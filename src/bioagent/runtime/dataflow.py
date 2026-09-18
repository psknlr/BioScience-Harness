"""Cross-step data binding — a step's arguments may reference earlier outputs.

A multi-step biomedical task is a chain, not a list:

    gene symbol -> Ensembl lookup -> ENSG id -> Open Targets -> associations -> ...

The runtime executed plans as independent invocations, so step 2 never saw
step 1's output and every step had to be fully parameterized up front — which a
planner cannot do, because the ENSG id does not exist until Ensembl answers.

Arguments may now contain references that are resolved at execution time:

    "${steps.<component_id>.output.<path>}"     by component id
    "${steps.<index>.output.<path>}"            by 0-based position in the plan
    "${task}"                                   the task text

`<path>` is a dotted key path with optional `[n]` indexes, e.g.
`data[0].id` or `results.hits[2].symbol`. A reference that is the *whole* string
yields the typed value (an int stays an int, a dict stays a dict); a reference
embedded in a longer string is substituted textually.

A reference that cannot be resolved — the step did not run, did not succeed, or
the path is absent — raises `DataflowError`. The runtime turns that into a
FAILED result for the dependent step rather than calling the backend with a
literal `"${steps...}"` string, which is what an unresolved template would
otherwise silently do.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

# Component ids contain dots (`public.connector.ensembl`), so the step key runs
# lazily up to the first `.output`/`.value`; the path after it is dotted keys
# and `[n]` indexes.
_REF = re.compile(r"\$\{(task|steps\.([^}\[]+?)\.(output|value)((?:\.[^.}\[]+|\[\d+\])*))\}")
#: Anything that still looks like a reference after binding is a malformed one.
_LEFTOVER = re.compile(r"\$\{\s*(steps|task)\b[^}]*\}")
_PATH = re.compile(r"\.([^.\[\]]+)|\[(\d+)\]")


class DataflowError(ValueError):
    """A step references data that is not available."""


def _walk(value: Any, path: str, origin: str) -> Any:
    for m in _PATH.finditer(path):
        key, idx = m.group(1), m.group(2)
        if idx is not None:
            if not isinstance(value, (list, tuple)):
                raise DataflowError(f"{origin}: [{idx}] applied to {type(value).__name__}")
            i = int(idx)
            if i >= len(value):
                raise DataflowError(f"{origin}: index {i} out of range ({len(value)} items)")
            value = value[i]
        else:
            if isinstance(value, Mapping):
                if key not in value:
                    have = ", ".join(list(map(str, value))[:8])
                    raise DataflowError(f"{origin}: key {key!r} not found (have: {have})")
                value = value[key]
            elif isinstance(value, (list, tuple)) and key.isdigit():
                value = value[int(key)]
            else:
                attr = getattr(value, key, _MISSING)
                if attr is _MISSING:
                    raise DataflowError(f"{origin}: {type(value).__name__} has no {key!r}")
                value = attr
    return value


_MISSING = object()


def _resolve_ref(m: "re.Match[str]", results: Mapping[str, Any], task: str) -> Any:
    whole = m.group(1)
    if whole == "task":
        return task
    step_key, _, path = m.group(2), m.group(3), m.group(4) or ""
    origin = "${" + whole + "}"
    res = results.get(step_key)
    if res is None:
        known = ", ".join(k for k in results if not k.isdigit())
        raise DataflowError(f"{origin}: no completed step {step_key!r} "
                            f"(completed: {known or 'none'})")
    status = getattr(res, "status", None)
    if status is not None and not getattr(status, "successful", False):
        raise DataflowError(f"{origin}: step {step_key!r} did not succeed "
                            f"({getattr(status, 'value', status)}: "
                            f"{str(getattr(res, 'error', '') or '')[:120]})")
    value = getattr(res, "value", res)
    return _walk(value, path, origin)


def bind_arguments(arguments: Mapping[str, Any] | None, results: Mapping[str, Any],
                   *, task: str = "") -> dict[str, Any]:
    """Resolve every `${...}` reference inside `arguments`.

    `results` maps both component ids and positional indexes (as strings) to the
    `CallResult` of the steps completed so far.
    """
    def sub(v: Any) -> Any:
        if isinstance(v, str):
            full = _REF.fullmatch(v.strip())
            if full:
                return _resolve_ref(full, results, task)
            out = _REF.sub(lambda m: str(_resolve_ref(m, results, task)), v)
            left = _LEFTOVER.search(out)
            if left:
                # Never hand a backend a template it could not resolve.
                raise DataflowError(f"malformed reference {left.group(0)!r}; expected "
                                    "${steps.<component_id>.output.<path>} or ${task}")
            return out
        if isinstance(v, Mapping):
            return {k: sub(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [sub(x) for x in v]
        return v

    return {k: sub(v) for k, v in dict(arguments or {}).items()}


def references(arguments: Mapping[str, Any] | None) -> list[str]:
    """Every reference string found in `arguments` (for plan inspection)."""
    found: list[str] = []

    def walk(v: Any) -> None:
        if isinstance(v, str):
            found.extend("${" + m.group(1) + "}" for m in _REF.finditer(v))
        elif isinstance(v, Mapping):
            for x in v.values():
                walk(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                walk(x)

    walk(dict(arguments or {}))
    return found
