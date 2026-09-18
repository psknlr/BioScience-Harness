"""LLM-backed planner.

The model client is injected, so the runtime has no vendor dependency. Three
things distinguish this from the earlier version:

* **It plans arguments, not just components.** The prompt used to ask only for
  `[{"id", "rationale"}]`, so the model could say "use Ensembl" but never "call
  Ensembl's gene_lookup with symbol=TP53" — and `PlanStep.arguments` stayed
  empty. Candidates are now presented with their argument surface (typed
  operations for public connectors, declared inputs otherwise) and the model
  returns `arguments` per step, including cross-step references such as
  `${steps.public.connector.ensembl.output.id}` (see `runtime.dataflow`).
* **It sees what went wrong.** Retry attempts pass a `PlanContext` with the
  previous failures, the critique and the retry hint, so a revised plan is a
  response to the failure rather than the same plan minus one step.
* **It falls back loudly.** With no client bound the plan is produced by the
  heuristic planner and `Plan.degraded` says so; the runtime records a
  `PlannerFallback` event. `AgentSpec(require_model=True)` refuses to run
  instead.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from .base import Plan, PlanContext, PlannerPlugin, PlanStep, register_planner
from .heuristic import HeuristicPlanner


def describe_arguments(manifest: Any) -> dict[str, Any]:
    """The argument surface of a component, as a model can use it."""
    cid = getattr(manifest, "id", "")
    if isinstance(cid, str) and cid.startswith("public.connector."):
        try:
            from ..providers.public_apis import BY_KEY

            src = BY_KEY[cid.rsplit(".", 1)[-1]]
            return {"operations": [
                {"operation": o.name, "description": o.description[:120],
                 "required": list(o.args), "example": dict(o.example)}
                for o in src.operations]}
        except (KeyError, ImportError):
            pass
    inputs = getattr(manifest, "inputs", None) or {}
    if isinstance(inputs, dict) and inputs:
        return {"inputs": {k: (v if isinstance(v, (str, int, float, bool)) else str(v)[:80])
                           for k, v in list(inputs.items())[:12]}}
    return {}


@register_planner
class LLMPlanner(PlannerPlugin):
    """Asks a model which retrieved components to call, in what order, and how."""

    name = "llm"

    def __init__(self, client: Callable[[str], str] | None = None, model: str = "",
                 fallback: PlannerPlugin | None = None, backends: Any = None) -> None:
        self._client = client
        self.model = model
        self.backends = backends
        # Accepting `backends=` matters: the runtime passes it, and a TypeError
        # here used to make the runtime silently construct a client-less planner.
        self.fallback = fallback or HeuristicPlanner(backends=backends)

    @property
    def client_bound(self) -> bool:
        return self._client is not None

    # ------------------------------------------------------------------ plan
    def plan(self, task: str, registry: Any, *, max_steps: int = 5,
             context: PlanContext | None = None) -> Plan:
        excluded = set(context.excluded) if context else set()
        candidates = [m for m in registry.search(task, limit=max_steps * 6) if m.id not in excluded]
        if not candidates:
            return Plan(task=task, steps=[], planner=self.name, notes="no candidates retrieved")
        if self._client is None:
            return self._fallback(task, registry, max_steps, context,
                                  "no model client bound to the runtime")

        menu = [{"id": m.id, "kind": m.kind, "backend": m.runtime.backend,
                 "state": m.state.value, "description": m.description[:160],
                 **describe_arguments(m)}
                for m in candidates]
        feedback = context.describe() if context else ""
        prompt = (
            "You are planning a biomedical analysis as an ordered sequence of component "
            "calls.\n"
            f"TASK: {task}\n\n"
            + (f"FEEDBACK FROM EARLIER ATTEMPTS:\n{feedback}\n\n" if feedback else "")
            + f"AVAILABLE COMPONENTS (JSON):\n{json.dumps(menu, indent=1)[:9000]}\n\n"
            "Rules:\n"
            "- Prefer components whose state is READY or RESOLVED; backend 'none' cannot execute.\n"
            "- For components listing `operations`, set arguments.operation to one of them and "
            "supply its `required` arguments (see `example`).\n"
            "- A later step may use an earlier step's output with a reference string: "
            "\"${steps.<component_id>.output.<dotted.path>}\" (e.g. "
            "\"${steps.public.connector.ensembl.output.id}\"). Use this instead of guessing "
            "identifiers that only exist after an earlier step runs.\n"
            "- Never invent argument names that the component does not list.\n"
            "Reply with ONLY a JSON array of objects "
            '[{"id": "...", "arguments": {...}, "rationale": "..."}] '
            f"with at most {max_steps} entries, in execution order."
        )
        try:
            raw = self._client(prompt)
            payload = self._extract_json(raw)
            valid = {m.id: m for m in candidates}
            steps: list[PlanStep] = []
            for item in payload:
                if not isinstance(item, dict) or item.get("id") not in valid:
                    continue
                args = item.get("arguments")
                steps.append(PlanStep(
                    component_id=item["id"], kind=valid[item["id"]].kind,
                    arguments=dict(args) if isinstance(args, dict) else {},
                    rationale=str(item.get("rationale", ""))[:200]))
            if not steps:
                raise ValueError("model returned no valid component ids")
            return Plan(task=task, steps=steps[:max_steps], planner=self.name,
                        notes=f"model={self.model or 'unspecified'}; "
                              f"chose {len(steps)} of {len(candidates)} candidates"
                              + (f"; attempt {context.attempt}" if context else ""))
        except Exception as exc:  # noqa: BLE001 - planning must not crash the run
            return self._fallback(task, registry, max_steps, context,
                                  f"model planning failed: {type(exc).__name__}: {exc}")

    def _fallback(self, task: str, registry: Any, max_steps: int,
                  context: PlanContext | None, why: str) -> Plan:
        from .base import plan_with_context

        p = plan_with_context(self.fallback, task, registry, max_steps=max_steps, context=context)
        p.planner = f"{self.name}->{self.fallback.name}"
        p.degraded = f"{self.name} planner requested but {why}; used {self.fallback.name}"
        p.notes = f"{p.notes} ({p.degraded})"
        return p

    @staticmethod
    def _extract_json(text: str) -> list:
        t = text.strip()
        start, end = t.find("["), t.rfind("]")
        if start >= 0 and end > start:
            return json.loads(t[start:end + 1])
        raise ValueError("no JSON array found in model response")
