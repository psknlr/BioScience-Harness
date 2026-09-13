"""Provider that turns the v1 capability catalogue into component manifests.

The catalogue stays valuable — it is the *component index*. What changes is its
status: it no longer implies executability. This provider maps each row to a
manifest, infers the execution backend from the row's integration mode and kind,
and attaches the real python-import requirements read from the upstream source
so the resolver can decide READY vs UNAVAILABLE honestly.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Iterable, Iterator, Mapping

from ..runtime.component import (ComponentManifest, LicenseSpec, Permissions,
                                 Provider, Requirements, RuntimeSpec, Validation)

_SLUG = re.compile(r"[^a-z0-9_.-]+")


def slug(text: str) -> str:
    return _SLUG.sub("-", str(text).strip().lower()).strip("-") or "unnamed"


#: project -> import root for python-backed tools
PROJECT_IMPORT_ROOT = {
    "Biomni": "biomni.tool",
    "STELLA": "stella",
    "GenoMAS": "genomas",
    "AutoBA": "autoba",
    "Agentomics": "agentomics",
    "BioMedArena": "harness.tools",
    "GeneAgent": "geneagent",
}


def infer_backend(kind: str, integration_mode: str, native_connectors: Iterable[str],
                  project: str) -> str:
    """Choose an execution backend from catalogue facts."""
    ncs = [c for c in native_connectors if c]
    if kind == "dataset":
        return "dataset"
    if kind in ("database", "connector"):
        return "mcp" if ncs else "http"
    if kind in ("benchmark", "software", "agent_role", "skill"):
        # These are declarative in the catalogue: no invocable entrypoint exists.
        return "none"
    if kind == "tool":
        if integration_mode == "vendor" and project in PROJECT_IMPORT_ROOT:
            return "python"
        if ncs:
            return "mcp"
        return "subprocess"
    return "none"


_IMPORT_CACHE: dict[tuple[str, float], tuple[str, ...]] = {}


def scan_module_imports(source_root: Path, rel_path: str) -> tuple[str, ...]:
    """Read the third-party top-level imports of the module a tool lives in.

    Memoized on (path, mtime): Biomni's 224 tools live in 22 modules, so the
    un-memoized version parsed each module ~10 times per registry build.
    """
    p = source_root / rel_path
    if not p.is_file() or p.suffix != ".py":
        return ()
    try:
        key = (str(p), p.stat().st_mtime)
    except OSError:
        return ()
    if key in _IMPORT_CACHE:
        return _IMPORT_CACHE[key]
    result = _scan_uncached(p)
    _IMPORT_CACHE[key] = result
    return result


def _scan_uncached(p: Path) -> tuple[str, ...]:
    try:
        tree = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
    except (SyntaxError, OSError):
        return ()
    import sys as _sys
    std = set(_sys.stdlib_module_names)
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                found.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
    return tuple(sorted(m for m in found if m not in std and not m.startswith("_")))


class CatalogueProvider:
    """Discovers components from the unified capability catalogue."""

    name = "catalogue"

    def __init__(self, rows: Iterable[Mapping], repos_root: Path | None = None,
                 commits: Mapping[str, str] | None = None) -> None:
        self.rows = list(rows)
        self.repos_root = Path(repos_root) if repos_root else None
        self.commits = dict(commits or {})

    def discover(self) -> Iterator[ComponentManifest]:
        for row in self.rows:
            yield self._to_manifest(row)

    # ------------------------------------------------------------------ mapping
    def _to_manifest(self, row: Mapping) -> ComponentManifest:
        def s(key: str, default: str = "") -> str:
            v = row.get(key, default)
            return "" if v is None else str(v)

        def lst(key: str) -> tuple[str, ...]:
            raw = s(key)
            if not raw or raw.lower() == "nan":
                return ()
            return tuple(x.strip() for x in raw.split(";") if x.strip())

        kind = s("kind", "tool")
        name = s("name")
        projects = lst("contributing_projects")
        primary = projects[0] if projects else ""
        # composite labels such as "ClawBio+K-Dense"
        if "+" in primary:
            primary = primary.split("+")[0]
        licenses = lst("licenses")
        mode_raw = s("integration_mode", "federated")
        mode = {"adapter-only": "federated"}.get(mode_raw, mode_raw)
        ncs = lst("native_connectors")
        source_paths = lst("source_paths")
        rel = source_paths[0] if source_paths else ""

        backend = infer_backend(kind, mode, ncs, primary)
        entrypoint = ""
        server = ""
        if backend == "python":
            root = PROJECT_IMPORT_ROOT.get(primary, slug(primary))
            module = root
            if rel.endswith(".py"):
                stem = Path(rel).stem
                if stem and stem not in ("__init__",):
                    module = f"{root}.{stem}" if not root.endswith(stem) else root
            entrypoint = f"{module}:{name}"
        elif backend == "mcp":
            server = ncs[0] if ncs else ""
            entrypoint = name

        py_reqs: tuple[str, ...] = ()
        if backend == "python" and self.repos_root and rel:
            py_reqs = scan_module_imports(self.repos_root, rel)

        deps = lst("external_deps")
        hosts = tuple(d.lower() for d in deps if "." in d and " " not in d)

        cid = f"{slug(primary) or 'unknown'}.{kind}.{slug(name)}"
        return ComponentManifest(
            id=cid, kind=kind, name=name,
            description=s("description")[:400], domain=s("domain"),
            omics_type=s("omics_type", "general"),
            provider=Provider(project=primary, commit=self.commits.get(primary, ""),
                              source_path=rel),
            runtime=RuntimeSpec(backend=backend, entrypoint=entrypoint, server=server),
            requires=Requirements(python=py_reqs,
                                  datasets=(name,) if kind == "dataset" else ()),
            permissions=Permissions(network=hosts,
                                    subprocess=(backend == "subprocess")),
            license=LicenseSpec(spdx=(licenses[0] if licenses else "NONE"),
                                integration_mode=mode),
            validation=Validation(),
            offline_capable=(kind == "dataset"),
            native_connectors=ncs,
            signature=s("signature"),
        )
