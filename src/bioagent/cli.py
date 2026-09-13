"""bioagent command line: fetch datasets, list what is fetchable, verify sources.

    python -m bioagent.cli fetch <component_id|filename> [--confirm]
    python -m bioagent.cli fetchable
    python -m bioagent.cli sources
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .acquisition import BulkDatasetProvider, Downloader, acquisition_for
from .config import data_lake_dir
from .runtime.registry import ComponentRegistry, Resolver


def _registry() -> ComponentRegistry:
    return ComponentRegistry(BulkDatasetProvider().discover())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="bioagent")
    ap.add_argument("--dest", default=None, help="data directory (default: data lake dir)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download a fetchable dataset")
    f.add_argument("target", help="component id or filename")
    f.add_argument("--confirm", action="store_true", help="allow downloads above the size gate")
    sub.add_parser("fetchable", help="list datasets that can be fetched")
    sub.add_parser("sources", help="list public API connectors")
    a = ap.parse_args(argv)

    dest = Path(a.dest) if getattr(a, "dest", None) else data_lake_dir()
    reg = _registry()
    if a.cmd == "fetchable":
        present = {p.name for p in dest.iterdir()} if dest.is_dir() else set()
        res = Resolver(reg, dataset_probe=lambda n: n in present)
        for m in reg:
            r = res.resolve(m.id)
            state = "present" if not r.missing_datasets else ("fetchable" if r.fetchable else "blocked")
            spec = acquisition_for(m)
            print(f"{state:<9} {m.id:<48} {spec.host if spec else ''}")
        return 0
    if a.cmd == "sources":
        from .providers.public_apis import SOURCES
        for s in SOURCES:
            print(f"{s.key:<15} {s.host:<36} {len(s.operations)} ops  {s.license}")
        return 0
    # fetch
    m = reg.get(a.target) or next((x for x in reg if x.name == a.target), None)
    if m is None:
        print(f"unknown dataset {a.target!r}; try `fetchable`", file=sys.stderr)
        return 2
    spec = acquisition_for(m)
    dl = Downloader(dest, log=lambda s: print(s, file=sys.stderr))
    try:
        out = dl.fetch(spec.url, spec.filename, checksum=spec.checksum,
                       expected_bytes=spec.expected_bytes, confirm=a.confirm)
    except Exception as exc:  # noqa: BLE001
        print(f"fetch failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"path": str(out.path), "bytes": out.bytes, "verified": out.verified,
                      "checksum": out.checksum, "from_cache": out.from_cache}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
