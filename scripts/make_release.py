#!/usr/bin/env python3
"""Build a release, refusing to ship a package that is known to be broken.

Two failures reached the v2.1 release tarball and both are mechanical, so both
are checked here rather than trusted to whoever runs the build:

1. **AppleDouble sidecars.** Archiving a source tree on macOS with the system
   `tar` writes a `._<name>` resource-fork file beside every real file. They are
   ordinary files to everyone else, and `._test_foo.py` is a filename pytest
   collects — so unpacking the release and running the suite failed at
   collection with ~80 unimportable modules before a single test ran.

2. **Missing runtime data.** The catalogue the registry loads was not declared
   as package data, so the wheel installed cleanly and then had nothing to
   index.

The build therefore: sanitizes the tree, builds sdist + wheel, and verifies each
artifact contains the runtime data and no sidecars. A failed check is a non-zero
exit, not a warning — a release that only warns is a release that ships.

    python scripts/make_release.py            # sanitize, build, verify
    python scripts/make_release.py --check    # verify the tree only, build nothing
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: Filenames that must never appear in a source tree or a release artifact.
JUNK_PATTERNS = ("._*", ".DS_Store", "Thumbs.db")
JUNK_DIRS = ("__MACOSX", ".ipynb_checkpoints")

#: Paths that must be present inside every built wheel, relative to the wheel root.
REQUIRED_IN_WHEEL = ("bioagent/data/unified_capability_catalogue.csv",)

SKIP_DIRS = {".git", ".venv", "venv", "build", "dist", ".mypy_cache",
             ".pytest_cache", "__pycache__", ".cache", ".eggs"}


def iter_tree(root: Path):
    for path in root.rglob("*"):
        if any(part in SKIP_DIRS or part.endswith(".egg-info") for part in path.parts):
            continue
        yield path


def find_junk(root: Path) -> list[Path]:
    """Every AppleDouble sidecar and OS metadata file in the tree."""
    out: list[Path] = []
    for path in iter_tree(root):
        if path.is_dir():
            if path.name in JUNK_DIRS:
                out.append(path)
            continue
        if any(path.match(pat) for pat in JUNK_PATTERNS):
            out.append(path)
    return sorted(out)


def find_unparsable(root: Path) -> list[tuple[Path, str]]:
    """Shipped python files that do not even parse.

    `demo_run.py` went out in v2.1 with an import spliced into the middle of a
    string literal, so the file raised SyntaxError on import. Nothing caught it
    because nothing tried to parse the demos.
    """
    import ast

    bad: list[tuple[Path, str]] = []
    for path in iter_tree(root):
        if path.suffix != ".py" or path.is_dir():
            continue
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError) as exc:
            bad.append((path.relative_to(root), f"{type(exc).__name__}: {exc}"))
    return bad


def sanitize(root: Path, *, dry_run: bool = False) -> list[Path]:
    """Delete the junk files, returning what was (or would be) removed."""
    junk = find_junk(root)
    if dry_run:
        return junk
    for path in junk:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
    return junk


def _archive_names(artifact: Path) -> list[str]:
    if artifact.suffix == ".whl":
        with zipfile.ZipFile(artifact) as zf:
            return zf.namelist()
    with tarfile.open(artifact) as tf:
        return tf.getnames()


def verify_artifact(artifact: Path) -> list[str]:
    """Problems found inside a built artifact; empty means it is releasable."""
    names = _archive_names(artifact)
    problems: list[str] = []
    junk = [n for n in names if Path(n).name.startswith("._")
            or Path(n).name in (".DS_Store", "Thumbs.db")
            or "__MACOSX" in Path(n).parts]
    if junk:
        problems.append(f"{len(junk)} AppleDouble/OS metadata entries, e.g. {junk[:3]}")
    if artifact.suffix == ".whl":
        for required in REQUIRED_IN_WHEEL:
            if required not in names:
                problems.append(f"missing runtime data: {required}")
    else:
        for required in REQUIRED_IN_WHEEL:
            leaf = Path(required).name
            if not any(Path(n).name == leaf for n in names):
                problems.append(f"missing runtime data: {leaf}")
    return problems


def build(outdir: Path) -> list[Path]:
    if outdir.exists():
        shutil.rmtree(outdir)
    subprocess.run([sys.executable, "-m", "build", "--sdist", "--wheel",
                    "--outdir", str(outdir), str(REPO)], check=True)
    return sorted(p for p in outdir.iterdir() if p.suffix in (".whl", ".gz"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="verify the working tree only; build nothing and delete nothing")
    ap.add_argument("--outdir", default=str(REPO / "dist"), help="where to write artifacts")
    args = ap.parse_args(argv)

    failures = 0

    junk = sanitize(REPO, dry_run=args.check)
    if junk:
        verb = "found" if args.check else "removed"
        print(f"[sanitize] {verb} {len(junk)} junk file(s):")
        for p in junk[:10]:
            print(f"           {p.relative_to(REPO)}")
        if len(junk) > 10:
            print(f"           ... and {len(junk) - 10} more")
        if args.check:
            failures += 1
    else:
        print("[sanitize] tree is clean")

    unparsable = find_unparsable(REPO)
    if unparsable:
        failures += 1
        print(f"[syntax]   {len(unparsable)} shipped python file(s) do not parse:")
        for rel, msg in unparsable:
            print(f"           {rel}: {msg}")
    else:
        print("[syntax]   every shipped python file parses")

    if args.check:
        print("OK" if not failures else "FAILED")
        return 1 if failures else 0

    artifacts = build(Path(args.outdir))
    for art in artifacts:
        problems = verify_artifact(art)
        if problems:
            failures += 1
            print(f"[verify]   {art.name}: FAILED")
            for pr in problems:
                print(f"           {pr}")
        else:
            print(f"[verify]   {art.name}: ok")

    print("OK" if not failures else "FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
