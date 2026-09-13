"""Release-hygiene tests: what actually ships must be usable.

Three defects reached the v2.1 release and none of them were logic bugs — they
were packaging bugs, invisible to a test suite that only ever ran from a source
checkout:

* the tarball carried ~80 macOS AppleDouble sidecars (`._name`), and pytest
  collects `._test_*.py` as modules, so unpacking the release and running the
  suite died during collection;
* the wheel contained no capability catalogue at all, because the data was never
  declared as package data — `pip install` produced a registry with nothing to
  index;
* `demo_run.py` shipped with an import spliced into the middle of a string
  literal, so it raised SyntaxError on import.

Building a wheel is slow, so that check is marked `integration`; the tree checks
are cheap and always run.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(REPO / "scripts"))
from make_release import (REQUIRED_IN_WHEEL, find_junk,  # noqa: E402
                          find_unparsable, verify_artifact)


def test_no_appledouble_or_os_metadata_files_in_the_tree() -> None:
    junk = find_junk(REPO)
    assert junk == [], (
        f"{len(junk)} AppleDouble/OS metadata file(s) in the tree, e.g. "
        f"{[str(p.relative_to(REPO)) for p in junk[:5]]}. "
        "Run: python scripts/make_release.py")


def test_every_shipped_python_file_parses() -> None:
    bad = find_unparsable(REPO)
    assert bad == [], f"unparsable python file(s): {bad}"


def test_the_catalogue_is_inside_the_package_not_only_the_repo() -> None:
    """`catalogue_path()` must resolve without a source checkout."""
    from bioagent.config import catalogue_path
    from bioagent.data import CATALOGUE_CSV

    assert CATALOGUE_CSV.exists(), "the packaged catalogue is missing"
    resolved = catalogue_path()
    assert resolved.exists()
    package_root = Path(__import__("bioagent").__file__).resolve().parent
    assert package_root in resolved.parents, (
        f"catalogue_path() resolved to {resolved}, outside the installed package; "
        "an installed wheel would find nothing there")


def test_the_packaged_catalogue_actually_loads() -> None:
    from bioagent import CapabilityRegistry
    from bioagent.config import catalogue_path

    registry = CapabilityRegistry.from_csv(catalogue_path())
    assert len(registry) > 1000, f"catalogue loaded only {len(registry)} rows"


def test_package_data_declares_every_runtime_data_file() -> None:
    """A new file under bioagent/data/ must be matched by pyproject's globs."""
    import fnmatch

    pyproject = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert '"bioagent.data"' in pyproject, "package-data entry for bioagent.data is missing"
    globs = ["*.csv", "*.json", "*.parquet"]
    for path in (REPO / "src" / "bioagent" / "data").iterdir():
        if path.name in ("__init__.py", "__pycache__"):
            continue
        assert any(fnmatch.fnmatch(path.name, g) for g in globs), (
            f"{path.name} is in bioagent/data/ but no package-data glob matches it; "
            "it would be dropped from the wheel")


@pytest.mark.integration
def test_the_built_wheel_contains_the_catalogue_and_no_sidecars(tmp_path) -> None:
    """The end-to-end check: build a real wheel and look inside it."""
    pytest.importorskip("build", reason="pip install build")
    subprocess.run([sys.executable, "-m", "build", "--wheel", "--outdir", str(tmp_path),
                    str(REPO)], check=True, capture_output=True)
    wheels = list(tmp_path.glob("*.whl"))
    assert wheels, "no wheel was produced"
    problems = verify_artifact(wheels[0])
    assert problems == [], f"wheel is not releasable: {problems}"
    names = zipfile.ZipFile(wheels[0]).namelist()
    for required in REQUIRED_IN_WHEEL:
        assert required in names
