# bioagent-harness v2.2 — a composable harness for biomedical AI agents

## v2.2 — trusted-core hardening

No new agents or connectors in this release. It closes gaps between what the
harness *declared* and what it *enforced*, each pinned by a regression test in
`tests/test_v22_trusted_core.py` and `tests/test_packaging.py`.

**Release integrity**
- The v2.1 tarball carried ~80 macOS AppleDouble sidecars (`._name`). `._test_*.py`
  is a filename pytest collects, so unpacking the release and running the suite
  failed during collection. `scripts/make_release.py` now sanitizes the tree and
  *refuses* to publish an artifact containing them.
- The wheel shipped no capability catalogue: it was never declared as package
  data, and `catalogue_path()` returned a path that resolves only in a source
  checkout. The catalogue now lives in `src/bioagent/data/` and travels with the
  distribution. CI installs the wheel into a clean venv, outside the repository,
  and loads the catalogue from it — the check a source-checkout suite structurally
  cannot make.
- `demo_run.py` shipped with an import spliced into the middle of a string
  literal, so it raised `SyntaxError` on import. Fixed, and the release check now
  parses every shipped python file.

**Policy enforcement**
- *auto-fetch no longer bypasses the agent's profile.* `Runtime.fetch()`
  authorized with a hardcoded `profile="biomedical-research"` whatever the spec
  asked for, so an `offline-analysis` agent with `auto_fetch=True` completed a
  download and was only then denied at invoke time — the bytes were already on
  disk. The fetch now runs under the spec's own profile, and a caller that
  supplies none falls back to the most restrictive profile rather than the most
  permissive.
- *Filesystem capabilities are checked.* `PermissionProfile` had carried
  `allow_filesystem_read` / `allow_filesystem_write` since v2.0 and
  `authorize()` had never looked at them. Declared paths are now ruled against
  the profile's permitted roots, `..` cannot walk out of a root, and — because
  the in-process python backend can enforce no write boundary at all — a
  component declaring writes is refused that backend outright instead of being
  granted a permission nothing enforces.
- *Policy propagates along data lineage.* Rulings covered only the component
  named in the request, so a permissively licensed tool whose
  `requires.datasets` pointed at a denied dataset was authorized on its own
  merits. `AuthorizationRequest.dependencies` now carries the resolved closure
  (`Resolver.dependency_contexts()`), most-restrictive-wins, and an
  unresolvable dependency fails closed.

**Scientific correctness**
- *Execution success is no longer reported as scientific acceptance.* A
  validator that ran cleanly and rejected the result produced
  `outcome=SUCCESS, ok=True, accepted=True` while its own critique said the
  validation had failed. `RunReport` now reports `execution_outcome` (did the
  machinery run) and `verdict` (`ACCEPTED` / `REJECTED` / `INCONCLUSIVE`, from
  the critique) separately, and `ok` requires both. `outcome` remains as an
  alias for `execution_outcome`.

**Environment measurement**
- *Container availability is measured, not asserted.* The resolver hardcoded
  "container backend requires a container runtime (none available)", so
  installing Docker changed nothing. `Resolver` takes a `backend_probe`, and
  `Runtime` supplies one backed by its live `BackendRegistry`.

## v2.1 — what changed
- Ten architectural defects fixed, each pinned by a regression test (`tests/test_v21_regressions.py`);
  one was security-class (path traversal into the immutable policy plane).
- `HTTPBackend` (REST + GraphQL, stdlib-only) with per-host rate limits, retries, response cache.
- 16 public data-source connectors, 45 typed operations, all verified live
  (`data/connector_live_verification.csv`): Ensembl, UniProt, NCBI E-utilities, ChEMBL, PubChem,
  ClinicalTrials.gov, openFDA, STRING, KEGG, Reactome, Open Targets, RCSB PDB, Europe PMC, gnomAD,
  MyGene, MyVariant.
- Acquisition layer: `AcquisitionSpec`, resumable checksum-verified `Downloader`, FETCHABLE resolution,
  `AgentSpec(auto_fetch=True)`, and a CLI:

      PYTHONPATH=src python -m bioagent.cli sources
      PYTHONPATH=src python -m bioagent.cli fetchable
      PYTHONPATH=src python -m bioagent.cli fetch hgnc.dataset.hgnc_complete_set_txt

- Executable-now components: 76 → 96. 764 more have a verified-live equivalent source (routable,
  not yet dispatched per component). See `data/capability_state_census_v21.csv`.


Built from a survey of 16 biomedical agent projects (2,567 catalogued capabilities).
v2 rebuilds the v1 scaffold as a **component runtime**: one manifest schema, a trusted
policy kernel, pluggable execution backends, event-sourced provenance, and a validated
self-evolution pipeline.

## What v2 corrects

v1 reported 621 capabilities as "confirmed routable". That was a routing decision, never a
dispatch. v2 measures executability against the live environment and reports it honestly:

    catalogued        2,567  (100%)
    dependency-ok     1,024  (39.9%)
    executable now       76  (3.0%)   <- all datasets

Nine v1 defects were reproduced empirically and fixed; each has a regression test in
`tests/test_regressions.py`.

## Layout

    src/bioagent/
      status.py            ExecutionStatus, LifecycleState, RunOutcome + transition table
      policy.py            immutable policy kernel (read-only license/permission tables)
      config.py            path resolution: argument -> env var -> repo-relative
      data/                packaged runtime data (the capability catalogue)
      runtime/
        component.py       ComponentManifest (the one composable unit)
        registry.py        ComponentRegistry / Resolver / Loader
        events.py          event-sourced provenance with graph replay
        hmr.py             transactional hot reload + LazyComponentSet
        agentspec.py       AgentSpec (data) + Runtime (executes any spec)
      backends/            python | mcp | dataset | subprocess | container | none
      providers/           discovery from catalogue rows and SKILL.md trees
      planners/            self-registering plugins: heuristic, llm
      evolution/           propose -> test -> benchmark -> policy -> promote
      workspace/           file workspace with an enforced trust boundary + git
      adapters/            v1 adapters (retained; superseded by backends)

## Quick start

    export PYTHONPATH=src
    export BIOAGENT_DATA_LAKE=/path/to/biomni_lake     # optional
    python demo_harness.py                             # full v2 demo
    python -m pytest -q                                # full suite
    python -m pytest -q -m unit                        # no data lake, no network

## Tests

    tests/test_regressions.py        pins all 8 v1 blocker defects
    tests/test_v2_harness.py         manifests, lifecycle, policy, backends, HMR, evolution
    tests/test_bioagent.py           v1 surface (migrated to status semantics)
    tests/test_v21_regressions.py    the ten v2 architectural-audit defects
    tests/test_v22_trusted_core.py   profile bypass, filesystem capabilities, lineage
                                     propagation, execution-vs-verdict, backend probing
    tests/test_packaging.py          release hygiene: sidecars, parseability, wheel contents

Building a release, with the checks that refuse a broken one:

    python scripts/make_release.py --check   # verify the tree; build nothing
    python scripts/make_release.py           # sanitize, build sdist + wheel, verify both

## What "trusted" means here, precisely

This is a **policy-gated, provenance-aware runtime**, not a sandbox. The
distinction is load-bearing, and `PolicyKernel.enforcement_report()` states it
per capability class at runtime rather than leaving it to prose:

| capability | how it is enforced |
| --- | --- |
| license / integration mode | **mechanism** — the invocation is refused |
| subprocess | **mechanism** — the invocation is refused |
| filesystem write | **mechanism** — backends that cannot confine writes are refused |
| filesystem read | **declaration** — declared paths are gated; reads are not intercepted |
| network hosts | **declaration** — declared hosts are gated; sockets are not intercepted |

"Declaration" means the kernel rules on what a component *declares* and records
the decision; it does not interpose on syscalls, so code that reaches a backend
still runs with the harness's own OS privileges. Genuine capability isolation
needs a container runtime, and `ContainerBackend` / `HardenedExecutor.guarantees()`
report whether one is present rather than assuming either way.

## Honesty notes

* Filesystem and network isolation depend on a container runtime; both
  `ContainerBackend.available()` and `HardenedExecutor.guarantees()` report what
  is actually present on the machine instead of implying it.
* This sandbox refuses to lower RLIMIT_AS, so `HardenedExecutor.guarantees()` probes what
  is actually enforced (CPU + file size) and reports the memory cap as unenforced.
* The Biomni data lake (15.1 GB) is not redistributable; only its inventory ships here.
* Unlicensed upstream projects are never vendored — they are invoked in place, and the
  policy kernel denies a vendor route for them on every call.

## License

MIT for this harness. `NOTICE` records every upstream project, its license, and whether it
is vendorable or federated-only.
