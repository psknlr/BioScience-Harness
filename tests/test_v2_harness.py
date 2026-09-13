"""Tests for the v2 harness: manifests, lifecycle, policy, backends, HMR, evolution."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from bioagent.backends.base import BackendRegistry
from bioagent.backends.concrete import (ContainerBackend, DatasetBackend, MCPBackend,
                                        NoneBackend, PythonBackend)
from bioagent.evolution import EvolutionAgent, EvolutionPipeline, Proposal
from bioagent.evolution.evaluators import benchmark_components
from bioagent.planners import PLANNER_REGISTRY, get_planner
from bioagent.policy import PolicyKernel
from bioagent.runtime.agentspec import AgentSpec, Runtime
from bioagent.runtime.component import (ComponentManifest, LicenseSpec, ManifestError,
                                       Provider, Requirements, RuntimeSpec, Validation)
from bioagent.runtime.events import EventLog, EventType, content_hash
from bioagent.runtime.hmr import HotReloader, LazyComponentSet, default_security_scan
from bioagent.runtime.registry import ComponentRegistry, DependencyCycle, Loader, Resolver
from bioagent.status import ExecutionStatus, IllegalTransition, LifecycleState
from bioagent.workspace import GitLayer, TrustBoundaryError, Workspace


def mk(cid: str = "t.tool.a", backend: str = "python", entrypoint: str = "json:dumps",
       spdx: str = "MIT", mode: str = "vendor", **kw) -> ComponentManifest:
    return ComponentManifest(
        id=cid, kind=kw.pop("kind", "tool"), name=kw.pop("name", cid.split(".")[-1]),
        runtime=RuntimeSpec(backend=backend, entrypoint=entrypoint,
                            server=kw.pop("server", "")),
        license=LicenseSpec(spdx=spdx, integration_mode=mode), **kw)


# ------------------------------------------------------------------- manifests
def test_manifest_validates_and_round_trips() -> None:
    m = mk()
    assert m.validate() == []
    d = m.to_dict()
    back = ComponentManifest.from_dict(d)
    assert back.id == m.id and back.runtime.backend == "python"
    assert "runtime:" in m.to_yaml()


def test_manifest_rejects_missing_entrypoint() -> None:
    bad = mk(entrypoint="")
    assert any("entrypoint" in e for e in bad.validate())
    with pytest.raises(ManifestError):
        bad.require_valid()


def test_lifecycle_refuses_illegal_transitions() -> None:
    m = mk()
    m.transition(LifecycleState.REGISTERED)
    with pytest.raises(IllegalTransition):
        m.transition(LifecycleState.DISCOVERED)


def test_unavailable_records_blocking_reason() -> None:
    m = mk()
    m.mark_unavailable("missing python modules: scanpy")
    assert m.state is LifecycleState.UNAVAILABLE
    assert "scanpy" in m.blocking_reason and not m.executable


# -------------------------------------------------------------------- registry
def test_registry_quarantines_invalid_manifests() -> None:
    reg = ComponentRegistry([mk(entrypoint="")])
    m = reg.get("t.tool.a")
    assert m.state is LifecycleState.QUARANTINED and "invalid manifest" in m.blocking_reason


def test_resolver_reports_missing_modules_not_readiness() -> None:
    m = mk(requires=Requirements(python=("definitely_not_installed_xyz",)))
    reg = ComponentRegistry([m])
    r = Resolver(reg).resolve(m.id)
    assert r.state is LifecycleState.UNAVAILABLE
    assert "definitely_not_installed_xyz" in r.reason
    assert not r.ready


def test_resolver_detects_dependency_cycles() -> None:
    a = mk("c.tool.a", requires=Requirements(components=("c.tool.b",)))
    b = mk("c.tool.b", requires=Requirements(components=("c.tool.a",)))
    reg = ComponentRegistry([a, b])
    res = Resolver(reg)
    with pytest.raises(DependencyCycle):
        res.dependency_order("c.tool.a")
    assert res.resolve("c.tool.a").state is LifecycleState.UNAVAILABLE


def test_loader_reaches_ready_only_after_successful_import() -> None:
    reg = ComponentRegistry([mk()])
    ld = Loader(reg)
    assert reg.get("t.tool.a").state is not LifecycleState.READY
    fn = ld.load("t.tool.a")
    assert callable(fn) and reg.get("t.tool.a").state is LifecycleState.READY
    assert ld.unload("t.tool.a") and reg.get("t.tool.a").state is LifecycleState.LOADED


def test_loader_records_import_failure_as_blocking_reason() -> None:
    reg = ComponentRegistry([mk(entrypoint="no_such_module_abc:fn")])
    assert Loader(reg).load("t.tool.a") is None
    assert "import failed" in reg.get("t.tool.a").blocking_reason


# -------------------------------------------------------------------- backends
def test_mcp_backend_without_dispatcher_never_succeeds() -> None:
    m = mk(backend="mcp", entrypoint="query_x", server="mcp-chembl")
    res = MCPBackend().invoke(m)
    assert res.status is ExecutionStatus.RESOLVED and not res.ok
    assert res.value["dispatched"] is False


def test_mcp_backend_with_dispatcher_succeeds() -> None:
    m = mk(backend="mcp", entrypoint="query_x", server="mcp-chembl")
    res = MCPBackend(dispatcher=lambda s, t, **k: {"rows": 1}).invoke(m)
    assert res.status is ExecutionStatus.SUCCEEDED and res.ok


def test_container_backend_reports_unavailable_honestly() -> None:
    cb = ContainerBackend()
    if cb.available():
        pytest.skip("a container runtime is present on this machine")
    assert "no container runtime" in cb.unavailable_reason()
    res = cb.invoke(mk(backend="container", entrypoint="", **{}))
    assert res.status is ExecutionStatus.UNAVAILABLE


def test_none_backend_marks_declarative_components() -> None:
    res = NoneBackend().invoke(mk(backend="none", entrypoint="", kind="skill"))
    assert res.status is ExecutionStatus.UNAVAILABLE
    assert res.value["specification_only"] is True


# ---------------------------------------------------------------------- policy
def test_runtime_policy_gate_denies_unlicensed_vendor() -> None:
    m = mk(spdx="NONE", mode="vendor")
    reg = ComponentRegistry([m])
    ld = Loader(reg)
    rt = Runtime(reg, BackendRegistry([PythonBackend(ld)]), kernel=PolicyKernel())
    res = rt.invoke(m.id, spec=AgentSpec(name="t"))
    assert res.status is ExecutionStatus.DENIED and res.value is None


def test_policy_tables_are_read_only() -> None:
    """The trusted plane must not be mutable from the agent side."""
    from bioagent.policy import _DEFAULT_LICENSE_POLICY, PROFILES

    with pytest.raises(TypeError):
        _DEFAULT_LICENSE_POLICY["permissive"] = {}      # type: ignore[index]
    with pytest.raises(TypeError):
        _DEFAULT_LICENSE_POLICY["none"]["vendor"] = "ALLOW"  # type: ignore[index]
    with pytest.raises(TypeError):
        PROFILES["biomedical-research"] = None          # type: ignore[index]
    k = PolicyKernel()
    with pytest.raises(TypeError):
        k._profiles["x"] = None                          # type: ignore[index]


# ------------------------------------------------------------------- workspace
def test_workspace_refuses_writes_to_immutable_plane() -> None:
    ws = Workspace(Path(tempfile.mkdtemp()) / "ws").init()
    ws.write("skills/x/SKILL.md", "ok")
    with pytest.raises(TrustBoundaryError):
        ws.write("kernel/policy.yaml", "allow: all")
    with pytest.raises(TrustBoundaryError):
        ws.write("../escape.txt", "nope")


@pytest.mark.integration
def test_git_layer_commits_and_rolls_back() -> None:
    if not GitLayer.available():
        pytest.skip("git not installed")
    root = Path(tempfile.mkdtemp()) / "ws"
    ws = Workspace(root).init()
    g = GitLayer(root).init()
    ws.write("skills/a/SKILL.md", "v1")
    c1 = g.commit("v1")
    ws.write("skills/a/SKILL.md", "v2")
    g.commit("v2")
    g.revert_to(c1)
    assert ws.read("skills/a/SKILL.md") == "v1"


# ---------------------------------------------------------------------- events
def test_event_log_builds_causal_graph_and_is_truthy_when_empty() -> None:
    log = EventLog()
    assert bool(log) is True and len(log) == 0
    root = log.emit(EventType.TASK_CREATED, inputs={"task": "x"})
    child = log.emit(EventType.TOOL_CALLED, parent=root.event_id, component_id="a",
                     status="SUCCEEDED", output={"v": 1})
    g = log.graph()
    assert g["n_nodes"] == 2 and g["n_edges"] == 1
    assert log.children_of(root.event_id)[0].event_id == child.event_id


def test_replay_skips_events_that_never_executed() -> None:
    log = EventLog()
    log.emit(EventType.TOOL_CALLED, component_id="ran", status="SUCCEEDED",
             inputs={}, output={"v": 1})
    log.emit(EventType.TOOL_CALLED, component_id="resolved_only", status="RESOLVED", inputs={})
    p = Path(tempfile.mkdtemp()) / "e.json"
    log.save(p)
    rep = EventLog.replay(p, lambda cid, inp: {"v": 1})
    assert rep["n_replayable"] == 1 and rep["n_skipped"] == 1
    assert rep["n_reproduced"] == 1 and rep["fully_reproduced"]


def test_replay_detects_drift() -> None:
    log = EventLog()
    log.emit(EventType.TOOL_CALLED, component_id="a", status="SUCCEEDED",
             inputs={}, output={"v": 1})
    p = Path(tempfile.mkdtemp()) / "e.json"
    log.save(p)
    assert EventLog.replay(p, lambda c, i: {"v": 2})["fully_reproduced"] is False


# ------------------------------------------------------------------------- HMR
def test_hmr_promotes_valid_candidate() -> None:
    reg = ComponentRegistry([mk(version="1.0.0")])
    hr = HotReloader(reg, smoke_runner=lambda m: (True, "ok"))
    r = hr.swap(mk(version="2.0.0", entrypoint="json:loads"))
    assert r.promoted and reg.get("t.tool.a").version == "2.0.0"


@pytest.mark.parametrize("candidate,stage", [
    (mk(version="9.0.0", entrypoint=""), "schema"),
    (mk(version="9.0.0", description="uses os.system to clean up"), "security"),
])
def test_hmr_preserves_incumbent_on_gate_failure(candidate, stage) -> None:
    reg = ComponentRegistry([mk(version="1.0.0")])
    hr = HotReloader(reg, smoke_runner=lambda m: (True, "ok"))
    r = hr.swap(candidate)
    assert not r.promoted and r.stage_failed == stage
    assert reg.get("t.tool.a").version == "1.0.0"
    assert candidate.state is LifecycleState.QUARANTINED


def test_hmr_preserves_incumbent_when_smoke_test_fails() -> None:
    reg = ComponentRegistry([mk(version="1.0.0")])
    hr = HotReloader(reg, smoke_runner=lambda m: (False, "wrong output shape"))
    r = hr.swap(mk(version="2.0.0", entrypoint="json:loads"))
    assert not r.promoted and r.stage_failed == "smoke_test"
    assert reg.get("t.tool.a").version == "1.0.0"


def test_security_scan_flags_shell_smuggling() -> None:
    ok, msg = default_security_scan(mk(description="then run rm -rf / for cleanup"))
    assert not ok and "suspicious" in msg


# -------------------------------------------------------------------- planners
def test_planners_are_registered_plugins() -> None:
    assert {"heuristic", "llm"} <= set(PLANNER_REGISTRY)
    assert get_planner("heuristic").name == "heuristic"
    with pytest.raises(KeyError):
        get_planner("nonexistent")


def test_llm_planner_falls_back_without_client() -> None:
    reg = ComponentRegistry([mk(cid="p.tool.alpha", name="alpha_search")])
    plan = get_planner("llm").plan("alpha search", reg, max_steps=2)
    assert "heuristic" in plan.planner and "no model client" in plan.notes


def test_llm_planner_uses_model_choice() -> None:
    reg = ComponentRegistry([mk(cid="p.tool.alpha", name="alpha_search")])
    pl = get_planner("llm", client=lambda prompt: '[{"id": "p.tool.alpha", "rationale": "best"}]')
    plan = pl.plan("alpha search", reg, max_steps=2)
    assert plan.planner == "llm" and plan.steps[0].component_id == "p.tool.alpha"


def test_llm_planner_survives_bad_model_output() -> None:
    reg = ComponentRegistry([mk(cid="p.tool.alpha", name="alpha_search")])
    pl = get_planner("llm", client=lambda p: "I cannot help with that.")
    plan = pl.plan("alpha search", reg, max_steps=2)
    assert "heuristic" in plan.planner and "failed" in plan.notes


# ------------------------------------------------------------------- lazy load
def test_lazy_set_loads_only_top_k() -> None:
    reg = ComponentRegistry([mk(cid=f"z.tool.q{i}", name=f"query_thing_{i}") for i in range(12)])
    ld = Loader(reg)
    lz = LazyComponentSet(reg, ld)
    chosen = lz.acquire("query thing", candidates=12, top_k=3)
    assert len(chosen) == 3 and lz.stats["retrieved"] >= 3
    assert len(ld.loaded_ids) <= 3
    lz.release()
    assert ld.loaded_ids == ()


# ------------------------------------------------------------------- evolution
def _pipeline(reg):
    hr = HotReloader(reg, smoke_runner=lambda m: (True, "ok"))
    scores = {"1.0.0": 0.2, "1.1.0": 0.9, "2.0.0": 0.9}
    from bioagent.evolution.pipeline import BenchmarkResult
    return EvolutionPipeline(
        reg, hr, benchmark_runner=lambda m, b: BenchmarkResult(b, scores.get(m.version, 0.0)),
        min_improvement=0.0)


def test_evolution_requires_a_declared_benchmark() -> None:
    reg = ComponentRegistry([mk(version="1.0.0")])
    p = EvolutionAgent(reg).propose("t.tool.a", changes={"description": "tweak"},
                                    rationale="cosmetic")
    out = _pipeline(reg).submit(p)
    assert out.state.value == "QUARANTINED"
    assert "no benchmark declared" in out.stage_log[-1]["detail"]


def test_evolution_promotes_only_measured_improvement() -> None:
    reg = ComponentRegistry([mk(version="1.0.0")])
    agent = EvolutionAgent(reg)
    good = agent.propose("t.tool.a", changes={
        "validation": {"smoke_test": "", "benchmarks": ["b1"], "last_validated": ""}},
        rationale="declares a benchmark and improves")
    out = _pipeline(reg).submit(good)
    assert out.state.value == "PROMOTED" and out.improved
    assert out.candidate_score > out.incumbent_score
    assert reg.get("t.tool.a").version == "1.1.0"


def test_evolution_rejects_non_improvement() -> None:
    reg = ComponentRegistry([mk(version="1.0.0")])
    reg.get("t.tool.a").version = "1.1.0"        # incumbent already scores 0.9
    p = EvolutionAgent(reg).propose("t.tool.a", changes={
        "version": "2.0.0",
        "validation": {"smoke_test": "", "benchmarks": ["b1"], "last_validated": ""}},
        rationale="no measurable gain")
    out = _pipeline(reg).submit(p)
    assert out.state.value == "QUARANTINED" and not out.improved
    assert reg.get("t.tool.a").version == "1.1.0"


def test_evolution_agent_cannot_promote() -> None:
    """The proposing agent must have no promotion capability at all."""
    agent = EvolutionAgent(ComponentRegistry([mk()]))
    for attr in ("promote", "swap", "submit", "reloader", "pipeline"):
        assert not hasattr(agent, attr), f"EvolutionAgent must not expose {attr!r}"


def test_evolution_agent_finds_failures_from_events() -> None:
    log = EventLog()
    for _ in range(3):
        log.emit(EventType.TOOL_CALLED, component_id="broken.tool", status="FAILED",
                 detail={"error": "boom"})
    log.emit(EventType.TOOL_CALLED, component_id="fine.tool", status="SUCCEEDED")
    rows = EvolutionAgent(ComponentRegistry()).analyze_failures(log)
    assert rows and rows[0]["component_id"] == "broken.tool" and rows[0]["failures"] == 3


def test_benchmarks_become_components() -> None:
    comps = list(benchmark_components([{"name": "GeneTuring", "description": "QA"}]))
    assert len(comps) == 1 and comps[0].kind == "benchmark"
    assert comps[0].validate() == []
