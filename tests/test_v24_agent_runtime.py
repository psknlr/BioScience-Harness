"""Regression tests for the v2.4 agent-runtime review.

The review's verdict: the declarative layer was sound, the execution layer did
not deliver an *agent* — steps could not consume each other's outputs, the LLM
planner planned components but never arguments (and silently ran heuristically
when no client was wired), 53 agent roles were permanently non-executable
metadata, memory was a directory, and nothing turned results into an answer.

Every test here was confirmed failing (or the feature absent) against the
reviewed code. A scripted model client stands in for a real one so the loops
are exercised deterministically; nothing here touches the network.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

from bioagent.adapters.base import CallResult
from bioagent.backends.agent import AgentBackend, RemoteAgentBackend
from bioagent.backends.base import BackendRegistry
from bioagent.backends.concrete import ContainerBackend, PythonBackend, SubprocessBackend
from bioagent.planners.base import Plan, PlanContext, PlanStep, plan_with_context
from bioagent.planners.heuristic import HeuristicPlanner
from bioagent.planners.llm import LLMPlanner
from bioagent.runtime.agentspec import AgentSpec, Runtime
from bioagent.runtime.component import (ComponentManifest, LicenseSpec, Provider,
                                        Requirements, RuntimeSpec)
from bioagent.runtime.dataflow import DataflowError, bind_arguments, references
from bioagent.runtime.memory import EpisodicMemory, WorkingMemory
from bioagent.runtime.orchestrator import Orchestrator
from bioagent.runtime.registry import ComponentRegistry, Loader, Resolver
from bioagent.runtime.synthesis import EvidenceSynthesizer
from bioagent.status import ExecutionStatus, LifecycleState, RunOutcome, ScientificVerdict


# ------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module", autouse=True)
def probe_module():
    """An importable module with a few deterministic 'tools'.

    Named uniquely: `test_v23_execution_semantics` also writes a `probe_impl`
    module, and Python caches the first one imported in `sys.modules`, so a
    shared name made this module's functions vanish in a full-suite run.
    """
    d = Path(tempfile.mkdtemp(prefix="bioagent-v24-"))
    (d / "probe_impl_v24.py").write_text(
        "def lookup(symbol='', **kw):\n"
        "    return {'id': 'ENSG00000141510', 'symbol': symbol}\n"
        "def associations(gene_id='', **kw):\n"
        "    return {'gene_id': gene_id, 'targets': [{'disease': 'EFO:0000311', 'score': 0.91}]}\n"
        "def broken(**kw):\n"
        "    raise RuntimeError('upstream exploded')\n"
        "def echo(**kw):\n"
        "    return {'called_with': kw}\n",
        encoding="utf-8")
    sys.path.insert(0, str(d))
    yield d
    sys.path.remove(str(d))
    shutil.rmtree(d, ignore_errors=True)


def tool(cid: str, fn: str, desc: str = "", **kw) -> ComponentManifest:
    return ComponentManifest(
        id=cid, kind="tool", name=cid.rsplit(".", 1)[-1], description=desc or cid,
        runtime=RuntimeSpec(backend="python", entrypoint=f"probe_impl_v24:{fn}"),
        license=LicenseSpec(spdx="MIT", integration_mode="vendor"), **kw)


def role(cid: str, desc: str, tools: tuple = (), handoffs: tuple = ()) -> ComponentManifest:
    return ComponentManifest(
        id=cid, kind="agent_role", name=cid.rsplit(".", 1)[-1], description=desc,
        runtime=RuntimeSpec(backend="agent"),
        inputs={"tools": list(tools), "handoffs": list(handoffs)},
        license=LicenseSpec(spdx="MIT", integration_mode="native"))


def registry() -> ComponentRegistry:
    return ComponentRegistry([
        tool("p.tool.ensembl_lookup", "lookup", "TP53 disease gene lookup by symbol"),
        tool("p.tool.opentargets", "associations", "TP53 disease associations for a gene id"),
        tool("p.tool.broken", "broken", "TP53 disease gene lookup that always fails"),
        tool("p.tool.echo", "echo", "echo arguments"),
        role("p.agent_role.manager", "manager: delegates",
             tools=("p.agent_role.genomics",), handoffs=("p.agent_role.genomics",)),
        role("p.agent_role.genomics", "genomics worker",
             tools=("p.tool.ensembl_lookup", "p.tool.opentargets")),
        role("p.agent_role.critic", "critic"),
    ])


def runtime(client=None, reg: ComponentRegistry | None = None, **kw) -> Runtime:
    reg = reg or registry()
    res = Resolver(reg)
    ld = Loader(reg, res)
    return Runtime(reg, BackendRegistry([PythonBackend(ld), AgentBackend()]),
                   resolver=res, loader=ld, llm_client=client, **kw)


def scripted_model(prompt: str) -> str:
    """A deterministic stand-in for a model, branching on prompt markers."""
    if "You are planning" in prompt:
        if "FEEDBACK FROM EARLIER ATTEMPTS" in prompt:
            return json.dumps([
                {"id": "p.tool.ensembl_lookup", "arguments": {"symbol": "TP53"}},
                {"id": "p.tool.opentargets",
                 "arguments": {"gene_id": "${steps.p.tool.ensembl_lookup.output.id}"}}])
        return json.dumps([
            {"id": "p.tool.broken", "arguments": {"symbol": "TP53"}},
            {"id": "p.tool.opentargets",
             "arguments": {"gene_id": "${steps.p.tool.broken.output.id}"}}])
    transcript = prompt.split("TRANSCRIPT SO FAR")[1] if "TRANSCRIPT SO FAR" in prompt else ""
    if "ROLE: p.agent_role.manager" in prompt:
        if "p.agent_role.genomics" in transcript:
            return json.dumps({"action": "final", "answer": "TP53 associates with EFO:0000311",
                               "evidence": ["p.agent_role.genomics"], "confidence": "moderate"})
        return json.dumps({"action": "call", "component_id": "p.agent_role.genomics",
                           "arguments": {"task": "associations for TP53"}})
    if "ROLE: p.agent_role.genomics" in prompt:
        if "p.tool.opentargets" in transcript:
            return json.dumps({"action": "final", "answer": "EFO:0000311 score 0.91",
                               "evidence": ["p.tool.ensembl_lookup", "p.tool.opentargets"],
                               "confidence": "high"})
        if "p.tool.ensembl_lookup" in transcript:
            return json.dumps({"action": "call", "component_id": "p.tool.opentargets",
                               "arguments": {"gene_id": "ENSG00000141510"}})
        return json.dumps({"action": "call", "component_id": "p.tool.ensembl_lookup",
                           "arguments": {"symbol": "TP53"}})
    if "ROLE: p.agent_role.critic" in prompt:
        return json.dumps({"action": "final", "answer": "ACCEPT — cites tool outputs"})
    if "integrating evidence" in prompt:
        return json.dumps({
            "answer": "TP53 is associated with EFO:0000311.",
            "findings": [
                {"statement": "TP53 maps to ENSG00000141510",
                 "source_step": "p.tool.ensembl_lookup", "confidence": "high"},
                {"statement": "HALLUCINATED", "source_step": "p.tool.nonexistent",
                 "confidence": "high"}],
            "conflicts": [], "uncertainty": "single source"})
    return "{}"


# =========================================================== concrete bugs
def test_container_reports_the_manifest_defect_before_the_environment() -> None:
    """The regression test for this backend failed on machines without Docker."""
    b = ContainerBackend()
    b.runtime_bin = None                                       # a machine with no runtime
    m = ComponentManifest(id="c.tool.boxed", kind="tool", name="boxed",
                          runtime=RuntimeSpec(backend="container", entrypoint="",
                                              image="example/img:1"),
                          license=LicenseSpec(spdx="MIT", integration_mode="federated"))
    res = b.invoke(m)
    assert res.status is ExecutionStatus.UNAVAILABLE
    assert "no runtime.entrypoint" in (res.error or "")


def test_subprocess_imports_an_uninstalled_upstream_checkout(tmp_path) -> None:
    """`python -I` drops the cwd from sys.path, so `cwd=<upstream root>` never
    made the upstream package importable — "invoke in place" failed with
    ModuleNotFoundError for any checkout that was not pip-installed."""
    pkg = tmp_path / "demo_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("def add(a=0, b=0):\n    return {'sum': a + b}\n")
    m = ComponentManifest(id="demo.tool.add", kind="tool", name="add",
                          provider=Provider(project="demo"),
                          runtime=RuntimeSpec(backend="subprocess", entrypoint="demo_pkg:add"),
                          license=LicenseSpec(spdx="MIT", integration_mode="federated"))
    res = SubprocessBackend(project_roots={"demo": tmp_path}).invoke(m, a=2, b=3)
    assert res.status is ExecutionStatus.SUCCEEDED, res.error
    assert res.value == {"sum": 5}, "arguments were not delivered to the entrypoint"


def test_catalogue_gives_subprocess_components_an_entrypoint_and_http_a_server() -> None:
    import pandas as pd

    from bioagent.config import catalogue_path
    from bioagent.providers.catalogue import CatalogueProvider

    rows = pd.read_csv(catalogue_path()).to_dict("records")
    ms = list(CatalogueProvider(rows).discover())
    sub = [m for m in ms if m.runtime.backend == "subprocess"]
    http = [m for m in ms if m.runtime.backend == "http"]
    roles = [m for m in ms if m.kind == "agent_role"]
    assert sub and all(":" in m.runtime.entrypoint for m in sub), \
        "subprocess components without a module:function entrypoint"
    assert http and sum(bool(m.runtime.server) for m in http) >= len(http) - 2
    assert roles and all(m.runtime.backend == "agent" for m in roles)


def test_http_component_without_a_server_is_structurally_invalid() -> None:
    m = ComponentManifest(id="d.database.x", kind="database", name="x",
                          runtime=RuntimeSpec(backend="http"),
                          license=LicenseSpec(spdx="MIT", integration_mode="native"))
    assert any("runtime.server" in e for e in m.validate())


# ================================================================ dataflow
def _results() -> dict:
    ok = CallResult(capability="public.connector.ensembl", adapter="a",
                    status=ExecutionStatus.SUCCEEDED,
                    value={"id": "ENSG1", "hits": [{"symbol": "TP53"}]})
    bad = CallResult(capability="p.tool.broken", adapter="a", status=ExecutionStatus.FAILED,
                     error="boom")
    return {"public.connector.ensembl": ok, "0": ok, "p.tool.broken": bad, "1": bad}


def test_dataflow_binds_dotted_component_ids_indexes_and_task() -> None:
    out = bind_arguments({"gene_id": "${steps.public.connector.ensembl.output.id}",
                          "sym": "${steps.0.output.hits[0].symbol}",
                          "label": "gene=${steps.0.output.id}", "t": "${task}",
                          "nested": {"ids": ["${steps.0.output.id}"]}},
                         _results(), task="T")
    assert out == {"gene_id": "ENSG1", "sym": "TP53", "label": "gene=ENSG1", "t": "T",
                   "nested": {"ids": ["ENSG1"]}}
    assert references({"a": "${steps.0.output.id}"}) == ["${steps.0.output.id}"]


@pytest.mark.parametrize("ref, fragment", [
    ("${steps.p.tool.broken.output.id}", "did not succeed"),
    ("${steps.missing.output.id}", "no completed step"),
    ("${steps.0.output.nope}", "key 'nope' not found"),
    ("${steps.0.output.hits[9]}", "out of range"),
    ("${steps.0.outpt.id}", "malformed reference"),
])
def test_dataflow_refuses_every_unresolvable_reference(ref, fragment) -> None:
    """A template that cannot be resolved must never reach a backend as text."""
    with pytest.raises(DataflowError, match=fragment):
        bind_arguments({"x": ref}, _results())


def test_a_step_consumes_an_earlier_step_output_at_run_time() -> None:
    class P(HeuristicPlanner):
        name = "chain"

        def plan(self, task, registry, *, max_steps=5, context=None):
            return Plan(task=task, planner=self.name, steps=[
                PlanStep("p.tool.ensembl_lookup", "tool", {"symbol": "TP53"}),
                PlanStep("p.tool.opentargets", "tool",
                         {"gene_id": "${steps.p.tool.ensembl_lookup.output.id}"})])

    rep = runtime().run("t", AgentSpec(name="a", max_attempts=1, synthesize=False), planner=P())
    assert [r.status for r in rep.results] == [ExecutionStatus.SUCCEEDED] * 2
    assert rep.results[1].value["gene_id"] == "ENSG00000141510"


def test_an_unresolvable_input_fails_the_step_without_calling_the_backend() -> None:
    class P(HeuristicPlanner):
        name = "bad-chain"

        def plan(self, task, registry, *, max_steps=5, context=None):
            return Plan(task=task, planner=self.name, steps=[
                PlanStep("p.tool.broken", "tool"),
                PlanStep("p.tool.echo", "tool", {"gene_id": "${steps.p.tool.broken.output.id}"})])

    rep = runtime().run("t", AgentSpec(name="a", max_attempts=1, synthesize=False), planner=P())
    echo = rep.results[1]
    assert echo.status is ExecutionStatus.CANCELLED and echo.adapter == "dataflow"
    assert not echo.executed and not echo.status.successful
    assert "did not succeed" in (echo.error or "")
    assert any(e.event_type == "DataflowFailed" for e in rep.events)


# ============================================================ planning layer
def test_heuristic_exclusion_happens_before_ranking() -> None:
    """Filtering after top-k could only shrink the plan; excluding before it
    reaches the next-best candidate."""
    reg = ComponentRegistry([tool(f"p.tool.t{i}", "echo", "TP53 disease tool") for i in range(6)])
    pl = HeuristicPlanner()
    first = pl.plan("TP53 disease", reg, max_steps=3)
    top = [s.component_id for s in first.steps]
    assert len(top) == 3
    again = pl.plan("TP53 disease", reg, max_steps=3,
                    context=PlanContext(attempt=2, excluded=frozenset(top)))
    assert len(again.steps) == 3
    assert not set(s.component_id for s in again.steps) & set(top)


def test_plan_with_context_tolerates_planners_without_a_context_parameter() -> None:
    class Old(HeuristicPlanner):
        name = "old"

        def plan(self, task, registry, *, max_steps=5):       # pre-context signature
            return Plan(task=task, planner=self.name)

    p = plan_with_context(Old(), "t", registry(), max_steps=3,
                          context=PlanContext(attempt=2))
    assert p.planner == "old"


def test_llm_planner_returns_arguments_and_receives_failure_feedback() -> None:
    seen: list[str] = []

    def model(prompt: str) -> str:
        seen.append(prompt)
        return scripted_model(prompt)

    rep = runtime(model).run("TP53 disease associations",
                             AgentSpec(name="a", planner="llm", max_attempts=2, model="fake"))
    assert rep.plan.planner == "llm" and not rep.plan.degraded
    assert rep.attempts == 2
    feedback = seen[1]
    assert "FEEDBACK FROM EARLIER ATTEMPTS" in feedback
    assert "upstream exploded" in feedback and "Do not use: p.tool.broken" in feedback
    assert [s.arguments for s in rep.plan.steps] == [
        {"symbol": "TP53"}, {"gene_id": "${steps.p.tool.ensembl_lookup.output.id}"}]
    assert rep.results[1].value["gene_id"] == "ENSG00000141510"
    assert rep.execution_outcome == RunOutcome.SUCCESS.value


def test_retry_excludes_only_components_that_failed_on_their_own() -> None:
    """A step blocked by an unresolved input is not a broken component."""
    rep = runtime(scripted_model).run("TP53 disease associations",
                                      AgentSpec(name="a", planner="llm", max_attempts=2))
    ids = [r.capability for r in rep.results]
    assert "p.tool.opentargets" in ids, "the consumer was excluded for the producer's failure"
    assert "p.tool.broken" not in ids


def test_llm_planner_falls_back_loudly_without_a_client() -> None:
    rep = runtime(None).run("TP53 disease", AgentSpec(name="a", planner="llm", max_attempts=1))
    assert rep.plan.planner == "llm->heuristic"
    assert "no model client bound" in rep.plan.degraded
    assert rep.summary()["planner_degraded"]
    assert sum(e.event_type == "PlannerFallback" for e in rep.events) == 1


def test_require_model_refuses_to_run_without_a_client() -> None:
    with pytest.raises(RuntimeError, match="requires a model"):
        runtime(None).run("t", AgentSpec(name="a", planner="llm", require_model=True))


def test_runtime_hands_the_planner_its_client_and_model() -> None:
    """`AgentSpec(planner="llm")` used to construct `LLMPlanner()` with no client."""
    rt = runtime(scripted_model)
    pl = rt._planner_for(AgentSpec(name="a", planner="llm", model="m-1"))
    assert isinstance(pl, LLMPlanner)
    assert pl.client_bound and pl.model == "m-1"


# ================================================================ agents
def test_agent_backend_is_unavailable_without_a_model_and_says_so() -> None:
    rt = runtime(None)
    res = rt.invoke("p.agent_role.genomics", spec=AgentSpec(name="a"), task="t")
    assert res.status is ExecutionStatus.UNAVAILABLE
    assert "no model client bound" in (res.error or "")


def test_agent_role_runs_a_tool_loop_through_the_policy_gate() -> None:
    rt = runtime(scripted_model)
    from bioagent.runtime.events import EventLog

    ev = EventLog()
    res = rt.invoke("p.agent_role.genomics", spec=AgentSpec(name="a"), events=ev,
                    task="associations for TP53")
    assert res.status is ExecutionStatus.SUCCEEDED, res.error
    calls = [c["component_id"] for c in res.value["tool_calls"]]
    assert calls == ["p.tool.ensembl_lookup", "p.tool.opentargets"]
    assert res.value["evidence"] == calls
    # every tool the agent called went through resolve -> POLICY -> backend
    assert sum(e.event_type == "PolicyChecked" for e in ev) >= 3
    assert sum(e.event_type == "AgentTurn" for e in ev) == 3


def test_agent_cannot_call_a_tool_outside_its_allowlist() -> None:
    def model(prompt: str) -> str:
        if "not among this role's tools" in prompt:
            return json.dumps({"action": "final", "answer": "gave up", "evidence": []})
        return json.dumps({"action": "call", "component_id": "p.tool.echo", "arguments": {}})

    res = runtime(model).invoke("p.agent_role.genomics", spec=AgentSpec(name="a"), task="t")
    assert res.status is ExecutionStatus.SUCCEEDED
    assert res.value["tool_calls"] == [], "a disallowed tool was executed"


def test_agent_that_never_finishes_is_a_failure_not_a_partial_success() -> None:
    model = lambda prompt: json.dumps({"action": "call", "component_id": "p.tool.ensembl_lookup",  # noqa: E731
                                       "arguments": {"symbol": "TP53"}})
    res = runtime(model).invoke("p.agent_role.genomics", spec=AgentSpec(name="a"), task="t",
                                max_turns=2)
    assert res.status is ExecutionStatus.FAILED
    assert "no final answer within 2" in (res.error or "")


def test_agent_handoff_returns_the_delegate_result_and_records_it() -> None:
    def model(prompt: str) -> str:
        if "ROLE: p.agent_role.manager" in prompt:
            return json.dumps({"action": "handoff", "to": "p.agent_role.genomics", "task": "t2"})
        return scripted_model(prompt)

    res = runtime(model).invoke("p.agent_role.manager", spec=AgentSpec(name="a"), task="t")
    assert res.status is ExecutionStatus.SUCCEEDED
    assert res.value["handed_off_from"] == "p.agent_role.manager"
    assert res.value["answer"] == "EFO:0000311 score 0.91"
    assert res.value["tool_calls"][0]["handoff"] is True


def test_agent_delegation_depth_is_bounded() -> None:
    reg = ComponentRegistry([role("p.agent_role.a", "a", tools=("p.agent_role.b",)),
                             role("p.agent_role.b", "b", tools=("p.agent_role.a",))])
    model = lambda prompt: json.dumps({"action": "call",  # noqa: E731
                                       "component_id": ("p.agent_role.b" if "ROLE: p.agent_role.a" in prompt
                                                        else "p.agent_role.a"),
                                       "arguments": {}})
    res = runtime(model, reg).invoke("p.agent_role.a", spec=AgentSpec(name="a"), task="t",
                                     max_turns=1)
    assert res.status is ExecutionStatus.FAILED           # terminated, not recursed forever


def test_remote_agent_backend_is_honest_without_a_transport() -> None:
    m = ComponentManifest(id="r.agent_role.x", kind="agent_role", name="x",
                          runtime=RuntimeSpec(backend="remote_agent", server="https://agents.example"),
                          license=LicenseSpec(spdx="MIT", integration_mode="federated"))
    assert RemoteAgentBackend().invoke(m, task="t").status is ExecutionStatus.UNAVAILABLE
    seen = {}
    b = RemoteAgentBackend(transport=lambda server, agent, **kw: seen.update(kw) or {"ok": 1})
    res = b.invoke(m, task="t", spec=object(), events=None)
    assert res.status is ExecutionStatus.SUCCEEDED and seen == {"task": "t"}


# ============================================================ orchestration
def test_orchestrator_runs_manager_worker_critic_with_shared_memory() -> None:
    o = Orchestrator(runtime(scripted_model)).run(
        "TP53 disease associations", manager="p.agent_role.manager",
        workers=["p.agent_role.genomics"], critic="p.agent_role.critic")
    assert o.ok and o.critic_accepted is True and o.rounds == 1
    kinds = [(m.kind, m.sender.rsplit(".", 1)[-1]) for m in o.transcript]
    assert ("task", "manager") in kinds and ("result", "genomics") in kinds
    assert ("critique", "critic") in kinds
    assert "p.agent_role.genomics.p.tool.opentargets" in o.working_memory["slots"]
    # synthesis cites the leaf tool the worker called, not just the worker
    assert [f.source_step for f in o.synthesis.findings] == ["p.tool.ensembl_lookup"]
    assert o.synthesis.verdict == ScientificVerdict.INCONCLUSIVE.value
    assert sum(e.event_type == "AgentMessage" for e in o.events) == len(o.transcript)


def test_a_rejecting_critic_triggers_a_revision_round() -> None:
    critic_calls = {"n": 0}

    def model(prompt: str) -> str:
        if "ROLE: p.agent_role.critic" in prompt:
            critic_calls["n"] += 1
            verdict = "REJECT — no evidence" if critic_calls["n"] == 1 else "ACCEPT"
            return json.dumps({"action": "final", "answer": verdict})
        if "ROLE: p.agent_role.manager" in prompt and "REVISE" not in prompt \
                and "Critic feedback" not in prompt:
            return json.dumps({"action": "final", "answer": "first draft", "evidence": []})
        return scripted_model(prompt)

    o = Orchestrator(runtime(model), max_rounds=2).run(
        "TP53 disease associations", manager="p.agent_role.manager",
        workers=["p.agent_role.genomics"], critic="p.agent_role.critic")
    assert o.rounds == 2 and o.critic_accepted is True
    assert [m.status for m in o.transcript if m.kind == "critique"] == ["REJECT", "ACCEPT"]


def test_orchestrator_reports_unavailability_instead_of_simulating_a_team() -> None:
    o = Orchestrator(runtime(None)).run("t", manager="p.agent_role.manager")
    assert not o.ok and "no model client bound" in o.error
    assert o.transcript == []


# ================================================================== memory
def test_episodic_memory_writes_through_the_workspace_and_searches() -> None:
    from bioagent.workspace import Workspace

    ws = Workspace(Path(tempfile.mkdtemp()) / "ws").init()
    mem = EpisodicMemory(workspace=ws)
    mem.write("TP53 lookups should use homo_sapiens", kind="fact", tags=("ensembl",))
    mem.write("unrelated note about STRING", kind="note")
    hits = mem.search("TP53 ensembl", k=3)
    assert [h.kind for h in hits] == ["fact"]
    assert ws.exists("memory/episodic/entries.jsonl")
    assert len(EpisodicMemory(workspace=ws)) == 2        # persisted, re-loadable


def test_runs_are_remembered_and_recalled() -> None:
    mem = EpisodicMemory(path=Path(tempfile.mkdtemp()) / "m.jsonl")
    rt = runtime(scripted_model, memory=mem)
    rep = rt.run("TP53 disease associations", AgentSpec(name="a", planner="llm", max_attempts=2))
    assert any(e.event_type == "MemoryWritten" for e in rep.events)
    assert len(mem) == 1 and mem.recent(1)[0].kind == "episode"
    assert "TP53" in mem.search("TP53", k=1)[0].text


def test_working_memory_is_a_bounded_shared_scratchpad() -> None:
    wm = WorkingMemory()
    wm.put("gene", {"id": "ENSG1"}, by="worker")
    wm.note("consider paralogs", by="critic")
    assert wm.get("gene") == {"id": "ENSG1"}
    text = wm.describe(limit=80)
    assert len(text) <= 80 and "gene" in text
    assert wm.snapshot()["n_ops"] == 1


# =============================================================== synthesis
def test_synthesis_without_a_model_aggregates_and_never_accepts() -> None:
    rep = runtime(None).run("TP53 disease", AgentSpec(name="a", max_attempts=1))
    assert rep.synthesis is not None and rep.synthesis.method == "deterministic"
    assert rep.synthesis.verdict == ScientificVerdict.INCONCLUSIVE.value
    assert all(f.source_step for f in rep.synthesis.findings)
    assert rep.summary()["answer"]


def test_model_synthesis_keeps_only_findings_that_cite_a_real_step() -> None:
    results = [CallResult(capability="p.tool.ensembl_lookup", adapter="a",
                          status=ExecutionStatus.SUCCEEDED, value={"id": "ENSG1"})]
    s = EvidenceSynthesizer(scripted_model, model="fake").synthesize("t", results)
    assert s.method == "model"
    assert [f.source_step for f in s.findings] == ["p.tool.ensembl_lookup"]
    assert s.verdict == ScientificVerdict.INCONCLUSIVE.value


def test_model_synthesis_failure_degrades_to_aggregation() -> None:
    results = [CallResult(capability="x", adapter="a", status=ExecutionStatus.SUCCEEDED, value=1)]
    s = EvidenceSynthesizer(lambda p: "not json").synthesize("t", results)
    assert s.method == "deterministic" and "model synthesis failed" in s.uncertainty


# ============================================================= http dispatch
def test_http_backend_dispatches_a_typed_operation() -> None:
    from bioagent.backends.http import HTTPBackend
    from bioagent.providers.public_apis import PublicAPIProvider

    ens = next(m for m in PublicAPIProvider().discover() if m.id == "public.connector.ensembl")
    b = HTTPBackend(cache_dir=Path(tempfile.mkdtemp()))
    captured = {}

    def fake_request(req, use_cache=True):
        captured["req"] = req
        return ExecutionStatus.SUCCEEDED, {"id": "ENSG1"}, None, {}

    b.request = fake_request
    res = b.invoke(ens, operation="gene_lookup", symbol="TP53", species="homo_sapiens")
    assert res.status is ExecutionStatus.SUCCEEDED
    assert captured["req"].full_url.startswith(
        "https://rest.ensembl.org/lookup/symbol/homo_sapiens/TP53")
    assert "no operation 'nope'" in (b.invoke(ens, operation="nope").error or "")
    assert "requires ['symbol']" in (b.invoke(ens, operation="gene_lookup").error or "")


def test_operation_dispatch_is_refused_for_untyped_http_components() -> None:
    from bioagent.backends.http import HTTPBackend

    m = ComponentManifest(id="d.database.x", kind="database", name="x",
                          runtime=RuntimeSpec(backend="http", server="https://example.invalid"),
                          permissions=__import__("bioagent.runtime.component", fromlist=["Permissions"])
                          .Permissions(network=("example.invalid",)),
                          license=LicenseSpec(spdx="MIT", integration_mode="native"))
    res = HTTPBackend(cache_dir=Path(tempfile.mkdtemp())).invoke(m, operation="anything")
    assert res.status is ExecutionStatus.FAILED and "not a typed public connector" in res.error
