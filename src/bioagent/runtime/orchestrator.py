"""Multi-agent orchestration — manager, workers, critic, shared memory.

`Runtime.run()` is planner → ordered component calls → critique. That is one
agent. The projects the catalogue federates (PantheonOS, STELLA, GenoMAS,
CellAgent, BioMedAgent, …) are built on manager/worker teams, delegation,
handoffs and a critic. This module runs such a team over the same runtime.

The mechanics are deliberately plain:

* **Roles are components.** Manager, workers and critic are `agent_role`
  manifests on the `agent` backend; the manager's tools are the workers, so
  delegation is an ordinary policy-gated `Runtime.invoke` (agent-as-tool).
* **Messages are data.** Every task sent, result returned, handoff and critique
  is an `AgentMessage` in the transcript, and each is also an `AgentMessage`
  event in the run's provenance graph.
* **Memory is shared.** One `WorkingMemory` is handed to every role; anything a
  worker records is visible to the manager's next turn and to the critic.
* **The critic gates.** After the manager answers, the critic (if any) judges
  it; a rejection triggers one revision round with the critique as feedback,
  bounded by `max_rounds`.
* **Synthesis closes.** The final answer with per-finding provenance comes from
  `EvidenceSynthesizer`, and its verdict is never ACCEPTED — a critic's
  acceptance is recorded as the critic's, not promoted to a scientific finding.

Without a model client the whole thing is UNAVAILABLE and reports why. It does
not simulate a team.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from ..adapters.base import CallResult
from ..status import ExecutionStatus
from .events import EventLog, EventType
from .memory import WorkingMemory
from .synthesis import EvidenceSynthesizer, Synthesis


@dataclass
class AgentMessage:
    kind: str                      # task | result | handoff | critique | final
    sender: str
    recipient: str = ""
    content: Any = ""
    status: str = ""
    msg_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OrchestrationReport:
    task: str
    manager: str
    workers: tuple[str, ...]
    critic: str | None
    transcript: list[AgentMessage] = field(default_factory=list)
    manager_result: CallResult | None = None
    critic_result: CallResult | None = None
    rounds: int = 0
    critic_accepted: bool | None = None
    synthesis: Synthesis | None = None
    working_memory: dict[str, Any] = field(default_factory=dict)
    events: EventLog | None = None
    error: str = ""

    @property
    def answer(self) -> str:
        if self.manager_result is not None and isinstance(self.manager_result.value, dict):
            return str(self.manager_result.value.get("answer", ""))
        return ""

    @property
    def ok(self) -> bool:
        return (self.manager_result is not None and self.manager_result.status.successful
                and self.critic_accepted is not False)

    def summary(self) -> dict[str, Any]:
        return {
            "task": self.task, "manager": self.manager, "workers": list(self.workers),
            "critic": self.critic, "rounds": self.rounds, "ok": self.ok,
            "manager_status": (self.manager_result.status.value
                               if self.manager_result else None),
            "critic_accepted": self.critic_accepted,
            "n_messages": len(self.transcript),
            "delegations": sum(1 for m in self.transcript if m.kind == "task"),
            "handoffs": sum(1 for m in self.transcript if m.kind == "handoff"),
            "verdict": self.synthesis.verdict if self.synthesis else "INCONCLUSIVE",
            "answer": self.answer[:300], "error": self.error or None,
        }


class Orchestrator:
    """Runs a manager/worker/critic team of agent roles over a runtime."""

    def __init__(self, runtime: Any, *, max_rounds: int = 2) -> None:
        self.runtime = runtime
        self.max_rounds = max_rounds

    def _check(self, role_ids: Sequence[str]) -> str:
        reg = self.runtime.registry
        for rid in role_ids:
            m = reg.get(rid)
            if m is None:
                return f"role {rid!r} is not in the registry"
            if m.runtime.backend != "agent":
                return f"role {rid!r} is on backend {m.runtime.backend!r}, not 'agent'"
        backend = self.runtime.backends.get("agent")
        if backend is None:
            return "no agent backend registered"
        if not backend.available():
            return backend.unavailable_reason()
        return ""

    def run(self, task: str, *, manager: str, workers: Sequence[str] = (),
            critic: str | None = None, spec: Any = None,
            tools: Sequence[str] = ()) -> OrchestrationReport:
        from .agentspec import AgentSpec

        spec = spec or AgentSpec(name=f"team:{manager}")
        report = OrchestrationReport(task=task, manager=manager, workers=tuple(workers),
                                     critic=critic)
        problem = self._check([manager, *workers, *([critic] if critic else [])])
        if problem:
            report.error = problem
            return report

        events = EventLog(catalogue_version=self.runtime.catalogue_version,
                          git_commit=self.runtime.git_commit)
        report.events = events
        root = events.emit(EventType.TASK_CREATED, inputs={"task": task},
                           detail={"team": {"manager": manager, "workers": list(workers),
                                            "critic": critic}}).event_id
        memory = WorkingMemory()
        raw_transcript: list[dict] = []
        feedback = ""

        for round_no in range(1, self.max_rounds + 1):
            report.rounds = round_no
            self._send(report, events, root, AgentMessage(
                "task", sender="orchestrator", recipient=manager,
                content=task if not feedback else f"{task}\n\nREVISE. Critic said: {feedback}"))
            res = self.runtime.invoke(
                manager, spec=spec, events=events, parent_event=root,
                task=task, context=(f"Critic feedback from round {round_no - 1}: {feedback}"
                                    if feedback else ""),
                tools=[*workers, *tools], handoffs=list(workers),
                transcript=raw_transcript, working_memory=memory)
            report.manager_result = res
            self._drain(report, events, root, raw_transcript)
            self._send(report, events, root, AgentMessage(
                "result", sender=manager, recipient="orchestrator",
                content=(res.value.get("answer") if isinstance(res.value, dict) else res.error),
                status=res.status.value))
            if not res.status.successful:
                break

            if not critic:
                report.critic_accepted = None
                break
            crit = self.runtime.invoke(
                critic, spec=spec, events=events, parent_event=root,
                task=(f"Critically evaluate this answer to the task. TASK: {task}\n"
                      f"ANSWER: {report.answer}\nReply with a final answer that begins with "
                      "ACCEPT or REJECT, followed by your reasons."),
                context="", tools=list(tools), transcript=raw_transcript,
                working_memory=memory)
            report.critic_result = crit
            self._drain(report, events, root, raw_transcript)
            verdict_text = (crit.value.get("answer", "") if isinstance(crit.value, dict)
                            else (crit.error or ""))
            accepted = crit.status.successful and verdict_text.strip().upper().startswith("ACCEPT")
            report.critic_accepted = bool(accepted) if crit.status.successful else None
            self._send(report, events, root, AgentMessage(
                "critique", sender=critic, recipient=manager, content=verdict_text,
                status="ACCEPT" if accepted else "REJECT"))
            if accepted or not crit.status.successful:
                break
            feedback = verdict_text[:800]

        report.working_memory = memory.snapshot()
        synth = EvidenceSynthesizer(self.runtime.llm_client, model=spec.model)
        evidence_results = self._evidence(report)
        report.synthesis = synth.synthesize(task, evidence_results)
        events.emit(EventType.SYNTHESIS_COMPLETED, parent=root,
                    status=report.synthesis.method.upper(),
                    output=report.synthesis.to_dict(),
                    detail={"n_findings": len(report.synthesis.findings)})
        events.emit(EventType.RUN_COMPLETED, parent=root,
                    status="SUCCESS" if report.ok else "FAILED", detail=report.summary())
        return report

    # ---------------------------------------------------------------- plumbing
    def _send(self, report: OrchestrationReport, events: EventLog, root: str,
              msg: AgentMessage) -> None:
        report.transcript.append(msg)
        events.emit(EventType.AGENT_MESSAGE, parent=root, component_id=msg.sender,
                    status=msg.kind.upper(),
                    detail={"recipient": msg.recipient, "status": msg.status,
                            "content": str(msg.content)[:400], "msg_id": msg.msg_id})

    def _drain(self, report: OrchestrationReport, events: EventLog, root: str,
               raw: list[dict]) -> None:
        """Lift the agents' own transcript entries into typed messages."""
        seen = {m.msg_id for m in report.transcript}
        for entry in raw:
            key = entry.get("_msg_id")
            if key in seen:
                continue
            msg = AgentMessage(kind=str(entry.get("kind", "note")),
                               sender=str(entry.get("sender", "")),
                               recipient=str(entry.get("recipient", "")),
                               content=entry.get("content", ""),
                               status=str(entry.get("status", "")))
            entry["_msg_id"] = msg.msg_id
            self._send(report, events, root, msg)

    @staticmethod
    def _evidence(report: OrchestrationReport) -> list[CallResult]:
        """Every tool call any role made — nested delegations flattened — as
        synthesis evidence, so findings can cite the leaf tool, not the agent."""
        out: list[CallResult] = []
        seen: set[tuple[str, str]] = set()

        def walk(calls: list[dict]) -> None:
            for call in calls:
                cid = str(call.get("component_id"))
                key = (cid, str(call.get("output"))[:200])
                if key not in seen:
                    seen.add(key)
                    try:
                        status = ExecutionStatus(call.get("status", "FAILED"))
                    except ValueError:
                        status = ExecutionStatus.FAILED
                    out.append(CallResult(capability=cid, adapter="agent", status=status,
                                          value=call.get("output"), error=call.get("error")))
                walk(list(call.get("nested_tool_calls") or []))

        for res in (report.manager_result, report.critic_result):
            if res is not None and isinstance(res.value, dict):
                walk(list(res.value.get("tool_calls", [])))
        if report.manager_result is not None and report.manager_result.status.successful:
            out.append(CallResult(capability=report.manager, adapter="agent",
                                  status=ExecutionStatus.SUCCEEDED,
                                  value={"answer": report.answer}))
        return out
