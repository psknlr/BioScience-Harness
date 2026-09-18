"""Agent-role execution — a catalogued role becomes a running agent.

The catalogue carries 53 `agent_role` components (manager, critic, planner,
tool-creation and domain-specialist agents from the surveyed projects). They
were mapped to `backend="none"`, so every one of them was permanently
non-executable metadata. This backend executes them.

An agent role runs as a bounded tool-use loop over a model client: the role's
description (or `inputs.system_prompt`) is the system instruction; the
components it is allowed to call (`requires.components` plus `inputs.tools`)
are its tools; every tool call goes back through `Runtime.invoke`, so it is
resolved, policy-gated and recorded like any other call. Other agent roles may
be among its tools — that is agent-as-tool, and it is how a manager delegates.
A role may also **hand off** the whole task to another role it is permitted to
reach, and the result is returned as its own with the handoff recorded.

What it is not: without a model client bound it reports UNAVAILABLE with that
reason. It never pretends a role ran. A model that never reaches a final
answer within `max_turns` is a FAILED call, not a partial success.
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Mapping

from ..adapters.base import CallResult
from ..runtime.component import ComponentManifest
from ..status import ExecutionStatus
from .base import Backend

#: keyword arguments that belong to orchestration, never to a tool's signature
_ORCHESTRATION_KWARGS = ("task", "context", "tools", "handoffs", "max_turns", "transcript",
                         "working_memory", "depth", "spec", "events", "parent_event")


def _digest(value: Any, limit: int = 600) -> Any:
    try:
        blob = json.dumps(value, default=str)
    except (TypeError, ValueError):
        blob = repr(value)
    if len(blob) <= limit:
        try:
            return json.loads(blob)
        except ValueError:
            return blob
    return blob[:limit] + "…"


class AgentBackend(Backend):
    """Runs `agent_role` components as bounded, policy-gated tool-use loops."""

    backend = "agent"
    #: Runtime.invoke forwards spec/events/parent_event to backends that set this.
    wants_runtime_context = True

    def __init__(self, client: Callable[[str], str] | None = None, *, max_turns: int = 6,
                 max_depth: int = 3, model: str = "") -> None:
        self._client = client
        self._runtime: Any = None
        self.max_turns = max_turns
        self.max_depth = max_depth
        self.model = model

    # ------------------------------------------------------------- wiring
    def bind_runtime(self, runtime: Any) -> None:
        self._runtime = runtime
        if self._client is None:
            self._client = getattr(runtime, "llm_client", None)

    @property
    def client(self) -> Callable[[str], str] | None:
        return self._client

    def available(self) -> bool:
        return self._client is not None and self._runtime is not None

    def unavailable_reason(self) -> str:
        if self._runtime is None:
            return "agent backend is not bound to a runtime"
        if self._client is None:
            return ("no model client bound: pass Runtime(llm_client=...) or "
                    "AgentBackend(client=...) — an agent role cannot run without a model")
        return ""

    # ------------------------------------------------------------ helpers
    @staticmethod
    def allowed_tools(manifest: ComponentManifest, extra: Any = None) -> list[str]:
        declared = list(manifest.requires.components)
        inputs = manifest.inputs if isinstance(manifest.inputs, Mapping) else {}
        for t in inputs.get("tools", ()) or ():
            if t not in declared:
                declared.append(str(t))
        for t in extra or ():
            if t not in declared:
                declared.append(str(t))
        return declared

    @staticmethod
    def handoff_targets(manifest: ComponentManifest, extra: Any = None) -> list[str]:
        inputs = manifest.inputs if isinstance(manifest.inputs, Mapping) else {}
        out = [str(t) for t in (inputs.get("handoffs", ()) or ())]
        for t in extra or ():
            if t not in out:
                out.append(str(t))
        return out

    def _describe_tool(self, cid: str) -> dict[str, Any]:
        from ..planners.llm import describe_arguments

        m = self._runtime.registry.get(cid) if self._runtime else None
        if m is None:
            return {"id": cid, "note": "not in registry"}
        return {"id": cid, "kind": m.kind, "description": m.description[:160],
                "state": m.state.value, **describe_arguments(m)}

    def _prompt(self, manifest: ComponentManifest, task: str, context: str, tools: list[str],
                handoffs: list[str], transcript: list[dict], memory_text: str) -> str:
        inputs = manifest.inputs if isinstance(manifest.inputs, Mapping) else {}
        system = str(inputs.get("system_prompt") or manifest.description or manifest.name)
        menu = [self._describe_tool(t) for t in tools]
        lines = [
            f"ROLE: {manifest.id}",
            f"You are: {system}",
            f"TASK: {task}",
        ]
        if context:
            lines.append(f"CONTEXT:\n{context}")
        if memory_text:
            lines.append(f"SHARED WORKING MEMORY:\n{memory_text}")
        lines.append(f"TOOLS YOU MAY CALL (JSON):\n{json.dumps(menu, indent=1)[:7000]}")
        if handoffs:
            lines.append("ROLES YOU MAY HAND THE WHOLE TASK TO: " + ", ".join(handoffs))
        if transcript:
            lines.append("TRANSCRIPT SO FAR (JSON):\n"
                         + json.dumps(transcript, indent=1, default=str)[-6000:])
        lines.append(
            "Reply with ONLY one JSON object. Either call a tool: "
            '{"action": "call", "component_id": "...", "arguments": {...}, "why": "..."} '
            "— or hand off: "
            '{"action": "handoff", "to": "<role id>", "task": "...", "why": "..."} '
            "— or finish: "
            '{"action": "final", "answer": "...", "evidence": ["<component_id>", ...], '
            '"confidence": "low|moderate|high"}. '
            "Cite only tools you actually called in `evidence`. Do not invent results.")
        return "\n\n".join(lines)

    @staticmethod
    def _extract(text: str) -> dict:
        t = text.strip()
        start, end = t.find("{"), t.rfind("}")
        if start >= 0 and end > start:
            obj = json.loads(t[start:end + 1])
            if isinstance(obj, dict):
                return obj
        raise ValueError("no JSON object in model response")

    # ------------------------------------------------------------- invoke
    def invoke(self, manifest: ComponentManifest, *, task: str = "", context: str = "",
               tools: Any = None, handoffs: Any = None, max_turns: int | None = None,
               transcript: list | None = None, working_memory: Any = None, depth: int = 0,
               spec: Any = None, events: Any = None, parent_event: str | None = None,
               **kwargs: Any) -> CallResult:
        t0 = time.perf_counter()
        if not self.available():
            return self._result(manifest, ExecutionStatus.UNAVAILABLE, t0,
                                error=self.unavailable_reason())
        if depth > self.max_depth:
            return self._result(manifest, ExecutionStatus.FAILED, t0,
                                error=f"agent delegation deeper than max_depth={self.max_depth}")
        from ..runtime.agentspec import AgentSpec

        spec = spec or AgentSpec(name=manifest.id, model=self.model)
        task = task or str(kwargs.get("query") or kwargs.get("input") or "")
        if not task:
            return self._result(manifest, ExecutionStatus.FAILED, t0,
                                error="agent role invoked without a task")
        allowed = self.allowed_tools(manifest, tools)
        targets = self.handoff_targets(manifest, handoffs)
        turns_limit = max_turns or self.max_turns
        transcript = transcript if transcript is not None else []
        local: list[dict] = []
        tool_calls: list[dict] = []
        turn_ids: list[str] = []

        for turn in range(1, turns_limit + 1):
            memory_text = working_memory.describe() if working_memory is not None else ""
            prompt = self._prompt(manifest, task, context, allowed, targets, local, memory_text)
            try:
                raw = self._client(prompt)
                action = self._extract(raw)
            except Exception as exc:  # noqa: BLE001 - a bad model turn is an observation
                local.append({"turn": turn, "error": f"{type(exc).__name__}: {exc}"})
                continue
            kind = str(action.get("action", "")).lower()
            if events is not None:
                ev = events.emit("AgentTurn", parent=parent_event, component_id=manifest.id,
                                 status=kind.upper() or "MALFORMED", model=spec.model,
                                 detail={"turn": turn, "depth": depth,
                                         "why": str(action.get("why", ""))[:200]})
                turn_ids.append(ev.event_id)

            if kind == "final":
                evidence = [e for e in (action.get("evidence") or [])
                            if any(c["component_id"] == e for c in tool_calls)]
                answer = str(action.get("answer", "")).strip()
                if not answer:
                    local.append({"turn": turn, "error": "final answer was empty"})
                    continue
                value = {"answer": answer, "evidence": evidence,
                         "confidence": (action.get("confidence")
                                        if action.get("confidence") in ("low", "moderate", "high")
                                        else "unassessed"),
                         "tool_calls": tool_calls, "turns": turn, "role": manifest.id}
                transcript.append({"kind": "final", "sender": manifest.id, "content": answer,
                                   "evidence": evidence})
                if working_memory is not None:
                    working_memory.put(f"{manifest.id}.answer", answer, by=manifest.id)
                return self._result(manifest, ExecutionStatus.SUCCEEDED, t0, value=value,
                                    metadata={"turns": turn, "n_tool_calls": len(tool_calls)})

            if kind == "handoff":
                to = str(action.get("to", ""))
                if to not in targets:
                    local.append({"turn": turn, "error": f"handoff to {to!r} not permitted; "
                                  f"allowed: {targets}"})
                    continue
                sub_task = str(action.get("task") or task)
                transcript.append({"kind": "handoff", "sender": manifest.id, "recipient": to,
                                   "content": sub_task})
                sub = self._runtime.invoke(
                    to, spec=spec, events=events, parent_event=parent_event,
                    task=sub_task, context=context, transcript=transcript,
                    working_memory=working_memory, depth=depth + 1)
                tool_calls.append({"component_id": to, "arguments": {"task": sub_task},
                                   "status": sub.status.value, "output": _digest(sub.value),
                                   "error": sub.error, "handoff": True,
                                   "nested_tool_calls": (list(sub.value.get("tool_calls", []))
                                                         if isinstance(sub.value, dict) else [])})
                if sub.status.successful and isinstance(sub.value, dict):
                    value = {**sub.value, "handed_off_from": manifest.id, "role": manifest.id,
                             "tool_calls": tool_calls, "turns": turn}
                    return self._result(manifest, ExecutionStatus.SUCCEEDED, t0, value=value,
                                        metadata={"turns": turn, "handoff": to})
                local.append({"turn": turn, "handoff": to, "status": sub.status.value,
                              "error": sub.error})
                continue

            if kind == "call":
                cid = str(action.get("component_id", ""))
                args = action.get("arguments") if isinstance(action.get("arguments"), dict) else {}
                if cid not in allowed:
                    local.append({"turn": turn, "error": f"{cid!r} is not among this role's "
                                  f"tools; allowed: {allowed}"})
                    continue
                target = self._runtime.registry.get(cid)
                nested = bool(target is not None and target.runtime.backend == "agent")
                fwd = dict(args)
                if nested:
                    fwd.update({"task": str(args.get("task") or task), "context": context,
                                "transcript": transcript, "working_memory": working_memory,
                                "depth": depth + 1})
                    transcript.append({"kind": "task", "sender": manifest.id, "recipient": cid,
                                       "content": fwd["task"]})
                res = self._runtime.invoke(cid, spec=spec, events=events,
                                           parent_event=parent_event, **fwd)
                record = {"component_id": cid, "arguments": _digest(args, 300),
                          "status": res.status.value, "output": _digest(res.value),
                          "error": res.error}
                if nested and isinstance(res.value, dict):
                    # keep the delegate's own calls intact (digest would truncate
                    # them) so leaf-level evidence survives to synthesis
                    record["nested_tool_calls"] = list(res.value.get("tool_calls", []))
                tool_calls.append(record)
                local.append({"turn": turn, "call": record})
                if nested:
                    transcript.append({"kind": "result", "sender": cid, "recipient": manifest.id,
                                       "content": _digest(res.value, 400),
                                       "status": res.status.value})
                if working_memory is not None and res.status.successful:
                    working_memory.put(f"{manifest.id}.{cid}", _digest(res.value), by=manifest.id)
                continue

            local.append({"turn": turn, "error": f"unknown action {kind!r}"})

        return self._result(
            manifest, ExecutionStatus.FAILED, t0,
            value={"tool_calls": tool_calls, "turns": turns_limit, "role": manifest.id},
            error=f"agent {manifest.id} produced no final answer within {turns_limit} turn(s)",
            metadata={"turns": turns_limit, "n_tool_calls": len(tool_calls)})


class RemoteAgentBackend(Backend):
    """Delegates to an agent running elsewhere, through an injected transport.

    `remote_agent` was a declared backend name with no implementation. As with
    `MCPBackend`, the transport is injected: `transport(server, agent, **kwargs)`
    returns the remote result. With none bound the backend is unavailable and
    says so — it never reports a remote call it did not make.
    """

    backend = "remote_agent"

    def __init__(self, transport: Callable[..., Any] | None = None) -> None:
        self._transport = transport

    def available(self) -> bool:
        return self._transport is not None

    def unavailable_reason(self) -> str:
        return "" if self._transport else "no remote-agent transport bound to this runtime"

    def invoke(self, manifest: ComponentManifest, **kwargs: Any) -> CallResult:
        t0 = time.perf_counter()
        if not self.available():
            return self._result(manifest, ExecutionStatus.UNAVAILABLE, t0,
                                error=self.unavailable_reason())
        server = manifest.runtime.server
        if not server:
            return self._result(manifest, ExecutionStatus.UNAVAILABLE, t0,
                                error="component declares no runtime.server for the remote agent")
        clean = {k: v for k, v in kwargs.items() if k not in _ORCHESTRATION_KWARGS
                 or k in ("task", "context")}
        try:
            value = self._transport(server, manifest.runtime.entrypoint or manifest.name, **clean)
            return self._result(manifest, ExecutionStatus.SUCCEEDED, t0, value=value,
                                metadata={"server": server})
        except Exception as exc:  # noqa: BLE001
            return self._result(manifest, ExecutionStatus.FAILED, t0,
                                error=f"{type(exc).__name__}: {exc}", metadata={"server": server})
