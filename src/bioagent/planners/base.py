"""Planner plugin interface and registry.

v1 hardcoded a retrieval planner inside the agent, so swapping in an LLM planner
meant editing the agent. Here planners register themselves and an AgentSpec names
one, so the runtime never imports a specific planner.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence


@dataclass
class PlanStep:
    """One planned component invocation."""

    component_id: str
    kind: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""


@dataclass
class Plan:
    task: str
    steps: list[PlanStep] = field(default_factory=list)
    planner: str = ""
    notes: str = ""
    #: Non-empty when the plan was NOT produced by the planner the spec asked
    #: for. `AgentSpec(planner="llm")` used to fall back to heuristic ordering
    #: silently; the runtime now records this in the event log and the report.
    degraded: str = ""

    def __len__(self) -> int:
        return len(self.steps)


@dataclass(frozen=True)
class PlanContext:
    """What a planner should know about the attempts before this one.

    Retrying used to mean "plan again from nothing and delete the failed steps
    afterwards": the planner never saw the error, the critique's retry hint, or
    the ids to avoid, and because exclusion happened *after* top-k selection a
    plan of five could only ever shrink toward zero instead of reaching for the
    sixth candidate.
    """

    attempt: int = 1
    excluded: frozenset[str] = frozenset()
    #: {"component_id", "status", "error"} for each step that did not succeed
    failures: tuple[dict, ...] = ()
    retry_hint: str | None = None
    previous_critique: str | None = None
    #: retrieved memory snippets a planner may condition on
    memory: tuple[str, ...] = ()

    def describe(self) -> str:
        """Human/model-readable feedback block; empty on the first attempt."""
        if self.attempt <= 1 and not self.failures and not self.memory:
            return ""
        lines = [f"ATTEMPT {self.attempt}."]
        if self.previous_critique:
            lines.append(f"Previous critique: {self.previous_critique}")
        if self.retry_hint:
            lines.append(f"Retry hint: {self.retry_hint}")
        for f in self.failures[:6]:
            lines.append(f"- {f.get('component_id')} -> {f.get('status')}: "
                         f"{str(f.get('error') or '')[:160]}")
        if self.excluded:
            lines.append("Do not use: " + ", ".join(sorted(self.excluded)[:12]))
        if self.memory:
            lines.append("Relevant memory:")
            lines += [f"  * {m[:200]}" for m in self.memory[:5]]
        return "\n".join(lines)


@dataclass
class Critique:
    """A planner's judgement of a run.

    `accepted` answers "should the runtime stop retrying?"; `verdict` answers
    "does the science hold?". They are not the same question, and conflating
    them is how "both tools returned 200" became a scientific result. A planner
    that only watched steps execute must leave `verdict` as INCONCLUSIVE; only a
    validator that compared against a metric, threshold, benchmark or ground
    truth may raise it to ACCEPTED.
    """

    accepted: bool
    reason: str
    retry_hint: str | None = None
    #: None means "this critique did not judge the science"; the report then
    #: falls back to deriving the verdict from `accepted`. The built-in planners
    #: always set it explicitly, so their runs never claim an unearned ACCEPTED.
    verdict: str | None = None


class PlannerPlugin(abc.ABC):
    """Turns a task into an ordered plan over the component registry."""

    #: registry key used by AgentSpec.planner
    name: str = "planner"

    @abc.abstractmethod
    def plan(self, task: str, registry: Any, *, max_steps: int = 5,
             context: PlanContext | None = None) -> Plan:
        """Turn a task into a plan. `context` carries feedback from prior attempts."""

    def critique(self, plan: Plan, results: Sequence[Any]) -> Critique:
        """Default: accept only if something executed and nothing failed."""
        if not results:
            return Critique(False, "no steps were executed", verdict="REJECTED")
        executed = [r for r in results if getattr(r, "executed", False)]
        failed = [r for r in results if not getattr(r, "status", None) or
                  not getattr(r.status, "successful", False)]
        if not executed:
            return Critique(False, "no step actually executed (all merely resolved)",
                            retry_hint="bind a dispatcher or choose executable components",
                            verdict="REJECTED")
        if failed:
            names = ", ".join(getattr(r, "capability", "?") for r in failed[:3])
            return Critique(False, f"{len(failed)}/{len(results)} step(s) did not succeed: {names}",
                            retry_hint="drop failing components or pick alternatives",
                            verdict="REJECTED")
        # Execution is complete and nothing judged the result. That is
        # INCONCLUSIVE, not ACCEPTED: a statistical test can run cleanly and
        # return p = 0.83, and a model can return an accuracy of 0.30. Both are
        # successful executions of a negative result.
        return Critique(True, f"all {len(results)} step(s) succeeded; no validator "
                              "judged the result, so the finding is not established",
                        verdict="INCONCLUSIVE")


def merge_step_arguments(step: Any, step_kwargs: dict | None) -> dict:
    """Combine run-wide kwargs with a plan step's own arguments.

    `PlanStep.arguments` existed on both planner interfaces and no execution path
    ever read it, so every REST path, query parameter and tool argument a planner
    produced was silently dropped and the backend was called with the run-wide
    defaults alone. A planner could "plan" a call to `/lookup/id/ENSG…` and the
    runtime would invoke the component with nothing.

    Step arguments win over the run-wide defaults: they are the specific decision
    the planner made for this step, and a global default must not overwrite it.
    """
    merged = dict(step_kwargs or {})
    merged.update(getattr(step, "arguments", None) or {})
    return merged


def plan_with_context(planner: PlannerPlugin, task: str, registry: Any, *,
                      max_steps: int, context: PlanContext | None) -> Plan:
    """Call `plan()` with the context if the planner accepts one.

    Third-party planners written against the older two-argument signature keep
    working; they simply do not see the feedback.
    """
    import inspect

    try:
        params = inspect.signature(planner.plan).parameters
    except (TypeError, ValueError):
        params = {}
    if "context" in params or any(p.kind is inspect.Parameter.VAR_KEYWORD
                                  for p in params.values()):
        return planner.plan(task, registry, max_steps=max_steps, context=context)
    return planner.plan(task, registry, max_steps=max_steps)


PLANNER_REGISTRY: dict[str, type[PlannerPlugin]] = {}


def register_planner(cls: type[PlannerPlugin]) -> type[PlannerPlugin]:
    PLANNER_REGISTRY[cls.name] = cls
    return cls


def get_planner(name: str, **kwargs: Any) -> PlannerPlugin:
    if name not in PLANNER_REGISTRY:
        raise KeyError(f"unknown planner {name!r}; registered: {sorted(PLANNER_REGISTRY)}")
    return PLANNER_REGISTRY[name](**kwargs)
