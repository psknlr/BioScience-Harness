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

    def __len__(self) -> int:
        return len(self.steps)


@dataclass
class Critique:
    accepted: bool
    reason: str
    retry_hint: str | None = None


class PlannerPlugin(abc.ABC):
    """Turns a task into an ordered plan over the component registry."""

    #: registry key used by AgentSpec.planner
    name: str = "planner"

    @abc.abstractmethod
    def plan(self, task: str, registry: Any, *, max_steps: int = 5) -> Plan:
        ...

    def critique(self, plan: Plan, results: Sequence[Any]) -> Critique:
        """Default: accept only if something executed and nothing failed."""
        if not results:
            return Critique(False, "no steps were executed")
        executed = [r for r in results if getattr(r, "executed", False)]
        failed = [r for r in results if not getattr(r, "status", None) or
                  not getattr(r.status, "successful", False)]
        if not executed:
            return Critique(False, "no step actually executed (all merely resolved)",
                            retry_hint="bind a dispatcher or choose executable components")
        if failed:
            names = ", ".join(getattr(r, "capability", "?") for r in failed[:3])
            return Critique(False, f"{len(failed)}/{len(results)} step(s) did not succeed: {names}",
                            retry_hint="drop failing components or pick alternatives")
        return Critique(True, f"all {len(results)} step(s) succeeded")


PLANNER_REGISTRY: dict[str, type[PlannerPlugin]] = {}


def register_planner(cls: type[PlannerPlugin]) -> type[PlannerPlugin]:
    PLANNER_REGISTRY[cls.name] = cls
    return cls


def get_planner(name: str, **kwargs: Any) -> PlannerPlugin:
    if name not in PLANNER_REGISTRY:
        raise KeyError(f"unknown planner {name!r}; registered: {sorted(PLANNER_REGISTRY)}")
    return PLANNER_REGISTRY[name](**kwargs)
