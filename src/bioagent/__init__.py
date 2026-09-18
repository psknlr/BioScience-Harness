"""bioagent — a federating biomedical agent over 16 open-source agent projects.

Contains no third-party source code: capability metadata is extracted by static
analysis, and capabilities are invoked through adapters that respect each
upstream project's license.
"""

from .agent import BioAgent, RunReport
from .core.executor import SandboxExecutor
from .core.planner import Plan, Planner, PlanStep, RetrievalPlanner
from .core.provenance import ProvenanceLog
from .core.executor import HardenedExecutor
from .policy import (Authorization, AuthorizationRequest, LicensePolicy,
                     PolicyDecision, PolicyKernel)
from .registry import Capability, CapabilityRegistry
from .status import ExecutionStatus, LifecycleState, RunOutcome, ScientificVerdict

__version__ = "0.2.4"

__all__ = [
    "BioAgent",
    "RunReport",
    "Capability",
    "CapabilityRegistry",
    "Plan",
    "PlanStep",
    "Planner",
    "RetrievalPlanner",
    "ProvenanceLog",
    "SandboxExecutor",
    "HardenedExecutor",
    "ExecutionStatus",
    "LifecycleState",
    "RunOutcome",
    "ScientificVerdict",
    "PolicyKernel",
    "LicensePolicy",
    "PolicyDecision",
    "Authorization",
    "AuthorizationRequest",
]
