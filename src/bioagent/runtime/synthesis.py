"""Evidence synthesis — the final answer a run produces, with its provenance.

`Runtime.run()` returned a plan, a list of `CallResult`s, a critique and an
event log. Nothing turned five successful tool calls into an answer: no merging
of results, no conflict detection, no statement of what was found or how sure
the run is. This module produces that last layer.

With a model client the synthesis is written by the model from the structured
evidence; without one it is a deterministic aggregation. Either way the output
is a `Synthesis` whose every finding cites the step it came from, and whose
verdict is never ACCEPTED — synthesis integrates evidence, it does not validate
it. Only a validator comparing against a metric, threshold, benchmark or ground
truth may establish a finding.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Sequence


@dataclass
class Finding:
    """One thing the run established, and where it came from."""

    statement: str
    source_step: str                      # component id
    evidence: Any = None                  # the (digested) value it rests on
    confidence: str = "unassessed"        # unassessed | low | moderate | high


@dataclass
class Synthesis:
    answer: str
    findings: list[Finding] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    uncertainty: str = ""
    verdict: str = "INCONCLUSIVE"
    #: "model" or "deterministic"
    method: str = "deterministic"
    model: str = ""
    n_sources: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def digest(value: Any, limit: int = 400) -> Any:
    """A bounded, JSON-safe view of a step output for prompts and reports."""
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


class EvidenceSynthesizer:
    """Integrates step results into one answer with provenance."""

    def __init__(self, client: Callable[[str], str] | None = None, model: str = "") -> None:
        self._client = client
        self.model = model

    # ----------------------------------------------------------- deterministic
    @staticmethod
    def aggregate(task: str, results: Sequence[Any]) -> Synthesis:
        succeeded = [r for r in results if getattr(r, "status", None) is not None
                     and r.status.successful]
        failed = [r for r in results if r not in succeeded]
        findings = [Finding(statement=f"{r.capability} returned a result",
                            source_step=r.capability, evidence=digest(r.value))
                    for r in succeeded]
        conflicts: list[str] = []
        if not succeeded:
            answer = f"No step produced a result for: {task}"
        else:
            answer = (f"{len(succeeded)} of {len(results)} step(s) produced results for: "
                      f"{task}. Evidence is listed per step; no model was available to "
                      "integrate it, so no interpretation is offered.")
        uncertainty = ("deterministic aggregation only — findings are per-step outputs, "
                       "not an integrated interpretation")
        if failed:
            uncertainty += f"; {len(failed)} step(s) did not succeed: " + ", ".join(
                f"{r.capability} ({r.status.value})" for r in failed[:4])
        return Synthesis(answer=answer, findings=findings, conflicts=conflicts,
                         uncertainty=uncertainty, verdict="INCONCLUSIVE",
                         method="deterministic", n_sources=len(succeeded))

    # ------------------------------------------------------------------ model
    def synthesize(self, task: str, results: Sequence[Any]) -> Synthesis:
        base = self.aggregate(task, results)
        if self._client is None or not base.findings:
            return base
        evidence = [{"step": f.source_step, "output": f.evidence} for f in base.findings]
        prompt = (
            "You are integrating evidence gathered by tools for a biomedical task. "
            "Use ONLY the evidence below; do not add facts from memory.\n"
            f"TASK: {task}\n\nEVIDENCE (JSON):\n{json.dumps(evidence, indent=1)[:12000]}\n\n"
            "Reply with ONLY a JSON object: {\"answer\": str, "
            "\"findings\": [{\"statement\": str, \"source_step\": str, "
            "\"confidence\": \"low|moderate|high\"}], "
            "\"conflicts\": [str], \"uncertainty\": str}. Every finding must cite a "
            "source_step from the evidence."
        )
        try:
            raw = self._client(prompt)
            obj = _extract_object(raw)
            valid_steps = {f.source_step for f in base.findings}
            findings = []
            for item in obj.get("findings", []):
                if not isinstance(item, dict) or item.get("source_step") not in valid_steps:
                    continue          # an uncited or mis-cited finding is dropped
                findings.append(Finding(
                    statement=str(item.get("statement", ""))[:600],
                    source_step=str(item["source_step"]),
                    evidence=next((f.evidence for f in base.findings
                                   if f.source_step == item["source_step"]), None),
                    confidence=(str(item.get("confidence", "unassessed"))
                                if item.get("confidence") in ("low", "moderate", "high")
                                else "unassessed")))
            if not findings:
                raise ValueError("model produced no finding that cites a step")
            return Synthesis(
                answer=str(obj.get("answer", ""))[:4000], findings=findings,
                conflicts=[str(c)[:400] for c in obj.get("conflicts", []) if c][:10],
                uncertainty=str(obj.get("uncertainty", ""))[:1000],
                verdict="INCONCLUSIVE", method="model", model=self.model,
                n_sources=base.n_sources)
        except Exception as exc:  # noqa: BLE001 - synthesis must not crash the run
            base.uncertainty += f"; model synthesis failed ({type(exc).__name__}: {exc})"
            return base


def _extract_object(text: str) -> dict:
    t = text.strip()
    start, end = t.find("{"), t.rfind("}")
    if start >= 0 and end > start:
        obj = json.loads(t[start:end + 1])
        if isinstance(obj, dict):
            return obj
    raise ValueError("no JSON object found in model response")
