"""Agent memory — working memory for a run or a team, episodic memory across runs.

The workspace had `memory/episodic`, `memory/semantic`, `memory/project` and
`memory/user` directories and nothing that wrote to or read from them: memory
was a storage namespace, not a subsystem. This module gives it an API and the
runtime calls it — runs are remembered, planners and agents recall.

Two layers:

* `WorkingMemory` — the shared scratchpad of one run or one team of agents.
  Step outputs, notes and handoff context live here for the duration of the
  task, and every agent in an orchestration reads the same one.
* `EpisodicMemory` — persistent, append-only, searchable. Backed by a JSON Lines
  file, written through the `Workspace` trust boundary when one is supplied so
  agent-authored memory can never land outside the mutable plane.

Retrieval is term-overlap ranking with a recency tiebreak — the same mechanism
the component registry uses for search — not an embedding index. It is honest
about that: `search()` returns what lexically overlaps, and a caller wanting
semantic recall should supply a `ranker`.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable


def _terms(text: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", str(text).lower()) if len(t) > 2}


@dataclass
class MemoryEntry:
    entry_id: str
    kind: str                       # episode | note | fact | handoff
    text: str
    tags: tuple[str, ...] = ()
    created_at: float = field(default_factory=time.time)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["tags"] = list(self.tags)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryEntry":
        return cls(entry_id=str(d.get("entry_id", "")), kind=str(d.get("kind", "note")),
                   text=str(d.get("text", "")), tags=tuple(d.get("tags") or ()),
                   created_at=float(d.get("created_at", 0.0)), meta=dict(d.get("meta") or {}))


class WorkingMemory:
    """Shared scratchpad for one run or one team. In-process, not persisted."""

    def __init__(self) -> None:
        self._slots: dict[str, Any] = {}
        self._notes: list[dict[str, Any]] = []
        self._history: list[dict[str, Any]] = []

    def put(self, key: str, value: Any, *, by: str = "") -> None:
        self._slots[key] = value
        self._history.append({"op": "put", "key": key, "by": by, "at": time.time()})

    def get(self, key: str, default: Any = None) -> Any:
        return self._slots.get(key, default)

    def note(self, text: str, *, by: str = "") -> None:
        self._notes.append({"text": str(text), "by": by, "at": time.time()})

    def keys(self) -> list[str]:
        return list(self._slots)

    def notes(self) -> list[dict[str, Any]]:
        return list(self._notes)

    def snapshot(self) -> dict[str, Any]:
        return {"slots": dict(self._slots), "notes": list(self._notes),
                "n_ops": len(self._history)}

    def describe(self, limit: int = 1500) -> str:
        """A bounded text view for prompts."""
        parts: list[str] = []
        for k, v in self._slots.items():
            try:
                blob = json.dumps(v, default=str)
            except (TypeError, ValueError):
                blob = repr(v)
            parts.append(f"{k}: {blob[:300]}")
        parts += [f"note ({n['by'] or 'anon'}): {n['text'][:300]}" for n in self._notes[-8:]]
        text = "\n".join(parts)
        return text[:limit]


class EpisodicMemory:
    """Persistent, searchable memory of past runs, notes and facts."""

    REL_PATH = "memory/episodic/entries.jsonl"

    def __init__(self, workspace: Any = None, path: str | Path | None = None,
                 ranker: Callable[[str, MemoryEntry], float] | None = None) -> None:
        if workspace is None and path is None:
            raise ValueError("EpisodicMemory needs a workspace or a path")
        self._ws = workspace
        self._path = Path(path) if path else None
        self._ranker = ranker
        self._entries: list[MemoryEntry] | None = None

    # ---------------------------------------------------------------- storage
    def _read_raw(self) -> str:
        if self._ws is not None:
            return self._ws.read(self.REL_PATH) if self._ws.exists(self.REL_PATH) else ""
        return self._path.read_text(encoding="utf-8") if self._path.exists() else ""

    def _write_raw(self, text: str) -> None:
        if self._ws is not None:
            # Through the trust boundary: memory/ is mutable, so agent writes are
            # permitted here and refused anywhere the boundary protects.
            self._ws.write(self.REL_PATH, text, agent=True)
        else:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(text, encoding="utf-8")

    def _load(self) -> list[MemoryEntry]:
        if self._entries is None:
            out: list[MemoryEntry] = []
            for line in self._read_raw().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(MemoryEntry.from_dict(json.loads(line)))
                except (ValueError, TypeError):
                    continue
            self._entries = out
        return self._entries

    def __len__(self) -> int:
        return len(self._load())

    # ------------------------------------------------------------------ write
    def write(self, text: str, *, kind: str = "note", tags: Iterable[str] = (),
              meta: dict | None = None) -> MemoryEntry:
        entry = MemoryEntry(entry_id=uuid.uuid4().hex[:12], kind=kind, text=str(text),
                            tags=tuple(str(t) for t in tags), meta=dict(meta or {}))
        entries = self._load()
        entries.append(entry)
        self._write_raw("".join(json.dumps(e.to_dict(), default=str) + "\n" for e in entries))
        return entry

    def remember_run(self, task: str, report: Any) -> MemoryEntry:
        """Record a run as an episode: what was asked, what ran, what came of it."""
        steps = [r.capability for r in getattr(report, "results", [])]
        answer = ""
        synth = getattr(report, "synthesis", None)
        if synth is not None:
            answer = str(getattr(synth, "answer", ""))[:400]
        text = (f"task: {task} | execution: {getattr(report, 'execution_outcome', '?')} | "
                f"verdict: {getattr(report, 'verdict', '?')} | steps: {', '.join(steps[:8])}"
                + (f" | answer: {answer}" if answer else ""))
        return self.write(text, kind="episode", tags=("run",),
                          meta={"steps": steps, "attempts": getattr(report, "attempts", 1)})

    # ------------------------------------------------------------------- read
    def recent(self, k: int = 5) -> list[MemoryEntry]:
        return sorted(self._load(), key=lambda e: -e.created_at)[:k]

    def search(self, query: str, k: int = 5, *, kinds: Iterable[str] = ()) -> list[MemoryEntry]:
        entries = self._load()
        if kinds:
            wanted = set(kinds)
            entries = [e for e in entries if e.kind in wanted]
        if self._ranker is not None:
            scored = [(self._ranker(query, e), e) for e in entries]
        else:
            q = _terms(query)
            scored = [(len(q & (_terms(e.text) | set(t.lower() for t in e.tags))), e)
                      for e in entries]
        scored = [(sc, e) for sc, e in scored if sc > 0]
        scored.sort(key=lambda x: (-x[0], -x[1].created_at))
        return [e for _, e in scored[:k]]

    def summarize(self, k: int = 10) -> str:
        return "\n".join(f"[{e.kind}] {e.text[:200]}" for e in self.recent(k))
