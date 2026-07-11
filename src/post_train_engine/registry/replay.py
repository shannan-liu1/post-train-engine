"""JSONL replay buffer for useful traces."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from post_train_engine.jsonl import append_jsonl, read_jsonl


@dataclass(frozen=True)
class ReplayTrace:
    trace_id: str
    task: str
    example_id: str
    prompt: str
    generation: str
    grade: dict[str, Any]
    source_candidate_id: str
    difficulty: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.trace_id:
            raise ValueError("trace_id must be non-empty")
        if not self.task:
            raise ValueError("task must be non-empty")


class ReplayBuffer:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, trace: ReplayTrace) -> None:
        append_jsonl(self.path, asdict(trace))

    def traces(self) -> tuple[ReplayTrace, ...]:
        if not self.path.exists():
            return ()
        return tuple(ReplayTrace(**row) for row in read_jsonl(self.path))

    def by_task(self, task: str) -> tuple[ReplayTrace, ...]:
        return tuple(trace for trace in self.traces() if trace.task == task)
