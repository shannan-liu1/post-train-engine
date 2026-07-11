"""JSONL checkpoint registry."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from post_train_engine.jsonl import append_jsonl, read_jsonl


@dataclass(frozen=True)
class CheckpointRecord:
    candidate_id: str
    path: str
    parent_id: str | None
    score: float
    metrics: dict[str, float]
    promoted: bool
    rejection_reason: str | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    remote_ref: str | None = None
    remote_artifacts: dict[str, str] = field(default_factory=dict)
    local_state: str = "available"

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id must be non-empty")
        if self.promoted and self.rejection_reason is not None:
            raise ValueError("promoted checkpoint cannot have a rejection reason")


class CheckpointRegistry:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, record: CheckpointRecord) -> None:
        append_jsonl(self.path, asdict(record))

    def records(self) -> tuple[CheckpointRecord, ...]:
        if not self.path.exists():
            return ()
        return tuple(CheckpointRecord(**row) for row in read_jsonl(self.path))

    def best_promoted(self) -> CheckpointRecord | None:
        promoted = [record for record in self.records() if record.promoted]
        if not promoted:
            return None
        return max(promoted, key=lambda record: record.score)
