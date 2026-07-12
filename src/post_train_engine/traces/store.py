"""Local JSONL persistence for typed trace records."""

from __future__ import annotations

from pathlib import Path

from post_train_engine.jsonl import append_jsonl, read_jsonl
from post_train_engine.traces.schema import TraceRecord


_TRAINING_ELIGIBLE_ROLES = frozenset({"train", "probe", "replay"})


class JsonlTraceStore:
    """Append-only local trace store backed by JSONL."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, trace: TraceRecord) -> None:
        append_jsonl(self.path, trace.model_dump(mode="json"))

    def read_all(self) -> list[TraceRecord]:
        if not self.path.exists():
            return []
        return [TraceRecord.model_validate(row) for row in read_jsonl(self.path)]

    def training_eligible(self, *, task_id: str | None = None) -> tuple[TraceRecord, ...]:
        """Query replayable evidence without admitting protected evaluation roles."""

        if task_id == "":
            raise ValueError("task_id must be non-empty when provided")
        return tuple(
            trace
            for trace in self.read_all()
            if trace.split_role in _TRAINING_ELIGIBLE_ROLES
            and (task_id is None or trace.task_id == task_id)
        )


__all__ = ["JsonlTraceStore"]
