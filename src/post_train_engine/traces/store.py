"""Local JSONL persistence for typed trace records."""

from __future__ import annotations

from pathlib import Path

from post_train_engine.jsonl import append_jsonl, read_jsonl
from post_train_engine.traces.schema import TraceRecord


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


__all__ = ["JsonlTraceStore"]
