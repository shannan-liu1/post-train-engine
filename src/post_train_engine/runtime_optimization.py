"""Output-equivalent batching, model reuse, and asynchronous evidence queues."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from post_train_engine.runtime_evidence import PolicyUse

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


@dataclass(frozen=True)
class RuntimeMeasurement:
    example_count: int
    model_load_count: int
    batch_count: int


class ReusableBatchEvaluator(Generic[InputT, OutputT]):
    """Load one model once and evaluate deterministic batches in input order."""

    def __init__(
        self,
        *,
        load_model: Callable[[str], Any],
        evaluate_batch: Callable[[Any, Sequence[InputT]], Sequence[OutputT]],
        batch_size: int,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._load_model = load_model
        self._evaluate_batch = evaluate_batch
        self._batch_size = batch_size

    def evaluate(
        self,
        model_ref: str,
        rows: Sequence[InputT],
    ) -> tuple[list[OutputT], RuntimeMeasurement]:
        model = self._load_model(model_ref)
        outputs: list[OutputT] = []
        batch_count = 0
        for start in range(0, len(rows), self._batch_size):
            batch = rows[start : start + self._batch_size]
            result = list(self._evaluate_batch(model, batch))
            if len(result) != len(batch):
                raise ValueError("batch evaluator output count does not match input count")
            outputs.extend(result)
            batch_count += 1
        return outputs, RuntimeMeasurement(
            example_count=len(rows),
            model_load_count=1,
            batch_count=batch_count,
        )


class AsyncEvidenceQueue:
    """Minimal FIFO executor that validates policy lag before releasing evidence."""

    def __init__(self, *, max_staleness_steps: int) -> None:
        if max_staleness_steps < 0:
            raise ValueError("max_staleness_steps must be non-negative")
        self._max_staleness_steps = max_staleness_steps
        self._pending: deque[tuple[str, int]] = deque()

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def submit(self, trace_id: str, *, generated_policy_step: int) -> None:
        if not trace_id or generated_policy_step < 0:
            raise ValueError("trace ID and generated policy step must be valid")
        self._pending.append((trace_id, generated_policy_step))

    def consume(self, *, current_policy_step: int) -> str:
        if not self._pending:
            raise ValueError("asynchronous evidence queue is empty")
        trace_id, generated_step = self._pending[0]
        PolicyUse(
            generated_policy_step=generated_step,
            consumed_policy_step=current_policy_step,
            max_staleness_steps=self._max_staleness_steps,
        )
        self._pending.popleft()
        return trace_id


__all__ = [
    "AsyncEvidenceQueue",
    "ReusableBatchEvaluator",
    "RuntimeMeasurement",
]
