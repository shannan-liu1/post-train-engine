from __future__ import annotations

import pytest

from post_train_engine.runtime_optimization import (
    AsyncEvidenceQueue,
    ReusableBatchEvaluator,
)


def test_batching_and_model_reuse_preserve_outputs_with_less_work() -> None:
    load_calls = {"count": 0}
    batch_calls = {"count": 0}

    def load_model(model_ref: str):
        load_calls["count"] += 1
        return {"model_ref": model_ref}

    def evaluate_batch(model, rows):
        batch_calls["count"] += 1
        return [f"{model['model_ref']}:{row * 2}" for row in rows]

    evaluator = ReusableBatchEvaluator(
        load_model=load_model,
        evaluate_batch=evaluate_batch,
        batch_size=2,
    )

    outputs, measurement = evaluator.evaluate("model-1", [1, 2, 3, 4])

    assert outputs == ["model-1:2", "model-1:4", "model-1:6", "model-1:8"]
    assert measurement.model_load_count == 1
    assert measurement.batch_count == 2
    assert load_calls["count"] == 1
    assert batch_calls["count"] == 2


def test_async_queue_enforces_policy_staleness_before_consumption() -> None:
    queue = AsyncEvidenceQueue(max_staleness_steps=2)
    queue.submit("trace-1", generated_policy_step=10)

    with pytest.raises(ValueError, match="policy staleness"):
        queue.consume(current_policy_step=13)

    assert queue.pending_count == 1
    assert queue.consume(current_policy_step=12) == "trace-1"
