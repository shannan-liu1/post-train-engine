from __future__ import annotations

import pytest

from post_train_engine.evals.grades import Grade
from post_train_engine.evals.harness import EvalReport, EvalSample, MetricResult
from post_train_engine.evals.promotion import eval_report_to_artifact


def test_eval_report_converts_to_paired_promotion_artifact() -> None:
    report = EvalReport(
        spec_name="gsm8k-promotion",
        candidate_id="candidate-1",
        source="gsm8k",
        primary_metric="greedy_exact_accuracy@1",
        metrics={
            "greedy_exact_accuracy@1": MetricResult(
                name="greedy_exact_accuracy@1",
                value=0.5,
                n=2,
            )
        },
        slices=(),
        details=(
            EvalSample(
                example_id="easy",
                sample_index=0,
                prompt="easy prompt",
                expected_answer="4",
                generation="<answer>4</answer>",
                grade=Grade(
                    parsed_answer="4",
                    parse_success=True,
                    is_correct=True,
                    reason="exact",
                    score=1.0,
                    metadata={"completion_tokens": 12},
                ),
                slices={"bucket": "easy_stable"},
            ),
            EvalSample(
                example_id="frontier",
                sample_index=0,
                prompt="frontier prompt",
                expected_answer="7",
                generation="<answer>8</answer>",
                grade=Grade(
                    parsed_answer="8",
                    parse_success=True,
                    is_correct=False,
                    reason="wrong",
                    score=0.0,
                    metadata={"completion_tokens": 15},
                ),
                slices={"bucket": "frontier"},
            ),
        ),
        metadata={"model_id": "Qwen/Qwen2.5-0.5B-Instruct", "split_hash": "sha256:split"},
    )

    artifact = eval_report_to_artifact(report)

    assert artifact.artifact_id == "candidate-1"
    assert artifact.primary_metric == "greedy_exact_accuracy@1"
    assert artifact.metrics["greedy_exact_accuracy@1"] == 0.5
    assert [row.example_id for row in artifact.examples] == ["easy", "frontier"]
    assert artifact.examples[0].correct is True
    assert artifact.examples[0].parse_ok is True
    assert artifact.examples[0].tokens == 12
    assert artifact.examples[0].bucket == "easy_stable"
    assert artifact.metadata["model_id"] == "Qwen/Qwen2.5-0.5B-Instruct"


def test_eval_report_conversion_rejects_missing_tokens() -> None:
    report = EvalReport(
        spec_name="gsm8k-promotion",
        candidate_id="candidate-1",
        source="gsm8k",
        primary_metric="greedy_exact_accuracy@1",
        metrics={
            "greedy_exact_accuracy@1": MetricResult(
                name="greedy_exact_accuracy@1",
                value=1.0,
                n=1,
            )
        },
        slices=(),
        details=(
            EvalSample(
                example_id="missing-tokens",
                sample_index=0,
                prompt="prompt",
                expected_answer="4",
                generation="<answer>4</answer>",
                grade=Grade(
                    parsed_answer="4",
                    parse_success=True,
                    is_correct=True,
                    reason="exact",
                    score=1.0,
                ),
            ),
        ),
    )

    with pytest.raises(ValueError, match="completion token"):
        eval_report_to_artifact(report)
