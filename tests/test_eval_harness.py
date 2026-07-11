from __future__ import annotations

import json
from pathlib import Path

import pytest

from post_train_engine.evals.grades import Grade
from post_train_engine.evals.harness import (
    MetricResult,
    EvalSpec,
    evaluate_model,
    write_eval_report,
)
from post_train_engine.evals.source import EvalSource
from post_train_engine.tasks.schema import Example


def _source() -> EvalSource:
    examples = [
        Example(
            id="easy",
            source="toy",
            prompt="easy",
            final_answer="yes",
            category="math",
            difficulty=1,
        ),
        Example(
            id="hard",
            source="toy",
            prompt="hard",
            final_answer="yes",
            category="logic",
            difficulty=5,
        ),
    ]
    return _source_with_examples(examples)


def _source_with_examples(examples: list[Example]) -> EvalSource:
    def score(parsed: str | None, _example: Example) -> Grade:
        is_correct = parsed == "yes"
        return Grade(
            parsed_answer=parsed,
            parse_success=parsed is not None,
            is_correct=is_correct,
            reason="exact",
            score=1.0 if is_correct else 0.0,
        )

    return EvalSource(
        name="toy",
        load_examples=lambda: examples,
        extract_answer=lambda generation: generation.strip() or None,
        score=score,
        default_max_new_tokens=4,
    )


def test_metric_result_rejects_boolean_values() -> None:
    with pytest.raises(ValueError, match="metric 'accuracy' must be finite"):
        MetricResult(name="accuracy", value=True, n=1)


def test_eval_harness_computes_sample_aware_metrics_and_slices() -> None:
    generations = {
        ("easy", 0): "yes",
        ("easy", 1): "wrong",
        ("hard", 0): "wrong",
        ("hard", 1): "yes",
    }
    spec = EvalSpec(
        name="toy-eval",
        primary_metric="pass@1",
        samples_per_example=2,
        metrics=(
            "pass@1",
            "pass@2",
            "exact_accuracy@1",
            "parse_success_rate@1",
            "mean_score@1",
        ),
        slice_fields=("category", "difficulty"),
    )

    report = evaluate_model(
        spec,
        _source(),
        lambda example, sample_index: generations[(example.id, sample_index)],
        candidate_id="candidate",
    )

    assert report.metrics["pass@1"].value == pytest.approx(0.5)
    assert report.metrics["pass@2"].value == pytest.approx(1.0)
    assert report.metrics["exact_accuracy@1"].value == pytest.approx(0.5)
    assert report.metrics["parse_success_rate@1"].value == pytest.approx(1.0)
    assert report.metrics["mean_score@1"].value == pytest.approx(0.5)
    assert {
        (slice_report.field, slice_report.value): slice_report.metrics[
            "exact_accuracy@1"
        ].value
        for slice_report in report.slices
    } == {
        ("category", "math"): 1.0,
        ("category", "logic"): 0.0,
        ("difficulty", "1"): 1.0,
        ("difficulty", "5"): 0.0,
    }


def test_pass_at_k_uses_all_samples_when_estimating_lower_k() -> None:
    examples = [
        Example(id="a", source="toy", prompt="a", final_answer="yes"),
        Example(id="b", source="toy", prompt="b", final_answer="yes"),
    ]
    source = _source_with_examples(examples)
    generations = {
        ("a", 0): "wrong",
        ("a", 1): "yes",
        ("a", 2): "yes",
        ("b", 0): "wrong",
        ("b", 1): "wrong",
        ("b", 2): "yes",
    }

    report = evaluate_model(
        EvalSpec(
            name="pass-estimator",
            primary_metric="pass@1",
            samples_per_example=3,
            metrics=("pass@1", "pass@2", "pass@3"),
        ),
        source,
        lambda example, sample_index: generations[(example.id, sample_index)],
        candidate_id="candidate",
    )

    assert report.metrics["pass@1"].value == pytest.approx((2 / 3 + 1 / 3) / 2)
    assert report.metrics["pass@2"].value == pytest.approx((1.0 + 2 / 3) / 2)
    assert report.metrics["pass@3"].value == pytest.approx(1.0)


def test_eval_harness_computes_macro_f1_from_first_sample_labels() -> None:
    spec = EvalSpec(
        name="toy-f1",
        primary_metric="macro_f1@1",
        samples_per_example=1,
        metrics=("macro_f1@1",),
    )

    report = evaluate_model(
        spec,
        _source(),
        lambda example, _sample_index: "yes" if example.id == "easy" else "no",
        candidate_id="candidate",
    )

    assert report.metrics["macro_f1@1"].value == pytest.approx(1 / 3)


def test_eval_report_writes_manifest_summary_and_details(tmp_path: Path) -> None:
    report = evaluate_model(
        EvalSpec(
            name="toy-eval",
            primary_metric="pass@1",
            samples_per_example=1,
            metrics=("pass@1",),
            slice_fields=("category",),
        ),
        _source(),
        lambda _example, _sample_index: "yes",
        candidate_id="candidate",
    )

    write_eval_report(report, tmp_path)

    manifest = json.loads((tmp_path / "eval_report.json").read_text(encoding="utf-8"))
    details = json.loads((tmp_path / "details.json").read_text(encoding="utf-8"))
    assert manifest["candidate_id"] == "candidate"
    assert manifest["metrics"]["pass@1"]["value"] == 1.0
    assert len(details) == 2
