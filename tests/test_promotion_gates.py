from __future__ import annotations

from dataclasses import replace

import pytest

from post_train_engine.evals.promotion import (
    CanaryDecision,
    EvalArtifact,
    EvalExampleResult,
    PromotionGateConfig,
    SliceGateConfig,
    decide_promotion,
    load_promotion_gate_config,
)


_EVAL_CONTRACT_HASH = "sha256:" + "1" * 64
_OTHER_EVAL_CONTRACT_HASH = "sha256:" + "2" * 64


def _artifact(
    artifact_id: str,
    correct: list[bool],
    *,
    parse_ok: list[bool] | None = None,
    tokens: int | None = 100,
) -> EvalArtifact:
    parse_ok = parse_ok or [True] * len(correct)
    examples = tuple(
        EvalExampleResult(
            example_id=f"e{idx:03d}",
            correct=value,
            parse_ok=parse_ok[idx],
            tokens=tokens,
            bucket="easy_stable" if idx < 10 else "frontier",
        )
        for idx, value in enumerate(correct)
    )
    accuracy = sum(correct) / len(correct)
    return EvalArtifact(
        artifact_id=artifact_id,
        primary_metric="greedy_exact_accuracy@1",
        evaluation_contract_hash=_EVAL_CONTRACT_HASH,
        examples=examples,
        metrics={"greedy_exact_accuracy@1": accuracy},
    )


def test_promotion_rejects_mismatched_evaluation_contracts() -> None:
    baseline = _artifact("old", [False, False])
    candidate = replace(
        _artifact("new", [True, False]),
        evaluation_contract_hash=_OTHER_EVAL_CONTRACT_HASH,
    )

    with pytest.raises(ValueError, match="evaluation contract"):
        decide_promotion(baseline, candidate, PromotionGateConfig())


def test_promotion_rejects_flat_two_percent_with_high_churn() -> None:
    old = [True] * 50 + [False] * 50
    new = [True] * 45 + [False] * 5 + [True] * 7 + [False] * 43

    decision = decide_promotion(
        _artifact("old", old),
        _artifact("new", new),
        PromotionGateConfig(min_primary_delta=0.02, min_primary_ci_low=-1.0),
    )

    assert decision.primary_delta == pytest.approx(0.02)
    assert decision.gates["mcnemar"] == "fail"
    assert decision.decision == "reject"


def test_promotion_accepts_two_percent_with_low_churn() -> None:
    old = [True] * 50 + [False] * 950
    new = [True] * 70 + [False] * 930

    decision = decide_promotion(
        _artifact("old", old),
        _artifact("new", new),
        PromotionGateConfig(
            min_primary_delta=0.02,
            min_primary_ci_low=0.0,
            max_mcnemar_p=0.05,
        ),
    )

    assert decision.primary_delta == pytest.approx(0.02)
    assert decision.gates["min_delta"] == "pass"
    assert decision.gates["mcnemar"] == "pass"
    assert decision.decision == "promote"


def test_promotion_rejects_when_token_counts_are_missing() -> None:
    old = [False] * 100
    new = [True] * 20 + [False] * 80

    decision = decide_promotion(
        _artifact("old", old, tokens=None),
        _artifact("new", new, tokens=None),
        PromotionGateConfig(min_primary_delta=0.02, min_primary_ci_low=0.0),
    )

    assert decision.gates["token_budget"] == "fail"
    assert decision.decision == "reject"


def test_promotion_rejects_when_easy_regression_evidence_is_missing() -> None:
    baseline = EvalArtifact(
        artifact_id="old",
        primary_metric="greedy_exact_accuracy@1",
        evaluation_contract_hash=_EVAL_CONTRACT_HASH,
        examples=tuple(
            EvalExampleResult(example_id=f"e{idx:03d}", correct=False, tokens=10)
            for idx in range(100)
        ),
    )
    candidate = EvalArtifact(
        artifact_id="new",
        primary_metric="greedy_exact_accuracy@1",
        evaluation_contract_hash=_EVAL_CONTRACT_HASH,
        examples=tuple(
            EvalExampleResult(
                example_id=f"e{idx:03d}",
                correct=idx == 0,
                tokens=10,
            )
            for idx in range(100)
        ),
    )

    decision = decide_promotion(
        baseline,
        candidate,
        PromotionGateConfig(
            min_primary_delta=0.0,
            min_primary_ci_low=-1.0,
            max_mcnemar_p=1.0,
            max_easy_regression=0.0,
        ),
    )

    assert decision.gates["easy_regression"] == "fail"
    assert decision.decision == "reject"
    assert decision.rejection_reasons == (
        "easy_regression missing easy_stable evidence",
    )


def test_eval_artifact_rejects_duplicate_example_ids() -> None:
    with pytest.raises(ValueError, match="duplicate eval example id: e001"):
        EvalArtifact.from_dict(
            {
                "artifact_id": "eval",
                "primary_metric": "greedy_exact_accuracy@1",
                "examples": [
                    {"example_id": "e001", "correct": True, "tokens": 10},
                    {"example_id": "e001", "correct": False, "tokens": 11},
                ],
            }
        )


def test_eval_artifact_rejects_string_booleans() -> None:
    with pytest.raises(ValueError, match="correct must be a boolean"):
        EvalArtifact.from_dict(
            {
                "artifact_id": "eval",
                "examples": [
                    {"example_id": "e001", "correct": "false", "tokens": 10},
                ],
            }
        )


def test_eval_artifact_rejects_boolean_metric_values() -> None:
    with pytest.raises(ValueError, match="metrics metric 'accuracy' must be finite"):
        EvalArtifact.from_dict(
            {
                "artifact_id": "eval",
                "primary_metric": "accuracy",
                "metrics": {"accuracy": True},
                "examples": [
                    {"example_id": "e001", "correct": True, "tokens": 10},
                ],
            }
        )


def test_promotion_gate_config_rejects_unused_large_gain_delta(tmp_path) -> None:
    config = tmp_path / "promotion.yaml"
    config.write_text("large_gain_delta: 0.05\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unknown promotion gate config field"):
        load_promotion_gate_config(config)


def test_promotion_gate_config_loads_required_slice_gates(tmp_path) -> None:
    config = tmp_path / "promotion.yaml"
    config.write_text(
        "\n".join(
            [
                "min_primary_delta: 0.0",
                "min_primary_ci_low: -1.0",
                "max_mcnemar_p: 1.0",
                "max_parse_regression: 0.005",
                "max_easy_regression: 0.01",
                "max_token_increase_ratio: 1.25",
                "required_slice_gates:",
                "  - slice_name: format",
                "    metric: greedy_exact_accuracy@1",
                "    max_regression: 0.0",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_promotion_gate_config(config)

    assert cfg.required_slice_gates == (
        SliceGateConfig(
            slice_name="format",
            metric="greedy_exact_accuracy@1",
            max_regression=0.0,
        ),
    )


def test_promotion_gate_config_rejects_boolean_numeric_fields() -> None:
    with pytest.raises(ValueError, match="min_primary_delta must be a finite number"):
        PromotionGateConfig(min_primary_delta=True)


def test_promotion_fails_closed_on_mismatched_primary_metric_contracts() -> None:
    baseline = _artifact("old", [False, False])
    candidate = EvalArtifact(
        artifact_id="new",
        primary_metric="sampled_pass@16",
        evaluation_contract_hash=_EVAL_CONTRACT_HASH,
        examples=tuple(
            EvalExampleResult(
                example_id=f"e{idx:03d}",
                correct=True,
                tokens=100,
                bucket="easy_stable",
            )
            for idx in range(2)
        ),
        metrics={"sampled_pass@16": 1.0},
    )

    with pytest.raises(ValueError, match="same primary_metric"):
        decide_promotion(baseline, candidate, PromotionGateConfig())


def test_promotion_fails_closed_on_unsupported_primary_metric_contracts() -> None:
    baseline = EvalArtifact(
        artifact_id="old",
        primary_metric="sampled_pass@16",
        evaluation_contract_hash=_EVAL_CONTRACT_HASH,
        examples=tuple(
            EvalExampleResult(
                example_id=f"e{idx:03d}",
                correct=False,
                tokens=100,
                bucket="easy_stable",
            )
            for idx in range(2)
        ),
        metrics={"sampled_pass@16": 0.0},
    )
    candidate = EvalArtifact(
        artifact_id="new",
        primary_metric="sampled_pass@16",
        evaluation_contract_hash=_EVAL_CONTRACT_HASH,
        examples=tuple(
            EvalExampleResult(
                example_id=f"e{idx:03d}",
                correct=True,
                tokens=100,
                bucket="easy_stable",
            )
            for idx in range(2)
        ),
        metrics={"sampled_pass@16": 1.0},
    )

    with pytest.raises(ValueError, match="example-level correctness"):
        decide_promotion(baseline, candidate, PromotionGateConfig())


def test_promotion_rejects_required_slice_regression_even_when_global_delta_passes() -> None:
    baseline_rows: list[EvalExampleResult] = []
    candidate_rows: list[EvalExampleResult] = []
    for idx in range(10):
        baseline_rows.append(
            EvalExampleResult(
                example_id=f"easy-{idx}",
                correct=True,
                tokens=10,
                bucket="easy_stable",
            )
        )
        candidate_rows.append(
            EvalExampleResult(
                example_id=f"easy-{idx}",
                correct=True,
                tokens=10,
                bucket="easy_stable",
            )
        )
    for idx in range(5):
        baseline_rows.append(
            EvalExampleResult(
                example_id=f"format-{idx}",
                correct=True,
                tokens=10,
                bucket="format",
            )
        )
        candidate_rows.append(
            EvalExampleResult(
                example_id=f"format-{idx}",
                correct=idx != 0,
                tokens=10,
                bucket="format",
            )
        )
    for idx in range(85):
        baseline_rows.append(
            EvalExampleResult(
                example_id=f"frontier-{idx}",
                correct=False,
                tokens=10,
                bucket="frontier",
            )
        )
        candidate_rows.append(
            EvalExampleResult(
                example_id=f"frontier-{idx}",
                correct=idx < 30,
                tokens=10,
                bucket="frontier",
            )
        )

    decision = decide_promotion(
        EvalArtifact(
            artifact_id="old",
            primary_metric="greedy_exact_accuracy@1",
            evaluation_contract_hash=_EVAL_CONTRACT_HASH,
            examples=tuple(baseline_rows),
        ),
        EvalArtifact(
            artifact_id="new",
            primary_metric="greedy_exact_accuracy@1",
            evaluation_contract_hash=_EVAL_CONTRACT_HASH,
            examples=tuple(candidate_rows),
        ),
        PromotionGateConfig(
            min_primary_delta=0.0,
            min_primary_ci_low=-1.0,
            max_mcnemar_p=1.0,
            required_slice_gates=(
                SliceGateConfig(
                    slice_name="format",
                    metric="greedy_exact_accuracy@1",
                    max_regression=0.0,
                ),
            ),
        ),
    )

    assert decision.primary_delta > 0
    assert decision.gates["slice:format:greedy_exact_accuracy@1"] == "fail"
    assert decision.decision == "reject"


def test_promotion_rejects_critical_or_high_severity_regressions() -> None:
    baseline = _artifact("old", [True] * 10 + [False] * 90)
    candidate = _artifact("new", [False] + [True] * 39 + [False] * 60)

    decision = decide_promotion(
        baseline,
        candidate,
        PromotionGateConfig(
            min_primary_delta=0.0,
            min_primary_ci_low=-1.0,
            max_mcnemar_p=1.0,
            max_easy_regression=1.0,
            max_high_severity_regressions=0,
        ),
    )

    assert decision.primary_delta > 0
    assert decision.gates["high_severity"] == "fail"
    assert decision.severity_summary["high"] == 1
    assert decision.decision == "reject"

    protected_candidate = EvalArtifact(
        artifact_id="new-protected",
        primary_metric=baseline.primary_metric,
        evaluation_contract_hash=_EVAL_CONTRACT_HASH,
        examples=(
            EvalExampleResult(
                example_id="e000",
                correct=False,
                tokens=100,
                bucket="easy_stable",
                protected=True,
            ),
            *candidate.examples[1:],
        ),
    )
    protected_decision = decide_promotion(
        baseline,
        protected_candidate,
        PromotionGateConfig(
            min_primary_delta=0.0,
            min_primary_ci_low=-1.0,
            max_mcnemar_p=1.0,
            max_easy_regression=1.0,
        ),
    )

    assert protected_decision.gates["critical_severity"] == "fail"
    assert protected_decision.severity_summary["critical"] == 1
    assert protected_decision.decision == "reject"


def test_promotion_rejects_failing_canary_decision() -> None:
    baseline = _artifact("old", [False] * 100)
    candidate = _artifact("new", [True] * 20 + [False] * 80)

    decision = decide_promotion(
        baseline,
        candidate,
        PromotionGateConfig(
            min_primary_delta=0.0,
            min_primary_ci_low=-1.0,
            max_mcnemar_p=1.0,
        ),
        canary_decision=CanaryDecision(
            decision="fail",
            failed_examples=("canary-format",),
            failure_types={"canary-format": "schema"},
        ),
    )

    assert decision.gates["canary"] == "fail"
    assert decision.canary_decision == {
        "decision": "fail",
        "failed_examples": ["canary-format"],
        "failure_types": {"canary-format": "schema"},
    }
    assert decision.decision == "reject"
