from __future__ import annotations

from pathlib import Path

from post_train_engine.engine import RunEngine, RunPlan
from post_train_engine.evals.contract import EvalContract
from post_train_engine.evidence_safety import (
    VerifierSeparation,
    certify_content_separation,
)
from post_train_engine.task_adapters import ExactMathRunAdapter


def test_second_executable_verifier_task_runs_through_engine(tmp_path: Path) -> None:
    plan = RunPlan(
        run_id="exact-math-1",
        candidate_id="exact-math-candidate-1",
        parent_candidate_id="seed",
        task_name="exact_math_tool",
        model_id="deterministic-calculator-policy",
        output_dir=str(tmp_path / "run"),
        training_example_ids=tuple(f"math-{index}" for index in range(4)),
        promotion_example_ids=tuple(f"promotion-{index}" for index in range(4)),
        evaluation_contract=EvalContract.from_components(
            suite_id="exact-math-promotion",
            suite_version="v1",
            example_ids=tuple(f"promotion-{index}" for index in range(4)),
            example_content=("9 + 8", "7 * 6", "20 - 3", "18 / 3"),
            prompt_contract={"task": "exact_math_tool"},
            verifier_contract={"verifier": "exact-integer-v1"},
            generation_contract={"backend": "deterministic"},
            primary_metric="accuracy",
        ),
        content_separation=certify_content_separation(
            training_texts=("2 + 2", "3 * 5", "12 - 7", "8 / 2"),
            protected_texts=("9 + 8", "7 * 6", "20 - 3", "18 / 3"),
            ngram_size=2,
        ),
        verifier_separation=VerifierSeparation(
            verifier_kind="executable_ground_truth",
            training_verifier_id="exact-integer-v1",
            promotion_verifier_id="exact-integer-v1",
        ),
        promotion_gate={
            "min_primary_delta": 0.1,
            "min_primary_ci_low": -1.0,
            "max_mcnemar_p": 1.0,
            "max_parse_regression": 0.0,
            "max_easy_regression": 0.0,
            "max_token_increase_ratio": 1.0,
        },
        inputs={
            "model": {
                "kind": "model",
                "requested_id": "deterministic-calculator-policy",
                "resolved_id": "deterministic-calculator-policy",
                "resolved_revision": "v1",
                "resolution_state": "exact",
            },
            "dataset": {
                "kind": "dataset",
                "requested_id": "exact-math-fixture",
                "resolved_id": "exact-math-fixture",
                "resolved_revision": "v1",
                "resolution_state": "exact",
            },
        },
    )

    execution = RunEngine().execute(plan, ExactMathRunAdapter())

    assert execution.manifest.task_name == "exact_math_tool"
    assert execution.manifest.status == "promoted"
    assert execution.manifest.metadata["cost_usd"] == 0.0
