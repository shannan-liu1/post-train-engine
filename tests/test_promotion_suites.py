from __future__ import annotations

from post_train_engine.evals.suites import (
    PromotionSuiteState,
    SuiteRotationPolicy,
    record_accepted_promotion,
    record_candidate_evaluated,
    record_suite_test,
    rotation_recommendation,
)


def test_promotion_suite_accounting_uses_clear_counter_names() -> None:
    state = PromotionSuiteState(
        suite_id="toy-promotion",
        suite_version="2026-06-a",
        example_count=2,
        example_id_hash="sha256:ids",
        prompt_hash="sha256:prompts",
        slice_distribution={"easy_stable": 1, "frontier": 1},
    )

    state = record_suite_test(state)
    state = record_candidate_evaluated(state, "candidate-1")
    state = record_candidate_evaluated(state, "candidate-1")
    state = record_accepted_promotion(state)

    assert state.num_times_suite_tested == 1
    assert state.num_candidates_evaluated == 1
    assert state.accepted_promotion_count == 1
    assert state.train_and_promotion_overlap_count == 0


def test_suite_overlap_fails_closed_instead_of_rotating() -> None:
    state = PromotionSuiteState(
        suite_id="toy-promotion",
        suite_version="2026-06-a",
        example_count=2,
        example_id_hash="sha256:ids",
        train_and_promotion_overlap_count=1,
        train_and_promotion_overlap_ids=("leaked",),
    )

    recommendation = rotation_recommendation(state, SuiteRotationPolicy())

    assert recommendation.action == "fail_closed"
    assert recommendation.reason == "train_and_promotion_overlap_count > 0"


def test_suite_rotation_retires_overused_suite_without_hiding_old_evidence() -> None:
    state = PromotionSuiteState(
        suite_id="toy-promotion",
        suite_version="2026-06-a",
        example_count=2,
        example_id_hash="sha256:ids",
        num_times_suite_tested=51,
    )

    recommendation = rotation_recommendation(
        state,
        SuiteRotationPolicy(max_num_times_suite_tested=50),
    )

    assert recommendation.action == "rotate"
    assert recommendation.reason == "num_times_suite_tested exceeded 50"
