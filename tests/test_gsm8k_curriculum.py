from __future__ import annotations

from post_train_engine.data_builders.gsm8k_curriculum import (
    build_gsm8k_curriculum,
    frontier_weight,
)
from post_train_engine.difficulty import (
    DifficultyBucket,
    DifficultyBucketConfig,
    DifficultyBucketRecord,
    bucket_probe_rollouts,
)
from post_train_engine.probe import (
    EarlyExitConfig,
    full_filter_reason,
    should_continue_after_early,
    should_train_after_full,
)
from post_train_engine.tasks.gsm8k import GSM8KExample


def _rollouts(correct: int, parse_ok: int = 16) -> list[dict[str, object]]:
    rows = []
    for idx in range(16):
        rows.append(
            {
                "run_id": "probe",
                "example_id": "gsm8k/train/000001",
                "rollout_id": idx,
                "trace_id": f"t{idx}",
                "correct": idx < correct,
                "parse_ok": idx < parse_ok,
                "reward": 1.0 if idx < correct else 0.0,
                "completion_tokens": 20 + idx,
                "gold_answer": "18",
                "completion": "<answer>18</answer>" if idx < correct else "<answer>19</answer>",
                "parser": "answer_tag",
            }
        )
    return rows


def _bucket(bucket: DifficultyBucket, pass_rate: float) -> DifficultyBucketRecord:
    return DifficultyBucketRecord(
        example_id="gsm8k/train/000001",
        num_rollouts=16,
        num_correct=int(pass_rate * 16),
        num_parse_ok=16,
        pass_rate=pass_rate,
        parse_rate=1.0,
        mean_reward=pass_rate,
        mean_tokens=30,
        bucket=bucket,
        bucket_reason="test",
        successful_trace_ids=("t0",),
        failed_trace_ids=("t1",),
    )


def _example() -> GSM8KExample:
    return GSM8KExample(
        id="gsm8k/train/000001",
        split="train",
        question="What is 9+9?",
        gold_solution="9+9=18 #### 18",
        gold_answer="18",
        source="openai/gsm8k",
        metadata={},
    )


def test_early_exit_accepts_one_two_three_of_four() -> None:
    cfg = EarlyExitConfig()

    assert [should_continue_after_early(i, cfg) for i in range(1, 4)] == [True] * 3


def test_early_exit_rejects_zero_and_four_of_four() -> None:
    cfg = EarlyExitConfig()

    assert should_continue_after_early(0, cfg) is False
    assert should_continue_after_early(4, cfg) is False


def test_early_exit_config_rejects_impossible_success_thresholds() -> None:
    try:
        EarlyExitConfig(g_total=16, g_early=4, early_min_successes=5)
    except ValueError as exc:
        assert "early_min_successes must be between 0 and g_early" in str(exc)
    else:
        raise AssertionError("expected impossible early threshold to be rejected")


def test_full_filter_accepts_two_through_twelve_of_sixteen() -> None:
    cfg = EarlyExitConfig()

    assert all(should_train_after_full(i, cfg) for i in range(2, 13))
    assert full_filter_reason(1, cfg) == "full_too_hard"
    assert full_filter_reason(13, cfg) == "full_too_easy"


def test_bucket_parser_issue() -> None:
    record = bucket_probe_rollouts("gsm8k/train/000001", _rollouts(correct=0, parse_ok=12))

    assert record.bucket is DifficultyBucket.PARSER_ISSUE


def test_bucket_easy_stable() -> None:
    record = bucket_probe_rollouts("gsm8k/train/000001", _rollouts(correct=14))

    assert record.bucket is DifficultyBucket.EASY_STABLE


def test_bucket_frontier() -> None:
    record = bucket_probe_rollouts("gsm8k/train/000001", _rollouts(correct=5))

    assert record.bucket is DifficultyBucket.FRONTIER


def test_bucket_config_rejects_inverted_frontier_thresholds() -> None:
    try:
        DifficultyBucketConfig(frontier_min_pass_rate=0.8, frontier_max_pass_rate=0.2)
    except ValueError as exc:
        assert "frontier_min_pass_rate cannot exceed frontier_max_pass_rate" in str(exc)
    else:
        raise AssertionError("expected inverted frontier thresholds to be rejected")


def test_bucket_hard_solved() -> None:
    record = bucket_probe_rollouts("gsm8k/train/000001", _rollouts(correct=1))

    assert record.bucket is DifficultyBucket.HARD_SOLVED


def test_bucket_unsolved_parseable() -> None:
    record = bucket_probe_rollouts("gsm8k/train/000001", _rollouts(correct=0))

    assert record.bucket is DifficultyBucket.UNSOLVED_PARSEABLE


def test_probe_artifact_schema() -> None:
    record = bucket_probe_rollouts(
        "gsm8k/train/000001",
        _rollouts(correct=5),
        DifficultyBucketConfig(),
    )
    body = record.to_json()

    assert body["example_id"] == "gsm8k/train/000001"
    assert body["pass_rate"] == 5 / 16
    assert body["successful_trace_ids"] == ["t0", "t1", "t2", "t3", "t4"]


def test_build_curriculum_emits_frontier_and_opsd_sets() -> None:
    examples = {_example().id: _example()}
    frontier = _bucket(DifficultyBucket.FRONTIER, pass_rate=5 / 16)
    hard = _bucket(DifficultyBucket.HARD_SOLVED, pass_rate=1 / 16)
    rows = _rollouts(correct=1)

    curriculum = build_gsm8k_curriculum(
        examples,
        (frontier, hard),
        rows,
        source_probe_run_id="probe",
    )

    assert curriculum.grpo_frontier[0]["weight"] == frontier_weight(5 / 16)
    assert curriculum.opsd_hard[0]["source_probe_run_id"] == "probe"
    assert curriculum.opsd_hard[0]["teacher_privileged_info"]["reference_source"] == (
        "verified_student_trace"
    )
