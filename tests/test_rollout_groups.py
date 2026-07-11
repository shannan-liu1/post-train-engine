from __future__ import annotations

import pytest

from post_train_engine.traces import (
    RolloutGroup,
    TraceRecord,
    build_rollout_group,
    stable_prompt_hash,
)


def test_build_rollout_group_derives_ids_size_and_reward_shape() -> None:
    traces = (_trace("trace-001"), _trace("trace-002"))

    group = build_rollout_group(
        group_id="group-001",
        traces=traces,
        rewards=(1.0, 0.0),
    )

    assert group.group_id == "group-001"
    assert group.trace_ids == ("trace-001", "trace-002")
    assert group.group_size == 2
    assert group.rewards == (1.0, 0.0)
    assert group.reward_variance == pytest.approx(0.25)
    assert group.degenerate_group is False


def test_rollout_group_rejects_malformed_group_contracts() -> None:
    with pytest.raises(ValueError, match="trace_ids"):
        RolloutGroup(
            group_id="group",
            trace_ids=("trace-001", "trace-001"),
            group_size=2,
        )
    with pytest.raises(ValueError, match="group_size"):
        RolloutGroup(group_id="group", trace_ids=("trace-001",), group_size=2)
    with pytest.raises(ValueError, match="rewards"):
        RolloutGroup(
            group_id="group",
            trace_ids=("trace-001",),
            group_size=1,
            rewards=(True,),
        )
    with pytest.raises(ValueError, match="reward_variance"):
        RolloutGroup(
            group_id="group",
            trace_ids=("trace-001",),
            group_size=1,
            reward_variance=float("inf"),
        )
    with pytest.raises(ValueError, match="degenerate_group"):
        RolloutGroup(
            group_id="group",
            trace_ids=("trace-001", "trace-002"),
            group_size=2,
            rewards=(1.0, 1.0),
            reward_variance=0.0,
            degenerate_group=False,
        )


def _trace(trace_id: str) -> TraceRecord:
    return TraceRecord(
        trace_id=trace_id,
        run_id="run-001",
        task_id="gsm8k",
        example_id="example-001",
        split_role="train",
        prompt_hash=stable_prompt_hash("What is 2 + 2?"),
        source_checkpoint="checkpoints/seed",
        policy_version="seed-v1",
        policy_step=0,
        policy_step_evidence="static",
        rollout_group_id="group-001",
        generation_backend="local",
        sampling_config={"temperature": 0.0},
        verifier_id="gsm8k_numeric_v1",
    )
