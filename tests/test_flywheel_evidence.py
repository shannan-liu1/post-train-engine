from __future__ import annotations

import pytest

from post_train_engine.flywheel import (
    CostRecord,
    PriceSnapshot,
    ResourceTopology,
    ShardInfo,
    ShardPlan,
)


def test_cost_record_attributes_measured_accelerator_usage() -> None:
    price = PriceSnapshot(
        snapshot_id="runpod-a100-2026-06-15",
        provider="runpod",
        accelerator_type="A100-80GB",
        accelerator_hour_usd=1.89,
    )

    cost = CostRecord.from_usage(
        price,
        accelerator_count=4,
        wall_seconds=1800,
        input_tokens=120_000,
        output_tokens=40_000,
        checkpoint_bytes=1024,
    )

    assert cost.accelerator_seconds == pytest.approx(7200)
    assert cost.estimated_usd == pytest.approx(3.78)


def test_shard_plan_validates_complete_disjoint_eval_coverage() -> None:
    plan = ShardPlan(
        role="eval",
        world_size=2,
        shards=(
            ShardInfo(
                role="eval",
                shard_id=0,
                world_size=2,
                rank=0,
                example_ids=("a", "c"),
                seed=1000,
            ),
            ShardInfo(
                role="eval",
                shard_id=1,
                world_size=2,
                rank=1,
                example_ids=("b", "d"),
                seed=1001,
            ),
        ),
    )

    assert plan.covered_example_ids == ("a", "b", "c", "d")


def test_shard_plan_rejects_missing_or_duplicate_shards() -> None:
    with pytest.raises(ValueError, match="expected shard ids"):
        ShardPlan(
            role="rollout",
            world_size=2,
            shards=(
                ShardInfo(
                    role="rollout",
                    shard_id=0,
                    world_size=2,
                    rank=0,
                    example_ids=("a",),
                    seed=1000,
                ),
            ),
        )
    with pytest.raises(ValueError, match="duplicate example id"):
        ShardPlan(
            role="rollout",
            world_size=2,
            shards=(
                ShardInfo(
                    role="rollout",
                    shard_id=0,
                    world_size=2,
                    rank=0,
                    example_ids=("a", "b"),
                    seed=1000,
                ),
                ShardInfo(
                    role="rollout",
                    shard_id=1,
                    world_size=2,
                    rank=1,
                    example_ids=("b", "c"),
                    seed=1001,
                ),
            ),
        )


def test_cost_and_topology_models_reject_boolean_numeric_fields() -> None:
    with pytest.raises(ValueError, match="accelerator_hour_usd"):
        PriceSnapshot(
            snapshot_id="bad",
            provider="runpod",
            accelerator_type="A100",
            accelerator_hour_usd=True,
        )
    with pytest.raises(ValueError, match="estimated_usd"):
        CostRecord(
            price_snapshot_id="price",
            provider="runpod",
            accelerator_type="A100",
            accelerator_count=1,
            wall_seconds=10,
            accelerator_seconds=10,
            estimated_usd=True,
        )
    with pytest.raises(ValueError, match="num_nodes"):
        ResourceTopology(num_nodes=True)
