from __future__ import annotations

from pathlib import Path

import pytest

from post_train_engine.runtime_evidence import (
    EvaluationCache,
    ExecutionTopology,
    PhaseCostRecord,
    PolicyUse,
    summarize_costs,
)


def test_cost_summary_reports_efficiency_units_and_missing_cost() -> None:
    summary = summarize_costs(
        (
            PhaseCostRecord(
                phase="train",
                duration_seconds=60.0,
                resource="gpu",
                resource_count=2,
                unit_price_usd=1.5,
            ),
            PhaseCostRecord(
                phase="provider_eval",
                duration_seconds=5.0,
                resource="remote_api",
                missing_reason="provider did not return billed cost",
            ),
        ),
        candidates=1,
        useful_traces=10,
        evaluations=20,
        promoted_metric_gain=0.1,
    )

    assert summary["certifying"] is False
    assert summary["measured_cost_usd"] == pytest.approx(0.05)
    assert summary["cost_per_useful_trace_usd"] == pytest.approx(0.005)
    assert summary["cost_per_promoted_metric_gain_usd"] == pytest.approx(0.5)


def test_evaluation_cache_requires_exact_contract_hash(tmp_path: Path) -> None:
    cache = EvaluationCache(tmp_path / "cache")
    contract = {
        "model": "sha256:model",
        "suite": "sha256:suite",
        "generation": "sha256:generation",
        "verifier": "sha256:verifier",
    }
    cache.put(contract, {"accuracy": 0.5})

    assert cache.get(contract) == {"accuracy": 0.5}
    assert cache.get({**contract, "verifier": "sha256:changed"}) is None


def test_topology_and_policy_staleness_fail_closed() -> None:
    with pytest.raises(ValueError, match="executed topology"):
        ExecutionTopology(
            modeled_world_size=4,
            executed_world_size=2,
            launcher="accelerate",
        )

    with pytest.raises(ValueError, match="policy staleness"):
        PolicyUse(
            generated_policy_step=10,
            consumed_policy_step=13,
            max_staleness_steps=2,
        )
