from __future__ import annotations

import numpy as np
import pytest

from post_train_engine.evals.statistics import (
    hierarchical_bootstrap_rollout_metric_delta,
    mcnemar_exact_p,
    paired_accuracy_stats,
    paired_bootstrap_ci,
)


def test_paired_accuracy_stats_counts_new_only_old_only() -> None:
    stats = paired_accuracy_stats(
        old=[True, True, False, False],
        new=[True, False, True, False],
        bootstrap_samples=200,
    )

    assert stats.old_correct == 2
    assert stats.new_correct == 2
    assert stats.new_only == 1
    assert stats.old_only == 1
    assert stats.delta == 0.0


def test_paired_bootstrap_ci_reproducible() -> None:
    old = np.array([True, False, False, True])
    new = np.array([True, True, False, False])

    first = paired_bootstrap_ci(
        old,
        new,
        metric=lambda diffs: float(np.mean(diffs)),
        samples=200,
        seed=7,
    )
    second = paired_bootstrap_ci(
        old,
        new,
        metric=lambda diffs: float(np.mean(diffs)),
        samples=200,
        seed=7,
    )

    assert first == second


def test_mcnemar_exact_rejects_60_40_with_net_20_if_gate_requires_p05() -> None:
    assert mcnemar_exact_p(new_only=60, old_only=40) > 0.05


def test_hierarchical_bootstrap_rollout_metric_delta_pass_at_k() -> None:
    old = np.array([[False, False], [True, False]])
    new = np.array([[True, False], [True, True]])

    result = hierarchical_bootstrap_rollout_metric_delta(
        old,
        new,
        metric="pass_at_k",
        k=2,
        samples=200,
        seed=3,
    )

    assert result.delta == pytest.approx(0.5)
    assert result.samples == 200
