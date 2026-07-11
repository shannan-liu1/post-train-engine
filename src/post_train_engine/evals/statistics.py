"""Statistical comparisons for paired eval artifacts."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class PairedAccuracyStats:
    n: int
    old_correct: int
    new_correct: int
    both_correct: int
    both_wrong: int
    new_only: int
    old_only: int
    delta: float
    se: float
    normal_ci_low: float
    normal_ci_high: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float
    mcnemar_p: float


@dataclass(frozen=True)
class BootstrapResult:
    delta: float
    ci_low: float
    ci_high: float
    samples: int


def paired_accuracy_stats(
    old: Sequence[bool],
    new: Sequence[bool],
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 1337,
) -> PairedAccuracyStats:
    if len(old) != len(new):
        raise ValueError("old and new must have the same length")
    if not old:
        raise ValueError("paired accuracy stats require at least one example")
    old_arr = np.asarray(old, dtype=bool)
    new_arr = np.asarray(new, dtype=bool)
    diffs = new_arr.astype(float) - old_arr.astype(float)
    delta = float(diffs.mean())
    se = _standard_error(diffs)
    ci_low = delta - 1.96 * se
    ci_high = delta + 1.96 * se
    boot_low, boot_high = paired_bootstrap_ci(
        old_arr,
        new_arr,
        metric=lambda sampled_diffs: float(np.mean(sampled_diffs)),
        samples=bootstrap_samples,
        seed=seed,
    )
    new_only = int(np.logical_and(new_arr, ~old_arr).sum())
    old_only = int(np.logical_and(old_arr, ~new_arr).sum())
    return PairedAccuracyStats(
        n=len(old_arr),
        old_correct=int(old_arr.sum()),
        new_correct=int(new_arr.sum()),
        both_correct=int(np.logical_and(old_arr, new_arr).sum()),
        both_wrong=int(np.logical_and(~old_arr, ~new_arr).sum()),
        new_only=new_only,
        old_only=old_only,
        delta=delta,
        se=se,
        normal_ci_low=ci_low,
        normal_ci_high=ci_high,
        bootstrap_ci_low=boot_low,
        bootstrap_ci_high=boot_high,
        mcnemar_p=mcnemar_exact_p(new_only, old_only),
    )


def paired_bootstrap_ci(
    old: np.ndarray,
    new: np.ndarray,
    *,
    metric: Callable[[np.ndarray], float],
    samples: int = 10_000,
    seed: int = 1337,
    alpha: float = 0.05,
) -> tuple[float, float]:
    old_arr = np.asarray(old)
    new_arr = np.asarray(new)
    if old_arr.shape != new_arr.shape:
        raise ValueError("old and new arrays must have matching shapes")
    if old_arr.ndim != 1:
        raise ValueError("paired_bootstrap_ci expects one-dimensional arrays")
    if old_arr.size == 0:
        raise ValueError("bootstrap requires at least one observation")
    if samples <= 0:
        raise ValueError("samples must be positive")
    diffs = new_arr.astype(float) - old_arr.astype(float)
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=float)
    for idx in range(samples):
        draw = rng.integers(0, diffs.size, size=diffs.size)
        estimates[idx] = metric(diffs[draw])
    low, high = np.quantile(estimates, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def mcnemar_exact_p(new_only: int, old_only: int) -> float:
    if new_only < 0 or old_only < 0:
        raise ValueError("discordant counts must be non-negative")
    discordant = new_only + old_only
    if discordant == 0:
        return 1.0
    tail = min(new_only, old_only)
    probability = sum(math.comb(discordant, k) for k in range(tail + 1)) / (2**discordant)
    return min(1.0, 2.0 * probability)


def hierarchical_bootstrap_rollout_metric_delta(
    old_correct: np.ndarray,
    new_correct: np.ndarray,
    metric: Literal["sampled_accuracy_at_1", "pass_at_k"],
    k: int,
    samples: int = 10_000,
    seed: int = 1337,
) -> BootstrapResult:
    old_arr = np.asarray(old_correct, dtype=bool)
    new_arr = np.asarray(new_correct, dtype=bool)
    if old_arr.shape != new_arr.shape:
        raise ValueError("old_correct and new_correct must have matching shapes")
    if old_arr.ndim != 2:
        raise ValueError("rollout metrics require arrays shaped [examples, rollouts]")
    if not 1 <= k <= old_arr.shape[1]:
        raise ValueError("k must be between 1 and the number of rollout slots")
    if samples <= 0:
        raise ValueError("samples must be positive")
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=float)
    n_examples = old_arr.shape[0]
    for sample_idx in range(samples):
        example_idx = rng.integers(0, n_examples, size=n_examples)
        old_sample = old_arr[example_idx]
        new_sample = new_arr[example_idx]
        estimates[sample_idx] = _rollout_metric(new_sample, metric, k) - _rollout_metric(
            old_sample,
            metric,
            k,
        )
    low, high = np.quantile(estimates, [0.025, 0.975])
    return BootstrapResult(
        delta=float(_rollout_metric(new_arr, metric, k) - _rollout_metric(old_arr, metric, k)),
        ci_low=float(low),
        ci_high=float(high),
        samples=samples,
    )


def _standard_error(diffs: np.ndarray) -> float:
    if diffs.size <= 1:
        return 0.0
    return float(diffs.std(ddof=1) / math.sqrt(diffs.size))


def _rollout_metric(
    correct: np.ndarray,
    metric: Literal["sampled_accuracy_at_1", "pass_at_k"],
    k: int,
) -> float:
    if metric == "sampled_accuracy_at_1":
        return float(correct[:, :k].mean(axis=1).mean())
    if metric == "pass_at_k":
        return float(correct[:, :k].any(axis=1).mean())
    raise ValueError(f"unknown rollout metric: {metric}")
