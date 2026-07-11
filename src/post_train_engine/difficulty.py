"""Difficulty maps built from probe traces."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from statistics import mean
from typing import Any

from post_train_engine.probe import ProbeResult, ProbeTrace


class DifficultyBand(str, Enum):
    EASY = "easy"
    LEARNABLE = "learnable"
    UNSOLVED = "unsolved"
    PARSER_ISSUE = "parser_issue"


class DifficultyBucket(str, Enum):
    PARSER_ISSUE = "parser_issue"
    EASY_STABLE = "easy_stable"
    FRONTIER = "frontier"
    HARD_SOLVED = "hard_solved"
    UNSOLVED_PARSEABLE = "unsolved_parseable"
    LABEL_OR_VERIFIER_SUSPECT = "label_or_verifier_suspect"


@dataclass(frozen=True)
class DifficultyRecord:
    example_id: str
    band: DifficultyBand
    n: int
    accuracy: float
    parse_success_rate: float
    mean_score: float
    learning_value: float


@dataclass(frozen=True)
class DifficultyBucketConfig:
    g_total: int = 16
    parser_issue_parse_rate_max: float = 0.80
    easy_min_pass_rate: float = 0.85
    frontier_min_pass_rate: float = 0.10
    frontier_max_pass_rate: float = 0.80
    hard_solved_max_pass_rate: float = 0.10

    def __post_init__(self) -> None:
        if self.g_total <= 0:
            raise ValueError("g_total must be positive")
        for name, value in (
            ("parser_issue_parse_rate_max", self.parser_issue_parse_rate_max),
            ("easy_min_pass_rate", self.easy_min_pass_rate),
            ("frontier_min_pass_rate", self.frontier_min_pass_rate),
            ("frontier_max_pass_rate", self.frontier_max_pass_rate),
            ("hard_solved_max_pass_rate", self.hard_solved_max_pass_rate),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.frontier_min_pass_rate > self.frontier_max_pass_rate:
            raise ValueError("frontier_min_pass_rate cannot exceed frontier_max_pass_rate")


@dataclass(frozen=True)
class DifficultyBucketRecord:
    example_id: str
    num_rollouts: int
    num_correct: int
    num_parse_ok: int
    pass_rate: float
    parse_rate: float
    mean_reward: float
    mean_tokens: float
    bucket: DifficultyBucket
    bucket_reason: str
    successful_trace_ids: tuple[str, ...]
    failed_trace_ids: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "num_rollouts": self.num_rollouts,
            "num_correct": self.num_correct,
            "num_parse_ok": self.num_parse_ok,
            "pass_rate": self.pass_rate,
            "parse_rate": self.parse_rate,
            "mean_reward": self.mean_reward,
            "mean_tokens": self.mean_tokens,
            "bucket": self.bucket.value,
            "bucket_reason": self.bucket_reason,
            "successful_trace_ids": list(self.successful_trace_ids),
            "failed_trace_ids": list(self.failed_trace_ids),
        }


@dataclass(frozen=True)
class DifficultyMap:
    records: tuple[DifficultyRecord, ...]

    @classmethod
    def from_probe(
        cls,
        probe: ProbeResult,
        *,
        solved_threshold: float = 0.80,
        parser_issue_threshold: float = 0.50,
    ) -> DifficultyMap:
        if not 0.0 <= solved_threshold <= 1.0:
            raise ValueError("solved_threshold must be between 0 and 1")
        if not 0.0 <= parser_issue_threshold <= 1.0:
            raise ValueError("parser_issue_threshold must be between 0 and 1")

        grouped: dict[str, list[ProbeTrace]] = defaultdict(list)
        for trace in probe.traces:
            grouped[trace.example_id].append(trace)

        records = [
            _record_for_group(
                example_id,
                traces,
                solved_threshold=solved_threshold,
                parser_issue_threshold=parser_issue_threshold,
            )
            for example_id, traces in grouped.items()
        ]
        return cls(records=tuple(records))

    def get(self, example_id: str) -> DifficultyRecord:
        for record in self.records:
            if record.example_id == example_id:
                return record
        raise KeyError(f"unknown example id: {example_id}")

    def by_band(self, band: DifficultyBand) -> tuple[DifficultyRecord, ...]:
        return tuple(record for record in self.records if record.band is band)

    def training_candidates(self) -> tuple[DifficultyRecord, ...]:
        candidates = [
            record
            for record in self.records
            if record.band in {DifficultyBand.LEARNABLE, DifficultyBand.UNSOLVED}
        ]
        return tuple(
            sorted(
                candidates,
                key=lambda record: (-record.learning_value, record.example_id),
            )
        )


def _record_for_group(
    example_id: str,
    traces: list[ProbeTrace],
    *,
    solved_threshold: float,
    parser_issue_threshold: float,
) -> DifficultyRecord:
    n = len(traces)
    if n == 0:
        raise ValueError("cannot build difficulty record from an empty trace group")
    accuracy = sum(trace.grade.is_correct for trace in traces) / n
    parse_success_rate = sum(trace.grade.parse_success for trace in traces) / n
    mean_score = mean(float(trace.grade.score) for trace in traces)

    if parse_success_rate < parser_issue_threshold:
        band = DifficultyBand.PARSER_ISSUE
        learning_value = 0.0
    elif accuracy >= solved_threshold:
        band = DifficultyBand.EASY
        learning_value = 0.0
    elif accuracy > 0.0:
        band = DifficultyBand.LEARNABLE
        learning_value = 1.0 - abs(0.5 - accuracy)
    else:
        band = DifficultyBand.UNSOLVED
        learning_value = 0.5 * parse_success_rate

    return DifficultyRecord(
        example_id=example_id,
        band=band,
        n=n,
        accuracy=accuracy,
        parse_success_rate=parse_success_rate,
        mean_score=mean_score,
        learning_value=learning_value,
    )


def bucket_probe_rollouts(
    example_id: str,
    rollouts: Sequence[Mapping[str, Any]],
    cfg: DifficultyBucketConfig | None = None,
) -> DifficultyBucketRecord:
    cfg = cfg or DifficultyBucketConfig()
    if not rollouts:
        raise ValueError("cannot bucket an example with no rollouts")

    num_rollouts = len(rollouts)
    num_correct = sum(1 for row in rollouts if bool(row.get("correct")))
    num_parse_ok = sum(1 for row in rollouts if bool(row.get("parse_ok")))
    pass_rate = num_correct / num_rollouts
    parse_rate = num_parse_ok / num_rollouts
    mean_reward = mean(float(row.get("reward", 0.0)) for row in rollouts)
    mean_tokens = mean(float(row.get("completion_tokens", 0.0)) for row in rollouts)
    successful = tuple(
        _trace_id(row)
        for row in rollouts
        if bool(row.get("correct"))
    )
    failed = tuple(
        _trace_id(row)
        for row in rollouts
        if not bool(row.get("correct"))
    )
    bucket, reason = _bucket_from_rates(pass_rate, parse_rate, rollouts, cfg)
    return DifficultyBucketRecord(
        example_id=example_id,
        num_rollouts=num_rollouts,
        num_correct=num_correct,
        num_parse_ok=num_parse_ok,
        pass_rate=pass_rate,
        parse_rate=parse_rate,
        mean_reward=mean_reward,
        mean_tokens=mean_tokens,
        bucket=bucket,
        bucket_reason=reason,
        successful_trace_ids=successful,
        failed_trace_ids=failed,
    )


def bucket_probe_artifact(
    rows: Sequence[Mapping[str, Any]],
    cfg: DifficultyBucketConfig | None = None,
) -> tuple[DifficultyBucketRecord, ...]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["example_id"])].append(row)
    return tuple(
        bucket_probe_rollouts(example_id, grouped[example_id], cfg)
        for example_id in sorted(grouped)
    )


def _bucket_from_rates(
    pass_rate: float,
    parse_rate: float,
    rollouts: Sequence[Mapping[str, Any]],
    cfg: DifficultyBucketConfig,
) -> tuple[DifficultyBucket, str]:
    if any(bool(row.get("label_or_verifier_suspect")) for row in rollouts):
        return DifficultyBucket.LABEL_OR_VERIFIER_SUSPECT, "explicit_suspect_flag"
    if any(row.get("gold_answer") in {"", None} for row in rollouts):
        return DifficultyBucket.LABEL_OR_VERIFIER_SUSPECT, "missing_gold_answer"
    if parse_rate < cfg.parser_issue_parse_rate_max:
        return DifficultyBucket.PARSER_ISSUE, "parse_rate_below_threshold"
    if pass_rate >= cfg.easy_min_pass_rate:
        return DifficultyBucket.EASY_STABLE, "pass_rate_above_easy_threshold"
    if cfg.frontier_min_pass_rate <= pass_rate <= cfg.frontier_max_pass_rate:
        return DifficultyBucket.FRONTIER, "pass_rate_in_frontier_range"
    if 0.0 < pass_rate < cfg.hard_solved_max_pass_rate:
        return DifficultyBucket.HARD_SOLVED, "rare_successes_below_hard_threshold"
    if pass_rate == 0.0:
        return DifficultyBucket.UNSOLVED_PARSEABLE, "no_successes_but_parseable"
    return DifficultyBucket.LABEL_OR_VERIFIER_SUSPECT, "bucket_threshold_gap"


def _trace_id(row: Mapping[str, Any]) -> str:
    trace_id = row.get("trace_id")
    if trace_id is not None:
        return str(trace_id)
    rollout_id = row.get("rollout_id", "")
    run_id = row.get("run_id", "")
    example_id = row.get("example_id", "")
    return f"{run_id}:{example_id}:{rollout_id}"
