"""Fail-closed promotion gates for paired eval artifacts."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

import yaml

from post_train_engine.evals.statistics import PairedAccuracyStats, paired_accuracy_stats
from post_train_engine.evals.harness import EvalReport, EvalSample

Severity = Literal["none", "low", "medium", "high", "critical"]

_PAIRED_CORRECTNESS_PRIMARY_METRICS = {
    "accuracy",
    "accuracy@1",
    "exact_accuracy@1",
    "greedy_exact_accuracy@1",
    "pass@1",
}
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

_SEVERITY_ORDER: dict[Severity, int] = {
    "none": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}


@dataclass(frozen=True)
class EvalExampleResult:
    example_id: str
    correct: bool
    parse_ok: bool = True
    tokens: int | float | None = None
    bucket: str | None = None
    failure_type: str | None = None
    severity: Severity = "none"
    protected: bool = False

    def __post_init__(self) -> None:
        if not self.example_id:
            raise ValueError("eval example id must be non-empty")
        if type(self.correct) is not bool:
            raise ValueError("correct must be a boolean")
        if type(self.parse_ok) is not bool:
            raise ValueError("parse_ok must be a boolean")
        if self.tokens is not None:
            if (
                type(self.tokens) is bool
                or not isinstance(self.tokens, int | float)
                or not math.isfinite(float(self.tokens))
            ):
                raise ValueError("tokens must be a finite number when provided")
            if float(self.tokens) < 0:
                raise ValueError("tokens must be non-negative")
        if self.failure_type is not None and not self.failure_type:
            raise ValueError("failure_type must be non-empty when provided")
        if self.severity not in _SEVERITY_ORDER:
            raise ValueError("severity must be none, low, medium, high, or critical")
        if type(self.protected) is not bool:
            raise ValueError("protected must be a boolean")


@dataclass(frozen=True)
class EvalArtifact:
    artifact_id: str
    primary_metric: str
    examples: tuple[EvalExampleResult, ...]
    evaluation_contract_hash: str | None = None
    metrics: Mapping[str, float] = field(default_factory=dict)
    slices: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.artifact_id:
            raise ValueError("artifact_id must be non-empty")
        if not self.primary_metric:
            raise ValueError("primary_metric must be non-empty")
        if self.evaluation_contract_hash is not None and not _SHA256_RE.fullmatch(
            self.evaluation_contract_hash
        ):
            raise ValueError(
                "evaluation_contract_hash must use sha256:<64 lowercase hex chars>"
            )
        _assert_unique_eval_example_ids(self.examples)
        _assert_finite_metric_map(self.metrics, label="metrics")
        for slice_name, values in self.slices.items():
            if not slice_name:
                raise ValueError("slice name must be non-empty")
            _assert_finite_metric_map(values, label=f"slice {slice_name!r}")

    @classmethod
    def from_dict(cls, body: Mapping[str, Any]) -> EvalArtifact:
        raw_examples = (
            body["examples"] if "examples" in body else body.get("example_results")
        )
        if raw_examples is None:
            raise ValueError("eval artifact must include examples or example_results")
        examples = tuple(_example_from_mapping(row) for row in raw_examples)
        artifact_id = str(
            body.get("artifact_id")
            or body.get("run_id")
            or body.get("candidate_id")
            or "unknown"
        )
        primary_metric = str(body.get("primary_metric") or "greedy_exact_accuracy@1")
        metrics = _metric_map_from_mapping(body.get("metrics", {}), label="metrics")
        if primary_metric not in metrics:
            metrics = dict(metrics)
            metrics[primary_metric] = _accuracy(examples)
        return cls(
            artifact_id=artifact_id,
            primary_metric=primary_metric,
            evaluation_contract_hash=(
                str(body["evaluation_contract_hash"])
                if body.get("evaluation_contract_hash") is not None
                else None
            ),
            examples=examples,
            metrics=metrics,
            slices={
                str(name): _metric_map_from_mapping(
                    values,
                    label=f"slice {name!r}",
                )
                for name, values in dict(body.get("slices", {})).items()
            },
            metadata=dict(body.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "primary_metric": self.primary_metric,
            "evaluation_contract_hash": self.evaluation_contract_hash,
            "metrics": dict(self.metrics),
            "slices": {key: dict(value) for key, value in self.slices.items()},
            "examples": [asdict(example) for example in self.examples],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SliceGateConfig:
    slice_name: str
    metric: str
    min_delta: float | None = None
    min_ci_low: float | None = None
    max_regression: float = 0.0
    required: bool = True

    def __post_init__(self) -> None:
        if not self.slice_name:
            raise ValueError("slice_name must be non-empty")
        if not self.metric:
            raise ValueError("slice gate metric must be non-empty")
        if self.min_delta is not None and not _is_finite_number(self.min_delta):
            raise ValueError("slice gate min_delta must be finite")
        if self.min_ci_low is not None and not _is_finite_number(self.min_ci_low):
            raise ValueError("slice gate min_ci_low must be finite")
        if not _is_finite_number(self.max_regression):
            raise ValueError("slice gate max_regression must be finite")
        if self.max_regression < 0.0:
            raise ValueError("slice gate max_regression must be non-negative")
        if type(self.required) is not bool:
            raise ValueError("slice gate required must be a boolean")


@dataclass(frozen=True)
class CanaryDecision:
    decision: Literal["pass", "fail"]
    failed_examples: tuple[str, ...] = ()
    failure_types: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.decision not in {"pass", "fail"}:
            raise ValueError("canary decision must be pass or fail")
        if self.decision == "pass" and self.failed_examples:
            raise ValueError("passing canary decision cannot include failed examples")
        if len(set(self.failed_examples)) != len(self.failed_examples):
            raise ValueError("failed canary examples must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "failed_examples": list(self.failed_examples),
            "failure_types": dict(self.failure_types),
        }


@dataclass(frozen=True)
class PromotionGateConfig:
    min_examples: int = 1
    min_primary_delta: float = 0.02
    min_primary_ci_low: float = 0.005
    max_mcnemar_p: float = 0.05
    max_parse_regression: float = 0.005
    max_easy_regression: float = 0.01
    max_token_increase_ratio: float = 1.25
    required_slice_gates: tuple[SliceGateConfig, ...] = ()
    max_critical_regressions: int = 0
    max_high_severity_regressions: int = 0
    max_medium_severity_regression_rate: float = 0.005

    def __post_init__(self) -> None:
        if type(self.min_examples) is bool or not isinstance(self.min_examples, int):
            raise ValueError("min_examples must be an integer")
        if self.min_examples <= 0:
            raise ValueError("min_examples must be positive")
        for name in (
            "min_primary_delta",
            "min_primary_ci_low",
            "max_mcnemar_p",
            "max_parse_regression",
            "max_easy_regression",
            "max_token_increase_ratio",
            "max_medium_severity_regression_rate",
        ):
            if not _is_finite_number(getattr(self, name)):
                raise ValueError(f"{name} must be a finite number")
        for name in ("max_critical_regressions", "max_high_severity_regressions"):
            value = getattr(self, name)
            if type(value) is bool or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.min_primary_delta < 0.0:
            raise ValueError("min_primary_delta must be non-negative")
        if not 0.0 <= self.max_mcnemar_p <= 1.0:
            raise ValueError("max_mcnemar_p must be between 0 and 1")
        if self.max_parse_regression < 0.0:
            raise ValueError("max_parse_regression must be non-negative")
        if self.max_easy_regression < 0.0:
            raise ValueError("max_easy_regression must be non-negative")
        if self.max_token_increase_ratio <= 0.0:
            raise ValueError("max_token_increase_ratio must be positive")
        if not 0.0 <= self.max_medium_severity_regression_rate <= 1.0:
            raise ValueError(
                "max_medium_severity_regression_rate must be between 0 and 1"
            )
        for gate in self.required_slice_gates:
            if not isinstance(gate, SliceGateConfig):
                raise ValueError("required_slice_gates must contain SliceGateConfig")


@dataclass(frozen=True)
class PromotionDecision:
    decision: Literal["promote", "reject"]
    primary_metric: str
    primary_delta: float
    primary_ci95: tuple[float, float]
    mcnemar_p: float
    gates: Mapping[str, Literal["pass", "fail"]]
    rejection_reasons: tuple[str, ...]
    new_only: int
    old_only: int
    stats: PairedAccuracyStats
    severity_summary: Mapping[Severity, int] = field(default_factory=dict)
    canary_decision: Mapping[str, Any] | None = None
    baseline_metrics: Mapping[str, float] = field(default_factory=dict)
    candidate_metrics: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "primary_metric": self.primary_metric,
            "primary_delta": self.primary_delta,
            "primary_ci95": list(self.primary_ci95),
            "mcnemar_p": self.mcnemar_p,
            "gates": dict(self.gates),
            "rejection_reasons": list(self.rejection_reasons),
            "new_only": self.new_only,
            "old_only": self.old_only,
            "stats": asdict(self.stats),
            "severity_summary": dict(self.severity_summary),
            "canary_decision": self.canary_decision,
            "baseline_metrics": dict(self.baseline_metrics),
            "candidate_metrics": dict(self.candidate_metrics),
        }


def decide_promotion(
    baseline_eval: EvalArtifact,
    candidate_eval: EvalArtifact,
    cfg: PromotionGateConfig,
    *,
    canary_decision: CanaryDecision | Mapping[str, Any] | None = None,
) -> PromotionDecision:
    if (
        baseline_eval.evaluation_contract_hash is None
        or candidate_eval.evaluation_contract_hash is None
    ):
        raise ValueError("promotion artifacts require an evaluation contract")
    if (
        baseline_eval.evaluation_contract_hash
        != candidate_eval.evaluation_contract_hash
    ):
        raise ValueError("promotion artifacts use different evaluation contracts")
    if baseline_eval.primary_metric != candidate_eval.primary_metric:
        raise ValueError(
            "eval artifacts must declare the same primary_metric; "
            f"baseline={baseline_eval.primary_metric!r}, "
            f"candidate={candidate_eval.primary_metric!r}"
        )
    if candidate_eval.primary_metric not in _PAIRED_CORRECTNESS_PRIMARY_METRICS:
        raise ValueError(
            "paired promotion only supports example-level correctness primary metrics; "
            f"got {candidate_eval.primary_metric!r}"
        )
    paired_rows = _paired_rows(baseline_eval, candidate_eval)
    old = [baseline.correct for baseline, _candidate in paired_rows]
    new = [candidate.correct for _baseline, candidate in paired_rows]
    stats = paired_accuracy_stats(old, new)
    primary_metric = candidate_eval.primary_metric
    primary_delta = stats.delta
    primary_ci = (stats.bootstrap_ci_low, stats.bootstrap_ci_high)
    gates: dict[str, Literal["pass", "fail"]] = {}
    reasons: list[str] = []

    _gate(
        "min_examples",
        stats.n >= cfg.min_examples,
        gates,
        reasons,
        f"underpowered_eval n={stats.n} < {cfg.min_examples}",
    )

    _gate(
        "min_delta",
        primary_delta >= cfg.min_primary_delta,
        gates,
        reasons,
        f"primary_delta {primary_delta:.6f} < {cfg.min_primary_delta:.6f}",
    )
    _gate(
        "ci_low",
        primary_ci[0] >= cfg.min_primary_ci_low,
        gates,
        reasons,
        f"primary_ci_low {primary_ci[0]:.6f} < {cfg.min_primary_ci_low:.6f}",
    )
    _gate(
        "mcnemar",
        stats.mcnemar_p <= cfg.max_mcnemar_p,
        gates,
        reasons,
        f"mcnemar_p {stats.mcnemar_p:.6f} > {cfg.max_mcnemar_p:.6f}",
    )
    parse_regression = _parse_rate(baseline_eval.examples) - _parse_rate(
        candidate_eval.examples,
    )
    _gate(
        "parse_regression",
        parse_regression <= cfg.max_parse_regression,
        gates,
        reasons,
        f"parse_regression {parse_regression:.6f} > {cfg.max_parse_regression:.6f}",
    )
    baseline_easy = _slice_metric(baseline_eval, "easy_stable", primary_metric)
    candidate_easy = _slice_metric(candidate_eval, "easy_stable", primary_metric)
    if baseline_easy is None or candidate_easy is None:
        _gate(
            "easy_regression",
            False,
            gates,
            reasons,
            "easy_regression missing easy_stable evidence",
        )
    else:
        easy_regression = baseline_easy - candidate_easy
        _gate(
            "easy_regression",
            easy_regression <= cfg.max_easy_regression,
            gates,
            reasons,
            f"easy_regression {easy_regression:.6f} > {cfg.max_easy_regression:.6f}",
        )
    token_ratio = _token_ratio(baseline_eval.examples, candidate_eval.examples)
    _gate(
        "token_budget",
        token_ratio <= cfg.max_token_increase_ratio,
        gates,
        reasons,
        f"token_increase_ratio {token_ratio:.6f} > {cfg.max_token_increase_ratio:.6f}",
    )
    for slice_gate in cfg.required_slice_gates:
        _apply_slice_gate(
            baseline_eval,
            candidate_eval,
            slice_gate,
            primary_metric=primary_metric,
            gates=gates,
            reasons=reasons,
        )

    severity_summary = _severity_summary(paired_rows)
    _gate(
        "critical_severity",
        severity_summary["critical"] <= cfg.max_critical_regressions,
        gates,
        reasons,
        (
            f"critical_regressions {severity_summary['critical']} > "
            f"{cfg.max_critical_regressions}"
        ),
    )
    _gate(
        "high_severity",
        severity_summary["high"] <= cfg.max_high_severity_regressions,
        gates,
        reasons,
        (
            f"high_severity_regressions {severity_summary['high']} > "
            f"{cfg.max_high_severity_regressions}"
        ),
    )
    medium_rate = severity_summary["medium"] / len(paired_rows)
    _gate(
        "medium_severity_rate",
        medium_rate <= cfg.max_medium_severity_regression_rate,
        gates,
        reasons,
        (
            f"medium_severity_regression_rate {medium_rate:.6f} > "
            f"{cfg.max_medium_severity_regression_rate:.6f}"
        ),
    )
    canary = _coerce_canary_decision(canary_decision)
    if canary is not None:
        _gate(
            "canary",
            canary.decision == "pass",
            gates,
            reasons,
            "canary decision failed",
        )

    return PromotionDecision(
        decision="promote" if not reasons else "reject",
        primary_metric=primary_metric,
        primary_delta=primary_delta,
        primary_ci95=primary_ci,
        mcnemar_p=stats.mcnemar_p,
        gates=gates,
        rejection_reasons=tuple(reasons),
        new_only=stats.new_only,
        old_only=stats.old_only,
        stats=stats,
        severity_summary=severity_summary,
        canary_decision=None if canary is None else canary.to_dict(),
        baseline_metrics=dict(baseline_eval.metrics),
        candidate_metrics=dict(candidate_eval.metrics),
    )


def load_eval_artifact(path: str | Path) -> EvalArtifact:
    return EvalArtifact.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def eval_report_to_artifact(report: EvalReport) -> EvalArtifact:
    """Convert a generic eval harness report into a paired promotion artifact."""

    details = _first_sample_details(report.details)
    return EvalArtifact(
        artifact_id=report.candidate_id,
        primary_metric=report.primary_metric,
        examples=tuple(_result_from_sample(sample) for sample in details),
        metrics={
            name: result.value
            for name, result in report.metrics.items()
        },
        slices=_slice_metrics(report),
        metadata={
            **dict(report.metadata),
            "source": report.source,
            "spec_name": report.spec_name,
        },
    )


def write_promotion_decision(decision: PromotionDecision, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(decision.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_promotion_gate_config(path: str | Path) -> PromotionGateConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if raw is None:
        return PromotionGateConfig()
    if not isinstance(raw, dict):
        raise ValueError("promotion gate config root must be a mapping")
    known = {field.name for field in fields(PromotionGateConfig)}
    unknown = sorted(str(key) for key in raw if str(key) not in known)
    if unknown:
        joined = ", ".join(unknown)
        raise ValueError(f"unknown promotion gate config field(s): {joined}")
    values = dict(raw)
    if "required_slice_gates" in values:
        values["required_slice_gates"] = tuple(
            gate
            if isinstance(gate, SliceGateConfig)
            else SliceGateConfig(**dict(gate))
            for gate in values["required_slice_gates"]
        )
    return PromotionGateConfig(**values)


def canary_decision_from_artifact(artifact: EvalArtifact) -> CanaryDecision:
    failed_examples: list[str] = []
    failure_types: dict[str, str] = {}
    for example in artifact.examples:
        if example.correct and example.parse_ok and example.severity == "none":
            continue
        failed_examples.append(example.example_id)
        failure_types[example.example_id] = (
            example.failure_type
            or ("parse" if not example.parse_ok else "incorrect")
        )
    return CanaryDecision(
        decision="fail" if failed_examples else "pass",
        failed_examples=tuple(failed_examples),
        failure_types=failure_types,
    )


def _first_sample_details(samples: Sequence[EvalSample]) -> tuple[EvalSample, ...]:
    selected: dict[str, EvalSample] = {}
    for sample in samples:
        if sample.sample_index != 0:
            continue
        if sample.example_id in selected:
            raise ValueError(f"duplicate first-sample eval detail: {sample.example_id}")
        selected[sample.example_id] = sample
    if not selected and samples:
        raise ValueError("eval report has no sample_index=0 details")
    return tuple(selected[example_id] for example_id in sorted(selected))


def _result_from_sample(sample: EvalSample) -> EvalExampleResult:
    tokens = _completion_tokens(sample)
    return EvalExampleResult(
        example_id=sample.example_id,
        correct=sample.grade.is_correct,
        parse_ok=sample.grade.parse_success,
        tokens=tokens,
        bucket=_bucket_from_sample(sample),
    )


def _completion_tokens(sample: EvalSample) -> int | float:
    for key in ("completion_tokens", "tokens"):
        value = sample.grade.metadata.get(key)
        if value is not None:
            return value
    raise ValueError(f"eval sample {sample.example_id} is missing completion token count")


def _bucket_from_sample(sample: EvalSample) -> str | None:
    for key in ("bucket", "difficulty_bucket", "category"):
        value = sample.slices.get(key) or sample.grade.metadata.get(key)
        if value is not None:
            return str(value)
    return None


def _slice_metrics(report: EvalReport) -> dict[str, dict[str, float]]:
    slices: dict[str, dict[str, float]] = {}
    for slice_report in report.slices:
        key = slice_report.value
        if slice_report.field != "bucket":
            key = f"{slice_report.field}:{slice_report.value}"
        slices[key] = {
            metric_name: metric.value
            for metric_name, metric in slice_report.metrics.items()
        }
    return slices


def _example_from_mapping(row: Mapping[str, Any]) -> EvalExampleResult:
    return EvalExampleResult(
        example_id=str(row["example_id"]),
        correct=_required_bool(row, "correct", fallback="greedy_correct"),
        parse_ok=_optional_bool(row, "parse_ok", default=True),
        tokens=row.get("tokens", row.get("completion_tokens")),
        bucket=None if row.get("bucket") is None else str(row.get("bucket")),
        failure_type=(
            None if row.get("failure_type") is None else str(row.get("failure_type"))
        ),
        severity=str(row.get("severity", "none")),
        protected=_optional_bool(row, "protected", default=False),
    )


def _required_bool(
    row: Mapping[str, Any],
    key: str,
    *,
    fallback: str | None = None,
) -> bool:
    if key in row:
        value = row[key]
    elif fallback is not None and fallback in row:
        value = row[fallback]
    else:
        raise ValueError(f"{key} must be present")
    if type(value) is not bool:
        raise ValueError(f"{key} must be a boolean")
    return value


def _optional_bool(row: Mapping[str, Any], key: str, *, default: bool) -> bool:
    if key not in row:
        return default
    value = row[key]
    if type(value) is not bool:
        raise ValueError(f"{key} must be a boolean")
    return value


def _assert_unique_eval_example_ids(examples: Sequence[EvalExampleResult]) -> None:
    seen: set[str] = set()
    for example in examples:
        if example.example_id in seen:
            raise ValueError(f"duplicate eval example id: {example.example_id}")
        seen.add(example.example_id)


def _assert_finite_metric_map(values: Mapping[str, float], *, label: str) -> None:
    for name, value in values.items():
        if not name:
            raise ValueError(f"{label} contains an empty metric name")
        if (
            type(value) is bool
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"{label} metric {name!r} must be finite")


def _metric_map_from_mapping(values: Any, *, label: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name, value in dict(values).items():
        metric_name = str(name)
        if not metric_name:
            raise ValueError(f"{label} contains an empty metric name")
        if not _is_finite_number(value):
            raise ValueError(f"{label} metric {metric_name!r} must be finite")
        metrics[metric_name] = float(value)
    return metrics


def _is_finite_number(value: object) -> bool:
    return (
        type(value) is not bool
        and isinstance(value, int | float)
        and math.isfinite(float(value))
    )


def _paired_correctness(
    baseline: EvalArtifact,
    candidate: EvalArtifact,
) -> tuple[list[bool], list[bool]]:
    paired = _paired_rows(baseline, candidate)
    return (
        [baseline_row.correct for baseline_row, _candidate_row in paired],
        [candidate_row.correct for _baseline_row, candidate_row in paired],
    )


def _paired_rows(
    baseline: EvalArtifact,
    candidate: EvalArtifact,
) -> tuple[tuple[EvalExampleResult, EvalExampleResult], ...]:
    baseline_by_id = {row.example_id: row for row in baseline.examples}
    candidate_by_id = {row.example_id: row for row in candidate.examples}
    ids = sorted(set(baseline_by_id) & set(candidate_by_id))
    if not ids:
        raise ValueError("eval artifacts have no paired example ids")
    missing = set(baseline_by_id) ^ set(candidate_by_id)
    if missing:
        raise ValueError(f"eval artifacts are not paired; first missing id: {sorted(missing)[0]}")
    return tuple(
        (baseline_by_id[example_id], candidate_by_id[example_id])
        for example_id in ids
    )


def _apply_slice_gate(
    baseline: EvalArtifact,
    candidate: EvalArtifact,
    slice_gate: SliceGateConfig,
    *,
    primary_metric: str,
    gates: dict[str, Literal["pass", "fail"]],
    reasons: list[str],
) -> None:
    gate_name = f"slice:{slice_gate.slice_name}:{slice_gate.metric}"
    baseline_metric = _slice_metric(baseline, slice_gate.slice_name, slice_gate.metric)
    candidate_metric = _slice_metric(candidate, slice_gate.slice_name, slice_gate.metric)
    if baseline_metric is None or candidate_metric is None:
        _gate(
            gate_name,
            not slice_gate.required,
            gates,
            reasons,
            f"{gate_name} missing required slice evidence",
        )
        return
    delta = candidate_metric - baseline_metric
    regression = baseline_metric - candidate_metric
    checks = [regression <= slice_gate.max_regression]
    reason_bits = [
        f"regression {regression:.6f} > {slice_gate.max_regression:.6f}",
    ]
    if slice_gate.min_delta is not None:
        checks.append(delta >= slice_gate.min_delta)
        reason_bits.append(f"delta {delta:.6f} < {slice_gate.min_delta:.6f}")
    if slice_gate.min_ci_low is not None:
        if slice_gate.metric != primary_metric:
            checks.append(False)
            reason_bits.append("ci requested for non-primary slice metric")
        else:
            stats = _slice_paired_stats(baseline, candidate, slice_gate.slice_name)
            if stats is None:
                checks.append(False)
                reason_bits.append("ci requested but slice pairs are missing")
            else:
                checks.append(stats.bootstrap_ci_low >= slice_gate.min_ci_low)
                reason_bits.append(
                    "ci_low "
                    f"{stats.bootstrap_ci_low:.6f} < {slice_gate.min_ci_low:.6f}"
                )
    passed = all(checks)
    _gate(
        gate_name,
        passed,
        gates,
        reasons,
        f"{gate_name} failed: {', '.join(reason_bits)}",
    )


def _slice_paired_stats(
    baseline: EvalArtifact,
    candidate: EvalArtifact,
    slice_name: str,
) -> PairedAccuracyStats | None:
    rows = [
        (baseline_row.correct, candidate_row.correct)
        for baseline_row, candidate_row in _paired_rows(baseline, candidate)
        if baseline_row.bucket == slice_name or candidate_row.bucket == slice_name
    ]
    if not rows:
        return None
    return paired_accuracy_stats(
        [baseline_correct for baseline_correct, _candidate_correct in rows],
        [candidate_correct for _baseline_correct, candidate_correct in rows],
    )


def _severity_summary(
    paired_rows: Sequence[tuple[EvalExampleResult, EvalExampleResult]],
) -> dict[Severity, int]:
    summary: dict[Severity, int] = {
        "none": 0,
        "low": 0,
        "medium": 0,
        "high": 0,
        "critical": 0,
    }
    for baseline, candidate in paired_rows:
        summary[_regression_severity(baseline, candidate)] += 1
    return summary


def _regression_severity(
    baseline: EvalExampleResult,
    candidate: EvalExampleResult,
) -> Severity:
    explicit = candidate.severity
    inferred: Severity = "none"
    damaged = (
        (baseline.correct and not candidate.correct)
        or (baseline.parse_ok and not candidate.parse_ok)
        or explicit != "none"
    )
    if (
        damaged
        and (
            candidate.protected
            or candidate.failure_type in {"canary", "schema", "safety", "tool"}
        )
    ):
        inferred = "critical"
    elif baseline.correct and not candidate.correct and candidate.bucket == "easy_stable":
        inferred = "high"
    elif baseline.parse_ok and not candidate.parse_ok:
        inferred = "medium"
    elif baseline.correct and not candidate.correct:
        inferred = "low"
    if _SEVERITY_ORDER[explicit] > _SEVERITY_ORDER[inferred]:
        return explicit
    return inferred


def _coerce_canary_decision(
    canary_decision: CanaryDecision | Mapping[str, Any] | None,
) -> CanaryDecision | None:
    if canary_decision is None:
        return None
    if isinstance(canary_decision, CanaryDecision):
        return canary_decision
    return CanaryDecision(
        decision=str(canary_decision["decision"]),
        failed_examples=tuple(str(item) for item in canary_decision.get("failed_examples", ())),
        failure_types={
            str(key): str(value)
            for key, value in dict(canary_decision.get("failure_types", {})).items()
        },
    )


def _accuracy(examples: Sequence[EvalExampleResult]) -> float:
    if not examples:
        return 0.0
    return sum(example.correct for example in examples) / len(examples)


def _parse_rate(examples: Sequence[EvalExampleResult]) -> float:
    if not examples:
        return 0.0
    return sum(example.parse_ok for example in examples) / len(examples)


def _slice_metric(artifact: EvalArtifact, slice_name: str, metric: str) -> float | None:
    if slice_name in artifact.slices and metric in artifact.slices[slice_name]:
        return artifact.slices[slice_name][metric]
    rows = [example for example in artifact.examples if example.bucket == slice_name]
    if not rows:
        return None
    return _accuracy(rows)


def _token_ratio(
    baseline_examples: Sequence[EvalExampleResult],
    candidate_examples: Sequence[EvalExampleResult],
) -> float:
    baseline = _mean_tokens(baseline_examples)
    candidate = _mean_tokens(candidate_examples)
    if baseline is None or candidate is None:
        return float("inf")
    if baseline == 0.0:
        return 1.0 if candidate == 0.0 else float("inf")
    return candidate / baseline


def _mean_tokens(examples: Sequence[EvalExampleResult]) -> float | None:
    tokens: list[float] = []
    for example in examples:
        if example.tokens is None:
            return None
        value = float(example.tokens)
        if not math.isfinite(value):
            return None
        tokens.append(value)
    if not tokens:
        return None
    return sum(tokens) / len(tokens)


def _gate(
    name: str,
    passed: bool,
    gates: dict[str, Literal["pass", "fail"]],
    reasons: list[str],
    reason: str,
) -> None:
    gates[name] = "pass" if passed else "fail"
    if not passed:
        reasons.append(reason)
