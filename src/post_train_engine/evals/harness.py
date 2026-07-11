"""Evaluation harness for metric, slice, and per-example evidence."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from math import comb, isfinite
from pathlib import Path
from typing import Any

from post_train_engine.evals.grades import Grade
from post_train_engine.evals.source import EvalSource
from post_train_engine.probe import ProbeRunner
from post_train_engine.tasks.schema import Example

GenerateSample = Callable[[Example, int], str]


@dataclass(frozen=True)
class EvalSpec:
    name: str
    primary_metric: str
    metrics: Sequence[str]
    samples_per_example: int = 1
    slice_fields: Sequence[str] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("eval spec name must be non-empty")
        if not self.primary_metric:
            raise ValueError("primary_metric must be non-empty")
        if self.samples_per_example <= 0:
            raise ValueError("samples_per_example must be positive")
        if not self.metrics:
            raise ValueError("eval spec must include at least one metric")
        if self.primary_metric not in self.metrics:
            raise ValueError("primary_metric must be included in metrics")
        for metric in self.metrics:
            _validate_metric_name(metric, self.samples_per_example)


@dataclass(frozen=True)
class MetricResult:
    name: str
    value: float
    n: int
    higher_is_better: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("metric name must be non-empty")
        if not _is_finite_number(self.value):
            raise ValueError(f"metric {self.name!r} must be finite")
        if self.n < 0:
            raise ValueError("metric n must be non-negative")


@dataclass(frozen=True)
class EvalSample:
    example_id: str
    sample_index: int
    prompt: str
    expected_answer: str | None
    generation: str
    grade: Grade
    slices: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SliceReport:
    field: str
    value: str
    n_examples: int
    metrics: Mapping[str, MetricResult]


@dataclass(frozen=True)
class EvalReport:
    spec_name: str
    candidate_id: str
    source: str
    primary_metric: str
    metrics: Mapping[str, MetricResult]
    slices: tuple[SliceReport, ...]
    details: tuple[EvalSample, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def metric(self, name: str) -> float:
        try:
            return self.metrics[name].value
        except KeyError as exc:
            raise ValueError(f"missing eval metric: {name}") from exc

    def to_metric_map(self) -> dict[str, float]:
        return {name: result.value for name, result in self.metrics.items()}


def evaluate_model(
    spec: EvalSpec,
    source: EvalSource,
    generate: GenerateSample,
    *,
    candidate_id: str,
    examples: Sequence[Example] | None = None,
) -> EvalReport:
    if not candidate_id:
        raise ValueError("candidate_id must be non-empty")
    rows = tuple(examples if examples is not None else source.load_examples())
    examples_by_id = {example.id: example for example in rows}
    probe = ProbeRunner(source).run(
        generate,
        examples=rows,
        samples_per_example=spec.samples_per_example,
    )
    details = tuple(
        EvalSample(
            example_id=trace.example_id,
            sample_index=trace.sample_index,
            prompt=trace.prompt,
            expected_answer=examples_by_id[trace.example_id].final_answer,
            generation=trace.generation,
            grade=trace.grade,
            slices=_slice_values(examples_by_id[trace.example_id], spec.slice_fields),
        )
        for trace in probe.traces
    )
    metrics = _compute_metrics(spec.metrics, details)
    slices = _slice_reports(spec.metrics, details, rows, spec.slice_fields)
    return EvalReport(
        spec_name=spec.name,
        candidate_id=candidate_id,
        source=source.name,
        primary_metric=spec.primary_metric,
        metrics=metrics,
        slices=slices,
        details=details,
        metadata=dict(spec.metadata),
    )


def write_eval_report(report: EvalReport, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_report.json").write_text(
        json.dumps(_report_manifest(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (out_dir / "details.json").write_text(
        json.dumps([_sample_row(sample) for sample in report.details], indent=2),
        encoding="utf-8",
    )


def _report_manifest(report: EvalReport) -> dict[str, Any]:
    return {
        "spec_name": report.spec_name,
        "candidate_id": report.candidate_id,
        "source": report.source,
        "primary_metric": report.primary_metric,
        "metrics": {
            name: asdict(metric) for name, metric in sorted(report.metrics.items())
        },
        "slices": [
            {
                "field": slice_report.field,
                "value": slice_report.value,
                "n_examples": slice_report.n_examples,
                "metrics": {
                    name: asdict(metric)
                    for name, metric in sorted(slice_report.metrics.items())
                },
            }
            for slice_report in report.slices
        ],
        "metadata": dict(report.metadata),
    }


def _sample_row(sample: EvalSample) -> dict[str, Any]:
    return {
        "example_id": sample.example_id,
        "sample_index": sample.sample_index,
        "prompt": sample.prompt,
        "expected_answer": sample.expected_answer,
        "generation": sample.generation,
        "grade": asdict(sample.grade),
        "slices": dict(sample.slices),
    }


def _compute_metrics(
    metric_names: Sequence[str],
    samples: Sequence[EvalSample],
) -> dict[str, MetricResult]:
    return {
        metric: _compute_metric(metric, samples)
        for metric in metric_names
    }


def _compute_metric(metric: str, samples: Sequence[EvalSample]) -> MetricResult:
    family, k = _parse_metric_name(metric)
    grouped = _group_by_example(samples)
    if family == "pass":
        value = _pass_at_k(grouped, k)
        n = len(grouped)
    elif family in {"accuracy", "exact_accuracy"}:
        selected = _first_samples(grouped, k)
        value = _mean([sample.grade.is_correct for sample in selected])
        n = len(selected)
    elif family == "parse_success_rate":
        selected = _samples_up_to_k(grouped, k)
        value = _mean([sample.grade.parse_success for sample in selected])
        n = len(selected)
    elif family == "mean_score":
        selected = _samples_up_to_k(grouped, k)
        value = _mean([float(sample.grade.score) for sample in selected])
        n = len(selected)
    elif family == "macro_f1":
        selected = _first_samples(grouped, k)
        value = _macro_f1(selected)
        n = len(selected)
    else:
        raise ValueError(f"unknown metric: {metric}")
    return MetricResult(name=metric, value=value, n=n)


def _slice_reports(
    metric_names: Sequence[str],
    samples: Sequence[EvalSample],
    examples: Sequence[Example],
    slice_fields: Sequence[str],
) -> tuple[SliceReport, ...]:
    reports: list[SliceReport] = []
    examples_by_slice: dict[tuple[str, str], set[str]] = defaultdict(set)
    for example in examples:
        for slice_field, value in _slice_values(example, slice_fields).items():
            examples_by_slice[(slice_field, value)].add(example.id)

    for slice_field, value in sorted(examples_by_slice):
        example_ids = examples_by_slice[(slice_field, value)]
        slice_samples = [
            sample for sample in samples if sample.example_id in example_ids
        ]
        reports.append(
            SliceReport(
                field=slice_field,
                value=value,
                n_examples=len(example_ids),
                metrics=_compute_metrics(metric_names, slice_samples),
            )
        )
    return tuple(reports)


def _slice_values(example: Example, fields: Sequence[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for slice_field in fields:
        value = getattr(example, slice_field, None)
        if value is None:
            value = example.metadata.get(slice_field)
        if value is not None:
            values[slice_field] = str(value)
    return values


def _group_by_example(samples: Sequence[EvalSample]) -> dict[str, list[EvalSample]]:
    grouped: dict[str, list[EvalSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.example_id].append(sample)
    for rows in grouped.values():
        rows.sort(key=lambda sample: sample.sample_index)
    return dict(grouped)


def _samples_up_to_k(
    grouped: Mapping[str, Sequence[EvalSample]],
    k: int,
) -> list[EvalSample]:
    return [
        sample
        for rows in grouped.values()
        for sample in rows
        if sample.sample_index < k
    ]


def _first_samples(
    grouped: Mapping[str, Sequence[EvalSample]],
    k: int,
) -> list[EvalSample]:
    if k != 1:
        raise ValueError("first-sample metrics currently support only @1")
    return [rows[0] for rows in grouped.values() if rows]


def _pass_at_k(grouped: Mapping[str, Sequence[EvalSample]], k: int) -> float:
    if not grouped:
        return 0.0
    estimates: list[float] = []
    for rows in grouped.values():
        n = len(rows)
        correct = sum(sample.grade.is_correct for sample in rows)
        if n < k:
            raise ValueError(f"pass@{k} requires at least {k} samples per example")
        if correct == 0:
            estimates.append(0.0)
        elif n - correct < k:
            estimates.append(1.0)
        else:
            estimates.append(1.0 - comb(n - correct, k) / comb(n, k))
    return sum(estimates) / len(estimates)


def _macro_f1(samples: Sequence[EvalSample]) -> float:
    if not samples:
        return 0.0
    labels = {
        _expected_label(sample) for sample in samples
    }
    labels.update(sample.grade.parsed_answer for sample in samples)
    labels.discard(None)
    if not labels:
        return 0.0

    expected_counts: Counter[str] = Counter()
    predicted_counts: Counter[str] = Counter()
    true_positive: Counter[str] = Counter()
    for sample in samples:
        expected = _expected_label(sample)
        predicted = sample.grade.parsed_answer
        if expected is not None:
            expected_counts[str(expected)] += 1
        if predicted is not None:
            predicted_counts[str(predicted)] += 1
        if expected is not None and predicted is not None and str(expected) == predicted:
            true_positive[str(expected)] += 1

    f1s: list[float] = []
    for label in labels:
        label = str(label)
        tp = true_positive[label]
        fp = predicted_counts[label] - tp
        fn = expected_counts[label] - tp
        denominator = 2 * tp + fp + fn
        f1s.append(0.0 if denominator == 0 else (2 * tp) / denominator)
    return sum(f1s) / len(f1s)


def _expected_label(sample: EvalSample) -> str | None:
    label = (
        sample.expected_answer
        or sample.grade.metadata.get("expected_answer")
        or sample.grade.metadata.get("label")
    )
    return None if label is None else str(label)


def _mean(values: Sequence[float | bool]) -> float:
    if not values:
        return 0.0
    return sum(float(value) for value in values) / len(values)


def _is_finite_number(value: object) -> bool:
    return (
        type(value) is not bool
        and isinstance(value, int | float)
        and isfinite(float(value))
    )


def _parse_metric_name(metric: str) -> tuple[str, int]:
    if "@" in metric:
        family, raw_k = metric.rsplit("@", 1)
        try:
            k = int(raw_k)
        except ValueError as exc:
            raise ValueError(f"invalid metric sample count: {metric}") from exc
    else:
        family = metric
        k = 1
    if family == "pass":
        return family, k
    if family in {"accuracy", "exact_accuracy"}:
        return family, k
    return family, k


def _validate_metric_name(metric: str, samples_per_example: int) -> None:
    family, k = _parse_metric_name(metric)
    if k <= 0:
        raise ValueError(f"metric {metric!r} must use k > 0")
    if k > samples_per_example:
        raise ValueError(
            f"metric {metric!r} requires k={k}, but samples_per_example="
            f"{samples_per_example}"
        )
    if family in {"accuracy", "exact_accuracy", "macro_f1"} and k != 1:
        raise ValueError(f"metric {metric!r} currently supports only @1")
    known = {
        "pass",
        "accuracy",
        "exact_accuracy",
        "parse_success_rate",
        "mean_score",
        "macro_f1",
    }
    if family not in known:
        raise ValueError(f"unknown metric: {metric}")
