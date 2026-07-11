"""Probe a model against verifier-backed examples."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from post_train_engine.evals.grades import Grade, summarize_grades
from post_train_engine.evals.source import EvalSource
from post_train_engine.jsonl import read_jsonl, write_jsonl
from post_train_engine.tasks.schema import Example, validate_unique_example_ids

ProbeGenerate = Callable[[Example, int], str]


@dataclass(frozen=True)
class ProbeTrace:
    source: str
    example_id: str
    prompt: str
    sample_index: int
    generation: str
    grade: Grade


@dataclass(frozen=True)
class EarlyExitConfig:
    g_total: int = 16
    g_early: int = 4
    early_min_successes: int = 1
    early_max_successes: int = 3
    full_min_successes: int = 2
    full_max_successes: int = 12

    def __post_init__(self) -> None:
        if self.g_total <= 0 or self.g_early <= 0:
            raise ValueError("g_total and g_early must be positive")
        if self.g_early > self.g_total:
            raise ValueError("g_early cannot exceed g_total")
        if not 0 <= self.early_min_successes <= self.g_early:
            raise ValueError("early_min_successes must be between 0 and g_early")
        if self.early_max_successes < 0:
            raise ValueError("early_max_successes must be non-negative")
        if not 0 <= self.full_min_successes <= self.g_total:
            raise ValueError("full_min_successes must be between 0 and g_total")
        if self.full_max_successes < 0:
            raise ValueError("full_max_successes must be non-negative")
        if self.early_min_successes > self.early_max_successes:
            raise ValueError("early_min_successes cannot exceed early_max_successes")
        if self.full_min_successes > self.full_max_successes:
            raise ValueError("full_min_successes cannot exceed full_max_successes")


@dataclass(frozen=True)
class ProbeArtifactRow:
    run_id: str
    example_id: str
    bucket_source_split: str
    model_id: str
    checkpoint_id: str
    prompt_hash: str
    prompt: str
    rollout_id: int
    seed: int
    generation_config: Mapping[str, Any]
    completion: str
    completion_tokens: int
    finish_reason: str
    parsed_answer: str | None
    parse_ok: bool
    parser: str | None
    gold_answer: str
    correct: bool
    reward: float
    latency_ms: int | float
    error: str | None
    question: str | None = None
    gold_solution: str | None = None
    trace_id: str | None = None
    filter_reason: str | None = None

    def to_json(self) -> dict[str, Any]:
        body = asdict(self)
        if body["trace_id"] is None:
            body["trace_id"] = f"{self.run_id}:{self.example_id}:{self.rollout_id}"
        return body


@dataclass(frozen=True)
class ProbeResult:
    source: str
    summary: dict[str, float]
    traces: tuple[ProbeTrace, ...]
    artifacts: Mapping[str, Any] = field(default_factory=dict)

    @property
    def example_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(trace.example_id for trace in self.traces))


class ProbeRunner:
    """Run one or more generations per example and keep verifier traces."""

    def __init__(self, source: EvalSource) -> None:
        self.source = source

    def run(
        self,
        generate: ProbeGenerate,
        *,
        examples: Sequence[Example] | None = None,
        samples_per_example: int = 1,
    ) -> ProbeResult:
        if samples_per_example <= 0:
            raise ValueError("samples_per_example must be positive")

        rows = tuple(examples if examples is not None else self.source.load_examples())
        validate_unique_example_ids(rows)
        traces: list[ProbeTrace] = []
        for example in rows:
            for sample_index in range(samples_per_example):
                generation = generate(example, sample_index)
                traces.append(
                    ProbeTrace(
                        source=self.source.name,
                        example_id=example.id,
                        prompt=example.prompt,
                        sample_index=sample_index,
                        generation=generation,
                        grade=self.source.grade(generation, example),
                    )
                )
        return ProbeResult(
            source=self.source.name,
            summary=summarize_grades(trace.grade for trace in traces),
            traces=tuple(traces),
        )


def should_continue_after_early(successes: int, cfg: EarlyExitConfig) -> bool:
    return cfg.early_min_successes <= successes <= cfg.early_max_successes


def should_train_after_full(successes: int, cfg: EarlyExitConfig) -> bool:
    return cfg.full_min_successes <= successes <= cfg.full_max_successes


def early_exit_rejection_reason(successes: int, cfg: EarlyExitConfig) -> str | None:
    if should_continue_after_early(successes, cfg):
        return None
    return "early_too_hard" if successes < cfg.early_min_successes else "early_too_easy"


def full_filter_reason(successes: int, cfg: EarlyExitConfig) -> str:
    if should_train_after_full(successes, cfg):
        return "accepted_frontier"
    return "full_too_hard" if successes < cfg.full_min_successes else "full_too_easy"


def write_probe_jsonl(rows: Sequence[ProbeArtifactRow], path: str | Path) -> None:
    write_jsonl(path, (row.to_json() for row in rows))


def read_probe_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return read_jsonl(path)
