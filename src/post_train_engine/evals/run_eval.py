"""Small deterministic eval runner.

Model loading and GPU generation live outside this module for now. The runner
accepts a generation callable so tests and early integrations can exercise the
evaluation contract without downloading models.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from post_train_engine.evals.grades import Grade, summarize_grades
from post_train_engine.evals.source import EvalSource
from post_train_engine.tasks.schema import Example

Generate = Callable[[Example], str]


@dataclass(frozen=True)
class EvalDetail:
    example_id: str
    prompt: str
    generation: str
    grade: Grade


@dataclass(frozen=True)
class EvalRunResult:
    source: str
    summary: dict[str, float]
    details: tuple[EvalDetail, ...]


def run_eval_source(
    source: EvalSource,
    generate: Generate,
    *,
    examples: Sequence[Example] | None = None,
) -> EvalRunResult:
    rows = tuple(examples if examples is not None else source.load_examples())
    details = tuple(
        EvalDetail(
            example_id=example.id,
            prompt=example.prompt,
            generation=(generation := generate(example)),
            grade=source.grade(generation, example),
        )
        for example in rows
    )
    return EvalRunResult(
        source=source.name,
        summary=summarize_grades(detail.grade for detail in details),
        details=details,
    )


def write_eval_result(result: EvalRunResult, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps({"source": result.source, "summary": result.summary}, indent=2),
        encoding="utf-8",
    )
    detail_rows: list[dict[str, Any]] = [
        {
            "example_id": detail.example_id,
            "prompt": detail.prompt,
            "generation": detail.generation,
            "grade": asdict(detail.grade),
        }
        for detail in result.details
    ]
    (out_dir / "details.json").write_text(
        json.dumps(detail_rows, indent=2),
        encoding="utf-8",
    )
