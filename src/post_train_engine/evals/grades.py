"""Generic verifier grade shape."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Grade:
    """Result of grading one model generation."""

    parsed_answer: str | None
    parse_success: bool
    is_correct: bool
    reason: str
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.reason:
            raise ValueError("grade reason must be non-empty")


def summarize_grades(grades: Iterable[Grade]) -> dict[str, float]:
    rows = list(grades)
    if not rows:
        return {
            "n": 0.0,
            "parse_success_rate": 0.0,
            "accuracy": 0.0,
            "mean_score": 0.0,
        }
    return {
        "n": float(len(rows)),
        "parse_success_rate": sum(row.parse_success for row in rows) / len(rows),
        "accuracy": sum(row.is_correct for row in rows) / len(rows),
        "mean_score": sum(float(row.score) for row in rows) / len(rows),
    }
