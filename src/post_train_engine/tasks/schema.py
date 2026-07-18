"""Common record schema for post-training data.

The trainer consumes normalized ``Example`` records so it does not need
loader-specific record shapes. Task-specific metadata lives in optional fields.

Validation is enforced by Pydantic at construction time. If a loader tries to
build an ``Example`` with bad data, such as an empty required field or an
out-of-range difficulty, Pydantic raises ``ValidationError`` and the
bad record never enters the pipeline. The loader is responsible for catching
and reporting; the schema is responsible for refusing.
"""

from __future__ import annotations

from typing import Any

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

# Dataset identifiers remain extensible, so the source contract is a non-empty string.
Source = str


class Example(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(..., min_length=1)
    source: Source = Field(..., min_length=1)
    prompt: str = Field(..., min_length=1)
    response: str | None = None
    final_answer: str | None = None
    category: str | None = None
    difficulty: int | None = Field(default=None, ge=1, le=5)
    metadata: dict[str, Any] = Field(default_factory=dict)


def validate_unique_example_ids(examples: Sequence[Example]) -> None:
    """Reject duplicate IDs before eval/probe metrics group by example."""

    seen: set[str] = set()
    for example in examples:
        if example.id in seen:
            raise ValueError(f"duplicate example id: {example.id}")
        seen.add(example.id)
