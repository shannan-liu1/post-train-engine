"""Typed trace schema for local rollout evidence."""

from __future__ import annotations

import hashlib
import math
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SplitRole = Literal[
    "train",
    "probe",
    "replay",
    "diagnostic",
    "selection",
    "promotion",
    "canary",
    "unseen",
]

_FROZEN_FORBID = ConfigDict(frozen=True, extra="forbid")


class TraceRecord(BaseModel):
    """One generated trace with enough provenance to reconstruct its source."""

    model_config = _FROZEN_FORBID

    trace_id: str = Field(..., min_length=1)
    run_id: str = Field(..., min_length=1)
    task_id: str = Field(..., min_length=1)
    example_id: str = Field(..., min_length=1)
    split_role: SplitRole
    prompt_hash: str = Field(..., min_length=1)
    source_checkpoint: str = Field(..., min_length=1)
    policy_version: str = Field(..., min_length=1)
    policy_step: int = Field(..., ge=0)
    policy_step_evidence: Literal["exact", "inferred_batch", "static"]
    rollout_group_id: str = Field(..., min_length=1)
    generation_backend: str = Field(..., min_length=1)
    sampling_config: dict[str, Any]
    verifier_id: str = Field(..., min_length=1)
    prompt: str | None = None
    completion: str | None = None
    parsed_answer: str | None = None
    parser_status: dict[str, Any] = Field(default_factory=dict)
    verifier_result: dict[str, Any] = Field(default_factory=dict)
    reward_components: dict[str, float] = Field(default_factory=dict)
    token_counts: dict[str, float] = Field(default_factory=dict)
    privileged_visibility: Literal["none", "gold_answer", "environment", "critic"] = "none"
    artifact_refs: dict[str, str] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    @field_validator("reward_components", mode="before")
    @classmethod
    def _reward_components_must_be_finite(cls, value: Any) -> Any:
        return _validate_number_mapping(
            {} if value is None else value,
            "reward_components",
        )

    @field_validator("token_counts", mode="before")
    @classmethod
    def _token_counts_must_be_non_negative_finite(cls, value: Any) -> Any:
        return _validate_number_mapping(
            {} if value is None else value,
            "token_counts",
            allow_negative=False,
        )

    @field_validator("sampling_config", mode="before")
    @classmethod
    def _sampling_config_must_be_nonempty(cls, value: Any) -> Any:
        if not isinstance(value, dict) or not value:
            raise ValueError("sampling_config must be a non-empty mapping")
        return value


def stable_prompt_hash(prompt: str) -> str:
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _validate_number_mapping(
    value: Any,
    field_name: str,
    *,
    allow_negative: bool = True,
) -> Any:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a mapping")
    for name, number in value.items():
        if (
            not name
            or type(number) is bool
            or not isinstance(number, int | float)
            or not math.isfinite(float(number))
        ):
            raise ValueError(f"{field_name} value {name!r} must be finite")
        if not allow_negative and float(number) < 0.0:
            raise ValueError(f"{field_name} value {name!r} must be non-negative")
    return value


__all__ = ["SplitRole", "TraceRecord", "stable_prompt_hash"]
