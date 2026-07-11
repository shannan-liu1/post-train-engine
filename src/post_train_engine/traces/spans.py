"""Span-level trace contracts for ECHO-style action and observation evidence."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_FROZEN_FORBID = ConfigDict(frozen=True, extra="forbid")

SpanRole = Literal[
    "system",
    "user",
    "assistant_action",
    "environment_observation",
    "critic",
    "verifier",
]
LossMaskKind = Literal["none", "policy", "value", "observation_aux", "verifier_aux"]


class TraceSpan(BaseModel):
    """Token span with enough metadata to separate policy, value, and observation losses."""

    model_config = _FROZEN_FORBID

    span_id: str = Field(..., min_length=1)
    role: SpanRole
    start_token: int = Field(..., ge=0)
    end_token: int = Field(..., gt=0)
    text_hash: str = Field(..., pattern=r"^sha256:[0-9a-f]{64}$")
    tool_name: str | None = None
    command_hash: str | None = Field(default=None, pattern=r"^sha256:[0-9a-f]{64}$")
    stdout_hash: str | None = Field(default=None, pattern=r"^sha256:[0-9a-f]{64}$")
    stderr_hash: str | None = Field(default=None, pattern=r"^sha256:[0-9a-f]{64}$")
    loss_mask_kind: LossMaskKind = "none"

    @field_validator("tool_name")
    @classmethod
    def _tool_name_nonempty(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("tool_name must be non-empty when provided")
        return value

    @model_validator(mode="after")
    def _valid_span_range(self) -> TraceSpan:
        if self.end_token <= self.start_token:
            raise ValueError("end_token must be greater than start_token")
        if self.role == "environment_observation" and not self.tool_name:
            raise ValueError("environment observations must record tool_name")
        return self


def build_loss_mask(
    spans: Iterable[TraceSpan],
    *,
    total_tokens: int,
    include_kinds: set[LossMaskKind],
) -> list[int]:
    if total_tokens < 0:
        raise ValueError("total_tokens must be non-negative")
    mask = [0 for _ in range(total_tokens)]
    for span in spans:
        if span.end_token > total_tokens:
            raise ValueError("span end_token exceeds total_tokens")
        if span.loss_mask_kind in include_kinds:
            for idx in range(span.start_token, span.end_token):
                mask[idx] = 1
    return mask


__all__ = ["LossMaskKind", "SpanRole", "TraceSpan", "build_loss_mask"]
