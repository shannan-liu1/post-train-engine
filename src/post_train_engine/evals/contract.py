"""Immutable evaluation law bound into every canonical RunPlan."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


_FROZEN_FORBID = ConfigDict(frozen=True, extra="forbid")
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class EvalContract(BaseModel):
    """Content-addressed promotion suite, prompt, verifier, and generation law."""

    model_config = _FROZEN_FORBID

    schema_version: Literal["eval_contract_v1"] = "eval_contract_v1"
    suite_id: str = Field(..., min_length=1)
    suite_version: str = Field(..., min_length=1)
    role: Literal["promotion"] = "promotion"
    example_ids_sha256: str
    example_content_sha256: str
    prompt_contract_sha256: str
    verifier_contract_sha256: str
    generation_contract_sha256: str
    primary_metric: str = Field(..., min_length=1)
    disclosure: Literal["aggregate"] = "aggregate"

    @field_validator(
        "example_ids_sha256",
        "example_content_sha256",
        "prompt_contract_sha256",
        "verifier_contract_sha256",
        "generation_contract_sha256",
    )
    @classmethod
    def _identity_must_be_sha256(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("evaluation contract identities must use sha256:<digest>")
        return value

    @classmethod
    def from_components(
        cls,
        *,
        suite_id: str,
        suite_version: str,
        example_ids: Sequence[str],
        example_content: Sequence[Any],
        prompt_contract: Mapping[str, Any],
        verifier_contract: Mapping[str, Any],
        generation_contract: Mapping[str, Any],
        primary_metric: str,
    ) -> EvalContract:
        normalized_ids = tuple(str(value) for value in example_ids)
        if not normalized_ids or any(not value for value in normalized_ids):
            raise ValueError("evaluation contract requires non-empty example IDs")
        if len(normalized_ids) != len(set(normalized_ids)):
            raise ValueError("evaluation contract example IDs must be unique")
        normalized_content = tuple(example_content)
        if len(normalized_content) != len(normalized_ids):
            raise ValueError(
                "evaluation contract content rows must match example ID count"
            )
        return cls(
            suite_id=suite_id,
            suite_version=suite_version,
            example_ids_sha256=hash_example_ids(normalized_ids),
            example_content_sha256=_stable_hash(normalized_content),
            prompt_contract_sha256=_stable_hash(prompt_contract),
            verifier_contract_sha256=_stable_hash(verifier_contract),
            generation_contract_sha256=_stable_hash(generation_contract),
            primary_metric=primary_metric,
        )

    @property
    def contract_hash(self) -> str:
        return _stable_hash(self.model_dump(mode="json"))


def hash_example_ids(example_ids: Sequence[str]) -> str:
    """Hash ordered suite membership without disclosing row contents."""

    return _stable_hash([str(value) for value in example_ids])


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = ["EvalContract", "hash_example_ids"]
