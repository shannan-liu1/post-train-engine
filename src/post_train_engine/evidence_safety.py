"""Content contamination and verifier-independence gates."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

_FROZEN_FORBID = ConfigDict(frozen=True, extra="forbid")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


class ContentSeparationCertificate(BaseModel):
    model_config = _FROZEN_FORBID

    training_count: int = Field(..., ge=0)
    protected_count: int = Field(..., ge=0)
    ngram_size: int = Field(..., gt=0)
    max_allowed_jaccard: float = Field(..., ge=0.0, le=1.0)
    observed_max_jaccard: float = Field(..., ge=0.0, le=1.0)
    closest_training_index: int | None = Field(default=None, ge=0)
    closest_protected_index: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _observed_overlap_must_pass(self) -> ContentSeparationCertificate:
        if self.observed_max_jaccard > self.max_allowed_jaccard:
            raise ValueError(
                "training and protected evaluation content overlap exceeds threshold: "
                f"{self.observed_max_jaccard:.6f} > {self.max_allowed_jaccard:.6f}"
            )
        if (self.closest_training_index is None) != (
            self.closest_protected_index is None
        ):
            raise ValueError("closest content indices must be both present or both absent")
        if (
            self.closest_training_index is not None
            and self.closest_training_index >= self.training_count
        ):
            raise ValueError("closest training index exceeds the certified row count")
        if (
            self.closest_protected_index is not None
            and self.closest_protected_index >= self.protected_count
        ):
            raise ValueError("closest protected index exceeds the certified row count")
        return self


class VerifierSeparation(BaseModel):
    model_config = _FROZEN_FORBID

    verifier_kind: Literal["executable_ground_truth", "learned_proxy", "heuristic"]
    training_verifier_id: str = Field(..., min_length=1)
    promotion_verifier_id: str = Field(..., min_length=1)
    independent_audit_artifact_id: str | None = None

    @model_validator(mode="after")
    def _proxy_must_not_grade_itself(self) -> VerifierSeparation:
        if self.verifier_kind == "executable_ground_truth":
            return self
        if self.training_verifier_id == self.promotion_verifier_id:
            raise ValueError(
                "learned or heuristic training reward requires an independent promotion verifier"
            )
        if not self.independent_audit_artifact_id:
            raise ValueError(
                "learned or heuristic verifier requires independent_audit_artifact_id"
            )
        return self


def certify_content_separation(
    *,
    training_texts: Sequence[str],
    protected_texts: Sequence[str],
    ngram_size: int = 5,
    max_jaccard: float = 0.8,
) -> ContentSeparationCertificate:
    if ngram_size <= 0:
        raise ValueError("ngram_size must be positive")
    if not 0.0 <= max_jaccard <= 1.0:
        raise ValueError("max_jaccard must be between zero and one")
    maximum = 0.0
    closest: tuple[int, int] | None = None
    training_ngrams = [_ngrams(text, ngram_size) for text in training_texts]
    protected_ngrams = [_ngrams(text, ngram_size) for text in protected_texts]
    for training_index, left in enumerate(training_ngrams):
        for protected_index, right in enumerate(protected_ngrams):
            union = left | right
            similarity = len(left & right) / len(union) if union else 1.0
            if similarity > maximum:
                maximum = similarity
                closest = (training_index, protected_index)
    certificate = ContentSeparationCertificate(
        training_count=len(training_texts),
        protected_count=len(protected_texts),
        ngram_size=ngram_size,
        max_allowed_jaccard=max_jaccard,
        observed_max_jaccard=maximum,
        closest_training_index=None if closest is None else closest[0],
        closest_protected_index=None if closest is None else closest[1],
    )
    if maximum > max_jaccard:
        raise ValueError(
            "training and protected evaluation content overlap exceeds threshold: "
            f"{maximum:.6f} > {max_jaccard:.6f}"
        )
    return certificate


def _ngrams(text: str, size: int) -> set[tuple[str, ...]]:
    tokens = _TOKEN_RE.findall(text.lower())
    if len(tokens) < size:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1)}


__all__ = [
    "ContentSeparationCertificate",
    "VerifierSeparation",
    "certify_content_separation",
]
