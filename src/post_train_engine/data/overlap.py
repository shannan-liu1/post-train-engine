"""Fail-closed split overlap checks for promotion data."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Literal

from post_train_engine.tasks.schema import Example

SplitRole = Literal["probe", "train", "promotion", "canary", "regression"]


class DataLeakageError(ValueError):
    """Raised when training data overlaps protected evaluation data."""


@dataclass(frozen=True)
class EvalSplitManifest:
    suite_id: str
    suite_version: str
    role: SplitRole
    example_ids: tuple[str, ...]
    example_id_hash: str
    prompt_hashes: Mapping[str, str]
    source_dataset: str
    source_revision: str | None = None

    def __post_init__(self) -> None:
        if not self.suite_id:
            raise ValueError("suite_id must be non-empty")
        if not self.suite_version:
            raise ValueError("suite_version must be non-empty")
        if not self.example_id_hash:
            raise ValueError("example_id_hash must be non-empty")
        if not self.source_dataset:
            raise ValueError("source_dataset must be non-empty")
        if len(set(self.example_ids)) != len(self.example_ids):
            raise ValueError("promotion manifest contains duplicate example_ids")
        missing_hashes = set(self.example_ids) - set(self.prompt_hashes)
        if missing_hashes:
            raise ValueError(
                "promotion manifest missing prompt hash for example "
                f"{sorted(missing_hashes)[0]}"
            )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DataOverlapReport:
    train_example_count: int
    promotion_example_count: int
    train_and_promotion_overlap_count: int
    train_and_promotion_overlap_ids: tuple[str, ...]
    prompt_hash_overlap_count: int
    prompt_hash_overlap_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def certify_no_training_promotion_overlap(
    train_examples: Sequence[Example],
    promotion_manifest: EvalSplitManifest,
) -> DataOverlapReport:
    """Return overlap evidence or raise before protected eval data can train."""

    train_ids = {example.id for example in train_examples}
    promotion_ids = set(promotion_manifest.example_ids)
    overlapping_ids = tuple(sorted(train_ids & promotion_ids))

    train_prompt_hashes = {
        example.id: prompt_sha256(example.prompt)
        for example in train_examples
    }
    promotion_prompt_hashes_by_hash = {
        prompt_hash: example_id
        for example_id, prompt_hash in promotion_manifest.prompt_hashes.items()
    }
    overlapping_prompt_hashes = tuple(
        sorted(
            example_id
            for example_id, prompt_hash in train_prompt_hashes.items()
            if prompt_hash in promotion_prompt_hashes_by_hash
        )
    )

    report = DataOverlapReport(
        train_example_count=len(train_examples),
        promotion_example_count=len(promotion_manifest.example_ids),
        train_and_promotion_overlap_count=len(overlapping_ids),
        train_and_promotion_overlap_ids=overlapping_ids,
        prompt_hash_overlap_count=len(overlapping_prompt_hashes),
        prompt_hash_overlap_ids=overlapping_prompt_hashes,
    )
    if overlapping_ids:
        raise DataLeakageError(
            "training/promotion example id overlap: "
            f"{', '.join(overlapping_ids[:5])}"
        )
    if overlapping_prompt_hashes:
        raise DataLeakageError(
            "training/promotion prompt hash overlap: "
            f"{', '.join(overlapping_prompt_hashes[:5])}"
        )
    return report


def prompt_sha256(prompt: str) -> str:
    if not prompt:
        raise ValueError("prompt must be non-empty")
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"
