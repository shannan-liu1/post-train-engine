from __future__ import annotations

import pytest

from post_train_engine.data.overlap import (
    DataLeakageError,
    EvalSplitManifest,
    certify_no_training_promotion_overlap,
    prompt_sha256,
)
from post_train_engine.tasks.schema import Example


def test_training_promotion_overlap_certification_accepts_disjoint_data() -> None:
    report = certify_no_training_promotion_overlap(
        [
            Example(id="train-1", source="toy", prompt="train prompt"),
            Example(id="train-2", source="toy", prompt="another train prompt"),
        ],
        EvalSplitManifest(
            suite_id="toy-promotion",
            suite_version="2026-06-a",
            role="promotion",
            example_ids=("promotion-1",),
            example_id_hash="sha256:promotion-ids",
            prompt_hashes={"promotion-1": prompt_sha256("heldout prompt")},
            source_dataset="toy",
            source_revision="rev1",
        ),
    )

    assert report.train_example_count == 2
    assert report.promotion_example_count == 1
    assert report.train_and_promotion_overlap_count == 0
    assert report.prompt_hash_overlap_count == 0


def test_training_promotion_overlap_certification_rejects_example_id_leakage() -> None:
    with pytest.raises(DataLeakageError, match="training/promotion example id overlap"):
        certify_no_training_promotion_overlap(
            [Example(id="leaked", source="toy", prompt="train prompt")],
            EvalSplitManifest(
                suite_id="toy-promotion",
                suite_version="2026-06-a",
                role="promotion",
                example_ids=("leaked",),
                example_id_hash="sha256:promotion-ids",
                prompt_hashes={"leaked": prompt_sha256("heldout prompt")},
                source_dataset="toy",
                source_revision="rev1",
            ),
        )


def test_training_promotion_overlap_certification_rejects_prompt_hash_leakage() -> None:
    with pytest.raises(DataLeakageError, match="training/promotion prompt hash overlap"):
        certify_no_training_promotion_overlap(
            [Example(id="train-1", source="toy", prompt="same prompt")],
            EvalSplitManifest(
                suite_id="toy-promotion",
                suite_version="2026-06-a",
                role="promotion",
                example_ids=("promotion-1",),
                example_id_hash="sha256:promotion-ids",
                prompt_hashes={"promotion-1": prompt_sha256("same prompt")},
                source_dataset="toy",
                source_revision="rev1",
            ),
        )
