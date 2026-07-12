from __future__ import annotations

import pytest

from post_train_engine.evidence_safety import (
    ContentSeparationCertificate,
    VerifierSeparation,
    certify_content_separation,
)


def test_content_separation_certificate_rejects_failed_threshold() -> None:
    with pytest.raises(ValueError, match="exceeds threshold"):
        ContentSeparationCertificate(
            training_count=1,
            protected_count=1,
            ngram_size=3,
            max_allowed_jaccard=0.8,
            observed_max_jaccard=0.9,
            closest_training_index=0,
            closest_protected_index=0,
        )


def test_content_separation_certificate_rejects_out_of_range_evidence() -> None:
    with pytest.raises(ValueError, match="closest training index"):
        ContentSeparationCertificate(
            training_count=1,
            protected_count=1,
            ngram_size=5,
            max_allowed_jaccard=0.8,
            observed_max_jaccard=0.1,
            closest_training_index=1,
            closest_protected_index=0,
        )


def test_content_separation_rejects_near_duplicate_protected_text() -> None:
    with pytest.raises(ValueError, match="content overlap"):
        certify_content_separation(
            training_texts=("Solve the problem and return the exact answer forty two",),
            protected_texts=("Solve the problem and return the exact answer forty two!",),
            ngram_size=3,
            max_jaccard=0.8,
        )


def test_learned_training_proxy_requires_independent_promotion_verifier() -> None:
    with pytest.raises(ValueError, match="independent promotion verifier"):
        VerifierSeparation(
            verifier_kind="learned_proxy",
            training_verifier_id="reward-model-v1",
            promotion_verifier_id="reward-model-v1",
        )

    shared_ground_truth = VerifierSeparation(
        verifier_kind="executable_ground_truth",
        training_verifier_id="exact-math-v1",
        promotion_verifier_id="exact-math-v1",
    )
    assert shared_ground_truth.promotion_verifier_id == "exact-math-v1"
