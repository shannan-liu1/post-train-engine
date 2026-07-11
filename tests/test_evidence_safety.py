from __future__ import annotations

import pytest

from post_train_engine.evidence_safety import (
    VerifierSeparation,
    certify_content_separation,
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
