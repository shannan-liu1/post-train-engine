from __future__ import annotations

import pytest

from post_train_engine.evaluation_roles import EvaluationRoles


def test_selection_and_promotion_roles_must_be_disjoint() -> None:
    with pytest.raises(ValueError, match="selection and promotion roles overlap"):
        EvaluationRoles(
            selection_example_ids=("select-1", "shared"),
            promotion_example_ids=("shared", "promote-1"),
            canary_example_ids=(),
        )


def test_diagnostic_and_promotion_roles_must_be_disjoint() -> None:
    with pytest.raises(ValueError, match="diagnostic and promotion roles overlap"):
        EvaluationRoles(
            diagnostic_example_ids=("diagnostic-1", "shared"),
            promotion_example_ids=("shared", "promote-1"),
        )


def test_training_roles_cannot_contain_protected_examples() -> None:
    roles = EvaluationRoles(
        selection_example_ids=("select-1",),
        promotion_example_ids=("promote-1",),
        canary_example_ids=("canary-1",),
    )

    with pytest.raises(ValueError, match="protected evaluation examples"):
        roles.require_training_eligible(("train-1", "canary-1"))


def test_training_roles_cannot_contain_diagnostic_examples() -> None:
    roles = EvaluationRoles(diagnostic_example_ids=("diagnostic-1",))

    with pytest.raises(ValueError, match="protected evaluation examples"):
        roles.require_training_eligible(("train-1", "diagnostic-1"))
