"""Canonical evaluation-role boundaries."""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, model_validator


class EvaluationRoles(BaseModel):
    """Immutable example assignments for selection and protected evaluation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    selection_example_ids: tuple[str, ...] = ()
    diagnostic_example_ids: tuple[str, ...] = ()
    promotion_example_ids: tuple[str, ...] = ()
    canary_example_ids: tuple[str, ...] = ()
    unseen_example_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _roles_must_be_valid(self) -> EvaluationRoles:
        named = {
            "selection": self.selection_example_ids,
            "promotion": self.promotion_example_ids,
            "canary": self.canary_example_ids,
            "unseen": self.unseen_example_ids,
        }
        for name, values in named.items():
            if len(values) != len(set(values)):
                raise ValueError(f"{name} role contains duplicate example IDs")
        names = tuple(named)
        for index, left_name in enumerate(names):
            for right_name in names[index + 1 :]:
                overlap = set(named[left_name]) & set(named[right_name])
                if overlap:
                    first = sorted(overlap)[0]
                    raise ValueError(
                        f"{left_name} and {right_name} roles overlap: {first}"
                    )
        return self

    @property
    def protected_example_ids(self) -> frozenset[str]:
        return frozenset(
            (
                *self.selection_example_ids,
                *self.promotion_example_ids,
                *self.canary_example_ids,
                *self.unseen_example_ids,
            )
        )

    def require_training_eligible(self, example_ids: Iterable[str]) -> None:
        overlap = self.protected_example_ids & set(example_ids)
        if overlap:
            first = sorted(overlap)[0]
            raise ValueError(
                f"training data contains protected evaluation examples: {first}"
            )


__all__ = ["EvaluationRoles"]
