"""Promotion suite accounting and rotation policy."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Literal

RotationAction = Literal["keep", "rotate", "fail_closed"]


@dataclass(frozen=True)
class PromotionSuiteState:
    suite_id: str
    suite_version: str
    example_count: int
    example_id_hash: str
    prompt_hash: str | None = None
    num_times_suite_tested: int = 0
    num_candidates_evaluated: int = 0
    accepted_promotion_count: int = 0
    train_and_promotion_overlap_count: int = 0
    train_and_promotion_overlap_ids: tuple[str, ...] = ()
    canary_failure_count: int = 0
    slice_distribution: Mapping[str, int] = field(default_factory=dict)
    evaluated_candidate_ids: tuple[str, ...] = ()
    retired: bool = False
    retire_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.suite_id:
            raise ValueError("suite_id must be non-empty")
        if not self.suite_version:
            raise ValueError("suite_version must be non-empty")
        if self.example_count < 0:
            raise ValueError("example_count must be non-negative")
        if not self.example_id_hash:
            raise ValueError("example_id_hash must be non-empty")
        for name in (
            "num_times_suite_tested",
            "num_candidates_evaluated",
            "accepted_promotion_count",
            "train_and_promotion_overlap_count",
            "canary_failure_count",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if len(set(self.evaluated_candidate_ids)) != len(self.evaluated_candidate_ids):
            raise ValueError("evaluated_candidate_ids must be unique")
        if self.train_and_promotion_overlap_count != len(
            self.train_and_promotion_overlap_ids
        ):
            raise ValueError(
                "train_and_promotion_overlap_count must match overlap id count"
            )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SuiteRotationPolicy:
    max_num_times_suite_tested: int = 50
    max_num_candidates_evaluated: int = 40
    max_accepted_promotion_count: int = 3
    max_train_and_promotion_overlap_count: int = 0
    require_bridge_eval_on_rotation: bool = True


@dataclass(frozen=True)
class SuiteRotationRecommendation:
    action: RotationAction
    reason: str
    require_bridge_eval: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def record_suite_test(state: PromotionSuiteState) -> PromotionSuiteState:
    return _replace(
        state,
        num_times_suite_tested=state.num_times_suite_tested + 1,
    )


def record_candidate_evaluated(
    state: PromotionSuiteState,
    candidate_id: str,
) -> PromotionSuiteState:
    if not candidate_id:
        raise ValueError("candidate_id must be non-empty")
    if candidate_id in state.evaluated_candidate_ids:
        return state
    evaluated = tuple(sorted((*state.evaluated_candidate_ids, candidate_id)))
    return _replace(
        state,
        evaluated_candidate_ids=evaluated,
        num_candidates_evaluated=len(evaluated),
    )


def record_accepted_promotion(state: PromotionSuiteState) -> PromotionSuiteState:
    return _replace(
        state,
        accepted_promotion_count=state.accepted_promotion_count + 1,
    )


def with_overlap_counts(
    state: PromotionSuiteState,
    *,
    train_and_promotion_overlap_count: int,
    train_and_promotion_overlap_ids: tuple[str, ...] = (),
) -> PromotionSuiteState:
    return _replace(
        state,
        train_and_promotion_overlap_count=train_and_promotion_overlap_count,
        train_and_promotion_overlap_ids=train_and_promotion_overlap_ids,
    )


def rotation_recommendation(
    state: PromotionSuiteState,
    policy: SuiteRotationPolicy,
) -> SuiteRotationRecommendation:
    if (
        state.train_and_promotion_overlap_count
        > policy.max_train_and_promotion_overlap_count
    ):
        return SuiteRotationRecommendation(
            action="fail_closed",
            reason="train_and_promotion_overlap_count > 0",
            require_bridge_eval=False,
        )
    if state.num_times_suite_tested > policy.max_num_times_suite_tested:
        return SuiteRotationRecommendation(
            action="rotate",
            reason=(
                "num_times_suite_tested exceeded "
                f"{policy.max_num_times_suite_tested}"
            ),
            require_bridge_eval=policy.require_bridge_eval_on_rotation,
        )
    if state.num_candidates_evaluated > policy.max_num_candidates_evaluated:
        return SuiteRotationRecommendation(
            action="rotate",
            reason=(
                "num_candidates_evaluated exceeded "
                f"{policy.max_num_candidates_evaluated}"
            ),
            require_bridge_eval=policy.require_bridge_eval_on_rotation,
        )
    if state.accepted_promotion_count > policy.max_accepted_promotion_count:
        return SuiteRotationRecommendation(
            action="rotate",
            reason=(
                "accepted_promotion_count exceeded "
                f"{policy.max_accepted_promotion_count}"
            ),
            require_bridge_eval=policy.require_bridge_eval_on_rotation,
        )
    return SuiteRotationRecommendation(
        action="keep",
        reason="suite within exposure limits",
        require_bridge_eval=False,
    )


def _replace(state: PromotionSuiteState, **updates: object) -> PromotionSuiteState:
    body = state.to_dict()
    body.update(updates)
    return PromotionSuiteState(**body)
