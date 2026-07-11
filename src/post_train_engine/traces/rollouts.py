"""Rollout grouping primitives."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from post_train_engine.traces.schema import TraceRecord

_FROZEN_FORBID = ConfigDict(frozen=True, extra="forbid")


class RolloutGroup(BaseModel):
    """A typed group of sibling rollout traces for one example."""

    model_config = _FROZEN_FORBID

    group_id: str = Field(..., min_length=1)
    trace_ids: tuple[str, ...] = Field(..., min_length=1)
    group_size: int = Field(..., ge=1)
    rewards: tuple[float, ...] = ()
    reward_variance: float | None = Field(default=None, ge=0.0)
    degenerate_group: bool = False

    @field_validator("group_size", "reward_variance", mode="before")
    @classmethod
    def _reject_boolean_numbers(cls, value: Any, info: Any) -> Any:
        if value is not None and type(value) is bool:
            raise ValueError(f"{info.field_name} must not be a boolean")
        if info.field_name == "reward_variance" and value is not None:
            if not isinstance(value, int | float) or not math.isfinite(float(value)):
                raise ValueError("reward_variance must be finite")
        return value

    @field_validator("rewards", mode="before")
    @classmethod
    def _rewards_must_be_finite(cls, value: Any) -> Any:
        if value is None:
            return ()
        rewards = tuple(value)
        for reward in rewards:
            if type(reward) is bool or not isinstance(reward, int | float):
                raise ValueError("rewards must be finite numbers")
            if not math.isfinite(float(reward)):
                raise ValueError("rewards must be finite numbers")
        return rewards

    @model_validator(mode="after")
    def _validate_group_contract(self) -> RolloutGroup:
        if len(set(self.trace_ids)) != len(self.trace_ids):
            raise ValueError("trace_ids must be unique")
        if self.group_size != len(self.trace_ids):
            raise ValueError("group_size must match trace_ids length")
        if self.rewards:
            if len(self.rewards) != self.group_size:
                raise ValueError("rewards length must match group_size")
            expected_degenerate = all(
                reward == self.rewards[0] for reward in self.rewards
            )
            if self.degenerate_group != expected_degenerate:
                raise ValueError("degenerate_group must match reward equality")
        return self


def build_rollout_group(
    *,
    group_id: str,
    traces: Iterable[TraceRecord] | None = None,
    trace_ids: Iterable[str] | None = None,
    rewards: Iterable[float] | None = None,
) -> RolloutGroup:
    if traces is not None and trace_ids is not None:
        raise ValueError("provide traces or trace_ids, not both")
    trace_values = tuple(traces) if traces is not None else None
    if trace_values is not None:
        mismatched = [
            trace.trace_id
            for trace in trace_values
            if trace.rollout_group_id != group_id
        ]
        if mismatched:
            raise ValueError(
                "trace rollout_group_id does not match group_id: " + mismatched[0]
            )
    ids = (
        tuple(trace.trace_id for trace in trace_values)
        if trace_values is not None
        else tuple(() if trace_ids is None else trace_ids)
    )
    reward_values = tuple(() if rewards is None else rewards)
    reward_variance = _population_variance(reward_values) if reward_values else None
    degenerate_group = bool(
        reward_values and all(reward == reward_values[0] for reward in reward_values)
    )
    return RolloutGroup(
        group_id=group_id,
        trace_ids=ids,
        group_size=len(ids),
        rewards=reward_values,
        reward_variance=reward_variance,
        degenerate_group=degenerate_group,
    )


def _population_variance(values: tuple[float, ...]) -> float:
    mean = sum(float(value) for value in values) / len(values)
    return sum((float(value) - mean) ** 2 for value in values) / len(values)


__all__ = ["RolloutGroup", "build_rollout_group"]
