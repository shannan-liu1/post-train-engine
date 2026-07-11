"""PPO trajectory primitives with critic/value fields."""

from __future__ import annotations

import math
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator

_FROZEN_FORBID = ConfigDict(frozen=True, extra="forbid")


class PPOTransition(BaseModel):
    """One PPO transition tied back to a preserved trace."""

    model_config = _FROZEN_FORBID

    trace_id: str = Field(..., min_length=1)
    reward: float
    value: float
    logprob: float
    done: bool
    action_token_count: int = Field(default=0, ge=0)
    observation_token_count: int = Field(default=0, ge=0)

    @field_validator("reward", "value", "logprob")
    @classmethod
    def _finite(cls, value: float, info: object) -> float:
        if type(value) is bool or not isinstance(value, int | float) or not math.isfinite(float(value)):
            raise ValueError(f"{getattr(info, 'field_name', 'value')} must be finite")
        return float(value)


class GAEEstimate(BaseModel):
    model_config = _FROZEN_FORBID

    advantages: tuple[float, ...]
    returns: tuple[float, ...]

    @field_validator("advantages", "returns", mode="before")
    @classmethod
    def _finite_sequence(cls, values: Sequence[float]) -> tuple[float, ...]:
        out = tuple(float(value) for value in values)
        if any(not math.isfinite(value) for value in out):
            raise ValueError("GAE values must be finite")
        return out


def compute_gae(
    transitions: Sequence[PPOTransition],
    *,
    gamma: float = 0.99,
    lam: float = 0.95,
    bootstrap_value: float = 0.0,
) -> GAEEstimate:
    """Compute generalized advantage estimates for one trajectory."""

    if not transitions:
        raise ValueError("compute_gae requires at least one transition")
    for name, value in {
        "gamma": gamma,
        "lam": lam,
        "bootstrap_value": bootstrap_value,
    }.items():
        if type(value) is bool or not isinstance(value, int | float) or not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be between 0 and 1")
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lam must be between 0 and 1")

    advantages = [0.0 for _ in transitions]
    last_gae = 0.0
    next_value = float(bootstrap_value)
    for idx in range(len(transitions) - 1, -1, -1):
        transition = transitions[idx]
        nonterminal = 0.0 if transition.done else 1.0
        delta = transition.reward + gamma * next_value * nonterminal - transition.value
        last_gae = delta + gamma * lam * nonterminal * last_gae
        advantages[idx] = last_gae
        next_value = transition.value
    returns = [advantage + transition.value for advantage, transition in zip(advantages, transitions, strict=True)]
    return GAEEstimate(advantages=tuple(advantages), returns=tuple(returns))


__all__ = ["GAEEstimate", "PPOTransition", "compute_gae"]
