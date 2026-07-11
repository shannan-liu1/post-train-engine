"""GSM8K verifier reward."""

from __future__ import annotations

from dataclasses import dataclass

from post_train_engine.tasks.gsm8k import parse_model_answer, verify_answer


@dataclass(frozen=True)
class GSM8KRewardConfig:
    parse_bonus: float = 0.02
    length_penalty_weight: float = 0.05
    max_new_tokens: int = 512
    use_lenient_parse_for_probe: bool = True
    use_strict_parse_for_reward: bool = True

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.parse_bonus < 0.0:
            raise ValueError("parse_bonus must be non-negative")
        if self.length_penalty_weight < 0.0:
            raise ValueError("length_penalty_weight must be non-negative")


@dataclass(frozen=True)
class GSM8KReward:
    reward: float
    task_reward: float
    parse_bonus: float
    length_penalty: float
    parse_ok: bool
    correct: bool
    parsed_answer: str | None
    verifier_error: str | None


def compute_gsm8k_reward(
    completion: str,
    gold_answer: str,
    rho_q: float,
    completion_tokens: int,
    config: GSM8KRewardConfig,
) -> GSM8KReward:
    if completion_tokens < 0:
        raise ValueError("completion_tokens must be non-negative")
    if not 0.0 <= rho_q <= 1.0:
        raise ValueError("rho_q must be between 0 and 1")

    mode = "strict" if config.use_strict_parse_for_reward else "lenient"
    parsed = parse_model_answer(completion, mode=mode)
    verification = (
        verify_answer(parsed.answer, gold_answer)
        if parsed.parse_ok and parsed.answer is not None
        else None
    )
    correct = bool(verification and verification.correct)
    task_reward = 1.0 if correct else 0.0
    parse_bonus = config.parse_bonus if parsed.parse_ok else 0.0
    length_penalty = (
        config.length_penalty_weight
        * rho_q
        * min(completion_tokens, config.max_new_tokens)
        / config.max_new_tokens
    )
    reward = task_reward + parse_bonus - length_penalty
    return GSM8KReward(
        reward=reward,
        task_reward=task_reward,
        parse_bonus=parse_bonus,
        length_penalty=length_penalty,
        parse_ok=parsed.parse_ok,
        correct=correct,
        parsed_answer=parsed.answer,
        verifier_error=parsed.error if verification is None else verification.error,
    )
