from __future__ import annotations

from post_train_engine.rewards.gsm8k import GSM8KRewardConfig, compute_gsm8k_reward


def test_reward_correctness_dominates_parse_bonus_and_length_penalty() -> None:
    cfg = GSM8KRewardConfig(parse_bonus=0.02, length_penalty_weight=0.05, max_new_tokens=10)

    worst_correct = compute_gsm8k_reward(
        "<answer>18</answer>",
        "18",
        rho_q=1.0,
        completion_tokens=10,
        config=cfg,
    )
    best_incorrect = compute_gsm8k_reward(
        "<answer>19</answer>",
        "18",
        rho_q=0.0,
        completion_tokens=0,
        config=cfg,
    )

    assert worst_correct.correct is True
    assert best_incorrect.correct is False
    assert worst_correct.reward > best_incorrect.reward
