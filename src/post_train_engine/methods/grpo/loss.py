"""Group Relative Policy Optimization primitives."""

from __future__ import annotations

from typing import Any

import torch

from post_train_engine.core.logprobs import sequence_log_probs
from post_train_engine.core.masking import IGNORE_INDEX


def group_relative_advantages(
    rewards: torch.Tensor,
    *,
    eps: float = 1e-8,
    normalize: bool = True,
) -> torch.Tensor:
    if rewards.ndim != 2:
        raise ValueError(f"rewards must have shape (prompts, group); got {tuple(rewards.shape)}")
    if rewards.size(1) < 2:
        raise ValueError("GRPO requires at least two completions per prompt group")
    centered = rewards - rewards.mean(dim=-1, keepdim=True)
    if not normalize:
        return centered
    std = rewards.std(dim=-1, unbiased=False, keepdim=True)
    return torch.where(std > eps, centered / std.clamp_min(eps), torch.zeros_like(centered))


def token_log_probs_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[torch.Tensor, torch.Tensor]:
    token_logps, response_mask = sequence_log_probs(
        logits,
        labels,
        ignore_index=ignore_index,
    )
    counts = response_mask.sum(dim=-1)
    if (counts == 0).any():
        bad_rows = torch.nonzero(counts == 0, as_tuple=False).flatten().tolist()
        raise ValueError(f"GRPO row(s) have no response labels after shifting: {bad_rows}")
    return token_logps, response_mask


def grpo_token_loss(
    *,
    policy_logps: torch.Tensor,
    old_logps: torch.Tensor,
    ref_logps: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    beta: float,
    clip_range: float | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if beta < 0.0:
        raise ValueError("beta must be non-negative")
    if clip_range is not None and clip_range <= 0.0:
        raise ValueError("clip_range must be positive when provided")
    if policy_logps.shape != old_logps.shape or policy_logps.shape != ref_logps.shape:
        raise ValueError("policy, old, and reference log-prob tensors must match")
    if response_mask.shape != policy_logps.shape:
        raise ValueError("response_mask must match log-prob tensor shape")
    if advantages.shape != (policy_logps.size(0),):
        raise ValueError(f"advantages must have shape ({policy_logps.size(0)},)")
    if not response_mask.any():
        raise ValueError("response_mask contains no response tokens")

    ratio = torch.exp(policy_logps - old_logps)
    token_advantages = advantages.unsqueeze(-1)
    objective = ratio * token_advantages
    if clip_range is not None:
        clipped_ratio = ratio.clamp(1.0 - clip_range, 1.0 + clip_range)
        objective = torch.minimum(objective, clipped_ratio * token_advantages)

    log_ratio_ref_policy = ref_logps - policy_logps
    kl = torch.exp(log_ratio_ref_policy) - log_ratio_ref_policy - 1.0
    token_loss = -(objective - beta * kl)
    mask = response_mask.to(token_loss.dtype)
    loss = (token_loss * mask).sum() / mask.sum()
    return loss, {
        "loss": loss.detach(),
        "mean_reward_objective": ((objective.detach() * mask).sum() / mask.sum()),
        "mean_kl": ((kl.detach() * mask).sum() / mask.sum()),
        "mean_advantage": advantages.detach().float().mean(),
        "mean_ratio": ((ratio.detach() * mask).sum() / mask.sum()),
    }


def grpo_loss_from_logits(
    *,
    policy_logits: torch.Tensor,
    labels: torch.Tensor,
    old_logps: torch.Tensor,
    ref_logps: torch.Tensor,
    advantages: torch.Tensor,
    beta: float,
    clip_range: float | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    policy_logps, response_mask = token_log_probs_from_logits(policy_logits, labels)
    return grpo_token_loss(
        policy_logps=policy_logps,
        old_logps=old_logps,
        ref_logps=ref_logps,
        advantages=advantages,
        response_mask=response_mask,
        beta=beta,
        clip_range=clip_range,
    )
