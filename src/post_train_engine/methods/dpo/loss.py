"""Direct Preference Optimization loss primitives."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from post_train_engine.core.logprobs import sequence_log_probs as token_log_probs
from post_train_engine.core.masking import IGNORE_INDEX


def sequence_log_probs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[torch.Tensor, torch.Tensor]:
    token_logps, response_mask = token_log_probs(
        logits,
        labels,
        ignore_index=ignore_index,
    )
    counts = response_mask.sum(dim=-1)
    if (counts == 0).any():
        bad_rows = torch.nonzero(counts == 0, as_tuple=False).flatten().tolist()
        raise ValueError(f"DPO row(s) have no response labels after shifting: {bad_rows}")
    return token_logps.sum(dim=-1), counts


def compute_dpo_loss(
    *,
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float,
    sample_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if beta <= 0.0:
        raise ValueError("beta must be positive")

    policy_margin = policy_chosen_logps - policy_rejected_logps
    reference_margin = ref_chosen_logps - ref_rejected_logps
    logits = beta * (policy_margin - reference_margin)
    per_pair_loss = -F.logsigmoid(logits)

    weights: torch.Tensor | None = None
    if sample_weights is not None:
        if sample_weights.shape != per_pair_loss.shape:
            raise ValueError("sample_weights must match the DPO batch shape")
        weights = sample_weights.to(device=per_pair_loss.device, dtype=per_pair_loss.dtype)
        if (weights <= 0).any():
            raise ValueError("sample_weights must be strictly positive")
        loss = (per_pair_loss * weights).sum() / weights.sum()
    else:
        loss = per_pair_loss.mean()

    preference_hits = (logits.detach() > 0).float()
    if weights is None:
        preference_accuracy = preference_hits.mean()
    else:
        preference_accuracy = (preference_hits * weights.detach()).sum() / weights.sum()
    return loss, {
        "loss": loss.detach(),
        "preference_accuracy": preference_accuracy,
        "policy_margin": policy_margin.detach().mean(),
        "reference_margin": reference_margin.detach().mean(),
        "reward_margin": logits.detach().mean(),
    }


def dpo_loss_from_logps(
    *,
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    chosen_counts: torch.Tensor,
    rejected_counts: torch.Tensor,
    beta: float,
    sample_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    loss, metrics = compute_dpo_loss(
        policy_chosen_logps=policy_chosen_logps,
        policy_rejected_logps=policy_rejected_logps,
        ref_chosen_logps=ref_chosen_logps,
        ref_rejected_logps=ref_rejected_logps,
        beta=beta,
        sample_weights=sample_weights,
    )
    metrics["chosen_response_tokens"] = chosen_counts.detach().float().mean()
    metrics["rejected_response_tokens"] = rejected_counts.detach().float().mean()
    return loss, metrics


def dpo_loss_from_policy_logits(
    *,
    policy_chosen_logits: torch.Tensor,
    policy_rejected_logits: torch.Tensor,
    chosen_labels: torch.Tensor,
    rejected_labels: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float,
    sample_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    policy_chosen_logps, chosen_counts = sequence_log_probs(policy_chosen_logits, chosen_labels)
    policy_rejected_logps, rejected_counts = sequence_log_probs(
        policy_rejected_logits,
        rejected_labels,
    )
    return dpo_loss_from_logps(
        policy_chosen_logps=policy_chosen_logps,
        policy_rejected_logps=policy_rejected_logps,
        ref_chosen_logps=ref_chosen_logps,
        ref_rejected_logps=ref_rejected_logps,
        chosen_counts=chosen_counts,
        rejected_counts=rejected_counts,
        beta=beta,
        sample_weights=sample_weights,
    )
