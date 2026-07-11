"""Supervised Fine-Tuning loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from post_train_engine.core.masking import IGNORE_INDEX


def compute_masked_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_labels = shift_labels.view(-1)
    valid_labels = flat_labels != IGNORE_INDEX
    if not valid_labels.any():
        return flat_logits.float().sum() * 0.0
    return F.cross_entropy(
        flat_logits.float(),
        flat_labels,
        ignore_index=IGNORE_INDEX,
        reduction="mean",
    )
