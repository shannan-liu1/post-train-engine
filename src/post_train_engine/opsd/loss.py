"""OPSD loss primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class OPSDLossConfig:
    divergence: Literal["forward_kl"] = "forward_kl"
    teacher_update_mode: Literal["frozen", "current", "ema"] = "frozen"
    temperature: float = 1.0
    pointwise_kl_clip: float | None = 10.0
    loss_on_rollout_tokens_only: bool = True

    def __post_init__(self) -> None:
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if self.pointwise_kl_clip is not None and self.pointwise_kl_clip <= 0.0:
            raise ValueError("pointwise_kl_clip must be positive when provided")


def opsd_forward_kl_loss(
    student_logits: Any,
    teacher_logits: Any,
    rollout_token_mask: Any,
    cfg: OPSDLossConfig,
) -> Any:
    """Forward KL(p_teacher || p_student) over student rollout positions only."""

    import torch
    import torch.nn.functional as F

    if cfg.divergence != "forward_kl":
        raise ValueError(f"unsupported OPSD divergence: {cfg.divergence}")
    if not cfg.loss_on_rollout_tokens_only:
        raise ValueError("OPSD loss currently supports rollout-token loss only")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student_logits and teacher_logits must have matching shapes")
    if student_logits.ndim != 3:
        raise ValueError("logits must have shape [batch, seq, vocab]")
    if rollout_token_mask.shape != student_logits.shape[:2]:
        raise ValueError("rollout_token_mask must have shape [batch, seq]")
    if not rollout_token_mask.any():
        raise ValueError("rollout_token_mask selects no rollout tokens")

    mask = rollout_token_mask.to(dtype=torch.bool)
    student_selected = student_logits[mask].float() / cfg.temperature
    teacher_selected = teacher_logits.detach()[mask].float() / cfg.temperature
    if student_selected.shape[0] != teacher_selected.shape[0]:
        raise ValueError("student and teacher rollout-token lengths differ")

    teacher_probs = F.softmax(teacher_selected, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_selected, dim=-1)
    student_log_probs = F.log_softmax(student_selected, dim=-1)
    pointwise_kl = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)
    if cfg.pointwise_kl_clip is not None:
        pointwise_kl = pointwise_kl.clamp(max=cfg.pointwise_kl_clip)
    return pointwise_kl.mean()
