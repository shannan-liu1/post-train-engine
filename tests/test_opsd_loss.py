from __future__ import annotations

import pytest

from post_train_engine.opsd.loss import OPSDLossConfig, opsd_forward_kl_loss


def test_opsd_loss_masks_privileged_tokens() -> None:
    torch = pytest.importorskip("torch")
    cfg = OPSDLossConfig()
    student = torch.zeros(1, 3, 4)
    teacher = torch.zeros(1, 3, 4)
    mask = torch.tensor([[False, False, True]])

    baseline = opsd_forward_kl_loss(student, teacher, mask, cfg)
    changed_prompt_teacher = teacher.clone()
    changed_prompt_teacher[:, :2, :] = 100.0
    changed = opsd_forward_kl_loss(student, changed_prompt_teacher, mask, cfg)

    assert float(changed) == pytest.approx(float(baseline))


def test_opsd_loss_detaches_teacher_logits() -> None:
    torch = pytest.importorskip("torch")
    cfg = OPSDLossConfig()
    student = torch.randn(1, 2, 4, requires_grad=True)
    teacher = torch.randn(1, 2, 4, requires_grad=True)
    mask = torch.tensor([[False, True]])

    loss = opsd_forward_kl_loss(student, teacher, mask, cfg)
    loss.backward()

    assert student.grad is not None
    assert teacher.grad is None


def test_opsd_loss_fails_closed_if_rollout_only_masking_is_disabled() -> None:
    torch = pytest.importorskip("torch")
    cfg = OPSDLossConfig(loss_on_rollout_tokens_only=False)
    logits = torch.zeros(1, 1, 2)
    mask = torch.tensor([[True]])

    with pytest.raises(ValueError, match="rollout-token loss only"):
        opsd_forward_kl_loss(logits, logits, mask, cfg)
