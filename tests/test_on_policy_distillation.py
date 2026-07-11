from __future__ import annotations

from pathlib import Path

import pytest

from post_train_engine.methods.opd import (
    DistillationOODGuard,
    DistillationTeacher,
    OnPolicyDistillationRow,
    TeacherSignal,
    build_mopd_peer_context,
    build_multi_teacher_view,
    multi_teacher_kl_loss,
)
from post_train_engine.traces import TraceRecord, stable_prompt_hash


def test_distillation_ood_guard_requires_measured_transfer() -> None:
    with pytest.raises(ValueError, match="OOD teacher advantage"):
        DistillationOODGuard(
            source_domain_ids=("math",),
            target_domain_id="code",
            measured_teacher_advantage=-0.1,
            min_teacher_advantage=0.0,
            evaluation_artifact_id="eval-1",
        )


def test_multi_teacher_view_records_teacher_weights_peer_context_and_training_view(
    tmp_path: Path,
) -> None:
    rows = [
        OnPolicyDistillationRow(
            row_id="row-1",
            target_trace_id="trace-target",
            run_id="run-1",
            task_id="gsm8k",
            example_id="ex-1",
            split_role="train",
            prompt="What is 2+2?",
            student_completion="bad <answer>5</answer>",
            teacher_signals=(
                TeacherSignal(
                    teacher_id="teacher-large",
                    teacher_kind="external",
                    weight=0.7,
                    supervision_kind="token_distribution",
                    visibility="none",
                    score=0.8,
                ),
                TeacherSignal(
                    teacher_id="teacher-peer",
                    teacher_kind="peer_conditioned",
                    weight=0.3,
                    supervision_kind="critique",
                    visibility="verifier_feedback",
                    score=0.6,
                ),
            ),
            peer_context={"success_trace_ids": ("trace-good",), "failure_trace_ids": ("trace-bad",)},
        )
    ]
    data_path = tmp_path / "views" / "mopd.jsonl"

    view = build_multi_teacher_view(
        view_id="run-1:mopd",
        run_id="run-1",
        task_id="gsm8k",
        rows=rows,
        data_path=data_path,
        ood_guard=DistillationOODGuard(
            source_domain_ids=("gsm8k",),
            target_domain_id="gsm8k-heldout-template",
            measured_teacher_advantage=0.1,
            min_teacher_advantage=0.0,
            evaluation_artifact_id="teacher-ood-eval-1",
        ),
    )

    assert data_path.is_file()
    assert view.view_type == "multi_teacher_opd"
    assert view.method_compatibility == ("opd", "multi_teacher_opd")
    assert view.source_trace_ids == ("trace-target",)
    assert view.metadata["teacher_ids"] == ["teacher-large", "teacher-peer"]
    assert view.metadata["aggregation"] == "weighted_probability_mixture"
    assert view.metadata["ood_guard"]["evaluation_artifact_id"] == "teacher-ood-eval-1"


def test_on_policy_distillation_row_fails_closed_on_bad_sources() -> None:
    signal = TeacherSignal(
        teacher_id="teacher",
        teacher_kind="external",
        weight=1.0,
        supervision_kind="token_distribution",
        visibility="none",
    )

    with pytest.raises(ValueError, match="protected evaluation split"):
        OnPolicyDistillationRow(
            row_id="row",
            target_trace_id="trace",
            run_id="run",
            task_id="gsm8k",
            example_id="ex",
            split_role="promotion",
            prompt="p",
            student_completion="c",
            teacher_signals=(signal,),
        )

    with pytest.raises(ValueError, match="privileged"):
        OnPolicyDistillationRow(
            row_id="row",
            target_trace_id="trace",
            run_id="run",
            task_id="gsm8k",
            example_id="ex",
            split_role="train",
            prompt="p",
            student_completion="c",
            teacher_signals=(
                TeacherSignal(
                    teacher_id="self",
                    teacher_kind="self_privileged",
                    weight=1.0,
                    supervision_kind="token_distribution",
                    visibility="none",
                ),
            ),
        )


def test_mopd_peer_context_partitions_successes_and_failures_without_target() -> None:
    traces = [
        _trace("target", reward=0.0),
        _trace("success-1", reward=1.0),
        _trace("success-2", reward=0.9),
        _trace("failure-1", reward=0.1),
    ]

    context = build_mopd_peer_context(
        target_trace_id="target",
        traces=traces,
        success_threshold=0.5,
        max_successes=2,
        max_failures=1,
    )

    assert context["success_trace_ids"] == ("success-1", "success-2")
    assert context["failure_trace_ids"] == ("failure-1",)
    assert "target" not in context["success_trace_ids"]
    assert "target" not in context["failure_trace_ids"]


def test_distillation_teacher_rejects_duplicate_ids_and_bad_weights() -> None:
    with pytest.raises(ValueError, match="weight"):
        TeacherSignal(
            teacher_id="t",
            teacher_kind="external",
            weight=float("nan"),
            supervision_kind="token_distribution",
            visibility="none",
        )

    with pytest.raises(ValueError, match="unique"):
        DistillationTeacher.validate_unique(
            (
                DistillationTeacher(teacher_id="t", teacher_kind="external", model_id="a"),
                DistillationTeacher(teacher_id="t", teacher_kind="external", model_id="b"),
            )
        )


def test_multi_teacher_kl_loss_uses_weighted_teacher_mixture_and_masks_rollout_tokens() -> None:
    torch = pytest.importorskip("torch")
    student_logits = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]], requires_grad=True)
    teacher_logits = torch.tensor(
        [
            [[[0.0, 2.0], [2.0, 0.0]]],
            [[[2.0, 0.0], [2.0, 0.0]]],
        ],
        requires_grad=True,
    )
    weights = torch.tensor([0.25, 0.75])
    mask = torch.tensor([[False, True]])

    loss = multi_teacher_kl_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        teacher_weights=weights,
        rollout_token_mask=mask,
    )
    loss.backward()

    assert float(loss.detach()) > 0.0
    assert student_logits.grad is not None
    assert teacher_logits.grad is None

    with pytest.raises(ValueError, match="selects no rollout tokens"):
        multi_teacher_kl_loss(
            student_logits=student_logits.detach(),
            teacher_logits=teacher_logits.detach(),
            teacher_weights=weights,
            rollout_token_mask=torch.tensor([[False, False]]),
        )


def _trace(trace_id: str, *, reward: float) -> TraceRecord:
    return TraceRecord(
        trace_id=trace_id,
        run_id="run",
        task_id="gsm8k",
        example_id="ex-1",
        split_role="train",
        prompt_hash=stable_prompt_hash("p"),
        source_checkpoint="student",
        policy_version="student-step-1",
        policy_step=1,
        policy_step_evidence="exact",
        rollout_group_id="group-1",
        generation_backend="trl_grpo",
        sampling_config={"temperature": 0.8},
        verifier_id="gsm8k_numeric_v1",
        prompt="p",
        completion=f"{trace_id} completion",
        reward_components={"reward": reward},
    )
