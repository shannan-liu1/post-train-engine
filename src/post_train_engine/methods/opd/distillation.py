"""On-policy and multi-teacher distillation primitives.

These contracts intentionally stop at evidence construction and loss math. A
GPU trainer should consume these rows; it should not invent a second format for
student traces, teacher supervision, or privileged context.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from post_train_engine.training_views.builders import build_training_view_artifact
from post_train_engine.training_views.schema import TrainingViewArtifact
from post_train_engine.traces.schema import SplitRole, TraceRecord

_FROZEN_FORBID = ConfigDict(frozen=True, extra="forbid")

TeacherKind = Literal["external", "self_privileged", "peer_conditioned", "committee"]
SupervisionKind = Literal["token_distribution", "sequence_score", "critique", "reference_completion"]
TeacherVisibility = Literal["none", "gold_answer", "verifier_feedback", "privileged_context", "environment"]


class DistillationOODGuard(BaseModel):
    """Fail-closed evidence that teacher advantage transfers to a target domain."""

    model_config = _FROZEN_FORBID

    source_domain_ids: tuple[str, ...] = Field(..., min_length=1)
    target_domain_id: str = Field(..., min_length=1)
    measured_teacher_advantage: float
    min_teacher_advantage: float = 0.0
    evaluation_artifact_id: str = Field(..., min_length=1)

    @field_validator("measured_teacher_advantage", "min_teacher_advantage")
    @classmethod
    def _advantage_must_be_finite(cls, value: float) -> float:
        if type(value) is bool or not isinstance(value, int | float) or not math.isfinite(value):
            raise ValueError("teacher advantage must be finite")
        return float(value)

    @model_validator(mode="after")
    def _ood_advantage_must_clear_gate(self) -> DistillationOODGuard:
        if self.target_domain_id in self.source_domain_ids:
            raise ValueError("OOD target domain must differ from source domains")
        if self.measured_teacher_advantage < self.min_teacher_advantage:
            raise ValueError(
                "OOD teacher advantage does not clear the configured minimum"
            )
        return self


class DistillationTeacher(BaseModel):
    """A teacher identity that can be audited across distillation rows."""

    model_config = _FROZEN_FORBID

    teacher_id: str = Field(..., min_length=1)
    teacher_kind: TeacherKind
    model_id: str = Field(..., min_length=1)
    tokenizer_id: str | None = None
    weight: float = Field(default=1.0, gt=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("weight")
    @classmethod
    def _weight_finite(cls, value: float) -> float:
        if type(value) is bool or not isinstance(value, int | float) or not math.isfinite(float(value)):
            raise ValueError("weight must be finite")
        return float(value)

    @staticmethod
    def validate_unique(teachers: Sequence[DistillationTeacher]) -> tuple[DistillationTeacher, ...]:
        if not teachers:
            raise ValueError("at least one teacher is required")
        ids = [teacher.teacher_id for teacher in teachers]
        if len(ids) != len(set(ids)):
            raise ValueError("teacher_id values must be unique")
        return tuple(teachers)


class TeacherSignal(BaseModel):
    """Teacher supervision attached to one on-policy student trace."""

    model_config = _FROZEN_FORBID

    teacher_id: str = Field(..., min_length=1)
    teacher_kind: TeacherKind
    weight: float = Field(..., gt=0.0)
    supervision_kind: SupervisionKind
    visibility: TeacherVisibility
    score: float | None = None
    artifact_ref: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("weight", "score")
    @classmethod
    def _finite_optional(cls, value: float | None, info: object) -> float | None:
        if value is None:
            return None
        if type(value) is bool or not isinstance(value, int | float) or not math.isfinite(float(value)):
            raise ValueError(f"{getattr(info, 'field_name', 'value')} must be finite")
        return float(value)


class OnPolicyDistillationRow(BaseModel):
    """One on-policy student trace plus one or more teacher signals."""

    model_config = _FROZEN_FORBID

    row_id: str = Field(..., min_length=1)
    target_trace_id: str = Field(..., min_length=1)
    run_id: str = Field(..., min_length=1)
    task_id: str = Field(..., min_length=1)
    example_id: str = Field(..., min_length=1)
    split_role: SplitRole
    prompt: str = Field(..., min_length=1)
    student_completion: str
    teacher_signals: tuple[TeacherSignal, ...] = Field(..., min_length=1)
    peer_context: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_distillation_contract(self) -> OnPolicyDistillationRow:
        if self.split_role in {"selection", "promotion", "canary", "unseen"}:
            raise ValueError(
                "on-policy distillation rows cannot use protected evaluation split sources"
            )
        teacher_ids = [signal.teacher_id for signal in self.teacher_signals]
        if len(teacher_ids) != len(set(teacher_ids)):
            raise ValueError("teacher_signals teacher_id values must be unique per row")
        if any(
            signal.teacher_kind == "self_privileged" and signal.visibility == "none"
            for signal in self.teacher_signals
        ):
            raise ValueError("self_privileged teacher signals must declare privileged visibility")
        return self

    def to_json(self) -> dict[str, Any]:
        body = self.model_dump(mode="json")
        body["source_trace_ids"] = [self.target_trace_id]
        body["source_split_roles"] = [self.split_role]
        body["teacher_weight_sum"] = sum(signal.weight for signal in self.teacher_signals)
        return body


def build_mopd_peer_context(
    *,
    target_trace_id: str,
    traces: Iterable[TraceRecord],
    success_threshold: float,
    max_successes: int = 2,
    max_failures: int = 1,
) -> dict[str, Any]:
    if not math.isfinite(success_threshold):
        raise ValueError("success_threshold must be finite")
    if max_successes < 0 or max_failures < 0:
        raise ValueError("max_successes and max_failures must be non-negative")
    successes: list[TraceRecord] = []
    failures: list[TraceRecord] = []
    target_seen = False
    for trace in traces:
        if trace.trace_id == target_trace_id:
            target_seen = True
            continue
        reward = trace.reward_components.get("reward")
        if reward is None:
            continue
        if reward >= success_threshold:
            successes.append(trace)
        else:
            failures.append(trace)
    if not target_seen:
        raise ValueError(f"target trace not found in rollout group: {target_trace_id}")
    successes = sorted(successes, key=lambda trace: (-trace.reward_components["reward"], trace.trace_id))
    failures = sorted(failures, key=lambda trace: (trace.reward_components["reward"], trace.trace_id))
    return {
        "strategy": "contrastive_success_failure",
        "success_threshold": success_threshold,
        "success_trace_ids": tuple(trace.trace_id for trace in successes[:max_successes]),
        "failure_trace_ids": tuple(trace.trace_id for trace in failures[:max_failures]),
    }


def build_multi_teacher_view(
    *,
    view_id: str,
    run_id: str,
    task_id: str,
    rows: Sequence[OnPolicyDistillationRow],
    data_path: str | Path,
    ood_guard: DistillationOODGuard,
) -> TrainingViewArtifact:
    if not rows:
        raise ValueError("multi-teacher view requires at least one row")
    path = Path(data_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized_rows = [row.to_json() for row in rows]
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in serialized_rows),
        encoding="utf-8",
    )
    teacher_ids = sorted({signal.teacher_id for row in rows for signal in row.teacher_signals})
    visibility = _view_visibility(rows)
    return build_training_view_artifact(
        view_id=view_id,
        run_id=run_id,
        task_id=task_id,
        view_type="multi_teacher_opd",
        method_compatibility=("opd", "multi_teacher_opd"),
        data_path=path,
        artifact_root=path.parent,
        data_kind="multi_teacher_opd_jsonl",
        rows=serialized_rows,
        privileged_visibility=visibility,
        metadata={
            "teacher_ids": teacher_ids,
            "aggregation": "weighted_probability_mixture",
            "row_count": len(rows),
            "peer_conditioned": any(row.peer_context for row in rows),
            "ood_guard": ood_guard.model_dump(mode="json"),
        },
    )


def _view_visibility(rows: Sequence[OnPolicyDistillationRow]) -> Literal[
    "none",
    "gold_answer",
    "verifier_feedback",
    "privileged_context",
    "unknown",
]:
    visibility_order = {
        "none": 0,
        "verifier_feedback": 1,
        "gold_answer": 2,
        "environment": 2,
        "privileged_context": 3,
    }
    strongest = "none"
    strongest_rank = 0
    for row in rows:
        for signal in row.teacher_signals:
            rank = visibility_order[signal.visibility]
            if rank > strongest_rank:
                strongest = signal.visibility
                strongest_rank = rank
    if strongest == "environment":
        return "privileged_context"
    return cast(Literal["none", "gold_answer", "verifier_feedback", "privileged_context", "unknown"], strongest)


def multi_teacher_kl_loss(
    *,
    student_logits: Any,
    teacher_logits: Any,
    teacher_weights: Any,
    rollout_token_mask: Any,
    temperature: float = 1.0,
) -> Any:
    """Forward KL from a weighted teacher probability mixture to student policy."""

    import torch
    import torch.nn.functional as F

    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if student_logits.ndim != 3:
        raise ValueError("student_logits must have shape [batch, seq, vocab]")
    if teacher_logits.ndim != 4:
        raise ValueError("teacher_logits must have shape [teachers, batch, seq, vocab]")
    if teacher_logits.shape[1:] != student_logits.shape:
        raise ValueError("teacher logits batch/seq/vocab dimensions must match student logits")
    if teacher_weights.shape != (teacher_logits.shape[0],):
        raise ValueError("teacher_weights must have shape [teachers]")
    if rollout_token_mask.shape != student_logits.shape[:2]:
        raise ValueError("rollout_token_mask must have shape [batch, seq]")
    if not rollout_token_mask.any():
        raise ValueError("rollout_token_mask selects no rollout tokens")
    weights = teacher_weights.float()
    if (weights <= 0.0).any() or not torch.isfinite(weights).all():
        raise ValueError("teacher_weights must be positive finite values")
    weights = weights / weights.sum()
    mask = rollout_token_mask.to(dtype=torch.bool)
    selected_student = student_logits[mask].float() / temperature
    selected_teacher = teacher_logits.detach()[:, mask, :].float() / temperature
    teacher_probs = F.softmax(selected_teacher, dim=-1)
    mixed_teacher_probs = (teacher_probs * weights[:, None, None]).sum(dim=0)
    mixed_teacher_probs = mixed_teacher_probs.clamp_min(1e-12)
    mixed_teacher_log_probs = mixed_teacher_probs.log()
    student_log_probs = F.log_softmax(selected_student, dim=-1)
    return (mixed_teacher_probs * (mixed_teacher_log_probs - student_log_probs)).sum(dim=-1).mean()


__all__ = [
    "DistillationTeacher",
    "DistillationOODGuard",
    "OnPolicyDistillationRow",
    "TeacherSignal",
    "build_mopd_peer_context",
    "build_multi_teacher_view",
    "multi_teacher_kl_loss",
]
