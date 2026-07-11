"""On-policy distillation contracts."""

from post_train_engine.methods.opd.distillation import (
    DistillationOODGuard,
    DistillationTeacher,
    OnPolicyDistillationRow,
    TeacherSignal,
    build_mopd_peer_context,
    build_multi_teacher_view,
    multi_teacher_kl_loss,
)

__all__ = [
    "DistillationOODGuard",
    "DistillationTeacher",
    "OnPolicyDistillationRow",
    "TeacherSignal",
    "build_mopd_peer_context",
    "build_multi_teacher_view",
    "multi_teacher_kl_loss",
]
