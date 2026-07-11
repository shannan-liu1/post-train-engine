"""OPSD scaffolding for hard-example distillation."""

from post_train_engine.opsd.context import (
    OPSDPrivilegedInfo,
    build_opsd_student_context,
    build_opsd_teacher_context,
)
from post_train_engine.opsd.loss import OPSDLossConfig, opsd_forward_kl_loss

__all__ = [
    "OPSDLossConfig",
    "OPSDPrivilegedInfo",
    "build_opsd_student_context",
    "build_opsd_teacher_context",
    "opsd_forward_kl_loss",
]
