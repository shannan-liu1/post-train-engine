from __future__ import annotations

from post_train_engine.opsd.context import (
    OPSDPrivilegedInfo,
    build_opsd_student_context,
    build_opsd_teacher_context,
)


def _info() -> OPSDPrivilegedInfo:
    return OPSDPrivilegedInfo(
        gold_final_answer="18",
        reference_solution="9+9=18",
        reference_source="gsm8k_gold_solution",
        verifier_name="gsm8k_numeric_v1",
        canonical_answer="18",
        verified=True,
    )


def test_opsd_student_context_has_no_gold_answer() -> None:
    context = build_opsd_student_context("What is 9+9?")

    assert "18" not in context
    assert "What is 9+9?" in context


def test_opsd_teacher_context_has_gold_answer_and_reference_solution() -> None:
    context = build_opsd_teacher_context("What is 9+9?", _info())

    assert "9+9=18" in context
    assert "Canonical final answer:\n18" in context
