"""OPSD student and teacher contexts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class OPSDPrivilegedInfo:
    gold_final_answer: str
    reference_solution: str
    reference_source: Literal[
        "gsm8k_gold_solution",
        "verified_student_trace",
        "external_verified_trace",
    ]
    verifier_name: str
    canonical_answer: str
    verified: bool


def build_opsd_student_context(question: str) -> str:
    return f"""Problem:
{question}

Solve the problem. Put only the final answer inside <answer>...</answer>.
Assistant:
"""


def build_opsd_teacher_context(question: str, info: OPSDPrivilegedInfo) -> str:
    return f"""Problem:
{question}

Privileged verified reference solution:
{info.reference_solution}

Canonical final answer:
{info.gold_final_answer}

The reference solution and final answer are verified. After understanding the reference,
evaluate and guide the student's attempted solution. Use the reference as privileged
training information only.

Assistant:
"""
