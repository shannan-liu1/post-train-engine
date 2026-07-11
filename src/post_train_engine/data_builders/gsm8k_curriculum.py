"""Build GSM8K curriculum datasets from probe difficulty buckets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from post_train_engine.difficulty import DifficultyBucket, DifficultyBucketRecord
from post_train_engine.jsonl import write_jsonl
from post_train_engine.opsd.context import build_opsd_student_context
from post_train_engine.tasks.gsm8k import GSM8KExample, format_prompt, verify_answer


@dataclass(frozen=True)
class GSM8KCurriculum:
    grpo_frontier: tuple[dict[str, Any], ...]
    easy_regression: tuple[dict[str, Any], ...]
    opsd_hard: tuple[dict[str, Any], ...]
    quarantine: tuple[dict[str, Any], ...]


def frontier_weight(pass_rate: float) -> float:
    if not 0.0 <= pass_rate <= 1.0:
        raise ValueError("pass_rate must be between 0 and 1")
    frontier_mid = 1.0 - abs(pass_rate - 0.5) / 0.5
    return max(0.25, frontier_mid)


def build_gsm8k_curriculum(
    examples: Mapping[str, GSM8KExample],
    buckets: Sequence[DifficultyBucketRecord],
    probe_rows: Sequence[Mapping[str, Any]],
    *,
    source_probe_run_id: str,
    prompt_style: str = "thinking_tags",
) -> GSM8KCurriculum:
    rows_by_example: dict[str, list[Mapping[str, Any]]] = {}
    for row in probe_rows:
        rows_by_example.setdefault(str(row["example_id"]), []).append(row)

    grpo_frontier: list[dict[str, Any]] = []
    easy_regression: list[dict[str, Any]] = []
    opsd_hard: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []

    for bucket in buckets:
        example = examples[bucket.example_id]
        if bucket.bucket is DifficultyBucket.FRONTIER:
            grpo_frontier.append(
                {
                    "example_id": example.id,
                    "question": example.question,
                    "gold_answer": example.gold_answer,
                    "gold_solution": example.gold_solution,
                    "rho_q": bucket.pass_rate,
                    "prompt": format_prompt(example.question, prompt_style),
                    "bucket": bucket.bucket.value,
                    "source_probe_run_id": source_probe_run_id,
                    "source_trace_ids": _source_trace_ids(bucket),
                    "source_split_roles": ["probe"],
                    "weight": frontier_weight(bucket.pass_rate),
                }
            )
        elif bucket.bucket is DifficultyBucket.EASY_STABLE:
            easy_regression.append(
                {
                    "example_id": example.id,
                    "question": example.question,
                    "gold_answer": example.gold_answer,
                    "prompt": format_prompt(example.question, prompt_style),
                    "bucket": bucket.bucket.value,
                    "source_probe_run_id": source_probe_run_id,
                    "source_trace_ids": _source_trace_ids(bucket),
                    "source_split_roles": ["probe"],
                }
            )
        elif bucket.bucket in {
            DifficultyBucket.HARD_SOLVED,
            DifficultyBucket.UNSOLVED_PARSEABLE,
        }:
            opsd_hard.append(
                build_opsd_hard_record(
                    example,
                    bucket,
                    rows_by_example.get(example.id, ()),
                    source_probe_run_id=source_probe_run_id,
                )
            )
        elif bucket.bucket in {
            DifficultyBucket.PARSER_ISSUE,
            DifficultyBucket.LABEL_OR_VERIFIER_SUSPECT,
        }:
            quarantine.append(
                {
                    "example_id": example.id,
                    "question": example.question,
                    "gold_answer": example.gold_answer,
                    "bucket": bucket.bucket.value,
                    "bucket_reason": bucket.bucket_reason,
                    "source_probe_run_id": source_probe_run_id,
                }
            )

    return GSM8KCurriculum(
        grpo_frontier=tuple(grpo_frontier),
        easy_regression=tuple(easy_regression),
        opsd_hard=tuple(opsd_hard),
        quarantine=tuple(quarantine),
    )


def build_opsd_hard_record(
    example: GSM8KExample,
    bucket: DifficultyBucketRecord,
    probe_rows: Sequence[Mapping[str, Any]],
    *,
    source_probe_run_id: str,
) -> dict[str, Any]:
    if bucket.bucket is DifficultyBucket.HARD_SOLVED:
        reference_solution = _best_successful_trace(probe_rows)
        reference_source = "verified_student_trace"
    elif bucket.bucket is DifficultyBucket.UNSOLVED_PARSEABLE:
        reference_solution = example.gold_solution
        reference_source = "gsm8k_gold_solution"
    else:
        raise ValueError(f"OPSD hard record cannot use bucket {bucket.bucket.value}")

    verification = verify_answer(example.gold_answer, example.gold_answer)
    return {
        "example_id": example.id,
        "question": example.question,
        "student_prompt": build_opsd_student_context(example.question),
        "teacher_privileged_info": {
            "gold_final_answer": example.gold_answer,
            "reference_solution": reference_solution,
            "reference_source": reference_source,
            "verifier": {
                "name": verification.verifier,
                "canonical_answer": verification.gold_canonical,
                "verified": verification.correct,
            },
        },
        "bucket": bucket.bucket.value,
        "source_probe_run_id": source_probe_run_id,
        "source_trace_ids": _source_trace_ids(bucket),
        "source_split_roles": ["probe"],
        "privileged_visibility": "gold_answer",
    }


def write_gsm8k_curriculum(curriculum: GSM8KCurriculum, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "grpo_frontier.jsonl", curriculum.grpo_frontier)
    write_jsonl(out_dir / "easy_regression.jsonl", curriculum.easy_regression)
    write_jsonl(out_dir / "opsd_hard.jsonl", curriculum.opsd_hard)
    write_jsonl(out_dir / "quarantine.jsonl", curriculum.quarantine)


def _best_successful_trace(rows: Sequence[Mapping[str, Any]]) -> str:
    successful = [row for row in rows if bool(row.get("correct"))]
    if not successful:
        raise ValueError("hard_solved example has no verified successful trace")

    def sort_key(row: Mapping[str, Any]) -> tuple[int, int, int, str]:
        completion = str(row.get("completion", ""))
        strict_parse = 0 if row.get("parser") in {"answer_tag", "hash_marker", "boxed"} else 1
        repeated = 1 if _has_repeated_loops(completion) else 0
        return (
            int(row.get("completion_tokens", len(completion.split()))),
            strict_parse,
            repeated,
            completion,
        )

    return str(sorted(successful, key=sort_key)[0].get("completion", ""))


def _source_trace_ids(bucket: DifficultyBucketRecord) -> list[str]:
    return sorted(set(bucket.successful_trace_ids + bucket.failed_trace_ids))


def _has_repeated_loops(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return len(lines) != len(set(lines))
