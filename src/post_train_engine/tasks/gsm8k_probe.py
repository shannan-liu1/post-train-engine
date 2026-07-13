"""GSM8K probe evidence primitive consumed by canonical Run adapters."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from post_train_engine.probe import (
    EarlyExitConfig,
    ProbeArtifactRow,
    early_exit_rejection_reason,
    full_filter_reason,
    should_continue_after_early,
)
from post_train_engine.rewards.gsm8k import GSM8KRewardConfig, compute_gsm8k_reward
from post_train_engine.tasks.gsm8k import (
    GSM8KExample,
    ParseMode,
    format_prompt,
    parse_model_answer,
    verify_answer,
)

GenerationFn = Callable[
    [GSM8KExample, int, str, Mapping[str, Any]],
    str | Mapping[str, Any],
]


def probe_gsm8k_examples(
    examples: Sequence[GSM8KExample],
    generate: GenerationFn,
    *,
    run_id: str,
    bucket_source_split: str,
    model_id: str,
    checkpoint_id: str,
    prompt_style: str,
    generation_config: Mapping[str, Any],
    early_exit: EarlyExitConfig,
    seed: int,
    reward_config: GSM8KRewardConfig | None = None,
) -> tuple[ProbeArtifactRow, ...]:
    reward_config = reward_config or GSM8KRewardConfig(
        max_new_tokens=int(generation_config.get("max_new_tokens", 512)),
    )
    rows: list[ProbeArtifactRow] = []
    probe_parse_mode: ParseMode = (
        "lenient" if reward_config.use_lenient_parse_for_probe else "strict"
    )
    for example in examples:
        prompt = format_prompt(example.question, prompt_style)
        prompt_hash = _sha256(prompt)
        staged: list[dict[str, Any]] = []
        for rollout_id in range(early_exit.g_early):
            staged.append(
                _rollout_to_probe_fields(
                    example,
                    generate(example, rollout_id, prompt, generation_config),
                    rollout_id=rollout_id,
                    parse_mode=probe_parse_mode,
                )
            )
        early_successes = sum(1 for row in staged if row["correct"])
        early_reason = early_exit_rejection_reason(early_successes, early_exit)
        if should_continue_after_early(early_successes, early_exit):
            for rollout_id in range(early_exit.g_early, early_exit.g_total):
                staged.append(
                    _rollout_to_probe_fields(
                        example,
                        generate(example, rollout_id, prompt, generation_config),
                        rollout_id=rollout_id,
                        parse_mode=probe_parse_mode,
                    )
                )
            full_reason = full_filter_reason(
                sum(1 for row in staged if row["correct"]),
                early_exit,
            )
        else:
            full_reason = early_reason or "accepted_frontier"

        rho_q = sum(1 for row in staged if row["correct"]) / len(staged)
        for row in staged:
            reward = compute_gsm8k_reward(
                str(row["completion"]),
                example.gold_answer,
                rho_q=rho_q,
                completion_tokens=int(row["completion_tokens"]),
                config=reward_config,
            )
            rows.append(
                ProbeArtifactRow(
                    run_id=run_id,
                    example_id=example.id,
                    bucket_source_split=bucket_source_split,
                    model_id=model_id,
                    checkpoint_id=checkpoint_id,
                    prompt_hash=prompt_hash,
                    prompt=prompt,
                    rollout_id=int(row["rollout_id"]),
                    seed=seed + int(row["rollout_id"]),
                    generation_config=dict(generation_config),
                    completion=str(row["completion"]),
                    completion_tokens=int(row["completion_tokens"]),
                    finish_reason=str(row["finish_reason"]),
                    parsed_answer=(
                        str(row["parsed_answer"])
                        if row["parsed_answer"] is not None
                        else None
                    ),
                    parse_ok=bool(row["parse_ok"]),
                    parser=str(row["parser"]) if row["parser"] is not None else None,
                    gold_answer=example.gold_answer,
                    correct=bool(row["correct"]),
                    reward=reward.reward,
                    latency_ms=float(row["latency_ms"]),
                    error=row["error"],
                    question=example.question,
                    gold_solution=example.gold_solution,
                    filter_reason=full_reason,
                )
            )
    return tuple(rows)


def _rollout_to_probe_fields(
    example: GSM8KExample,
    generated: str | Mapping[str, Any],
    *,
    rollout_id: int,
    parse_mode: ParseMode = "lenient",
) -> dict[str, Any]:
    if isinstance(generated, str):
        completion = generated
        completion_tokens = len(generated.split())
        finish_reason = "stop"
        latency_ms = 0
        error = None
    else:
        completion = str(generated.get("completion", ""))
        completion_tokens = int(
            generated.get("completion_tokens", len(completion.split()))
        )
        finish_reason = str(generated.get("finish_reason", "stop"))
        latency_ms = float(generated.get("latency_ms", 0))
        error = generated.get("error")
    parsed = parse_model_answer(completion, mode=parse_mode)
    verification = (
        verify_answer(parsed.answer, example.gold_answer)
        if parsed.parse_ok and parsed.answer is not None
        else None
    )
    return {
        "rollout_id": rollout_id,
        "completion": completion,
        "completion_tokens": completion_tokens,
        "finish_reason": finish_reason,
        "parsed_answer": parsed.answer,
        "parse_ok": parsed.parse_ok,
        "parser": parsed.parser,
        "correct": bool(verification and verification.correct),
        "latency_ms": latency_ms,
        "error": error
        or parsed.error
        or (verification.error if verification else None),
    }


def _sha256(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


__all__ = ["GenerationFn", "probe_gsm8k_examples"]
