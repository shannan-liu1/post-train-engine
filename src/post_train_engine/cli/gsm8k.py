"""GSM8K hill-climbing artifact CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from post_train_engine.data_builders.gsm8k_curriculum import (
    build_gsm8k_curriculum,
    write_gsm8k_curriculum,
)
from post_train_engine.difficulty import bucket_probe_artifact
from post_train_engine.generation import GenerationFn
from post_train_engine.ledger import make_run_ledger_entry, write_ledger_entry
from post_train_engine.probe import (
    EarlyExitConfig,
    ProbeArtifactRow,
    early_exit_rejection_reason,
    full_filter_reason,
    read_probe_jsonl,
    should_continue_after_early,
)
from post_train_engine.rewards.gsm8k import GSM8KRewardConfig, compute_gsm8k_reward
from post_train_engine.tasks.gsm8k import (
    GSM8KExample,
    ParseMode,
    format_prompt,
    load_gsm8k,
    make_gsm8k_splits,
    parse_model_answer,
    verify_answer,
)


def register_gsm8k_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("gsm8k")
    gsm8k_sub = parser.add_subparsers(dest="gsm8k_command", required=True)
    _register_gsm8k_commands(gsm8k_sub)


def _register_gsm8k_commands(
    gsm8k_sub: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:

    make_splits = gsm8k_sub.add_parser("make-splits")
    make_splits.add_argument("--dataset", default="openai/gsm8k")
    make_splits.add_argument("--seed", type=int, required=True)
    make_splits.add_argument("--train-pool-size", type=int, required=True)
    make_splits.add_argument("--dev-promotion-size", type=int, required=True)
    make_splits.add_argument("--dev-diagnostics-size", type=int)
    make_splits.add_argument("--out", required=True)
    make_splits.set_defaults(func=cmd_make_splits)

    curriculum = gsm8k_sub.add_parser("build-curriculum")
    curriculum.add_argument("--probe", required=True)
    curriculum.add_argument("--splits", required=True)
    curriculum.add_argument("--out-dir", required=True)
    curriculum.add_argument("--prompt-style", default="thinking_tags")
    curriculum.set_defaults(func=cmd_build_curriculum)


def cmd_make_splits(args: argparse.Namespace) -> None:
    train = load_gsm8k("train", args.dataset)
    test = load_gsm8k("test", args.dataset)
    dataset_revision = _dataset_revision(train)
    splits = make_gsm8k_splits(
        train,
        seed=args.seed,
        train_pool_size=args.train_pool_size,
        dev_promotion_size=args.dev_promotion_size,
        dev_diagnostics_size=args.dev_diagnostics_size,
        official_examples=test,
        dataset=args.dataset,
        dataset_revision=dataset_revision,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(splits.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    _write_default_ledger(
        out,
        run_id=f"gsm8k_splits_{args.seed}",
        command=sys.argv,
        dataset=args.dataset,
        dataset_revision=dataset_revision,
        split_hash=splits.split_hash,
        artifact_paths={"splits": str(out)},
        seeds=[args.seed],
    )


def cmd_build_curriculum(args: argparse.Namespace) -> None:
    rows = read_probe_jsonl(args.probe)
    buckets = bucket_probe_artifact(rows)
    split_artifact = _load_split_file(args.splits)
    examples = {
        example.id: example
        for example in _load_examples_for_ids(
            split_artifact,
            sorted({str(row["example_id"]) for row in rows}),
        )
    }
    run_id = str(rows[0].get("run_id", "unknown_probe")) if rows else "unknown_probe"
    curriculum = build_gsm8k_curriculum(
        examples,
        buckets,
        rows,
        source_probe_run_id=run_id,
        prompt_style=args.prompt_style,
    )
    write_gsm8k_curriculum(curriculum, args.out_dir)
    _write_default_ledger(
        Path(args.out_dir) / "ledger.jsonl",
        run_id=f"curriculum_{int(time.time())}",
        parent_run_id=run_id,
        command=sys.argv,
        dataset=split_artifact.get("dataset"),
        dataset_revision=split_artifact.get("dataset_revision"),
        split_hash=split_artifact.get("split_hash"),
        prompt_template=args.prompt_style,
        artifact_paths={"curriculum": str(args.out_dir)},
    )


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
                    parsed_answer=str(row["parsed_answer"])
                    if row["parsed_answer"] is not None
                    else None,
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
        completion_tokens = int(generated.get("completion_tokens", len(completion.split())))
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
        "error": error or parsed.error or (verification.error if verification else None),
    }


def _load_split_file(path: str | Path) -> dict[str, Any]:
    body = json.loads(Path(path).read_text(encoding="utf-8"))
    if "splits" not in body:
        raise ValueError("split file must include splits")
    return body


def _load_examples_for_ids(
    split_artifact: Mapping[str, Any],
    ids: Sequence[str],
) -> list[GSM8KExample]:
    dataset = str(split_artifact.get("dataset", "openai/gsm8k"))
    need_train = any("/train/" in example_id for example_id in ids)
    need_test = any("/test/" in example_id for example_id in ids)
    examples: list[GSM8KExample] = []
    if need_train:
        examples.extend(load_gsm8k("train", dataset))
    if need_test:
        examples.extend(load_gsm8k("test", dataset))
    by_id = {example.id: example for example in examples}
    return [by_id[example_id] for example_id in ids]


def _dataset_revision(examples: Sequence[GSM8KExample]) -> str:
    if not examples:
        return "unknown"
    return str(examples[0].metadata.get("dataset_revision", "unknown"))


def _sha256(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_default_ledger(path_hint: Path, **kwargs: Any) -> None:
    ledger = path_hint.parent / "ledger.jsonl" if path_hint.suffix else path_hint
    entry = make_run_ledger_entry(**kwargs)
    write_ledger_entry(ledger, entry)
