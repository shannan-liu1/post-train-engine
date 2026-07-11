"""GSM8K hill-climbing artifact CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from post_train_engine.data_builders.gsm8k_curriculum import (
    build_gsm8k_curriculum,
    write_gsm8k_curriculum,
)
from post_train_engine.difficulty import bucket_probe_artifact
from post_train_engine.evals.promotion import EvalArtifact, EvalExampleResult
from post_train_engine.generation import (
    DEFAULT_ROLLOUT_BACKEND,
    GenerationFn,
    build_generation_fn,
)
from post_train_engine.ledger import make_run_ledger_entry, write_ledger_entry
from post_train_engine.probe import (
    EarlyExitConfig,
    ProbeArtifactRow,
    early_exit_rejection_reason,
    full_filter_reason,
    read_probe_jsonl,
    should_continue_after_early,
    write_probe_jsonl,
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

    probe = gsm8k_sub.add_parser("probe")
    probe.add_argument("--model", required=True)
    probe.add_argument("--checkpoint-id", default="base")
    probe.add_argument("--split-file", required=True)
    probe.add_argument("--split", required=True)
    probe.add_argument("--sample-size", type=int)
    probe.add_argument("--rollouts", type=int, default=16)
    probe.add_argument("--early-rollouts", type=int, default=4)
    probe.add_argument("--temperature", type=float, default=1.0)
    probe.add_argument("--top-p", type=float, default=0.97)
    probe.add_argument("--max-new-tokens", type=int, default=512)
    probe.add_argument("--prompt-style", default="thinking_tags")
    probe.add_argument(
        "--generation-backend",
        choices=["vllm", "hf"],
        default=DEFAULT_ROLLOUT_BACKEND,
    )
    probe.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    probe.add_argument("--vllm-dtype", default="auto")
    probe.add_argument("--trust-remote-code", action="store_true")
    probe.add_argument("--seed", type=int, default=1000)
    probe.add_argument("--out", required=True)
    probe.set_defaults(func=cmd_probe)

    curriculum = gsm8k_sub.add_parser("build-curriculum")
    curriculum.add_argument("--probe", required=True)
    curriculum.add_argument("--splits", required=True)
    curriculum.add_argument("--out-dir", required=True)
    curriculum.add_argument("--prompt-style", default="thinking_tags")
    curriculum.set_defaults(func=cmd_build_curriculum)

    eval_parser = gsm8k_sub.add_parser("eval")
    eval_parser.add_argument("--model", required=True)
    eval_parser.add_argument("--split-file", required=True)
    eval_parser.add_argument("--split", required=True)
    eval_parser.add_argument("--greedy", action="store_true")
    eval_parser.add_argument("--sampled-rollouts", type=int, default=0)
    eval_parser.add_argument("--temperature", type=float, default=1.0)
    eval_parser.add_argument("--top-p", type=float, default=0.97)
    eval_parser.add_argument("--max-new-tokens", type=int, default=512)
    eval_parser.add_argument("--prompt-style", default="thinking_tags")
    eval_parser.add_argument(
        "--generation-backend",
        choices=["vllm", "hf"],
        default=DEFAULT_ROLLOUT_BACKEND,
    )
    eval_parser.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    eval_parser.add_argument("--vllm-dtype", default="auto")
    eval_parser.add_argument("--trust-remote-code", action="store_true")
    eval_parser.add_argument("--seed", type=int, default=1000)
    eval_parser.add_argument("--out", required=True)
    eval_parser.set_defaults(func=cmd_eval)


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


def cmd_probe(args: argparse.Namespace) -> None:
    _validate_probe_args(args)
    split_artifact = _load_split_file(args.split_file)
    examples = _load_examples_for_split(split_artifact, args.split)
    if args.sample_size is not None:
        examples = examples[: args.sample_size]
    generation_config = _generation_config_from_args(args, greedy=False)
    generate = _build_generation_fn_from_args(args, default_greedy=False)
    rows = probe_gsm8k_examples(
        examples,
        generate,
        run_id=f"probe_{int(time.time())}",
        bucket_source_split=args.split,
        model_id=args.model,
        checkpoint_id=args.checkpoint_id,
        prompt_style=args.prompt_style,
        generation_config=generation_config,
        early_exit=EarlyExitConfig(g_total=args.rollouts, g_early=args.early_rollouts),
        seed=args.seed,
    )
    out = Path(args.out)
    write_probe_jsonl(rows, out)
    _write_default_ledger(
        out,
        run_id=rows[0].run_id if rows else "probe_empty",
        command=sys.argv,
        base_model=args.model,
        checkpoint=args.checkpoint_id,
        dataset=split_artifact.get("dataset"),
        dataset_revision=split_artifact.get("dataset_revision"),
        split_hash=split_artifact.get("split_hash"),
        prompt_template=args.prompt_style,
        reward_config=asdict(GSM8KRewardConfig(max_new_tokens=args.max_new_tokens)),
        artifact_paths={"probe": str(out)},
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


def cmd_eval(args: argparse.Namespace) -> None:
    if not args.greedy:
        raise ValueError("gsm8k eval requires --greedy for greedy_exact_accuracy@1")
    _validate_eval_args(args)
    split_artifact = _load_split_file(args.split_file)
    examples = _load_examples_for_split(split_artifact, args.split)
    raw_generate = _build_generation_fn_from_args(args, default_greedy=args.greedy)

    def greedy_generate(
        example: GSM8KExample,
        rollout_id: int,
        prompt: str,
        config: Mapping[str, Any],
    ) -> str | Mapping[str, Any]:
        return raw_generate(
            example,
            rollout_id,
            prompt,
            {**dict(config), "greedy": True},
        )

    sampled_generate: GenerationFn | None = None
    if args.sampled_rollouts > 0:

        def sampled_generate(
            example: GSM8KExample,
            rollout_id: int,
            prompt: str,
            config: Mapping[str, Any],
        ) -> str | Mapping[str, Any]:
            return raw_generate(
                example,
                rollout_id,
                prompt,
                {**dict(config), "greedy": False},
            )

    artifact = evaluate_gsm8k_examples(
        examples,
        greedy_generate,
        artifact_id=Path(args.out).stem,
        model_id=args.model,
        prompt_style=args.prompt_style,
        generation_config=_generation_config_from_args(
            args,
            greedy=bool(args.greedy),
            sampled_rollouts=args.sampled_rollouts,
        ),
        sampled_generate=sampled_generate,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    _write_default_ledger(
        out,
        run_id=artifact.artifact_id,
        command=sys.argv,
        base_model=args.model,
        dataset=split_artifact.get("dataset"),
        dataset_revision=split_artifact.get("dataset_revision"),
        split_hash=split_artifact.get("split_hash"),
        prompt_template=args.prompt_style,
        artifact_paths={"eval": str(out)},
        seeds=[args.seed],
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


def evaluate_gsm8k_examples(
    examples: Sequence[GSM8KExample],
    generate: GenerationFn,
    *,
    artifact_id: str,
    model_id: str,
    prompt_style: str,
    generation_config: Mapping[str, Any],
    sampled_generate: GenerationFn | None = None,
) -> EvalArtifact:
    results: list[EvalExampleResult] = []
    sampled_by_example: list[list[bool]] = []
    sampled_rollouts = int(generation_config.get("sampled_rollouts", 0) or 0)
    if sampled_rollouts < 0:
        raise ValueError("sampled_rollouts must be non-negative")
    for example in examples:
        prompt = format_prompt(example.question, prompt_style)
        generated = generate(example, 0, prompt, generation_config)
        fields = _rollout_to_probe_fields(
            example,
            generated,
            rollout_id=0,
            parse_mode="strict",
        )
        results.append(
            EvalExampleResult(
                example_id=example.id,
                correct=bool(fields["correct"]),
                parse_ok=bool(fields["parse_ok"]),
                tokens=int(fields["completion_tokens"]),
            )
        )
        sampled_correct: list[bool] = []
        if sampled_rollouts:
            sampler = sampled_generate or generate
            for rollout_id in range(sampled_rollouts):
                sampled_fields = _rollout_to_probe_fields(
                    example,
                    sampler(example, rollout_id + 1, prompt, generation_config),
                    rollout_id=rollout_id + 1,
                    parse_mode="strict",
                )
                sampled_correct.append(bool(sampled_fields["correct"]))
        sampled_by_example.append(sampled_correct)
    accuracy = sum(row.correct for row in results) / len(results) if results else 0.0
    parse_rate = sum(row.parse_ok for row in results) / len(results) if results else 0.0
    mean_tokens = (
        sum(float(row.tokens or 0) for row in results) / len(results) if results else 0.0
    )
    metrics = {
        "greedy_exact_accuracy@1": accuracy,
        "parse_success@1": parse_rate,
        "mean_tokens": mean_tokens,
    }
    if sampled_rollouts:
        sampled_values = [
            correct for per_example in sampled_by_example for correct in per_example
        ]
        metrics["sampled_exact_accuracy@1"] = (
            sum(sampled_values) / len(sampled_values) if sampled_values else 0.0
        )
        metrics[f"sampled_pass@{sampled_rollouts}"] = (
            sum(any(per_example) for per_example in sampled_by_example)
            / len(sampled_by_example)
            if sampled_by_example
            else 0.0
        )
    return EvalArtifact(
        artifact_id=artifact_id,
        primary_metric="greedy_exact_accuracy@1",
        examples=tuple(results),
        metrics=metrics,
        metadata={"model_id": model_id, "generation_config": dict(generation_config)},
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="pte gsm8k")
    subparsers = parser.add_subparsers(dest="gsm8k_command", required=True)
    _register_gsm8k_commands(subparsers)
    args = parser.parse_args(argv)
    args.func(args)


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


def _generation_config_from_args(
    args: argparse.Namespace,
    *,
    greedy: bool,
    sampled_rollouts: int | None = None,
) -> dict[str, Any]:
    config = {
        "backend": args.generation_backend,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "greedy": greedy,
        "vllm_tensor_parallel_size": args.vllm_tensor_parallel_size,
        "vllm_dtype": args.vllm_dtype,
        "trust_remote_code": args.trust_remote_code,
        "seed": args.seed,
    }
    if sampled_rollouts is not None:
        config["sampled_rollouts"] = sampled_rollouts
    return config


def _validate_probe_args(args: argparse.Namespace) -> None:
    _require_positive_int(args.rollouts, "rollouts")
    _require_positive_int(args.early_rollouts, "early-rollouts")
    if args.early_rollouts > args.rollouts:
        raise ValueError("early-rollouts cannot exceed rollouts")
    _require_generation_args(args)
    if args.sample_size is not None:
        _require_positive_int(args.sample_size, "sample-size")


def _validate_eval_args(args: argparse.Namespace) -> None:
    _require_generation_args(args)
    if args.sampled_rollouts < 0:
        raise ValueError("sampled-rollouts must be non-negative")


def _require_generation_args(args: argparse.Namespace) -> None:
    _require_positive_int(args.max_new_tokens, "max-new-tokens")
    _require_positive_int(args.vllm_tensor_parallel_size, "vllm-tensor-parallel-size")
    if args.temperature < 0.0:
        raise ValueError("temperature must be non-negative")
    if not 0.0 < args.top_p <= 1.0:
        raise ValueError("top-p must be > 0 and <= 1")


def _require_positive_int(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _build_generation_fn_from_args(
    args: argparse.Namespace,
    *,
    default_greedy: bool,
) -> GenerationFn:
    return build_generation_fn(
        args.model,
        backend=args.generation_backend,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        default_greedy=default_greedy,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        vllm_dtype=args.vllm_dtype,
        trust_remote_code=args.trust_remote_code,
    )


def _load_split_file(path: str | Path) -> dict[str, Any]:
    body = json.loads(Path(path).read_text(encoding="utf-8"))
    if "splits" not in body:
        raise ValueError("split file must include splits")
    return body


def _load_examples_for_split(split_artifact: Mapping[str, Any], split_name: str) -> list[GSM8KExample]:
    ids = split_artifact["splits"][split_name]
    return _load_examples_for_ids(split_artifact, ids)


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
