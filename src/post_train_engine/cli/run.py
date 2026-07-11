"""Generic run command for local flywheel smoke bundles."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

import yaml

from post_train_engine.cli.gsm8k import probe_gsm8k_examples
from post_train_engine.config import HuggingFaceLifecycleConfig, ModelLifecycleConfig
from post_train_engine.data.overlap import (
    EvalSplitManifest,
    certify_no_training_promotion_overlap,
    prompt_sha256,
)
from post_train_engine.data_builders.gsm8k_curriculum import (
    GSM8KCurriculum,
    build_gsm8k_curriculum,
    write_gsm8k_curriculum,
)
from post_train_engine.difficulty import bucket_probe_artifact
from post_train_engine.evals.promotion import (
    EvalArtifact,
    EvalExampleResult,
    PromotionGateConfig,
    load_eval_artifact,
)
from post_train_engine.engine import (
    RunEngine,
    RunPlan,
    RunStage,
    StageOutput,
    require_nonfailed_manifest,
)
from post_train_engine.flywheel import CostRecord, PriceSnapshot, ResourceTopology
from post_train_engine.ledger import make_run_ledger_entry, stable_hash
from post_train_engine.lifecycle import (
    CheckpointLifecycleInput,
    CheckpointLifecycleManager,
)
from post_train_engine.jsonl import write_jsonl
from post_train_engine.probe import EarlyExitConfig, write_probe_jsonl
from post_train_engine.registry.checkpoints import CheckpointRecord
from post_train_engine.run_bundle import RunManifest
from post_train_engine.tasks.gsm8k import (
    GSM8KExample,
    GSM8KSplits,
    format_prompt,
)
from post_train_engine.tasks.schema import Example
from post_train_engine.traces import TraceRecord, build_rollout_group
from post_train_engine.training_views import (
    build_training_view_artifact,
    write_training_view_artifact,
)
from post_train_engine.training.optimizers import DEFAULT_OPTIMIZER_FRAMEWORK

Runner = Callable[[Mapping[str, Any], Path], RunManifest]


def register_run_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser("run")
    parser.add_argument("--config", required=True)
    env_group = parser.add_mutually_exclusive_group()
    env_group.add_argument("--env", default=".env")
    env_group.add_argument(
        "--no-env",
        action="store_true",
        help="Do not read a dotenv file; resolve env vars from the process only.",
    )
    parser.set_defaults(func=cmd_run)


def cmd_run(args: argparse.Namespace) -> None:
    execute_run_config(
        args.config,
        env_path=None if args.no_env else args.env,
    )


def execute_run_config(
    config_path: str | Path,
    *,
    env_path: str | Path | None = ".env",
) -> RunManifest | dict[str, Any]:
    raw = _load_config(config_path)
    kind = str(raw.get("kind") or "")
    runner = RUNNERS.get(kind)
    if runner is not None:
        return runner(raw, Path(config_path))
    if raw.get("schema_version") == "runpod_grpo_hillclimb_v1":
        from post_train_engine.runpod_grpo import run_runpod_grpo_hillclimb

        return run_runpod_grpo_hillclimb(config_path)
    if {"run", "dataset", "baseline", "providers"}.issubset(raw):
        from post_train_engine.api_hillclimb import run_hillclimb

        return run_hillclimb(config_path, env_path=env_path)
    raise ValueError(
        f"unsupported run config kind {kind!r}; "
        f"supported run kinds: {', '.join(sorted(RUNNERS))}, "
        "api_hillclimb, runpod_grpo_hillclimb",
    )


def run_local_gsm8k_smoke(
    raw_config: Mapping[str, Any],
    *,
    config_path: Path,
) -> RunManifest:
    """Write a deterministic GSM8K dry-run bundle without model inference."""

    plan, adapter = _compile_local_gsm8k_smoke(raw_config, config_path=config_path)
    execution = RunEngine().execute(plan, adapter)
    require_nonfailed_manifest(execution.manifest, plan.output_dir)
    return execution.manifest

def _write_training_views(
    out_dir: Path,
    *,
    run_id: str,
    curriculum_dir: Path,
    curriculum: GSM8KCurriculum,
) -> dict[str, Path]:
    paths = {
        "grpo_rollout_view": out_dir / "grpo_rollout_view.json",
        "easy_regression_view": out_dir / "easy_regression_view.json",
        "opsd_view": out_dir / "opsd_view.json",
    }
    write_training_view_artifact(
        build_training_view_artifact(
            view_id=f"{run_id}:grpo_frontier",
            run_id=run_id,
            task_id="gsm8k",
            view_type="grpo_rollout",
            method_compatibility=("grpo",),
            data_path=curriculum_dir / "grpo_frontier.jsonl",
            artifact_root=out_dir.parent,
            data_kind="grpo_frontier",
            rows=curriculum.grpo_frontier,
            privileged_visibility="none",
            metadata={"bucket_policy": "frontier"},
        ),
        paths["grpo_rollout_view"],
    )
    write_training_view_artifact(
        build_training_view_artifact(
            view_id=f"{run_id}:easy_regression",
            run_id=run_id,
            task_id="gsm8k",
            view_type="sft",
            method_compatibility=("sft", "grpo"),
            data_path=curriculum_dir / "easy_regression.jsonl",
            artifact_root=out_dir.parent,
            data_kind="easy_regression",
            rows=curriculum.easy_regression,
            privileged_visibility="none",
            metadata={"bucket_policy": "easy_regression_rehearsal"},
        ),
        paths["easy_regression_view"],
    )
    write_training_view_artifact(
        build_training_view_artifact(
            view_id=f"{run_id}:opsd_hard",
            run_id=run_id,
            task_id="gsm8k",
            view_type="opsd",
            method_compatibility=("opsd",),
            data_path=curriculum_dir / "opsd_hard.jsonl",
            artifact_root=out_dir.parent,
            data_kind="opsd_hard",
            rows=curriculum.opsd_hard,
            privileged_visibility="gold_answer",
            metadata={"bucket_policy": "hard_or_unsolved_parseable"},
        ),
        paths["opsd_view"],
    )
    return paths


def _run_gsm8k_local_smoke(
    raw: Mapping[str, Any],
    config_path: Path,
) -> RunManifest:
    return run_local_gsm8k_smoke(raw, config_path=config_path)


RUNNERS: dict[str, Runner] = {
    "gsm8k_local_smoke": _run_gsm8k_local_smoke,
}


class _SmokeConfig:
    def __init__(
        self,
        *,
        run_id: str,
        out_dir: Path,
        seed: int,
        model_id: str,
        prompt_style: str,
        rollouts: int,
        early_rollouts: int,
        max_new_tokens: int,
    ) -> None:
        self.run_id = run_id
        self.out_dir = out_dir
        self.seed = seed
        self.model_id = model_id
        self.prompt_style = prompt_style
        self.rollouts = rollouts
        self.early_rollouts = early_rollouts
        self.max_new_tokens = max_new_tokens

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> _SmokeConfig:
        run_id = str(raw.get("run_id") or "gsm8k-local-smoke")
        out_dir = Path(str(raw.get("out_dir") or Path("runs") / run_id))
        seed = _positive_or_zero_int(raw.get("seed", 123), "seed")
        rollouts = _positive_int(raw.get("rollouts", 4), "rollouts")
        early_rollouts = _positive_int(raw.get("early_rollouts", 2), "early_rollouts")
        if early_rollouts > rollouts:
            raise ValueError("early_rollouts cannot exceed rollouts")
        return cls(
            run_id=run_id,
            out_dir=out_dir,
            seed=seed,
            model_id=str(raw.get("model_id") or "local-deterministic-gsm8k"),
            prompt_style=str(raw.get("prompt_style") or "thinking_tags"),
            rollouts=rollouts,
            early_rollouts=early_rollouts,
            max_new_tokens=_positive_int(raw.get("max_new_tokens", 64), "max_new_tokens"),
        )


def _compile_local_gsm8k_smoke(
    raw_config: Mapping[str, Any],
    *,
    config_path: Path,
) -> tuple[RunPlan, _LocalGSM8KSmokeAdapter]:
    cfg = _SmokeConfig.from_mapping(raw_config)
    train_examples, promotion_examples = _fixture_examples()
    promotion_gate = PromotionGateConfig(
        min_primary_delta=0.10,
        min_primary_ci_low=-1.0,
        max_mcnemar_p=1.0,
        max_parse_regression=0.0,
        max_easy_regression=0.0,
        max_token_increase_ratio=1.0,
    )
    price = PriceSnapshot(
        snapshot_id="local-smoke-2026-06-16",
        provider="local",
        accelerator_type="none",
        accelerator_hour_usd=0.0,
    )
    topology = ResourceTopology(launcher="local", num_nodes=1, gpus_per_node=0)
    costs = (
        CostRecord.from_usage(
            price,
            accelerator_count=0,
            wall_seconds=0.0,
        ),
    )
    plan = RunPlan(
        run_id=cfg.run_id,
        candidate_id="candidate-dry-run",
        parent_candidate_id="seed",
        task_name="gsm8k",
        model_id=cfg.model_id,
        output_dir=str(cfg.out_dir),
        source_root=str(Path(__file__).resolve().parents[3]),
        inputs={
            "model": {
                "kind": "model",
                "requested_id": cfg.model_id,
                "resolved_id": cfg.model_id,
                "resolved_revision": "local-deterministic-fixture-v1",
                "resolution_state": "exact",
            },
            "dataset": {
                "kind": "dataset",
                "requested_id": "local_gsm8k_fixture",
                "resolved_id": "local_gsm8k_fixture",
                "resolved_revision": "2026-06-16",
                "resolution_state": "exact",
            },
        },
        training_example_ids=tuple(example.id for example in train_examples),
        promotion_example_ids=tuple(example.id for example in promotion_examples),
        promotion_gate=asdict(promotion_gate),
        metadata={
            "compute_required": "none",
            "smoke_runner": "local_deterministic_fixture",
            "generation_backend": "deterministic_fixture",
            "checkpoint_ref": "checkpoints/candidate-dry-run",
            "eval_spec_hash": stable_hash({"split": "dev_promotion"}),
            "training_config_hash": stable_hash(
                {"method": "grpo_vanilla", "data": "grpo_frontier.jsonl"},
            ),
            "optimizer_config": {"framework": DEFAULT_OPTIMIZER_FRAMEWORK},
            "resource_topology": topology.model_dump(mode="json"),
            "price_snapshot": price.model_dump(mode="json"),
            "costs": [cost.model_dump(mode="json") for cost in costs],
            "scores": {"greedy_exact_accuracy@1": 0.5},
            "promotion_suite_id": "gsm8k-local-smoke",
            "promotion_suite_version": "2026-06-16",
            "num_times_suite_tested": 1,
            "num_candidates_evaluated": 1,
            "accepted_promotion_count": 0,
            "train_and_promotion_overlap_count": 0,
        },
    )
    return plan, _LocalGSM8KSmokeAdapter(cfg=cfg, config_path=config_path)


class _LocalGSM8KSmokeAdapter:
    """Deterministic fixture adapter behind the canonical RunEngine seam."""

    def __init__(self, *, cfg: _SmokeConfig, config_path: Path) -> None:
        self.cfg = cfg
        self.config_path = config_path

    def execute_stage(
        self,
        stage: RunStage,
        plan: RunPlan,
        prior: dict[str, StageOutput],
    ) -> StageOutput:
        if stage == "promote":
            raise ValueError("RunEngine owns the promote stage")
        handlers = {
            "prepare": self._prepare,
            "data": self._data,
            "evidence": self._evidence,
            "train": self._train,
            "select": self._select,
            "evaluate": self._evaluate,
            "finalize": self._finalize,
        }
        return handlers[stage](plan, prior)

    def _prepare(
        self,
        plan: RunPlan,
        _prior: dict[str, StageOutput],
    ) -> StageOutput:
        path = Path(plan.output_dir) / "run_plan.json"
        _write_json(path, plan.model_dump(mode="json"))
        return _local_stage_output(artifacts={"run_plan": path})

    def _data(
        self,
        plan: RunPlan,
        _prior: dict[str, StageOutput],
    ) -> StageOutput:
        run_dir = Path(plan.output_dir)
        train_examples, promotion_examples = _fixture_examples()
        splits_path = run_dir / "splits.json"
        splits = _write_splits(
            splits_path,
            train_examples=train_examples,
            promotion_examples=promotion_examples,
            seed=self.cfg.seed,
        )
        overlap_report = certify_no_training_promotion_overlap(
            [
                Example(
                    id=example.id,
                    source=example.source,
                    prompt=format_prompt(example.question, self.cfg.prompt_style),
                    final_answer=example.gold_answer,
                )
                for example in train_examples
            ],
            _promotion_manifest(
                promotion_examples,
                prompt_style=self.cfg.prompt_style,
                split_hash=splits.split_hash,
            ),
        )
        overlap_path = run_dir / "data_overlap_report.json"
        _write_json(overlap_path, overlap_report.to_dict())
        return _local_stage_output(
            artifacts={"splits": splits_path, "data_overlap_report": overlap_path},
            values={"split_hash": splits.split_hash},
        )

    def _evidence(
        self,
        plan: RunPlan,
        _prior: dict[str, StageOutput],
    ) -> StageOutput:
        run_dir = Path(plan.output_dir)
        train_examples, _ = _fixture_examples()
        generation_config = {
            "backend": "deterministic_fixture",
            "temperature": 0.0,
            "top_p": 1.0,
            "max_new_tokens": self.cfg.max_new_tokens,
            "greedy": False,
            "seed": self.cfg.seed,
        }
        probe_rows = probe_gsm8k_examples(
            train_examples,
            _fixture_probe_generation,
            run_id=f"{self.cfg.run_id}-probe",
            bucket_source_split="train_pool",
            model_id=self.cfg.model_id,
            checkpoint_id="seed",
            prompt_style=self.cfg.prompt_style,
            generation_config=generation_config,
            early_exit=EarlyExitConfig(
                g_total=self.cfg.rollouts,
                g_early=self.cfg.early_rollouts,
            ),
            seed=self.cfg.seed,
        )
        probe_path = run_dir / "probe.jsonl"
        write_probe_jsonl(probe_rows, probe_path)
        serialized = [row.to_json() for row in probe_rows]
        traces = _trace_records_from_probe_rows(
            serialized,
            task_id="gsm8k",
            source_checkpoint="seed",
        )
        traces_path = run_dir / "traces.jsonl"
        write_jsonl(
            traces_path,
            (trace.model_dump(mode="json") for trace in traces),
        )
        rollout_groups_path = run_dir / "rollout_groups.jsonl"
        write_jsonl(
            rollout_groups_path,
            (
                group.model_dump(mode="json")
                for group in _rollout_groups_from_probe_rows(serialized, traces)
            ),
        )
        buckets = bucket_probe_artifact(serialized)
        difficulty_path = run_dir / "difficulty_map.json"
        _write_json(difficulty_path, [bucket.to_json() for bucket in buckets])
        curriculum_dir = run_dir / "curriculum"
        curriculum = build_gsm8k_curriculum(
            {example.id: example for example in train_examples},
            buckets,
            serialized,
            source_probe_run_id=f"{self.cfg.run_id}-probe",
            prompt_style=self.cfg.prompt_style,
        )
        write_gsm8k_curriculum(curriculum, curriculum_dir)
        views = _write_training_views(
            run_dir / "training_views",
            run_id=self.cfg.run_id,
            curriculum_dir=curriculum_dir,
            curriculum=curriculum,
        )
        return _local_stage_output(
            artifacts={
                "probe": probe_path,
                "traces": traces_path,
                "rollout_groups": rollout_groups_path,
                "difficulty_map": difficulty_path,
                "grpo_frontier": curriculum_dir / "grpo_frontier.jsonl",
                "easy_regression": curriculum_dir / "easy_regression.jsonl",
                "opsd_hard": curriculum_dir / "opsd_hard.jsonl",
                **views,
            },
        )

    def _train(
        self,
        plan: RunPlan,
        _prior: dict[str, StageOutput],
    ) -> StageOutput:
        checkpoint_dir = Path(plan.output_dir) / "checkpoints" / plan.candidate_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = checkpoint_dir / "checkpoint_manifest.json"
        _write_json(
            path,
            {
                "checkpoint_type": "dry_run_smoke_manifest",
                "method": "grpo_vanilla",
                "model": self.cfg.model_id,
                "optimizer": {"framework": DEFAULT_OPTIMIZER_FRAMEWORK},
                "data": str(Path(plan.output_dir) / "curriculum" / "grpo_frontier.jsonl"),
                "status": "validated_not_trained",
            },
        )
        return _local_stage_output(artifacts={"checkpoint_manifest": path})

    def _select(
        self,
        plan: RunPlan,
        _prior: dict[str, StageOutput],
    ) -> StageOutput:
        path = Path(plan.output_dir) / "checkpoint_selection.json"
        _write_json(
            path,
            {
                "candidate_id": plan.candidate_id,
                "rule": "single_validated_candidate",
                "selected": True,
            },
        )
        return _local_stage_output(artifacts={"checkpoint_selection": path})

    def _evaluate(
        self,
        plan: RunPlan,
        _prior: dict[str, StageOutput],
    ) -> StageOutput:
        _, promotion_examples = _fixture_examples()
        baseline = _fixture_eval_artifact(
            promotion_examples,
            artifact_id="seed",
            model_id=self.cfg.model_id,
        )
        candidate = _fixture_eval_artifact(
            promotion_examples,
            artifact_id=plan.candidate_id,
            model_id=self.cfg.model_id,
        )
        baseline_path = Path(plan.output_dir) / "eval" / "baseline.json"
        candidate_path = Path(plan.output_dir) / "eval" / "candidate.json"
        _write_json(baseline_path, baseline.to_dict())
        _write_json(candidate_path, candidate.to_dict())
        return _local_stage_output(
            artifacts={
                "baseline_eval": baseline_path,
                "candidate_eval": candidate_path,
            },
        )

    def _finalize(
        self,
        plan: RunPlan,
        prior: dict[str, StageOutput],
    ) -> StageOutput:
        run_dir = Path(plan.output_dir)
        baseline = load_eval_artifact(prior["evaluate"].artifacts["baseline_eval"])
        candidate = load_eval_artifact(prior["evaluate"].artifacts["candidate_eval"])
        decision = _read_json(prior["promote"].artifacts["promotion_decision"])
        overlap = _read_json(prior["data"].artifacts["data_overlap_report"])
        split_hash = str(prior["data"].values["split_hash"])
        checkpoint_dir = run_dir / "checkpoints" / plan.candidate_id
        promoted = decision["decision"] == "promote"
        rejection_reasons = tuple(str(item) for item in decision["rejection_reasons"])
        lifecycle = CheckpointLifecycleManager(
            ModelLifecycleConfig(
                artifact_dir=run_dir / "lifecycle",
                discard_rejected_local=False,
                hf=HuggingFaceLifecycleConfig(enabled=False),
            ),
            date_provider=lambda: "2026-06-16",
        ).finalize(
            CheckpointLifecycleInput(
                candidate_id=plan.candidate_id,
                checkpoint_ref=str(checkpoint_dir),
                task_name=plan.task_name,
                parent_candidate_id=plan.parent_candidate_id or "seed",
                parent_checkpoint_ref="seed",
                previous_incumbent_candidate_id="seed",
                previous_incumbent_checkpoint_ref="seed",
                previous_incumbent_remote_ref=None,
                promoted=promoted,
                score=candidate.metrics[candidate.primary_metric],
                incumbent_score=baseline.metrics[baseline.primary_metric],
                metrics=dict(candidate.metrics),
                evaluation_artifacts={
                    "baseline_eval": prior["evaluate"].artifacts["baseline_eval"],
                    "candidate_eval": prior["evaluate"].artifacts["candidate_eval"],
                    "promotion_eval_artifact": candidate.to_dict(),
                },
                evaluation_metadata={"split": "dev_promotion", "split_hash": split_hash},
                train_artifacts={
                    "grpo_frontier": prior["evidence"].artifacts["grpo_frontier"],
                },
                train_metrics={"train_loss": 0.0},
                train_metadata={
                    "costs": {"estimated_usd": 0.0},
                    "train_and_promotion_overlap_count": 0,
                },
                promotion_gate={
                    "decision": decision["decision"],
                    "reasons": list(rejection_reasons),
                },
                promotion_decision=decision,
                data_overlap_report=overlap,
                severity_summary=dict(decision["severity_summary"]),
                rejection_reason="; ".join(rejection_reasons) or None,
            ),
        )
        registry_path = run_dir / "checkpoint_registry.jsonl"
        records = (
            CheckpointRecord(
                candidate_id="seed",
                path="seed",
                parent_id=None,
                score=baseline.metrics[baseline.primary_metric],
                metrics=dict(baseline.metrics),
                promoted=True,
            ),
            CheckpointRecord(
                candidate_id=plan.candidate_id,
                path=str(checkpoint_dir),
                parent_id=plan.parent_candidate_id,
                score=candidate.metrics[candidate.primary_metric],
                metrics=dict(candidate.metrics),
                promoted=promoted,
                rejection_reason=None if promoted else "; ".join(rejection_reasons),
                artifacts={"lifecycle": lifecycle.local_artifacts},
                metadata={
                    "promotion_decision": decision,
                    "lifecycle_evidence_path": str(lifecycle.evidence_path),
                },
                local_state=lifecycle.local_state,
            ),
        )
        write_jsonl(registry_path, (asdict(record) for record in records))
        ledger_path = run_dir / "ledger.jsonl"
        ledger = make_run_ledger_entry(
            run_id=plan.run_id,
            command=["pte", "run", "--config", str(self.config_path)],
            base_model=plan.model_id,
            checkpoint=str(checkpoint_dir),
            dataset="local_gsm8k_fixture",
            dataset_revision="2026-06-16",
            split_hash=split_hash,
            prompt_template=self.cfg.prompt_style,
            reward_config={"verifier": "gsm8k_numeric_v1"},
            trainer_config={"method": "grpo_vanilla", "dry_run": True},
            seeds=[self.cfg.seed],
            artifact_paths={
                "splits": prior["data"].artifacts["splits"],
                "probe": prior["evidence"].artifacts["probe"],
                "difficulty_map": prior["evidence"].artifacts["difficulty_map"],
                "promotion_decision": prior["promote"].artifacts["promotion_decision"],
            },
        )
        write_jsonl(ledger_path, (ledger.to_json(),))
        return _local_stage_output(
            artifacts={
                "checkpoint_registry": registry_path,
                "ledger": ledger_path,
                "lifecycle": lifecycle.evidence_path,
            },
            values={"decision": decision["decision"]},
        )


def _local_stage_output(
    *,
    artifacts: Mapping[str, Path],
    values: Mapping[str, Any] | None = None,
) -> StageOutput:
    return StageOutput(
        artifacts={name: str(path) for name, path in artifacts.items()},
        values=dict(values or {}),
        cost_usd=0.0,
    )


def _read_json(path: str | Path) -> dict[str, Any]:
    body = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(body, dict):
        raise ValueError(f"expected JSON object: {path}")
    return body


def _fixture_examples() -> tuple[list[GSM8KExample], list[GSM8KExample]]:
    train = [
        _gsm8k_example(0, "If Ana has 2 apples and buys 3, how many apples?", "5"),
        _gsm8k_example(1, "What is 4 plus 4?", "8"),
        _gsm8k_example(2, "What is 5 plus 6?", "11"),
    ]
    promotion = [
        _gsm8k_example(3, "What is 7 minus 1?", "6"),
        _gsm8k_example(4, "What is 3 times 4?", "12"),
    ]
    return train, promotion


def _gsm8k_example(index: int, question: str, answer: str) -> GSM8KExample:
    return GSM8KExample(
        id=f"gsm8k/train/{index:06d}",
        split="train",
        question=question,
        gold_solution=f"fixture solution #### {answer}",
        gold_answer=answer,
        source="local_gsm8k_fixture",
        metadata={"dataset_revision": "2026-06-16"},
    )


def _fixture_probe_generation(
    example: GSM8KExample,
    rollout_id: int,
    _prompt: str,
    _config: Mapping[str, Any],
) -> str:
    correct_by_example = {
        "gsm8k/train/000000": rollout_id in {0, 2},
        "gsm8k/train/000001": True,
        "gsm8k/train/000002": False,
    }
    answer = example.gold_answer if correct_by_example[example.id] else "999"
    return f"Reasoning trace {rollout_id}. <answer>{answer}</answer>"


def _fixture_eval_artifact(
    examples: Sequence[GSM8KExample],
    *,
    artifact_id: str,
    model_id: str,
) -> EvalArtifact:
    results = (
        EvalExampleResult(
            example_id=examples[0].id,
            correct=True,
            parse_ok=True,
            tokens=4,
            bucket="easy_stable",
        ),
        EvalExampleResult(
            example_id=examples[1].id,
            correct=False,
            parse_ok=True,
            tokens=4,
            bucket="easy_stable",
        ),
    )
    accuracy = sum(row.correct for row in results) / len(results)
    return EvalArtifact(
        artifact_id=artifact_id,
        primary_metric="greedy_exact_accuracy@1",
        examples=results,
        metrics={
            "greedy_exact_accuracy@1": accuracy,
            "parse_success@1": 1.0,
            "mean_tokens": 4.0,
        },
        slices={"easy_stable": {"greedy_exact_accuracy@1": accuracy}},
        metadata={"model_id": model_id, "split": "dev_promotion"},
    )


def _trace_records_from_probe_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    task_id: str,
    source_checkpoint: str,
) -> tuple[TraceRecord, ...]:
    return tuple(
        TraceRecord(
            trace_id=str(row["trace_id"]),
            run_id=str(row["run_id"]),
            task_id=task_id,
            example_id=str(row["example_id"]),
            split_role="probe",
            prompt_hash=str(row["prompt_hash"]),
            source_checkpoint=source_checkpoint,
            policy_version=source_checkpoint,
            policy_step=0,
            policy_step_evidence="static",
            rollout_group_id=f"{row['example_id']}:rollouts",
            generation_backend=str(dict(row["generation_config"]).get("backend", "unknown")),
            sampling_config=dict(row["generation_config"]),
            verifier_id="gsm8k_numeric_v1",
            reward_components={"gsm8k_reward": float(row["reward"])},
            token_counts={"completion": float(row["completion_tokens"])},
        )
        for row in rows
    )


def _rollout_groups_from_probe_rows(
    rows: Sequence[Mapping[str, Any]],
    trace_records: Sequence[TraceRecord],
) -> tuple[Any, ...]:
    trace_by_id = {trace.trace_id: trace for trace in trace_records}
    rows_by_example: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        rows_by_example.setdefault(str(row["example_id"]), []).append(row)
    return tuple(
        build_rollout_group(
            group_id=f"{example_id}:rollouts",
            traces=tuple(trace_by_id[str(row["trace_id"])] for row in grouped_rows),
            rewards=tuple(float(row["reward"]) for row in grouped_rows),
        )
        for example_id, grouped_rows in sorted(rows_by_example.items())
    )


def _write_splits(
    path: Path,
    *,
    train_examples: Sequence[GSM8KExample],
    promotion_examples: Sequence[GSM8KExample],
    seed: int,
) -> GSM8KSplits:
    body = {
        "dataset": "local_gsm8k_fixture",
        "dataset_revision": "2026-06-16",
        "seed": seed,
        "splits": {
            "train_pool": [example.id for example in train_examples],
            "dev_promotion": [example.id for example in promotion_examples],
            "dev_diagnostics": [],
            "official_val": [],
        },
    }
    split_hash = stable_hash(body)
    splits = GSM8KSplits(
        dataset=body["dataset"],
        dataset_revision=body["dataset_revision"],
        seed=seed,
        splits=body["splits"],
        split_hash=split_hash,
    )
    _write_json(path, splits.to_dict())
    return splits


def _promotion_manifest(
    examples: Sequence[GSM8KExample],
    *,
    prompt_style: str,
    split_hash: str,
) -> EvalSplitManifest:
    return EvalSplitManifest(
        suite_id="gsm8k-local-smoke",
        suite_version="2026-06-16",
        role="promotion",
        example_ids=tuple(example.id for example in examples),
        example_id_hash=split_hash,
        prompt_hashes={
            example.id: prompt_sha256(format_prompt(example.question, prompt_style))
            for example in examples
        },
        source_dataset="local_gsm8k_fixture",
        source_revision="2026-06-16",
    )


def _load_config(path: str | Path) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("run config root must be a mapping")
    return raw


def _write_json(path: Path, body: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(body), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    return value


def _positive_int(value: object, name: str) -> int:
    if type(value) is bool or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_or_zero_int(value: object, name: str) -> int:
    if type(value) is bool or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value
