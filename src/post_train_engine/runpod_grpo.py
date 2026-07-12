"""Manual-RunPod GSM8K GRPO hill-climb path.

This module is intentionally concrete. It is for the first paid RunPod
experiment where the user opens a pod, installs the repo, and runs
``python -m post_train_engine.cli hillclimb --config ...`` inside that pod.
It does not submit pods or hide provider state behind an API client.
"""

from __future__ import annotations

import inspect
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from post_train_engine.artifacts import require_valid_run_bundle
from post_train_engine.artifact_store import ArtifactStore
from post_train_engine.config import HuggingFaceLifecycleConfig, ModelLifecycleConfig
from post_train_engine.engine import (
    CampaignBinding,
    RunEngine,
    RunPlan,
    RunStage,
    StageOutput,
    require_nonfailed_manifest,
)
from post_train_engine.evals.promotion import (
    EvalArtifact,
    EvalExampleResult,
)
from post_train_engine.evals.contract import EvalContract
from post_train_engine.evidence_safety import (
    VerifierSeparation,
    certify_content_separation,
)
from post_train_engine.evaluation_roles import EvaluationRoles
from post_train_engine.hub_identity import (
    is_huggingface_commit,
    resolve_huggingface_revision,
)
from post_train_engine.jsonl import read_jsonl, write_jsonl
from post_train_engine.lifecycle import (
    CheckpointLifecycleInput,
    CheckpointLifecycleManager,
    HuggingFaceCheckpointUploader,
)
from post_train_engine.rewards.gsm8k import GSM8KRewardConfig, compute_gsm8k_reward
from post_train_engine.runpod import cuda_version_from_image, validate_cuda_runtime
from post_train_engine.runtime_evidence import (
    PhaseCostRecord,
    measure_runtime_pair,
    summarize_costs,
)
from post_train_engine.tasks.gsm8k import (
    GSM8KExample,
    format_prompt,
    load_gsm8k,
    parse_model_answer,
    verify_answer,
)
from post_train_engine.traces.schema import TraceRecord, stable_prompt_hash
from post_train_engine.traces.rollouts import build_rollout_group
from post_train_engine.traces.store import JsonlTraceStore
from post_train_engine.training_views import (
    TrainingViewArtifact,
    build_training_view_artifact,
    write_training_view_artifact,
)


_FROZEN_FORBID = ConfigDict(frozen=True, extra="forbid")
_OPTIONAL_GRPO_CONFIG_FIELDS = {
    "generation_kwargs",
    "use_vllm",
    "vllm_gpu_memory_utilization",
}
_STABLE_GENERATION_KWARGS = {
    "remove_invalid_values": True,
    "renormalize_logits": True,
}


class ManualRunPodExecution(BaseModel):
    model_config = _FROZEN_FORBID

    mode: Literal["runpod_manual_grpo"]
    provider: Literal["runpod"] = "runpod"
    gpu_type: str = Field(..., min_length=1)
    gpu_count: int = Field(default=1, gt=0)
    container_image: str = Field(..., min_length=1)
    disk_gb: int = Field(default=100, ge=50)
    volume_gb: int = Field(default=150, ge=0)
    accelerator_hour_usd: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def require_cuda_version_in_image(self) -> ManualRunPodExecution:
        cuda_version_from_image(self.container_image)
        return self

    @property
    def cuda_version(self) -> str:
        """RunPod allocation filter derived from the pinned image tag."""

        return cuda_version_from_image(self.container_image)


class RunSpec(BaseModel):
    model_config = _FROZEN_FORBID

    run_id: str = Field(..., min_length=1)
    certification_mode: Literal["non_certifying_smoke", "certifying"]
    campaign: CampaignBinding | None = None
    output_dir: str = Field(..., min_length=1)
    seed: int = Field(default=42, ge=0)
    overwrite: bool = False
    distributed_timeout_seconds: float = Field(default=120.0, gt=0.0)

    @model_validator(mode="after")
    def _certification_has_authority(self) -> RunSpec:
        if self.certification_mode == "certifying" and self.campaign is None:
            raise ValueError("certifying run requires campaign binding")
        if self.certification_mode == "non_certifying_smoke" and self.campaign is not None:
            raise ValueError("non-certifying smoke cannot bind a campaign")
        return self


class ModelSpec(BaseModel):
    model_config = _FROZEN_FORBID

    base_model_id: str = Field(default="Qwen/Qwen2.5-0.5B-Instruct", min_length=1)
    revision: str = Field(default="main", min_length=1)
    resolved_revision: str | None = None
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    trust_remote_code: bool = False
    use_safetensors: bool = True
    attn_implementation: str | None = "eager"
    gradient_checkpointing: bool = True


class DatasetSpec(BaseModel):
    model_config = _FROZEN_FORBID

    source: Literal["huggingface_gsm8k", "embedded_gsm8k_tiny"] = "huggingface_gsm8k"
    dataset_name: str = "openai/gsm8k"
    revision: str = Field(default="main", min_length=1)
    resolved_revision: str | None = None
    train_size: int = Field(..., gt=0)
    selection_size: int = Field(..., gt=0)
    eval_size: int = Field(..., gt=0)
    split_seed: int = Field(default=42, ge=0)
    prompt_style: Literal["plain", "chat", "thinking_tags"] = "thinking_tags"


class GRPOTrainingSpec(BaseModel):
    model_config = _FROZEN_FORBID

    method: Literal["grpo"] = "grpo"
    max_steps: int = Field(..., gt=0)
    learning_rate: float = Field(default=5.0e-7, gt=0.0)
    per_device_train_batch_size: int = Field(default=2, gt=0)
    gradient_accumulation_steps: int = Field(default=8, gt=0)
    num_generations: int = Field(default=2, ge=2)
    max_completion_length: int = Field(default=256, gt=0)
    temperature: float = Field(default=0.8, gt=0.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    beta: float = Field(default=0.02, ge=0.0)
    save_steps: int = Field(default=50, gt=0)
    logging_steps: int = Field(default=5, gt=0)
    save_total_limit: int = Field(default=3, gt=0)
    report_to: Literal["none", "wandb"] = "none"
    run_name: str | None = None
    use_vllm: bool = False
    vllm_gpu_memory_utilization: float = Field(default=0.3, gt=0.0, le=1.0)
    parse_bonus: float = Field(default=0.02, ge=0.0)
    length_penalty_weight: float = Field(default=0.05, ge=0.0)

    @model_validator(mode="after")
    def _save_steps_not_after_training(self) -> GRPOTrainingSpec:
        if self.save_steps > self.max_steps:
            raise ValueError("training.save_steps must be <= training.max_steps")
        return self

    def reward_config(self) -> GSM8KRewardConfig:
        return GSM8KRewardConfig(
            parse_bonus=self.parse_bonus,
            length_penalty_weight=self.length_penalty_weight,
            max_new_tokens=self.max_completion_length,
        )


class EvalSpec(BaseModel):
    model_config = _FROZEN_FORBID

    max_new_tokens: int = Field(default=256, gt=0)
    temperature: float = Field(default=0.0, ge=0.0)
    do_sample: bool = False
    batch_size: int = Field(default=1, gt=0)


class PromotionSpec(BaseModel):
    model_config = _FROZEN_FORBID

    min_eval_examples: int = Field(default=8, gt=0)
    min_accuracy_delta: float = 0.0
    min_paired_delta_ci_low: float = -1.0
    max_mcnemar_p: float = 1.0
    max_parse_regression: float = 0.0
    max_easy_regression: float = 0.0
    max_token_increase_ratio: float = 1.25

    @field_validator(
        "min_accuracy_delta",
        "min_paired_delta_ci_low",
        "max_mcnemar_p",
        "max_parse_regression",
        "max_easy_regression",
        "max_token_increase_ratio",
    )
    @classmethod
    def _finite(cls, value: float) -> float:
        if type(value) is bool or not isinstance(value, int | float) or not math.isfinite(value):
            raise ValueError("promotion numeric fields must be finite")
        return float(value)


class TraceCaptureSpec(BaseModel):
    model_config = _FROZEN_FORBID

    enabled: bool = True
    train_trace_dir: str = "rollouts/train"


class CheckpointSelectionSpec(BaseModel):
    model_config = _FROZEN_FORBID

    enabled: bool = True
    metric: Literal["accuracy", "parse_success_rate"] = "accuracy"
    include_final: bool = True


class HFUploadSpec(BaseModel):
    model_config = _FROZEN_FORBID

    enabled: bool = False
    repo_id: str | None = Field(default=None, min_length=1)
    repo_type: Literal["model", "dataset", "space"] = "model"
    private: bool = True
    token_env: str = "PTE_REMOTE_HF_WRITE"
    path_template: str = "tasks/{task}/checkpoints/{date}/{candidate_id}"
    upload_evidence: bool = True
    upload_promoted_checkpoints: bool = True
    upload_rejected_checkpoints: bool = False

    @model_validator(mode="after")
    def _enabled_requires_repo(self) -> HFUploadSpec:
        if self.enabled and not self.repo_id:
            raise ValueError("hf_upload.repo_id is required when hf_upload.enabled=true")
        required_fields = ("{task}", "{date}", "{candidate_id}")
        missing = [field for field in required_fields if field not in self.path_template]
        if missing:
            raise ValueError(f"hf_upload.path_template must include {', '.join(missing)}")
        return self


class RunPodGRPOConfig(BaseModel):
    model_config = _FROZEN_FORBID

    schema_version: Literal["runpod_grpo_hillclimb_v1"]
    execution: ManualRunPodExecution
    run: RunSpec
    model: ModelSpec = Field(default_factory=ModelSpec)
    dataset: DatasetSpec
    training: GRPOTrainingSpec
    eval: EvalSpec = Field(default_factory=EvalSpec)
    promotion: PromotionSpec = Field(default_factory=PromotionSpec)
    trace_capture: TraceCaptureSpec = Field(default_factory=TraceCaptureSpec)
    checkpoint_selection: CheckpointSelectionSpec = Field(default_factory=CheckpointSelectionSpec)
    hf_upload: HFUploadSpec = Field(default_factory=HFUploadSpec)

    def training_reward_config(self) -> GSM8KRewardConfig:
        return self.training.reward_config()


@dataclass(frozen=True)
class EvalRow:
    example_id: str
    prompt: str
    completion: str
    parsed_answer: str | None
    gold_answer: str
    correct: bool
    parse_ok: bool
    completion_tokens: int
    sample_index: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "prompt": self.prompt,
            "completion": self.completion,
            "parsed_answer": self.parsed_answer,
            "gold_answer": self.gold_answer,
            "correct": self.correct,
            "parse_ok": self.parse_ok,
            "completion_tokens": self.completion_tokens,
            "sample_index": self.sample_index,
        }


@dataclass(frozen=True)
class DistributedContext:
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0

    @classmethod
    def from_env(cls) -> DistributedContext:
        return cls(
            world_size=_positive_env_int("WORLD_SIZE", default=1),
            rank=_nonnegative_env_int("RANK", default=0),
            local_rank=_nonnegative_env_int("LOCAL_RANK", default=0),
        )

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    def to_json(self) -> dict[str, int | bool]:
        return {
            "world_size": self.world_size,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "is_distributed": self.is_distributed,
            "is_main_process": self.is_main_process,
        }


def is_runpod_grpo_config(path: str | Path) -> bool:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return isinstance(raw, dict) and raw.get("schema_version") == "runpod_grpo_hillclimb_v1"


def load_runpod_grpo_config(path: str | Path) -> RunPodGRPOConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("RunPod GRPO hillclimb config root must be a mapping")
    return RunPodGRPOConfig.model_validate(raw)


def _resolve_hub_revisions(cfg: RunPodGRPOConfig) -> RunPodGRPOConfig:
    model_revision = resolve_huggingface_revision(
        cfg.model.base_model_id,
        kind="model",
        requested_revision=cfg.model.revision,
    )
    dataset_revision = (
        None
        if cfg.dataset.source == "embedded_gsm8k_tiny"
        else resolve_huggingface_revision(
            cfg.dataset.dataset_name,
            kind="dataset",
            requested_revision=cfg.dataset.revision,
        )
    )
    return cfg.model_copy(
        update={
            "model": cfg.model.model_copy(
                update={"resolved_revision": model_revision}
            ),
            "dataset": cfg.dataset.model_copy(
                update={"resolved_revision": dataset_revision}
            ),
        }
    )


def run_runpod_grpo_hillclimb(config_path: str | Path) -> dict[str, Any]:
    cfg = load_runpod_grpo_config(config_path)
    dist = DistributedContext.from_env()
    _validate_launch_topology(cfg, dist)
    _validate_grpo_runtime_shape(cfg.training, world_size=dist.world_size)
    if dist.is_main_process:
        _validate_hf_upload_config(cfg)
    state = _distributed_state(dist)
    run_dir = Path(cfg.run.output_dir)
    store: ArtifactStore | None = None
    resumable = False
    if dist.is_main_process:
        resumable = run_dir.is_dir() and (
            (run_dir / "manifest.json").is_file() or (run_dir / "state").is_dir()
        )
        store = (
            ArtifactStore(run_dir, resume=True)
            if resumable
            else ArtifactStore(run_dir, overwrite=cfg.run.overwrite)
        )
    _wait_for_everyone(state, timeout_seconds=cfg.run.distributed_timeout_seconds)
    if (run_dir / "manifest.json").is_file():
        bundle = require_valid_run_bundle(run_dir)
        require_nonfailed_manifest(bundle.manifest, run_dir)
        return _read_json_object(run_dir / "final_report.json")

    if store is not None and not resumable:
        _event(
            store,
            "run_started",
            {"config": str(config_path), "distributed": dist.to_json()},
        )
        store.copy_file(config_path, "config.raw.yaml")
        store.write_json("config.resolved.json", cfg.model_dump(mode="json"))
        store.write_text("command.txt", " ".join(sys.argv) + "\n")
        store.write_json("environment.json", runtime_environment(dist))
    _wait_for_everyone(state, timeout_seconds=cfg.run.distributed_timeout_seconds)
    _require_cuda(cfg)
    if store is not None and not resumable:
        cfg = _resolve_hub_revisions(cfg)
        store.write_json("config.resolved.json", cfg.model_dump(mode="json"))
    elif store is not None:
        cfg = RunPodGRPOConfig.model_validate(
            _read_json_object(run_dir / "config.resolved.json")
        )
    _wait_for_everyone(state, timeout_seconds=cfg.run.distributed_timeout_seconds)
    if store is None:
        cfg = RunPodGRPOConfig.model_validate(
            _read_json_object(run_dir / "config.resolved.json")
        )

    train_examples, selection_examples, promotion_examples = _load_and_split_dataset(cfg)
    roles = EvaluationRoles(
        selection_example_ids=tuple(row.id for row in selection_examples),
        promotion_example_ids=tuple(row.id for row in promotion_examples),
    )
    roles.require_training_eligible(row.id for row in train_examples)
    plan = _compile_runpod_plan(
        cfg,
        dist=dist,
        train_examples=train_examples,
        selection_examples=selection_examples,
        promotion_examples=promotion_examples,
    )
    adapter = _RunPodGRPOAdapter(
        cfg=cfg,
        config_path=Path(config_path),
        dist=dist,
        store=store,
        train_examples=train_examples,
        selection_examples=selection_examples,
        promotion_examples=promotion_examples,
    )
    coordinator = (
        _AccelerateRunCoordinator(dist, state)
        if dist.is_distributed
        else None
    )
    execution = RunEngine().execute(plan, adapter, coordinator=coordinator)
    require_nonfailed_manifest(execution.manifest, run_dir)
    return _read_json_object(run_dir / "final_report.json")

class _AccelerateRunCoordinator:
    def __init__(self, dist: DistributedContext, state: Any) -> None:
        self.is_main_process = dist.is_main_process
        self._state = state

    def wait(self, timeout_seconds: float) -> None:
        from datetime import timedelta

        import torch.distributed as torch_dist

        work = torch_dist.barrier(async_op=True)
        if not work.wait(timeout=timedelta(seconds=timeout_seconds)):
            raise TimeoutError("distributed stage barrier timed out")

    def collect_errors(
        self,
        error: str | None,
        timeout_seconds: float,
    ) -> tuple[str, ...]:
        import torch.distributed as torch_dist

        self.wait(timeout_seconds)
        world_size = int(self._state.num_processes)
        gathered: list[str | None] = [None for _ in range(world_size)]
        torch_dist.all_gather_object(gathered, error)
        return tuple(item for item in gathered if item is not None)


def _compile_runpod_plan(
    cfg: RunPodGRPOConfig,
    *,
    dist: DistributedContext,
    train_examples: Sequence[GSM8KExample],
    selection_examples: Sequence[GSM8KExample],
    promotion_examples: Sequence[GSM8KExample],
) -> RunPlan:
    model_exact = is_huggingface_commit(cfg.model.resolved_revision)
    dataset_revision = (
        "embedded-gsm8k-tiny-v1"
        if cfg.dataset.source == "embedded_gsm8k_tiny"
        else cfg.dataset.resolved_revision
    )
    dataset_exact = (
        cfg.dataset.source == "embedded_gsm8k_tiny"
        or is_huggingface_commit(dataset_revision)
    )
    return RunPlan(
        certification_mode=cfg.run.certification_mode,
        run_id=cfg.run.run_id,
        candidate_id=f"{cfg.run.run_id}-grpo-candidate",
        parent_candidate_id="baseline",
        task_name="gsm8k",
        model_id=cfg.model.base_model_id,
        output_dir=cfg.run.output_dir,
        distributed_timeout_seconds=cfg.run.distributed_timeout_seconds,
        source_root=str(Path(__file__).resolve().parents[2]),
        inputs={
            "model": {
                "kind": "model",
                "requested_id": cfg.model.base_model_id,
                "resolved_id": cfg.model.base_model_id,
                "requested_revision": cfg.model.revision,
                "resolved_revision": cfg.model.resolved_revision,
                "resolution_state": "exact" if model_exact else "provider_managed",
                "non_certifying_reason": (
                    None if model_exact else "model revision did not resolve to a commit"
                ),
            },
            "dataset": {
                "kind": "dataset",
                "requested_id": cfg.dataset.dataset_name,
                "resolved_id": cfg.dataset.dataset_name,
                "requested_revision": cfg.dataset.revision,
                "resolved_revision": dataset_revision,
                "resolution_state": "exact" if dataset_exact else "provider_managed",
                "non_certifying_reason": (
                    None if dataset_exact else "dataset revision did not resolve to a commit"
                ),
            },
        },
        training_example_ids=tuple(row.id for row in train_examples),
        selection_example_ids=tuple(row.id for row in selection_examples),
        promotion_example_ids=tuple(row.id for row in promotion_examples),
        evaluation_contract=EvalContract.from_components(
            suite_id="gsm8k-runpod-promotion",
            suite_version=(
                f"{cfg.dataset.dataset_name}:{dataset_revision}:"
                f"seed={cfg.dataset.split_seed}"
            ),
            example_ids=tuple(row.id for row in promotion_examples),
            example_content=tuple(
                {
                    "id": row.id,
                    "question": row.question,
                    "gold_answer": row.gold_answer,
                }
                for row in promotion_examples
            ),
            prompt_contract={"prompt_style": cfg.dataset.prompt_style},
            verifier_contract={"task": "gsm8k", "verifier": "exact-answer-v1"},
            generation_contract=cfg.eval.model_dump(mode="json"),
            primary_metric="accuracy",
        ),
        content_separation=certify_content_separation(
            training_texts=tuple(row.question for row in train_examples),
            protected_texts=tuple(
                row.question for row in (*selection_examples, *promotion_examples)
            ),
        ),
        verifier_separation=VerifierSeparation(
            verifier_kind="executable_ground_truth",
            training_verifier_id="gsm8k-exact-answer-v1",
            promotion_verifier_id="gsm8k-exact-answer-v1",
        ),
        promotion_gate={
            "min_examples": cfg.promotion.min_eval_examples,
            "min_primary_delta": max(cfg.promotion.min_accuracy_delta, 1e-12),
            "min_primary_ci_low": cfg.promotion.min_paired_delta_ci_low,
            "max_mcnemar_p": cfg.promotion.max_mcnemar_p,
            "max_parse_regression": cfg.promotion.max_parse_regression,
            "max_easy_regression": cfg.promotion.max_easy_regression,
            "max_token_increase_ratio": cfg.promotion.max_token_increase_ratio,
        },
        campaign=cfg.run.campaign,
        metadata={
            "execution_mode": "runpod_grpo",
            "execution": cfg.execution.model_dump(mode="json"),
            "distributed": {
                "world_size": dist.world_size,
                "is_distributed": dist.is_distributed,
            },
            "eval_sharding": "rank_modulo_example_order",
            "trace_capture": cfg.trace_capture.model_dump(mode="json"),
            "checkpoint_selection": cfg.checkpoint_selection.model_dump(mode="json"),
            "hf_upload": {
                **cfg.hf_upload.model_dump(mode="json"),
                "token_env": cfg.hf_upload.token_env,
                "token_present": bool(os.environ.get(cfg.hf_upload.token_env)),
            },
        },
    )


class _RunPodGRPOAdapter:
    def __init__(
        self,
        *,
        cfg: RunPodGRPOConfig,
        config_path: Path,
        dist: DistributedContext,
        store: ArtifactStore | None,
        train_examples: Sequence[GSM8KExample],
        selection_examples: Sequence[GSM8KExample],
        promotion_examples: Sequence[GSM8KExample],
    ) -> None:
        self.cfg = cfg
        self.config_path = config_path
        self.dist = dist
        self.store = store
        self.train_examples = list(train_examples)
        self.selection_examples = list(selection_examples)
        self.promotion_examples = list(promotion_examples)

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
        _plan: RunPlan,
        _prior: dict[str, StageOutput],
    ) -> StageOutput:
        if self.store is None:
            return _runpod_worker_output()
        self.store.copy_file(self.config_path, "config.raw.yaml")
        self.store.write_json("config.resolved.json", self.cfg.model_dump(mode="json"))
        self.store.write_text("command.txt", " ".join(sys.argv) + "\n")
        self.store.write_json("environment.json", runtime_environment(self.dist))
        baseline = {
            "candidate_id": "baseline",
            "model_id": self.cfg.model.base_model_id,
            "adapter_kind": "base",
        }
        self.store.write_json("candidates/baseline.json", baseline)
        _event(self.store, "preflight_passed", {"cuda": True})
        return _runpod_output(
            self.store,
            artifacts={
                "config_raw": "config.raw.yaml",
                "config_resolved": "config.resolved.json",
                "environment": "environment.json",
                "command": "command.txt",
                "baseline_candidate": "candidates/baseline.json",
            },
        )

    def _data(
        self,
        _plan: RunPlan,
        _prior: dict[str, StageOutput],
    ) -> StageOutput:
        if self.store is None:
            return _runpod_worker_output()
        _write_dataset_artifacts(
            self.store,
            self.cfg,
            self.train_examples,
            self.selection_examples,
            self.promotion_examples,
        )
        return _runpod_output(
            self.store,
            artifacts={
                "dataset_splits": "datasets/splits.json",
                "train_examples": "datasets/train.jsonl",
                "selection_examples": "datasets/selection.jsonl",
                "promotion_examples": "datasets/promotion.jsonl",
            },
        )

    def _evidence(
        self,
        _plan: RunPlan,
        _prior: dict[str, StageOutput],
    ) -> StageOutput:
        started = time.perf_counter()
        local_rows = _evaluate_hf_model(
            cfg=self.cfg,
            model_ref=self.cfg.model.base_model_id,
            examples=_shard_sequence(self.train_examples, self.dist),
            dist=self.dist,
            mode="training_probe",
            samples_per_example=self.cfg.training.num_generations,
        )
        rows = _gather_eval_rows(local_rows, self.dist)
        if self.store is None:
            return _runpod_worker_output()
        view = _write_measured_training_view(
            self.store,
            self.cfg,
            self.train_examples,
            rows,
        )
        artifacts = {
            "input_traces": "evidence/input_traces.jsonl",
            "rollout_groups": "evidence/rollout_groups.jsonl",
        }
        values: dict[str, Any]
        if view is None:
            artifacts["non_training_outcome"] = "evidence/non_training_outcome.json"
            values = {
                "training_eligible": False,
                "non_training_reason": "no measured parent-policy frontier examples",
            }
        else:
            artifacts.update(
                {
                    "selected_training_examples": "datasets/train_selected.jsonl",
                    "method_training_view": "evidence/method_training_view.json",
                }
            )
            values = {
                "training_eligible": True,
                "useful_trace_count": len(view.source_trace_ids),
            }
        return _runpod_output(
            self.store,
            artifacts=artifacts,
            values=values,
            phase_cost=_runpod_phase_cost(
                self.cfg,
                "parent_policy_evidence",
                time.perf_counter() - started,
            ),
        )

    def _train(
        self,
        _plan: RunPlan,
        prior: dict[str, StageOutput],
    ) -> StageOutput:
        if not bool(prior["evidence"].values["training_eligible"]):
            if self.store is None:
                return _runpod_worker_output()
            result = {
                "status": "skipped_no_learnable_evidence",
                "metrics": {},
                "distributed": self.dist.to_json(),
            }
            self.store.write_json("train/trainer_result.json", result)
            return _runpod_output(
                self.store,
                artifacts={"train_result": "train/trainer_result.json"},
                values={"training_outcome": result["status"]},
            )
        view = TrainingViewArtifact.model_validate_json(
            Path(prior["evidence"].artifacts["method_training_view"]).read_text(
                encoding="utf-8"
            )
        )
        started = time.perf_counter()
        train_result = _train_grpo(self.cfg, view, self.store, dist=self.dist)
        if self.store is None:
            return _runpod_worker_output()
        self.store.write_json("train/trainer_result.json", train_result)
        return _runpod_output(
            self.store,
            artifacts={"train_result": "train/trainer_result.json"},
            values={"training_outcome": "trained"},
            phase_cost=_runpod_phase_cost(
                self.cfg,
                "training",
                time.perf_counter() - started,
            ),
        )

    def _select(
        self,
        plan: RunPlan,
        prior: dict[str, StageOutput],
    ) -> StageOutput:
        trained = bool(prior["evidence"].values["training_eligible"])
        started = time.perf_counter()
        if trained:
            checkpoint_refs = _candidate_checkpoint_refs(Path(plan.output_dir))
            selection = _evaluate_and_select_checkpoints(
                cfg=self.cfg,
                checkpoint_refs=checkpoint_refs,
                eval_examples=self.selection_examples,
                dist=self.dist,
                store=self.store,
            )
            if self.store is None:
                return _runpod_worker_output()
            selected_path = str(selection["selected_checkpoint_path"])
        else:
            if self.store is None:
                return _runpod_worker_output()
            selected_path = self.cfg.model.base_model_id
            selection = {
                "selected_checkpoint_id": "baseline_no_training",
                "selected_checkpoint_path": selected_path,
                "selected_score": None,
                "selection_metric": self.cfg.checkpoint_selection.metric,
                "selection_reason": "no_learnable_evidence",
                "evaluated_checkpoints": [],
            }
            self.store.write_json("train/checkpoint_selection.json", selection)
        candidate = {
            "candidate_id": plan.candidate_id,
            "model_id": selected_path,
            "parent_id": "baseline",
            "adapter_kind": "full_model_checkpoint" if trained else "no_training_outcome",
            "training_method": "grpo",
            "checkpoint_selection": {
                key: value for key, value in selection.items() if key != "selected_eval_rows"
            },
        }
        self.store.write_json("candidates/candidate.json", candidate)
        return _runpod_output(
            self.store,
            artifacts={
                "checkpoint_selection": "train/checkpoint_selection.json",
                "candidate": "candidates/candidate.json",
            },
            values={
                "selected_checkpoint_path": selected_path,
                "trained": trained,
            },
            phase_cost=(
                _runpod_phase_cost(
                    self.cfg,
                    "checkpoint_selection",
                    time.perf_counter() - started,
                )
                if trained
                else None
            ),
        )

    def _evaluate(
        self,
        plan: RunPlan,
        prior: dict[str, StageOutput],
    ) -> StageOutput:
        started = time.perf_counter()
        baseline_local = _evaluate_hf_model(
            cfg=self.cfg,
            model_ref=self.cfg.model.base_model_id,
            examples=_shard_sequence(self.promotion_examples, self.dist),
            dist=self.dist,
        )
        baseline_rows = _gather_eval_rows(baseline_local, self.dist)
        candidate_local = _evaluate_hf_model(
            cfg=self.cfg,
            model_ref=str(prior["select"].values["selected_checkpoint_path"]),
            examples=_shard_sequence(self.promotion_examples, self.dist),
            dist=self.dist,
        )
        candidate_rows = _gather_eval_rows(candidate_local, self.dist)
        if self.store is None:
            return _runpod_worker_output()
        _write_eval_outputs(self.store, "baseline", baseline_rows)
        _write_eval_outputs(self.store, "candidate", candidate_rows)
        self.store.write_json(
            "evals/baseline.json",
            _runpod_promotion_artifact(
                "baseline",
                baseline_rows,
                evaluation_contract_hash=plan.evaluation_contract.contract_hash,
            ).to_dict(),
        )
        self.store.write_json(
            "evals/candidate.json",
            _runpod_promotion_artifact(
                "candidate",
                candidate_rows,
                evaluation_contract_hash=plan.evaluation_contract.contract_hash,
            ).to_dict(),
        )
        return _runpod_output(
            self.store,
            artifacts={
                "baseline_eval": "evals/baseline.json",
                "candidate_eval": "evals/candidate.json",
            },
            values={"evaluation_count": len(baseline_rows) + len(candidate_rows)},
            phase_cost=_runpod_phase_cost(
                self.cfg,
                "promotion_evaluation",
                time.perf_counter() - started,
            ),
        )

    def _finalize(
        self,
        _plan: RunPlan,
        prior: dict[str, StageOutput],
    ) -> StageOutput:
        if self.store is None:
            return _runpod_worker_output()
        decision = _read_json_object(prior["promote"].artifacts["promotion_decision"])
        next_experiment = _read_json_object(
            prior["promote"].artifacts["next_experiment"]
        )
        candidate = _read_json_object(prior["select"].artifacts["candidate"])
        train_result = _read_json_object(prior["train"].artifacts["train_result"])
        phase_records = tuple(
            PhaseCostRecord.model_validate(output.values["phase_cost"])
            for output in prior.values()
            if "phase_cost" in output.values
        )
        cost_summary = summarize_costs(
            phase_records,
            candidates=1,
            useful_traces=int(prior["evidence"].values.get("useful_trace_count", 0)),
            evaluations=int(prior["evaluate"].values["evaluation_count"]),
            promoted_metric_gain=max(0.0, float(decision["primary_delta"])),
        )
        self.store.write_json("costs.json", cost_summary)
        self.store.write_json("metrics.json", _promotion_metrics(decision))
        lifecycle = (
            {
                "enabled": False,
                "promoted": False,
                "reason": "no_training_outcome",
                "local_artifacts": {},
            }
            if not bool(prior["select"].values["trained"])
            else _finalize_lifecycle_if_configured(
                cfg=self.cfg,
                store=self.store,
                candidate=candidate,
                train_result=train_result,
                decision=decision,
            )
        )
        if lifecycle.get("enabled"):
            lifecycle_artifact = {"lifecycle": "lifecycle/runpod_grpo_lifecycle.json"}
        else:
            self.store.write_json("lifecycle/runpod_grpo_lifecycle.json", lifecycle)
            lifecycle_artifact = {"lifecycle": "lifecycle/runpod_grpo_lifecycle.json"}
        baseline = {
            "candidate_id": "baseline",
            "model_id": self.cfg.model.base_model_id,
            "adapter_kind": "base",
        }
        report = _final_report(
            self.cfg,
            baseline,
            candidate,
            train_result,
            decision,
            lifecycle,
            cost_summary,
            next_experiment,
        )
        self.store.write_json("final_report.json", report)
        self.store.write_text("final_report.md", _markdown_report(report))
        _event(self.store, "run_finished", {"decision": decision["decision"]})
        return _runpod_output(
            self.store,
            artifacts={
                "metrics": "metrics.json",
                "costs": "costs.json",
                "final_report_json": "final_report.json",
                "final_report_md": "final_report.md",
                "logs": "logs/events.jsonl",
                **lifecycle_artifact,
            },
            values={"decision": decision["decision"]},
        )


def _runpod_output(
    store: ArtifactStore,
    *,
    artifacts: Mapping[str, str],
    values: Mapping[str, Any] | None = None,
    phase_cost: PhaseCostRecord | None = None,
) -> StageOutput:
    output_values = dict(values or {})
    if phase_cost is not None:
        output_values["phase_cost"] = phase_cost.model_dump(mode="json")
    cost_usd = None if phase_cost is None else phase_cost.cost_usd
    missing_reason = None
    if phase_cost is not None and cost_usd is None:
        missing_reason = phase_cost.missing_reason
    return StageOutput(
        artifacts={
            name: str(store.run_dir / relative)
            for name, relative in artifacts.items()
        },
        values=output_values,
        cost_usd=0.0 if phase_cost is None else cost_usd,
        cost_missing_reason=missing_reason,
    )


def _runpod_worker_output() -> StageOutput:
    return StageOutput(cost_usd=0.0)


def _read_json_object(path: str | Path) -> dict[str, Any]:
    body = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(body, dict):
        raise ValueError(f"expected JSON object: {path}")
    return body


def runtime_environment(dist: DistributedContext | None = None) -> dict[str, Any]:
    packages = {}
    for name in ("torch", "transformers", "trl", "datasets", "accelerate", "peft"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = "NOT_INSTALLED"
    cuda: dict[str, Any]
    try:
        import torch

        cuda = {
            "available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
            "version": torch.version.cuda,
            "devices": [
                torch.cuda.get_device_name(idx)
                for idx in range(torch.cuda.device_count())
            ],
        }
    except Exception as exc:  # pragma: no cover - defensive runtime artifact
        cuda = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "cuda": cuda,
        "distributed": (dist or DistributedContext.from_env()).to_json(),
    }


def _require_cuda(cfg: RunPodGRPOConfig) -> dict[str, Any]:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "RunPod GRPO hillclimb requires CUDA and PyTorch. Start a GPU pod "
            "with the training dependencies installed."
        ) from exc

    try:
        return validate_cuda_runtime(
            torch_module=torch,
            expected_cuda_version=cfg.execution.cuda_version,
            expected_gpu_count=cfg.execution.gpu_count,
            expected_gpu_type=cfg.execution.gpu_type,
        )
    except RuntimeError as exc:
        raise RuntimeError(
            "RunPod GRPO hillclimb requires CUDA and the configured runtime: "
            f"{exc}"
        ) from exc


def _validate_launch_topology(cfg: RunPodGRPOConfig, dist: DistributedContext) -> None:
    expected = cfg.execution.gpu_count
    if expected > 1 and not dist.is_distributed:
        raise ValueError(
            "RunPod GRPO config requests multiple GPUs and requires accelerate launch. "
            f"Use: accelerate launch --num_processes {expected} -m post_train_engine.cli "
            f"hillclimb --config <config>. Config gpu_count={expected}, WORLD_SIZE=1."
        )
    if dist.world_size != expected:
        raise ValueError(
            "RunPod GRPO launch topology mismatch: "
            f"config execution.gpu_count={expected}, WORLD_SIZE={dist.world_size}."
        )
    if dist.rank >= dist.world_size:
        raise ValueError(f"RANK must be < WORLD_SIZE; got rank={dist.rank}, world={dist.world_size}")
    if dist.local_rank >= dist.world_size:
        raise ValueError(
            f"LOCAL_RANK must be < WORLD_SIZE for single-node RunPod launch; "
            f"got local_rank={dist.local_rank}, world={dist.world_size}"
        )


def _validate_grpo_runtime_shape(training: GRPOTrainingSpec, *, world_size: int | None = None) -> None:
    resolved_world_size = _world_size_from_env() if world_size is None else world_size
    global_prompt_batch = training.per_device_train_batch_size * resolved_world_size
    if global_prompt_batch % training.num_generations != 0:
        raise ValueError(
            "TRL GRPO requires per_device_train_batch_size * world_size to be "
            "divisible by num_generations. Got "
            f"{training.per_device_train_batch_size} * {resolved_world_size} = "
            f"{global_prompt_batch}, num_generations={training.num_generations}."
        )


def _world_size_from_env() -> int:
    return _positive_env_int("WORLD_SIZE", default=1)


def _positive_env_int(name: str, *, default: int) -> int:
    value = _env_int(name, default=default)
    if value <= 0:
        raise ValueError(f"{name} must be positive; got {value}")
    return value


def _nonnegative_env_int(name: str, *, default: int) -> int:
    value = _env_int(name, default=default)
    if value < 0:
        raise ValueError(f"{name} must be non-negative; got {value}")
    return value


def _env_int(name: str, *, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer; got {raw!r}") from exc


def _distributed_state(dist: DistributedContext) -> Any | None:
    if not dist.is_distributed:
        return None
    try:
        from accelerate import PartialState
    except ImportError as exc:
        raise RuntimeError(
            'Accelerate is required for multi-GPU RunPod GRPO. Run: python -m pip install -e ".[dev,rlvr]"'
        ) from exc
    return PartialState()


def _wait_for_everyone(
    state: Any | None,
    *,
    timeout_seconds: float = 120.0,
) -> None:
    if state is None:
        return
    from datetime import timedelta

    import torch.distributed as torch_dist

    work = torch_dist.barrier(async_op=True)
    if not work.wait(timeout=timedelta(seconds=timeout_seconds)):
        raise TimeoutError("distributed preparation barrier timed out")


def _load_and_split_dataset(
    cfg: RunPodGRPOConfig,
) -> tuple[list[GSM8KExample], list[GSM8KExample], list[GSM8KExample]]:
    if cfg.dataset.source == "huggingface_gsm8k":
        examples = load_gsm8k(
            "train",
            cfg.dataset.dataset_name,
            revision=cfg.dataset.resolved_revision or cfg.dataset.revision,
        )
    else:
        from post_train_engine.tasks.gsm8k import embedded_gsm8k_examples

        examples = embedded_gsm8k_examples()
    required = (
        cfg.dataset.train_size
        + cfg.dataset.selection_size
        + cfg.dataset.eval_size
    )
    if required > len(examples):
        raise ValueError(f"requested {required} examples but dataset has {len(examples)}")
    shuffled = list(examples)
    random.Random(cfg.dataset.split_seed).shuffle(shuffled)
    train_end = cfg.dataset.train_size
    selection_end = train_end + cfg.dataset.selection_size
    return (
        shuffled[:train_end],
        shuffled[train_end:selection_end],
        shuffled[selection_end:required],
    )


def _write_dataset_artifacts(
    store: ArtifactStore,
    cfg: RunPodGRPOConfig,
    train_examples: Sequence[GSM8KExample],
    selection_examples: Sequence[GSM8KExample],
    promotion_examples: Sequence[GSM8KExample],
) -> None:
    store.write_json(
        "datasets/splits.json",
        {
            "dataset": "gsm8k",
            "source": cfg.dataset.source,
            "dataset_name": cfg.dataset.dataset_name,
            "split_seed": cfg.dataset.split_seed,
            "train_ids": [row.id for row in train_examples],
            "selection_ids": [row.id for row in selection_examples],
            "promotion_ids": [row.id for row in promotion_examples],
        },
    )
    store.write_jsonl(
        "datasets/train.jsonl",
        [_example_json(cfg, row, "train") for row in train_examples],
    )
    store.write_jsonl(
        "datasets/selection.jsonl",
        [_example_json(cfg, row, "selection") for row in selection_examples],
    )
    store.write_jsonl(
        "datasets/promotion.jsonl",
        [_example_json(cfg, row, "promotion") for row in promotion_examples],
    )

def _write_measured_training_view(
    store: ArtifactStore,
    cfg: RunPodGRPOConfig,
    train_examples: Sequence[GSM8KExample],
    probe_rows: Sequence[EvalRow],
) -> TrainingViewArtifact | None:
    examples_by_id = {example.id: example for example in train_examples}
    grouped: dict[str, list[EvalRow]] = {}
    for row in probe_rows:
        if row.example_id not in examples_by_id:
            raise ValueError(f"parent probe returned unknown train example: {row.example_id}")
        grouped.setdefault(row.example_id, []).append(row)
    missing = sorted(set(examples_by_id).difference(grouped))
    if missing:
        raise ValueError(f"parent probe missing train examples: {missing}")

    traces: list[TraceRecord] = []
    traces_by_example: dict[str, list[TraceRecord]] = {}
    for example_id, rows in sorted(grouped.items()):
        seen_samples: set[int] = set()
        group_id = f"{cfg.run.run_id}:parent-probe:{example_id}"
        for row in sorted(rows, key=lambda item: item.sample_index):
            if row.sample_index in seen_samples:
                raise ValueError(
                    f"parent probe has duplicate sample_index for {example_id}: "
                    f"{row.sample_index}"
                )
            seen_samples.add(row.sample_index)
            trace = TraceRecord(
                trace_id=f"{group_id}:{row.sample_index}",
                run_id=cfg.run.run_id,
                task_id="gsm8k",
                example_id=example_id,
                split_role="train",
                prompt_hash=stable_prompt_hash(row.prompt),
                source_checkpoint=cfg.model.base_model_id,
                policy_version=cfg.model.base_model_id,
                policy_step=0,
                policy_step_evidence="static",
                rollout_group_id=group_id,
                generation_backend="transformers_parent_probe",
                sampling_config={
                    "do_sample": True,
                    "temperature": cfg.training.temperature,
                    "top_p": cfg.training.top_p,
                    "max_new_tokens": cfg.training.max_completion_length,
                    "num_generations": cfg.training.num_generations,
                },
                verifier_id="gsm8k_numeric_v1",
                prompt=row.prompt,
                completion=row.completion,
                parsed_answer=row.parsed_answer,
                parser_status={"parse_ok": row.parse_ok},
                verifier_result={"correct": row.correct},
                reward_components={"correct": float(row.correct)},
                token_counts={"completion": float(row.completion_tokens)},
                privileged_visibility="none",
            )
            traces.append(trace)
            traces_by_example.setdefault(example_id, []).append(trace)
    store.write_jsonl(
        "evidence/input_traces.jsonl",
        [trace.model_dump(mode="json") for trace in traces],
    )
    groups = [
        build_rollout_group(
            group_id=f"{cfg.run.run_id}:parent-probe:{example_id}",
            traces=traces_by_example[example_id],
            rewards=[float(row.correct) for row in sorted(rows, key=lambda item: item.sample_index)],
        )
        for example_id, rows in sorted(grouped.items())
    ]
    store.write_jsonl(
        "evidence/rollout_groups.jsonl",
        [group.model_dump(mode="json") for group in groups],
    )
    frontier_ids = {
        example_id
        for example_id, rows in grouped.items()
        if 0.0 < sum(float(row.correct) for row in rows) / len(rows) < 1.0
    }
    if not frontier_ids:
        store.write_json(
            "evidence/non_training_outcome.json",
            {
                "outcome": "no_learnable_evidence",
                "selection_policy": "parent_success_rate_frontier",
                "measured_example_count": len(grouped),
            },
        )
        return None
    selected_rows = [
        {
            **_example_json(cfg, examples_by_id[example_id], "train"),
            "source_trace_ids": [trace.trace_id for trace in traces_by_example[example_id]],
            "source_split_roles": ["train"],
        }
        for example_id in sorted(frontier_ids)
    ]
    train_path = store.write_jsonl("datasets/train_selected.jsonl", selected_rows)
    view = build_training_view_artifact(
        view_id=f"{cfg.run.run_id}:grpo-training",
        run_id=cfg.run.run_id,
        task_id="gsm8k",
        view_type="grpo_rollout",
        method_compatibility=("grpo",),
        data_path=train_path,
        artifact_root=store.run_dir,
        data_kind="grpo_training_rows",
        rows=selected_rows,
        privileged_visibility="gold_answer",
        metadata={
            "selection_policy": "parent_success_rate_frontier",
            "selection_evidence": "measured_parent_success_rate",
            "selected_example_ids": sorted(frontier_ids),
            "parent_policy_version": cfg.model.base_model_id,
        },
    )
    write_training_view_artifact(
        view,
        store.run_dir / "evidence" / "method_training_view.json",
    )
    return view


def _shard_sequence(rows: Sequence[Any], dist: DistributedContext) -> list[Any]:
    if not dist.is_distributed:
        return list(rows)
    return [
        row
        for idx, row in enumerate(rows)
        if idx % dist.world_size == dist.rank
    ]


def _example_json(cfg: RunPodGRPOConfig, example: GSM8KExample, split_role: str) -> dict[str, Any]:
    return {
        "example_id": example.id,
        "split_role": split_role,
        "question": example.question,
        "gold_solution": example.gold_solution,
        "gold_answer": example.gold_answer,
        "prompt": format_prompt(example.question, cfg.dataset.prompt_style),
        "source": example.source,
        "metadata": dict(example.metadata),
    }


def _train_grpo(
    cfg: RunPodGRPOConfig,
    training_view: TrainingViewArtifact,
    store: ArtifactStore | None,
    *,
    dist: DistributedContext,
) -> dict[str, Any]:
    if "grpo" not in training_view.method_compatibility:
        raise ValueError("RunPod GRPO requires a GRPO-compatible TrainingView")
    training_rows = read_jsonl(
        training_view.require_data_integrity(cfg.run.output_dir)
    )
    if not training_rows:
        raise ValueError("RunPod GRPO TrainingView contains no rows")
    try:
        import torch
        from datasets import Dataset
        from transformers import AutoModelForCausalLM
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise RuntimeError(
            'RunPod GRPO requires GPU dependencies. Run: python -m pip install -e ".[dev,rlvr]"'
        ) from exc

    tokenizer = _load_tokenizer(cfg)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.base_model_id,
        revision=cfg.model.resolved_revision or cfg.model.revision,
        torch_dtype=_torch_dtype(torch, cfg.model.dtype),
        trust_remote_code=cfg.model.trust_remote_code,
        use_safetensors=cfg.model.use_safetensors,
        attn_implementation=cfg.model.attn_implementation,
    )
    _align_model_with_tokenizer(model, tokenizer)
    if cfg.model.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    training_args = GRPOConfig(
        **_filter_trl_config_kwargs(_grpo_config_kwargs(cfg), GRPOConfig)
    )
    reward_config = cfg.training.reward_config()
    trace_path = None
    if cfg.trace_capture.enabled:
        trace_path = (
            Path(cfg.run.output_dir)
            / cfg.trace_capture.train_trace_dir
            / f"rank{dist.rank}.jsonl"
        )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=_gsm8k_reward_func(
            tokenizer,
            reward_config,
            trace_path=trace_path,
            run_id=cfg.run.run_id,
            source_checkpoint=cfg.model.base_model_id,
            rank=dist.rank,
            sampling_config={
                "temperature": cfg.training.temperature,
                "top_p": cfg.training.top_p,
                "num_generations": cfg.training.num_generations,
                "max_completion_length": cfg.training.max_completion_length,
            },
        ),
        args=training_args,
        train_dataset=Dataset.from_list(
            [
                {
                    "prompt": str(row["prompt"]),
                    "answer": str(row["gold_answer"]),
                    "example_id": str(row["example_id"]),
                }
                for row in training_rows
            ]
        ),
        processing_class=tokenizer,
    )
    _event(store, "training_started", {"max_steps": cfg.training.max_steps, "distributed": dist.to_json()})
    result = trainer.train()
    final_dir = Path(cfg.run.output_dir) / "train" / "final"
    if _trainer_is_main_process(trainer):
        trainer.save_model(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
    metrics = _finite_metrics(getattr(result, "metrics", {}))
    _event(store, "training_finished", {"metrics": metrics, "final_dir": str(final_dir)})
    del trainer
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "trainer": "TRL_GRPOTrainer",
        "method": "grpo",
        "base_model_id": cfg.model.base_model_id,
        "final_checkpoint": str(final_dir),
        "metrics": metrics,
        "distributed": dist.to_json(),
        "training_view": {
            "view_id": training_view.view_id,
            "source_trace_ids": list(training_view.source_trace_ids),
            "source_split_roles": list(training_view.source_split_roles),
        },
        "reward": {
            "verifier": "gsm8k_numeric_v1",
            "parse_bonus": cfg.training.parse_bonus,
            "length_penalty_weight": cfg.training.length_penalty_weight,
            "max_new_tokens": cfg.training.max_completion_length,
        },
        "trace_capture": {
            "enabled": cfg.trace_capture.enabled,
            "trace_path": None if trace_path is None else str(trace_path),
            "rank": dist.rank,
        },
    }


def _candidate_checkpoint_refs(run_dir: str | Path) -> list[dict[str, str]]:
    train_dir = Path(run_dir) / "train"
    refs: list[dict[str, str]] = []
    if train_dir.exists():
        for path in sorted(train_dir.glob("checkpoint-*"), key=_checkpoint_sort_key):
            if path.is_dir():
                refs.append({"checkpoint_id": path.name, "path": str(path)})
        final_dir = train_dir / "final"
        if final_dir.is_dir():
            refs.append({"checkpoint_id": "final", "path": str(final_dir)})
    if not refs:
        raise ValueError(f"no candidate checkpoints found under {train_dir}")
    return refs


def _checkpoint_sort_key(path: Path) -> tuple[int, str]:
    suffix = path.name.removeprefix("checkpoint-")
    try:
        return (int(suffix), path.name)
    except ValueError:
        return (sys.maxsize, path.name)


def _evaluate_and_select_checkpoints(
    *,
    cfg: RunPodGRPOConfig,
    checkpoint_refs: Sequence[Mapping[str, str]],
    eval_examples: Sequence[GSM8KExample],
    dist: DistributedContext,
    store: ArtifactStore | None,
) -> dict[str, Any]:
    evaluated: list[dict[str, Any]] = []
    refs = list(checkpoint_refs)
    if not cfg.checkpoint_selection.include_final:
        refs = [ref for ref in refs if ref["checkpoint_id"] != "final"]
    if not cfg.checkpoint_selection.enabled:
        refs = [refs[-1]]
    for ref in refs:
        local_rows = _evaluate_hf_model(
            cfg=cfg,
            model_ref=ref["path"],
            examples=_shard_sequence(eval_examples, dist),
            dist=dist,
        )
        rows = _gather_eval_rows(local_rows, dist)
        if dist.is_main_process:
            artifact_prefix = f"checkpoint_selection/{_safe_artifact_name(ref['checkpoint_id'])}"
            assert store is not None
            _write_eval_outputs(store, artifact_prefix, rows)
            store.write_json(
                f"evals/{artifact_prefix}.json",
                _eval_artifact(ref["checkpoint_id"], rows),
            )
            evaluated.append(
                {
                    "checkpoint_id": ref["checkpoint_id"],
                    "path": ref["path"],
                    "rows": rows,
                    "metrics": _eval_metrics(rows),
                }
            )
    if not dist.is_main_process:
        return {}
    selection = _select_checkpoint(evaluated, metric=cfg.checkpoint_selection.metric)
    assert store is not None
    store.write_json(
        "train/checkpoint_selection.json",
        {key: value for key, value in selection.items() if key != "selected_eval_rows"},
    )
    _event(
        store,
        "checkpoint_selected",
        {
            "checkpoint_id": selection["selected_checkpoint_id"],
            "metric": cfg.checkpoint_selection.metric,
            "score": selection["selected_score"],
        },
    )
    return selection


def _select_checkpoint(
    evaluated: Sequence[Mapping[str, Any]],
    *,
    metric: str,
) -> dict[str, Any]:
    if not evaluated:
        raise ValueError("checkpoint selection requires at least one evaluated checkpoint")
    ranked = sorted(
        evaluated,
        key=lambda item: (
            _checkpoint_score(item["rows"], metric),
            -_eval_metrics(item["rows"])["mean_tokens"],
            str(item["checkpoint_id"]) == "final",
        ),
        reverse=True,
    )
    selected = ranked[0]
    return {
        "selected_checkpoint_id": selected["checkpoint_id"],
        "selected_checkpoint_path": selected["path"],
        "selected_score": _checkpoint_score(selected["rows"], metric),
        "selection_metric": metric,
        "selection_reason": f"max_{metric}_then_min_mean_tokens",
        "evaluated_checkpoints": [
            {
                "checkpoint_id": item["checkpoint_id"],
                "path": item["path"],
                "metrics": _eval_metrics(item["rows"]),
            }
            for item in ranked
        ],
        "selected_eval_rows": list(selected["rows"]),
    }


def _checkpoint_score(rows: Sequence[EvalRow], metric: str) -> float:
    metrics = _eval_metrics(rows)
    if metric not in metrics:
        raise ValueError(f"unsupported checkpoint selection metric: {metric}")
    value = metrics[metric]
    if not math.isfinite(value):
        raise ValueError(f"checkpoint selection metric is non-finite: {metric}")
    return value


def _safe_artifact_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "-" for char in value)


def _evaluate_hf_model(
    *,
    cfg: RunPodGRPOConfig,
    model_ref: str,
    examples: Sequence[GSM8KExample],
    dist: DistributedContext,
    mode: Literal["evaluation", "training_probe"] = "evaluation",
    samples_per_example: int = 1,
) -> list[EvalRow]:
    if samples_per_example <= 0:
        raise ValueError("samples_per_example must be positive")
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError("Transformers and torch are required for RunPod eval") from exc

    tokenizer = _load_tokenizer(cfg, model_ref=model_ref)
    model = AutoModelForCausalLM.from_pretrained(
        model_ref,
        revision=(
            cfg.model.resolved_revision or cfg.model.revision
            if model_ref == cfg.model.base_model_id
            else None
        ),
        torch_dtype=_torch_dtype(torch, cfg.model.dtype),
        trust_remote_code=cfg.model.trust_remote_code,
        use_safetensors=cfg.model.use_safetensors,
        attn_implementation=cfg.model.attn_implementation,
    ).eval()
    _align_model_with_tokenizer(model, tokenizer)
    device = torch.device(f"cuda:{dist.local_rank}")
    model.to(device)
    rows: list[EvalRow] = []
    expanded = [
        (example, sample_index)
        for example in examples
        for sample_index in range(samples_per_example)
    ]
    for start in range(0, len(expanded), cfg.eval.batch_size):
        batch = expanded[start : start + cfg.eval.batch_size]
        prompts = [
            format_prompt(example.question, cfg.dataset.prompt_style)
            for example, _sample_index in batch
        ]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
        ).to(device)
        generate_kwargs = {
            "max_new_tokens": (
                cfg.training.max_completion_length
                if mode == "training_probe"
                else cfg.eval.max_new_tokens
            ),
            "do_sample": mode == "training_probe" or cfg.eval.do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "renormalize_logits": True,
            "remove_invalid_values": True,
        }
        if generate_kwargs["do_sample"]:
            generate_kwargs["temperature"] = (
                cfg.training.temperature
                if mode == "training_probe"
                else cfg.eval.temperature
            )
            if mode == "training_probe":
                generate_kwargs["top_p"] = cfg.training.top_p
        with torch.no_grad():
            output_batch = model.generate(
                **inputs,
                **generate_kwargs,
            )
        input_length = inputs["input_ids"].shape[-1]
        for (example, sample_index), prompt, output_ids in zip(
            batch,
            prompts,
            output_batch,
            strict=True,
        ):
            completion_ids = output_ids[input_length:]
            completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
            parsed = parse_model_answer(completion, mode="strict")
            verification = (
                verify_answer(parsed.answer, example.gold_answer)
                if parsed.parse_ok and parsed.answer is not None
                else None
            )
            rows.append(
                EvalRow(
                    example_id=example.id,
                    prompt=prompt,
                    completion=completion,
                    parsed_answer=parsed.answer,
                    gold_answer=example.gold_answer,
                    correct=bool(verification and verification.correct),
                    parse_ok=parsed.parse_ok,
                    completion_tokens=int(len(completion_ids)),
                    sample_index=sample_index,
                )
            )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def _gather_eval_rows(rows: Sequence[EvalRow], dist: DistributedContext) -> list[EvalRow]:
    if not dist.is_distributed:
        return list(rows)
    try:
        import torch.distributed as torch_dist
    except ImportError as exc:  # pragma: no cover - torch always present in this path
        raise RuntimeError("torch.distributed is required for distributed eval merge") from exc
    if not torch_dist.is_available() or not torch_dist.is_initialized():
        raise RuntimeError("distributed eval merge requires an initialized process group")
    gathered: list[list[dict[str, Any]] | None] = [None for _ in range(dist.world_size)]
    torch_dist.all_gather_object(gathered, [row.to_json() for row in rows])
    if not dist.is_main_process:
        return []
    merged: list[EvalRow] = []
    for shard in gathered:
        if shard is None:
            raise RuntimeError("distributed eval merge received a missing shard")
        for row in shard:
            merged.append(EvalRow(**row))
    return sorted(merged, key=lambda row: (row.example_id, row.sample_index))


def _write_eval_outputs(store: ArtifactStore, artifact_prefix: str, rows: Sequence[EvalRow]) -> None:
    store.write_jsonl(
        f"evals/{artifact_prefix}_raw_outputs.jsonl",
        [row.to_json() for row in rows],
    )


def _load_tokenizer(cfg: RunPodGRPOConfig, *, model_ref: str | None = None) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_ref or cfg.model.base_model_id,
        revision=(
            cfg.model.resolved_revision or cfg.model.revision
            if model_ref is None or model_ref == cfg.model.base_model_id
            else None
        ),
        trust_remote_code=cfg.model.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def _torch_dtype(torch: Any, dtype: str) -> Any:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]


def _align_model_with_tokenizer(model: Any, tokenizer: Any) -> None:
    tokenizer.padding_side = "left"
    token_count = len(tokenizer)
    embedding_count = model.get_input_embeddings().num_embeddings
    if token_count > embedding_count:
        model.resize_token_embeddings(token_count)
    for token_attr in ("pad_token_id", "eos_token_id", "bos_token_id"):
        token_id = getattr(tokenizer, token_attr, None)
        if token_id is None:
            continue
        setattr(model.config, token_attr, token_id)
        generation_config = getattr(model, "generation_config", None)
        if generation_config is not None:
            setattr(generation_config, token_attr, token_id)


def _grpo_config_kwargs(cfg: RunPodGRPOConfig) -> dict[str, Any]:
    return {
        "output_dir": str(Path(cfg.run.output_dir) / "train"),
        "max_steps": cfg.training.max_steps,
        "learning_rate": cfg.training.learning_rate,
        "per_device_train_batch_size": cfg.training.per_device_train_batch_size,
        "gradient_accumulation_steps": cfg.training.gradient_accumulation_steps,
        "bf16": cfg.model.dtype == "bfloat16",
        "fp16": cfg.model.dtype == "float16",
        "tf32": True,
        "gradient_checkpointing": cfg.model.gradient_checkpointing,
        "logging_steps": cfg.training.logging_steps,
        "save_steps": cfg.training.save_steps,
        "save_total_limit": cfg.training.save_total_limit,
        "report_to": cfg.training.report_to,
        "run_name": cfg.training.run_name or cfg.run.run_id,
        "remove_unused_columns": False,
        "seed": cfg.run.seed,
        "num_generations": cfg.training.num_generations,
        "max_completion_length": cfg.training.max_completion_length,
        "temperature": cfg.training.temperature,
        "top_p": cfg.training.top_p,
        "beta": cfg.training.beta,
        "generation_kwargs": _STABLE_GENERATION_KWARGS,
        "use_vllm": cfg.training.use_vllm,
        "vllm_gpu_memory_utilization": cfg.training.vllm_gpu_memory_utilization,
    }


def _filter_trl_config_kwargs(kwargs: dict[str, Any], config_cls: type[Any]) -> dict[str, Any]:
    signature = inspect.signature(config_cls.__init__)
    parameters = signature.parameters
    if any(param.kind is inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return kwargs
    accepted = set(parameters)
    unsupported = set(kwargs) - accepted
    non_optional = unsupported - _OPTIONAL_GRPO_CONFIG_FIELDS
    if non_optional:
        raise TypeError(
            f"Installed {config_cls.__name__} does not accept required kwargs: "
            f"{sorted(non_optional)}. Accepted kwargs include: {sorted(accepted)}"
        )
    return {key: value for key, value in kwargs.items() if key not in unsupported}


def _trainer_is_main_process(trainer: Any) -> bool:
    is_world_process_zero = getattr(trainer, "is_world_process_zero", None)
    if callable(is_world_process_zero):
        return bool(is_world_process_zero())
    accelerator = getattr(trainer, "accelerator", None)
    if accelerator is not None and hasattr(accelerator, "is_main_process"):
        return bool(accelerator.is_main_process)
    args = getattr(trainer, "args", None)
    if args is not None and hasattr(args, "process_index"):
        return int(args.process_index) == 0
    return DistributedContext.from_env().is_main_process


def _gsm8k_reward_func(
    tokenizer: Any,
    config: GSM8KRewardConfig,
    *,
    trace_path: str | Path | None = None,
    run_id: str = "unknown-run",
    source_checkpoint: str = "unknown-checkpoint",
    rank: int = 0,
    sampling_config: Mapping[str, Any] | None = None,
) -> Any:
    trace_store = None if trace_path is None else JsonlTraceStore(trace_path)
    counter = {"value": 0, "batch": 0}

    def reward_func(
        *args: Any,
        completions: list[Any] | None = None,
        answer: Any = None,
        gold_answer: Any = None,
        prompt: Any = None,
        prompts: Any = None,
        example_id: Any = None,
        **metadata: Any,
    ) -> list[float]:
        if completions is None:
            if len(args) == 1:
                completions = args[0]
            elif len(args) >= 2:
                completions = args[1]
            else:
                raise ValueError("completions are required to score GSM8K rewards")
        gold_values = gold_answer if gold_answer is not None else answer
        gold_batch = _as_batch(gold_values, n=len(completions), field_name="answer")
        prompt_values = prompts if prompts is not None else prompt
        prompt_batch = _optional_batch(prompt_values, n=len(completions), default="")
        example_id_batch = _optional_batch(
            example_id,
            n=len(completions),
            default="unknown-example",
        )
        raw_step = metadata.get("global_step", metadata.get("step"))
        if type(raw_step) is not bool and isinstance(raw_step, int) and raw_step >= 0:
            policy_step = raw_step
            step_evidence = "exact"
            policy_version = f"{source_checkpoint}:optimizer-step:{raw_step}"
        else:
            policy_step = counter["batch"]
            step_evidence = "inferred_batch"
            policy_version = (
                f"{run_id}:rank{rank}:reward-batch:{counter['batch']:08d}"
            )
        rewards: list[float] = []
        for completion, gold, prompt_text, example in zip(
            completions,
            gold_batch,
            prompt_batch,
            example_id_batch,
            strict=True,
        ):
            text = _completion_to_text(completion)
            token_count = len(tokenizer.encode(text, add_special_tokens=False))
            result = compute_gsm8k_reward(
                text,
                gold,
                rho_q=1.0,
                completion_tokens=token_count,
                config=config,
            )
            rewards.append(float(result.reward))
            if trace_store is not None:
                trace_store.append(
                    TraceRecord(
                        trace_id=f"{run_id}:grpo-train:rank{rank}:{counter['value']:08d}",
                        run_id=run_id,
                        task_id="gsm8k",
                        example_id=str(example),
                        split_role="train",
                        prompt_hash=stable_prompt_hash(str(prompt_text)),
                        source_checkpoint=source_checkpoint,
                        policy_version=policy_version,
                        policy_step=policy_step,
                        policy_step_evidence=step_evidence,
                        rollout_group_id=(
                            f"{run_id}:rank{rank}:batch{counter['batch']:08d}:"
                            f"{example}"
                        ),
                        generation_backend="trl_grpo",
                        sampling_config=dict(
                            sampling_config or {"source": "not_reported"}
                        ),
                        verifier_id="gsm8k_numeric_v1",
                        prompt=str(prompt_text),
                        completion=text,
                        parsed_answer=result.parsed_answer,
                        parser_status={
                            "parse_ok": result.parse_ok,
                            "mode": "strict" if config.use_strict_parse_for_reward else "lenient",
                            "error": result.verifier_error,
                        },
                        verifier_result={
                            "verifier": "gsm8k_numeric_v1",
                            "correct": result.correct,
                            "gold_answer": str(gold),
                            "error": result.verifier_error,
                        },
                        reward_components={
                            "reward": float(result.reward),
                            "task_reward": float(result.task_reward),
                            "parse_bonus": float(result.parse_bonus),
                            "length_penalty": float(result.length_penalty),
                        },
                        token_counts={"completion": float(token_count)},
                        privileged_visibility="gold_answer",
                    )
                )
                counter["value"] += 1
        counter["batch"] += 1
        return rewards

    return reward_func


def _completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, Mapping):
        content = completion.get("content")
        return str(content if content is not None else completion)
    if isinstance(completion, Sequence):
        parts = []
        for item in completion:
            if isinstance(item, Mapping) and item.get("content") is not None:
                parts.append(str(item["content"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(completion)


def _as_batch(values: Any, *, n: int, field_name: str) -> list[str]:
    if values is None:
        raise ValueError(f"{field_name} is required to score GSM8K rewards")
    if isinstance(values, str):
        return [values] * n
    if isinstance(values, Sequence):
        if len(values) == 0:
            raise ValueError(f"{field_name} must not be empty")
        items = [str(value) for value in values]
        if len(items) == n:
            return items
        if n % len(items) == 0:
            repeats = n // len(items)
            return [item for item in items for _ in range(repeats)]
        raise ValueError(f"{field_name} length {len(values)} does not match completions length {n}")
    return [str(values)] * n


def _optional_batch(values: Any, *, n: int, default: str) -> list[str]:
    if values is None:
        return [default] * n
    if isinstance(values, str):
        return [values] * n
    if isinstance(values, Sequence):
        items = [str(value) for value in values]
        if len(items) == n:
            return items
        if len(items) == 1:
            return items * n
    return [str(values)] * n


def _eval_artifact(candidate_id: str, rows: Sequence[EvalRow]) -> dict[str, Any]:
    if not rows:
        raise ValueError("eval artifact requires at least one row")
    metrics = _eval_metrics(rows)
    return {
        "candidate_id": candidate_id,
        "metrics": metrics,
        "examples": [row.to_json() for row in rows],
    }


def _eval_metrics(rows: Sequence[EvalRow]) -> dict[str, float]:
    return {
        "accuracy": sum(row.correct for row in rows) / len(rows),
        "parse_success_rate": sum(row.parse_ok for row in rows) / len(rows),
        "mean_tokens": sum(row.completion_tokens for row in rows) / len(rows),
    }


def _runpod_promotion_artifact(
    artifact_id: str,
    rows: Sequence[EvalRow],
    *,
    evaluation_contract_hash: str,
) -> EvalArtifact:
    metrics = _eval_metrics(rows)
    examples = tuple(
        EvalExampleResult(
            example_id=row.example_id,
            correct=row.correct,
            parse_ok=row.parse_ok,
            tokens=row.completion_tokens,
            bucket="easy_stable",
        )
        for row in rows
    )
    return EvalArtifact(
        artifact_id=artifact_id,
        primary_metric="accuracy",
        evaluation_contract_hash=evaluation_contract_hash,
        examples=examples,
        metrics=metrics,
        slices={"easy_stable": {"accuracy": metrics["accuracy"]}},
    )


def _promotion_metrics(decision: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "baseline": dict(decision["baseline_metrics"]),
        "candidate": dict(decision["candidate_metrics"]),
        "delta": decision["primary_delta"],
        "paired_ci95": list(decision["primary_ci95"]),
        "mcnemar_p": decision["mcnemar_p"],
        "paired_stats": dict(decision["stats"]),
    }


def _validate_hf_upload_config(cfg: RunPodGRPOConfig) -> None:
    hf = cfg.hf_upload
    if not hf.enabled:
        return
    if not hf.repo_id:
        raise ValueError("hf_upload.repo_id is required when hf_upload.enabled=true")
    if not os.environ.get(hf.token_env):
        raise ValueError(f"missing required HF token env: {hf.token_env}")


def _finalize_lifecycle_if_configured(
    *,
    cfg: RunPodGRPOConfig,
    store: ArtifactStore,
    candidate: Mapping[str, Any],
    train_result: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> dict[str, Any]:
    hf = cfg.hf_upload
    promoted = decision["decision"] == "promote"
    if not hf.enabled:
        return {
            "enabled": False,
            "promoted": promoted,
            "remote_ref": None,
            "local_artifacts": {},
        }
    receipt_path = store.run_dir / "lifecycle" / "runpod_grpo_lifecycle.json"
    if receipt_path.is_file():
        existing = _read_json_object(receipt_path)
        if existing.get("promoted") is not promoted:
            raise ValueError("existing lifecycle receipt has a different promotion decision")
        return existing
    transaction_path = store.run_dir / "lifecycle" / "transaction.json"
    if transaction_path.is_file():
        transaction = _read_json_object(transaction_path)
        if transaction.get("state") == "started":
            raise ValueError(
                "remote lifecycle transaction is ambiguous; reconcile the deterministic "
                "Hugging Face path before retrying"
            )
    store.write_json(
        "lifecycle/transaction.json",
        {
            "state": "started",
            "candidate_id": candidate["candidate_id"],
            "promoted": promoted,
            "repo_id": hf.repo_id,
            "path_template": hf.path_template,
        },
    )
    uploader = HuggingFaceCheckpointUploader(token=os.environ[hf.token_env])
    lifecycle = CheckpointLifecycleManager(
        ModelLifecycleConfig(
            artifact_dir=store.run_dir / "lifecycle",
            discard_rejected_local=False,
            keep_only_latest_promoted_local=False,
            hf=HuggingFaceLifecycleConfig(
                enabled=True,
                repo_id=hf.repo_id,
                repo_type=hf.repo_type,
                private=hf.private,
                path_template=hf.path_template,
                upload_evidence=hf.upload_evidence,
                upload_promoted_checkpoints=hf.upload_promoted_checkpoints,
                upload_rejected_checkpoints=hf.upload_rejected_checkpoints,
            ),
        ),
        uploader=uploader,
    )
    outcome = lifecycle.finalize(
        CheckpointLifecycleInput(
            candidate_id=str(candidate["candidate_id"]),
            checkpoint_ref=str(candidate["model_id"]),
            task_name="gsm8k",
            parent_candidate_id="baseline",
            parent_checkpoint_ref=cfg.model.base_model_id,
            previous_incumbent_candidate_id="baseline",
            previous_incumbent_checkpoint_ref=cfg.model.base_model_id,
            previous_incumbent_remote_ref=None,
            promoted=promoted,
            score=float(decision["candidate_metrics"]["accuracy"]),
            incumbent_score=float(decision["baseline_metrics"]["accuracy"]),
            metrics={
                name: float(value)
                for name, value in decision["candidate_metrics"].items()
                if isinstance(value, int | float) and math.isfinite(float(value))
            },
            evaluation_artifacts={
                "candidate_eval": str(store.run_dir / "evals" / "candidate.json"),
                "raw_outputs": str(store.run_dir / "evals" / "candidate_raw_outputs.jsonl"),
                "checkpoint_selection": str(store.run_dir / "train" / "checkpoint_selection.json"),
            },
            evaluation_metadata={
                "split": "eval",
                "dataset": cfg.dataset.source,
                "dataset_name": cfg.dataset.dataset_name,
                "split_seed": cfg.dataset.split_seed,
            },
            train_artifacts={
                "trainer_result": str(store.run_dir / "train" / "trainer_result.json"),
                "trace_dir": str(store.run_dir / cfg.trace_capture.train_trace_dir),
            },
            train_metrics=dict(train_result.get("metrics", {})),
            train_metadata={
                "distributed": train_result.get("distributed", {}),
                "trace_capture": train_result.get("trace_capture", {}),
                "costs": {"source": "not_reported"},
            },
            promotion_gate=decision.get("gates", {}),
            promotion_decision=decision,
            rejection_reason=None if promoted else "; ".join(decision.get("rejection_reasons", [])),
        )
    )
    body = {
        "enabled": True,
        "promoted": promoted,
        "remote_ref": outcome.remote_ref,
        "local_state": outcome.local_state,
        "evidence_path": str(outcome.evidence_path),
        "local_artifacts": outcome.local_artifacts,
        "remote_artifacts": outcome.remote_artifacts,
        "discarded_paths": list(outcome.discarded_paths),
    }
    store.write_json("lifecycle/runpod_grpo_lifecycle.json", body)
    store.write_json(
        "lifecycle/transaction.json",
        {
            "state": "completed",
            "candidate_id": candidate["candidate_id"],
            "promoted": promoted,
            "receipt": "lifecycle/runpod_grpo_lifecycle.json",
        },
    )
    return body


def _finite_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in metrics.items():
        if type(value) is bool or not isinstance(value, int | float):
            continue
        numeric = float(value)
        if math.isfinite(numeric):
            out[str(key)] = numeric
    return out


def _runpod_phase_cost(
    cfg: RunPodGRPOConfig,
    phase: str,
    duration_seconds: float,
) -> PhaseCostRecord:
    price = cfg.execution.accelerator_hour_usd
    return PhaseCostRecord(
        phase=phase,
        duration_seconds=duration_seconds,
        resource=f"runpod:{cfg.execution.gpu_type}",
        resource_count=float(cfg.execution.gpu_count),
        unit_price_usd=price,
        missing_reason=(
            None if price is not None else "execution.accelerator_hour_usd not configured"
        ),
    )


def _final_report(
    cfg: RunPodGRPOConfig,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    train_result: dict[str, Any],
    decision: dict[str, Any],
    lifecycle: Mapping[str, Any],
    cost_summary: Mapping[str, Any],
    next_experiment: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "run_id": cfg.run.run_id,
        "status": decision["decision"],
        "execution": cfg.execution.model_dump(mode="json"),
        "model": cfg.model.model_dump(mode="json"),
        "training": cfg.training.model_dump(mode="json"),
        "baseline": baseline,
        "candidate": candidate,
        "train_result": train_result,
        "metrics": _promotion_metrics(decision),
        "promotion": decision,
        "lifecycle": dict(lifecycle),
        "cost": dict(cost_summary),
        "next_experiment": dict(next_experiment),
    }


def _markdown_report(report: Mapping[str, Any]) -> str:
    failures = report["promotion"]["rejection_reasons"] or ["none"]
    failure_lines = "\n".join(f"- {reason}" for reason in failures)
    return "\n".join(
        [
            f"# RunPod GRPO Hillclimb {report['run_id']}",
            "",
            f"- Status: {report['status']}",
            f"- Base model: {report['model']['base_model_id']}",
            f"- Candidate: {report['candidate']['candidate_id']}",
            f"- Steps: {report['training']['max_steps']}",
            f"- Accuracy delta: {report['metrics']['delta']}",
            f"- CI95: {report['metrics']['paired_ci95']}",
            f"- Promotion: {report['promotion']['decision']}",
            "",
            "## Failures",
            "",
            failure_lines,
            "",
            "## Next Experiment",
            "",
            (
                f"- {report['next_experiment']['category']}: "
                f"{report['next_experiment']['rationale']}"
            ),
            "",
        ]
    )


def _event(store: ArtifactStore | None, event: str, payload: Mapping[str, Any]) -> None:
    if store is None:
        return
    path = store.run_dir / "logs" / "events.jsonl"
    existing = read_jsonl(path) if path.is_file() else []
    retained = [row for row in existing if row.get("event") != event]
    write_jsonl(
        path,
        [
            *retained,
            {
                "created_at": datetime.now(UTC).isoformat(),
                "event": event,
                "payload": dict(payload),
            },
        ],
    )


def run_runpod_eval_benchmark(
    config_path: str | Path,
    out_path: str | Path,
) -> dict[str, Any] | None:
    """Certify batched evaluation with warmed, order-balanced runtime evidence."""
    cfg = load_runpod_grpo_config(config_path)
    dist = DistributedContext.from_env()
    _validate_launch_topology(cfg, dist)
    state = _distributed_state(dist)
    runtime_attestation = _require_cuda(cfg)
    cfg = _resolve_hub_revisions(cfg)
    _train, _selection, promotion = _load_and_split_dataset(cfg)
    local_examples = _shard_sequence(promotion, dist)
    scalar_cfg = cfg.model_copy(
        update={"eval": cfg.eval.model_copy(update={"batch_size": 1})}
    )

    def evaluate_scalar() -> list[dict[str, Any]]:
        local_rows = []
        for example in local_examples:
            local_rows.extend(
                _evaluate_hf_model(
                    cfg=scalar_cfg,
                    model_ref=cfg.model.base_model_id,
                    examples=[example],
                    dist=dist,
                )
            )
        return [row.to_json() for row in _gather_eval_rows(local_rows, dist)]

    def evaluate_optimized() -> list[dict[str, Any]]:
        local_rows = _evaluate_hf_model(
            cfg=cfg,
            model_ref=cfg.model.base_model_id,
            examples=local_examples,
            dist=dist,
        )
        return [row.to_json() for row in _gather_eval_rows(local_rows, dist)]

    def synchronize() -> None:
        _wait_for_everyone(state)
        import torch

        torch.cuda.synchronize()

    evidence = measure_runtime_pair(
        baseline=evaluate_scalar,
        optimized=evaluate_optimized,
        synchronize=synchronize,
        reduce_seconds=lambda seconds: _max_rank_seconds(seconds, dist),
    )
    if not dist.is_main_process:
        return None

    optimized_payload = evidence.output
    payload = json.dumps(
        optimized_payload,
        sort_keys=True,
        separators=(",", ":"),
    )
    result: dict[str, Any] = {
        "schema_version": "runpod_eval_runtime_benchmark_v2",
        "certifying": evidence.certifying,
        "config": str(Path(config_path)),
        "model_id": cfg.model.base_model_id,
        "model_revision": cfg.model.resolved_revision,
        "dataset_revision": (
            "embedded-gsm8k-tiny-v1"
            if cfg.dataset.source == "embedded_gsm8k_tiny"
            else cfg.dataset.resolved_revision
        ),
        "topology": dist.to_json(),
        "environment": runtime_environment(dist),
        "runtime_attestation": runtime_attestation,
        "example_count": len(promotion),
        "baseline": {
            "strategy": "one_model_load_per_example",
            "model_load_count_per_rank": len(local_examples),
            "batch_size": 1,
            "max_rank_wall_seconds_trials": list(evidence.baseline_seconds),
        },
        "optimized": {
            "strategy": "one_model_load_per_shard_with_batching",
            "model_load_count_per_rank": 1,
            "batch_size": cfg.eval.batch_size,
            "max_rank_wall_seconds_trials": list(evidence.optimized_seconds),
        },
        "measurement_order": ["baseline", "optimized", "optimized", "baseline"],
        "warmup_strategy": "optimized",
        "minimum_speedup": evidence.minimum_speedup,
        "speedup": evidence.conservative_speedup,
        "output_parity": evidence.output_parity,
        "output_sha256": "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    if not evidence.output_parity:
        raise RuntimeError("batched evaluation output drifted from scalar evaluation")
    if not evidence.certifying:
        raise RuntimeError(
            "conservative paired runtime speedup did not meet the certification margin; "
            "inspect benchmark artifact"
        )
    return result


def _max_rank_seconds(seconds: float, dist: DistributedContext) -> float:
    if not dist.is_distributed:
        return seconds
    import torch
    import torch.distributed as torch_dist

    value = torch.tensor(seconds, device=f"cuda:{dist.local_rank}")
    torch_dist.all_reduce(value, op=torch_dist.ReduceOp.MAX)
    return float(value.item())


__all__ = [
    "RunPodGRPOConfig",
    "DistributedContext",
    "_align_model_with_tokenizer",
    "_as_batch",
    "_candidate_checkpoint_refs",
    "_checkpoint_score",
    "_filter_trl_config_kwargs",
    "_grpo_config_kwargs",
    "_shard_sequence",
    "_select_checkpoint",
    "_validate_hf_upload_config",
    "_validate_grpo_runtime_shape",
    "_validate_launch_topology",
    "is_runpod_grpo_config",
    "load_runpod_grpo_config",
    "run_runpod_grpo_hillclimb",
    "run_runpod_eval_benchmark",
    "runtime_environment",
]
