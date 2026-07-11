"""Validated experiment configuration for post-training climbs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

_FROZEN_FORBID = ConfigDict(frozen=True, extra="forbid")

DType = Literal["bfloat16", "float32", "float16"]
HFRepoType = Literal["model", "dataset", "space"]
MethodName = Literal[
    "sft",
    "dpo",
    "grpo",
    "grpo_vanilla",
    "opsd",
    "reward_model",
    "custom",
]
OptimizerFramework = Literal["muon", "adamw"]


class ModelConfig(BaseModel):
    model_config = _FROZEN_FORBID

    base_model_id: str = Field(..., min_length=1)
    dtype: DType = "bfloat16"
    use_safetensors: bool = True
    trust_remote_code: bool = False
    gradient_checkpointing: bool = False
    attn_implementation: str | None = None


class TaskConfig(BaseModel):
    model_config = _FROZEN_FORBID

    name: str = Field(..., min_length=1)
    split: str = "train"
    seed: int = Field(default=42, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MethodConfig(BaseModel):
    model_config = _FROZEN_FORBID

    name: MethodName
    parameters: dict[str, Any] = Field(default_factory=dict)


class DataConfig(BaseModel):
    model_config = _FROZEN_FORBID

    train_path: Path | None = None
    eval_path: Path | None = None
    max_seq_len: int = Field(default=4096, gt=1)
    val_split_pct: float = Field(default=5.0, ge=0.0, lt=100.0)
    seed: int = Field(default=42, ge=0)


class TrainingConfig(BaseModel):
    model_config = _FROZEN_FORBID

    max_steps: int = Field(..., gt=0)
    lr: float = Field(..., gt=0.0)
    warmup_steps: int = Field(default=0, ge=0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    grad_accum_steps: int = Field(default=1, ge=1)
    grad_clip: float = Field(default=1.0, gt=0.0)
    per_device_batch_size: int = Field(default=1, ge=1)
    eval_every_n_steps: int = Field(default=100, gt=0)
    checkpoint_every_n_steps: int = Field(default=100, gt=0)

    @model_validator(mode="after")
    def _warmup_must_be_less_than_max(self) -> TrainingConfig:
        if self.warmup_steps >= self.max_steps:
            raise ValueError(
                f"warmup_steps ({self.warmup_steps}) must be < max_steps "
                f"({self.max_steps})"
            )
        return self


class OptimizerConfig(BaseModel):
    model_config = _FROZEN_FORBID

    framework: OptimizerFramework = "muon"
    hidden_lr: float | None = Field(default=None, gt=0.0)
    aux_lr: float | None = Field(default=None, gt=0.0)
    aux_betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float | None = Field(default=None, ge=0.0)


class EvalConfig(BaseModel):
    model_config = _FROZEN_FORBID

    source: str = Field(..., min_length=1)
    max_examples: int | None = Field(default=None, gt=0)
    seed: int = Field(default=42, ge=0)
    metrics: list[str] = Field(default_factory=lambda: ["accuracy"])
    constraints: dict[str, float] = Field(default_factory=dict)


class LoggingConfig(BaseModel):
    model_config = _FROZEN_FORBID

    wandb_project: str = "post-train-engine"
    run_name: str | None = None
    tags: list[str] = Field(default_factory=list)


class CheckpointConfig(BaseModel):
    model_config = _FROZEN_FORBID

    save_dir: Path = Path("results/checkpoints")
    retention_last_n: int = Field(default=3, ge=0)
    resume_from: Path | None = None


class HuggingFaceLifecycleConfig(BaseModel):
    model_config = _FROZEN_FORBID

    enabled: bool = False
    repo_id: str | None = Field(default=None, min_length=1)
    repo_type: HFRepoType = "model"
    private: bool = True
    path_template: str = "tasks/{task}/checkpoints/{date}/{candidate_id}"
    upload_evidence: bool = True
    upload_promoted_checkpoints: bool = True
    upload_rejected_checkpoints: bool = False

    @model_validator(mode="after")
    def _enabled_requires_repo(self) -> HuggingFaceLifecycleConfig:
        if self.enabled and not self.repo_id:
            raise ValueError("hf.repo_id is required when lifecycle HF upload is enabled")
        required_fields = ("{task}", "{date}", "{candidate_id}")
        missing = [field for field in required_fields if field not in self.path_template]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"hf.path_template must include {joined}")
        return self


class ModelLifecycleConfig(BaseModel):
    model_config = _FROZEN_FORBID

    artifact_dir: Path = Path("results/lifecycle")
    managed_checkpoint_root: Path | None = None
    discard_rejected_local: bool = True
    keep_only_latest_promoted_local: bool = True
    require_hf_evidence_before_discard: bool = True
    require_hf_checkpoint_before_pruning: bool = True
    hf: HuggingFaceLifecycleConfig = Field(default_factory=HuggingFaceLifecycleConfig)


class ExperimentConfig(BaseModel):
    """Top-level config for one candidate training/eval run."""

    model_config = _FROZEN_FORBID

    model: ModelConfig
    task: TaskConfig
    method: MethodConfig
    training: TrainingConfig
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    eval: EvalConfig
    data: DataConfig = Field(default_factory=DataConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    checkpointing: CheckpointConfig = Field(default_factory=CheckpointConfig)
    lifecycle: ModelLifecycleConfig = Field(default_factory=ModelLifecycleConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        path = Path(path)
        with path.open("r", encoding="utf-8") as fp:
            raw = yaml.safe_load(fp)
        if not isinstance(raw, dict):
            raise ValueError(f"YAML root must be a mapping; got {type(raw).__name__}")
        return cls.model_validate(raw)

    def to_yaml(self, path: str | Path) -> None:
        path = Path(path)
        with path.open("w", encoding="utf-8") as fp:
            yaml.safe_dump(
                self.model_dump(mode="json"),
                fp,
                sort_keys=False,
                default_flow_style=False,
            )
