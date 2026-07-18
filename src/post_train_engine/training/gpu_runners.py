"""Concrete GPU-backed method runners.

These runners are thin adapters over Hugging Face Transformers and TRL. They do
not own promotion, probing, or difficulty logic; they only turn an
``ExperimentConfig`` into one trained checkpoint.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from post_train_engine.config import ExperimentConfig
from post_train_engine.training.grpo_config import configured_modified_grpo_knobs
from post_train_engine.training.optimizers import build_optimizer
from post_train_engine.training.runner import MethodTrainingRequest, RunResult


RewardFunc = Callable[..., float]


@dataclass(frozen=True)
class _GpuRunnerBase:
    allow_cpu: bool = False
    extra_trainer_kwargs: Mapping[str, Any] = field(default_factory=dict)

    name: str = field(init=False, default="")

    def train(self, request: MethodTrainingRequest) -> RunResult:
        if not isinstance(request, MethodTrainingRequest):
            raise TypeError(
                "method runners require a MethodTrainingRequest with TrainingView"
            )
        config = request.config
        self._validate_config(config)
        request.training_view.require_data_integrity(
            request.artifact_root or Path.cwd()
        )
        torch = self._require_torch()
        result = self._train_after_validation(config, torch)
        return RunResult(
            candidate_id=result.candidate_id,
            checkpoint_path=result.checkpoint_path,
            metrics=result.metrics,
            artifacts=result.artifacts,
            metadata={
                **result.metadata,
                "training_view_id": request.training_view.view_id,
                "source_trace_ids": list(request.training_view.source_trace_ids),
                "source_split_roles": list(request.training_view.source_split_roles),
            },
        )

    def _validate_config(self, config: ExperimentConfig) -> None:
        if config.method.name != self.name:
            raise ValueError(
                f"{self.__class__.__name__} expected method {self.name}; "
                f"got {config.method.name}"
            )
        if config.data.train_path is None:
            raise ValueError(f"{self.name} runner requires data.train_path")

    def _require_torch(self) -> Any:
        import torch

        if not self.allow_cpu and not torch.cuda.is_available():
            raise RuntimeError(
                f"{self.name} GPU runner requires CUDA; pass allow_cpu=True only "
                "for local smoke tests"
            )
        return torch

    def _training_arguments(self, config: ExperimentConfig) -> Any:
        from transformers import TrainingArguments

        candidate_id = _candidate_id(config)
        return TrainingArguments(
            output_dir=str(config.checkpointing.save_dir / candidate_id),
            max_steps=config.training.max_steps,
            learning_rate=config.training.lr,
            warmup_steps=config.training.warmup_steps,
            weight_decay=config.training.weight_decay,
            gradient_accumulation_steps=config.training.grad_accum_steps,
            max_grad_norm=config.training.grad_clip,
            per_device_train_batch_size=config.training.per_device_batch_size,
            save_steps=config.training.checkpoint_every_n_steps,
            eval_steps=config.training.eval_every_n_steps,
            logging_steps=max(1, min(config.training.eval_every_n_steps, 25)),
            bf16=config.model.dtype == "bfloat16",
            fp16=config.model.dtype == "float16",
            report_to=[],
            remove_unused_columns=False,
        )

    def _load_model_and_tokenizer(
        self, config: ExperimentConfig, torch: Any
    ) -> tuple[Any, Any]:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[config.model.dtype]
        tokenizer = AutoTokenizer.from_pretrained(
            config.model.base_model_id,
            trust_remote_code=config.model.trust_remote_code,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            config.model.base_model_id,
            torch_dtype=dtype,
            trust_remote_code=config.model.trust_remote_code,
            use_safetensors=config.model.use_safetensors,
            attn_implementation=config.model.attn_implementation,
        )
        if config.model.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        return model, tokenizer

    def _load_json_dataset(self, config: ExperimentConfig) -> Any:
        from datasets import load_dataset

        train_path = config.data.train_path
        if train_path is None:
            raise ValueError(f"{self.name} runner requires data.train_path")
        return load_dataset("json", data_files=str(train_path), split="train")

    def _maybe_apply_lora(self, model: Any, config: ExperimentConfig) -> Any:
        lora = config.method.parameters.get("lora")
        if not lora:
            return model
        from peft import LoraConfig, get_peft_model

        if not isinstance(lora, dict):
            raise ValueError("method.parameters.lora must be a mapping")
        return get_peft_model(model, LoraConfig(**lora))

    def _finish(self, trainer: Any, config: ExperimentConfig) -> RunResult:
        result = trainer.train()
        output_dir = config.checkpointing.save_dir / _candidate_id(config)
        trainer.save_model(str(output_dir))
        metrics = _numeric_metrics(getattr(result, "metrics", {}))
        return RunResult(
            candidate_id=_candidate_id(config),
            checkpoint_path=str(output_dir),
            metrics=metrics,
            artifacts={"trainer": trainer.__class__.__name__},
            metadata={
                "method": self.name,
                "train_data_path": str(config.data.train_path),
                "train_overlap_certification": "not_provided",
            },
        )

    def _optimizer_tuple(
        self, model: Any, config: ExperimentConfig, torch: Any
    ) -> tuple[Any, None]:
        return build_optimizer(model, config.optimizer, config.training, torch), None

    def _train_after_validation(
        self, config: ExperimentConfig, torch: Any
    ) -> RunResult:
        raise NotImplementedError


@dataclass(frozen=True)
class SFTGpuRunner(_GpuRunnerBase):
    name: str = field(init=False, default="sft")

    def _train_after_validation(
        self, config: ExperimentConfig, torch: Any
    ) -> RunResult:
        from trl import SFTTrainer

        model, tokenizer = self._load_model_and_tokenizer(config, torch)
        model = self._maybe_apply_lora(model, config)
        dataset = self._load_json_dataset(config)
        args = self._training_arguments(config)
        params = config.method.parameters
        trainer_kwargs = {
            "model": model,
            "args": args,
            "train_dataset": dataset,
            "dataset_text_field": params.get("dataset_text_field", "text"),
            "optimizers": self._optimizer_tuple(model, config, torch),
            **dict(self.extra_trainer_kwargs),
        }
        trainer = _build_trainer(SFTTrainer, tokenizer, trainer_kwargs)
        return self._finish(trainer, config)


@dataclass(frozen=True)
class DPOGpuRunner(_GpuRunnerBase):
    name: str = field(init=False, default="dpo")

    def _train_after_validation(
        self, config: ExperimentConfig, torch: Any
    ) -> RunResult:
        from trl import DPOTrainer

        model, tokenizer = self._load_model_and_tokenizer(config, torch)
        model = self._maybe_apply_lora(model, config)
        dataset = self._load_json_dataset(config)
        args = self._training_arguments(config)
        trainer_kwargs = {
            "model": model,
            "args": args,
            "train_dataset": dataset,
            "beta": config.method.parameters.get("beta", 0.1),
            "optimizers": self._optimizer_tuple(model, config, torch),
            **dict(self.extra_trainer_kwargs),
        }
        trainer = _build_trainer(DPOTrainer, tokenizer, trainer_kwargs)
        return self._finish(trainer, config)


@dataclass(frozen=True)
class GRPOGpuRunner(_GpuRunnerBase):
    reward_funcs: Sequence[RewardFunc] | None = None
    name: str = field(init=False, default="grpo")

    def _validate_config(self, config: ExperimentConfig) -> None:
        super()._validate_config(config)
        if not self.reward_funcs:
            raise ValueError("GRPOGpuRunner requires reward_funcs")
        unsupported = configured_modified_grpo_knobs(config.method.parameters)
        if unsupported:
            raise ValueError(
                "GRPOGpuRunner cannot silently ignore modified-GRPO knobs: "
                + ", ".join(unsupported)
            )

    def _train_after_validation(
        self, config: ExperimentConfig, torch: Any
    ) -> RunResult:
        from trl import GRPOTrainer

        model, tokenizer = self._load_model_and_tokenizer(config, torch)
        model = self._maybe_apply_lora(model, config)
        dataset = self._load_json_dataset(config)
        args = self._training_arguments(config)
        trainer_kwargs = {
            "model": model,
            "args": args,
            "train_dataset": dataset,
            "reward_funcs": list(self.reward_funcs or ()),
            "optimizers": self._optimizer_tuple(model, config, torch),
            **dict(self.extra_trainer_kwargs),
        }
        trainer = _build_trainer(GRPOTrainer, tokenizer, trainer_kwargs)
        return self._finish(trainer, config)


def default_gpu_runners(
    *,
    allow_cpu: bool = False,
    grpo_reward_funcs: Sequence[RewardFunc] | None = None,
) -> dict[str, _GpuRunnerBase]:
    runners: list[_GpuRunnerBase] = [
        SFTGpuRunner(allow_cpu=allow_cpu),
        DPOGpuRunner(allow_cpu=allow_cpu),
        GRPOGpuRunner(allow_cpu=allow_cpu, reward_funcs=grpo_reward_funcs),
    ]
    return {runner.name: runner for runner in runners}


def _candidate_id(config: ExperimentConfig) -> str:
    if config.logging.run_name:
        return config.logging.run_name
    return f"{config.task.name}-{config.method.name}"


def _numeric_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in metrics.items():
        if type(value) is not bool and isinstance(value, int | float):
            out[key] = float(value)
    return out


def _build_trainer(
    trainer_cls: type[Any], tokenizer: Any, kwargs: dict[str, Any]
) -> Any:
    try:
        return trainer_cls(processing_class=tokenizer, **kwargs)
    except TypeError as exc:
        if "unexpected keyword argument 'processing_class'" not in str(exc):
            raise
        return trainer_cls(tokenizer=tokenizer, **kwargs)
