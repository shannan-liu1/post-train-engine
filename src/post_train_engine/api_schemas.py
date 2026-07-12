"""API-first hill-climb schemas.

These models describe the remote-compute orchestration layer. They are
deliberately separate from the legacy local ``ExperimentConfig``/GPU-runner
contracts so the core hill-climb CLI can orchestrate providers without assuming
CUDA, vLLM, Transformers, or a local checkpoint directory.
"""

from __future__ import annotations

import math
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from post_train_engine.engine import CampaignBinding

_FROZEN_FORBID = ConfigDict(frozen=True, extra="forbid")

DatasetSource = Literal["embedded_gsm8k_tiny", "huggingface_gsm8k"]
PromptStyle = Literal["plain", "chat", "thinking_tags"]
ProviderType = Literal[
    "fake",
    "fake_prompt_adapter",
    "chat_completions",
    "chat_completions_prompt_adapter",
]
JobType = Literal["rollout_generation", "candidate_adaptation", "evaluation"]
JobState = Literal["queued", "running", "succeeded", "failed", "timeout", "cancelled"]
SplitRole = Literal["train", "eval", "promotion", "probe"]


class RunSection(BaseModel):
    model_config = _FROZEN_FORBID

    run_id: str = Field(..., min_length=1)
    certification_mode: Literal["non_certifying_smoke", "certifying"]
    campaign: CampaignBinding | None = None
    output_dir: str = Field(..., min_length=1)
    seed: int = Field(default=123, ge=0)
    overwrite: bool = False

    @model_validator(mode="after")
    def _certification_has_authority(self) -> RunSection:
        if self.certification_mode == "certifying" and self.campaign is None:
            raise ValueError("certifying run requires campaign binding")
        if self.certification_mode == "non_certifying_smoke" and self.campaign is not None:
            raise ValueError("non-certifying smoke cannot bind a campaign")
        return self


class DatasetSpec(BaseModel):
    model_config = _FROZEN_FORBID

    name: Literal["gsm8k"] = "gsm8k"
    source: DatasetSource
    train_size: int = Field(..., gt=0)
    eval_size: int = Field(..., gt=0)
    split_seed: int = Field(default=123, ge=0)
    prompt_style: PromptStyle = "thinking_tags"
    dataset_name: str = "openai/gsm8k"
    dataset_revision: str | None = None


class Candidate(BaseModel):
    model_config = _FROZEN_FORBID

    candidate_id: str = Field(..., min_length=1)
    model_id: str = Field(..., min_length=1)
    parent_id: str | None = None
    system_prompt: str = ""
    prompt_prefix: str = ""
    prompt_suffix: str = ""
    adapter_kind: str = "base"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("parent_id")
    @classmethod
    def _parent_id_must_be_nonempty(cls, value: str | None) -> str | None:
        if value == "":
            raise ValueError("parent_id must be non-empty when provided")
        return value

    def to_json(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class CandidateSpec(BaseModel):
    model_config = _FROZEN_FORBID

    candidate_id: str = Field(..., min_length=1)
    model_id: str | None = None
    model_id_env: str | None = None
    system_prompt: str = ""
    prompt_prefix: str = ""
    prompt_suffix: str = ""

    @model_validator(mode="after")
    def _must_have_model_source(self) -> CandidateSpec:
        if not self.model_id and not self.model_id_env:
            raise ValueError("candidate requires model_id or model_id_env")
        return self


class ProviderSpec(BaseModel):
    model_config = _FROZEN_FORBID

    type: ProviderType
    provider_id: str = Field(..., min_length=1)
    base_url: str | None = None
    base_url_env: str | None = None
    api_key_env: str | None = None
    model: str | None = None
    model_env: str | None = None
    timeout_seconds: float = Field(default=60.0, gt=0.0)
    max_tokens_field: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"

    @model_validator(mode="after")
    def _provider_contract(self) -> ProviderSpec:
        if is_chat_completions_provider_type(self.type):
            if not self.api_key_env:
                raise ValueError(f"{self.type} requires api_key_env")
            if not self.base_url and not self.base_url_env:
                raise ValueError(f"{self.type} requires base_url or base_url_env")
            if not self.model and not self.model_env:
                raise ValueError(f"{self.type} requires model or model_env")
        return self


class ProviderBundleSpec(BaseModel):
    model_config = _FROZEN_FORBID

    inference: ProviderSpec
    training: ProviderSpec


class GenerationSpec(BaseModel):
    model_config = _FROZEN_FORBID

    samples_per_example: int = Field(default=1, gt=0)
    temperature: float = Field(default=0.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    max_output_tokens: int = Field(default=256, gt=0)


class PromotionPolicy(BaseModel):
    model_config = _FROZEN_FORBID

    primary_metric: Literal["accuracy"] = "accuracy"
    min_accuracy_delta: float = 0.0
    min_paired_delta_ci_low: float = 0.0
    min_eval_examples: int = Field(default=30, gt=0)
    max_mcnemar_p: float = Field(default=1.0, ge=0.0, le=1.0)
    max_parse_regression: float = Field(default=0.0, ge=0.0)
    max_easy_regression: float = Field(default=0.0, ge=0.0)
    max_token_increase_ratio: float = Field(default=1.25, gt=0.0)

    @field_validator(
        "min_accuracy_delta",
        "min_paired_delta_ci_low",
        "max_mcnemar_p",
        "max_parse_regression",
        "max_easy_regression",
        "max_token_increase_ratio",
    )
    @classmethod
    def _finite_float(cls, value: float) -> float:
        if type(value) is bool or not isinstance(value, int | float) or not math.isfinite(float(value)):
            raise ValueError("promotion numeric fields must be finite")
        return float(value)


class HillClimbConfig(BaseModel):
    model_config = _FROZEN_FORBID

    run: RunSection
    dataset: DatasetSpec
    baseline: CandidateSpec
    providers: ProviderBundleSpec
    rollout: GenerationSpec
    eval: GenerationSpec
    promotion: PromotionPolicy


class PromptRequest(BaseModel):
    model_config = _FROZEN_FORBID

    example_id: str = Field(..., min_length=1)
    prompt: str = Field(..., min_length=1)
    sample_index: int = Field(..., ge=0)
    split_role: SplitRole


class JobRequest(BaseModel):
    model_config = _FROZEN_FORBID

    job_id: str = Field(..., min_length=1)
    job_type: JobType
    provider_id: str = Field(..., min_length=1)
    payload: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class JobHandle(BaseModel):
    model_config = _FROZEN_FORBID

    job_id: str = Field(..., min_length=1)
    job_type: JobType
    provider_id: str = Field(..., min_length=1)
    provider_job_id: str = Field(..., min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class JobStatus(BaseModel):
    model_config = _FROZEN_FORBID

    state: JobState
    message: str | None = None

    @property
    def terminal(self) -> bool:
        return self.state in {"succeeded", "failed", "timeout", "cancelled"}

    def to_json(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class JobResult(BaseModel):
    model_config = _FROZEN_FORBID

    handle: JobHandle
    status: JobStatus
    payload: dict[str, Any]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _result_must_be_terminal(self) -> JobResult:
        if not self.status.terminal:
            raise ValueError("job result status must be terminal")
        return self

    def to_json(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def to_redacted_json(self) -> dict[str, Any]:
        return redact_secrets(self.to_json())


class EvalExampleRecord(BaseModel):
    model_config = _FROZEN_FORBID

    example_id: str = Field(..., min_length=1)
    prompt: str = Field(..., min_length=1)
    completion: str
    parsed_answer: str | None
    gold_answer: str = Field(..., min_length=1)
    correct: bool
    parse_ok: bool
    completion_tokens: int = Field(..., ge=0)
    finish_reason: str | None = None
    provider_job_id: str | None = None
    sample_index: int = Field(default=0, ge=0)


class EvalResult(BaseModel):
    model_config = _FROZEN_FORBID

    candidate_id: str = Field(..., min_length=1)
    metrics: dict[str, float]
    examples: tuple[EvalExampleRecord, ...]

    @field_validator("metrics", mode="before")
    @classmethod
    def _metrics_are_finite(cls, value: Any) -> Any:
        metrics = dict(value or {})
        if not metrics:
            raise ValueError("eval result requires metrics")
        for name, metric in metrics.items():
            if (
                not name
                or type(metric) is bool
                or not isinstance(metric, int | float)
                or not math.isfinite(float(metric))
            ):
                raise ValueError(f"metric {name!r} must be finite")
        return metrics

    def to_json(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


_SECRET_KEY_FRAGMENTS = ("API_KEY", "ACCESS_KEY", "SECRET", "PASSWORD", "AUTHORIZATION")
_SAFE_TOKEN_KEYS = {
    "completion_tokens",
    "input_tokens",
    "max_output_tokens",
    "output_tokens",
    "prompt_tokens",
    "token_counts",
    "total_tokens",
}


def redact_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            key_str = str(key)
            if _looks_secret_key(key_str):
                redacted[key_str] = "[REDACTED]"
            else:
                redacted[key_str] = redact_secrets(item)
        return redacted
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    if isinstance(value, tuple):
        return [redact_secrets(item) for item in value]
    if isinstance(value, str):
        return redact_secret_text(value)
    return value


def redact_secret_text(text: str) -> str:
    text = re.sub(
        r"(?i)\b(api[_-]?key|access[_-]?key|secret|password|authorization|token)\s*[:=]\s*[^,\s;]+",
        "[REDACTED]",
        text,
    )
    return re.sub(r"(?i)Bearer\s+[A-Za-z0-9._\-]+", "Bearer [REDACTED]", text)


def _looks_secret_key(key: str) -> bool:
    if key.lower() in _SAFE_TOKEN_KEYS or key.lower().endswith("_tokens"):
        return False
    upper = key.upper()
    return (
        any(fragment in upper for fragment in _SECRET_KEY_FRAGMENTS)
        or upper == "TOKEN"
        or upper.endswith("_TOKEN")
    )


def is_chat_completions_provider_type(provider_type: str) -> bool:
    return provider_type in {
        "chat_completions",
        "chat_completions_prompt_adapter",
    }


__all__ = [
    "Candidate",
    "CandidateSpec",
    "DatasetSpec",
    "EvalExampleRecord",
    "EvalResult",
    "GenerationSpec",
    "HillClimbConfig",
    "JobHandle",
    "JobRequest",
    "JobResult",
    "JobStatus",
    "PromotionPolicy",
    "ProviderSpec",
    "is_chat_completions_provider_type",
    "redact_secret_text",
    "redact_secrets",
]
