"""Cost and resource-topology evidence for canonical Runs."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)

_FROZEN_FORBID = ConfigDict(frozen=True, extra="forbid")

Launcher = Literal["local", "torchrun", "accelerate", "runpod"]
ShardRole = Literal["train", "rollout", "eval", "artifact_merge"]


class PriceSnapshot(BaseModel):
    """Versioned pricing input used for deterministic cost attribution."""

    model_config = _FROZEN_FORBID

    snapshot_id: str = Field(..., min_length=1)
    provider: str = Field(..., min_length=1)
    accelerator_type: str = Field(..., min_length=1)
    accelerator_hour_usd: float = Field(..., ge=0.0)
    currency: str = Field(default="USD", min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("accelerator_hour_usd", mode="before")
    @classmethod
    def _reject_boolean_price(cls, value: Any) -> Any:
        _reject_bool(value, "accelerator_hour_usd")
        return value


class CostRecord(BaseModel):
    """Measured usage and deterministic cost estimate for one run segment."""

    model_config = _FROZEN_FORBID

    price_snapshot_id: str = Field(..., min_length=1)
    provider: str = Field(..., min_length=1)
    accelerator_type: str = Field(..., min_length=1)
    accelerator_count: int = Field(..., ge=0)
    wall_seconds: float = Field(..., ge=0.0)
    accelerator_seconds: float = Field(..., ge=0.0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    checkpoint_bytes: int = Field(default=0, ge=0)
    estimated_usd: float = Field(..., ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "accelerator_count",
        "wall_seconds",
        "accelerator_seconds",
        "input_tokens",
        "output_tokens",
        "checkpoint_bytes",
        "estimated_usd",
        mode="before",
    )
    @classmethod
    def _reject_boolean_usage(cls, value: Any, info: Any) -> Any:
        _reject_bool(value, info.field_name)
        return value

    @classmethod
    def from_usage(
        cls,
        price: PriceSnapshot,
        *,
        accelerator_count: int,
        wall_seconds: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        checkpoint_bytes: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> CostRecord:
        accelerator_seconds = float(accelerator_count) * float(wall_seconds)
        estimated_usd = accelerator_seconds / 3600.0 * price.accelerator_hour_usd
        return cls(
            price_snapshot_id=price.snapshot_id,
            provider=price.provider,
            accelerator_type=price.accelerator_type,
            accelerator_count=accelerator_count,
            wall_seconds=wall_seconds,
            accelerator_seconds=accelerator_seconds,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            checkpoint_bytes=checkpoint_bytes,
            estimated_usd=estimated_usd,
            metadata=metadata or {},
        )


class ResourceTopology(BaseModel):
    """Execution topology for training, rollout, eval, or artifact jobs."""

    model_config = _FROZEN_FORBID

    launcher: Launcher = "local"
    num_nodes: int = Field(default=1, ge=1)
    gpus_per_node: int = Field(default=0, ge=0)
    gpu_type: str | None = None
    data_parallel_size: int = Field(default=1, ge=1)
    tensor_parallel_size: int = Field(default=1, ge=1)
    pipeline_parallel_size: int = Field(default=1, ge=1)
    fsdp: str | None = None
    deepspeed_zero_stage: int | None = Field(default=None, ge=0, le=3)
    cpu_offload: bool = False
    max_memory_gb_per_gpu: float | None = Field(default=None, gt=0.0)

    @field_validator(
        "num_nodes",
        "gpus_per_node",
        "data_parallel_size",
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "deepspeed_zero_stage",
        "max_memory_gb_per_gpu",
        mode="before",
    )
    @classmethod
    def _reject_boolean_topology(cls, value: Any, info: Any) -> Any:
        _reject_bool(value, info.field_name)
        return value

    @computed_field
    @property
    def total_gpus(self) -> int:
        return self.num_nodes * self.gpus_per_node

    @model_validator(mode="after")
    def _parallelism_must_fit_topology(self) -> ResourceTopology:
        parallel_product = (
            self.data_parallel_size
            * self.tensor_parallel_size
            * self.pipeline_parallel_size
        )
        if self.total_gpus == 0 and parallel_product != 1:
            raise ValueError("parallelism > 1 requires at least one GPU")
        if self.total_gpus and parallel_product > self.total_gpus:
            raise ValueError("parallelism product cannot exceed total_gpus")
        return self


class ShardInfo(BaseModel):
    """Identity and deterministic assignment for one distributed work shard."""

    model_config = _FROZEN_FORBID

    role: ShardRole
    shard_id: int = Field(..., ge=0)
    world_size: int = Field(..., ge=1)
    rank: int = Field(..., ge=0)
    example_ids: tuple[str, ...] = ()
    rollout_ids: tuple[int, ...] = ()
    seed: int = Field(..., ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("shard_id", "world_size", "rank", "seed", mode="before")
    @classmethod
    def _reject_boolean_shard_numbers(cls, value: Any, info: Any) -> Any:
        _reject_bool(value, info.field_name)
        return value

    @model_validator(mode="after")
    def _rank_and_shard_must_fit_world(self) -> ShardInfo:
        if self.shard_id >= self.world_size:
            raise ValueError("shard_id must be < world_size")
        if self.rank >= self.world_size:
            raise ValueError("rank must be < world_size")
        if len(set(self.example_ids)) != len(self.example_ids):
            raise ValueError("duplicate example id within shard")
        if len(set(self.rollout_ids)) != len(self.rollout_ids):
            raise ValueError("duplicate rollout id within shard")
        return self


class ShardPlan(BaseModel):
    """Complete shard assignment for one flywheel role."""

    model_config = _FROZEN_FORBID

    role: ShardRole
    world_size: int = Field(..., ge=1)
    shards: tuple[ShardInfo, ...]

    @field_validator("world_size", mode="before")
    @classmethod
    def _reject_boolean_world_size(cls, value: Any) -> Any:
        _reject_bool(value, "world_size")
        return value

    @computed_field
    @property
    def covered_example_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                example_id
                for shard in self.shards
                for example_id in shard.example_ids
            )
        )

    @model_validator(mode="after")
    def _validate_complete_disjoint_plan(self) -> ShardPlan:
        shard_ids = [shard.shard_id for shard in self.shards]
        expected = set(range(self.world_size))
        if set(shard_ids) != expected:
            raise ValueError(
                f"expected shard ids {sorted(expected)}, got {sorted(set(shard_ids))}"
            )
        if len(shard_ids) != len(set(shard_ids)):
            raise ValueError("duplicate shard id")
        seen_examples: set[str] = set()
        for shard in self.shards:
            if shard.role != self.role:
                raise ValueError("shard role must match plan role")
            if shard.world_size != self.world_size:
                raise ValueError("shard world_size must match plan world_size")
            for example_id in shard.example_ids:
                if example_id in seen_examples:
                    raise ValueError(f"duplicate example id across shards: {example_id}")
                seen_examples.add(example_id)
        return self


def _reject_bool(value: Any, field_name: str) -> None:
    if type(value) is bool:
        raise ValueError(f"{field_name} must not be a boolean")


__all__ = [
    "CostRecord",
    "PriceSnapshot",
    "ResourceTopology",
    "ShardInfo",
    "ShardPlan",
]
