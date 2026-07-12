"""Cost, cache, topology, and policy-staleness evidence contracts."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field, model_validator

_FROZEN_FORBID = ConfigDict(frozen=True, extra="forbid")
_CACHE_KEYS = frozenset({"model", "suite", "generation", "verifier"})


@dataclass(frozen=True)
class RuntimePairEvidence:
    baseline_seconds: tuple[float, float]
    optimized_seconds: tuple[float, float]
    conservative_speedup: float
    baseline_output_parity: tuple[bool, bool]
    optimized_output_parity: tuple[bool, bool]
    output_parity: bool
    certifying: bool
    minimum_speedup: float
    output: Any


def measure_runtime_pair(
    *,
    baseline: Callable[[], Any],
    optimized: Callable[[], Any],
    synchronize: Callable[[], None],
    reduce_seconds: Callable[[float], float],
    minimum_speedup: float = 1.05,
    clock: Callable[[], float] = time.perf_counter,
) -> RuntimePairEvidence:
    """Measure one warmed ABBA pair and certify only its worst-case speedup."""
    if not math.isfinite(minimum_speedup) or minimum_speedup <= 1.0:
        raise ValueError("minimum_speedup must be finite and greater than one")

    warm_output = optimized()
    observations: dict[str, list[tuple[float, Any]]] = {
        "baseline": [],
        "optimized": [],
    }
    for name, operation in (
        ("baseline", baseline),
        ("optimized", optimized),
        ("optimized", optimized),
        ("baseline", baseline),
    ):
        synchronize()
        started = clock()
        output = operation()
        synchronize()
        seconds = reduce_seconds(clock() - started)
        if not math.isfinite(seconds) or seconds <= 0.0:
            raise ValueError("runtime trial duration must be positive and finite")
        observations[name].append((seconds, output))

    baseline_seconds = tuple(value[0] for value in observations["baseline"])
    optimized_seconds = tuple(value[0] for value in observations["optimized"])
    baseline_output_parity = tuple(
        output == warm_output for _seconds, output in observations["baseline"]
    )
    optimized_output_parity = tuple(
        output == warm_output for _seconds, output in observations["optimized"]
    )
    output_parity = all((*baseline_output_parity, *optimized_output_parity))
    conservative_speedup = min(baseline_seconds) / max(optimized_seconds)
    return RuntimePairEvidence(
        baseline_seconds=(baseline_seconds[0], baseline_seconds[1]),
        optimized_seconds=(optimized_seconds[0], optimized_seconds[1]),
        conservative_speedup=conservative_speedup,
        baseline_output_parity=(
            baseline_output_parity[0],
            baseline_output_parity[1],
        ),
        optimized_output_parity=(
            optimized_output_parity[0],
            optimized_output_parity[1],
        ),
        output_parity=output_parity,
        certifying=output_parity and conservative_speedup >= minimum_speedup,
        minimum_speedup=minimum_speedup,
        output=warm_output,
    )


class PhaseCostRecord(BaseModel):
    model_config = _FROZEN_FORBID

    phase: str = Field(..., min_length=1)
    duration_seconds: float = Field(..., ge=0.0)
    resource: str = Field(..., min_length=1)
    resource_count: float = Field(default=1.0, ge=0.0)
    unit_price_usd: float | None = Field(default=None, ge=0.0)
    measured_cost_usd: float | None = Field(default=None, ge=0.0)
    missing_reason: str | None = None

    @model_validator(mode="after")
    def _cost_must_be_measured_or_explicitly_missing(self) -> PhaseCostRecord:
        if self.measured_cost_usd is None and self.unit_price_usd is None:
            if not self.missing_reason:
                raise ValueError("cost requires a price, measured value, or missing_reason")
        elif self.missing_reason:
            raise ValueError("measured cost evidence cannot also be marked missing")
        return self

    @property
    def cost_usd(self) -> float | None:
        if self.measured_cost_usd is not None:
            return self.measured_cost_usd
        if self.unit_price_usd is None:
            return None
        return self.duration_seconds / 3600.0 * self.resource_count * self.unit_price_usd


def summarize_costs(
    records: tuple[PhaseCostRecord, ...],
    *,
    candidates: int,
    useful_traces: int,
    evaluations: int,
    promoted_metric_gain: float,
) -> dict[str, Any]:
    for name, value in {
        "candidates": candidates,
        "useful_traces": useful_traces,
        "evaluations": evaluations,
    }.items():
        if type(value) is bool or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if not math.isfinite(promoted_metric_gain) or promoted_metric_gain < 0:
        raise ValueError("promoted_metric_gain must be finite and non-negative")
    measured = sum(record.cost_usd or 0.0 for record in records)
    missing = [record.phase for record in records if record.cost_usd is None]
    return {
        "certifying": not missing,
        "missing_cost_phases": missing,
        "measured_cost_usd": measured,
        "cost_per_candidate_usd": _ratio(measured, candidates),
        "cost_per_useful_trace_usd": _ratio(measured, useful_traces),
        "cost_per_evaluation_usd": _ratio(measured, evaluations),
        "cost_per_promoted_metric_gain_usd": _ratio(
            measured,
            promoted_metric_gain,
        ),
        "phases": [record.model_dump(mode="json") for record in records],
    }


class EvaluationCache:
    """Content-addressed cache that accepts only an exact evaluation contract."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def get(self, contract: dict[str, str]) -> dict[str, Any] | None:
        path = self._path(contract)
        if not path.is_file():
            return None
        body = json.loads(path.read_text(encoding="utf-8"))
        if body.get("contract") != contract or not isinstance(body.get("value"), dict):
            raise ValueError(f"evaluation cache entry is malformed: {path}")
        return dict(body["value"])

    def put(self, contract: dict[str, str], value: dict[str, Any]) -> Path:
        path = self._path(contract)
        temporary = path.with_name("." + path.name + ".tmp")
        temporary.write_text(
            json.dumps({"contract": contract, "value": value}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(path)
        return path

    def _path(self, contract: dict[str, str]) -> Path:
        if set(contract) != _CACHE_KEYS or any(not value for value in contract.values()):
            raise ValueError(
                "evaluation cache contract requires model, suite, generation, and verifier"
            )
        payload = json.dumps(contract, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return self.root / f"{digest}.json"


class ExecutionTopology(BaseModel):
    model_config = _FROZEN_FORBID

    modeled_world_size: int = Field(..., gt=0)
    executed_world_size: int = Field(..., gt=0)
    launcher: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def _executed_topology_must_match_plan(self) -> ExecutionTopology:
        if self.modeled_world_size != self.executed_world_size:
            raise ValueError(
                "executed topology does not match modeled topology: "
                f"{self.executed_world_size} != {self.modeled_world_size}"
            )
        return self


class PolicyUse(BaseModel):
    model_config = _FROZEN_FORBID

    generated_policy_step: int = Field(..., ge=0)
    consumed_policy_step: int = Field(..., ge=0)
    max_staleness_steps: int = Field(..., ge=0)

    @model_validator(mode="after")
    def _staleness_must_fit_budget(self) -> PolicyUse:
        staleness = self.consumed_policy_step - self.generated_policy_step
        if staleness < 0:
            raise ValueError("policy evidence cannot come from a future policy step")
        if staleness > self.max_staleness_steps:
            raise ValueError(
                f"policy staleness {staleness} exceeds {self.max_staleness_steps} steps"
            )
        return self


def _ratio(numerator: float, denominator: int | float) -> float | None:
    return None if denominator == 0 else numerator / denominator


__all__ = [
    "EvaluationCache",
    "ExecutionTopology",
    "PhaseCostRecord",
    "PolicyUse",
    "RuntimePairEvidence",
    "measure_runtime_pair",
    "summarize_costs",
]
