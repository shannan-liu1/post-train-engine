"""Common method-runner protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from post_train_engine.config import ExperimentConfig
from post_train_engine.training_views import TrainingViewArtifact


@dataclass(frozen=True)
class RunResult:
    candidate_id: str
    checkpoint_path: str
    metrics: dict[str, float]
    artifacts: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MethodTrainingRequest:
    """Only canonical input accepted by a post-training method adapter."""

    config: ExperimentConfig
    training_view: TrainingViewArtifact
    artifact_root: Path | None = None

    def __post_init__(self) -> None:
        method = self.config.method.name
        if method not in self.training_view.method_compatibility:
            raise ValueError(
                f"TrainingView {self.training_view.view_id!r} is not compatible with {method}"
            )
        configured_path = self.config.data.train_path
        if configured_path is not None:
            view_path = self.training_view.resolve_data_path(
                self.artifact_root or Path.cwd()
            )
            if Path(configured_path).resolve() != view_path.resolve():
                raise ValueError(
                    "config data.train_path must match the TrainingView data artifact"
                )


class MethodRunner(Protocol):
    name: str

    def train(self, request: MethodTrainingRequest) -> RunResult:
        """Train one candidate checkpoint."""


__all__ = ["MethodRunner", "MethodTrainingRequest", "RunResult"]
