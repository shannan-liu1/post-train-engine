"""Typed method-specific training-view artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    field_validator,
    model_validator,
)

from post_train_engine.traces.schema import SplitRole

TrainingViewType = Literal[
    "sft",
    "preference",
    "grpo_rollout",
    "opd",
    "multi_teacher_opd",
    "opsd",
    "answer_only_distillation",
    "reward_model",
]
PrivilegedVisibility = Literal[
    "none",
    "gold_answer",
    "verifier_feedback",
    "privileged_context",
    "environment",
    "critic",
    "unknown",
]

_FROZEN_FORBID = ConfigDict(frozen=True, extra="forbid")
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class TrainingDataRef(BaseModel):
    """Content identity for the data projected by one TrainingView."""

    model_config = _FROZEN_FORBID

    path: str = Field(..., min_length=1)
    kind: str = Field(..., min_length=1)
    sha256: str
    required: StrictBool = True

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        kind: str,
        artifact_root: str | Path | None = None,
        required: bool = True,
    ) -> TrainingDataRef:
        data_path = Path(path)
        if not data_path.is_file():
            raise ValueError(
                f"training data path must be an existing file: {data_path}"
            )
        if artifact_root is None:
            raise ValueError("training data path must be relative to artifact_root")
        try:
            stored_path = (
                data_path.resolve()
                .relative_to(Path(artifact_root).resolve())
                .as_posix()
            )
        except ValueError as exc:
            raise ValueError(
                "training data path must remain inside artifact_root"
            ) from exc
        digest = hashlib.sha256()
        with data_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return cls(
            path=stored_path,
            kind=kind,
            sha256=f"sha256:{digest.hexdigest()}",
            required=required,
        )

    @field_validator("path")
    @classmethod
    def _path_must_be_relative_and_contained(cls, value: str) -> str:
        posix = PurePosixPath(value.replace("\\", "/"))
        if (
            PurePosixPath(value).is_absolute()
            or PureWindowsPath(value).is_absolute()
            or ".." in posix.parts
        ):
            raise ValueError("training data path must be relative to artifact_root")
        return value

    @field_validator("sha256")
    @classmethod
    def _sha256_must_be_canonical(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError(
                "training data sha256 must use sha256:<64 lowercase hex chars>"
            )
        return value


class TrainingViewArtifact(BaseModel):
    """Immutable method-specific projection over trace evidence."""

    model_config = _FROZEN_FORBID

    view_id: str = Field(..., min_length=1)
    run_id: str = Field(..., min_length=1)
    task_id: str = Field(..., min_length=1)
    view_type: TrainingViewType
    method_compatibility: tuple[str, ...] = Field(..., min_length=1)
    data_artifact: TrainingDataRef
    source_trace_ids: tuple[str, ...] = Field(..., min_length=1)
    source_split_roles: tuple[SplitRole, ...] = Field(..., min_length=1)
    privileged_visibility: PrivilegedVisibility = "unknown"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("method_compatibility", "source_trace_ids", mode="before")
    @classmethod
    def _string_tuple_must_be_nonempty_unique(cls, value: Any, info: Any) -> Any:
        if isinstance(value, str):
            raise ValueError(f"{info.field_name} must be a sequence, not a string")
        items = tuple(value or ())
        if not items:
            raise ValueError(f"{info.field_name} must be non-empty")
        if any(not isinstance(item, str) or not item for item in items):
            raise ValueError(f"{info.field_name} entries must be non-empty strings")
        if len(set(items)) != len(items):
            raise ValueError(f"{info.field_name} entries must be unique")
        return items

    @field_validator("source_split_roles", mode="before")
    @classmethod
    def _source_split_roles_must_be_nonempty_unique(cls, value: Any) -> Any:
        if isinstance(value, str):
            raise ValueError("source_split_roles must be a sequence, not a string")
        roles = tuple(value or ())
        if not roles:
            raise ValueError("source_split_roles must be non-empty")
        if len(set(roles)) != len(roles):
            raise ValueError("source_split_roles entries must be unique")
        return roles

    @model_validator(mode="after")
    def _validate_training_view_contract(self) -> TrainingViewArtifact:
        protected = {
            "selection",
            "diagnostic",
            "promotion",
            "canary",
            "unseen",
        }.intersection(self.source_split_roles)
        if protected:
            raise ValueError(
                "training views cannot use protected evaluation split sources: "
                + ", ".join(sorted(protected))
            )
        if self.view_type == "grpo_rollout" and "grpo" not in self.method_compatibility:
            raise ValueError("grpo_rollout views must be compatible with grpo")
        if self.view_type == "opsd" and self.privileged_visibility == "none":
            raise ValueError("opsd views must declare privileged visibility")
        if (
            self.view_type == "multi_teacher_opd"
            and "multi_teacher_opd" not in self.method_compatibility
        ):
            raise ValueError(
                "multi_teacher_opd views must declare multi_teacher_opd compatibility"
            )
        return self

    def to_json(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def resolve_data_path(self, artifact_root: str | Path) -> Path:
        """Resolve one canonical relative reference through its artifact root."""

        return Path(artifact_root) / self.data_artifact.path

    def require_data_integrity(self, artifact_root: str | Path) -> Path:
        """Resolve and hash-check the immutable data immediately before use."""

        root = Path(artifact_root).resolve()
        path = self.resolve_data_path(root).resolve()
        if not path.is_relative_to(root):
            raise ValueError("training data path escapes artifact_root")
        if not path.is_file():
            if self.data_artifact.required:
                raise ValueError(f"required training data file is missing: {path}")
            return path
        actual = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != self.data_artifact.sha256:
            raise ValueError(
                "training data sha256 mismatch: "
                f"expected {self.data_artifact.sha256}, got {actual}"
            )
        return path


def write_training_view_artifact(
    view: TrainingViewArtifact,
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(view.to_json(), indent=2, sort_keys=True),
        encoding="utf-8",
    )


__all__ = [
    "PrivilegedVisibility",
    "TrainingDataRef",
    "TrainingViewArtifact",
    "TrainingViewType",
    "write_training_view_artifact",
]
