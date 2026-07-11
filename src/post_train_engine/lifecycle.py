"""Checkpoint lifecycle management for hill-climb flywheels."""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Protocol

from post_train_engine.config import HuggingFaceLifecycleConfig, ModelLifecycleConfig


class CheckpointUploader(Protocol):
    def ensure_repo(
        self,
        *,
        repo_id: str,
        repo_type: str,
        private: bool,
    ) -> None:
        """Create or validate the remote repository."""

    def upload_folder(
        self,
        *,
        local_dir: Path,
        repo_id: str,
        path_in_repo: str,
        repo_type: str,
        commit_message: str,
    ) -> str:
        """Upload a folder and return a durable remote artifact reference."""


class HuggingFaceCheckpointUploader:
    """Thin adapter around ``huggingface_hub.HfApi``."""

    def __init__(self, api: Any | None = None, *, token: str | None = None) -> None:
        if api is None:
            from huggingface_hub import HfApi

            api = HfApi(token=token)
        self.api = api

    def ensure_repo(
        self,
        *,
        repo_id: str,
        repo_type: str,
        private: bool,
    ) -> None:
        self.api.create_repo(
            repo_id=repo_id,
            repo_type=repo_type,
            private=private,
            exist_ok=True,
        )

    def upload_folder(
        self,
        *,
        local_dir: Path,
        repo_id: str,
        path_in_repo: str,
        repo_type: str,
        commit_message: str,
    ) -> str:
        result = self.api.upload_folder(
            folder_path=str(local_dir),
            repo_id=repo_id,
            path_in_repo=path_in_repo,
            repo_type=repo_type,
            commit_message=commit_message,
        )
        return str(getattr(result, "commit_url", None) or f"hf://{repo_id}/{path_in_repo}")


@dataclass(frozen=True)
class CheckpointLifecycleInput:
    candidate_id: str
    checkpoint_ref: str
    task_name: str
    parent_candidate_id: str | None
    parent_checkpoint_ref: str | None
    previous_incumbent_candidate_id: str | None
    previous_incumbent_checkpoint_ref: str | None
    previous_incumbent_remote_ref: str | None
    promoted: bool
    score: float
    incumbent_score: float | None
    metrics: Mapping[str, float]
    evaluation_artifacts: Mapping[str, Any]
    evaluation_metadata: Mapping[str, Any]
    train_artifacts: Mapping[str, Any]
    train_metrics: Mapping[str, float]
    train_metadata: Mapping[str, Any]
    promotion_gate: Mapping[str, Any]
    promotion_decision: Mapping[str, Any] | None = None
    suite_state: Mapping[str, Any] | None = None
    data_overlap_report: Mapping[str, Any] | None = None
    severity_summary: Mapping[str, Any] | None = None
    canary_decision: Mapping[str, Any] | None = None
    rejection_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id must be non-empty")
        if not self.checkpoint_ref:
            raise ValueError("checkpoint_ref must be non-empty")
        if not self.task_name:
            raise ValueError("task_name must be non-empty")
        if self.promoted and self.rejection_reason is not None:
            raise ValueError("promoted checkpoint cannot have a rejection reason")


@dataclass(frozen=True)
class CheckpointLifecycleOutcome:
    remote_ref: str | None
    local_state: str
    artifact_dir: Path
    evidence_path: Path
    local_artifacts: dict[str, str]
    remote_artifacts: dict[str, str]
    discarded_paths: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


class CheckpointLifecycleManager:
    """Persist, upload, and prune one checkpoint decision at a time."""

    def __init__(
        self,
        config: ModelLifecycleConfig | None = None,
        *,
        uploader: CheckpointUploader | None = None,
        date_provider: Callable[[], str | date] | None = None,
    ) -> None:
        self.config = config or ModelLifecycleConfig()
        self.uploader = uploader
        self.date_provider = date_provider or (lambda: date.today().isoformat())
        if self.config.hf.enabled and self.uploader is None:
            self.uploader = HuggingFaceCheckpointUploader()

    def finalize(self, checkpoint: CheckpointLifecycleInput) -> CheckpointLifecycleOutcome:
        hf = self.config.hf
        run_date = _date_string(self.date_provider())
        task_segment = _safe_path_segment(checkpoint.task_name)
        candidate_segment = _safe_path_segment(checkpoint.candidate_id)
        base_path = _format_remote_path(
            hf.path_template,
            task=task_segment,
            date=run_date,
            candidate_id=candidate_segment,
        )
        evidence_dir = self.config.artifact_dir / task_segment / run_date / candidate_segment
        local_artifacts = _write_evidence_bundle(
            evidence_dir,
            self._evidence_body(
                checkpoint,
                run_date=run_date,
                base_path=base_path,
                remote_ref=None,
                remote_artifacts={},
                local_state=_local_state(checkpoint.checkpoint_ref),
                discarded_paths=(),
            ),
        )

        remote_ref: str | None = None
        remote_artifacts: dict[str, str] = {}
        if hf.enabled:
            self._ensure_hf_repo(hf)
            if self._should_upload_checkpoint(checkpoint):
                checkpoint_dir = _required_checkpoint_dir(checkpoint.checkpoint_ref)
                checkpoint_remote = self._upload_folder(
                    local_dir=checkpoint_dir,
                    path_in_repo=f"{base_path}/checkpoint",
                    message=f"Save checkpoint {checkpoint.candidate_id}",
                )
                remote_artifacts["checkpoint"] = checkpoint_remote
                remote_ref = f"hf://{hf.repo_id}/{base_path}"

            body = self._evidence_body(
                checkpoint,
                run_date=run_date,
                base_path=base_path,
                remote_ref=remote_ref,
                remote_artifacts=remote_artifacts,
                local_state=_local_state(checkpoint.checkpoint_ref),
                discarded_paths=(),
            )
            local_artifacts = _write_evidence_bundle(evidence_dir, body)
            if hf.upload_evidence:
                remote_artifacts["evidence"] = self._upload_folder(
                    local_dir=evidence_dir,
                    path_in_repo=f"{base_path}/evidence",
                    message=f"Save lifecycle evidence {checkpoint.candidate_id}",
                )

        discarded_paths = self._discard_after_evidence_upload(
            checkpoint,
            remote_artifacts=remote_artifacts,
        )
        local_state = "discarded" if checkpoint.checkpoint_ref in discarded_paths else _local_state(
            checkpoint.checkpoint_ref,
        )
        body = self._evidence_body(
            checkpoint,
            run_date=run_date,
            base_path=base_path,
            remote_ref=remote_ref,
            remote_artifacts=remote_artifacts,
            local_state=local_state,
            discarded_paths=discarded_paths,
        )
        local_artifacts = _write_evidence_bundle(evidence_dir, body)
        if discarded_paths and hf.enabled and hf.upload_evidence:
            try:
                remote_artifacts["evidence"] = self._upload_folder(
                    local_dir=evidence_dir,
                    path_in_repo=f"{base_path}/evidence",
                    message=f"Record lifecycle discard {checkpoint.candidate_id}",
                )
            except Exception as exc:
                remote_artifacts["discard_receipt_upload_error"] = (
                    f"{type(exc).__name__}: {exc}"
                )
                body = self._evidence_body(
                    checkpoint,
                    run_date=run_date,
                    base_path=base_path,
                    remote_ref=remote_ref,
                    remote_artifacts=remote_artifacts,
                    local_state=local_state,
                    discarded_paths=discarded_paths,
                )
                local_artifacts = _write_evidence_bundle(evidence_dir, body)

        return CheckpointLifecycleOutcome(
            remote_ref=remote_ref,
            local_state=local_state,
            artifact_dir=evidence_dir,
            evidence_path=evidence_dir / "lifecycle.json",
            local_artifacts=local_artifacts,
            remote_artifacts=remote_artifacts,
            discarded_paths=discarded_paths,
            metadata={
                "lifecycle_date": run_date,
                "lifecycle_base_path": base_path,
                "lifecycle_evidence_path": str(evidence_dir / "lifecycle.json"),
            },
        )

    def _ensure_hf_repo(self, hf: HuggingFaceLifecycleConfig) -> None:
        if self.uploader is None or hf.repo_id is None:
            raise ValueError("HF lifecycle upload is enabled but no uploader/repo is configured")
        self.uploader.ensure_repo(
            repo_id=hf.repo_id,
            repo_type=hf.repo_type,
            private=hf.private,
        )

    def _upload_folder(
        self,
        *,
        local_dir: Path,
        path_in_repo: str,
        message: str,
    ) -> str:
        hf = self.config.hf
        if self.uploader is None or hf.repo_id is None:
            raise ValueError("HF lifecycle upload is enabled but no uploader/repo is configured")
        return self.uploader.upload_folder(
            local_dir=local_dir,
            repo_id=hf.repo_id,
            path_in_repo=path_in_repo,
            repo_type=hf.repo_type,
            commit_message=message,
        )

    def _should_upload_checkpoint(self, checkpoint: CheckpointLifecycleInput) -> bool:
        hf = self.config.hf
        return bool(
            hf.enabled
            and (
                (checkpoint.promoted and hf.upload_promoted_checkpoints)
                or (not checkpoint.promoted and hf.upload_rejected_checkpoints)
            )
        )

    def _discard_after_evidence_upload(
        self,
        checkpoint: CheckpointLifecycleInput,
        *,
        remote_artifacts: Mapping[str, str],
    ) -> tuple[str, ...]:
        discarded: list[str] = []
        if self.config.discard_rejected_local and not checkpoint.promoted:
            if self._evidence_uploaded_or_not_required(remote_artifacts):
                if _discard_checkpoint_dir(
                    checkpoint.checkpoint_ref,
                    managed_root=self.config.managed_checkpoint_root,
                ):
                    discarded.append(checkpoint.checkpoint_ref)

        if (
            self.config.keep_only_latest_promoted_local
            and checkpoint.promoted
            and checkpoint.previous_incumbent_checkpoint_ref
            and checkpoint.previous_incumbent_checkpoint_ref != checkpoint.checkpoint_ref
            and self._previous_incumbent_may_be_pruned(checkpoint)
        ):
            if _discard_checkpoint_dir(
                checkpoint.previous_incumbent_checkpoint_ref,
                managed_root=self.config.managed_checkpoint_root,
            ):
                discarded.append(checkpoint.previous_incumbent_checkpoint_ref)

        return tuple(discarded)

    def _evidence_uploaded_or_not_required(
        self,
        remote_artifacts: Mapping[str, str],
    ) -> bool:
        hf = self.config.hf
        if not hf.enabled or not hf.upload_evidence:
            return True
        return "evidence" in remote_artifacts or not self.config.require_hf_evidence_before_discard

    def _previous_incumbent_may_be_pruned(
        self,
        checkpoint: CheckpointLifecycleInput,
    ) -> bool:
        if not self.config.require_hf_checkpoint_before_pruning:
            return True
        return checkpoint.previous_incumbent_remote_ref is not None

    def _evidence_body(
        self,
        checkpoint: CheckpointLifecycleInput,
        *,
        run_date: str,
        base_path: str,
        remote_ref: str | None,
        remote_artifacts: Mapping[str, str],
        local_state: str,
        discarded_paths: tuple[str, ...],
    ) -> dict[str, Any]:
        body = {
            "candidate_id": checkpoint.candidate_id,
            "task": checkpoint.task_name,
            "decision": "promote" if checkpoint.promoted else "reject",
            "promoted": checkpoint.promoted,
            "checkpoint_ref": checkpoint.checkpoint_ref,
            "remote_ref": remote_ref,
            "remote_artifacts": dict(remote_artifacts),
            "local_state": local_state,
            "discarded_paths": list(discarded_paths),
            "created_at": datetime.now(tz=UTC).isoformat(),
            "date": run_date,
            "hf_path": base_path,
            "lineage": {
                "parent_candidate_id": checkpoint.parent_candidate_id,
                "parent_checkpoint_ref": checkpoint.parent_checkpoint_ref,
                "previous_incumbent_candidate_id": checkpoint.previous_incumbent_candidate_id,
                "previous_incumbent_checkpoint_ref": checkpoint.previous_incumbent_checkpoint_ref,
                "previous_incumbent_remote_ref": checkpoint.previous_incumbent_remote_ref,
            },
            "eval_performance": {
                "score": checkpoint.score,
                "incumbent_score": checkpoint.incumbent_score,
                "metrics": dict(checkpoint.metrics),
                "artifacts": dict(checkpoint.evaluation_artifacts),
                "metadata": dict(checkpoint.evaluation_metadata),
            },
            "costs": _extract_costs(checkpoint.train_metrics, checkpoint.train_metadata),
            "training": {
                "metrics": dict(checkpoint.train_metrics),
                "artifacts": dict(checkpoint.train_artifacts),
                "metadata": dict(checkpoint.train_metadata),
            },
            "promotion_gate": dict(checkpoint.promotion_gate),
            "promotion_decision": (
                None
                if checkpoint.promotion_decision is None
                else dict(checkpoint.promotion_decision)
            ),
            "suite_state": (
                None if checkpoint.suite_state is None else dict(checkpoint.suite_state)
            ),
            "data_overlap_report": (
                None
                if checkpoint.data_overlap_report is None
                else dict(checkpoint.data_overlap_report)
            ),
            "severity_summary": (
                None
                if checkpoint.severity_summary is None
                else dict(checkpoint.severity_summary)
            ),
            "canary_decision": (
                None
                if checkpoint.canary_decision is None
                else dict(checkpoint.canary_decision)
            ),
            "rejection_reason": checkpoint.rejection_reason,
        }
        return _jsonable(body)


def _write_evidence_bundle(evidence_dir: Path, body: Mapping[str, Any]) -> dict[str, str]:
    evidence_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "lifecycle": evidence_dir / "lifecycle.json",
        "eval_performance": evidence_dir / "eval_performance.json",
        "costs": evidence_dir / "costs.json",
        "promotion_gate": evidence_dir / "promotion_gate.json",
        "lineage": evidence_dir / "lineage.json",
    }
    optional_artifacts = {
        "promotion_decision": "promotion_decision.json",
        "suite_state": "suite_state.json",
        "data_overlap_report": "data_overlap_report.json",
        "severity_summary": "severity_summary.json",
        "canary_decision": "canary_decision.json",
    }
    for key, file_name in optional_artifacts.items():
        if body.get(key) is not None:
            artifacts[key] = evidence_dir / file_name
    _write_json(artifacts["lifecycle"], body)
    _write_json(artifacts["eval_performance"], body["eval_performance"])
    _write_json(artifacts["costs"], body["costs"])
    _write_json(artifacts["promotion_gate"], body["promotion_gate"])
    _write_json(artifacts["lineage"], body["lineage"])
    for key in optional_artifacts:
        if key in artifacts:
            _write_json(artifacts[key], body[key])
    local_artifacts = {name: str(path) for name, path in artifacts.items()}
    promotion_eval = body["eval_performance"]["artifacts"].get("promotion_eval_artifact")
    if isinstance(promotion_eval, Mapping):
        promotion_eval_path = evidence_dir / "promotion_eval_artifact.json"
        _write_json(promotion_eval_path, promotion_eval)
        local_artifacts["promotion_eval_artifact"] = str(promotion_eval_path)
    return local_artifacts


def _write_json(path: Path, body: Any) -> None:
    path.write_text(
        json.dumps(_jsonable(body), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _extract_costs(
    train_metrics: Mapping[str, float],
    train_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    raw_costs = train_metadata.get("costs")
    if isinstance(raw_costs, Mapping):
        return dict(raw_costs)
    metric_costs = {
        name: value
        for name, value in train_metrics.items()
        if name.endswith("_usd") or "cost" in name
    }
    if metric_costs:
        return metric_costs
    return {"source": "not_reported"}


def _format_remote_path(template: str, **values: str) -> str:
    try:
        rendered = template.format(**values)
    except KeyError as exc:
        raise ValueError(f"unknown lifecycle HF path template field: {exc.args[0]}") from exc
    return "/".join(segment for segment in rendered.replace("\\", "/").split("/") if segment)


def _safe_path_segment(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")
    if not safe:
        raise ValueError(f"cannot derive a safe path segment from {value!r}")
    return safe


def _date_string(value: str | date) -> str:
    return value.isoformat() if isinstance(value, date) else str(value)


def _local_state(checkpoint_ref: str) -> str:
    path = Path(checkpoint_ref)
    if not path.exists():
        return "reference_only"
    return "available" if path.is_dir() else "file"


def _required_checkpoint_dir(checkpoint_ref: str) -> Path:
    path = Path(checkpoint_ref)
    if not path.is_dir():
        raise ValueError(f"checkpoint path must be an existing directory: {checkpoint_ref}")
    _assert_checkpoint_like_dir(path)
    return path


def _discard_checkpoint_dir(
    checkpoint_ref: str,
    *,
    managed_root: Path | None,
) -> bool:
    path = Path(checkpoint_ref)
    if not path.exists():
        return False
    if not path.is_dir():
        raise ValueError(f"refusing to discard non-directory checkpoint: {checkpoint_ref}")
    _assert_safe_discard_path(path, managed_root=managed_root)
    shutil.rmtree(path)
    return True


def _assert_safe_discard_path(path: Path, *, managed_root: Path | None) -> None:
    if managed_root is None:
        raise ValueError("checkpoint deletion requires managed_checkpoint_root")
    resolved = path.resolve()
    resolved_root = managed_root.resolve()
    forbidden = {Path.cwd().resolve(), Path.home().resolve(), Path(resolved.anchor)}
    if resolved in forbidden or resolved.parent == resolved:
        raise ValueError(f"refusing to discard unsafe checkpoint path: {path}")
    if path.is_symlink() or getattr(path, "is_junction", lambda: False)():
        raise ValueError(f"refusing to discard linked checkpoint path: {path}")
    if resolved == resolved_root or not resolved.is_relative_to(resolved_root):
        raise ValueError(
            f"refusing to discard checkpoint outside managed checkpoint root: {path}"
        )
    _assert_checkpoint_like_dir(path)


def _assert_checkpoint_like_dir(path: Path) -> None:
    resolved = path.resolve()
    checkpoint_markers = (
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
        "adapter_model.safetensors",
        "checkpoint_manifest.json",
        "trainer_state.json",
    )
    has_marker = any((resolved / marker).exists() for marker in checkpoint_markers)
    has_sharded_weights = any(resolved.glob("*.safetensors")) or any(
        resolved.glob("pytorch_model-*.bin"),
    )
    if not has_marker and not has_sharded_weights:
        raise ValueError(f"refusing to manage non-checkpoint directory: {path}")


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    return value
