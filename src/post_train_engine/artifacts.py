"""Artifact bundle validation for flywheel run directories."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from post_train_engine.jsonl import read_jsonl
from post_train_engine.run_bundle import CANONICAL_RUN_SCHEMA_VERSION, RunBundle


@dataclass(frozen=True)
class ArtifactStatus:
    name: str
    kind: str
    path: str
    required: bool
    status: str
    exists: bool
    expected_sha256: str | None = None
    actual_sha256: str | None = None
    message: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def to_dict(self) -> dict[str, Any]:
        body = {
            "name": self.name,
            "kind": self.kind,
            "path": self.path,
            "required": self.required,
            "status": self.status,
            "exists": self.exists,
            "expected_sha256": self.expected_sha256,
            "actual_sha256": self.actual_sha256,
        }
        if self.message:
            body["message"] = self.message
        return body


def validate_run_bundle(
    run_dir: str | Path,
    *,
    write: bool = True,
) -> dict[str, Any]:
    """Validate the required artifact files referenced by ``manifest.json``."""

    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    manifest = _read_manifest(manifest_path)
    if manifest.get("schema_version") == CANONICAL_RUN_SCHEMA_VERSION:
        try:
            body = RunBundle.load(run_dir).validate()
        except ValidationError as exc:
            failure = {
                "name": "manifest",
                "kind": "manifest",
                "path": "manifest.json",
                "required": True,
                "status": "malformed",
                "exists": True,
                "expected_sha256": None,
                "actual_sha256": _sha256(manifest_path),
                "message": str(exc),
            }
            body = {
                "run_id": str(manifest.get("run_id", "")),
                "candidate_id": str(manifest.get("candidate_id", "")),
                "status": "failed",
                "required_count": 1,
                "ok_count": 0,
                "failure_count": 1,
                "failures": [failure],
                "artifacts": [failure],
            }
        if write:
            _write_json(run_dir / "artifact_status.json", body)
        return body
    artifacts = [
        _manifest_status(manifest_path),
        *[
            _artifact_status(name, ref)
            for name, ref in sorted(dict(manifest.get("artifacts", {})).items())
        ],
    ]
    artifacts.extend(_semantic_statuses(manifest))
    required = [artifact for artifact in artifacts if artifact.required]
    failures = [
        artifact.to_dict()
        for artifact in required
        if not artifact.ok
    ]
    body = {
        "run_id": str(manifest.get("run_id", "")),
        "candidate_id": str(manifest.get("candidate_id", "")),
        "status": "ok" if not failures else "failed",
        "required_count": len(required),
        "ok_count": sum(artifact.ok for artifact in required),
        "failure_count": len(failures),
        "failures": failures,
        "artifacts": [artifact.to_dict() for artifact in artifacts],
    }
    if write:
        _write_json(run_dir / "artifact_status.json", body)
    return body


def require_valid_run_bundle(run_dir: str | Path) -> dict[str, Any]:
    """Validate a run bundle and raise with evidence-linked failures."""

    status = validate_run_bundle(run_dir, write=True)
    if status["status"] != "ok":
        failed = ", ".join(
            _format_failure(failure)
            for failure in status["failures"]
        )
        raise ValueError(f"artifact bundle validation failed: {failed}")
    return status


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"manifest does not exist: {path}") from exc
    if not isinstance(body, dict):
        raise ValueError(f"manifest JSON root must be an object: {path}")
    return body


def _manifest_status(path: Path) -> ArtifactStatus:
    if not path.is_file():
        return ArtifactStatus(
            name="manifest",
            kind="manifest",
            path=str(path),
            required=True,
            status="missing",
            exists=False,
            message="manifest.json is missing",
        )
    digest = _sha256(path)
    return ArtifactStatus(
        name="manifest",
        kind="manifest",
        path=str(path),
        required=True,
        status="ok",
        exists=True,
        actual_sha256=digest,
        expected_sha256=digest,
    )


def _artifact_status(name: str, ref: Any) -> ArtifactStatus:
    if not isinstance(ref, dict):
        return ArtifactStatus(
            name=name,
            kind="unknown",
            path="",
            required=True,
            status="malformed",
            exists=False,
            message="artifact reference must be an object",
        )
    path = Path(str(ref.get("path", "")))
    kind = str(ref.get("kind") or name)
    raw_required = ref.get("required", True)
    if type(raw_required) is not bool:
        return ArtifactStatus(
            name=name,
            kind=kind,
            path=str(path),
            required=True,
            status="malformed",
            exists=path.is_file(),
            message="artifact required flag must be a boolean",
        )
    required = raw_required
    expected = ref.get("sha256")
    if not isinstance(expected, str) or not expected.startswith("sha256:"):
        return ArtifactStatus(
            name=name,
            kind=kind,
            path=str(path),
            required=required,
            status="malformed",
            exists=path.is_file(),
            message="artifact reference missing sha256:<digest>",
        )
    if not path.is_file():
        return ArtifactStatus(
            name=name,
            kind=kind,
            path=str(path),
            required=required,
            status="missing",
            exists=False,
            expected_sha256=expected,
        )
    actual = _sha256(path)
    if actual != expected:
        return ArtifactStatus(
            name=name,
            kind=kind,
            path=str(path),
            required=required,
            status="hash_mismatch",
            exists=True,
            expected_sha256=expected,
            actual_sha256=actual,
        )
    return ArtifactStatus(
        name=name,
        kind=kind,
        path=str(path),
        required=required,
        status="ok",
        exists=True,
        expected_sha256=expected,
        actual_sha256=actual,
    )


def _format_failure(failure: dict[str, Any]) -> str:
    label = f"{failure['name']}:{failure['status']}"
    message = failure.get("message")
    if message:
        return f"{label}({message})"
    return label


def _semantic_statuses(manifest: dict[str, Any]) -> list[ArtifactStatus]:
    return [
        _promotion_consistency_status(manifest),
        _training_view_leakage_status(manifest),
        _grpo_reward_evidence_status(manifest),
    ]


def _promotion_consistency_status(manifest: dict[str, Any]) -> ArtifactStatus:
    if manifest.get("status") in {"planned", "running", "failed"}:
        return _semantic_ok("promotion_consistency")
    try:
        promotion = _read_artifact_json(manifest, "promotion_decision")
        if manifest.get("schema_version") == "api_hillclimb_run_v1":
            return _api_promotion_consistency_status(manifest, promotion)
        lifecycle = _read_artifact_json(manifest, "lifecycle")
        promoted = lifecycle.get("promoted")
        if type(promoted) is not bool:
            return _semantic_failure(
                "promotion_consistency",
                "lifecycle promoted flag must be a boolean",
            )
        manifest_promoted = manifest.get("status") == "promoted"
        decision_promoted = promotion.get("decision") == "promote"
        if promoted != manifest_promoted or promoted != decision_promoted:
            return _semantic_failure(
                "promotion_consistency",
                "promoted lifecycle conflicts with manifest status or promotion decision",
            )
    except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
        return _semantic_failure("promotion_consistency", str(exc))
    return _semantic_ok("promotion_consistency")


def _api_promotion_consistency_status(
    manifest: dict[str, Any],
    promotion: dict[str, Any],
) -> ArtifactStatus:
    decision = promotion.get("decision")
    if decision not in {"promote", "reject"}:
        return _semantic_failure(
            "promotion_consistency",
            "promotion decision must be promote or reject",
        )
    expected_status = "promoted" if decision == "promote" else "rejected"
    if manifest.get("status") != expected_status:
        return _semantic_failure(
            "promotion_consistency",
            "manifest status conflicts with promotion decision",
        )
    report = _read_artifact_json(manifest, "final_report_json")
    report_promotion = report.get("promotion")
    if not isinstance(report_promotion, dict):
        return _semantic_failure(
            "promotion_consistency",
            "final report promotion must be an object",
        )
    if report_promotion.get("decision") != decision:
        return _semantic_failure(
            "promotion_consistency",
            "final report conflicts with promotion decision",
        )
    return _semantic_ok("promotion_consistency")


def _training_view_leakage_status(manifest: dict[str, Any]) -> ArtifactStatus:
    try:
        for name, ref in dict(manifest.get("artifacts", {})).items():
            if not _is_training_view_artifact(name, ref):
                continue
            view = _read_json(Path(str(ref["path"])))
            trace_ids = view.get("source_trace_ids")
            roles = view.get("source_split_roles")
            method_compatibility = view.get("method_compatibility")
            if not isinstance(trace_ids, list) or not trace_ids:
                return _semantic_failure(
                    "training_view_leakage",
                    f"{name} missing source_trace_ids",
                )
            if not isinstance(roles, list) or not roles:
                return _semantic_failure(
                    "training_view_leakage",
                    f"{name} missing source_split_roles",
                )
            if "promotion" in roles:
                return _semantic_failure(
                    "training_view_leakage",
                    f"{name} uses promotion split sources",
                )
            if not isinstance(method_compatibility, list) or not method_compatibility:
                return _semantic_failure(
                    "training_view_leakage",
                    f"{name} missing method_compatibility",
                )
    except (OSError, json.JSONDecodeError, ValueError, TypeError, KeyError) as exc:
        return _semantic_failure("training_view_leakage", str(exc))
    return _semantic_ok("training_view_leakage")


def _grpo_reward_evidence_status(manifest: dict[str, Any]) -> ArtifactStatus:
    artifacts = dict(manifest.get("artifacts", {}))
    if "grpo_rollout_view" not in artifacts:
        return _semantic_ok("grpo_reward_evidence")
    try:
        groups_path = Path(str(artifacts["rollout_groups"]["path"]))
        groups = read_jsonl(groups_path)
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        return _semantic_failure("grpo_reward_evidence", str(exc))
    if not groups:
        return _semantic_failure("grpo_reward_evidence", "missing rollout groups")
    has_reward_evidence = False
    has_non_degenerate_group = False
    for group in groups:
        rewards = group.get("rewards")
        variance = group.get("reward_variance")
        if isinstance(rewards, list) and rewards and _finite_number(variance):
            has_reward_evidence = True
        if group.get("degenerate_group") is False:
            has_non_degenerate_group = True
    if not has_reward_evidence:
        return _semantic_failure(
            "grpo_reward_evidence",
            "GRPO rollout view requires reward evidence",
        )
    if not has_non_degenerate_group:
        return _semantic_failure(
            "grpo_reward_evidence",
            "GRPO rollout view requires at least one non-degenerate reward group",
        )
    return _semantic_ok("grpo_reward_evidence")


def _semantic_ok(name: str) -> ArtifactStatus:
    return ArtifactStatus(
        name=name,
        kind="semantic",
        path=f"semantic://{name}",
        required=True,
        status="ok",
        exists=True,
    )


def _semantic_failure(name: str, message: str) -> ArtifactStatus:
    return ArtifactStatus(
        name=name,
        kind="semantic",
        path=f"semantic://{name}",
        required=True,
        status="semantic_failed",
        exists=True,
        message=message,
    )


def _is_training_view_artifact(name: str, ref: Any) -> bool:
    if name.endswith("_view"):
        return True
    return isinstance(ref, dict) and str(ref.get("kind", "")).endswith("_view")


def _read_artifact_json(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    try:
        path = Path(str(manifest["artifacts"][name]["path"]))
    except KeyError as exc:
        raise ValueError(f"manifest missing artifact {name!r}") from exc
    return _read_json(path)


def _read_json(path: Path) -> dict[str, Any]:
    body = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(body, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return body


def _finite_number(value: Any) -> bool:
    return (
        type(value) is not bool
        and isinstance(value, int | float)
        and math.isfinite(float(value))
    )


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, body: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(body, indent=2, sort_keys=True),
        encoding="utf-8",
    )


__all__ = [
    "ArtifactStatus",
    "require_valid_run_bundle",
    "validate_run_bundle",
]
