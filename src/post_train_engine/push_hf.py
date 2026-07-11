"""Dry-run Hugging Face push plans for flywheel evidence bundles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from post_train_engine.artifacts import require_valid_run_bundle


def write_hf_push_plan(
    run_dir: str | Path,
    *,
    repo_id: str,
    dry_run: bool,
    repo_type: str = "model",
    private: bool = False,
) -> dict[str, Any]:
    """Write an auditable HF upload plan without performing network I/O."""

    if not dry_run:
        raise RuntimeError("push-hf currently supports --dry-run only")
    run_dir = Path(run_dir)
    artifact_status = require_valid_run_bundle(run_dir)
    manifest = _read_json(run_dir / "manifest.json")
    lifecycle_path = _artifact_path(run_dir, manifest, "lifecycle")
    lifecycle = _read_json(lifecycle_path)
    promotion = _read_json(_artifact_path(run_dir, manifest, "promotion_decision"))
    hf_path = str(lifecycle.get("hf_path") or "")
    if not hf_path:
        raise ValueError("lifecycle artifact missing hf_path")

    promoted = _required_bool(lifecycle, "promoted")
    _validate_promotion_consistency(
        promoted=promoted,
        manifest=manifest,
        promotion=promotion,
    )
    evidence_dir = lifecycle_path.parent
    targets = [
        {
            "role": "evidence",
            "will_upload": True,
            "local_path": str(evidence_dir),
            "path_in_repo": f"{hf_path}/evidence",
            "reason": "always_upload_evidence",
        },
        {
            "role": "checkpoint",
            "will_upload": promoted,
            "local_path": str(
                manifest.get("checkpoint_ref")
                or manifest.get("metadata", {}).get("checkpoint_ref")
                or ""
            ),
            "path_in_repo": f"{hf_path}/checkpoint",
            "reason": "candidate_promoted" if promoted else "candidate_not_promoted",
        },
    ]
    plan = {
        "dry_run": True,
        "repo_id": repo_id,
        "repo_type": repo_type,
        "private": private,
        "run_id": manifest["run_id"],
        "candidate_id": manifest["candidate_id"],
        "artifact_status": artifact_status["status"],
        "artifact_status_path": str(run_dir / "artifact_status.json"),
        "targets": targets,
    }
    _write_json(run_dir / "push_plan.json", plan)
    return plan


def _artifact_path(run_dir: Path, manifest: dict[str, Any], name: str) -> Path:
    try:
        path = Path(manifest["artifacts"][name]["path"])
    except KeyError as exc:
        raise ValueError(f"manifest missing artifact {name!r}") from exc
    return path if path.is_absolute() else run_dir / path


def _read_json(path: str | Path) -> dict[str, Any]:
    body = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(body, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return body


def _required_bool(body: dict[str, Any], name: str) -> bool:
    value = body.get(name)
    if type(value) is not bool:
        raise ValueError(f"lifecycle {name} flag must be a boolean")
    return value


def _validate_promotion_consistency(
    *,
    promoted: bool,
    manifest: dict[str, Any],
    promotion: dict[str, Any],
) -> None:
    manifest_promoted = manifest.get("status") == "promoted"
    decision_promoted = promotion.get("decision") == "promote"
    if promoted != manifest_promoted or promoted != decision_promoted:
        raise ValueError(
            "promoted lifecycle conflicts with manifest status or promotion decision",
        )


def _write_json(path: Path, body: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(body, indent=2, sort_keys=True),
        encoding="utf-8",
    )


__all__ = ["write_hf_push_plan"]
