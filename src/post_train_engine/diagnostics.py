"""Deterministic diagnostics for flywheel run bundles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from post_train_engine.artifacts import validate_run_bundle


def write_run_diagnostics(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    artifact_status = validate_run_bundle(run_dir, write=True)
    manifest = _read_json(run_dir / "manifest.json")
    if manifest["status"] == "failed":
        evidence_path = _artifact_path(run_dir, manifest, "failure")
        failure = _read_json(evidence_path)
        reasons = [
            f"{failure.get('error_type', 'stage_failure')} at {failure.get('stage', 'unknown')}"
        ]
        primary_category = "stage_failure"
        evidence_name = "failure"
    else:
        evidence_path = _artifact_path(run_dir, manifest, "promotion_decision")
        promotion = _read_json(evidence_path)
        reasons = [str(reason) for reason in promotion.get("rejection_reasons", [])]
        primary_category = _primary_category(reasons)
        evidence_name = "promotion_decision"
    diagnostics = {
        "run_id": manifest["run_id"],
        "candidate_id": manifest["candidate_id"],
        "primary_category": primary_category,
        "evidence": reasons,
        "artifact_status": artifact_status["status"],
        "artifact_failures": list(artifact_status["failures"]),
        "artifact_refs": {
            "manifest": str(run_dir / "manifest.json"),
            evidence_name: str(evidence_path),
            "artifact_status": str(run_dir / "artifact_status.json"),
        },
    }
    runpod_plan_path = run_dir / "runpod_plan.json"
    if runpod_plan_path.is_file():
        diagnostics["artifact_refs"]["runpod_plan"] = str(runpod_plan_path)
    _write_json(run_dir / "diagnostics.json", diagnostics)
    return diagnostics


def _primary_category(reasons: list[str]) -> str:
    for reason in reasons:
        if reason.startswith("primary_delta"):
            return "under_min_delta"
    for reason in reasons:
        if reason.startswith("primary_ci_low"):
            return "underpowered_eval"
    for reason in reasons:
        if reason.startswith("parse_regression"):
            return "parse_regression"
    for reason in reasons:
        if reason.startswith("easy_regression"):
            return "easy_slice_regression"
    return "accepted" if not reasons else "promotion_gate_failed"


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


def _write_json(path: Path, body: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(body, indent=2, sort_keys=True),
        encoding="utf-8",
    )
