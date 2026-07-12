"""Run report writers for flywheel bundles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from post_train_engine.artifacts import validate_run_bundle
from post_train_engine.run_bundle import RunBundle


def write_run_report(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    artifact_status = validate_run_bundle(run_dir, write=True)
    bundle = RunBundle.load(run_dir)
    manifest = _read_json(run_dir / "manifest.json")
    if manifest["status"] == "failed":
        failure = _read_json(bundle.verified_artifact_path("failure"))
        promotion = {
            "decision": "failed",
            "primary_metric": None,
            "primary_delta": None,
            "rejection_reasons": [
                f"{failure.get('error_type', 'stage_failure')} at {failure.get('stage', 'unknown')}"
            ],
        }
    else:
        promotion = _read_json(bundle.verified_artifact_path("promotion_decision"))
    summary = {
        "run_id": manifest["run_id"],
        "candidate_id": manifest["candidate_id"],
        "task_name": manifest["task_name"],
        "model_id": manifest["model_id"],
        "status": manifest["status"],
        "promotion_decision": promotion["decision"],
        "primary_metric": promotion["primary_metric"],
        "primary_delta": promotion["primary_delta"],
        "rejection_reasons": list(promotion.get("rejection_reasons", [])),
        "scores": dict(
            manifest.get("scores", manifest.get("metadata", {}).get("scores", {})),
        ),
        "artifact_status": artifact_status["status"],
        "required_artifact_count": artifact_status["required_count"],
        "ok_artifact_count": artifact_status["ok_count"],
        "artifact_failures": list(artifact_status["failures"]),
        "artifacts": {
            name: ref["path"]
            for name, ref in dict(manifest.get("artifacts", {})).items()
            if ref.get("visibility", "standard") != "sealed"
        },
    }
    runpod_plan_path = run_dir / "runpod_plan.json"
    if runpod_plan_path.is_file():
        summary["runpod_plan"] = str(runpod_plan_path)
    _write_json(run_dir / "summary.json", summary)
    (run_dir / "report.md").write_text(_markdown_report(summary), encoding="utf-8")
    return summary


def _markdown_report(summary: dict[str, Any]) -> str:
    reasons = summary["rejection_reasons"]
    reason_lines = "\n".join(f"- {reason}" for reason in reasons) or "- none"
    return "\n".join(
        [
            f"# Run {summary['run_id']}",
            "",
            f"- Candidate: {summary['candidate_id']}",
            f"- Task: {summary['task_name']}",
            f"- Model: {summary['model_id']}",
            f"- Status: {summary['status']}",
            f"- Promotion decision: {summary['promotion_decision']}",
            f"- Primary metric: {summary['primary_metric']}",
            f"- Primary delta: {summary['primary_delta']}",
            f"- Artifact status: {summary['artifact_status']} "
            f"({summary['ok_artifact_count']}/{summary['required_artifact_count']})",
            "",
            "## Decision Evidence",
            "",
            reason_lines,
            "",
            "## Artifact Health",
            "",
            _artifact_health(summary),
            "",
        ],
    )


def _read_json(path: str | Path) -> dict[str, Any]:
    body = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(body, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return body


def _artifact_health(summary: dict[str, Any]) -> str:
    failures = summary["artifact_failures"]
    if not failures:
        return "- all required artifacts validated"
    return "\n".join(
        f"- {failure['name']}: {failure['status']}"
        for failure in failures
    )


def _write_json(path: Path, body: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(body, indent=2, sort_keys=True),
        encoding="utf-8",
    )
