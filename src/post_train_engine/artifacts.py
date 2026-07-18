"""Canonical RunBundle validation facade."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from post_train_engine.run_bundle import RunBundle


def validate_run_bundle(
    run_dir: str | Path,
    *,
    write: bool = True,
) -> dict[str, Any]:
    """Validate exactly one canonical RunBundle schema and its evidence."""

    root = Path(run_dir)
    manifest_path = root / "manifest.json"
    try:
        body = RunBundle.load(root).validate()
    except (
        FileNotFoundError,
        json.JSONDecodeError,
        ValidationError,
        ValueError,
    ) as exc:
        body = _manifest_failure(manifest_path, exc)
    if write:
        _write_json(root / "artifact_status.json", body)
    return body


def require_valid_run_bundle(run_dir: str | Path) -> dict[str, Any]:
    """Validate a RunBundle and raise with evidence-linked failures."""

    status = validate_run_bundle(run_dir, write=True)
    if status["status"] != "ok":
        failed = ", ".join(_format_failure(item) for item in status["failures"])
        raise ValueError(f"artifact bundle validation failed: {failed}")
    return status


def _manifest_failure(path: Path, error: Exception) -> dict[str, Any]:
    raw: dict[str, Any] = {}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(parsed, dict):
            raw = parsed
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    exists = path.is_file()
    failure = {
        "name": "manifest",
        "kind": "manifest",
        "path": "manifest.json",
        "required": True,
        "status": "malformed" if exists else "missing",
        "exists": exists,
        "expected_sha256": None,
        "actual_sha256": _sha256(path) if exists else None,
        "message": str(error),
    }
    return {
        "run_id": str(raw.get("run_id", "")),
        "candidate_id": str(raw.get("candidate_id", "")),
        "status": "failed",
        "required_count": 1,
        "ok_count": 0,
        "failure_count": 1,
        "failures": [failure],
        "artifacts": [failure],
    }


def _format_failure(failure: dict[str, Any]) -> str:
    label = f"{failure['name']}:{failure['status']}"
    message = failure.get("message")
    return f"{label}({message})" if message else label


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, body: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(
        json.dumps(body, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


__all__ = ["require_valid_run_bundle", "validate_run_bundle"]
