"""Local artifact store for API-first hill-climb runs."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from post_train_engine.api_schemas import (
    JobHandle,
    JobRequest,
    JobResult,
    redact_secret_text,
    redact_secrets,
)
from post_train_engine.jsonl import read_jsonl, write_jsonl
from post_train_engine.run_bundle import make_artifact_ref

_STORE_MARKER = ".post_train_artifact_store.json"
_STORE_MARKER_SCHEMA = "post_train_artifact_store_v1"


class ArtifactStore:
    """Write immutable-ish local artifacts under one run directory."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        overwrite: bool = False,
        resume: bool = False,
    ) -> None:
        self.run_dir = Path(run_dir)
        if resume:
            if overwrite:
                raise ValueError("resume and overwrite are mutually exclusive")
            if not self.run_dir.is_dir():
                raise ValueError(f"cannot resume missing run directory: {self.run_dir}")
            if not _is_safe_overwrite_target(self.run_dir):
                raise ValueError(f"cannot resume unmarked run directory: {self.run_dir}")
            return
        if self.run_dir.exists() and not overwrite:
            raise ValueError(f"run directory already exists: {self.run_dir}")
        if overwrite and self.run_dir.exists():
            _clear_run_dir(self.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.write_json(
            _STORE_MARKER,
            {
                "schema_version": _STORE_MARKER_SCHEMA,
                "run_dir_name": self.run_dir.name,
            },
        )

    def write_text(self, relative: str, text: str) -> Path:
        path = self._path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def write_json(self, relative: str, body: Any) -> Path:
        return self.write_text(
            relative,
            json.dumps(_jsonable(body), indent=2, sort_keys=True),
        )

    def write_jsonl(self, relative: str, rows: list[dict[str, Any]]) -> Path:
        path = self._path(relative)
        write_jsonl(path, rows)
        return path

    def copy_file(self, source: str | Path, relative: str) -> Path:
        path = self._path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, path)
        return path

    def append_provider_request(self, request: JobRequest) -> None:
        _upsert_jsonl(
            self.run_dir / "provider_requests.jsonl",
            redact_secrets(request.to_json()),
            job_id=request.job_id,
        )

    def append_provider_response(self, result: JobResult) -> None:
        _upsert_jsonl(
            self.run_dir / "provider_responses.jsonl",
            result.to_redacted_json(),
            job_id=result.handle.job_id,
        )

    def append_provider_error(
        self,
        request: JobRequest,
        error: BaseException,
    ) -> None:
        _upsert_jsonl(
            self.run_dir / "provider_errors.jsonl",
            {
                "job_id": request.job_id,
                "job_type": request.job_type,
                "provider_id": request.provider_id,
                "error_type": type(error).__name__,
                "message": redact_secret_text(str(error)),
                "request": redact_secrets(request.to_json()),
            },
            job_id=request.job_id,
        )

    def provider_operation(self, job_id: str) -> dict[str, Any] | None:
        path = self.run_dir / "provider_operations.jsonl"
        if not path.is_file():
            return None
        return next(
            (row for row in read_jsonl(path) if row.get("job_id") == job_id),
            None,
        )

    def record_provider_intent(
        self,
        request: JobRequest,
        *,
        request_sha256: str,
        recovery_policy: str,
    ) -> None:
        existing = self.provider_operation(request.job_id)
        if existing is not None:
            if existing.get("request_sha256") != request_sha256:
                raise ValueError("provider operation request differs from durable intent")
            return
        _upsert_jsonl(
            self.run_dir / "provider_operations.jsonl",
            {
                "schema_version": "provider_operation_v1",
                "job_id": request.job_id,
                "provider_id": request.provider_id,
                "job_type": request.job_type,
                "request_sha256": request_sha256,
                "recovery_policy": recovery_policy,
                "state": "intent",
                "request": redact_secrets(request.to_json()),
                "handle": None,
                "result": None,
            },
            job_id=request.job_id,
        )

    def record_provider_handle(self, handle: JobHandle) -> None:
        operation = self.provider_operation(handle.job_id)
        if operation is None:
            raise ValueError("provider handle requires a durable operation intent")
        operation["state"] = "submitted"
        operation["handle"] = handle.to_json()
        _upsert_jsonl(
            self.run_dir / "provider_operations.jsonl",
            operation,
            job_id=handle.job_id,
        )

    def record_provider_result(self, result: JobResult) -> None:
        operation = self.provider_operation(result.handle.job_id)
        if operation is None:
            raise ValueError("provider result requires a durable operation intent")
        operation["state"] = "completed"
        operation["handle"] = result.handle.to_json()
        operation["result"] = result.to_redacted_json()
        _upsert_jsonl(
            self.run_dir / "provider_operations.jsonl",
            operation,
            job_id=result.handle.job_id,
        )

    def artifact_ref(
        self,
        relative: str,
        *,
        kind: str,
        required: bool = True,
    ) -> dict[str, Any]:
        return make_artifact_ref(
            self.run_dir,
            self._path(relative),
            kind=kind,
            required=required,
        ).model_dump(mode="json")

    def sha256(self, relative: str) -> str:
        path = self._path(relative)
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()

    def _path(self, relative: str) -> Path:
        requested = Path(relative)
        if requested.is_absolute():
            raise ValueError(f"artifact path is outside run directory: {relative}")
        root = self.run_dir.resolve()
        target = (root / requested).resolve()
        if target == root or not target.is_relative_to(root):
            raise ValueError(f"artifact path is outside run directory: {relative}")
        return target


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    return value


def _upsert_jsonl(
    path: Path,
    row: dict[str, Any],
    *,
    job_id: str,
) -> None:
    existing = read_jsonl(path) if path.is_file() else []
    retained = [item for item in existing if _provider_log_job_id(item) != job_id]
    temporary = path.with_name("." + path.name + ".tmp")
    write_jsonl(temporary, [*retained, row])
    temporary.replace(path)


def _provider_log_job_id(row: dict[str, Any]) -> str | None:
    direct = row.get("job_id")
    if isinstance(direct, str):
        return direct
    handle = row.get("handle")
    if isinstance(handle, dict) and isinstance(handle.get("job_id"), str):
        return str(handle["job_id"])
    return None


def _clear_run_dir(path: Path) -> None:
    resolved = path.resolve()
    if resolved == resolved.parent:
        raise ValueError(f"refusing to overwrite filesystem root: {resolved}")
    if _is_linked_path(path) or not path.is_dir():
        raise ValueError(f"refusing to overwrite non-directory run path: {path}")
    if (path / ".git").exists():
        raise ValueError(f"refusing to overwrite repository root: {path}")
    if not _is_safe_overwrite_target(path):
        raise ValueError(f"refusing to overwrite unmarked run directory: {path}")
    for child in path.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def _is_safe_overwrite_target(path: Path) -> bool:
    if _is_linked_path(path):
        return False
    marker = path / _STORE_MARKER
    if not marker.is_file() or _is_linked_path(marker):
        return False
    try:
        body = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        isinstance(body, dict)
        and body.get("schema_version") == _STORE_MARKER_SCHEMA
        and body.get("run_dir_name") == path.name
    )


def _is_linked_path(path: Path) -> bool:
    return path.is_symlink() or bool(getattr(path, "is_junction", lambda: False)())


__all__ = ["ArtifactStore"]
