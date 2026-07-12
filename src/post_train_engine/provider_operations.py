"""Crash-safe execution boundary for external provider operations."""

from __future__ import annotations

import hashlib
import json
import time

from post_train_engine.api_schemas import (
    JobHandle,
    JobRequest,
    JobResult,
    redact_secret_text,
)
from post_train_engine.artifact_store import ArtifactStore
from post_train_engine.providers.base import RemoteProvider


class AmbiguousProviderOperationError(RuntimeError):
    """Provider work may have completed but cannot be reconciled safely."""


def execute_provider_operation(
    *,
    provider: RemoteProvider,
    store: ArtifactStore,
    request: JobRequest,
    timeout_seconds: float = 300.0,
    poll_interval_seconds: float = 0.05,
) -> JobResult:
    """Execute or reconcile one provider operation without blind resubmission."""

    request_sha256 = _request_sha256(request)
    operation = store.provider_operation(request.job_id)
    if operation is not None:
        if operation.get("request_sha256") != request_sha256:
            raise ValueError("provider operation request differs from durable intent")
        completed = operation.get("result")
        if isinstance(completed, dict):
            return JobResult.model_validate(completed)
        stored_handle = operation.get("handle")
        handle = (
            JobHandle.model_validate(stored_handle)
            if isinstance(stored_handle, dict)
            else None
        )
        reconciled = provider.reconcile_job(request, handle)
        if reconciled is not None:
            _validate_provider_handle(request, reconciled.handle)
            _validate_provider_result(reconciled.handle, reconciled)
            store.record_provider_result(reconciled)
            store.append_provider_response(reconciled)
            return reconciled
        if provider.recovery_policy != "replay_safe":
            raise AmbiguousProviderOperationError(
                f"provider operation {request.job_id!r} is ambiguous; "
                "reconciliation returned no durable result"
            )
    else:
        store.record_provider_intent(
            request,
            request_sha256=request_sha256,
            recovery_policy=provider.recovery_policy,
        )

    store.append_provider_request(request)
    try:
        handle = provider.submit_job(request)
        _validate_provider_handle(request, handle)
        store.record_provider_handle(handle)
        start = time.monotonic()
        while True:
            status = provider.poll_job(handle)
            if status.terminal:
                break
            if time.monotonic() - start > timeout_seconds:
                raise TimeoutError(f"provider job timed out: {handle.provider_job_id}")
            time.sleep(poll_interval_seconds)
        if status.state != "succeeded":
            raise RuntimeError(
                f"provider job failed closed: {handle.provider_job_id}: {status.message}"
            )
        result = provider.fetch_result(handle)
        _validate_provider_result(handle, result)
        store.record_provider_result(result)
    except Exception as exc:
        store.append_provider_error(request, exc)
        raise RuntimeError(
            f"provider job failed closed: {request.job_id}: "
            f"{redact_secret_text(str(exc))}"
        ) from exc
    store.append_provider_response(result)
    return result


def _request_sha256(request: JobRequest) -> str:
    payload = json.dumps(
        request.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_provider_result(handle: JobHandle, result: JobResult) -> None:
    if result.handle != handle:
        raise RuntimeError("provider result handle mismatch")
    if result.status.state != "succeeded":
        raise RuntimeError(f"provider result failed closed: {result.status.message}")


def _validate_provider_handle(request: JobRequest, handle: JobHandle) -> None:
    if (
        handle.job_id != request.job_id
        or handle.job_type != request.job_type
        or handle.provider_id != request.provider_id
    ):
        raise RuntimeError("provider handle does not match request")


__all__ = [
    "AmbiguousProviderOperationError",
    "execute_provider_operation",
]
