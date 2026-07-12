from __future__ import annotations

from pathlib import Path

import pytest

from post_train_engine.api_schemas import (
    JobHandle,
    JobRequest,
    JobResult,
    JobStatus,
)
from post_train_engine.artifact_store import ArtifactStore
from post_train_engine.provider_operations import execute_provider_operation
from post_train_engine.provider_operations import AmbiguousProviderOperationError


class RecoverableProvider:
    provider_id = "recoverable-provider"
    provider_type = "fixture"
    recovery_policy = "reconcile"

    def __init__(self) -> None:
        self.submission_count = 0
        self.results: dict[str, JobResult] = {}

    def submit_job(self, request: JobRequest) -> JobHandle:
        self.submission_count += 1
        handle = JobHandle(
            job_id=request.job_id,
            job_type=request.job_type,
            provider_id=request.provider_id,
            provider_job_id=f"provider://{request.job_id}",
        )
        self.results[request.job_id] = JobResult(
            handle=handle,
            status=JobStatus(state="succeeded"),
            payload={"value": 42},
        )
        return handle

    def reconcile_job(
        self,
        request: JobRequest,
        _handle: JobHandle | None,
    ) -> JobResult | None:
        return self.results.get(request.job_id)

    def poll_job(self, handle: JobHandle) -> JobStatus:
        return self.results[handle.job_id].status

    def fetch_result(self, handle: JobHandle) -> JobResult:
        return self.results[handle.job_id]


class CrashBeforeResultStore(ArtifactStore):
    def record_provider_result(self, result: JobResult) -> None:
        raise KeyboardInterrupt("simulated process death before result durability")


class NonReplayableProvider(RecoverableProvider):
    recovery_policy = "non_replayable"

    def reconcile_job(
        self,
        _request: JobRequest,
        _handle: JobHandle | None,
    ) -> JobResult | None:
        return None


class WrongHandleProvider(RecoverableProvider):
    def submit_job(self, request: JobRequest) -> JobHandle:
        self.submission_count += 1
        return JobHandle(
            job_id="different-job",
            job_type=request.job_type,
            provider_id=request.provider_id,
            provider_job_id="remote-wrong",
        )


def test_resume_reconciles_completed_provider_operation_without_resubmission(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    provider = RecoverableProvider()
    request = JobRequest(
        job_id="run-1:evaluate",
        job_type="evaluation",
        provider_id=provider.provider_id,
        payload={"candidate_id": "candidate-1"},
    )

    with pytest.raises(KeyboardInterrupt, match="process death"):
        execute_provider_operation(
            provider=provider,
            store=CrashBeforeResultStore(run_dir),
            request=request,
        )

    result = execute_provider_operation(
        provider=provider,
        store=ArtifactStore(run_dir, resume=True),
        request=request,
    )

    assert result.payload == {"value": 42}
    assert provider.submission_count == 1


def test_provider_rejects_handle_for_a_different_request(tmp_path: Path) -> None:
    provider = WrongHandleProvider()
    request = JobRequest(
        job_id="run-1:evaluate",
        job_type="evaluation",
        provider_id=provider.provider_id,
        payload={},
    )

    with pytest.raises(RuntimeError, match="provider handle does not match request"):
        execute_provider_operation(
            provider=provider,
            store=ArtifactStore(tmp_path / "run"),
            request=request,
        )


def test_ambiguous_non_replayable_operation_fails_without_resubmission(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    provider = NonReplayableProvider()
    request = JobRequest(
        job_id="run-1:adapt",
        job_type="candidate_adaptation",
        provider_id=provider.provider_id,
        payload={"candidate_id": "candidate-1"},
    )

    with pytest.raises(KeyboardInterrupt):
        execute_provider_operation(
            provider=provider,
            store=CrashBeforeResultStore(run_dir),
            request=request,
        )

    with pytest.raises(AmbiguousProviderOperationError, match="ambiguous"):
        execute_provider_operation(
            provider=provider,
            store=ArtifactStore(run_dir, resume=True),
            request=request,
        )

    assert provider.submission_count == 1
