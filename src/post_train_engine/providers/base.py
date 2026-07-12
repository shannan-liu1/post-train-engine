"""Provider protocols for API-first compute backends."""

from __future__ import annotations

from typing import Protocol

from post_train_engine.api_schemas import JobHandle, JobRequest, JobResult, JobStatus
from post_train_engine.provider_operations import RecoveryPolicy


class RemoteProvider(Protocol):
    provider_id: str
    provider_type: str
    recovery_policy: RecoveryPolicy

    def submit_job(self, request: JobRequest) -> JobHandle:
        """Submit or synchronously execute a provider job and return its handle."""

    def poll_job(self, handle: JobHandle) -> JobStatus:
        """Return the current provider job status."""

    def fetch_result(self, handle: JobHandle) -> JobResult:
        """Return a terminal job result, or fail loudly."""

    def reconcile_job(
        self,
        request: JobRequest,
        handle: JobHandle | None,
    ) -> JobResult | None:
        """Recover durable provider state without resubmitting work."""


__all__ = ["RemoteProvider"]
