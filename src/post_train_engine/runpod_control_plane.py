"""Crash-safe RunPod allocation and billing receipts."""

from __future__ import annotations

import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Protocol
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field, model_validator

from post_train_engine.runpod import cuda_version_from_image

_FROZEN_FORBID = ConfigDict(frozen=True, extra="forbid")
_MAX_USER_AUTHORIZED_SPEND_USD = 1.5


class RunPodTransport(Protocol):
    def request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> Any:
        """Send one authenticated RunPod API request."""


class RunPodRESTTransport:
    """Minimal official REST transport. It never serializes the API key."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://rest.runpod.io/v1",
        timeout_seconds: float = 30.0,
    ) -> None:
        if not api_key:
            raise ValueError("RunPod API key must be non-empty")
        if base_url != "https://rest.runpod.io/v1":
            raise ValueError("RunPod REST base URL must use the official v1 endpoint")
        if timeout_seconds <= 0.0:
            raise ValueError("RunPod REST timeout must be positive")
        self._api_key = api_key
        self._base_url = base_url
        self._timeout_seconds = timeout_seconds

    def request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> Any:
        if not path.startswith("/") or path.startswith("//"):
            raise ValueError("RunPod REST path must be absolute within the v1 API")
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = Request(
            self._base_url + path,
            data=data,
            method=method,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Accept": "application/json",
                **({"Content-Type": "application/json"} if data is not None else {}),
            },
        )
        with urlopen(request, timeout=self._timeout_seconds) as response:
            payload = response.read()
        return {} if not payload else json.loads(payload.decode("utf-8"))


class RunPodProviderTransport(RunPodRESTTransport):
    """Use provider-scheduled GraphQL creation and REST lifecycle operations."""

    _GRAPHQL_URL = "https://api.runpod.io/graphql"

    def request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> Any:
        if method != "POST" or path != "/pods":
            return super().request(method, path, body)
        if body is None:
            raise ValueError("RunPod Pod creation requires a request body")
        terminate_after = body.get("terminateAfter")
        if not isinstance(terminate_after, str) or not terminate_after:
            raise ValueError("RunPod Pod creation requires provider termination")
        gpu_types = body.get("gpuTypeIds")
        if not isinstance(gpu_types, list) or len(gpu_types) != 1:
            raise ValueError("RunPod GraphQL creation requires exactly one GPU type")
        cuda_versions = body.get("allowedCudaVersions")
        if not isinstance(cuda_versions, list) or len(cuda_versions) != 1:
            raise ValueError("RunPod GraphQL creation requires one CUDA version")
        provider_input = {
            "cloudType": body.get("cloudType"),
            "containerDiskInGb": body.get("containerDiskInGb"),
            "gpuCount": body.get("gpuCount"),
            "gpuTypeId": gpu_types[0],
            "imageName": body.get("imageName"),
            "minCudaVersion": cuda_versions[0],
            "name": body.get("name"),
            "ports": ",".join(body.get("ports", ())),
            "startSsh": True,
            "supportPublicIp": body.get("supportPublicIp"),
            "terminateAfter": terminate_after,
            "volumeInGb": body.get("volumeInGb"),
        }
        payload = json.dumps(
            {
                "query": (
                    "mutation createPod($input: PodFindAndDeployOnDemandInput!) { "
                    "podFindAndDeployOnDemand(input: $input) { "
                    "id name costPerHr desiredStatus } }"
                ),
                "variables": {"input": provider_input},
            }
        ).encode("utf-8")
        request = Request(
            self._GRAPHQL_URL,
            data=payload,
            method="POST",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
        )
        with urlopen(request, timeout=self._timeout_seconds) as response:
            raw = json.loads(response.read().decode("utf-8"))
        errors = raw.get("errors") if isinstance(raw, dict) else None
        if isinstance(errors, list) and errors:
            message = errors[0].get("message") if isinstance(errors[0], dict) else None
            raise RuntimeError(
                f"RunPod GraphQL create failed: {message or 'unknown error'}"
            )
        data = raw.get("data") if isinstance(raw, dict) else None
        pod = data.get("podFindAndDeployOnDemand") if isinstance(data, dict) else None
        if not isinstance(pod, dict):
            raise ValueError("RunPod GraphQL create response omitted the Pod")
        return pod


class RunPodBudget(BaseModel):
    model_config = _FROZEN_FORBID

    target_spend_usd: float = Field(..., gt=0.0, le=_MAX_USER_AUTHORIZED_SPEND_USD)
    settled_spend_usd: float = Field(..., ge=0.0)
    reserve_usd: float = Field(default=0.15, ge=0.0)
    minimum_runtime_seconds: int = Field(default=60, gt=0)
    max_runtime_seconds: int = Field(default=1200, gt=0)

    @model_validator(mode="after")
    def _settled_spend_and_reserve_leave_execution_budget(self) -> RunPodBudget:
        if self.settled_spend_usd + self.reserve_usd >= self.target_spend_usd:
            raise ValueError(
                "settled_spend_usd plus reserve_usd must be below target_spend_usd"
            )
        if self.max_runtime_seconds < self.minimum_runtime_seconds:
            raise ValueError(
                "max_runtime_seconds must be at least minimum_runtime_seconds"
            )
        return self

    def hard_deadline_seconds(self, pod_rate_usd_per_hour: float) -> int:
        if not math.isfinite(pod_rate_usd_per_hour) or pod_rate_usd_per_hour <= 0.0:
            raise ValueError(
                "RunPod create receipt requires a positive finite Pod rate"
            )
        cost_deadline_seconds = math.floor(
            (self.target_spend_usd - self.settled_spend_usd - self.reserve_usd)
            / pod_rate_usd_per_hour
            * 3600.0
        )
        seconds = min(cost_deadline_seconds, self.max_runtime_seconds)
        if seconds < self.minimum_runtime_seconds:
            raise ValueError(
                "authoritative Pod rate cannot fit the minimum runtime under the spend target"
            )
        return seconds


class RunPodAllocationPolicy(BaseModel):
    """Exact paid-compute envelope authorized for the current campaign."""

    model_config = _FROZEN_FORBID

    cloud_type: Literal["SECURE"] = "SECURE"
    gpu_type: Literal["NVIDIA A40"] = "NVIDIA A40"
    gpu_count: Literal[2] = 2
    container_disk_gb: Literal[40] = 40
    volume_gb: Literal[0] = 0

    def validate_request(self, request: dict[str, Any]) -> None:
        if request.get("cloudType") != self.cloud_type:
            raise ValueError("RunPod request must use Secure Cloud")
        if request.get("computeType") != "GPU":
            raise ValueError("RunPod request must use GPU compute")
        if request.get("gpuTypeIds") != [self.gpu_type]:
            raise ValueError("RunPod request must use exactly NVIDIA A40")
        if request.get("gpuCount") != self.gpu_count:
            raise ValueError("RunPod request must use exactly two A40 GPUs")
        if request.get("containerDiskInGb") != self.container_disk_gb:
            raise ValueError("RunPod request must use exactly 40 GB container disk")
        if request.get("volumeInGb") != self.volume_gb:
            raise ValueError("RunPod request must use zero persistent volume")
        if request.get("interruptible") is not False:
            raise ValueError(
                "RunPod request must explicitly disable interruptible mode"
            )
        if request.get("ports") != ["22/tcp"]:
            raise ValueError("RunPod request must expose SSH only")
        if request.get("supportPublicIp") is not True:
            raise ValueError("RunPod request must request a public SSH address")
        if "env" in request:
            raise ValueError(
                "RunPod create request must not persist environment values"
            )
        image = str(request.get("imageName") or "")
        cuda_version = cuda_version_from_image(image)
        if request.get("allowedCudaVersions") != [cuda_version]:
            raise ValueError("RunPod CUDA filter must match the pinned image")


class PodCreateReceipt(BaseModel):
    model_config = _FROZEN_FORBID

    pod_id: str = Field(..., min_length=1)
    pod_name: str = Field(..., min_length=1)
    pod_rate_usd_per_hour: float = Field(..., gt=0.0)
    hard_deadline_seconds: int = Field(..., gt=0)
    request_sha256: str = Field(..., pattern=r"^sha256:[0-9a-f]{64}$")
    recorded_at_unix: float = Field(..., gt=0.0)


class PodBillingReceipt(BaseModel):
    model_config = _FROZEN_FORBID

    pod_id: str = Field(..., min_length=1)
    settlement_state: Literal["pending", "provisional", "settled"]
    amount_usd: float | None = Field(default=None, ge=0.0)
    row_count: int = Field(..., ge=0)
    recorded_at_unix: float = Field(..., gt=0.0)

    @model_validator(mode="after")
    def _state_matches_amount(self) -> PodBillingReceipt:
        has_amount = self.amount_usd is not None
        if self.settlement_state == "pending" and has_amount:
            raise ValueError("pending billing receipt cannot include an amount")
        if self.settlement_state in {"provisional", "settled"} and not has_amount:
            raise ValueError("billing settlement state must match amount evidence")
        return self


class AmbiguousPodCreationError(RuntimeError):
    """The create request may have succeeded, so an automatic replay is unsafe."""


class RunPodControlPlane:
    """Own one durable, bounded RunPod Pod creation operation."""

    def __init__(
        self,
        transport: RunPodTransport,
        journal_path: str | Path,
        *,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.transport = transport
        self.journal_path = Path(journal_path)
        self.sleep = sleep
        self.clock = clock

    def create_pod(
        self,
        request: dict[str, Any],
        *,
        budget: RunPodBudget,
        allocation_policy: RunPodAllocationPolicy | None = None,
        arm_watchdog: Callable[[], dict[str, Any]] | None = None,
        operation_sha256: str | None = None,
    ) -> PodCreateReceipt:
        (allocation_policy or RunPodAllocationPolicy()).validate_request(request)
        if arm_watchdog is None:
            raise ValueError("RunPod creation requires a watchdog callback")
        pod_name = str(request.get("name") or "")
        if not pod_name:
            raise ValueError("RunPod create request requires a deterministic name")
        request_sha256 = _sha256_json(request)
        budget_json = budget.model_dump(mode="json")
        operation_sha256 = operation_sha256 or _sha256_json(
            {"request": request, "budget": budget_json}
        )
        if not operation_sha256.startswith("sha256:") or len(operation_sha256) != 71:
            raise ValueError("RunPod attempt identity must use sha256:<digest>")
        journal = self._read_journal()
        if journal is None:
            intent_at_unix = self.clock()
            operation = {
                "intent_at_unix": intent_at_unix,
                "pod_name": pod_name,
                "request_sha256": request_sha256,
                "operation_sha256": operation_sha256,
                "budget": budget_json,
                "request": request,
            }
            self._write_journal({"state": "intent", **operation})
            watchdog = arm_watchdog()
            if watchdog.get("state") != "armed":
                raise RuntimeError("RunPod watchdog did not enter armed state")
            self._write_journal({"state": "create_started", **operation})
            provider_request = {
                **request,
                "terminateAfter": _rfc3339_utc(
                    intent_at_unix + budget.max_runtime_seconds
                ),
            }
            try:
                raw = self.transport.request("POST", "/pods", provider_request)
            except BaseException as exc:
                self._write_journal(
                    {
                        "state": "ambiguous",
                        "intent_at_unix": intent_at_unix,
                        "pod_name": pod_name,
                        "request_sha256": request_sha256,
                        "operation_sha256": operation_sha256,
                        "budget": budget_json,
                        "request": request,
                    }
                )
                raw = self._reconcile_by_name_with_retries(pod_name)
                if raw is None:
                    raise AmbiguousPodCreationError(
                        "RunPod create response was ambiguous; no same-name Pod "
                        "appeared during bounded reconciliation"
                    ) from exc
        else:
            if journal.get("request_sha256") != request_sha256:
                raise ValueError(
                    "RunPod operation journal belongs to a different request"
                )
            if journal.get("budget") != budget_json:
                raise ValueError(
                    "RunPod operation journal belongs to a different budget"
                )
            if journal.get("operation_sha256") != operation_sha256:
                raise ValueError(
                    "RunPod operation journal belongs to a different attempt identity"
                )
            state = journal.get("state")
            if state not in {"intent", "create_started", "ambiguous", "created"}:
                raise RuntimeError(
                    f"RunPod create cannot resume terminal cleanup state {state!r}"
                )
            if state == "created":
                watchdog = arm_watchdog()
                if watchdog.get("state") != "armed":
                    raise RuntimeError("RunPod watchdog did not enter armed state")
                return PodCreateReceipt.model_validate(journal["receipt"])
            intent_at_unix = _positive_timestamp(
                journal.get("intent_at_unix"), label="RunPod create intent"
            )
            watchdog = arm_watchdog()
            if watchdog.get("state") != "armed":
                raise RuntimeError("RunPod watchdog did not enter armed state")
            if state == "intent":
                operation = {
                    key: value for key, value in journal.items() if key != "state"
                }
                self._write_journal({"state": "create_started", **operation})
                provider_request = {
                    **request,
                    "terminateAfter": _rfc3339_utc(
                        intent_at_unix + budget.max_runtime_seconds
                    ),
                }
                try:
                    raw = self.transport.request("POST", "/pods", provider_request)
                except BaseException as exc:
                    self._write_journal({"state": "ambiguous", **operation})
                    raw = self._reconcile_by_name_with_retries(pod_name)
                    if raw is None:
                        raise AmbiguousPodCreationError(
                            "RunPod create response was ambiguous; no same-name Pod "
                            "appeared during bounded reconciliation"
                        ) from exc
            else:
                raw = self._reconcile_by_name_with_retries(pod_name)
                if raw is None:
                    raise AmbiguousPodCreationError(
                        "RunPod creation remains ambiguous; no same-name Pod was found "
                        "and replay is disabled"
                    )

        pod_id = str(raw.get("id") or "") if isinstance(raw, dict) else ""
        if not pod_id:
            self._write_journal(
                {
                    "state": "ambiguous",
                    "intent_at_unix": intent_at_unix,
                    "pod_name": pod_name,
                    "request_sha256": request_sha256,
                    "operation_sha256": operation_sha256,
                    "budget": budget_json,
                    "request": request,
                }
            )
            raw = self._reconcile_by_name(pod_name)
            if raw is None:
                raise AmbiguousPodCreationError(
                    "RunPod create response omitted the Pod id and no same-name Pod was found"
                )
            pod_id = str(raw.get("id") or "")
            if not pod_id:
                raise AmbiguousPodCreationError(
                    "RunPod reconciliation found a same-name row without a Pod id"
                )
        try:
            rate = _pod_rate(raw)
            hard_deadline = budget.hard_deadline_seconds(rate)
        except BaseException:
            self.delete_pod_and_verify(pod_id)
            raise
        if self.clock() >= intent_at_unix + hard_deadline:
            self.delete_pod_and_verify(pod_id)
            raise TimeoutError(
                "RunPod create reconciliation exceeded its hard deadline"
            )
        receipt = PodCreateReceipt(
            pod_id=pod_id,
            pod_name=pod_name,
            pod_rate_usd_per_hour=rate,
            hard_deadline_seconds=hard_deadline,
            request_sha256=request_sha256,
            recorded_at_unix=intent_at_unix,
        )
        try:
            self._write_journal(
                {
                    "state": "created",
                    "pod_name": pod_name,
                    "request_sha256": request_sha256,
                    "operation_sha256": operation_sha256,
                    "budget": budget_json,
                    "request": request,
                    "receipt": receipt.model_dump(mode="json"),
                }
            )
        except BaseException:
            self.delete_pod_and_verify(pod_id)
            raise
        return receipt

    def list_pods(self) -> list[dict[str, Any]]:
        """Return the provider Pod inventory for pre-create and teardown gates."""

        return _list_rows(self.transport.request("GET", "/pods"))

    def created_receipt(self) -> PodCreateReceipt | None:
        """Return the durable receipt only for an unfinished created operation."""

        journal = self._read_journal()
        if journal is None or journal.get("state") != "created":
            return None
        return PodCreateReceipt.model_validate(journal.get("receipt"))

    def operation_state(self) -> str | None:
        journal = self._read_journal()
        return None if journal is None else str(journal.get("state") or "")

    def delete_pod(self, pod_id: str) -> None:
        if not pod_id:
            raise ValueError("pod_id must be non-empty")
        self.transport.request("DELETE", f"/pods/{pod_id}")
        self._record_delete_requested(pod_id)

    def verify_pod_absent(self, pod_id: str) -> bool:
        """Close the journal only after two provider absence observations."""

        if not pod_id:
            raise ValueError("pod_id must be non-empty")
        for observation in range(2):
            active_pods = _list_rows(self.transport.request("GET", "/pods"))
            unexpected = [
                str(row.get("id") or "<missing-id>")
                for row in active_pods
                if str(row.get("id")) != pod_id
            ]
            if unexpected:
                raise RuntimeError(
                    "RunPod teardown found unexpected active Pods: "
                    + ", ".join(unexpected)
                )
            if any(str(row.get("id")) == pod_id for row in active_pods):
                return False
            if observation == 0:
                self.sleep(0.25)
        self._record_deleted(pod_id, outcome="provider_absent")
        return True

    def require_only_created_pod(
        self,
        receipt: PodCreateReceipt,
        *,
        attempts: int = 5,
    ) -> None:
        """Require eventual visibility of exactly the just-created Pod."""

        if attempts <= 0:
            raise ValueError("inventory attempts must be positive")
        last_error: BaseException | None = None
        for attempt in range(attempts):
            try:
                active_pods = self.list_pods()
            except (OSError, TimeoutError) as exc:
                last_error = exc
                active_pods = []
            matching = [
                pod
                for pod in active_pods
                if str(pod.get("id")) == receipt.pod_id
                and str(pod.get("name")) == receipt.pod_name
            ]
            if len(matching) == 1 and len(active_pods) == 1:
                return
            if active_pods:
                raise RuntimeError(
                    "created RunPod attempt does not match the active Pod inventory"
                )
            if attempt + 1 < attempts:
                self.sleep(1.0)
        error = RuntimeError("created RunPod Pod never appeared in active inventory")
        if last_error is not None:
            raise error from last_error
        raise error

    def delete_pod_and_verify(self, pod_id: str, *, attempts: int = 3) -> int:
        """Delete a known Pod and require provider-confirmed absence."""

        if attempts <= 0:
            raise ValueError("delete attempts must be positive")
        last_error: BaseException | None = None
        for attempt in range(attempts):
            try:
                self.delete_pod(pod_id)
            except BaseException as exc:
                last_error = exc
            try:
                if self.verify_pod_absent(pod_id):
                    return attempt + 1
            except BaseException as exc:
                last_error = exc
            if attempt + 1 < attempts:
                self.sleep(1.0)
        self.record_delete_unverified(pod_id)
        error = RuntimeError(
            f"RunPod still reports Pod {pod_id!r} active after {attempts} delete attempts"
        )
        if last_error is not None:
            raise error from last_error
        raise error

    def get_pod(self, pod_id: str) -> dict[str, Any]:
        """Return one provider Pod only when its identity matches the request."""

        if not pod_id:
            raise ValueError("pod_id must be non-empty")
        raw = self.transport.request("GET", f"/pods/{pod_id}")
        if not isinstance(raw, dict) or str(raw.get("id")) != pod_id:
            raise ValueError("RunPod Pod response did not match the requested Pod id")
        return raw

    def record_delete_unverified(self, pod_id: str) -> None:
        """Keep the journal non-terminal when provider absence is unproven."""

        if not pod_id:
            raise ValueError("pod_id must be non-empty")
        journal = self._read_journal()
        receipt = None if journal is None else journal.get("receipt")
        if isinstance(receipt, dict) and str(receipt.get("pod_id")) != pod_id:
            return
        if journal is not None:
            retained = {
                key: value
                for key, value in journal.items()
                if key not in {"deleted_at_unix", "deleted_pod_id"}
            }
            self._write_journal(
                {
                    **retained,
                    "state": "delete_unverified",
                    "delete_unverified_pod_id": pod_id,
                    "delete_unverified_at_unix": time.time(),
                    "deletion_outcome": "provider_absence_unverified",
                }
            )

    def _record_deleted(self, pod_id: str, *, outcome: str) -> None:
        journal = self._read_journal()
        receipt = None if journal is None else journal.get("receipt")
        if isinstance(receipt, dict) and str(receipt.get("pod_id")) != pod_id:
            return
        if journal is not None:
            if (
                journal.get("state") == "deleted"
                and str(journal.get("deleted_pod_id")) == pod_id
            ):
                return
            self._write_journal(
                {
                    **journal,
                    "state": "deleted",
                    "deleted_pod_id": pod_id,
                    "deleted_at_unix": time.time(),
                    "deletion_outcome": outcome,
                }
            )

    def _record_delete_requested(self, pod_id: str) -> None:
        journal = self._read_journal()
        receipt = None if journal is None else journal.get("receipt")
        if isinstance(receipt, dict) and str(receipt.get("pod_id")) != pod_id:
            return
        if journal is not None:
            self._write_journal(
                {
                    **journal,
                    "state": "delete_requested",
                    "delete_requested_pod_id": pod_id,
                    "delete_requested_at_unix": self.clock(),
                    "deletion_outcome": "provider_delete_accepted",
                }
            )

    def fetch_billing(
        self,
        pod_id: str,
        *,
        start_time: str,
        end_time: str | None = None,
        final: bool = False,
    ) -> PodBillingReceipt:
        query = {
            "podId": pod_id,
            "startTime": start_time,
            "grouping": "podId",
            "bucketSize": "hour",
        }
        if end_time is not None:
            query["endTime"] = end_time
        raw = self.transport.request("GET", "/billing/pods?" + urlencode(query))
        rows = _list_rows(raw)
        pod_rows = [row for row in rows if str(row.get("podId")) == pod_id]
        if not pod_rows:
            return PodBillingReceipt(
                pod_id=pod_id,
                settlement_state="pending",
                amount_usd=None,
                row_count=0,
                recorded_at_unix=time.time(),
            )
        amounts = [_non_negative_amount(row.get("amount")) for row in pod_rows]
        amount = sum(amounts)
        if not final:
            self._write_billing_observation(pod_id, amount)
            return PodBillingReceipt(
                pod_id=pod_id,
                settlement_state="provisional",
                amount_usd=amount,
                row_count=len(pod_rows),
                recorded_at_unix=time.time(),
            )
        if end_time is None:
            raise ValueError("final billing settlement requires end_time")
        observation = self._read_billing_observation(pod_id)
        if observation is None or amount != observation:
            raise ValueError(
                "final billing amount must match a durable prior observation"
            )
        active_pods = _list_rows(self.transport.request("GET", "/pods"))
        if active_pods:
            raise ValueError("cannot finalize billing while any RunPod Pod is active")
        self._record_deleted(pod_id, outcome="provider_absent_at_billing_settlement")
        return PodBillingReceipt(
            pod_id=pod_id,
            settlement_state="settled",
            amount_usd=amount,
            row_count=len(pod_rows),
            recorded_at_unix=time.time(),
        )

    def _reconcile_by_name(self, pod_name: str) -> dict[str, Any] | None:
        rows = _list_rows(self.transport.request("GET", "/pods"))
        matches = [row for row in rows if str(row.get("name")) == pod_name]
        if len(matches) > 1:
            raise AmbiguousPodCreationError(
                f"multiple RunPod Pods share deterministic name {pod_name!r}"
            )
        return None if not matches else matches[0]

    def _reconcile_by_name_with_retries(
        self,
        pod_name: str,
        *,
        attempts: int = 3,
    ) -> dict[str, Any] | None:
        for attempt in range(attempts):
            try:
                match = self._reconcile_by_name(pod_name)
            except AmbiguousPodCreationError:
                raise
            except BaseException:
                match = None
            if match is not None:
                return match
            if attempt + 1 < attempts:
                self.sleep(1.0)
        return None

    def _read_journal(self) -> dict[str, Any] | None:
        if not self.journal_path.is_file():
            return None
        raw = json.loads(self.journal_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("RunPod operation journal must be a JSON object")
        return raw

    def _write_journal(self, body: dict[str, Any]) -> None:
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.journal_path.with_name("." + self.journal_path.name + ".tmp")
        temporary.write_text(
            json.dumps(body, indent=2, sort_keys=True), encoding="utf-8"
        )
        temporary.replace(self.journal_path)

    @property
    def _billing_journal_path(self) -> Path:
        return self.journal_path.with_name(self.journal_path.stem + ".billing.json")

    def _write_billing_observation(self, pod_id: str, amount_usd: float) -> None:
        path = self._billing_journal_path
        body: dict[str, Any] = {}
        if path.is_file():
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("RunPod billing journal must be a JSON object")
            body = raw
        body[pod_id] = {"amount_usd": amount_usd, "recorded_at_unix": time.time()}
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name("." + path.name + ".tmp")
        temporary.write_text(
            json.dumps(body, indent=2, sort_keys=True), encoding="utf-8"
        )
        temporary.replace(path)

    def _read_billing_observation(self, pod_id: str) -> float | None:
        path = self._billing_journal_path
        if not path.is_file():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or not isinstance(raw.get(pod_id), dict):
            return None
        return _non_negative_amount(raw[pod_id].get("amount_usd"))


def _pod_rate(raw: dict[str, Any]) -> float:
    value = raw.get("adjustedCostPerHr")
    if value is None:
        value = raw.get("costPerHr")
    if type(value) is bool or not isinstance(value, int | float | str):
        raise ValueError(
            "RunPod create response did not include an authoritative Pod rate"
        )
    try:
        rate = float(value)
    except ValueError as exc:
        raise ValueError("RunPod create response Pod rate must be numeric") from exc
    if not math.isfinite(rate) or rate <= 0.0:
        raise ValueError("RunPod create response Pod rate must be positive and finite")
    return rate


def _list_rows(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, dict):
        raw = raw.get("data", raw.get("pods", raw.get("items")))
    if not isinstance(raw, list) or any(not isinstance(row, dict) for row in raw):
        raise ValueError("RunPod list response must contain object rows")
    return list(raw)


def _non_negative_amount(value: Any) -> float:
    if type(value) is bool or not isinstance(value, int | float):
        raise ValueError("RunPod billing amount must be numeric")
    amount = float(value)
    if not math.isfinite(amount) or amount < 0.0:
        raise ValueError("RunPod billing amount must be finite and non-negative")
    return amount


def _sha256_json(body: dict[str, Any]) -> str:
    payload = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _positive_timestamp(value: Any, *, label: str) -> float:
    if (
        type(value) not in {int, float}
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{label} requires a positive finite timestamp")
    return float(value)


def _rfc3339_utc(timestamp: float) -> str:
    return (
        datetime.fromtimestamp(timestamp, timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


__all__ = [
    "AmbiguousPodCreationError",
    "PodBillingReceipt",
    "PodCreateReceipt",
    "RunPodBudget",
    "RunPodAllocationPolicy",
    "RunPodControlPlane",
    "RunPodProviderTransport",
    "RunPodRESTTransport",
    "RunPodTransport",
]
