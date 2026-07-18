from __future__ import annotations

import json
import subprocess
import urllib.error
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest

from post_train_engine.runpod_control_plane import (
    PodCreateReceipt,
    RunPodBudget,
    RunPodControlPlane,
    RunPodCreateRejectedError,
    RunPodProviderTransport,
)
from post_train_engine.runpod_watchdog import (
    launch_local_deletion_watchdog,
    run_local_deletion_watchdog,
)


class FakeTransport:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str, dict[str, Any] | None]] = []

    def request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> Any:
        self.calls.append((method, path, body))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def _request() -> dict[str, Any]:
    return {
        "name": "pte-r4-deadbeef",
        "cloudType": "SECURE",
        "computeType": "GPU",
        "gpuTypeIds": ["NVIDIA A40"],
        "gpuCount": 2,
        "imageName": "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04@sha256:cb154fcca15d1d6ce858cfa672b76505e30861ef981d28ec94bd44168767d853",
        "allowedCudaVersions": ["12.8"],
        "containerDiskInGb": 40,
        "volumeInGb": 0,
        "interruptible": False,
        "ports": ["22/tcp"],
        "supportPublicIp": True,
    }


def _arm_watchdog() -> dict[str, str]:
    return {"state": "armed"}


def test_provider_transport_uses_graphql_provider_termination_for_create(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[Any] = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        @staticmethod
        def read(_size: int) -> bytes:
            return json.dumps(
                {
                    "data": {
                        "podFindAndDeployOnDemand": {
                            "id": "pod-1",
                            "name": "pte-r4-deadbeef",
                            "costPerHr": 0.44,
                        }
                    }
                }
            ).encode()

    def open_request(request, *, timeout):
        captured.append((request, timeout))
        return Response()

    monkeypatch.setattr(
        "post_train_engine.runpod_control_plane.open_no_redirect", open_request
    )
    body = {
        **_request(),
        "terminateAfter": "2026-07-18T00:00:00Z",
    }

    result = RunPodProviderTransport("secret").request("POST", "/pods", body)

    request = captured[0][0]
    payload = json.loads(request.data)
    assert request.full_url == "https://api.runpod.io/graphql"
    assert request.get_header("User-agent") == "post-train-engine/0.0.1"
    assert payload["variables"]["input"]["terminateAfter"] == body["terminateAfter"]
    assert payload["variables"]["input"]["gpuTypeId"] == "NVIDIA A40"
    assert result["id"] == "pod-1"


def test_runpod_transport_rejects_oversized_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        @staticmethod
        def read(size: int) -> bytes:
            return b"x" * size

    monkeypatch.setattr(
        "post_train_engine.runpod_control_plane.open_no_redirect",
        lambda *_args, **_kwargs: Response(),
    )

    with pytest.raises(RuntimeError, match="response exceeded"):
        RunPodProviderTransport("secret").request("GET", "/pods")


def test_graphql_http_rejection_preserves_bounded_provider_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = urllib.error.HTTPError(
        "https://api.runpod.io/graphql",
        403,
        "Forbidden",
        {},
        BytesIO(b'Error 1010: browser signature banned; token=secret-value'),
    )
    monkeypatch.setattr(
        "post_train_engine.runpod_control_plane.open_no_redirect",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(error),
    )

    with pytest.raises(
        RunPodCreateRejectedError,
        match="HTTP 403: Error 1010: browser signature banned",
    ) as exc_info:
        RunPodProviderTransport("secret-value").request(
            "POST",
            "/pods",
            {**_request(), "terminateAfter": "2026-07-18T00:00:00Z"},
        )

    assert "secret-value" not in str(exc_info.value)


def test_create_persists_authoritative_rate_and_budget_deadline(tmp_path: Path) -> None:
    transport = FakeTransport(
        [{"id": "pod-1", "name": "pte-r4-deadbeef", "adjustedCostPerHr": 0.44}]
    )
    control = RunPodControlPlane(transport, tmp_path / "runpod_operation.json")

    receipt = control.create_pod(
        _request(),
        budget=RunPodBudget(
            target_spend_usd=1.5,
            settled_spend_usd=0.0,
            reserve_usd=0.15,
        ),
        arm_watchdog=_arm_watchdog,
    )

    assert receipt.pod_id == "pod-1"
    assert receipt.pod_rate_usd_per_hour == 0.44
    assert receipt.hard_deadline_seconds == 1200
    journal = json.loads((tmp_path / "runpod_operation.json").read_text("utf-8"))
    assert journal["state"] == "created"
    assert journal["receipt"]["pod_rate_usd_per_hour"] == 0.44


def test_create_requires_armed_watchdog_before_provider_contact(
    tmp_path: Path,
) -> None:
    transport = FakeTransport([])

    with pytest.raises(ValueError, match="watchdog callback"):
        RunPodControlPlane(
            transport,
            tmp_path / "runpod_operation.json",
        ).create_pod(
            _request(),
            budget=RunPodBudget(
                target_spend_usd=1.5,
                settled_spend_usd=0.0,
            ),
        )

    assert transport.calls == []


def test_create_retries_safe_intent_after_watchdog_launch_failure(
    tmp_path: Path,
) -> None:
    transport = FakeTransport(
        [{"id": "pod-1", "name": "pte-r4-deadbeef", "costPerHr": 0.44}]
    )
    control = RunPodControlPlane(transport, tmp_path / "runpod_operation.json")
    budget = RunPodBudget(target_spend_usd=1.5, settled_spend_usd=0.0)

    with pytest.raises(RuntimeError, match="watchdog failed"):
        control.create_pod(
            _request(),
            budget=budget,
            arm_watchdog=lambda: (_ for _ in ()).throw(RuntimeError("watchdog failed")),
        )

    assert transport.calls == []
    assert control.operation_state() == "intent"
    receipt = control.create_pod(_request(), budget=budget, arm_watchdog=_arm_watchdog)
    assert receipt.pod_id == "pod-1"
    assert [call[0] for call in transport.calls] == ["POST"]


def test_created_inventory_allows_bounded_eventual_visibility(
    tmp_path: Path,
) -> None:
    pod = {"id": "pod-1", "name": "pte-r4-deadbeef"}
    control = RunPodControlPlane(
        FakeTransport([[], [], [pod]]),
        tmp_path / "runpod_operation.json",
        sleep=lambda _seconds: None,
    )
    control.require_only_created_pod(
        PodCreateReceipt(
            pod_id="pod-1",
            pod_name="pte-r4-deadbeef",
            pod_rate_usd_per_hour=0.44,
            hard_deadline_seconds=1200,
            request_sha256="sha256:" + "a" * 64,
            recorded_at_unix=1.0,
        )
    )


def test_create_anchors_provider_termination_and_billing_to_intent(
    tmp_path: Path,
) -> None:
    transport = FakeTransport(
        [{"id": "pod-1", "name": "pte-r4-deadbeef", "costPerHr": 0.44}]
    )
    control = RunPodControlPlane(
        transport,
        tmp_path / "runpod_operation.json",
        clock=lambda: 1000.0,
    )

    receipt = control.create_pod(
        _request(),
        budget=RunPodBudget(
            target_spend_usd=1.5,
            settled_spend_usd=0.0,
            max_runtime_seconds=1200,
        ),
        arm_watchdog=_arm_watchdog,
    )

    assert receipt.recorded_at_unix == 1000.0
    assert transport.calls[0][2]["terminateAfter"] == "1970-01-01T00:36:40Z"


def test_delete_updates_the_canonical_operation_journal(tmp_path: Path) -> None:
    journal_path = tmp_path / "runpod_operation.json"
    transport = FakeTransport(
        [
            {"id": "pod-1", "name": "pte-r4-deadbeef", "costPerHr": 0.44},
            {},
        ]
    )
    control = RunPodControlPlane(transport, journal_path)
    control.create_pod(
        _request(),
        budget=RunPodBudget(target_spend_usd=1.5, settled_spend_usd=0.0),
        arm_watchdog=_arm_watchdog,
    )

    control.delete_pod("pod-1")

    journal = json.loads(journal_path.read_text("utf-8"))
    assert journal["state"] == "delete_requested"
    assert journal["delete_requested_pod_id"] == "pod-1"

    transport.responses.extend([[], []])
    assert control.verify_pod_absent("pod-1") is True
    journal = json.loads(journal_path.read_text("utf-8"))
    assert journal["state"] == "deleted"
    assert journal["deleted_pod_id"] == "pod-1"
    assert journal["deleted_at_unix"] > journal["receipt"]["recorded_at_unix"]


def test_corrupt_journal_cannot_block_provider_deletion(tmp_path: Path) -> None:
    journal_path = tmp_path / "runpod_operation.json"
    journal_path.write_text("{", encoding="utf-8")
    transport = FakeTransport([{}])

    with pytest.raises(json.JSONDecodeError):
        RunPodControlPlane(transport, journal_path).delete_pod("pod-1")

    assert transport.calls == [("DELETE", "/pods/pod-1", None)]


def test_budget_deadline_subtracts_settled_campaign_spend() -> None:
    budget = RunPodBudget(
        target_spend_usd=1.5,
        settled_spend_usd=0.4,
        reserve_usd=0.15,
    )

    assert budget.hard_deadline_seconds(0.44) == 1200


def test_budget_uses_cost_deadline_when_it_is_stricter_than_runtime_cap() -> None:
    budget = RunPodBudget(
        target_spend_usd=1.5,
        settled_spend_usd=1.2,
        reserve_usd=0.15,
        max_runtime_seconds=1800,
    )

    assert budget.hard_deadline_seconds(0.44) == 1227


def test_budget_requires_explicit_settled_campaign_spend() -> None:
    with pytest.raises(ValueError, match="settled_spend_usd"):
        RunPodBudget(target_spend_usd=1.5)


def test_create_reconciles_ambiguous_submission_without_second_post(
    tmp_path: Path,
) -> None:
    journal = tmp_path / "runpod_operation.json"
    transport = FakeTransport(
        [
            TimeoutError("response lost"),
            [{"id": "pod-1", "name": "pte-r4-deadbeef", "costPerHr": 0.44}],
        ]
    )
    control = RunPodControlPlane(transport, journal)

    receipt = control.create_pod(
        _request(),
        budget=RunPodBudget(target_spend_usd=1.5, settled_spend_usd=0.0),
        arm_watchdog=_arm_watchdog,
    )

    assert receipt.pod_id == "pod-1"
    assert [call[0] for call in transport.calls] == ["POST", "GET"]


def test_create_records_definitive_rejection_without_ambiguous_reconciliation(
    tmp_path: Path,
) -> None:
    journal = tmp_path / "runpod_operation.json"
    transport = FakeTransport([RunPodCreateRejectedError("HTTP 403")])
    control = RunPodControlPlane(transport, journal)

    with pytest.raises(RunPodCreateRejectedError, match="HTTP 403"):
        control.create_pod(
            _request(),
            budget=RunPodBudget(target_spend_usd=1.5, settled_spend_usd=0.0),
            arm_watchdog=_arm_watchdog,
        )

    assert [call[0] for call in transport.calls] == ["POST"]
    assert json.loads(journal.read_text(encoding="utf-8"))["state"] == "rejected"


def test_create_reconciles_success_response_without_pod_id(tmp_path: Path) -> None:
    transport = FakeTransport(
        [
            {"name": "pte-r4-deadbeef", "costPerHr": 0.44},
            [{"id": "pod-1", "name": "pte-r4-deadbeef", "costPerHr": 0.44}],
        ]
    )

    receipt = RunPodControlPlane(
        transport, tmp_path / "runpod_operation.json"
    ).create_pod(
        _request(),
        budget=RunPodBudget(target_spend_usd=1.5, settled_spend_usd=0.0),
        arm_watchdog=_arm_watchdog,
    )

    assert receipt.pod_id == "pod-1"
    assert [call[0] for call in transport.calls] == ["POST", "GET"]


def test_create_deletes_pod_when_rate_cannot_fit_minimum_runtime(
    tmp_path: Path,
) -> None:
    transport = FakeTransport(
        [
            {"id": "pod-1", "name": "pte-r4-deadbeef", "costPerHr": 4.0},
            {},
            [],
            [],
        ]
    )
    control = RunPodControlPlane(transport, tmp_path / "runpod_operation.json")

    with pytest.raises(ValueError, match="cannot fit the minimum runtime"):
        control.create_pod(
            _request(),
            budget=RunPodBudget(
                target_spend_usd=1.5,
                settled_spend_usd=0.0,
                reserve_usd=0.15,
                minimum_runtime_seconds=1800,
                max_runtime_seconds=1800,
            ),
            arm_watchdog=_arm_watchdog,
        )

    assert [call[:2] for call in transport.calls[-3:]] == [
        ("DELETE", "/pods/pod-1"),
        ("GET", "/pods"),
        ("GET", "/pods"),
    ]


def test_billing_remains_pending_until_provider_returns_pod_rows(
    tmp_path: Path,
) -> None:
    transport = FakeTransport(
        [
            [],
            [{"podId": "pod-1", "amount": 0.37}],
            [{"podId": "pod-1", "amount": 0.37}],
            [],
        ]
    )
    control = RunPodControlPlane(transport, tmp_path / "runpod_operation.json")

    pending = control.fetch_billing("pod-1", start_time="2026-07-11T00:00:00Z")
    provisional = control.fetch_billing("pod-1", start_time="2026-07-11T00:00:00Z")
    settled = control.fetch_billing(
        "pod-1",
        start_time="2026-07-11T00:00:00Z",
        end_time="2026-07-11T01:00:00Z",
        final=True,
    )

    assert pending.settlement_state == "pending"
    assert pending.amount_usd is None
    assert provisional.settlement_state == "provisional"
    assert provisional.amount_usd == 0.37
    assert settled.settlement_state == "settled"
    assert settled.amount_usd == 0.37
    assert "grouping=podId" in transport.calls[-2][1]
    assert transport.calls[-1][:2] == ("GET", "/pods")


def test_get_pod_requires_the_requested_provider_identity(tmp_path: Path) -> None:
    transport = FakeTransport(
        [{"id": "pod-1", "publicIp": "192.0.2.10", "portMappings": {"22": 12345}}]
    )

    pod = RunPodControlPlane(transport, tmp_path / "runpod_operation.json").get_pod(
        "pod-1"
    )

    assert pod["publicIp"] == "192.0.2.10"
    assert transport.calls == [("GET", "/pods/pod-1", None)]


def test_nonempty_billing_is_not_final_without_explicit_teardown_boundary(
    tmp_path: Path,
) -> None:
    transport = FakeTransport([[{"podId": "pod-1", "amount": 0.12}]])
    receipt = RunPodControlPlane(
        transport, tmp_path / "runpod_operation.json"
    ).fetch_billing("pod-1", start_time="2026-07-11T00:00:00Z")

    assert receipt.settlement_state == "provisional"
    assert receipt.amount_usd == 0.12


def test_malformed_create_rate_deletes_known_pod(tmp_path: Path) -> None:
    transport = FakeTransport(
        [
            {"id": "pod-1", "name": "pte-r4-deadbeef"},
            {},
            [{"id": "pod-1"}],
            {},
            [],
            [],
        ]
    )

    with pytest.raises(ValueError, match="authoritative Pod rate"):
        RunPodControlPlane(
            transport,
            tmp_path / "runpod_operation.json",
            sleep=lambda _seconds: None,
        ).create_pod(
            _request(),
            budget=RunPodBudget(target_spend_usd=1.5, settled_spend_usd=0.0),
            arm_watchdog=_arm_watchdog,
        )

    assert [call[:2] for call in transport.calls[1:]] == [
        ("DELETE", "/pods/pod-1"),
        ("GET", "/pods"),
        ("DELETE", "/pods/pod-1"),
        ("GET", "/pods"),
        ("GET", "/pods"),
    ]


def test_create_receipt_write_failure_deletes_known_pod(tmp_path: Path) -> None:
    class FailingReceiptControlPlane(RunPodControlPlane):
        writes = 0

        def _write_journal(self, body):
            self.writes += 1
            if self.writes == 3:
                raise OSError("receipt disk failure")
            super()._write_journal(body)

    transport = FakeTransport(
        [
            {"id": "pod-1", "name": "pte-r4-deadbeef", "costPerHr": 0.44},
            {},
            [],
            [],
        ]
    )

    with pytest.raises(OSError, match="receipt disk failure"):
        FailingReceiptControlPlane(
            transport, tmp_path / "runpod_operation.json"
        ).create_pod(
            _request(),
            budget=RunPodBudget(target_spend_usd=1.5, settled_spend_usd=0.0),
            arm_watchdog=_arm_watchdog,
        )

    assert [call[:2] for call in transport.calls[-3:]] == [
        ("DELETE", "/pods/pod-1"),
        ("GET", "/pods"),
        ("GET", "/pods"),
    ]


def test_create_rejects_non_secure_or_persistent_request_before_api(
    tmp_path: Path,
) -> None:
    request = _request()
    request["cloudType"] = "COMMUNITY"
    request["volumeInGb"] = 20
    transport = FakeTransport([])

    with pytest.raises(ValueError, match="Secure Cloud"):
        RunPodControlPlane(transport, tmp_path / "runpod_operation.json").create_pod(
            request,
            budget=RunPodBudget(target_spend_usd=1.5, settled_spend_usd=0.0),
        )

    assert transport.calls == []


def test_created_journal_cannot_be_replayed_under_different_budget(
    tmp_path: Path,
) -> None:
    transport = FakeTransport(
        [{"id": "pod-1", "name": "pte-r4-deadbeef", "costPerHr": 0.44}]
    )
    control = RunPodControlPlane(transport, tmp_path / "runpod_operation.json")
    control.create_pod(
        _request(),
        budget=RunPodBudget(target_spend_usd=1.5, settled_spend_usd=0.0),
        arm_watchdog=_arm_watchdog,
    )

    with pytest.raises(ValueError, match="different budget"):
        control.create_pod(
            _request(),
            budget=RunPodBudget(target_spend_usd=1.0, settled_spend_usd=0.0),
            arm_watchdog=_arm_watchdog,
        )


def test_created_journal_cannot_resume_under_different_attempt_identity(
    tmp_path: Path,
) -> None:
    transport = FakeTransport(
        [{"id": "pod-1", "name": "pte-r4-deadbeef", "costPerHr": 0.44}]
    )
    control = RunPodControlPlane(transport, tmp_path / "runpod_operation.json")
    control.create_pod(
        _request(),
        budget=RunPodBudget(target_spend_usd=1.5, settled_spend_usd=0.0),
        operation_sha256="sha256:" + "a" * 64,
        arm_watchdog=_arm_watchdog,
    )

    with pytest.raises(ValueError, match="different attempt identity"):
        control.create_pod(
            _request(),
            budget=RunPodBudget(target_spend_usd=1.5, settled_spend_usd=0.0),
            operation_sha256="sha256:" + "b" * 64,
            arm_watchdog=_arm_watchdog,
        )


def test_local_watchdog_launches_detached_without_serializing_api_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    journal = tmp_path / "runpod_operation.json"
    journal.write_text(
        json.dumps(
            {
                "state": "created",
                "receipt": {
                    "pod_id": "pod-1",
                    "hard_deadline_seconds": 120,
                    "recorded_at_unix": 1000.0,
                },
            }
        ),
        encoding="utf-8",
    )
    calls: list[tuple[list[str], dict[str, Any]]] = []

    class Process:
        pid = 1234

        @staticmethod
        def poll() -> None:
            return None

    def spawn(command: list[str], **kwargs: Any) -> Process:
        calls.append((command, kwargs))
        (tmp_path / "watchdog.json").write_text(
            json.dumps({"state": "ready"}), encoding="utf-8"
        )
        return Process()

    monkeypatch.setenv("UNRELATED_SECRET", "must-not-reach-child")
    receipt = launch_local_deletion_watchdog(
        journal_path=journal,
        receipt_path=tmp_path / "watchdog.json",
        log_path=tmp_path / "watchdog.log",
        api_key="super-secret",
        spawn=spawn,
        clock=lambda: 1001.0,
    )

    command, kwargs = calls[0]
    assert receipt["state"] == "armed"
    assert receipt["pod_id"] == "pod-1"
    assert receipt["pid"] == 1234
    assert receipt["delete_at_unix"] == 1120.0
    assert "super-secret" not in " ".join(command)
    assert kwargs["env"]["PTE_REMOTE_RUNPOD_ALL"] == "super-secret"
    assert "UNRELATED_SECRET" not in kwargs["env"]
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["close_fds"] is True


def test_local_watchdog_arms_from_precreate_intent(
    tmp_path: Path,
) -> None:
    journal = tmp_path / "runpod_operation.json"
    receipt_path = tmp_path / "watchdog.json"
    journal.write_text(
        json.dumps(
            {
                "state": "intent",
                "pod_name": "pte-r4-deadbeef",
                "intent_at_unix": 1000.0,
                "budget": {
                    "minimum_runtime_seconds": 60,
                    "max_runtime_seconds": 1200,
                },
            }
        ),
        encoding="utf-8",
    )

    class Process:
        pid = 1234

        @staticmethod
        def poll() -> None:
            return None

    def spawn(_command, **_kwargs):
        receipt_path.write_text(json.dumps({"state": "ready"}), encoding="utf-8")
        return Process()

    result = launch_local_deletion_watchdog(
        journal_path=journal,
        receipt_path=receipt_path,
        log_path=tmp_path / "watchdog.log",
        api_key="secret",
        spawn=spawn,
        sleep=lambda _seconds: None,
        clock=lambda: 1001.0,
    )

    assert result["state"] == "armed"
    assert result["pod_name"] == "pte-r4-deadbeef"
    assert result["delete_at_unix"] == 1060.0


def test_expired_watchdog_target_deletes_synchronously_without_spawning(
    tmp_path: Path,
) -> None:
    journal = tmp_path / "runpod_operation.json"
    journal.write_text(
        json.dumps(
            {
                "state": "created",
                "receipt": {
                    "pod_id": "pod-1",
                    "hard_deadline_seconds": 120,
                    "recorded_at_unix": 1000.0,
                },
            }
        ),
        encoding="utf-8",
    )
    transport = FakeTransport([{}, [], []])

    result = launch_local_deletion_watchdog(
        journal_path=journal,
        receipt_path=tmp_path / "watchdog.json",
        log_path=tmp_path / "watchdog.log",
        api_key="super-secret",
        spawn=lambda *_args, **_kwargs: pytest.fail("expired target must not spawn"),
        clock=lambda: 1120.0,
        transport_factory=lambda _key: transport,
    )

    assert result["state"] == "deleted"
    assert transport.calls == [
        ("DELETE", "/pods/pod-1", None),
        ("GET", "/pods", None),
        ("GET", "/pods", None),
    ]


def test_local_watchdog_deletes_literal_journal_pod_and_records_result(
    tmp_path: Path,
) -> None:
    journal = tmp_path / "runpod_operation.json"
    receipt_path = tmp_path / "watchdog.json"
    journal.write_text(
        json.dumps(
            {
                "state": "created",
                "receipt": {
                    "pod_id": "pod-1",
                    "hard_deadline_seconds": 120,
                    "recorded_at_unix": 1000.0,
                },
            }
        ),
        encoding="utf-8",
    )
    transport = FakeTransport([{}, [], []])
    sleeps: list[float] = []

    result = run_local_deletion_watchdog(
        journal_path=journal,
        receipt_path=receipt_path,
        api_key="super-secret",
        sleep=sleeps.append,
        clock=lambda: 1020.0,
        transport_factory=lambda _key: transport,
    )

    assert sleeps == [100.0, 0.25]
    assert transport.calls == [
        ("DELETE", "/pods/pod-1", None),
        ("GET", "/pods", None),
        ("GET", "/pods", None),
    ]
    assert result["state"] == "deleted"
    assert result["pod_id"] == "pod-1"
    assert "super-secret" not in receipt_path.read_text("utf-8")


def test_local_watchdog_reconciles_and_deletes_late_ambiguous_create(
    tmp_path: Path,
) -> None:
    journal = tmp_path / "runpod_operation.json"
    receipt_path = tmp_path / "watchdog.json"
    journal.write_text(
        json.dumps(
            {
                "state": "ambiguous",
                "pod_name": "pte-r4-deadbeef",
                "intent_at_unix": 1000.0,
                "budget": {
                    "minimum_runtime_seconds": 60,
                    "max_runtime_seconds": 1200,
                },
            }
        ),
        encoding="utf-8",
    )
    transport = FakeTransport(
        [
            [{"id": "pod-late", "name": "pte-r4-deadbeef"}],
            {},
            [],
            [],
        ]
    )

    result = run_local_deletion_watchdog(
        journal_path=journal,
        receipt_path=receipt_path,
        api_key="secret",
        sleep=lambda _seconds: None,
        clock=lambda: 1060.0,
        transport_factory=lambda _key: transport,
    )

    assert result["state"] == "deleted"
    assert result["pod_id"] == "pod-late"
    assert [call[:2] for call in transport.calls] == [
        ("GET", "/pods"),
        ("DELETE", "/pods/pod-late"),
        ("GET", "/pods"),
        ("GET", "/pods"),
    ]


def test_local_watchdog_closes_definitively_rejected_create(
    tmp_path: Path,
) -> None:
    journal = tmp_path / "runpod_operation.json"
    receipt_path = tmp_path / "watchdog.json"
    journal.write_text(
        json.dumps(
            {
                "state": "rejected",
                "pod_name": "pte-r4-deadbeef",
                "intent_at_unix": 1000.0,
                "budget": {
                    "minimum_runtime_seconds": 60,
                    "max_runtime_seconds": 1200,
                },
            }
        ),
        encoding="utf-8",
    )
    transport = FakeTransport([[]])

    result = run_local_deletion_watchdog(
        journal_path=journal,
        receipt_path=receipt_path,
        api_key="secret",
        sleep=lambda _seconds: None,
        clock=lambda: 1060.0,
        transport_factory=lambda _key: transport,
    )

    assert result["state"] == "absent"
    assert result["pod_name"] == "pte-r4-deadbeef"
    assert [call[:2] for call in transport.calls] == [("GET", "/pods")]


def test_local_watchdog_keeps_reconciling_ambiguous_create_until_provider_ttl(
    tmp_path: Path,
) -> None:
    journal = tmp_path / "runpod_operation.json"
    receipt_path = tmp_path / "watchdog.json"
    journal.write_text(
        json.dumps(
            {
                "state": "ambiguous",
                "pod_name": "pte-r4-deadbeef",
                "intent_at_unix": 1000.0,
                "budget": {
                    "minimum_runtime_seconds": 60,
                    "max_runtime_seconds": 1200,
                },
            }
        ),
        encoding="utf-8",
    )
    transport = FakeTransport(
        [
            [],
            [{"id": "pod-late", "name": "pte-r4-deadbeef"}],
            {},
            [],
            [],
        ]
    )
    now = [1060.0]

    def advance(seconds: float) -> None:
        now[0] += seconds

    result = run_local_deletion_watchdog(
        journal_path=journal,
        receipt_path=receipt_path,
        api_key="secret",
        sleep=advance,
        clock=lambda: now[0],
        transport_factory=lambda _key: transport,
    )

    assert result["state"] == "deleted"
    assert result["pod_id"] == "pod-late"
    assert now[0] == 1070.25
    assert [call[:2] for call in transport.calls] == [
        ("GET", "/pods"),
        ("GET", "/pods"),
        ("DELETE", "/pods/pod-late"),
        ("GET", "/pods"),
        ("GET", "/pods"),
    ]


def test_local_watchdog_reconciles_a_lost_delete_response_by_provider_absence(
    tmp_path: Path,
) -> None:
    journal = tmp_path / "runpod_operation.json"
    receipt_path = tmp_path / "watchdog.json"
    journal.write_text(
        json.dumps(
            {
                "state": "created",
                "receipt": {
                    "pod_id": "pod-1",
                    "hard_deadline_seconds": 120,
                    "recorded_at_unix": 1000.0,
                },
            }
        ),
        encoding="utf-8",
    )
    transport = FakeTransport([TimeoutError("response lost"), [], []])

    result = run_local_deletion_watchdog(
        journal_path=journal,
        receipt_path=receipt_path,
        api_key="super-secret",
        sleep=lambda _seconds: None,
        clock=lambda: 1120.0,
        transport_factory=lambda _key: transport,
    )

    assert result["state"] == "deleted"
    operation = json.loads(journal.read_text("utf-8"))
    assert operation["state"] == "deleted"
    assert operation["deletion_outcome"] == "provider_absent"


def test_local_watchdog_rechecks_provider_when_journal_claims_deleted(
    tmp_path: Path,
) -> None:
    journal = tmp_path / "runpod_operation.json"
    journal.write_text(
        json.dumps(
            {
                "state": "deleted",
                "receipt": {
                    "pod_id": "pod-1",
                    "hard_deadline_seconds": 1200,
                    "recorded_at_unix": 1000.0,
                },
                "deleted_pod_id": "pod-1",
            }
        ),
        encoding="utf-8",
    )
    transport = FakeTransport([[], []])

    result = run_local_deletion_watchdog(
        journal_path=journal,
        receipt_path=tmp_path / "watchdog.json",
        api_key="secret",
        sleep=lambda _seconds: None,
        clock=lambda: 1100.0,
        transport_factory=lambda _key: transport,
    )

    assert result["state"] == "absent"
    assert [call[:2] for call in transport.calls] == [
        ("GET", "/pods"),
        ("GET", "/pods"),
    ]


def test_local_watchdog_rejects_deleted_pod_identity_mismatch(tmp_path: Path) -> None:
    journal = tmp_path / "runpod_operation.json"
    journal.write_text(
        json.dumps(
            {
                "state": "deleted",
                "receipt": {
                    "pod_id": "pod-1",
                    "hard_deadline_seconds": 1200,
                    "recorded_at_unix": 1000.0,
                },
                "deleted_pod_id": "pod-other",
            }
        ),
        encoding="utf-8",
    )
    transport = FakeTransport([])

    with pytest.raises(ValueError, match="Pod identity changed"):
        run_local_deletion_watchdog(
            journal_path=journal,
            receipt_path=tmp_path / "watchdog.json",
            api_key="secret",
            sleep=lambda _seconds: None,
            clock=lambda: 1100.0,
            transport_factory=lambda _key: transport,
        )

    assert transport.calls == []


def test_local_watchdog_retries_until_provider_confirms_absence(
    tmp_path: Path,
) -> None:
    journal = tmp_path / "runpod_operation.json"
    receipt_path = tmp_path / "watchdog.json"
    journal.write_text(
        json.dumps(
            {
                "state": "created",
                "receipt": {
                    "pod_id": "pod-1",
                    "hard_deadline_seconds": 120,
                    "recorded_at_unix": 1000.0,
                },
            }
        ),
        encoding="utf-8",
    )
    transport = FakeTransport([{}, [{"id": "pod-1"}], {}, [], []])
    sleeps: list[float] = []

    result = run_local_deletion_watchdog(
        journal_path=journal,
        receipt_path=receipt_path,
        api_key="super-secret",
        sleep=sleeps.append,
        clock=lambda: 1120.0,
        transport_factory=lambda _key: transport,
    )

    assert result["state"] == "deleted"
    assert sleeps == [1.0, 0.25]
    assert [call[:2] for call in transport.calls] == [
        ("DELETE", "/pods/pod-1"),
        ("GET", "/pods"),
        ("DELETE", "/pods/pod-1"),
        ("GET", "/pods"),
        ("GET", "/pods"),
    ]


def test_local_watchdog_records_unverified_state_after_retry_exhaustion(
    tmp_path: Path,
) -> None:
    journal = tmp_path / "runpod_operation.json"
    receipt_path = tmp_path / "watchdog.json"
    journal.write_text(
        json.dumps(
            {
                "state": "created",
                "receipt": {
                    "pod_id": "pod-1",
                    "hard_deadline_seconds": 120,
                    "recorded_at_unix": 1000.0,
                },
            }
        ),
        encoding="utf-8",
    )
    transport = FakeTransport(
        [
            {},
            [{"id": "pod-1"}],
            {},
            [{"id": "pod-1"}],
            {},
            [{"id": "pod-1"}],
        ]
    )

    with pytest.raises(RuntimeError, match="still reports the Pod active"):
        run_local_deletion_watchdog(
            journal_path=journal,
            receipt_path=receipt_path,
            api_key="super-secret",
            sleep=lambda _seconds: None,
            clock=lambda: 1120.0,
            transport_factory=lambda _key: transport,
        )

    operation = json.loads(journal.read_text("utf-8"))
    assert operation["state"] == "delete_unverified"


def test_local_watchdog_persists_failure_when_journal_update_is_corrupt(
    tmp_path: Path,
) -> None:
    journal = tmp_path / "runpod_operation.json"
    receipt_path = tmp_path / "watchdog.json"
    journal.write_text(
        json.dumps(
            {
                "state": "created",
                "receipt": {
                    "pod_id": "pod-1",
                    "hard_deadline_seconds": 120,
                    "recorded_at_unix": 1000.0,
                },
            }
        ),
        encoding="utf-8",
    )

    class CorruptingTransport(FakeTransport):
        def request(self, method, path, body=None):
            result = super().request(method, path, body)
            if len(self.calls) == 6:
                journal.write_text("{", encoding="utf-8")
            return result

    transport = CorruptingTransport(
        [
            {},
            [{"id": "pod-1"}],
            {},
            [{"id": "pod-1"}],
            {},
            [{"id": "pod-1"}],
        ]
    )

    with pytest.raises(RuntimeError, match="still reports the Pod active"):
        run_local_deletion_watchdog(
            journal_path=journal,
            receipt_path=receipt_path,
            api_key="super-secret",
            sleep=lambda _seconds: None,
            clock=lambda: 1120.0,
            transport_factory=lambda _key: transport,
        )

    receipt = json.loads(receipt_path.read_text("utf-8"))
    assert receipt["state"] == "failed"
    assert receipt["journal_error_type"] == "JSONDecodeError"
