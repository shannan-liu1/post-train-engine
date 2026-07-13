from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from post_train_engine.runpod_control_plane import (
    AmbiguousPodCreationError,
    RunPodBudget,
    RunPodControlPlane,
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
        "imageName": "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04",
        "allowedCudaVersions": ["12.8"],
        "containerDiskInGb": 40,
        "volumeInGb": 0,
        "interruptible": False,
        "ports": ["22/tcp"],
        "supportPublicIp": True,
    }


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
    )

    assert receipt.pod_id == "pod-1"
    assert receipt.pod_rate_usd_per_hour == 0.44
    assert receipt.hard_deadline_seconds == 11045
    journal = json.loads((tmp_path / "runpod_operation.json").read_text("utf-8"))
    assert journal["state"] == "created"
    assert journal["receipt"]["pod_rate_usd_per_hour"] == 0.44


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
    )

    control.delete_pod("pod-1")

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

    assert budget.hard_deadline_seconds(0.44) == 7772


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

    with pytest.raises(AmbiguousPodCreationError):
        control.create_pod(
            _request(),
            budget=RunPodBudget(target_spend_usd=1.5, settled_spend_usd=0.0),
        )
    receipt = control.create_pod(
        _request(),
        budget=RunPodBudget(target_spend_usd=1.5, settled_spend_usd=0.0),
    )

    assert receipt.pod_id == "pod-1"
    assert [call[0] for call in transport.calls] == ["POST", "GET"]


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
            ),
        )

    assert transport.calls[-1][:2] == ("DELETE", "/pods/pod-1")


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
    provisional = control.fetch_billing(
        "pod-1", start_time="2026-07-11T00:00:00Z"
    )
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
        [{"id": "pod-1", "name": "pte-r4-deadbeef"}, {}]
    )

    with pytest.raises(ValueError, match="authoritative Pod rate"):
        RunPodControlPlane(
            transport, tmp_path / "runpod_operation.json"
        ).create_pod(
            _request(),
            budget=RunPodBudget(target_spend_usd=1.5, settled_spend_usd=0.0),
        )

    assert transport.calls[-1][:2] == ("DELETE", "/pods/pod-1")


def test_create_receipt_write_failure_deletes_known_pod(tmp_path: Path) -> None:
    class FailingReceiptControlPlane(RunPodControlPlane):
        writes = 0

        def _write_journal(self, body):
            self.writes += 1
            if self.writes == 2:
                raise OSError("receipt disk failure")
            super()._write_journal(body)

    transport = FakeTransport(
        [
            {"id": "pod-1", "name": "pte-r4-deadbeef", "costPerHr": 0.44},
            {},
        ]
    )

    with pytest.raises(OSError, match="receipt disk failure"):
        FailingReceiptControlPlane(
            transport, tmp_path / "runpod_operation.json"
        ).create_pod(
            _request(),
            budget=RunPodBudget(target_spend_usd=1.5, settled_spend_usd=0.0),
        )

    assert transport.calls[-1][:2] == ("DELETE", "/pods/pod-1")


def test_create_rejects_non_secure_or_persistent_request_before_api(
    tmp_path: Path,
) -> None:
    request = _request()
    request["cloudType"] = "COMMUNITY"
    request["volumeInGb"] = 20
    transport = FakeTransport([])

    with pytest.raises(ValueError, match="Secure Cloud"):
        RunPodControlPlane(
            transport, tmp_path / "runpod_operation.json"
        ).create_pod(
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
    )

    with pytest.raises(ValueError, match="different budget"):
        control.create_pod(
            _request(),
            budget=RunPodBudget(target_spend_usd=1.0, settled_spend_usd=0.0),
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
    assert kwargs["env"]["RUNPOD_API_KEY"] == "super-secret"
    assert "UNRELATED_SECRET" not in kwargs["env"]
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["close_fds"] is True


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
    transport = FakeTransport([{}, []])

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
    transport = FakeTransport([{}, []])
    sleeps: list[float] = []

    result = run_local_deletion_watchdog(
        journal_path=journal,
        receipt_path=receipt_path,
        api_key="super-secret",
        sleep=sleeps.append,
        clock=lambda: 1020.0,
        transport_factory=lambda _key: transport,
    )

    assert sleeps == [100.0]
    assert transport.calls == [
        ("DELETE", "/pods/pod-1", None),
        ("GET", "/pods", None),
    ]
    assert result["state"] == "deleted"
    assert result["pod_id"] == "pod-1"
    assert "super-secret" not in receipt_path.read_text("utf-8")


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
    transport = FakeTransport([TimeoutError("response lost"), []])

    result = run_local_deletion_watchdog(
        journal_path=journal,
        receipt_path=receipt_path,
        api_key="super-secret",
        sleep=lambda _seconds: None,
        clock=lambda: 1120.0,
        transport_factory=lambda _key: transport,
    )

    assert result["state"] == "absent"
    operation = json.loads(journal.read_text("utf-8"))
    assert operation["state"] == "deleted"
    assert operation["deletion_outcome"] == "provider_absent"


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
    transport = FakeTransport([{}, [{"id": "pod-1"}], {}, []])
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
    assert sleeps == [0.0, 2.0]
    assert [call[:2] for call in transport.calls] == [
        ("DELETE", "/pods/pod-1"),
        ("GET", "/pods"),
        ("DELETE", "/pods/pod-1"),
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
