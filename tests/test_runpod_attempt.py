from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pytest

from post_train_engine.runpod import write_runpod_plan
from post_train_engine.runpod_attempt import (
    RunPodAttemptRunner,
    SSHRunPodRemoteExecutor,
    prepare_runpod_attempt,
    settle_runpod_billing,
)
from post_train_engine.runpod_control_plane import RunPodControlPlane
from post_train_engine.cli.main import main


R4_COMMAND = (
    "accelerate launch --num_processes 2 -m post_train_engine.cli "
    "run --config configs/gsm8k_runpod_r4.yaml --no-env"
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


class FakeRemote:
    def __init__(self, *, r4_certified: bool) -> None:
        self.r4_certified = r4_certified
        self.calls: list[str] = []

    def bootstrap(self, pod_id: str, _spec: Any) -> None:
        self.calls.append(f"bootstrap:{pod_id}")

    def run_r4(self, pod_id: str, _spec: Any) -> bool:
        self.calls.append(f"r4:{pod_id}")
        return self.r4_certified

    def run_grpo(self, pod_id: str, _spec: Any) -> None:
        self.calls.append(f"grpo:{pod_id}")

    def download_evidence(self, pod_id: str, _spec: Any) -> None:
        self.calls.append(f"download:{pod_id}")


def test_attempt_runs_r4_then_grpo_and_always_deletes(tmp_path: Path) -> None:
    attempt_dir = tmp_path / "attempt"
    plan_path = tmp_path / "runpod_plan.json"
    write_runpod_plan(
        config_path="configs/gsm8k_runpod_smoke.yaml",
        out_path=plan_path,
        image=None,
        gpu_type=None,
        command=R4_COMMAND,
        dry_run=True,
    )
    spec = prepare_runpod_attempt(
        plan_path=plan_path,
        attempt_dir=attempt_dir,
        repo_url="https://github.com/shannan-liu1/post-train-engine.git",
        commit_sha="1" * 40,
        target_spend_usd=1.5,
        settled_spend_usd=0.0,
        max_runtime_seconds=1200,
    )
    transport = FakeTransport(
        [
            [],
            {"id": "pod-1", "name": spec.pod_name, "costPerHr": 0.44},
            {},
            [],
            [],
        ]
    )
    control = RunPodControlPlane(transport, attempt_dir / "runpod_operation.json")
    remote = FakeRemote(r4_certified=True)
    watchdog_calls: list[str] = []

    result = RunPodAttemptRunner(
        control=control,
        remote=remote,
        launch_watchdog=lambda _spec, receipt: watchdog_calls.append(receipt.pod_id)
        or {"state": "armed", "pod_id": receipt.pod_id},
        clock=lambda: 1000.0,
    ).execute(spec)

    assert result.state == "billing_pending"
    assert result.r4_certified is True
    assert result.grpo_ran is True
    assert watchdog_calls == ["pod-1"]
    assert remote.calls == [
        "bootstrap:pod-1",
        "r4:pod-1",
        "grpo:pod-1",
        "download:pod-1",
    ]
    operation = json.loads(
        (attempt_dir / "runpod_operation.json").read_text("utf-8")
    )
    billing_start = quote(
        datetime.fromtimestamp(
            operation["receipt"]["recorded_at_unix"], UTC
        ).isoformat(),
        safe="",
    )
    assert [call[:2] for call in transport.calls] == [
        ("GET", "/pods"),
        ("POST", "/pods"),
        ("DELETE", "/pods/pod-1"),
        ("GET", "/pods"),
        (
            "GET",
            f"/billing/pods?podId=pod-1&startTime={billing_start}"
            "&grouping=podId&bucketSize=hour",
        ),
    ]
    state = json.loads((attempt_dir / "attempt_state.json").read_text("utf-8"))
    assert state["state"] == "billing_pending"


def test_attempt_skips_grpo_when_r4_rejects_and_deletes(tmp_path: Path) -> None:
    attempt_dir = tmp_path / "attempt"
    plan_path = tmp_path / "runpod_plan.json"
    write_runpod_plan(
        config_path="configs/gsm8k_runpod_smoke.yaml",
        out_path=plan_path,
        image=None,
        gpu_type=None,
        command=R4_COMMAND,
        dry_run=True,
    )
    spec = prepare_runpod_attempt(
        plan_path=plan_path,
        attempt_dir=attempt_dir,
        repo_url="https://github.com/shannan-liu1/post-train-engine.git",
        commit_sha="2" * 40,
        target_spend_usd=1.5,
        settled_spend_usd=0.0,
        max_runtime_seconds=1200,
    )
    transport = FakeTransport(
        [
            [],
            {"id": "pod-1", "name": spec.pod_name, "costPerHr": 0.44},
            {},
            [],
            [],
        ]
    )
    remote = FakeRemote(r4_certified=False)

    result = RunPodAttemptRunner(
        control=RunPodControlPlane(
            transport, attempt_dir / "runpod_operation.json"
        ),
        remote=remote,
        launch_watchdog=lambda _spec, receipt: {
            "state": "armed",
            "pod_id": receipt.pod_id,
        },
        clock=lambda: 1000.0,
    ).execute(spec)

    assert result.r4_certified is False
    assert result.grpo_ran is False
    assert "grpo:pod-1" not in remote.calls
    assert ("DELETE", "/pods/pod-1", None) in transport.calls


def test_attempt_reconciles_lost_delete_response_by_provider_absence(
    tmp_path: Path,
) -> None:
    attempt_dir = tmp_path / "attempt"
    plan_path = tmp_path / "runpod_plan.json"
    write_runpod_plan(
        config_path="configs/gsm8k_runpod_smoke.yaml",
        out_path=plan_path,
        image=None,
        gpu_type=None,
        command=R4_COMMAND,
        dry_run=True,
    )
    spec = prepare_runpod_attempt(
        plan_path=plan_path,
        attempt_dir=attempt_dir,
        repo_url="https://github.com/shannan-liu1/post-train-engine.git",
        commit_sha="3" * 40,
        target_spend_usd=1.5,
        settled_spend_usd=0.0,
    )
    transport = FakeTransport(
        [
            [],
            {"id": "pod-1", "name": spec.pod_name, "costPerHr": 0.44},
            TimeoutError("delete response lost"),
            [],
            [],
        ]
    )

    result = RunPodAttemptRunner(
        control=RunPodControlPlane(
            transport, attempt_dir / "runpod_operation.json"
        ),
        remote=FakeRemote(r4_certified=False),
        launch_watchdog=lambda _spec, receipt: {
            "state": "armed",
            "pod_id": receipt.pod_id,
        },
        clock=lambda: 1000.0,
    ).execute(spec)

    assert result.state == "billing_pending"
    assert [call[:2] for call in transport.calls[2:4]] == [
        ("DELETE", "/pods/pod-1"),
        ("GET", "/pods"),
    ]


def test_attempt_watchdog_failure_still_deletes_before_raising(tmp_path: Path) -> None:
    attempt_dir = tmp_path / "attempt"
    plan_path = tmp_path / "runpod_plan.json"
    write_runpod_plan(
        config_path="configs/gsm8k_runpod_smoke.yaml",
        out_path=plan_path,
        image=None,
        gpu_type=None,
        command=R4_COMMAND,
        dry_run=True,
    )
    spec = prepare_runpod_attempt(
        plan_path=plan_path,
        attempt_dir=attempt_dir,
        repo_url="https://github.com/shannan-liu1/post-train-engine.git",
        commit_sha="4" * 40,
        target_spend_usd=1.5,
        settled_spend_usd=0.0,
    )
    transport = FakeTransport(
        [
            [],
            {"id": "pod-1", "name": spec.pod_name, "costPerHr": 0.44},
            {},
            [],
        ]
    )
    remote = FakeRemote(r4_certified=True)

    with pytest.raises(RuntimeError, match="watchdog launch failed"):
        RunPodAttemptRunner(
            control=RunPodControlPlane(
                transport, attempt_dir / "runpod_operation.json"
            ),
            remote=remote,
            launch_watchdog=lambda *_args: (_ for _ in ()).throw(
                RuntimeError("watchdog launch failed")
            ),
            clock=lambda: 1000.0,
        ).execute(spec)

    assert ("DELETE", "/pods/pod-1", None) in transport.calls
    assert remote.calls == []
    assert json.loads(
        (attempt_dir / "runpod_operation.json").read_text("utf-8")
    )["state"] == "deleted"


def test_attempt_refuses_to_create_while_any_pod_is_active(tmp_path: Path) -> None:
    attempt_dir = tmp_path / "attempt"
    plan_path = tmp_path / "runpod_plan.json"
    write_runpod_plan(
        config_path="configs/gsm8k_runpod_smoke.yaml",
        out_path=plan_path,
        image=None,
        gpu_type=None,
        command=R4_COMMAND,
        dry_run=True,
    )
    spec = prepare_runpod_attempt(
        plan_path=plan_path,
        attempt_dir=attempt_dir,
        repo_url="https://github.com/shannan-liu1/post-train-engine.git",
        commit_sha="7" * 40,
        target_spend_usd=1.5,
        settled_spend_usd=0.0,
    )
    transport = FakeTransport([[{"id": "other-pod"}]])

    with pytest.raises(RuntimeError, match="zero active Pods"):
        RunPodAttemptRunner(
            control=RunPodControlPlane(
                transport, attempt_dir / "runpod_operation.json"
            ),
            remote=FakeRemote(r4_certified=True),
            launch_watchdog=lambda *_args: pytest.fail("must not arm watchdog"),
        ).execute(spec)

    assert transport.calls == [("GET", "/pods", None)]


def test_attempt_resume_preserves_original_deadline_and_billing_window(
    tmp_path: Path,
) -> None:
    attempt_dir = tmp_path / "attempt"
    plan_path = tmp_path / "runpod_plan.json"
    write_runpod_plan(
        config_path="configs/gsm8k_runpod_smoke.yaml",
        out_path=plan_path,
        image=None,
        gpu_type=None,
        command=R4_COMMAND,
        dry_run=True,
    )
    spec = prepare_runpod_attempt(
        plan_path=plan_path,
        attempt_dir=attempt_dir,
        repo_url="https://github.com/shannan-liu1/post-train-engine.git",
        commit_sha="8" * 40,
        target_spend_usd=1.5,
        settled_spend_usd=0.0,
    )
    transport = FakeTransport(
        [
            {"id": "pod-1", "name": spec.pod_name, "costPerHr": 0.44},
            [{"id": "pod-1", "name": spec.pod_name}],
            {},
            [],
            [],
        ]
    )
    control = RunPodControlPlane(transport, attempt_dir / "runpod_operation.json")
    control.create_pod(spec.create_request, budget=spec.budget)
    journal_path = attempt_dir / "runpod_operation.json"
    journal = json.loads(journal_path.read_text("utf-8"))
    journal["receipt"]["recorded_at_unix"] = 1000.0
    journal_path.write_text(json.dumps(journal), encoding="utf-8")

    result = RunPodAttemptRunner(
        control=control,
        remote=FakeRemote(r4_certified=True),
        launch_watchdog=lambda _spec, receipt: {
            "state": "armed",
            "pod_id": receipt.pod_id,
        },
        clock=lambda: 2150.0,
    ).execute(spec)

    assert result.r4_certified is True
    assert result.grpo_ran is False
    assert [method for method, _path, _body in transport.calls].count("POST") == 1
    assert "startTime=1970-01-01T00%3A16%3A40%2B00%3A00" in transport.calls[-1][1]


def test_ssh_remote_bootstrap_pins_host_and_verifies_exact_source(
    tmp_path: Path,
) -> None:
    class Control:
        @staticmethod
        def get_pod(pod_id: str) -> dict[str, Any]:
            assert pod_id == "pod-1"
            return {
                "id": "pod-1",
                "publicIp": "192.0.2.10",
                "portMappings": {"22": 12345},
            }

    attempt_dir = tmp_path / "attempt"
    plan_path = tmp_path / "runpod_plan.json"
    key_path = tmp_path / "id_ed25519"
    key_path.write_text("not-a-real-test-key", encoding="utf-8")
    write_runpod_plan(
        config_path="configs/gsm8k_runpod_smoke.yaml",
        out_path=plan_path,
        image=None,
        gpu_type=None,
        command=R4_COMMAND,
        dry_run=True,
    )
    spec = prepare_runpod_attempt(
        plan_path=plan_path,
        attempt_dir=attempt_dir,
        repo_url="https://github.com/shannan-liu1/post-train-engine.git",
        commit_sha="5" * 40,
        target_spend_usd=1.5,
        settled_spend_usd=0.0,
    )
    calls: list[list[str]] = []

    def run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if command[0] == "ssh-keyscan":
            return subprocess.CompletedProcess(
                command, 0, "[192.0.2.10]:12345 ssh-ed25519 AAAATEST\n", ""
            )
        if command[0] == "scp":
            Path(command[-1]).write_bytes(b"evidence")
        return subprocess.CompletedProcess(command, 0, "", "")

    executor = SSHRunPodRemoteExecutor(
        control=Control(),
        ssh_private_key=key_path,
        command_runner=run,
        sleep=lambda _seconds: None,
    )
    executor.bootstrap("pod-1", spec)
    executor.download_evidence("pod-1", spec)

    assert calls[0][:4] == ["ssh-keyscan", "-T", "5", "-p"]
    ssh_command = calls[1]
    assert ssh_command[0] == "ssh"
    assert "BatchMode=yes" in ssh_command
    assert "PasswordAuthentication=no" in ssh_command
    assert "StrictHostKeyChecking=yes" in ssh_command
    remote_script = ssh_command[-1]
    assert spec.commit_sha in remote_script
    assert "git checkout --detach" in remote_script
    assert "runpod_preflight.py --constraints-only" in remote_script
    assert "pip install --no-deps" in remote_script
    scp_command = calls[-1]
    assert scp_command[0] == "scp"
    assert "-P" in scp_command
    evidence_receipt = json.loads(
        (attempt_dir / "evidence_download.json").read_text("utf-8")
    )
    assert evidence_receipt["byte_count"] == len(b"evidence")
    assert evidence_receipt["sha256"].startswith("sha256:")


def test_cli_prepares_immutable_attempt_without_contacting_provider(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "runpod_plan.json"
    attempt_dir = tmp_path / "attempt"
    write_runpod_plan(
        config_path="configs/gsm8k_runpod_smoke.yaml",
        out_path=plan_path,
        image=None,
        gpu_type=None,
        command=R4_COMMAND,
        dry_run=True,
    )

    main(
        [
            "runpod",
            "attempt",
            "prepare",
            "--plan",
            str(plan_path),
            "--attempt-dir",
            str(attempt_dir),
            "--repo-url",
            "https://github.com/shannan-liu1/post-train-engine.git",
            "--commit-sha",
            "6" * 40,
            "--target-spend-usd",
            "1.5",
            "--settled-spend-usd",
            "0",
            "--max-runtime-seconds",
            "1200",
        ]
    )

    prepared = json.loads((attempt_dir / "attempt.json").read_text("utf-8"))
    assert prepared["commit_sha"] == "6" * 40
    assert prepared["budget"]["max_runtime_seconds"] == 1200
    assert prepared["create_request"]["gpuCount"] == 2
    assert prepared["create_request"]["volumeInGb"] == 0


def test_billing_settlement_requires_matching_observations(tmp_path: Path) -> None:
    transport = FakeTransport(
        [
            [{"podId": "pod-1", "amount": 0.19}],
            [{"podId": "pod-1", "amount": 0.19}],
            [],
        ]
    )
    journal = tmp_path / "runpod_operation.json"
    journal.write_text(
        json.dumps(
            {"state": "created", "receipt": {"pod_id": "pod-1"}}
        ),
        encoding="utf-8",
    )
    control = RunPodControlPlane(transport, journal)
    out = tmp_path / "billing_receipt.json"

    provisional = settle_runpod_billing(
        control=control,
        pod_id="pod-1",
        start_time="2026-07-12T21:06:51Z",
        out=out,
    )
    settled = settle_runpod_billing(
        control=control,
        pod_id="pod-1",
        start_time="2026-07-12T21:06:51Z",
        end_time="2026-07-12T21:15:09Z",
        final=True,
        out=out,
    )

    assert provisional.settlement_state == "provisional"
    assert settled.settlement_state == "settled"
    assert json.loads(out.read_text("utf-8"))["amount_usd"] == 0.19
    assert json.loads(journal.read_text("utf-8"))["state"] == "deleted"
