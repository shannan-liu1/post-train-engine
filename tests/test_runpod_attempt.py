from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pytest

from post_train_engine.runpod import write_runpod_plan
from post_train_engine.runpod_attempt import (
    RunPodAttemptCleanupError,
    RunPodAttemptRunner,
    RunPodAttemptSpec,
    SSHRunPodRemoteExecutor,
    load_runpod_attempt,
    prepare_runpod_attempt,
    settle_runpod_billing,
    verify_runpod_attempt_source,
)
from post_train_engine.runpod_control_plane import RunPodBudget, RunPodControlPlane
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
            [{"id": "pod-1", "name": spec.pod_name}],
            {},
            [],
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
        launch_watchdog=lambda _spec, _receipt: (
            watchdog_calls.append(spec.pod_name)
            or {"state": "armed", "pod_name": spec.pod_name}
        ),
        clock=lambda: 1000.0,
    ).execute(spec)

    assert result.state == "billing_pending"
    assert result.r4_certified is True
    assert result.grpo_ran is True
    assert watchdog_calls == [spec.pod_name]
    assert remote.calls == [
        "bootstrap:pod-1",
        "r4:pod-1",
        "grpo:pod-1",
        "download:pod-1",
    ]
    operation = json.loads((attempt_dir / "runpod_operation.json").read_text("utf-8"))
    billing_start = quote(
        datetime.fromtimestamp(
            operation["receipt"]["recorded_at_unix"], UTC
        ).isoformat(),
        safe="",
    )
    assert [call[:2] for call in transport.calls] == [
        ("GET", "/pods"),
        ("POST", "/pods"),
        ("GET", "/pods"),
        ("DELETE", "/pods/pod-1"),
        ("GET", "/pods"),
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
            [{"id": "pod-1", "name": spec.pod_name}],
            {},
            [],
            [],
            [],
        ]
    )
    remote = FakeRemote(r4_certified=False)

    result = RunPodAttemptRunner(
        control=RunPodControlPlane(transport, attempt_dir / "runpod_operation.json"),
        remote=remote,
        launch_watchdog=lambda _spec, _receipt: {
            "state": "armed",
            "pod_name": spec.pod_name,
        },
        clock=lambda: 1000.0,
    ).execute(spec)

    assert result.r4_certified is False
    assert result.grpo_ran is False
    assert "grpo:pod-1" not in remote.calls
    assert ("DELETE", "/pods/pod-1", None) in transport.calls


def test_attempt_refuses_bootstrap_without_full_r4_cleanup_budget(
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
        commit_sha="a" * 40,
        target_spend_usd=1.5,
        settled_spend_usd=0.0,
        max_runtime_seconds=1200,
    )
    transport = FakeTransport(
        [
            [],
            {"id": "pod-1", "name": spec.pod_name, "costPerHr": 0.44},
            [{"id": "pod-1", "name": spec.pod_name}],
            {},
            [],
            [],
        ]
    )
    remote = FakeRemote(r4_certified=True)

    with pytest.raises(TimeoutError, match="full R4 and cleanup budget"):
        RunPodAttemptRunner(
            control=RunPodControlPlane(
                transport,
                attempt_dir / "runpod_operation.json",
                clock=lambda: 1000.0,
            ),
            remote=remote,
            launch_watchdog=lambda _spec, _receipt: {
                "state": "armed",
                "pod_name": spec.pod_name,
            },
            clock=lambda: 1121.0,
        ).execute(spec)

    assert remote.calls == ["download:pod-1"]
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
            [{"id": "pod-1", "name": spec.pod_name}],
            TimeoutError("delete response lost"),
            [],
            [],
            [],
        ]
    )

    result = RunPodAttemptRunner(
        control=RunPodControlPlane(transport, attempt_dir / "runpod_operation.json"),
        remote=FakeRemote(r4_certified=False),
        launch_watchdog=lambda _spec, _receipt: {
            "state": "armed",
            "pod_name": spec.pod_name,
        },
        clock=lambda: 1000.0,
    ).execute(spec)

    assert result.state == "billing_pending"
    assert [call[:2] for call in transport.calls[3:5]] == [
        ("DELETE", "/pods/pod-1"),
        ("GET", "/pods"),
    ]


def test_attempt_arms_name_watchdog_before_provider_create(tmp_path: Path) -> None:
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
        commit_sha="9" * 40,
        target_spend_usd=1.5,
        settled_spend_usd=0.0,
    )
    transport = FakeTransport([[], TimeoutError("create response lost"), [], [], []])
    control = RunPodControlPlane(
        transport,
        attempt_dir / "runpod_operation.json",
        sleep=lambda _seconds: None,
    )
    armed_after_calls: list[int] = []

    def arm(_spec, receipt):
        assert receipt is None
        operation = json.loads(
            (attempt_dir / "runpod_operation.json").read_text("utf-8")
        )
        assert operation["state"] == "intent"
        armed_after_calls.append(len(transport.calls))
        return {"state": "armed", "pod_name": spec.pod_name}

    with pytest.raises(Exception, match="ambiguous"):
        RunPodAttemptRunner(
            control=control,
            remote=FakeRemote(r4_certified=False),
            launch_watchdog=arm,
        ).execute(spec)

    assert armed_after_calls == [1]
    assert [call[0] for call in transport.calls] == ["GET", "POST", "GET", "GET", "GET"]


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

    assert transport.calls == [("GET", "/pods", None)]
    assert remote.calls == []
    assert (
        json.loads((attempt_dir / "runpod_operation.json").read_text("utf-8"))["state"]
        == "intent"
    )

    resume_transport = FakeTransport([[{"id": "unexpected-pod"}]])
    with pytest.raises(RuntimeError, match="zero active Pods"):
        RunPodAttemptRunner(
            control=RunPodControlPlane(
                resume_transport, attempt_dir / "runpod_operation.json"
            ),
            remote=remote,
            launch_watchdog=lambda *_args: pytest.fail("must not arm watchdog"),
        ).execute(spec)
    assert resume_transport.calls == [("GET", "/pods", None)]


def test_attempt_preserves_primary_and_deletion_failures(tmp_path: Path) -> None:
    class FailingRemote(FakeRemote):
        def bootstrap(self, pod_id: str, _spec: Any) -> None:
            self.calls.append(f"bootstrap:{pod_id}")
            raise RuntimeError("bootstrap exploded")

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
        commit_sha="e" * 40,
        target_spend_usd=1.5,
        settled_spend_usd=0.0,
    )
    active = [{"id": "pod-1", "name": spec.pod_name}]
    transport = FakeTransport(
        [
            [],
            {"id": "pod-1", "name": spec.pod_name, "costPerHr": 0.44},
            active,
            TimeoutError("delete one"),
            active,
            TimeoutError("delete two"),
            active,
            TimeoutError("delete three"),
            active,
        ]
    )

    with pytest.raises(RunPodAttemptCleanupError) as raised:
        RunPodAttemptRunner(
            control=RunPodControlPlane(
                transport,
                attempt_dir / "runpod_operation.json",
                sleep=lambda _seconds: None,
            ),
            remote=FailingRemote(r4_certified=False),
            launch_watchdog=lambda *_args: {
                "state": "armed",
                "pod_name": spec.pod_name,
            },
        ).execute(spec)

    assert "bootstrap exploded" in str(raised.value)
    assert "still reports Pod" in str(raised.value)


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


def test_attempt_refuses_concurrent_execution_lease(tmp_path: Path) -> None:
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
        commit_sha="a" * 40,
        target_spend_usd=1.5,
        settled_spend_usd=0.0,
    )
    (attempt_dir / "execution.lock").write_text("other-agent", encoding="utf-8")
    transport = FakeTransport([])

    with pytest.raises(RuntimeError, match="already executing"):
        RunPodAttemptRunner(
            control=RunPodControlPlane(
                transport, attempt_dir / "runpod_operation.json"
            ),
            remote=FakeRemote(r4_certified=False),
            launch_watchdog=lambda *_args: pytest.fail("must not arm watchdog"),
        ).execute(spec)

    assert transport.calls == []


def test_attempt_resume_preserves_original_deadline_and_refuses_late_work(
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
            [],
        ]
    )
    control = RunPodControlPlane(transport, attempt_dir / "runpod_operation.json")
    control.create_pod(
        spec.create_request,
        budget=spec.budget,
        arm_watchdog=lambda: {"state": "armed"},
        operation_sha256=spec.attempt_sha256,
    )
    journal_path = attempt_dir / "runpod_operation.json"
    journal = json.loads(journal_path.read_text("utf-8"))
    journal["receipt"]["recorded_at_unix"] = 1000.0
    journal_path.write_text(json.dumps(journal), encoding="utf-8")

    remote = FakeRemote(r4_certified=True)
    with pytest.raises(TimeoutError, match="full R4 and cleanup budget"):
        RunPodAttemptRunner(
            control=control,
            remote=remote,
            launch_watchdog=lambda _spec, _receipt: {
                "state": "armed",
                "pod_name": spec.pod_name,
            },
            clock=lambda: 2150.0,
        ).execute(spec)

    assert remote.calls == ["download:pod-1"]
    assert [method for method, _path, _body in transport.calls].count("POST") == 1
    assert all("/billing/pods" not in path for _method, path, _body in transport.calls)


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
    keyscan_attempts = 0
    probe_attempts = 0

    def run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        nonlocal keyscan_attempts, probe_attempts
        calls.append(command)
        if command[0] == "ssh-keyscan":
            keyscan_attempts += 1
            if keyscan_attempts == 1:
                raise subprocess.TimeoutExpired(command, timeout=5)
            return subprocess.CompletedProcess(
                command, 0, "[192.0.2.10]:12345 ssh-ed25519 AAAATEST\n", ""
            )
        if command[0] == "ssh" and command[-1] == "true":
            probe_attempts += 1
            if probe_attempts == 1:
                raise subprocess.TimeoutExpired(command, timeout=5)
        if command[0] == "scp":
            Path(command[-1]).write_bytes(b"evidence")
        if command[0] == "ssh" and "PTE_EVIDENCE_BYTES" in command[-1]:
            return subprocess.CompletedProcess(
                command,
                0,
                f"PTE_EVIDENCE_BYTES={len(b'evidence')}\n",
                "",
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    executor = SSHRunPodRemoteExecutor(
        control=Control(),
        ssh_private_key=key_path,
        command_runner=run,
        sleep=lambda _seconds: None,
    )
    executor.bootstrap("pod-1", spec)
    executor.run_grpo("pod-1", spec)
    executor.download_evidence("pod-1", spec)

    assert [call[0] for call in calls[:2]] == ["ssh-keyscan", "ssh-keyscan"]
    assert len(
        [call for call in calls if call[0] == "ssh" and call[-1] == "true"]
    ) == 2
    remote_ssh_commands = [
        call for call in calls if call[0] == "ssh" and call[-1] != "true"
    ]
    ssh_command = remote_ssh_commands[0]
    assert ssh_command[0] == "ssh"
    assert "BatchMode=yes" in ssh_command
    assert "PasswordAuthentication=no" in ssh_command
    assert "StrictHostKeyChecking=yes" in ssh_command
    remote_script = ssh_command[-1]
    assert spec.commit_sha in remote_script
    assert "git checkout --detach" in remote_script
    assert "runpod_preflight.py --constraints-only" in remote_script
    assert "pip install --no-deps" in remote_script
    grpo_script = remote_ssh_commands[1][-1]
    assert "if [ -f runs/gsm8k-runpod-smoke/manifest.json ]" not in grpo_script
    assert "pte artifacts validate --run runs/gsm8k-runpod-smoke" in grpo_script
    scp_command = calls[-1]
    assert scp_command[0] == "scp"
    assert "-P" in scp_command
    evidence_receipt = json.loads(
        (attempt_dir / "evidence_download.json").read_text("utf-8")
    )
    assert evidence_receipt["byte_count"] == len(b"evidence")
    assert evidence_receipt["sha256"].startswith("sha256:")


def test_ssh_remote_resume_rejects_changed_endpoint_without_repinning(
    tmp_path: Path,
) -> None:
    class Control:
        @staticmethod
        def get_pod(_pod_id: str) -> dict[str, Any]:
            return {
                "id": "pod-1",
                "publicIp": "192.0.2.11",
                "portMappings": {"22": 12346},
            }

    plan_path = tmp_path / "runpod_plan.json"
    attempt_dir = tmp_path / "attempt"
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
        commit_sha="e" * 40,
        target_spend_usd=1.5,
        settled_spend_usd=0.0,
    )
    known_hosts = attempt_dir / "known_hosts"
    known_hosts.write_text(
        "[192.0.2.10]:12345 ssh-ed25519 AAAATEST\n",
        encoding="utf-8",
    )
    original_known_hosts = known_hosts.read_bytes()
    (attempt_dir / "ssh_connection.json").write_text(
        json.dumps(
            {
                "pod_id": "pod-1",
                "host": "192.0.2.10",
                "port": 12345,
                "known_hosts_sha256": "sha256:"
                + hashlib.sha256(original_known_hosts).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    commands: list[list[str]] = []

    with pytest.raises(ValueError, match="endpoint changed"):
        SSHRunPodRemoteExecutor(
            control=Control(),
            ssh_private_key=key_path,
            command_runner=lambda command, **_kwargs: commands.append(command),
            sleep=lambda _seconds: None,
        ).bootstrap("pod-1", spec)

    assert commands == []
    assert known_hosts.read_bytes() == original_known_hosts


def test_ssh_bootstrap_uses_one_aggregate_deadline(tmp_path: Path) -> None:
    class Control:
        @staticmethod
        def get_pod(_pod_id: str) -> dict[str, Any]:
            return {
                "id": "pod-1",
                "publicIp": "192.0.2.10",
                "portMappings": {"22": 12345},
            }

    key_path = tmp_path / "id_ed25519"
    key_path.write_text("fixture", encoding="utf-8")
    spec = RunPodAttemptSpec(
        attempt_dir=str(tmp_path),
        plan_sha256="sha256:" + "a" * 64,
        repo_url="https://github.com/shannan-liu1/post-train-engine.git",
        commit_sha="a" * 40,
        pod_name="pte-r4-aaaaaaaaaaaa",
        create_request={},
        budget=RunPodBudget(target_spend_usd=1.5, settled_spend_usd=0.0),
    )
    times = iter((100.0, 100.0, 100.0, 101.0, 102.0, 130.0))
    calls: list[tuple[list[str], float]] = []

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((command, float(kwargs["timeout"])))
        stdout = (
            "[192.0.2.10]:12345 ssh-ed25519 AAAATEST\n"
            if command[0] == "ssh-keyscan"
            else ""
        )
        return subprocess.CompletedProcess(command, 0, stdout, "")

    executor = SSHRunPodRemoteExecutor(
        control=Control(),
        ssh_private_key=key_path,
        command_runner=run,
        sleep=lambda _seconds: None,
        monotonic=lambda: next(times, 130.0),
    )

    executor.bootstrap("pod-1", spec)

    bootstrap_timeout = calls[-1][1]
    assert calls[-1][0][0] == "ssh"
    assert calls[-1][0][-1] != "true"
    assert bootstrap_timeout == 450.0


def test_r4_command_and_executor_reject_failed_canonical_run(tmp_path: Path) -> None:
    key_path = tmp_path / "id_ed25519"
    key_path.write_text("fixture", encoding="utf-8")
    executor = SSHRunPodRemoteExecutor(
        control=object(),
        ssh_private_key=key_path,
        command_runner=lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 0, "PTE_R4_CERTIFIED=1\n", ""
        ),
    )
    known_hosts = tmp_path / "known_hosts"
    known_hosts.write_text("fixture", encoding="utf-8")
    executor._connections["pod-1"] = ("192.0.2.10", 12345, known_hosts)
    spec = RunPodAttemptSpec(
        attempt_dir=str(tmp_path),
        plan_sha256="sha256:" + "a" * 64,
        repo_url="https://github.com/shannan-liu1/post-train-engine.git",
        commit_sha="a" * 40,
        pod_name="pte-r4-aaaaaaaaaaaa",
        create_request={},
        budget=RunPodBudget(target_spend_usd=1.5, settled_spend_usd=0.0),
    )
    commands: list[list[str]] = []
    executor.command_runner = lambda command, **_kwargs: (
        commands.append(command)
        or subprocess.CompletedProcess(command, 17, "PTE_R4_CERTIFIED=1\n", "")
    )

    with pytest.raises(RuntimeError, match="without certifying evidence"):
        executor.run_r4("pod-1", spec)
    assert 'if [ "$r4_rc" -ne 0 ]; then exit "$r4_rc"; fi' in commands[0][-1]


def test_load_attempt_rejects_attempt_json_changed_after_review(tmp_path: Path) -> None:
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
    prepare_runpod_attempt(
        plan_path=plan_path,
        attempt_dir=attempt_dir,
        repo_url="https://github.com/shannan-liu1/post-train-engine.git",
        commit_sha="a" * 40,
        target_spend_usd=1.5,
        settled_spend_usd=0.0,
    )
    attempt_path = attempt_dir / "attempt.json"
    raw = json.loads(attempt_path.read_text(encoding="utf-8"))
    raw["minimum_grpo_remaining_seconds"] += 1
    attempt_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="changed after review"):
        load_runpod_attempt(attempt_path)


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
    assert prepared["schema_version"] == "runpod_attempt_v2"
    assert prepared["plan_sha256"].startswith("sha256:")
    assert (attempt_dir / "plan.json").is_file()
    assert prepared["minimum_grpo_remaining_seconds"] >= 540
    assert prepared["create_request"]["gpuCount"] == 2
    assert prepared["create_request"]["volumeInGb"] == 0


def test_plan_rejects_fields_the_attempt_cannot_execute(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "runpod_plan.json"
    with pytest.raises(ValueError, match="rejects overrides: setup_commands"):
        write_runpod_plan(
            config_path="configs/gsm8k_runpod_smoke.yaml",
            out_path=plan_path,
            image=None,
            gpu_type=None,
            command=R4_COMMAND,
            setup_commands=("untracked setup",),
            dry_run=True,
        )


def test_prepare_preserves_reviewed_remote_workdir(tmp_path: Path) -> None:
    plan_path = tmp_path / "runpod_plan.json"
    write_runpod_plan(
        config_path="configs/gsm8k_runpod_smoke.yaml",
        out_path=plan_path,
        image=None,
        gpu_type=None,
        command=R4_COMMAND,
        remote_workdir="/workspace/reviewed-pte",
        dry_run=True,
    )

    spec = prepare_runpod_attempt(
        plan_path=plan_path,
        attempt_dir=tmp_path / "attempt",
        repo_url="https://github.com/shannan-liu1/post-train-engine.git",
        commit_sha="f" * 40,
        target_spend_usd=1.5,
        settled_spend_usd=0.0,
    )

    assert spec.remote_workdir == "/workspace/reviewed-pte"


def test_attempt_refuses_r4_when_bootstrap_consumes_cleanup_reserve(
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
        commit_sha="1" * 40,
        target_spend_usd=1.5,
        settled_spend_usd=0.0,
    )
    transport = FakeTransport(
        [
            [],
            {"id": "pod-1", "name": spec.pod_name, "costPerHr": 0.44},
            [{"id": "pod-1", "name": spec.pod_name}],
            {},
            [],
            [],
        ]
    )
    now = [1000.0]

    class SlowBootstrap(FakeRemote):
        def bootstrap(self, pod_id: str, remote_spec: Any) -> None:
            super().bootstrap(pod_id, remote_spec)
            now[0] = 1601.0

    remote = SlowBootstrap(r4_certified=True)
    with pytest.raises(TimeoutError, match="R4 and cleanup reserve"):
        RunPodAttemptRunner(
            control=RunPodControlPlane(
                transport,
                attempt_dir / "runpod_operation.json",
                clock=lambda: 1000.0,
            ),
            remote=remote,
            launch_watchdog=lambda _spec, _receipt: {
                "state": "armed",
                "pod_name": spec.pod_name,
            },
            clock=lambda: now[0],
        ).execute(spec)

    assert remote.calls == ["bootstrap:pod-1", "download:pod-1"]
    assert ("DELETE", "/pods/pod-1", None) in transport.calls


def test_prepare_refuses_to_overwrite_a_different_attempt(tmp_path: Path) -> None:
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
    first = prepare_runpod_attempt(
        plan_path=plan_path,
        attempt_dir=attempt_dir,
        repo_url="https://github.com/shannan-liu1/post-train-engine.git",
        commit_sha="b" * 40,
        target_spend_usd=1.5,
        settled_spend_usd=0.0,
    )

    with pytest.raises(ValueError, match="different prepared attempt"):
        prepare_runpod_attempt(
            plan_path=plan_path,
            attempt_dir=attempt_dir,
            repo_url=first.repo_url,
            commit_sha="c" * 40,
            target_spend_usd=1.5,
            settled_spend_usd=0.0,
        )

    assert (
        json.loads((attempt_dir / "attempt.json").read_text("utf-8"))["commit_sha"]
        == "b" * 40
    )


def test_source_preflight_requires_clean_local_and_remote_main_identity(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "runpod_plan.json"
    write_runpod_plan(
        config_path="configs/gsm8k_runpod_smoke.yaml",
        out_path=plan_path,
        image=None,
        gpu_type=None,
        command=R4_COMMAND,
        dry_run=True,
    )
    sha = "d" * 40
    spec = prepare_runpod_attempt(
        plan_path=plan_path,
        attempt_dir=tmp_path / "attempt",
        repo_url="https://github.com/shannan-liu1/post-train-engine.git",
        commit_sha=sha,
        target_spend_usd=1.5,
        settled_spend_usd=0.0,
    )
    responses = iter(
        [
            subprocess.CompletedProcess([], 0, sha + "\n", ""),
            subprocess.CompletedProcess([], 0, "", ""),
            subprocess.CompletedProcess([], 0, f"{sha}\trefs/heads/main\n", ""),
        ]
    )

    receipt = verify_runpod_attempt_source(
        spec,
        repo_root=Path.cwd(),
        command_runner=lambda *_args, **_kwargs: next(responses),
    )

    assert receipt["state"] == "verified"
    assert receipt["commit_sha"] == sha


def test_source_preflight_rejects_changed_prepared_plan(tmp_path: Path) -> None:
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
        attempt_dir=tmp_path / "attempt",
        repo_url="https://github.com/shannan-liu1/post-train-engine.git",
        commit_sha="d" * 40,
        target_spend_usd=1.5,
        settled_spend_usd=0.0,
    )
    frozen_plan = Path(spec.attempt_dir) / "plan.json"
    raw = json.loads(frozen_plan.read_text(encoding="utf-8"))
    raw["job"]["remote_workdir"] = "/workspace/tampered"
    frozen_plan.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="plan changed"):
        verify_runpod_attempt_source(spec)


def test_source_preflight_rejects_changed_config_bytes(tmp_path: Path) -> None:
    config_path = tmp_path / "configs" / "gsm8k_runpod_smoke.yaml"
    config_path.parent.mkdir()
    config_path.write_bytes(Path("configs/gsm8k_runpod_smoke.yaml").read_bytes())
    plan_path = tmp_path / "runpod_plan.json"
    plan = write_runpod_plan(
        config_path="configs/gsm8k_runpod_smoke.yaml",
        out_path=plan_path,
        image=None,
        gpu_type=None,
        command=R4_COMMAND,
        dry_run=True,
    )
    plan["config"]["sha256"] = (
        "sha256:" + hashlib.sha256(config_path.read_bytes()).hexdigest()
    )
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    spec = prepare_runpod_attempt(
        plan_path=plan_path,
        attempt_dir=tmp_path / "attempt",
        repo_url="https://github.com/shannan-liu1/post-train-engine.git",
        commit_sha="d" * 40,
        target_spend_usd=1.5,
        settled_spend_usd=0.0,
    )
    config_path.write_text("tampered", encoding="utf-8")

    with pytest.raises(ValueError, match="config changed"):
        verify_runpod_attempt_source(spec, repo_root=tmp_path)


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
        json.dumps({"state": "created", "receipt": {"pod_id": "pod-1"}}),
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
