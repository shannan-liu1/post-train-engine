"""One bounded, durable RunPod infrastructure attempt."""

from __future__ import annotations

import json
import hashlib
import ipaddress
import os
import re
import shlex
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from post_train_engine.runpod import build_runpod_create_request
from post_train_engine.runpod_control_plane import (
    PodBillingReceipt,
    PodCreateReceipt,
    RunPodBudget,
    RunPodControlPlane,
)

_FROZEN_FORBID = ConfigDict(frozen=True, extra="forbid")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_PUBLIC_GITHUB_RE = re.compile(
    r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?"
)
CANONICAL_R4_COMMAND = (
    "accelerate launch --num_processes 2 -m post_train_engine.cli "
    "run --config configs/gsm8k_runpod_r4.yaml --no-env"
)
BOOTSTRAP_TIMEOUT_SECONDS = 480
R4_TIMEOUT_SECONDS = 360
GRPO_TIMEOUT_SECONDS = 300
EVIDENCE_ARCHIVE_TIMEOUT_SECONDS = 60
EVIDENCE_DOWNLOAD_TIMEOUT_SECONDS = 120
TEARDOWN_RESERVE_SECONDS = 60
MAX_EVIDENCE_ARCHIVE_BYTES = 256 * 1024 * 1024
R4_DOWNSTREAM_RESERVE_SECONDS = (
    R4_TIMEOUT_SECONDS
    + EVIDENCE_ARCHIVE_TIMEOUT_SECONDS
    + EVIDENCE_DOWNLOAD_TIMEOUT_SECONDS
    + TEARDOWN_RESERVE_SECONDS
)
R4_REQUIRED_REMAINING_SECONDS = (
    BOOTSTRAP_TIMEOUT_SECONDS + R4_DOWNSTREAM_RESERVE_SECONDS
)
GRPO_DOWNSTREAM_RESERVE_SECONDS = (
    GRPO_TIMEOUT_SECONDS
    + EVIDENCE_ARCHIVE_TIMEOUT_SECONDS
    + EVIDENCE_DOWNLOAD_TIMEOUT_SECONDS
    + TEARDOWN_RESERVE_SECONDS
)


class RunPodAttemptSpec(BaseModel):
    """Immutable local contract for one paid infrastructure attempt."""

    model_config = _FROZEN_FORBID

    schema_version: Literal["runpod_attempt_v2"] = "runpod_attempt_v2"
    attempt_dir: str = Field(..., min_length=1)
    plan_sha256: str = Field(..., pattern=r"^sha256:[0-9a-f]{64}$")
    repo_url: str = Field(..., min_length=1)
    commit_sha: str = Field(..., min_length=40, max_length=40)
    pod_name: str = Field(..., min_length=1, max_length=191)
    create_request: dict[str, Any]
    budget: RunPodBudget
    remote_workdir: str = "/workspace/post-train-engine"
    r4_config: str = "configs/gsm8k_runpod_r4.yaml"
    grpo_config: str = "configs/gsm8k_runpod_smoke.yaml"
    minimum_grpo_remaining_seconds: int = Field(
        default=GRPO_DOWNSTREAM_RESERVE_SECONDS, gt=0
    )

    @model_validator(mode="after")
    def _validate_source_and_time(self) -> RunPodAttemptSpec:
        if _COMMIT_RE.fullmatch(self.commit_sha) is None:
            raise ValueError("commit_sha must be a lowercase full Git SHA")
        if _PUBLIC_GITHUB_RE.fullmatch(self.repo_url) is None:
            raise ValueError("repo_url must be a public HTTPS GitHub repository")
        if not self.remote_workdir.startswith("/") or self.remote_workdir == "/":
            raise ValueError("remote_workdir must be a non-root absolute path")
        if self.budget.max_runtime_seconds < R4_REQUIRED_REMAINING_SECONDS:
            raise ValueError(
                "max_runtime_seconds cannot fit bootstrap, R4, evidence, and teardown"
            )
        if self.minimum_grpo_remaining_seconds < GRPO_DOWNSTREAM_RESERVE_SECONDS:
            raise ValueError(
                "minimum_grpo_remaining_seconds cannot undercut the canonical "
                "GRPO and cleanup reserve"
            )
        if self.minimum_grpo_remaining_seconds >= self.budget.max_runtime_seconds:
            raise ValueError("minimum_grpo_remaining_seconds must be below max runtime")
        return self

    @property
    def attempt_sha256(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return "sha256:" + hashlib.sha256(payload).hexdigest()


class RunPodAttemptResult(BaseModel):
    model_config = _FROZEN_FORBID

    state: Literal["billing_pending", "billing_provisional"]
    pod_id: str
    r4_certified: bool
    grpo_ran: bool
    billed_cost_usd: float | None


class RunPodAttemptCleanupError(RuntimeError):
    """Mandatory deletion or evidence cleanup failed after an attempt error."""


class RunPodRemoteExecutor(Protocol):
    def bootstrap(self, pod_id: str, spec: RunPodAttemptSpec) -> None: ...

    def run_r4(self, pod_id: str, spec: RunPodAttemptSpec) -> bool: ...

    def run_grpo(self, pod_id: str, spec: RunPodAttemptSpec) -> None: ...

    def download_evidence(self, pod_id: str, spec: RunPodAttemptSpec) -> None: ...


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


class SSHRunPodRemoteExecutor:
    """Execute the fixed remote sequence over one pinned batch-mode SSH channel."""

    def __init__(
        self,
        *,
        control: RunPodControlPlane,
        ssh_private_key: str | Path,
        command_runner: CommandRunner = subprocess.run,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.control = control
        self.ssh_private_key = Path(ssh_private_key)
        if not self.ssh_private_key.is_file():
            raise ValueError("SSH private key does not exist")
        self.command_runner = command_runner
        self.sleep = sleep
        self.monotonic = monotonic
        self._connections: dict[str, tuple[str, int, Path]] = {}

    def bootstrap(self, pod_id: str, spec: RunPodAttemptSpec) -> None:
        script = _bootstrap_script(spec)
        self._ssh(
            pod_id,
            spec,
            script,
            timeout=BOOTSTRAP_TIMEOUT_SECONDS,
            check=True,
        )

    def run_r4(self, pod_id: str, spec: RunPodAttemptSpec) -> bool:
        workdir = shlex.quote(spec.remote_workdir)
        config = shlex.quote(spec.r4_config)
        script = (
            "set -uo pipefail; "
            f"cd {workdir}; "
            "r4_rc=0; "
            "if [ ! -f runs/gsm8k-runpod-r4/manifest.json ]; then "
            f"accelerate launch --num_processes 2 -m post_train_engine.cli run --config {config} --no-env || r4_rc=$?; "
            "fi; "
            'if [ "$r4_rc" -ne 0 ]; then exit "$r4_rc"; fi; '
            "pte artifacts validate --run runs/gsm8k-runpod-r4 || exit $?; "
            "python -c \"import json; p=json.load(open('runs/gsm8k-runpod-r4/final_report.json')); "
            "print('PTE_R4_CERTIFIED=' + ('1' if p['runtime_certified'] else '0'))\""
        )
        result = self._ssh(
            pod_id,
            spec,
            script,
            timeout=R4_TIMEOUT_SECONDS,
            check=False,
        )
        markers = [
            line.strip()
            for line in result.stdout.splitlines()
            if line.startswith("PTE_R4_CERTIFIED=")
        ]
        if markers == ["PTE_R4_CERTIFIED=0"]:
            return False
        if result.returncode != 0 or markers != ["PTE_R4_CERTIFIED=1"]:
            raise RuntimeError("R4 remote execution ended without certifying evidence")
        return True

    def run_grpo(self, pod_id: str, spec: RunPodAttemptSpec) -> None:
        workdir = shlex.quote(spec.remote_workdir)
        config = shlex.quote(spec.grpo_config)
        script = (
            "set -euo pipefail; "
            f"cd {workdir}; "
            f"accelerate launch --num_processes 2 -m post_train_engine.cli run --config {config} --no-env; "
            "pte artifacts validate --run runs/gsm8k-runpod-smoke"
        )
        self._ssh(pod_id, spec, script, timeout=GRPO_TIMEOUT_SECONDS, check=True)

    def download_evidence(self, pod_id: str, spec: RunPodAttemptSpec) -> None:
        connection = self._connections.get(pod_id)
        if connection is None:
            raise RuntimeError("cannot download evidence before SSH is established")
        host, port, known_hosts = connection
        remote_archive = f"/tmp/pte-evidence-{pod_id}.tar.gz"
        workdir = shlex.quote(spec.remote_workdir)
        archive = shlex.quote(remote_archive)
        script = (
            "set -euo pipefail; "
            f"cd {workdir}; "
            "paths=(); "
            "for path in runs/gsm8k-runpod-r4 runs/gsm8k-runpod-smoke "
            "runs/runpod-preflight artifacts/runpod; do "
            'if [ -e "$path" ]; then paths+=("$path"); fi; done; '
            "if [ ${#paths[@]} -eq 0 ]; then "
            f"tar -czf {archive} --files-from /dev/null; "
            "else "
            f"tar -czf {archive} --exclude='*.safetensors' "
            "--exclude='checkpoint-*' \"${paths[@]}\"; fi; "
            f"bytes=$(stat -c %s {archive}); "
            "printf 'PTE_EVIDENCE_BYTES=%s\\n' \"$bytes\""
        )
        archive_result = self._ssh(
            pod_id,
            spec,
            script,
            timeout=EVIDENCE_ARCHIVE_TIMEOUT_SECONDS,
            check=True,
        )
        byte_markers = [
            line.removeprefix("PTE_EVIDENCE_BYTES=")
            for line in archive_result.stdout.splitlines()
            if line.startswith("PTE_EVIDENCE_BYTES=")
        ]
        if len(byte_markers) != 1 or not byte_markers[0].isdigit():
            raise RuntimeError("RunPod evidence archive omitted its byte count")
        remote_byte_count = int(byte_markers[0])
        if not 0 < remote_byte_count <= MAX_EVIDENCE_ARCHIVE_BYTES:
            raise RuntimeError("RunPod evidence archive exceeds the 256 MiB limit")
        destination = Path(spec.attempt_dir) / "evidence.tar.gz"
        command = [
            "scp",
            *self._ssh_options(port, known_hosts, scp=True),
            f"root@{host}:{remote_archive}",
            str(destination),
        ]
        self._run(command, timeout=EVIDENCE_DOWNLOAD_TIMEOUT_SECONDS, check=True)
        if not destination.is_file() or destination.stat().st_size != remote_byte_count:
            raise RuntimeError(
                "RunPod evidence download did not match the remote byte count"
            )
        _write_json(
            Path(spec.attempt_dir) / "evidence_download.json",
            {
                "pod_id": pod_id,
                "path": destination.name,
                "byte_count": destination.stat().st_size,
                "sha256": _sha256_file(destination),
            },
        )

    def _ssh(
        self,
        pod_id: str,
        spec: RunPodAttemptSpec,
        script: str,
        *,
        timeout: float,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        deadline = self.monotonic() + timeout
        host, port, known_hosts = self._connection(pod_id, spec, deadline=deadline)
        command = [
            "ssh",
            *self._ssh_options(port, known_hosts),
            f"root@{host}",
            "bash",
            "-lc",
            shlex.quote(script),
        ]
        remaining = deadline - self.monotonic()
        if remaining <= 0.0:
            raise TimeoutError("RunPod remote command exhausted its aggregate deadline")
        return self._run(command, timeout=remaining, check=check)

    def _connection(
        self,
        pod_id: str,
        spec: RunPodAttemptSpec,
        *,
        deadline: float,
    ) -> tuple[str, int, Path]:
        if pod_id in self._connections:
            return self._connections[pod_id]
        attempt_dir = Path(spec.attempt_dir)
        known_hosts = attempt_dir / "known_hosts"
        evidence_path = attempt_dir / "ssh_connection.json"
        pinned: dict[str, Any] | None = None
        if evidence_path.is_file():
            raw = json.loads(evidence_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict) or str(raw.get("pod_id")) != pod_id:
                raise ValueError("SSH connection evidence belongs to a different Pod")
            if not known_hosts.is_file():
                raise ValueError("pinned SSH connection is missing known_hosts")
            actual_hash = (
                "sha256:" + hashlib.sha256(known_hosts.read_bytes()).hexdigest()
            )
            if raw.get("known_hosts_sha256") != actual_hash:
                raise ValueError("pinned SSH known_hosts evidence changed")
            pinned = raw
        readiness_deadline = deadline
        while self.monotonic() < readiness_deadline:
            try:
                pod = self.control.get_pod(pod_id)
            except (OSError, TimeoutError):
                self._sleep_until_retry(readiness_deadline)
                continue
            desired_status = str(pod.get("desiredStatus") or "").upper()
            if desired_status in {"EXITED", "TERMINATED"}:
                detail = str(pod.get("lastStatusChange") or "no provider detail")
                raise RuntimeError(
                    f"RunPod entered terminal status {desired_status}: {detail[:500]}"
                )
            host_value = pod.get("publicIp")
            mappings = pod.get("portMappings")
            port_value = mappings.get("22") if isinstance(mappings, dict) else None
            if isinstance(host_value, str) and isinstance(port_value, int):
                host = str(ipaddress.ip_address(host_value))
                if not 0 < port_value < 65536:
                    raise ValueError("RunPod SSH port must be between 1 and 65535")
                if pinned is not None:
                    if pinned.get("host") != host or pinned.get("port") != port_value:
                        raise ValueError(
                            "RunPod SSH endpoint changed after it was pinned"
                        )
                else:
                    try:
                        scan = self._run(
                            ["ssh-keyscan", "-T", "5", "-p", str(port_value), host],
                            timeout=max(
                                0.1,
                                min(10.0, readiness_deadline - self.monotonic()),
                            ),
                            check=False,
                        )
                    except subprocess.TimeoutExpired:
                        self._sleep_until_retry(readiness_deadline)
                        continue
                    if scan.returncode != 0 or not scan.stdout.strip():
                        self._sleep_until_retry(readiness_deadline)
                        continue
                    known_hosts.parent.mkdir(parents=True, exist_ok=True)
                    known_hosts.write_text(scan.stdout, encoding="utf-8")
                try:
                    probe = self._run(
                        [
                            "ssh",
                            *self._ssh_options(port_value, known_hosts),
                            f"root@{host}",
                            "true",
                        ],
                        timeout=max(
                            0.1,
                            min(15.0, readiness_deadline - self.monotonic()),
                        ),
                        check=False,
                    )
                except subprocess.TimeoutExpired:
                    self._sleep_until_retry(readiness_deadline)
                    continue
                if probe.returncode != 0:
                    self._sleep_until_retry(readiness_deadline)
                    continue
                if pinned is None:
                    _write_json(
                        evidence_path,
                        {
                            "pod_id": pod_id,
                            "host": host,
                            "port": port_value,
                            "known_hosts_sha256": "sha256:"
                            + hashlib.sha256(known_hosts.read_bytes()).hexdigest(),
                        },
                    )
                connection = (host, port_value, known_hosts)
                self._connections[pod_id] = connection
                return connection
            self._sleep_until_retry(readiness_deadline)
        raise TimeoutError("RunPod did not expose SSH within the remote phase deadline")

    def _sleep_until_retry(self, deadline: float) -> None:
        remaining = deadline - self.monotonic()
        if remaining > 0.0:
            self.sleep(min(5.0, remaining))

    def _ssh_options(
        self,
        port: int,
        known_hosts: Path,
        *,
        scp: bool = False,
    ) -> list[str]:
        return [
            "-P" if scp else "-p",
            str(port),
            "-i",
            str(self.ssh_private_key),
            "-o",
            "BatchMode=yes",
            "-o",
            "PasswordAuthentication=no",
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            f"UserKnownHostsFile={known_hosts}",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "ConnectionAttempts=1",
        ]

    def _run(
        self,
        command: list[str],
        *,
        timeout: float,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        result = self.command_runner(
            command,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        if check and result.returncode != 0:
            stderr = result.stderr[-2000:].strip()
            raise RuntimeError(
                f"remote command failed with exit {result.returncode}: {stderr}"
            )
        return result


WatchdogLauncher = Callable[
    [RunPodAttemptSpec, PodCreateReceipt | None],
    dict[str, Any],
]


def prepare_runpod_attempt(
    *,
    plan_path: str | Path,
    attempt_dir: str | Path,
    repo_url: str,
    commit_sha: str,
    target_spend_usd: float,
    settled_spend_usd: float,
    max_runtime_seconds: int = 1200,
    reserve_usd: float = 0.15,
    minimum_grpo_remaining_seconds: int = GRPO_DOWNSTREAM_RESERVE_SECONDS,
) -> RunPodAttemptSpec:
    """Freeze one reviewed dry-run plan without contacting RunPod."""

    plan_path = Path(plan_path)
    raw = json.loads(plan_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("RunPod plan must be a JSON object")
    job = raw.get("job")
    if not isinstance(job, dict) or job.get("command") != CANONICAL_R4_COMMAND:
        raise ValueError("RunPod attempt plan must run the canonical R4 command first")
    environment = raw.get("environment")
    if not isinstance(environment, dict):
        raise ValueError("RunPod attempt plan requires an environment mapping")
    for field in ("setup_commands", "env", "secret_env"):
        if environment.get(field):
            raise ValueError(f"RunPod attempt plan contains unsupported {field}")
    remote_workdir = job.get("remote_workdir")
    if not isinstance(remote_workdir, str) or not remote_workdir:
        raise ValueError("RunPod attempt plan requires remote_workdir")
    if job.get("sync_artifacts") is not False:
        raise ValueError("RunPod attempt plan cannot use legacy artifact sync")
    attempt_dir = Path(attempt_dir)
    pod_name = f"pte-r4-{commit_sha[:12]}"
    request = build_runpod_create_request(raw, pod_name=pod_name)
    budget = RunPodBudget(
        target_spend_usd=target_spend_usd,
        settled_spend_usd=settled_spend_usd,
        reserve_usd=reserve_usd,
        max_runtime_seconds=max_runtime_seconds,
    )
    spec = RunPodAttemptSpec(
        attempt_dir=str(attempt_dir),
        plan_sha256=_sha256_json(raw),
        repo_url=repo_url,
        commit_sha=commit_sha,
        pod_name=pod_name,
        create_request=request,
        budget=budget,
        remote_workdir=remote_workdir,
        minimum_grpo_remaining_seconds=minimum_grpo_remaining_seconds,
    )
    attempt_path = attempt_dir / "attempt.json"
    if attempt_path.is_file():
        existing = RunPodAttemptSpec.model_validate_json(
            attempt_path.read_text(encoding="utf-8")
        )
        if existing != spec:
            raise ValueError("attempt directory contains a different prepared attempt")
        persisted_plan = attempt_dir / "plan.json"
        if (
            not persisted_plan.is_file()
            or _sha256_json(json.loads(persisted_plan.read_text(encoding="utf-8")))
            != spec.plan_sha256
        ):
            raise ValueError("attempt directory contains changed plan evidence")
        _require_attempt_identity(existing)
        return existing
    if attempt_dir.exists() and any(attempt_dir.iterdir()):
        raise ValueError("attempt directory is occupied without attempt.json")
    _write_json(attempt_dir / "plan.json", raw)
    _write_json(attempt_path, spec.model_dump(mode="json"))
    _write_json(
        attempt_dir / "attempt_state.json",
        {
            "state": "prepared",
            "attempt": "attempt.json",
            "attempt_sha256": spec.attempt_sha256,
        },
    )
    return spec


def verify_runpod_attempt_source(
    spec: RunPodAttemptSpec,
    *,
    repo_root: str | Path = ".",
    command_runner: CommandRunner = subprocess.run,
) -> dict[str, Any]:
    """Prove reviewed local main equals public remote main before allocation."""

    root = Path(repo_root).resolve()
    attempt_dir = Path(spec.attempt_dir).resolve()
    plan_path = attempt_dir / "plan.json"
    if not plan_path.is_file():
        raise ValueError("prepared RunPod attempt is missing plan.json")
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("prepared RunPod plan is not valid JSON") from exc
    if not isinstance(plan, dict) or _sha256_json(plan) != spec.plan_sha256:
        raise ValueError("prepared RunPod plan changed after review")
    _require_attempt_identity(spec)
    if spec.pod_name != f"pte-r4-{spec.commit_sha[:12]}":
        raise ValueError("prepared RunPod Pod name does not match its commit")
    if build_runpod_create_request(plan, pod_name=spec.pod_name) != spec.create_request:
        raise ValueError("prepared RunPod create request does not match its plan")
    config_ref = plan.get("config")
    if not isinstance(config_ref, dict):
        raise ValueError("prepared RunPod plan omitted its config evidence")
    config_path = (root / str(config_ref.get("path") or "")).resolve()
    try:
        config_path.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            "prepared RunPod config must remain inside the repository"
        ) from exc
    if not config_path.is_file() or _sha256_file(config_path) != config_ref.get(
        "sha256"
    ):
        raise ValueError("prepared RunPod config changed after plan creation")

    def run(command: list[str]) -> subprocess.CompletedProcess[str]:
        result = command_runner(
            command,
            cwd=root,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"source preflight failed: {' '.join(command[:2])}: "
                + result.stderr[-1000:].strip()
            )
        return result

    head = run(["git", "rev-parse", "HEAD"]).stdout.strip()
    if head != spec.commit_sha:
        raise ValueError("local HEAD does not match the prepared attempt commit")
    dirty = run(["git", "status", "--porcelain"]).stdout.strip()
    if dirty:
        raise ValueError("local worktree must be clean before paid execution")
    remote = run(["git", "ls-remote", spec.repo_url, "refs/heads/main"]).stdout.strip()
    expected = f"{spec.commit_sha}\trefs/heads/main"
    if remote != expected:
        raise ValueError(
            "public remote main does not match the prepared attempt commit"
        )
    receipt = {
        "state": "verified",
        "commit_sha": spec.commit_sha,
        "remote_ref": "refs/heads/main",
        "repo_url": spec.repo_url,
        "recorded_at": datetime.now(UTC).isoformat(),
    }
    _write_json(Path(spec.attempt_dir) / "source_preflight.json", receipt)
    return receipt


class RunPodAttemptRunner:
    """Execute one fail-closed attempt around the canonical remote ``pte run``."""

    def __init__(
        self,
        *,
        control: RunPodControlPlane,
        remote: RunPodRemoteExecutor,
        launch_watchdog: WatchdogLauncher,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.control = control
        self.remote = remote
        self.launch_watchdog = launch_watchdog
        self.clock = clock

    def execute(self, spec: RunPodAttemptSpec) -> RunPodAttemptResult:
        lease = Path(spec.attempt_dir) / "execution.lock"
        lease.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(
                lease,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
        except FileExistsError as exc:
            raise RuntimeError("RunPod attempt is already executing") from exc
        try:
            os.write(descriptor, str(os.getpid()).encode("ascii"))
            return self._execute_locked(spec)
        finally:
            os.close(descriptor)
            lease.unlink(missing_ok=True)

    def _execute_locked(self, spec: RunPodAttemptSpec) -> RunPodAttemptResult:
        receipt: PodCreateReceipt | None = None
        r4_certified = False
        grpo_ran = False
        primary_error: BaseException | None = None
        evidence_error: BaseException | None = None
        deletion_error: BaseException | None = None

        try:
            if self.control.operation_state() in {None, "intent"}:
                active_pods = self.control.list_pods()
                if active_pods:
                    raise RuntimeError("RunPod attempt requires zero active Pods")
            self._state(spec, "creating")
            receipt = self.control.create_pod(
                spec.create_request,
                budget=spec.budget,
                arm_watchdog=lambda: self.launch_watchdog(spec, None),
                operation_sha256=spec.attempt_sha256,
            )
            self.control.require_only_created_pod(receipt)
            self._state(spec, "created", pod_id=receipt.pod_id)
            self._state(spec, "watchdog_armed", pod_id=receipt.pod_id)
            remaining = (
                receipt.recorded_at_unix + receipt.hard_deadline_seconds - self.clock()
            )
            if remaining < R4_REQUIRED_REMAINING_SECONDS:
                raise TimeoutError(
                    "RunPod attempt lacks the full R4 and cleanup budget"
                )
            self.remote.bootstrap(receipt.pod_id, spec)
            self._state(spec, "preflight_passed", pod_id=receipt.pod_id)
            remaining = (
                receipt.recorded_at_unix + receipt.hard_deadline_seconds - self.clock()
            )
            if remaining < R4_DOWNSTREAM_RESERVE_SECONDS:
                raise TimeoutError("RunPod attempt lacks the R4 and cleanup reserve")
            r4_certified = self.remote.run_r4(receipt.pod_id, spec)
            self._state(
                spec,
                "r4_certified" if r4_certified else "r4_rejected",
                pod_id=receipt.pod_id,
            )
            remaining = (
                receipt.recorded_at_unix + receipt.hard_deadline_seconds - self.clock()
            )
            if r4_certified and remaining >= spec.minimum_grpo_remaining_seconds:
                self.remote.run_grpo(receipt.pod_id, spec)
                grpo_ran = True
                self._state(spec, "grpo_finished", pod_id=receipt.pod_id)
        except BaseException as exc:
            primary_error = exc
        finally:
            if receipt is not None:
                try:
                    self.remote.download_evidence(receipt.pod_id, spec)
                except BaseException as exc:
                    evidence_error = exc
                try:
                    self.control.delete_pod_and_verify(receipt.pod_id)
                except BaseException as exc:
                    deletion_error = exc

        cleanup_error = deletion_error or evidence_error
        if primary_error is not None or cleanup_error is not None:
            self._state(
                spec,
                "failed",
                **(
                    {"primary_error_type": type(primary_error).__name__}
                    if primary_error is not None
                    else {}
                ),
                **(
                    {"evidence_error_type": type(evidence_error).__name__}
                    if evidence_error is not None
                    else {}
                ),
                **(
                    {"deletion_error_type": type(deletion_error).__name__}
                    if deletion_error is not None
                    else {}
                ),
            )
        if primary_error is not None and cleanup_error is not None:
            details = [f"primary: {primary_error}"]
            if evidence_error is not None:
                details.append(f"evidence: {evidence_error}")
            if deletion_error is not None:
                details.append(f"deletion: {deletion_error}")
            raise RunPodAttemptCleanupError(
                "RunPod attempt and mandatory cleanup failed; " + "; ".join(details)
            ) from cleanup_error
        if primary_error is not None:
            raise primary_error
        if cleanup_error is not None:
            raise cleanup_error
        if receipt is None:
            raise RuntimeError("RunPod attempt ended without a create receipt")

        start_time = datetime.fromtimestamp(receipt.recorded_at_unix, UTC).isoformat()
        billing = self.control.fetch_billing(receipt.pod_id, start_time=start_time)
        state = (
            "billing_pending"
            if billing.settlement_state == "pending"
            else "billing_provisional"
        )
        result = RunPodAttemptResult(
            state=state,
            pod_id=receipt.pod_id,
            r4_certified=r4_certified,
            grpo_ran=grpo_ran,
            billed_cost_usd=billing.amount_usd,
        )
        result_evidence = result.model_dump(mode="json")
        result_evidence.pop("state")
        self._state(spec, state, **result_evidence)
        return result

    @staticmethod
    def _state(spec: RunPodAttemptSpec, state: str, **evidence: Any) -> None:
        _write_json(
            Path(spec.attempt_dir) / "attempt_state.json",
            {"state": state, "attempt_sha256": spec.attempt_sha256, **evidence},
        )


def load_runpod_attempt(path: str | Path) -> RunPodAttemptSpec:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    spec = RunPodAttemptSpec.model_validate(raw)
    _require_attempt_identity(spec)
    return spec


def _require_attempt_identity(spec: RunPodAttemptSpec) -> None:
    state_path = Path(spec.attempt_dir) / "attempt_state.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError(
            "prepared RunPod attempt is missing identity evidence"
        ) from exc
    if (
        not isinstance(state, dict)
        or state.get("attempt_sha256") != spec.attempt_sha256
    ):
        raise ValueError("prepared RunPod attempt changed after review")


def settle_runpod_billing(
    *,
    control: RunPodControlPlane,
    pod_id: str,
    start_time: str,
    out: str | Path,
    end_time: str | None = None,
    final: bool = False,
) -> PodBillingReceipt:
    """Persist one provisional or matching final provider billing observation."""

    receipt = control.fetch_billing(
        pod_id,
        start_time=start_time,
        end_time=end_time,
        final=final,
    )
    _write_json(Path(out), receipt.model_dump(mode="json"))
    return receipt


def _bootstrap_script(spec: RunPodAttemptSpec) -> str:
    workdir = shlex.quote(spec.remote_workdir)
    repo_url = shlex.quote(spec.repo_url)
    commit_sha = shlex.quote(spec.commit_sha)
    return (
        "set -euo pipefail; "
        f"if [ -d {workdir}/.git ]; then "
        f'cd {workdir}; test "$(git remote get-url origin)" = {repo_url}; '
        "else "
        f"git clone {repo_url} {workdir}; cd {workdir}; "
        "fi; "
        f"git checkout --detach {commit_sha}; "
        f'test "$(git rev-parse HEAD)" = {commit_sha}; '
        'test -z "$(git status --porcelain)"; '
        "mkdir -p artifacts/runpod; "
        "python src/post_train_engine/runpod_preflight.py --constraints-only; "
        'python -c "import torch; '
        "assert torch.cuda.is_available(), 'CUDA unavailable'; "
        "assert torch.cuda.device_count() == 2, 'expected exactly two GPUs'; "
        "names=[torch.cuda.get_device_name(i) for i in range(2)]; "
        "assert all('A40' in name for name in names), f'unexpected GPUs: {names}'; "
        'print(torch.__version__, torch.version.cuda, names)" '
        "> artifacts/runpod/gpu_before_install.txt; "
        'python -c "import torch; print(torch.__version__, torch.version.cuda)" '
        "> artifacts/runpod/torch_before.txt; "
        "python -m pip install --require-hashes -r requirements/runpod.txt; "
        "python -m pip install --no-deps -e '.[rlvr]'; "
        'python -c "import torch; print(torch.__version__, torch.version.cuda)" '
        "> artifacts/runpod/torch_after.txt; "
        "cmp artifacts/runpod/torch_before.txt artifacts/runpod/torch_after.txt; "
        "python scripts/check_cuda_stack.py --config configs/gsm8k_runpod_smoke.yaml; "
        "python scripts/runpod_preflight.py --out artifacts/runpod/paid_preflight.json "
        "--total-timeout-sec 300; "
        "accelerate env > artifacts/runpod/accelerate_env.txt; "
        "accelerate launch --num_processes 2 scripts/check_distributed_cuda.py "
        "> artifacts/runpod/distributed_cuda.txt"
    )


def _write_json(path: Path, body: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(body, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _sha256_json(body: dict[str, Any]) -> str:
    payload = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


__all__ = [
    "BOOTSTRAP_TIMEOUT_SECONDS",
    "CANONICAL_R4_COMMAND",
    "EVIDENCE_ARCHIVE_TIMEOUT_SECONDS",
    "EVIDENCE_DOWNLOAD_TIMEOUT_SECONDS",
    "GRPO_DOWNSTREAM_RESERVE_SECONDS",
    "GRPO_TIMEOUT_SECONDS",
    "MAX_EVIDENCE_ARCHIVE_BYTES",
    "R4_DOWNSTREAM_RESERVE_SECONDS",
    "R4_REQUIRED_REMAINING_SECONDS",
    "R4_TIMEOUT_SECONDS",
    "RunPodAttemptResult",
    "RunPodAttemptCleanupError",
    "RunPodAttemptRunner",
    "RunPodAttemptSpec",
    "RunPodRemoteExecutor",
    "SSHRunPodRemoteExecutor",
    "load_runpod_attempt",
    "prepare_runpod_attempt",
    "settle_runpod_billing",
    "TEARDOWN_RESERVE_SECONDS",
    "verify_runpod_attempt_source",
]
