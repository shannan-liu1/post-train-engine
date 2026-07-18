"""One bounded, durable RunPod infrastructure attempt."""

from __future__ import annotations

import json
import hashlib
import ipaddress
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


class RunPodAttemptSpec(BaseModel):
    """Immutable local contract for one paid infrastructure attempt."""

    model_config = _FROZEN_FORBID

    schema_version: Literal["runpod_attempt_v1"] = "runpod_attempt_v1"
    attempt_dir: str = Field(..., min_length=1)
    repo_url: str = Field(..., min_length=1)
    commit_sha: str = Field(..., min_length=40, max_length=40)
    pod_name: str = Field(..., min_length=1, max_length=191)
    create_request: dict[str, Any]
    budget: RunPodBudget
    remote_workdir: str = "/workspace/post-train-engine"
    r4_config: str = "configs/gsm8k_runpod_r4.yaml"
    grpo_config: str = "configs/gsm8k_runpod_smoke.yaml"
    minimum_grpo_remaining_seconds: int = Field(default=420, gt=0)

    @model_validator(mode="after")
    def _validate_source_and_time(self) -> RunPodAttemptSpec:
        if _COMMIT_RE.fullmatch(self.commit_sha) is None:
            raise ValueError("commit_sha must be a lowercase full Git SHA")
        if _PUBLIC_GITHUB_RE.fullmatch(self.repo_url) is None:
            raise ValueError("repo_url must be a public HTTPS GitHub repository")
        if not self.remote_workdir.startswith("/") or self.remote_workdir == "/":
            raise ValueError("remote_workdir must be a non-root absolute path")
        if self.minimum_grpo_remaining_seconds >= self.budget.max_runtime_seconds:
            raise ValueError(
                "minimum_grpo_remaining_seconds must be below max runtime"
            )
        return self


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
    ) -> None:
        self.control = control
        self.ssh_private_key = Path(ssh_private_key)
        if not self.ssh_private_key.is_file():
            raise ValueError("SSH private key does not exist")
        self.command_runner = command_runner
        self.sleep = sleep
        self._connections: dict[str, tuple[str, int, Path]] = {}

    def bootstrap(self, pod_id: str, spec: RunPodAttemptSpec) -> None:
        script = _bootstrap_script(spec)
        self._ssh(pod_id, spec, script, timeout=480, check=True)

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
            "pte artifacts validate --run runs/gsm8k-runpod-r4 || exit $?; "
            "python -c \"import json; p=json.load(open('runs/gsm8k-runpod-r4/final_report.json')); "
            "print('PTE_R4_CERTIFIED=' + ('1' if p['runtime_certified'] else '0'))\""
        )
        result = self._ssh(pod_id, spec, script, timeout=360, check=False)
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
            "if [ -f runs/gsm8k-runpod-smoke/manifest.json ]; then "
            "pte artifacts validate --run runs/gsm8k-runpod-smoke; "
            "else "
            f"accelerate launch --num_processes 2 -m post_train_engine.cli run --config {config} --no-env; "
            "fi"
        )
        self._ssh(pod_id, spec, script, timeout=300, check=True)

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
            "if [ -e \"$path\" ]; then paths+=(\"$path\"); fi; done; "
            "if [ ${#paths[@]} -eq 0 ]; then "
            f"tar -czf {archive} --files-from /dev/null; "
            "else "
            f"tar -czf {archive} --exclude='*.safetensors' "
            "--exclude='checkpoint-*' \"${paths[@]}\"; fi"
        )
        self._ssh(pod_id, spec, script, timeout=60, check=True)
        destination = Path(spec.attempt_dir) / "evidence.tar.gz"
        command = [
            "scp",
            *self._ssh_options(port, known_hosts, scp=True),
            f"root@{host}:{remote_archive}",
            str(destination),
        ]
        self._run(command, timeout=120, check=True)
        if not destination.is_file() or destination.stat().st_size <= 0:
            raise RuntimeError("RunPod evidence download produced no local archive")
        _write_json(
            Path(spec.attempt_dir) / "evidence_download.json",
            {
                "pod_id": pod_id,
                "path": destination.name,
                "byte_count": destination.stat().st_size,
                "sha256": "sha256:" + hashlib.sha256(destination.read_bytes()).hexdigest(),
            },
        )

    def _ssh(
        self,
        pod_id: str,
        spec: RunPodAttemptSpec,
        script: str,
        *,
        timeout: int,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        host, port, known_hosts = self._connection(pod_id, spec)
        command = [
            "ssh",
            *self._ssh_options(port, known_hosts),
            f"root@{host}",
            "bash",
            "-lc",
            shlex.quote(script),
        ]
        return self._run(command, timeout=timeout, check=check)

    def _connection(
        self,
        pod_id: str,
        spec: RunPodAttemptSpec,
    ) -> tuple[str, int, Path]:
        if pod_id in self._connections:
            return self._connections[pod_id]
        for _attempt in range(24):
            pod = self.control.get_pod(pod_id)
            host_value = pod.get("publicIp")
            mappings = pod.get("portMappings")
            port_value = mappings.get("22") if isinstance(mappings, dict) else None
            if isinstance(host_value, str) and isinstance(port_value, int):
                host = str(ipaddress.ip_address(host_value))
                if not 0 < port_value < 65536:
                    raise ValueError("RunPod SSH port must be between 1 and 65535")
                known_hosts = Path(spec.attempt_dir) / "known_hosts"
                scan = self._run(
                    ["ssh-keyscan", "-T", "5", "-p", str(port_value), host],
                    timeout=10,
                    check=True,
                )
                if not scan.stdout.strip():
                    raise RuntimeError("ssh-keyscan returned no RunPod host key")
                known_hosts.parent.mkdir(parents=True, exist_ok=True)
                known_hosts.write_text(scan.stdout, encoding="utf-8")
                fingerprint = hashlib.sha256(scan.stdout.encode("utf-8")).hexdigest()
                _write_json(
                    Path(spec.attempt_dir) / "ssh_connection.json",
                    {
                        "pod_id": pod_id,
                        "host": host,
                        "port": port_value,
                        "known_hosts_sha256": f"sha256:{fingerprint}",
                    },
                )
                connection = (host, port_value, known_hosts)
                self._connections[pod_id] = connection
                return connection
            self.sleep(5.0)
        raise TimeoutError("RunPod did not expose a public SSH mapping within 120 seconds")

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
        timeout: int,
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
    [RunPodAttemptSpec, PodCreateReceipt],
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
    minimum_grpo_remaining_seconds: int = 420,
) -> RunPodAttemptSpec:
    """Freeze one reviewed dry-run plan without contacting RunPod."""

    plan_path = Path(plan_path)
    raw = json.loads(plan_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("RunPod plan must be a JSON object")
    job = raw.get("job")
    if not isinstance(job, dict) or job.get("command") != CANONICAL_R4_COMMAND:
        raise ValueError("RunPod attempt plan must run the canonical R4 command first")
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
        repo_url=repo_url,
        commit_sha=commit_sha,
        pod_name=pod_name,
        create_request=request,
        budget=budget,
        minimum_grpo_remaining_seconds=minimum_grpo_remaining_seconds,
    )
    _write_json(attempt_dir / "attempt.json", spec.model_dump(mode="json"))
    _write_json(
        attempt_dir / "attempt_state.json",
        {"state": "prepared", "attempt": "attempt.json"},
    )
    return spec


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
        receipt: PodCreateReceipt | None = None
        r4_certified = False
        grpo_ran = False
        remote_started = False
        primary_error: BaseException | None = None
        cleanup_error: BaseException | None = None

        try:
            existing_receipt = self.control.created_receipt()
            active_pods = self.control.list_pods()
            if existing_receipt is None:
                if active_pods:
                    raise RuntimeError("RunPod attempt requires zero active Pods")
                self._state(spec, "creating")
                receipt = self.control.create_pod(
                    spec.create_request,
                    budget=spec.budget,
                )
            else:
                matching = [
                    pod
                    for pod in active_pods
                    if str(pod.get("id")) == existing_receipt.pod_id
                ]
                if len(matching) != 1 or len(active_pods) != 1:
                    raise RuntimeError(
                        "created RunPod attempt does not match the active Pod inventory"
                    )
                receipt = existing_receipt
            self._state(spec, "created", pod_id=receipt.pod_id)
            watchdog = self.launch_watchdog(spec, receipt)
            if watchdog.get("state") != "armed":
                raise RuntimeError("RunPod watchdog did not enter armed state")
            remote_started = True
            self._state(spec, "watchdog_armed", pod_id=receipt.pod_id)
            self.remote.bootstrap(receipt.pod_id, spec)
            self._state(spec, "preflight_passed", pod_id=receipt.pod_id)
            r4_certified = self.remote.run_r4(receipt.pod_id, spec)
            self._state(
                spec,
                "r4_certified" if r4_certified else "r4_rejected",
                pod_id=receipt.pod_id,
            )
            remaining = (
                receipt.recorded_at_unix
                + receipt.hard_deadline_seconds
                - self.clock()
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
                    if remote_started:
                        self.remote.download_evidence(receipt.pod_id, spec)
                except BaseException as exc:
                    cleanup_error = exc
                try:
                    delete_error: BaseException | None = None
                    try:
                        self.control.delete_pod(receipt.pod_id)
                    except BaseException as exc:
                        delete_error = exc
                    if not self.control.verify_pod_absent(receipt.pod_id):
                        self.control.record_delete_unverified(receipt.pod_id)
                        if delete_error is not None:
                            raise delete_error
                        raise RuntimeError("RunPod Pod remains active after deletion")
                except BaseException as exc:
                    cleanup_error = cleanup_error or exc

        if primary_error is not None and cleanup_error is not None:
            raise RunPodAttemptCleanupError(
                "RunPod attempt failed with "
                f"{type(primary_error).__name__}, and mandatory cleanup failed with "
                f"{type(cleanup_error).__name__}"
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
            {"state": state, **evidence},
        )


def load_runpod_attempt(path: str | Path) -> RunPodAttemptSpec:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return RunPodAttemptSpec.model_validate(raw)


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
        f"cd {workdir}; test \"$(git remote get-url origin)\" = {repo_url}; "
        "else "
        f"git clone {repo_url} {workdir}; cd {workdir}; "
        "fi; "
        f"git checkout --detach {commit_sha}; "
        f"test \"$(git rev-parse HEAD)\" = {commit_sha}; "
        "test -z \"$(git status --porcelain)\"; "
        "mkdir -p artifacts/runpod; "
        "python src/post_train_engine/runpod_preflight.py --constraints-only; "
        "python -c \"import torch; print(torch.__version__, torch.version.cuda)\" "
        "> artifacts/runpod/torch_before.txt; "
        "python -m pip install -r requirements/runpod.txt; "
        "python -m pip install --no-deps -e '.[rlvr]'; "
        "python -c \"import torch; print(torch.__version__, torch.version.cuda)\" "
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


__all__ = [
    "CANONICAL_R4_COMMAND",
    "RunPodAttemptResult",
    "RunPodAttemptCleanupError",
    "RunPodAttemptRunner",
    "RunPodAttemptSpec",
    "RunPodRemoteExecutor",
    "SSHRunPodRemoteExecutor",
    "load_runpod_attempt",
    "prepare_runpod_attempt",
    "settle_runpod_billing",
]
