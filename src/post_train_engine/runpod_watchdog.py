"""Detached local RunPod deletion watchdog."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from post_train_engine.runpod_control_plane import (
    RunPodControlPlane,
    RunPodRESTTransport,
    RunPodTransport,
)

RUNPOD_API_KEY_ENV = "RUNPOD_API_KEY"
_DELETE_VERIFY_ATTEMPTS = 3
_DELETE_VERIFY_DELAY_SECONDS = 2.0
_MINIMUM_DETACHED_LEAD_SECONDS = 5.0
_CHILD_ENV_ALLOWLIST = frozenset(
    {
        "APPDATA",
        "COMSPEC",
        "HOME",
        "LOCALAPPDATA",
        "PATH",
        "PATHEXT",
        "PROGRAMDATA",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "USERPROFILE",
        "WINDIR",
    }
)


class SpawnedProcess(Protocol):
    pid: int

    def poll(self) -> int | None: ...


def launch_local_deletion_watchdog(
    *,
    journal_path: str | Path,
    receipt_path: str | Path,
    log_path: str | Path,
    api_key: str,
    spawn: Callable[..., SpawnedProcess] = subprocess.Popen,
    clock: Callable[[], float] = time.time,
    transport_factory: Callable[[str], RunPodTransport] = RunPodRESTTransport,
) -> dict[str, Any]:
    """Launch the provider-authoritative watchdog without serializing its key."""

    if not api_key:
        raise ValueError("RunPod API key must be non-empty")
    journal = Path(journal_path).resolve()
    receipt = Path(receipt_path).resolve()
    log = Path(log_path).resolve()
    operation = _load_created_operation(journal)
    pod_id, deadline_seconds, delete_at_unix = _watchdog_target(operation)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    if delete_at_unix - clock() <= _MINIMUM_DETACHED_LEAD_SECONDS:
        return run_local_deletion_watchdog(
            journal_path=journal,
            receipt_path=receipt,
            api_key=api_key,
            sleep=lambda _seconds: None,
            clock=clock,
            transport_factory=transport_factory,
        )
    _write_json_atomic(
        receipt,
        {
            "state": "launching",
            "pod_id": pod_id,
            "hard_deadline_seconds": deadline_seconds,
            "delete_at_unix": delete_at_unix,
            "recorded_at_unix": time.time(),
        },
    )
    command = [
        sys.executable,
        "-m",
        "post_train_engine.runpod_watchdog",
        "--worker",
        "--journal",
        str(journal),
        "--receipt",
        str(receipt),
    ]
    environment = _watchdog_environment(os.environ, api_key=api_key)
    spawn_kwargs: dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "stderr": subprocess.STDOUT,
        "close_fds": True,
        "env": environment,
    }
    if os.name == "nt":
        spawn_kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        )
    else:
        spawn_kwargs["start_new_session"] = True
    with log.open("ab") as log_stream:
        process = spawn(command, stdout=log_stream, **spawn_kwargs)
    if process.poll() is not None:
        raise RuntimeError("RunPod deletion watchdog exited during launch")
    result = {
        "state": "armed",
        "pod_id": pod_id,
        "pid": process.pid,
        "hard_deadline_seconds": deadline_seconds,
        "delete_at_unix": delete_at_unix,
        "recorded_at_unix": time.time(),
        "log_path": str(log),
    }
    _write_json_atomic(receipt, result)
    return result


def run_local_deletion_watchdog(
    *,
    journal_path: str | Path,
    receipt_path: str | Path,
    api_key: str,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.time,
    transport_factory: Callable[[str], RunPodTransport] = RunPodRESTTransport,
) -> dict[str, Any]:
    """Wait for the journal deadline, delete its literal Pod, and persist evidence."""

    if not api_key:
        raise ValueError("RunPod API key must be non-empty")
    journal = Path(journal_path).resolve()
    receipt = Path(receipt_path).resolve()
    operation = _load_created_operation(journal)
    pod_id, _deadline_seconds, delete_at_unix = _watchdog_target(operation)
    sleep(max(0.0, delete_at_unix - clock()))
    control = RunPodControlPlane(transport_factory(api_key), journal)
    last_error: Exception | None = None
    for attempt in range(_DELETE_VERIFY_ATTEMPTS):
        delete_error: Exception | None = None
        try:
            control.delete_pod(pod_id)
        except Exception as exc:
            delete_error = exc
            last_error = exc
        try:
            absent = control.verify_pod_absent(pod_id)
        except Exception as exc:
            absent = False
            last_error = exc
        if absent:
            result = {
                "state": "absent" if delete_error is not None else "deleted",
                "pod_id": pod_id,
                "delete_attempts": attempt + 1,
                "recorded_at_unix": time.time(),
            }
            _write_json_atomic(receipt, result)
            return result
        if attempt + 1 < _DELETE_VERIFY_ATTEMPTS:
            sleep(_DELETE_VERIFY_DELAY_SECONDS)

    journal_error: Exception | None = None
    try:
        control.record_delete_unverified(pod_id)
    except Exception as exc:
        journal_error = exc
    result = {
        "state": "failed",
        "pod_id": pod_id,
        "delete_attempts": _DELETE_VERIFY_ATTEMPTS,
        "error_type": (
            type(last_error).__name__
            if last_error is not None
            else "PodStillActiveError"
        ),
        "error": "provider still reports the Pod active after deletion retries",
        **(
            {"journal_error_type": type(journal_error).__name__}
            if journal_error is not None
            else {}
        ),
        "recorded_at_unix": time.time(),
    }
    _write_json_atomic(receipt, result)
    raise RuntimeError(result["error"]) from (journal_error or last_error)


def _load_created_operation(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("state") != "created":
        raise ValueError("watchdog requires a created RunPod operation journal")
    return raw


def _watchdog_target(operation: Mapping[str, Any]) -> tuple[str, int, float]:
    receipt = operation.get("receipt")
    if not isinstance(receipt, dict):
        raise ValueError("RunPod operation journal is missing its create receipt")
    pod_id = str(receipt.get("pod_id") or "")
    deadline = receipt.get("hard_deadline_seconds")
    recorded_at = receipt.get("recorded_at_unix")
    if not pod_id:
        raise ValueError("RunPod create receipt is missing the Pod id")
    if type(deadline) is not int or deadline <= 0:
        raise ValueError("RunPod create receipt requires a positive hard deadline")
    if (
        type(recorded_at) not in {int, float}
        or not math.isfinite(recorded_at)
        or recorded_at <= 0.0
    ):
        raise ValueError("RunPod create receipt requires a valid recorded time")
    return pod_id, deadline, float(recorded_at) + deadline


def _watchdog_environment(
    source: Mapping[str, str], *, api_key: str
) -> dict[str, str]:
    environment = {
        name: value
        for name, value in source.items()
        if name.upper() in _CHILD_ENV_ALLOWLIST
    }
    environment[RUNPOD_API_KEY_ENV] = api_key
    return environment


def _write_json_atomic(path: Path, body: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(body, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    api_key = os.environ.get(RUNPOD_API_KEY_ENV, "")
    run_local_deletion_watchdog(
        journal_path=args.journal,
        receipt_path=args.receipt,
        api_key=api_key,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "RUNPOD_API_KEY_ENV",
    "launch_local_deletion_watchdog",
    "run_local_deletion_watchdog",
]
