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

RUNPOD_API_KEY_ENV = "PTE_REMOTE_RUNPOD_ALL"
_DELETE_VERIFY_ATTEMPTS = 3
_MINIMUM_DETACHED_LEAD_SECONDS = 5.0
_AMBIGUOUS_RECONCILIATION_INTERVAL_SECONDS = 10.0
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
    sleep: Callable[[float], None] = time.sleep,
    transport_factory: Callable[[str], RunPodTransport] = RunPodRESTTransport,
) -> dict[str, Any]:
    """Launch the provider-authoritative watchdog without serializing its key."""

    if not api_key:
        raise ValueError("RunPod API key must be non-empty")
    journal = Path(journal_path).resolve()
    receipt = Path(receipt_path).resolve()
    log = Path(log_path).resolve()
    operation = _load_guarded_operation(journal)
    target = _watchdog_target(operation)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    if (
        "pod_id" in target
        and target["delete_at_unix"] - clock() <= _MINIMUM_DETACHED_LEAD_SECONDS
    ):
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
            **target,
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
    ready = False
    for _attempt in range(20):
        if process.poll() is not None:
            raise RuntimeError("RunPod deletion watchdog exited during launch")
        try:
            child_receipt = json.loads(receipt.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            child_receipt = {}
        if isinstance(child_receipt, dict) and child_receipt.get("state") == "ready":
            ready = True
            break
        sleep(0.1)
    if not ready:
        raise RuntimeError("RunPod deletion watchdog did not confirm readiness")
    result = {
        "state": "armed",
        **target,
        "pid": process.pid,
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
    """Supervise one intent until its Pod is absent or provider TTL takes over."""

    if not api_key:
        raise ValueError("RunPod API key must be non-empty")
    journal = Path(journal_path).resolve()
    receipt = Path(receipt_path).resolve()
    operation = _load_guarded_operation(journal)
    target = _watchdog_target(operation)
    _write_json_atomic(
        receipt,
        {"state": "ready", **target, "recorded_at_unix": time.time()},
    )
    control = RunPodControlPlane(
        transport_factory(api_key),
        journal,
        sleep=sleep,
        clock=clock,
    )
    while True:
        operation = _load_guarded_operation(journal)
        if operation.get("state") == "deleted":
            return _confirm_deleted_operation(control, operation, receipt)
        target = _watchdog_target(operation)
        delay = target["delete_at_unix"] - clock()
        if delay > 0.0:
            sleep(delay)
            refreshed = _load_guarded_operation(journal)
            if refreshed.get("state") == "deleted":
                continue
            refreshed_target = _watchdog_target(refreshed)
            if refreshed_target["delete_at_unix"] > target["delete_at_unix"]:
                continue
            target = refreshed_target
        break

    pod_id = target.get("pod_id")
    while not isinstance(pod_id, str) or not pod_id:
        operation = _load_guarded_operation(journal)
        if operation.get("state") == "deleted":
            return _confirm_deleted_operation(control, operation, receipt)
        refreshed_target = _watchdog_target(operation)
        pod_id = refreshed_target.get("pod_id")
        if isinstance(pod_id, str) and pod_id:
            break
        pod_name = str(target.get("pod_name") or "")
        inventory_error: BaseException | None = None
        try:
            pods = control.list_pods()
        except (OSError, TimeoutError) as exc:
            inventory_error = exc
            pods = []
        matches = [row for row in pods if str(row.get("name")) == pod_name]
        if len(matches) > 1:
            raise RuntimeError(f"multiple RunPod Pods share watchdog name {pod_name!r}")
        if matches:
            pod_id = str(matches[0].get("id") or "")
            if not pod_id:
                raise RuntimeError("RunPod watchdog reconciliation found no Pod id")
            break
        provider_ttl_unix = _provider_ttl_unix(operation)
        now = clock()
        if inventory_error is not None:
            if now >= provider_ttl_unix:
                result = {
                    "state": "failed",
                    "pod_name": pod_name,
                    "error_type": type(inventory_error).__name__,
                    "error": "provider inventory remained unavailable through provider TTL",
                    "recorded_at_unix": time.time(),
                }
                _write_json_atomic(receipt, result)
                raise RuntimeError(result["error"]) from inventory_error
            sleep(
                min(
                    _AMBIGUOUS_RECONCILIATION_INTERVAL_SECONDS,
                    provider_ttl_unix - now,
                )
            )
            continue
        if now >= provider_ttl_unix:
            result = {
                "state": "absent",
                "pod_name": pod_name,
                "recorded_at_unix": time.time(),
            }
            _write_json_atomic(receipt, result)
            return result
        sleep(
            min(
                _AMBIGUOUS_RECONCILIATION_INTERVAL_SECONDS,
                provider_ttl_unix - now,
            )
        )
    try:
        delete_attempts = control.delete_pod_and_verify(
            pod_id, attempts=_DELETE_VERIFY_ATTEMPTS
        )
    except Exception as exc:
        result = {
            "state": "failed",
            "pod_id": pod_id,
            "delete_attempts": _DELETE_VERIFY_ATTEMPTS,
            "error_type": type(exc).__name__,
            **(
                {"journal_error_type": type(exc).__name__}
                if isinstance(exc, (json.JSONDecodeError, OSError))
                else {}
            ),
            "error": "provider still reports the Pod active after deletion retries",
            "recorded_at_unix": time.time(),
        }
        _write_json_atomic(receipt, result)
        raise RuntimeError(result["error"]) from exc
    result = {
        "state": "deleted",
        "pod_id": pod_id,
        "delete_attempts": delete_attempts,
        "recorded_at_unix": time.time(),
    }
    _write_json_atomic(receipt, result)
    return result


def _confirm_deleted_operation(
    control: RunPodControlPlane,
    operation: Mapping[str, Any],
    receipt_path: Path,
) -> dict[str, Any]:
    deleted_pod_id = str(operation.get("deleted_pod_id") or "")
    create_receipt = operation.get("receipt")
    receipt_pod_id = (
        str(create_receipt.get("pod_id") or "")
        if isinstance(create_receipt, dict)
        else ""
    )
    if deleted_pod_id and receipt_pod_id and deleted_pod_id != receipt_pod_id:
        raise ValueError("RunPod journal Pod identity changed after creation")
    pod_id = deleted_pod_id or receipt_pod_id
    if not pod_id:
        raise ValueError("deleted RunPod journal is missing the Pod id")
    if not control.verify_pod_absent(pod_id):
        control.delete_pod_and_verify(pod_id, attempts=_DELETE_VERIFY_ATTEMPTS)
    result = {
        "state": "absent",
        "pod_id": pod_id,
        "recorded_at_unix": time.time(),
    }
    _write_json_atomic(receipt_path, result)
    return result


def _load_guarded_operation(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    allowed = {
        "intent",
        "create_started",
        "ambiguous",
        "created",
        "delete_requested",
        "delete_unverified",
        "deleted",
    }
    if not isinstance(raw, dict) or raw.get("state") not in allowed:
        raise ValueError("watchdog requires a guarded RunPod operation journal")
    return raw


def _watchdog_target(operation: Mapping[str, Any]) -> dict[str, Any]:
    receipt = operation.get("receipt")
    if isinstance(receipt, dict):
        pod_id = str(receipt.get("pod_id") or "")
        deadline = receipt.get("hard_deadline_seconds")
        recorded_at = receipt.get("recorded_at_unix")
        if not pod_id:
            raise ValueError("RunPod create receipt is missing the Pod id")
        identity = {"pod_id": pod_id}
    else:
        pod_name = str(operation.get("pod_name") or "")
        budget = operation.get("budget")
        deadline = (
            budget.get("minimum_runtime_seconds") if isinstance(budget, dict) else None
        )
        recorded_at = operation.get("intent_at_unix")
        if not pod_name:
            raise ValueError("RunPod create intent is missing the Pod name")
        identity = {"pod_name": pod_name}
    if type(deadline) is not int or deadline <= 0:
        raise ValueError("RunPod create receipt requires a positive hard deadline")
    if (
        type(recorded_at) not in {int, float}
        or not math.isfinite(recorded_at)
        or recorded_at <= 0.0
    ):
        raise ValueError("RunPod create receipt requires a valid recorded time")
    return {
        **identity,
        "hard_deadline_seconds": deadline,
        "delete_at_unix": float(recorded_at) + deadline,
    }


def _provider_ttl_unix(operation: Mapping[str, Any]) -> float:
    budget = operation.get("budget")
    max_runtime = (
        budget.get("max_runtime_seconds") if isinstance(budget, dict) else None
    )
    intent_at = operation.get("intent_at_unix")
    if type(max_runtime) is not int or max_runtime <= 0:
        raise ValueError("RunPod create intent requires a positive provider TTL")
    if (
        type(intent_at) not in {int, float}
        or not math.isfinite(intent_at)
        or intent_at <= 0.0
    ):
        raise ValueError("RunPod create intent requires a valid intent time")
    return float(intent_at) + max_runtime


def _watchdog_environment(source: Mapping[str, str], *, api_key: str) -> dict[str, str]:
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
    try:
        run_local_deletion_watchdog(
            journal_path=args.journal,
            receipt_path=args.receipt,
            api_key=api_key,
        )
    except BaseException as exc:
        try:
            current = json.loads(args.receipt.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            current = {}
        if not isinstance(current, dict) or current.get("state") != "failed":
            _write_json_atomic(
                args.receipt,
                {
                    "state": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "recorded_at_unix": time.time(),
                },
            )
        raise


if __name__ == "__main__":
    main()


__all__ = [
    "RUNPOD_API_KEY_ENV",
    "launch_local_deletion_watchdog",
    "run_local_deletion_watchdog",
]
