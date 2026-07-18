"""Fail-fast RunPod readiness checks with one aggregate time budget."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

DEFAULT_OUT = Path("runs/runpod-preflight/preflight_report.json")
CommandSpec = tuple[str, list[str]]


def command_specs(require_cuda: bool) -> list[CommandSpec]:
    """Return only checks that need the paid remote environment."""
    python = sys.executable
    specs: list[CommandSpec] = [
        (
            "dependency_lock",
            [
                python,
                "-c",
                (
                    "from post_train_engine.runpod_preflight import "
                    "verify_runpod_constraints; verify_runpod_constraints(); "
                    "print('RunPod constraints OK')"
                ),
            ],
        ),
        (
            "config_validation",
            [
                python,
                "-c",
                (
                    "from post_train_engine.runpod_grpo import load_runpod_grpo_config; "
                    "load_runpod_grpo_config('configs/gsm8k_runpod_smoke.yaml'); "
                    "print('RunPod GRPO config OK')"
                ),
            ],
        ),
        (
            "grpo_trl_config_smoke",
            [
                python,
                "-c",
                (
                    "from trl import GRPOConfig; "
                    "from post_train_engine.runpod_grpo import "
                    "load_runpod_grpo_config,_filter_trl_config_kwargs,_grpo_config_kwargs; "
                    "cfg=load_runpod_grpo_config('configs/gsm8k_runpod_smoke.yaml'); "
                    "kwargs=_grpo_config_kwargs(cfg); "
                    "kwargs['per_device_train_batch_size']=cfg.training.num_generations; "
                    "GRPOConfig(**_filter_trl_config_kwargs(kwargs, GRPOConfig)); "
                    "print('GRPOConfig OK')"
                ),
            ],
        ),
    ]
    if require_cuda:
        specs.insert(1, ("cuda_probe", [python, "scripts/check_cuda_stack.py"]))
    return specs


def verify_runpod_constraints(root: str | Path = ".") -> None:
    """Bind the Torch-excluding RunPod requirements to the normalized uv lock."""
    root = Path(root)
    lock = root / "uv.lock"
    constraints = root / "requirements" / "runpod.txt"
    if not lock.is_file() or not constraints.is_file():
        raise ValueError("RunPod constraints require uv.lock and requirements/runpod.txt")
    normalized_lock = lock.read_text(encoding="utf-8").encode("utf-8")
    digest = hashlib.sha256(normalized_lock).hexdigest()
    lines = constraints.read_text(encoding="utf-8").splitlines()
    if f"# uv-lock-sha256: {digest}" not in lines[:3]:
        raise ValueError("requirements/runpod.txt does not match the normalized uv.lock")
    if not any("--hash=sha256:" in line for line in lines):
        raise ValueError("requirements/runpod.txt must include package hashes")
    if any(
        re.match(r"torch(?:\s|[<>=!~@\[]|$)", line.strip(), re.IGNORECASE)
        for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    ):
        raise ValueError("requirements/runpod.txt must preserve image-provided Torch")


def _run_one(name: str, command: list[str], timeout_sec: float) -> dict[str, Any]:
    start = time.perf_counter()
    started_at = datetime.now(UTC).isoformat()
    try:
        proc = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "name": name,
            "command": command,
            "returncode": None,
            "ok": False,
            "timed_out": True,
            "error_type": "TimeoutExpired",
            "timeout_sec": timeout_sec,
            "started_at": started_at,
            "ended_at": datetime.now(UTC).isoformat(),
            "elapsed_sec": round(time.perf_counter() - start, 3),
            "stdout_tail": _tail(exc.stdout),
            "stderr_tail": _tail(exc.stderr),
        }
    except OSError as exc:
        return {
            "name": name,
            "command": command,
            "returncode": None,
            "ok": False,
            "timed_out": False,
            "error_type": type(exc).__name__,
            "timeout_sec": timeout_sec,
            "started_at": started_at,
            "ended_at": datetime.now(UTC).isoformat(),
            "elapsed_sec": round(time.perf_counter() - start, 3),
            "stdout_tail": "",
            "stderr_tail": str(exc)[-4000:],
        }
    return {
        "name": name,
        "command": command,
        "returncode": proc.returncode,
        "ok": proc.returncode == 0,
        "timed_out": False,
        "error_type": None,
        "timeout_sec": timeout_sec,
        "started_at": started_at,
        "ended_at": datetime.now(UTC).isoformat(),
        "elapsed_sec": round(time.perf_counter() - start, 3),
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def run_preflight(
    *,
    out: Path,
    command_timeout_sec: int,
    total_timeout_sec: int,
    cuda_required: bool,
    checks: Sequence[CommandSpec] | None = None,
) -> dict[str, Any]:
    """Run gates until the first failure and always persist the resulting evidence."""
    if command_timeout_sec <= 0 or total_timeout_sec <= 0:
        raise ValueError("preflight timeouts must be positive")
    specs = list(checks) if checks is not None else command_specs(cuda_required)
    deadline = time.monotonic() + total_timeout_sec
    results: list[dict[str, Any]] = []
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        for name, command in specs:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                results.append(
                    {
                        "name": name,
                        "command": command,
                        "returncode": None,
                        "ok": False,
                        "timed_out": True,
                        "error_type": "AggregateTimeout",
                        "timeout_sec": 0,
                        "started_at": None,
                        "ended_at": datetime.now(UTC).isoformat(),
                        "elapsed_sec": 0.0,
                        "stdout_tail": "",
                        "stderr_tail": "aggregate preflight deadline expired",
                    }
                )
                break
            result = _run_one(
                name,
                command,
                min(float(command_timeout_sec), remaining),
            )
            results.append(result)
            if not result["ok"]:
                break
    finally:
        report = {
            "schema_version": "runpod_preflight_v2",
            "ok": bool(results) and all(result["ok"] for result in results) and len(results) == len(specs),
            "generated_at": datetime.now(UTC).isoformat(),
            "cuda_required": cuda_required,
            "command_timeout_sec": command_timeout_sec,
            "total_timeout_sec": total_timeout_sec,
            "results": results,
        }
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _tail(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return value[-4000:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--constraints-only", action="store_true")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--timeout-sec", type=int, default=120)
    parser.add_argument("--total-timeout-sec", type=int, default=300)
    parser.add_argument("--no-cuda-required", action="store_true")
    args = parser.parse_args()
    if args.constraints_only:
        verify_runpod_constraints(args.root)
        print("RunPod constraints OK")
        return
    if args.root != Path("."):
        parser.error("--root is valid only with --constraints-only")
    report = run_preflight(
        out=args.out,
        command_timeout_sec=args.timeout_sec,
        total_timeout_sec=args.total_timeout_sec,
        cuda_required=not args.no_cuda_required,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()


__all__ = [
    "CommandSpec",
    "command_specs",
    "main",
    "run_preflight",
    "verify_runpod_constraints",
]
