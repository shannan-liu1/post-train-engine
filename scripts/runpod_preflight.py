"""Run minimum post-train-engine RunPod readiness checks.

This is a paid-pod gate, not a local dry-run success signal. By default it
requires CUDA because the next command is a real GRPO hillclimb.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


DEFAULT_OUT = Path("runs/runpod-preflight/preflight_report.json")


def _command_specs(require_cuda: bool) -> list[tuple[str, list[str]]]:
    python = sys.executable
    specs: list[tuple[str, list[str]]] = [
        (
            "config_validation",
            [
                python,
                "-c",
                (
                    "from post_train_engine.runpod_grpo import load_runpod_grpo_config; "
                    "load_runpod_grpo_config('configs/gsm8k_runpod_smoke.yaml'); "
                    "load_runpod_grpo_config('configs/gsm8k_runpod_300step.yaml'); "
                    "load_runpod_grpo_config('configs/gsm8k_runpod_300step_4gpu.yaml'); "
                    "print('RunPod GRPO configs OK')"
                ),
            ],
        ),
        (
            "focused_pytest",
            [
                python,
                "-m",
                "pytest",
                "tests/test_runpod_grpo_hillclimb.py",
                "tests/test_agentic_rl_contracts.py",
                "tests/test_on_policy_distillation.py",
                "tests/test_gsm8k_reward.py",
                "tests/test_gsm8k_task.py",
                "-q",
            ],
        ),
        ("ruff", [python, "-m", "ruff", "check", "."]),
        ("git_diff_check", ["git", "diff", "--check"]),
    ]
    if require_cuda:
        specs.append(
            (
                "cuda_probe",
                [python, "scripts/check_cuda_stack.py"],
            )
        )
        specs.append(
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
            )
        )
    return specs


def _run_one(name: str, command: list[str], timeout_sec: int) -> dict[str, Any]:
    start = time.perf_counter()
    started_at = datetime.now(UTC).isoformat()
    proc = subprocess.run(
        command,
        text=True,
        capture_output=True,
        timeout=timeout_sec,
        check=False,
    )
    ended_at = datetime.now(UTC).isoformat()
    return {
        "name": name,
        "command": command,
        "returncode": proc.returncode,
        "ok": proc.returncode == 0,
        "started_at": started_at,
        "ended_at": ended_at,
        "elapsed_sec": round(time.perf_counter() - start, 3),
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--timeout-sec", type=int, default=900)
    parser.add_argument(
        "--no-cuda-required",
        action="store_true",
        help="Skip CUDA probe. Use only for local static validation.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results = [
        _run_one(name, command, args.timeout_sec)
        for name, command in _command_specs(require_cuda=not args.no_cuda_required)
    ]
    report = {
        "ok": all(result["ok"] for result in results),
        "generated_at": datetime.now(UTC).isoformat(),
        "cuda_required": not args.no_cuda_required,
        "results": results,
    }
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
