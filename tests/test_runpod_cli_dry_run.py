from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from post_train_engine.cli.main import main
from post_train_engine.runpod import validate_cuda_runtime


class _FakeCuda:
    def __init__(self, names: list[str], *, available: bool = True) -> None:
        self._names = names
        self._available = available

    def is_available(self) -> bool:
        return self._available

    def device_count(self) -> int:
        return len(self._names)

    def get_device_name(self, index: int) -> str:
        return self._names[index]


def test_runpod_runtime_validation_binds_cuda_count_and_gpu_type() -> None:
    torch_module = SimpleNamespace(
        version=SimpleNamespace(cuda="12.8"),
        cuda=_FakeCuda(["NVIDIA A40", "NVIDIA A40"]),
    )

    inventory = validate_cuda_runtime(
        torch_module=torch_module,
        expected_cuda_version="12.8",
        expected_gpu_count=2,
        expected_gpu_type="A40",
    )

    assert inventory == {
        "cuda_version": "12.8",
        "device_count": 2,
        "device_names": ["NVIDIA A40", "NVIDIA A40"],
    }


@pytest.mark.parametrize(
    ("cuda_version", "device_names", "message"),
    [
        ("12.4", ["NVIDIA A40", "NVIDIA A40"], "Torch CUDA version"),
        ("12.8", ["NVIDIA A40"], "device count"),
        ("12.8", ["NVIDIA RTX 3090", "NVIDIA RTX 3090"], "GPU type"),
    ],
)
def test_runpod_runtime_validation_rejects_hardware_drift(
    cuda_version: str,
    device_names: list[str],
    message: str,
) -> None:
    torch_module = SimpleNamespace(
        version=SimpleNamespace(cuda=cuda_version),
        cuda=_FakeCuda(device_names),
    )

    with pytest.raises(RuntimeError, match=message):
        validate_cuda_runtime(
            torch_module=torch_module,
            expected_cuda_version="12.8",
            expected_gpu_count=2,
            expected_gpu_type="A40",
        )


def test_runpod_dry_run_plan_writes_remote_execution_contract(
    tmp_path: Path,
) -> None:
    config = _write_smoke_config(tmp_path)
    out_path = tmp_path / "runpod_plan.json"

    main(
        [
            "runpod",
            "plan",
            "--config",
            str(config),
            "--out",
            str(out_path),
            "--image",
            "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04",
            "--gpu-type",
            "L40S",
            "--gpu-count",
            "2",
            "--command",
            "pte run --config configs/experiments/gsm8k_smoke.yaml",
            "--setup-command",
            "uv sync",
            "--env",
            "PTE_RUN_MODE=remote",
            "--secret-env",
            "PTE_REMOTE_HF_WRITE",
            "--dry-run",
        ],
    )

    plan = json.loads(out_path.read_text(encoding="utf-8"))
    assert plan["dry_run"] is True
    assert plan["provider"] == "runpod"
    assert plan["config"]["path"] == str(config)
    assert plan["environment"]["image"] == (
        "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"
    )
    assert plan["environment"]["allowed_cuda_versions"] == ["12.8"]
    assert plan["environment"]["gpu_type"] == "L40S"
    assert plan["environment"]["gpu_count"] == 2
    assert plan["environment"]["setup_commands"] == ["uv sync"]
    assert plan["environment"]["env"] == {"PTE_RUN_MODE": "remote"}
    assert plan["environment"]["secret_env"] == ["PTE_REMOTE_HF_WRITE"]
    assert plan["job"]["command"] == "pte run --config configs/experiments/gsm8k_smoke.yaml"
    assert plan["job"]["remote_workdir"] == "/workspace/post-train-engine"
    assert plan["resource_topology"]["gpus_per_node"] == 2
    assert plan["resource_topology"]["data_parallel_size"] == 2
    assert plan["resource_topology"]["total_gpus"] == 2


def test_runpod_plan_derives_deployment_environment_from_runpod_config(
    tmp_path: Path,
) -> None:
    out_path = tmp_path / "runpod_plan.json"

    main(
        [
            "runpod",
            "plan",
            "--config",
            "configs/gsm8k_runpod_smoke.yaml",
            "--out",
            str(out_path),
            "--command",
            (
                "accelerate launch --num_processes 2 -m post_train_engine.cli "
                "run --config configs/gsm8k_runpod_smoke.yaml"
            ),
            "--dry-run",
        ]
    )

    environment = json.loads(out_path.read_text(encoding="utf-8"))["environment"]
    assert environment["image"] == (
        "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"
    )
    assert environment["allowed_cuda_versions"] == ["12.8"]
    assert environment["gpu_type"] == "NVIDIA A40"
    assert environment["gpu_count"] == 2
    assert environment["container_disk_gb"] == 100
    assert environment["volume_gb"] == 150


def test_runpod_plan_rejects_environment_override_that_conflicts_with_config(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="image does not match RunPod config"):
        main(
            [
                "runpod",
                "plan",
                "--config",
                "configs/gsm8k_runpod_smoke.yaml",
                "--out",
                str(tmp_path / "runpod_plan.json"),
                "--image",
                "runpod/pytorch:2.7.0-py3.11-cuda12.6.3-cudnn-devel-ubuntu22.04",
                "--command",
                "python -m post_train_engine.cli run --config configs/gsm8k_runpod_smoke.yaml",
                "--dry-run",
            ]
        )


def test_runpod_plan_can_reference_valid_run_bundle(tmp_path: Path) -> None:
    config = _write_smoke_config(tmp_path)
    run_dir = tmp_path / "runs" / "gsm8k-smoke"
    main(["run", "--config", str(config)])

    main(
        [
            "runpod",
            "plan",
            "--run",
            str(run_dir),
            "--image",
            "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04",
            "--gpu-type",
            "NVIDIA A100 80GB PCIe",
            "--command",
            "pte report --run runs/gsm8k-smoke",
            "--dry-run",
        ],
    )

    plan = json.loads((run_dir / "runpod_plan.json").read_text(encoding="utf-8"))
    assert plan["run"]["path"] == str(run_dir)
    assert plan["run"]["artifact_status"] == "ok"
    assert plan["job"]["sync_artifacts"] is True

    main(["report", "--run", str(run_dir)])
    main(["diagnose", "--run", str(run_dir)])
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    diagnostics = json.loads((run_dir / "diagnostics.json").read_text(encoding="utf-8"))
    assert summary["runpod_plan"].endswith("runpod_plan.json")
    assert diagnostics["artifact_refs"]["runpod_plan"].endswith("runpod_plan.json")


def test_runpod_plan_refuses_real_execution_until_adapter_exists(
    tmp_path: Path,
) -> None:
    config = _write_smoke_config(tmp_path)

    with pytest.raises(RuntimeError, match="RunPod execution is not implemented"):
        main(
            [
                "runpod",
                "plan",
                "--config",
                str(config),
                "--image",
                "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04",
                "--gpu-type",
                "NVIDIA A100 80GB PCIe",
                "--command",
                "pte run --config configs/experiments/gsm8k_smoke.yaml",
            ],
        )


def test_runpod_plan_refuses_secret_values_in_plain_env(
    tmp_path: Path,
) -> None:
    config = _write_smoke_config(tmp_path)

    with pytest.raises(ValueError, match="secret-like env key must use --secret-env"):
        main(
            [
                "runpod",
                "plan",
                "--config",
                str(config),
                "--out",
                str(tmp_path / "runpod_plan.json"),
                "--image",
                "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04",
                "--gpu-type",
                "NVIDIA A100 80GB PCIe",
                "--command",
                "pte run --config configs/experiments/gsm8k_smoke.yaml",
                "--env",
                "HF_TOKEN=unsafe",
                "--dry-run",
            ],
        )


@pytest.mark.parametrize(
    "secret_name",
    [
        "PTE_REMOTE_RUNPOD_ALL",
        "PTE_REMOTE_WANDB_API",
        "PTE_REMOTE_HF_WRITE",
        "PTE_REMOTE_HYPERBOLIC_GPU",
    ],
)
def test_runpod_plan_refuses_known_remote_secret_values_in_plain_env(
    tmp_path: Path,
    secret_name: str,
) -> None:
    config = _write_smoke_config(tmp_path)

    with pytest.raises(ValueError, match="secret-like env key must use --secret-env"):
        main(
            [
                "runpod",
                "plan",
                "--config",
                str(config),
                "--out",
                str(tmp_path / "runpod_plan.json"),
                "--image",
                "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04",
                "--gpu-type",
                "L40S",
                "--command",
                "pte run --config configs/experiments/gsm8k_smoke.yaml",
                "--env",
                f"{secret_name}=unsafe",
                "--dry-run",
            ],
        )


def test_runpod_plan_refuses_duplicate_plain_env_keys(tmp_path: Path) -> None:
    config = _write_smoke_config(tmp_path)

    with pytest.raises(ValueError, match="--env keys must be unique"):
        main(
            [
                "runpod",
                "plan",
                "--config",
                str(config),
                "--out",
                str(tmp_path / "runpod_plan.json"),
                "--image",
                "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04",
                "--gpu-type",
                "NVIDIA A100 80GB PCIe",
                "--command",
                "pte run --config configs/experiments/gsm8k_smoke.yaml",
                "--env",
                "PTE_RUN_MODE=remote",
                "--env",
                "PTE_RUN_MODE=other",
                "--dry-run",
            ],
        )


def test_runpod_plan_refuses_non_positive_gpu_count(tmp_path: Path) -> None:
    config = _write_smoke_config(tmp_path)

    with pytest.raises(ValueError, match="gpu_count must be a positive integer"):
        main(
            [
                "runpod",
                "plan",
                "--config",
                str(config),
                "--out",
                str(tmp_path / "runpod_plan.json"),
                "--image",
                "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04",
                "--gpu-type",
                "L40S",
                "--gpu-count",
                "0",
                "--command",
                "pte run --config configs/experiments/gsm8k_smoke.yaml",
                "--dry-run",
            ],
        )


def _write_smoke_config(tmp_path: Path) -> Path:
    config = tmp_path / "gsm8k_smoke.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "kind": "gsm8k_local_smoke",
                "run_id": "gsm8k-smoke",
                "out_dir": str(tmp_path / "runs" / "gsm8k-smoke"),
                "seed": 123,
                "model_id": "local-deterministic-gsm8k",
                "prompt_style": "thinking_tags",
                "rollouts": 4,
                "early_rollouts": 2,
                "max_new_tokens": 64,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config
