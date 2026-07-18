from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from post_train_engine.cli.main import main
from post_train_engine.runpod import build_runpod_create_request, validate_cuda_runtime
from post_train_engine.runpod_control_plane import RunPodAllocationPolicy


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


def test_runpod_plan_has_one_config_derived_allocation(tmp_path: Path) -> None:
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
                "run --config configs/gsm8k_runpod_r4.yaml --no-env"
            ),
            "--dry-run",
        ]
    )

    plan = json.loads(out_path.read_text(encoding="utf-8"))
    environment = plan["environment"]
    assert environment == {
        "allowed_cuda_versions": ["12.8"],
        "container_disk_gb": 40,
        "gpu_count": 2,
        "gpu_type": "NVIDIA A40",
        "image": "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04",
        "volume_gb": 0,
    }
    request = build_runpod_create_request(plan, pod_name="pte-r4-deadbeef")
    RunPodAllocationPolicy().validate_request(request)


def test_runpod_plan_rejects_removed_override_surface(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "runpod",
                "plan",
                "--config",
                "configs/gsm8k_runpod_smoke.yaml",
                "--out",
                str(tmp_path / "runpod_plan.json"),
                "--command",
                "unused",
                "--gpu-type",
                "L40S",
                "--dry-run",
            ]
        )


def test_runpod_plan_is_local_only(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="use --dry-run"):
        main(
            [
                "runpod",
                "plan",
                "--config",
                "configs/gsm8k_runpod_smoke.yaml",
                "--out",
                str(tmp_path / "runpod_plan.json"),
                "--command",
                "unused",
            ]
        )
