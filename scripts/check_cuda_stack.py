"""Fail-fast CUDA stack check for RunPod-style GRPO pods."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata

from post_train_engine.runpod import validate_cuda_runtime
from post_train_engine.runpod_grpo import load_runpod_grpo_config


def _installed_packages() -> dict[str, str]:
    packages: dict[str, str] = {}
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        if name:
            packages[name.lower()] = dist.version
    return packages


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/gsm8k_runpod_smoke.yaml")
    args = parser.parse_args(argv)
    try:
        config = load_runpod_grpo_config(args.config)
    except Exception as exc:
        print(f"FAIL: could not load RunPod config: {exc!r}")
        return 1
    try:
        import torch
    except Exception as exc:
        print(f"FAIL: could not import torch: {exc!r}")
        return 1

    packages = _installed_packages()
    torch_cuda = torch.version.cuda or "cpu"
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    devices = (
        [torch.cuda.get_device_name(idx) for idx in range(device_count)]
        if cuda_available
        else []
    )
    print(f"torch: {torch.__version__}")
    print(f"torch cuda: {torch_cuda}")
    print(f"cuda available: {cuda_available}")
    print(f"device count: {device_count}")
    print(f"devices: {devices}")

    cu13_packages = sorted(
        name
        for name in packages
        if name.endswith("-cu13")
        or (name.startswith(("cuda-", "nvidia-")) and packages[name].startswith("13."))
    )
    if cu13_packages and torch_cuda.startswith("12."):
        print("FAIL: CUDA 13 pip packages are installed next to a Torch CUDA 12 build:")
        for name in cu13_packages:
            print(f"  {name}=={packages[name]}")
        return 1

    for package in ("torchvision", "torchaudio"):
        if package not in packages:
            continue
        try:
            __import__(package)
        except Exception as exc:
            print(
                f"FAIL: optional package {package}=={packages[package]} is installed "
                f"but cannot be imported: {exc!r}"
            )
            print("Repair hint: uninstall the broken optional wheel on the disposable pod.")
            return 1

    try:
        validate_cuda_runtime(
            torch_module=torch,
            expected_cuda_version=config.execution.cuda_version,
            expected_gpu_count=config.execution.gpu_count,
            expected_gpu_type=config.execution.gpu_type,
        )
    except RuntimeError as exc:
        print(f"FAIL: {exc}")
        return 1

    print("PASS: CUDA stack and hardware match the RunPod config.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
