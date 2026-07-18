"""Dry-run RunPod remote execution plans."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from post_train_engine.flywheel import ResourceTopology


def cuda_version_from_image(image: str) -> str:
    """Return the RunPod CUDA allocation filter encoded by an image tag."""

    for segment in image.split("-"):
        if not segment.startswith("cuda"):
            continue
        version_parts = segment.removeprefix("cuda").split(".")
        if len(version_parts) >= 2 and all(
            part.isdigit() for part in version_parts[:2]
        ):
            return ".".join(version_parts[:2])
    raise ValueError("image must include a parseable cudaMAJOR.MINOR tag")


def validate_cuda_runtime(
    *,
    torch_module: Any,
    expected_cuda_version: str,
    expected_gpu_count: int,
    expected_gpu_type: str,
) -> dict[str, Any]:
    """Fail closed unless the visible CUDA runtime matches the RunPod plan."""

    if not torch_module.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is false")
    actual_cuda = str(torch_module.version.cuda or "cpu")
    if _major_minor(actual_cuda) != expected_cuda_version:
        raise RuntimeError(
            "Torch CUDA version does not match the RunPod image filter: "
            f"expected {expected_cuda_version}, got {actual_cuda}"
        )
    device_count = int(torch_module.cuda.device_count())
    if device_count != expected_gpu_count:
        raise RuntimeError(
            "Visible CUDA device count does not match the RunPod plan: "
            f"expected {expected_gpu_count}, got {device_count}"
        )
    device_names = [
        str(torch_module.cuda.get_device_name(index)) for index in range(device_count)
    ]
    expected_name = _normalize_gpu_type(expected_gpu_type)
    mismatches = [
        name for name in device_names if _normalize_gpu_type(name) != expected_name
    ]
    if mismatches:
        raise RuntimeError(
            "Visible GPU type does not match the RunPod plan: "
            f"expected {expected_gpu_type!r}, got {device_names!r}"
        )
    return {
        "cuda_version": _major_minor(actual_cuda),
        "device_count": device_count,
        "device_names": device_names,
    }


def _major_minor(value: str) -> str:
    parts = value.split(".")
    if len(parts) < 2 or not all(part.isdigit() for part in parts[:2]):
        raise RuntimeError(f"CUDA version must contain major.minor; got {value!r}")
    return ".".join(parts[:2])


def _normalize_gpu_type(value: str) -> str:
    normalized = "".join(
        character for character in value.lower() if character.isalnum()
    )
    for prefix in ("nvidiageforce", "nvidiatesla", "nvidia", "geforce", "tesla"):
        if normalized.startswith(prefix):
            return normalized.removeprefix(prefix)
    return normalized


def write_runpod_plan(
    *,
    image: str | None = None,
    gpu_type: str | None = None,
    command: str,
    dry_run: bool,
    run_dir: str | Path | None = None,
    config_path: str | Path | None = None,
    out_path: str | Path | None = None,
    repo_root: str | Path = ".",
    remote_workdir: str = "/workspace/post-train-engine",
    setup_commands: tuple[str, ...] = (),
    env: tuple[str, ...] = (),
    secret_env: tuple[str, ...] = (),
    gpu_count: int | None = None,
    container_disk_gb: int | None = None,
    volume_gb: int | None = None,
) -> dict[str, Any]:
    """Write a local RunPod execution plan without submitting a remote job."""

    if not dry_run:
        raise RuntimeError("RunPod execution is not implemented; use --dry-run")
    if config_path is None:
        raise ValueError("runpod plan requires --config")
    supplied = sorted(
        name
        for name, is_supplied in {
            "run_dir": run_dir is not None,
            "image": image is not None,
            "gpu_type": gpu_type is not None,
            "repo_root": Path(repo_root) != Path("."),
            "setup_commands": bool(setup_commands),
            "env": bool(env),
            "secret_env": bool(secret_env),
            "gpu_count": gpu_count is not None,
            "container_disk_gb": container_disk_gb is not None,
            "volume_gb": volume_gb is not None,
        }.items()
        if is_supplied
    )
    if supplied:
        raise ValueError(
            "canonical RunPod plan rejects overrides: " + ", ".join(supplied)
        )
    if not command:
        raise ValueError("command is required")
    runpod_execution = _runpod_execution_from_config(config_path)
    if runpod_execution is None:
        raise ValueError("runpod plan requires a RunPod GRPO config")
    image = str(runpod_execution["container_image"])
    gpu_type = str(runpod_execution["gpu_type"])
    gpu_count = int(runpod_execution["gpu_count"])
    container_disk_gb = int(runpod_execution["disk_gb"])
    volume_gb = int(runpod_execution["volume_gb"])
    cuda_version = cuda_version_from_image(image)

    topology = ResourceTopology(
        launcher="runpod",
        num_nodes=1,
        gpus_per_node=gpu_count,
        gpu_type=gpu_type,
        data_parallel_size=gpu_count,
    )
    config_path = Path(config_path).resolve()
    if not config_path.is_file():
        raise ValueError(f"config path does not exist: {config_path}")
    try:
        config_relative = config_path.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("RunPod config must remain inside the repository") from exc
    config_block = {"path": config_relative, "sha256": _sha256(config_path)}
    if out_path is None:
        raise ValueError("runpod plan requires --out")
    out_path = Path(out_path)

    plan = {
        "provider": "runpod",
        "dry_run": True,
        "will_submit": False,
        "config": config_block,
        "environment": {
            "image": image,
            "allowed_cuda_versions": [cuda_version],
            "gpu_type": gpu_type,
            "gpu_count": gpu_count,
            "container_disk_gb": container_disk_gb,
            "volume_gb": volume_gb,
        },
        "job": {
            "command": command,
            "remote_workdir": remote_workdir,
            "sync_artifacts": False,
        },
        "resource_topology": topology.model_dump(mode="json"),
    }
    _write_json(out_path, plan)
    return plan


def build_runpod_create_request(
    plan: dict[str, Any],
    *,
    pod_name: str,
) -> dict[str, Any]:
    """Compile a dry-run plan into the sole authorized REST create shape."""

    if not pod_name:
        raise ValueError("pod_name must be non-empty")
    if plan.get("provider") != "runpod" or plan.get("will_submit") is not False:
        raise ValueError("RunPod create request requires a validated dry-run plan")
    environment = plan.get("environment")
    if not isinstance(environment, dict):
        raise ValueError("RunPod plan environment must be a mapping")
    allowed_cuda_versions = environment.get("allowed_cuda_versions")
    if allowed_cuda_versions != [
        cuda_version_from_image(str(environment.get("image")))
    ]:
        raise ValueError("RunPod plan CUDA filter does not match its image")
    return {
        "name": pod_name,
        "allowedCudaVersions": list(allowed_cuda_versions),
        "cloudType": "SECURE",
        "computeType": "GPU",
        "containerDiskInGb": int(environment["container_disk_gb"]),
        "globalNetworking": True,
        "gpuCount": int(environment["gpu_count"]),
        "gpuTypeIds": [str(environment["gpu_type"])],
        "gpuTypePriority": "availability",
        "imageName": str(environment["image"]),
        "interruptible": False,
        "ports": ["22/tcp"],
        "supportPublicIp": True,
        "volumeInGb": int(environment["volume_gb"]),
    }


def _runpod_execution_from_config(
    config_path: str | Path | None,
) -> dict[str, Any] | None:
    if config_path is None:
        return None
    path = Path(config_path)
    if not path.is_file():
        raise ValueError(f"config path does not exist: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if (
        not isinstance(raw, dict)
        or raw.get("schema_version") != "runpod_grpo_hillclimb_v1"
    ):
        return None
    execution = raw.get("execution")
    if not isinstance(execution, dict):
        raise ValueError("RunPod config execution block must be a mapping")
    required = {
        "container_image",
        "gpu_type",
        "gpu_count",
        "disk_gb",
        "volume_gb",
    }
    missing = sorted(required.difference(execution))
    if missing:
        raise ValueError(f"RunPod config execution is missing fields: {missing}")
    return execution


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, body: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(body, indent=2, sort_keys=True),
        encoding="utf-8",
    )


__all__ = [
    "build_runpod_create_request",
    "cuda_version_from_image",
    "validate_cuda_runtime",
    "write_runpod_plan",
]
