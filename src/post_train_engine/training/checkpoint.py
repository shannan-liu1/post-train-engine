"""Atomic checkpoint save/load and bounded retention."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, NamedTuple

import safetensors.torch
import torch

_STEP_PREFIX = "step-"
_STEP_DIGITS = 8
_MODEL_FILENAME = "model.safetensors"
_STATE_FILENAME = "state.pt"


class CheckpointState(NamedTuple):
    model_state_dict: dict[str, torch.Tensor]
    optimizer_state_dict: dict[str, Any]
    scheduler_state_dict: dict[str, Any]
    step: int
    rng_states: dict[str, Any]
    config: dict[str, Any]


def save_checkpoint(
    directory: Path,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    rng_states: dict[str, Any],
    config: Any,
) -> Path:
    directory = Path(directory)
    final_dir = directory / f"{_STEP_PREFIX}{step:0{_STEP_DIGITS}d}"
    tmp_dir = directory / f"{_STEP_PREFIX}{step:0{_STEP_DIGITS}d}.tmp"
    tmp_dir.mkdir(parents=True)

    safetensors.torch.save_model(model, str(tmp_dir / _MODEL_FILENAME))
    config_payload = (
        config.model_dump(mode="json")
        if hasattr(config, "model_dump")
        else dict(config)
        if isinstance(config, dict)
        else config
    )
    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng_states": rng_states,
            "config": config_payload,
        },
        tmp_dir / _STATE_FILENAME,
    )
    os.replace(tmp_dir, final_dir)
    return final_dir


def load_checkpoint(path: Path) -> CheckpointState:
    path = Path(path)
    payload = torch.load(path / _STATE_FILENAME, weights_only=True)
    return CheckpointState(
        model_state_dict=safetensors.torch.load_file(str(path / _MODEL_FILENAME)),
        optimizer_state_dict=payload["optimizer"],
        scheduler_state_dict=payload["scheduler"],
        step=int(payload["step"]),
        rng_states=payload["rng_states"],
        config=payload["config"],
    )


def apply_retention_policy(
    directory: Path,
    last_n: int,
    best_so_far: Path | None,
) -> None:
    if last_n < 0:
        raise ValueError("last_n must be non-negative")
    directory = Path(directory)
    if not directory.exists():
        return

    candidate_steps: list[tuple[int, Path]] = []
    for path in directory.iterdir():
        step = _checkpoint_step(path)
        if step is None or not path.is_dir():
            continue
        if path.is_symlink() or bool(
            getattr(path, "is_junction", lambda: False)()
        ):
            raise ValueError(f"refusing to manage linked checkpoint directory: {path}")
        candidate_steps.append((step, path))
    candidate_steps = sorted(
        candidate_steps,
        key=lambda row: row[0],
        reverse=True,
    )
    candidates = [path for _step, path in candidate_steps]
    keep: set[Path] = set(candidates[:last_n])
    if best_so_far is not None:
        best_resolved = Path(best_so_far).resolve()
        keep.update(cand for cand in candidates if cand.resolve() == best_resolved)
    for candidate in candidates:
        if candidate not in keep:
            shutil.rmtree(candidate)


def _checkpoint_step(path: Path) -> int | None:
    name = path.name
    if not name.startswith(_STEP_PREFIX):
        return None
    raw_step = name[len(_STEP_PREFIX) :]
    if len(raw_step) != _STEP_DIGITS or not raw_step.isdigit():
        return None
    return int(raw_step)
