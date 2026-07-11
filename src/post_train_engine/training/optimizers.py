"""Optimizer selection for GPU training runners."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

from post_train_engine.config import OptimizerConfig, TrainingConfig

OptimizerFramework = Literal["muon", "adamw"]
DEFAULT_OPTIMIZER_FRAMEWORK: OptimizerFramework = "muon"

_AUX_NAME_MARKERS = (
    "embed",
    "embedding",
    "lm_head",
    "classifier",
    "score",
    "bias",
)


@dataclass(frozen=True)
class MuonParameterSplit:
    hidden: tuple[Any, ...]
    aux: tuple[Any, ...]


def split_muon_parameters(named_parameters: Iterable[tuple[str, Any]]) -> MuonParameterSplit:
    """Split trainable params into Muon hidden weights and auxiliary AdamW params."""

    hidden: list[Any] = []
    aux: list[Any] = []
    for name, param in named_parameters:
        if not bool(getattr(param, "requires_grad", True)):
            continue
        ndim = int(getattr(param, "ndim", 0))
        if ndim >= 2 and not _is_auxiliary_name(name):
            hidden.append(param)
        else:
            aux.append(param)
    return MuonParameterSplit(hidden=tuple(hidden), aux=tuple(aux))


def build_optimizer(
    model: Any,
    optimizer_config: OptimizerConfig,
    training_config: TrainingConfig,
    torch: Any,
) -> Any:
    weight_decay = optimizer_config.weight_decay
    if weight_decay is None:
        weight_decay = training_config.weight_decay
    if optimizer_config.framework == "adamw":
        return torch.optim.AdamW(
            [param for param in model.parameters() if getattr(param, "requires_grad", True)],
            lr=training_config.lr,
            betas=optimizer_config.aux_betas,
            weight_decay=weight_decay,
        )
    if optimizer_config.framework != "muon":
        raise ValueError(f"unknown optimizer framework: {optimizer_config.framework!r}")
    try:
        from muon import MuonWithAuxAdam
    except ImportError as exc:
        raise RuntimeError(
            "optimizer.framework=muon is the default, but the `muon` package is "
            "not installed. Install the optimizer extras on the training host, "
            "or set optimizer.framework: adamw for an explicit AdamW fallback."
        ) from exc

    split = split_muon_parameters(model.named_parameters())
    if not split.hidden:
        raise ValueError("Muon optimizer selected, but no trainable hidden weight matrices were found")
    param_groups: list[dict[str, Any]] = [
        {
            "params": list(split.hidden),
            "use_muon": True,
            "lr": optimizer_config.hidden_lr or training_config.lr,
            "weight_decay": weight_decay,
        }
    ]
    if split.aux:
        param_groups.append(
            {
                "params": list(split.aux),
                "use_muon": False,
                "lr": optimizer_config.aux_lr or training_config.lr,
                "betas": optimizer_config.aux_betas,
                "weight_decay": weight_decay,
            }
        )
    return MuonWithAuxAdam(param_groups)


def _is_auxiliary_name(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in _AUX_NAME_MARKERS)
