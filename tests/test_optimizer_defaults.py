from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from post_train_engine.config import ExperimentConfig, OptimizerConfig, TrainingConfig
from post_train_engine.training.optimizers import (
    DEFAULT_OPTIMIZER_FRAMEWORK,
    build_optimizer,
    split_muon_parameters,
)


class _Param:
    def __init__(self, ndim: int, *, requires_grad: bool = True) -> None:
        self.ndim = ndim
        self.requires_grad = requires_grad


def test_experiment_config_defaults_to_muon_optimizer() -> None:
    config = ExperimentConfig.model_validate(
        {
            "model": {"base_model_id": "base"},
            "task": {"name": "gsm8k"},
            "method": {"name": "grpo"},
            "training": {"max_steps": 1, "lr": 1e-6},
            "eval": {"source": "gsm8k"},
        }
    )

    assert DEFAULT_OPTIMIZER_FRAMEWORK == "muon"
    assert config.optimizer.framework == "muon"


def test_muon_parameter_split_keeps_embeddings_heads_and_biases_on_aux_adam() -> None:
    block = _Param(2)
    embed = _Param(2)
    head = _Param(2)
    bias = _Param(1)
    frozen = _Param(2, requires_grad=False)

    split = split_muon_parameters(
        [
            ("model.layers.0.mlp.down_proj.weight", block),
            ("model.embed_tokens.weight", embed),
            ("lm_head.weight", head),
            ("model.layers.0.mlp.down_proj.bias", bias),
            ("model.layers.1.mlp.down_proj.weight", frozen),
        ]
    )

    assert split.hidden == (block,)
    assert split.aux == (embed, head, bias)


def test_default_muon_optimizer_fails_closed_when_muon_package_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "muon":
            raise ImportError("no muon here")
        return real_import(name, *args, **kwargs)

    class Model:
        def named_parameters(self) -> list[tuple[str, _Param]]:
            return [("model.layers.0.mlp.down_proj.weight", _Param(2))]

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="optimizer.framework=muon is the default"):
        build_optimizer(
            Model(),
            OptimizerConfig(),
            TrainingConfig(max_steps=1, lr=1e-6),
            torch=object(),
        )


def test_explicit_adamw_optimizer_fallback_uses_trainable_params_only() -> None:
    trainable = _Param(2)
    frozen = _Param(2, requires_grad=False)

    class AdamW:
        def __init__(
            self,
            params: list[_Param],
            *,
            lr: float,
            betas: tuple[float, float],
            weight_decay: float,
        ) -> None:
            self.params = params
            self.lr = lr
            self.betas = betas
            self.weight_decay = weight_decay

    class Torch:
        optim = SimpleNamespace(AdamW=AdamW)

    class Model:
        def parameters(self) -> list[_Param]:
            return [trainable, frozen]

    optimizer = build_optimizer(
        Model(),
        OptimizerConfig(framework="adamw", weight_decay=0.02),
        TrainingConfig(max_steps=1, lr=1e-6, weight_decay=0.01),
        Torch,
    )

    assert optimizer.params == [trainable]
    assert optimizer.lr == 1e-6
    assert optimizer.betas == (0.9, 0.95)
    assert optimizer.weight_decay == 0.02
