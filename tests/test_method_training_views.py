from __future__ import annotations

from pathlib import Path

import pytest

from post_train_engine.config import ExperimentConfig
from post_train_engine.training.gpu_runners import SFTGpuRunner


def test_method_adapter_rejects_raw_config_without_training_view(tmp_path: Path) -> None:
    config = ExperimentConfig.model_validate(
        {
            "model": {"base_model_id": "base"},
            "task": {"name": "toy"},
            "method": {"name": "sft"},
            "training": {"max_steps": 1, "lr": 1e-5},
            "eval": {"source": "toy"},
            "data": {"train_path": str(tmp_path / "train.jsonl")},
        }
    )

    with pytest.raises(TypeError, match="MethodTrainingRequest"):
        SFTGpuRunner(allow_cpu=True).train(config)  # type: ignore[arg-type]
