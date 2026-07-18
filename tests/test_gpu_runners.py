from __future__ import annotations

from pathlib import Path

import pytest

from post_train_engine.config import ExperimentConfig
from post_train_engine.training.gpu_runners import (
    DPOGpuRunner,
    GRPOGpuRunner,
    SFTGpuRunner,
    _build_trainer,
    _numeric_metrics,
    default_gpu_runners,
)
from post_train_engine.training.runner import MethodTrainingRequest
from post_train_engine.training_views import TrainingDataRef, TrainingViewArtifact


def test_trainer_compatibility_does_not_mask_internal_type_error() -> None:
    class BrokenTrainer:
        def __init__(self, *, processing_class, **_kwargs):
            raise TypeError("trainer internal type failure")

    with pytest.raises(TypeError, match="trainer internal type failure"):
        _build_trainer(BrokenTrainer, object(), {})


def _config(
    method: str, tmp_path: Path, *, train_path: Path | None
) -> ExperimentConfig:
    raw = {
        "model": {"base_model_id": "base"},
        "task": {"name": "toy"},
        "method": {"name": method},
        "training": {"max_steps": 1, "lr": 1e-5},
        "eval": {"source": "toy"},
        "logging": {"run_name": f"{method}-run"},
        "checkpointing": {"save_dir": str(tmp_path / "ckpts")},
    }
    if train_path is not None:
        raw["data"] = {"train_path": str(train_path)}
    return ExperimentConfig.model_validate(raw)


def _request(
    config: ExperimentConfig, method: str, tmp_path: Path
) -> MethodTrainingRequest:
    path = config.data.train_path or tmp_path / "unused.jsonl"
    return MethodTrainingRequest(
        config,
        TrainingViewArtifact(
            view_id=f"{method}-view",
            run_id="run-1",
            task_id="toy",
            view_type="grpo_rollout" if method == "grpo" else "sft",
            method_compatibility=(method,),
            data_artifact=TrainingDataRef(
                path=path.resolve().relative_to(tmp_path.resolve()).as_posix(),
                kind="training_data",
                sha256="sha256:" + "0" * 64,
            ),
            source_trace_ids=("trace-1",),
            source_split_roles=("probe",),
            privileged_visibility="none",
        ),
        artifact_root=tmp_path,
    )


def test_sft_gpu_runner_fails_fast_on_wrong_method(tmp_path: Path) -> None:
    runner = SFTGpuRunner(allow_cpu=True)
    config = _config("dpo", tmp_path, train_path=tmp_path / "train.jsonl")

    with pytest.raises(ValueError, match="expected method sft"):
        runner.train(_request(config, "dpo", tmp_path))


def test_dpo_gpu_runner_requires_training_data_path_before_loading_gpu_stack(
    tmp_path: Path,
) -> None:
    runner = DPOGpuRunner(allow_cpu=True)
    config = _config("dpo", tmp_path, train_path=None)

    with pytest.raises(ValueError, match="data.train_path"):
        runner.train(_request(config, "dpo", tmp_path))


def test_grpo_gpu_runner_requires_reward_functions(tmp_path: Path) -> None:
    runner = GRPOGpuRunner(allow_cpu=True)
    config = _config("grpo", tmp_path, train_path=tmp_path / "train.jsonl")

    with pytest.raises(ValueError, match="reward_funcs"):
        runner.train(_request(config, "grpo", tmp_path))


def test_grpo_gpu_runner_fails_closed_on_malformed_modified_knob(
    tmp_path: Path,
) -> None:
    runner = GRPOGpuRunner(allow_cpu=True, reward_funcs=[lambda *_: 0.0])
    raw = _config("grpo", tmp_path, train_path=tmp_path / "train.jsonl").model_dump()
    raw["method"]["parameters"] = {"adaptive_entropy": True}
    config = ExperimentConfig.model_validate(raw)

    with pytest.raises(ValueError, match="adaptive_entropy"):
        runner.train(_request(config, "grpo", tmp_path))


def test_default_gpu_runners_register_expected_methods() -> None:
    runners = default_gpu_runners(allow_cpu=True, grpo_reward_funcs=[lambda *_: 0.0])

    assert sorted(runners) == ["dpo", "grpo", "sft"]


def test_numeric_metrics_excludes_booleans() -> None:
    assert _numeric_metrics({"loss": 0.5, "finished": True, "epoch": 1}) == {
        "loss": 0.5,
        "epoch": 1.0,
    }
