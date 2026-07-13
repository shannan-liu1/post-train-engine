from __future__ import annotations

from typing import Any

import pytest

from post_train_engine.cli.gsm8k import (
    probe_gsm8k_examples,
)
from post_train_engine.generation import (
    DEFAULT_ROLLOUT_BACKEND,
    build_generation_fn,
    resolve_generation_backend,
)
from post_train_engine.ledger import make_run_ledger_entry
from post_train_engine.probe import EarlyExitConfig
from post_train_engine.tasks.gsm8k import GSM8KExample


def _example(idx: int = 1) -> GSM8KExample:
    return GSM8KExample(
        id=f"gsm8k/train/{idx:06d}",
        split="train",
        question="What is 9+9?",
        gold_solution="9+9=18 #### 18",
        gold_answer="18",
        source="openai/gsm8k",
        metadata={},
    )


def _generate_correct(*_args: object) -> str:
    return "Reasoning <answer>18</answer>"


def test_probe_artifact_schema() -> None:
    rows = probe_gsm8k_examples(
        [_example()],
        _generate_correct,
        run_id="probe",
        bucket_source_split="train_pool",
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        checkpoint_id="base",
        prompt_style="thinking_tags",
        generation_config={"temperature": 1.0, "top_p": 0.97, "max_new_tokens": 512},
        early_exit=EarlyExitConfig(g_total=4, g_early=4, early_max_successes=4),
        seed=1000,
    )
    body = rows[0].to_json()

    assert body["run_id"] == "probe"
    assert body["example_id"] == "gsm8k/train/000001"
    assert body["prompt_hash"].startswith("sha256:")
    assert body["parse_ok"] is True
    assert body["correct"] is True
    assert body["reward"] > 0


def test_default_rollout_backend_is_vllm() -> None:
    assert DEFAULT_ROLLOUT_BACKEND == "vllm"
    assert resolve_generation_backend(None) == "vllm"


def test_default_vllm_generation_backend_fails_closed_when_vllm_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "vllm":
            raise ImportError("no vllm here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="vLLM is the default rollout backend"):
        build_generation_fn(
            "model",
            max_new_tokens=16,
            temperature=1.0,
            top_p=0.97,
        )


def test_generation_backend_is_persisted_in_probe_artifacts() -> None:
    rows = probe_gsm8k_examples(
        [_example()],
        _generate_correct,
        run_id="probe",
        bucket_source_split="train_pool",
        model_id="model",
        checkpoint_id="base",
        prompt_style="plain",
        generation_config={
            "backend": "vllm",
            "temperature": 1.0,
            "top_p": 0.97,
            "max_new_tokens": 512,
        },
        early_exit=EarlyExitConfig(g_total=4, g_early=4, early_max_successes=4),
        seed=1000,
    )

    assert rows[0].generation_config["backend"] == "vllm"


def test_run_ledger_contains_required_hashes() -> None:
    entry = make_run_ledger_entry(
        run_id="run",
        parent_run_id="probe",
        command=["pte", "gsm8k", "probe"],
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        base_model_revision="rev",
        checkpoint="base",
        dataset="openai/gsm8k",
        dataset_revision="rev",
        split_hash="sha256:split",
        prompt_template="prompt",
        reward_config={"parse_bonus": 0.02},
        trainer_config={"max_steps": 1},
        seeds=[1337],
        artifact_paths={"probe": "probe.jsonl"},
    ).to_json()

    for key in [
        "prompt_template_hash",
        "verifier_hash",
        "reward_config_hash",
        "trainer_config_hash",
    ]:
        assert entry[key].startswith("sha256:")
