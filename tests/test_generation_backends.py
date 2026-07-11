from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest

from post_train_engine.generation import build_generation_fn
from post_train_engine.tasks.gsm8k import GSM8KExample


def test_vllm_generation_uses_seed_plus_rollout_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class SamplingParams:
        def __init__(self, **kwargs: Any) -> None:
            captured["sampling_params"] = kwargs

    class LLM:
        def __init__(self, **kwargs: Any) -> None:
            captured["llm"] = kwargs

        def generate(
            self,
            prompts: list[str],
            sampling_params: SamplingParams,
            *,
            use_tqdm: bool,
        ) -> list[Any]:
            captured["prompts"] = prompts
            captured["sampling_params_obj"] = sampling_params
            captured["use_tqdm"] = use_tqdm
            return [
                SimpleNamespace(
                    outputs=[
                        SimpleNamespace(
                            text="<answer>18</answer>",
                            token_ids=[1, 2, 3],
                            finish_reason="stop",
                        )
                    ]
                )
            ]

    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(LLM=LLM, SamplingParams=SamplingParams),
    )
    generate = build_generation_fn(
        "Qwen/Qwen2.5-0.5B-Instruct",
        max_new_tokens=64,
        temperature=1.0,
        top_p=0.97,
    )

    row = generate(
        GSM8KExample(
            id="gsm8k/train/000001",
            split="train",
            question="What is 9+9?",
            gold_solution="9+9=18 #### 18",
            gold_answer="18",
            source="openai/gsm8k",
            metadata={},
        ),
        7,
        "prompt",
        {"seed": 1000, "max_new_tokens": 64, "temperature": 1.0, "top_p": 0.97},
    )

    assert captured["sampling_params"]["seed"] == 1007
    assert captured["sampling_params"]["max_tokens"] == 64
    assert captured["prompts"] == ["prompt"]
    assert captured["use_tqdm"] is False
    assert row["completion"] == "<answer>18</answer>"
    assert row["completion_tokens"] == 3


def test_hf_generation_passes_trust_remote_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: Any) -> Any:
            captured["tokenizer"] = {"model_id": model_id, **kwargs}
            return SimpleNamespace(pad_token_id=0, eos_token="<eos>")

    class AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: Any) -> Any:
            captured["model"] = {"model_id": model_id, **kwargs}
            return SimpleNamespace(eval=lambda: None)

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModelForCausalLM=AutoModelForCausalLM,
            AutoTokenizer=AutoTokenizer,
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())

    build_generation_fn(
        "custom/model",
        backend="hf",
        max_new_tokens=64,
        temperature=1.0,
        top_p=0.97,
        trust_remote_code=True,
    )

    assert captured["tokenizer"]["trust_remote_code"] is True
    assert captured["model"]["trust_remote_code"] is True
