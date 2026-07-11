"""Model generation backends for probe and eval rollouts."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any, Literal, cast

from post_train_engine.tasks.gsm8k import GSM8KExample

GenerationBackend = Literal["vllm", "hf"]
GenerationFn = Callable[
    [GSM8KExample, int, str, Mapping[str, Any]],
    str | Mapping[str, Any],
]

DEFAULT_ROLLOUT_BACKEND: GenerationBackend = "vllm"


def resolve_generation_backend(backend: str | None) -> GenerationBackend:
    resolved = (backend or DEFAULT_ROLLOUT_BACKEND).lower()
    if resolved not in {"vllm", "hf"}:
        raise ValueError(f"unknown generation backend: {backend!r}")
    return cast(GenerationBackend, resolved)


def build_generation_fn(
    model_id: str,
    *,
    backend: str | None = None,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    default_greedy: bool = False,
    vllm_tensor_parallel_size: int = 1,
    vllm_dtype: str = "auto",
    trust_remote_code: bool = False,
) -> GenerationFn:
    resolved = resolve_generation_backend(backend)
    if resolved == "vllm":
        return build_vllm_generation_fn(
            model_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            default_greedy=default_greedy,
            tensor_parallel_size=vllm_tensor_parallel_size,
            dtype=vllm_dtype,
            trust_remote_code=trust_remote_code,
        )
    return build_hf_generation_fn(
        model_id,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        default_greedy=default_greedy,
        trust_remote_code=trust_remote_code,
    )


def build_vllm_generation_fn(
    model_id: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    default_greedy: bool = False,
    tensor_parallel_size: int = 1,
    dtype: str = "auto",
    trust_remote_code: bool = False,
) -> GenerationFn:
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError(
            "vLLM is the default rollout backend, but the `vllm` package is not "
            "installed. Install the rollout extras on a GPU host or pass "
            "--generation-backend hf for a slower Transformers fallback."
        ) from exc

    llm = LLM(
        model=model_id,
        dtype=dtype,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=trust_remote_code,
    )

    def generate(
        _example: GSM8KExample,
        _rollout_id: int,
        prompt: str,
        generation_config: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        greedy = bool(generation_config.get("greedy", default_greedy))
        rollout_seed = _rollout_seed(generation_config, _rollout_id)
        sampling_kwargs: dict[str, Any] = {
            "max_tokens": int(generation_config.get("max_new_tokens", max_new_tokens)),
            "temperature": 0.0
            if greedy
            else float(generation_config.get("temperature", temperature)),
            "top_p": 1.0 if greedy else float(generation_config.get("top_p", top_p)),
        }
        if rollout_seed is not None:
            sampling_kwargs["seed"] = rollout_seed
        sampling_params = SamplingParams(
            **sampling_kwargs,
        )
        start = time.perf_counter()
        outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
        elapsed_ms = (time.perf_counter() - start) * 1000
        completion_output = outputs[0].outputs[0]
        completion = str(completion_output.text)
        token_ids = getattr(completion_output, "token_ids", None)
        return {
            "completion": completion,
            "completion_tokens": (
                len(token_ids) if token_ids is not None else len(completion.split())
            ),
            "finish_reason": str(getattr(completion_output, "finish_reason", "stop")),
            "latency_ms": elapsed_ms,
            "error": None,
        }

    return generate


def build_hf_generation_fn(
    model_id: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    default_greedy: bool = False,
    trust_remote_code: bool = False,
) -> GenerationFn:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    def generate(
        _example: GSM8KExample,
        _rollout_id: int,
        prompt: str,
        generation_config: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        greedy = bool(generation_config.get("greedy", default_greedy))
        rollout_seed = _rollout_seed(generation_config, _rollout_id)
        inputs = tokenizer(prompt, return_tensors="pt")
        start = time.perf_counter()
        with torch.no_grad():
            if rollout_seed is not None:
                torch.manual_seed(rollout_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(rollout_seed)
            generate_kwargs: dict[str, Any] = {
                **inputs,
                "do_sample": not greedy,
                "max_new_tokens": int(generation_config.get("max_new_tokens", max_new_tokens)),
                "pad_token_id": tokenizer.pad_token_id,
            }
            if not greedy:
                generate_kwargs["temperature"] = float(
                    generation_config.get("temperature", temperature)
                )
                generate_kwargs["top_p"] = float(generation_config.get("top_p", top_p))
            output = model.generate(**generate_kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        generated_ids = output[0, inputs["input_ids"].shape[-1] :]
        completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
        return {
            "completion": completion,
            "completion_tokens": int(generated_ids.shape[-1]),
            "finish_reason": "stop",
            "latency_ms": elapsed_ms,
            "error": None,
        }

    return generate


def _rollout_seed(generation_config: Mapping[str, Any], rollout_id: int) -> int | None:
    seed = generation_config.get("seed")
    if seed is None:
        return None
    return int(seed) + int(rollout_id)
