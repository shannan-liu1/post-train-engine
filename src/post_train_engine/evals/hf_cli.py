"""Hugging Face generation eval CLI."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from post_train_engine.evals.grades import Grade
from post_train_engine.evals.run_eval import EvalRunResult, run_eval_source, write_eval_result
from post_train_engine.evals.source import EvalSource
from post_train_engine.tasks.schema import Example

AnswerExtractor = Literal["last_line", "full_text"]
Generator = Callable[[Example], str]


@dataclass(frozen=True)
class HfEvalArgs:
    model_id: str
    examples_jsonl: Path
    out_dir: Path
    source_name: str = "jsonl_exact"
    max_examples: int | None = None
    max_new_tokens: int | None = None
    dtype: str = "bfloat16"
    device_map: str = "auto"
    trust_remote_code: bool = False
    answer_extractor: AnswerExtractor = "last_line"


def load_jsonl_exact_source(
    path: str | Path,
    *,
    source_name: str = "jsonl_exact",
    answer_extractor: AnswerExtractor = "last_line",
    default_max_new_tokens: int = 128,
) -> EvalSource:
    path = Path(path)

    def load_examples() -> list[Example]:
        rows: list[Example] = []
        with path.open("r", encoding="utf-8") as fp:
            for line_number, line in enumerate(fp, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                example = Example.model_validate(json.loads(stripped))
                if example.final_answer is None:
                    raise ValueError(
                        f"{path}:{line_number} is missing required final_answer"
                    )
                rows.append(example)
        return rows

    def extract_answer(generation: str) -> str | None:
        if answer_extractor == "full_text":
            answer = generation.strip()
        else:
            lines = [line.strip() for line in generation.splitlines() if line.strip()]
            answer = lines[-1] if lines else ""
        return answer or None

    def score(parsed_answer: str | None, example: Example) -> Grade:
        expected = example.final_answer
        if expected is None:
            raise ValueError(f"example {example.id!r} is missing final_answer")
        if parsed_answer is None:
            return Grade(
                parsed_answer=None,
                parse_success=False,
                is_correct=False,
                reason="no parsed answer",
                score=0.0,
            )
        is_correct = parsed_answer.strip() == expected.strip()
        return Grade(
            parsed_answer=parsed_answer,
            parse_success=True,
            is_correct=is_correct,
            reason="exact match" if is_correct else "exact mismatch",
            score=1.0 if is_correct else 0.0,
        )

    return EvalSource(
        name=source_name,
        load_examples=load_examples,
        extract_answer=extract_answer,
        score=score,
        default_max_new_tokens=default_max_new_tokens,
    )


def build_hf_generator(args: HfEvalArgs) -> Generator:
    """Build a Transformers-backed greedy generator.

    Imports are lazy so the rest of the eval framework remains usable in local
    tests without installing or importing GPU libraries.
    """

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(args.dtype)
    if dtype is None:
        raise ValueError(f"unsupported dtype: {args.dtype}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
        use_safetensors=True,
    )
    model.eval()

    def generate(example: Example) -> str:
        encoded = tokenizer(example.prompt, return_tensors="pt")
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens
                or load_jsonl_exact_source(
                    args.examples_jsonl,
                    source_name=args.source_name,
                    answer_extractor=args.answer_extractor,
                ).default_max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        prompt_len = encoded["input_ids"].shape[-1]
        generated = output[0, prompt_len:]
        return tokenizer.decode(generated, skip_special_tokens=True)

    return generate


def run_hf_eval(args: HfEvalArgs, *, generator: Generator | None = None) -> EvalRunResult:
    source = load_jsonl_exact_source(
        args.examples_jsonl,
        source_name=args.source_name,
        answer_extractor=args.answer_extractor,
        default_max_new_tokens=args.max_new_tokens or 128,
    )
    examples = source.load_examples()
    if args.max_examples is not None:
        examples = examples[: args.max_examples]
    result = run_eval_source(source, generator or build_hf_generator(args), examples=examples)
    write_eval_result(result, args.out_dir)
    return result


def parse_args(argv: list[str] | None = None) -> HfEvalArgs:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--examples-jsonl", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--source-name", default="jsonl_exact")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("bfloat16", "float16", "float32"),
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--answer-extractor",
        default="last_line",
        choices=("last_line", "full_text"),
    )
    parsed = parser.parse_args(argv)
    return HfEvalArgs(**vars(parsed))


def main(argv: list[str] | None = None) -> int:
    run_hf_eval(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
