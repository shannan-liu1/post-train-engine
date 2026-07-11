from __future__ import annotations

import json
from pathlib import Path

from post_train_engine.evals.hf_cli import HfEvalArgs, load_jsonl_exact_source, run_hf_eval
from post_train_engine.tasks.schema import Example


def test_hf_eval_cli_runs_exact_match_jsonl_source_with_injected_generator(
    tmp_path: Path,
) -> None:
    examples_path = tmp_path / "examples.jsonl"
    examples_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "a",
                        "source": "toy",
                        "prompt": "1+1",
                        "final_answer": "2",
                    }
                ),
                json.dumps(
                    {
                        "id": "b",
                        "source": "toy",
                        "prompt": "1+2",
                        "final_answer": "3",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "eval"

    result = run_hf_eval(
        HfEvalArgs(
            model_id="unused-in-test",
            examples_jsonl=examples_path,
            out_dir=out_dir,
            source_name="toy",
        ),
        generator=lambda example: "2" if example.id == "a" else "wrong",
    )

    assert result.summary["accuracy"] == 0.5
    assert json.loads((out_dir / "summary.json").read_text(encoding="utf-8")) == {
        "source": "toy",
        "summary": {
            "n": 2.0,
            "parse_success_rate": 1.0,
            "accuracy": 0.5,
            "mean_score": 0.5,
        },
    }


def test_load_jsonl_exact_source_rejects_examples_without_final_answer(
    tmp_path: Path,
) -> None:
    examples_path = tmp_path / "bad.jsonl"
    examples_path.write_text(
        json.dumps({"id": "bad", "source": "toy", "prompt": "missing answer"})
        + "\n",
        encoding="utf-8",
    )

    source = load_jsonl_exact_source(examples_path, source_name="toy")

    try:
        source.load_examples()
    except ValueError as exc:
        assert "final_answer" in str(exc)
    else:
        raise AssertionError("expected load_examples to reject missing final_answer")


def test_load_jsonl_exact_source_extracts_last_non_empty_line(tmp_path: Path) -> None:
    examples_path = tmp_path / "examples.jsonl"
    examples_path.write_text(
        json.dumps(
            {
                "id": "a",
                "source": "toy",
                "prompt": "prompt",
                "final_answer": "answer",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    source = load_jsonl_exact_source(examples_path, source_name="toy")

    example = Example(id="a", source="toy", prompt="prompt", final_answer="answer")

    assert source.grade("scratch\n\nanswer", example).is_correct is True
