from __future__ import annotations

import pytest

from post_train_engine.config import ExperimentConfig
from post_train_engine.evals.grades import Grade, summarize_grades
from post_train_engine.evals.source import EvalSource
from post_train_engine.jsonl import read_jsonl
from post_train_engine.registry.checkpoints import CheckpointRecord, CheckpointRegistry
from post_train_engine.tasks.registry import TaskRegistry, TaskSpec
from post_train_engine.tasks.schema import Example


def test_experiment_config_validates_method_task_and_schedule() -> None:
    config = ExperimentConfig.model_validate(
        {
            "model": {"base_model_id": "Qwen/Qwen2.5-0.5B"},
            "task": {"name": "toy"},
            "method": {"name": "sft"},
            "training": {"max_steps": 10, "lr": 1e-5, "warmup_steps": 2},
            "eval": {"source": "toy_eval"},
        }
    )

    assert config.method.name == "sft"
    assert config.logging.wandb_project == "post-train-engine"

    with pytest.raises(ValueError, match="warmup_steps"):
        ExperimentConfig.model_validate(
            {
                "model": {"base_model_id": "Qwen/Qwen2.5-0.5B"},
                "task": {"name": "toy"},
                "method": {"name": "sft"},
                "training": {"max_steps": 10, "lr": 1e-5, "warmup_steps": 10},
                "eval": {"source": "toy_eval"},
            }
        )


def test_example_schema_rejects_empty_required_fields() -> None:
    with pytest.raises(ValueError, match="String should have at least 1 character"):
        Example(id="", source="toy", prompt="2 + 2?")


def test_task_registry_and_eval_source_use_normalized_examples() -> None:
    example = Example(
        id="ex-1",
        source="toy",
        prompt="2 + 2?",
        final_answer="4",
    )
    task = TaskSpec(
        name="toy",
        load_train=lambda: [example],
        load_eval=lambda: [example],
    )
    tasks = TaskRegistry()
    tasks.register(task)

    source = EvalSource(
        name="toy_eval",
        load_examples=tasks.get("toy").load_eval,
        extract_answer=lambda generation: generation.strip(),
        score=lambda parsed, ex: Grade(
            parsed_answer=parsed,
            parse_success=parsed is not None,
            is_correct=parsed == ex.final_answer,
            reason="correct" if parsed == ex.final_answer else "wrong",
            score=1.0 if parsed == ex.final_answer else 0.0,
        ),
        default_max_new_tokens=8,
    )

    grade = source.grade("4", example)

    assert grade.is_correct is True
    assert summarize_grades([grade]) == {
        "n": 1.0,
        "parse_success_rate": 1.0,
        "accuracy": 1.0,
        "mean_score": 1.0,
    }


@pytest.mark.parametrize("difficulty", [0, 6])
def test_normalized_example_rejects_out_of_range_difficulty(
    difficulty: int,
) -> None:
    with pytest.raises(ValueError, match="difficulty"):
        Example(
            id="example",
            source="toy",
            prompt="2 + 2?",
            difficulty=difficulty,
        )


def test_checkpoint_registry_and_replay_buffer_persist_jsonl(tmp_path) -> None:
    checkpoints = CheckpointRegistry(tmp_path / "checkpoints.jsonl")
    checkpoints.append(
        CheckpointRecord(
            candidate_id="candidate-1",
            path="results/checkpoints/step-00000010",
            parent_id="seed",
            score=0.62,
            metrics={"accuracy": 0.62},
            promoted=True,
        )
    )
    checkpoints.append(
        CheckpointRecord(
            candidate_id="candidate-2",
            path="results/checkpoints/step-00000020",
            parent_id="candidate-1",
            score=0.61,
            metrics={"accuracy": 0.61},
            promoted=False,
            rejection_reason="no objective improvement",
        )
    )

    assert checkpoints.best_promoted().candidate_id == "candidate-1"
    assert [
        row["candidate_id"] for row in read_jsonl(tmp_path / "checkpoints.jsonl")
    ] == [
        "candidate-1",
        "candidate-2",
    ]
