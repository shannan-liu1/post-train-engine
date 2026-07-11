from __future__ import annotations

import pytest

from post_train_engine.difficulty import DifficultyBand, DifficultyMap
from post_train_engine.evals.grades import Grade
from post_train_engine.evals.source import EvalSource
from post_train_engine.probe import ProbeRunner
from post_train_engine.tasks.schema import Example


def _source(examples: list[Example]) -> EvalSource:
    def score(parsed: str | None, example: Example) -> Grade:
        expected = example.final_answer
        if parsed is None:
            return Grade(
                parsed_answer=None,
                parse_success=False,
                is_correct=False,
                score=0.0,
                reason="parse failed",
            )
        return Grade(
            parsed_answer=parsed,
            parse_success=True,
            is_correct=parsed == expected,
            score=1.0 if parsed == expected else 0.0,
            reason="exact",
        )

    return EvalSource(
        name="toy",
        load_examples=lambda: examples,
        extract_answer=lambda generation: None
        if generation == "NO_PARSE"
        else generation.strip(),
        score=score,
        default_max_new_tokens=8,
    )


def test_probe_runner_records_each_sample_and_summarizes_grades() -> None:
    examples = [
        Example(id="easy", source="toy", prompt="1+1", final_answer="2"),
        Example(id="miss", source="toy", prompt="1+2", final_answer="3"),
    ]
    generations = {
        ("easy", 0): "2",
        ("easy", 1): "2",
        ("miss", 0): "NO_PARSE",
        ("miss", 1): "4",
    }

    result = ProbeRunner(_source(examples)).run(
        lambda example, sample_index: generations[(example.id, sample_index)],
        samples_per_example=2,
    )

    assert result.summary == {
        "n": 4.0,
        "parse_success_rate": pytest.approx(0.75),
        "accuracy": pytest.approx(0.50),
        "mean_score": pytest.approx(0.50),
    }
    assert [(trace.example_id, trace.sample_index) for trace in result.traces] == [
        ("easy", 0),
        ("easy", 1),
        ("miss", 0),
        ("miss", 1),
    ]


def test_probe_runner_rejects_duplicate_example_ids() -> None:
    examples = [
        Example(id="dup", source="toy", prompt="first", final_answer="1"),
        Example(id="dup", source="toy", prompt="second", final_answer="2"),
    ]

    with pytest.raises(ValueError, match="duplicate example id: dup"):
        ProbeRunner(_source(examples)).run(
            lambda _example, _sample_index: "1",
            samples_per_example=1,
        )


def test_difficulty_map_finds_easy_learnable_unsolved_and_parser_issue_bands() -> None:
    examples = [
        Example(id="easy", source="toy", prompt="easy", final_answer="ok"),
        Example(id="learnable", source="toy", prompt="mixed", final_answer="ok"),
        Example(id="unsolved", source="toy", prompt="hard", final_answer="ok"),
        Example(id="parser", source="toy", prompt="bad parse", final_answer="ok"),
    ]
    generations = {
        ("easy", 0): "ok",
        ("easy", 1): "ok",
        ("learnable", 0): "ok",
        ("learnable", 1): "wrong",
        ("unsolved", 0): "wrong",
        ("unsolved", 1): "wrong",
        ("parser", 0): "NO_PARSE",
        ("parser", 1): "NO_PARSE",
    }
    probe = ProbeRunner(_source(examples)).run(
        lambda example, sample_index: generations[(example.id, sample_index)],
        samples_per_example=2,
    )

    difficulty = DifficultyMap.from_probe(probe)

    assert difficulty.get("easy").band is DifficultyBand.EASY
    assert difficulty.get("learnable").band is DifficultyBand.LEARNABLE
    assert difficulty.get("unsolved").band is DifficultyBand.UNSOLVED
    assert difficulty.get("parser").band is DifficultyBand.PARSER_ISSUE
    assert [record.example_id for record in difficulty.training_candidates()] == [
        "learnable",
        "unsolved",
    ]
