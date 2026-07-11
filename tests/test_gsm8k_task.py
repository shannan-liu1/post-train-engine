from __future__ import annotations

import pytest

from post_train_engine.tasks.gsm8k import (
    GSM8KExample,
    extract_gold_answer,
    make_gsm8k_splits,
    normalize_numeric_answer,
    parse_model_answer,
    verify_answer,
)


def _examples(split: str, n: int) -> list[GSM8KExample]:
    return [
        GSM8KExample(
            id=f"gsm8k/{split}/{idx:06d}",
            split=split,  # type: ignore[arg-type]
            question=f"q{idx}",
            gold_solution=f"work #### {idx}",
            gold_answer=str(idx),
            source="openai/gsm8k",
            metadata={"dataset_revision": "rev"},
        )
        for idx in range(n)
    ]


def test_extract_gold_answer_after_hashes() -> None:
    assert extract_gold_answer("Reasoning\n#### 1,234") == "1,234"


def test_normalize_numeric_answer_commas_currency_decimals_fractions() -> None:
    assert normalize_numeric_answer("$1,200.00.") == "1200"
    assert normalize_numeric_answer("\u00a31,200") == "1200"
    assert normalize_numeric_answer("\u20ac1,200") == "1200"
    assert normalize_numeric_answer("\u00a51,200") == "1200"
    assert normalize_numeric_answer(" 2 / 4 ") == "0.5"
    assert normalize_numeric_answer(".50") == "0.5"


def test_parse_model_answer_answer_tags() -> None:
    parsed = parse_model_answer("work <answer> 18 </answer>", mode="strict")

    assert parsed.parse_ok is True
    assert parsed.answer == "18"
    assert parsed.parser == "answer_tag"


def test_parse_model_answer_boxed() -> None:
    parsed = parse_model_answer(r"therefore \boxed{42}", mode="strict")

    assert parsed.parse_ok is True
    assert parsed.answer == "42"
    assert parsed.parser == "boxed"


def test_parse_model_answer_lenient_last_number() -> None:
    parsed = parse_model_answer("try 7, then final value 9", mode="lenient")

    assert parsed.parse_ok is True
    assert parsed.answer == "9"
    assert parsed.parser == "last_number"


def test_parse_model_answer_rejects_conflicting_tagged_answers() -> None:
    parsed = parse_model_answer("<answer>1</answer><answer>2</answer>", mode="strict")

    assert parsed.parse_ok is False
    assert parsed.error == "multiple conflicting answers"


def test_verify_answer_exact_numeric_equivalence() -> None:
    result = verify_answer("2/4", "0.5")

    assert result.correct is True
    assert result.predicted_canonical == "0.5"
    assert result.gold_canonical == "0.5"


def test_verify_answer_fails_closed_on_nan_inf() -> None:
    result = verify_answer("NaN", "1")

    assert result.correct is False
    assert result.predicted_canonical is None
    assert result.error == "non-finite number"


def test_make_splits_deterministic() -> None:
    train = _examples("train", 20)

    first = make_gsm8k_splits(train, seed=1337, train_pool_size=10, dev_promotion_size=5)
    second = make_gsm8k_splits(train, seed=1337, train_pool_size=10, dev_promotion_size=5)

    assert first.splits == second.splits
    assert first.split_hash == second.split_hash


def test_make_splits_fails_when_requested_sizes_exceed_available_train_ids() -> None:
    with pytest.raises(ValueError, match="requested 6 train split examples"):
        make_gsm8k_splits(
            _examples("train", 5),
            seed=1337,
            train_pool_size=3,
            dev_promotion_size=3,
        )


def test_official_val_never_in_train_pool() -> None:
    splits = make_gsm8k_splits(
        _examples("train", 20),
        seed=1,
        train_pool_size=10,
        dev_promotion_size=5,
        official_examples=_examples("test", 5),
    )

    assert not set(splits.splits["official_val"]) & set(splits.splits["train_pool"])


def test_overlapping_official_examples_fail_closed() -> None:
    train = _examples("train", 5)

    with pytest.raises(ValueError, match="overlap"):
        make_gsm8k_splits(train, seed=1, official_examples=[train[0]])
