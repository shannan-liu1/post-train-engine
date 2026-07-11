"""GSM8K task loading, parsing, verification, and split artifacts."""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass, field
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any, Literal


SplitName = Literal["train", "test"]
PromptStyle = Literal["plain", "chat", "thinking_tags"]
ParseMode = Literal["strict", "lenient"]

_ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
_HASH_ANSWER_RE = re.compile(r"####\s*([^\n\r]+)")
_BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
_ANSWER_IS_RE = re.compile(
    r"(?:final\s+)?answer\s+is\s*[:=]?\s*([-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:\s*/\s*[-+]?\d[\d,]*)?)",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"(?<![\w/])[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:\s*/\s*[-+]?\d[\d,]*)?(?![\w/])")
_CURRENCY_RE = re.compile(r"[$\u00a3\u20ac\u00a5]")
_NON_FINITE_RE = re.compile(r"(?i)\b(?:nan|inf|infinity)\b")


@dataclass(frozen=True)
class GSM8KExample:
    id: str
    split: SplitName
    question: str
    gold_solution: str
    gold_answer: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParsedAnswer:
    parse_ok: bool
    answer: str | None
    parser: str | None
    confidence: Literal["strict", "lenient", "none"]
    error: str | None = None


@dataclass(frozen=True)
class VerificationResult:
    correct: bool
    predicted_canonical: str | None
    gold_canonical: str
    verifier: str
    error: str | None = None


@dataclass(frozen=True)
class GSM8KSplits:
    dataset: str
    dataset_revision: str
    seed: int
    splits: dict[str, list[str]]
    split_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_gsm8k(
    split: SplitName,
    dataset_name: str = "openai/gsm8k",
    revision: str | None = None,
) -> list[GSM8KExample]:
    """Load GSM8K through Hugging Face Datasets.

    The import is intentionally lazy so parser/verifier tests do not need the
    datasets stack or network access.
    """

    from datasets import load_dataset

    dataset = load_dataset(dataset_name, "main", split=split, revision=revision)
    dataset_version = getattr(dataset.info, "version", None)
    dataset_revision = (
        revision
        if revision is not None
        else str(dataset_version) if dataset_version is not None else "unknown"
    )
    examples: list[GSM8KExample] = []
    for idx, row in enumerate(dataset):
        gold_solution = str(row["answer"])
        examples.append(
            GSM8KExample(
                id=f"gsm8k/{split}/{idx:06d}",
                split=split,
                question=str(row["question"]),
                gold_solution=gold_solution,
                gold_answer=extract_gold_answer(gold_solution),
                source=dataset_name,
                metadata={
                    "dataset_revision": dataset_revision,
                    "dataset_requested_revision": revision,
                },
            )
        )
    return examples


def extract_gold_answer(gold_solution: str) -> str:
    """Extract answer after GSM8K's #### marker."""

    match = _HASH_ANSWER_RE.search(gold_solution)
    if not match:
        raise ValueError("GSM8K gold solution is missing #### answer marker")
    return match.group(1).strip()


def normalize_numeric_answer(text: str) -> str:
    """Normalize common numeric answer spellings into a canonical string."""

    value = _number_value(text)
    if value.denominator == 1:
        return str(value.numerator)
    decimal = _finite_decimal_string(value)
    if decimal is not None:
        return decimal
    return f"{value.numerator}/{value.denominator}"


def parse_model_answer(text: str, mode: ParseMode = "strict") -> ParsedAnswer:
    """Parse a final answer from model text.

    Strict order:
      1. <answer>...</answer>
      2. #### ...
      3. \\boxed{...}

    Lenient fallback:
      4. final 'answer is ...'
      5. last standalone number
    """

    if mode not in {"strict", "lenient"}:
        raise ValueError(f"unknown parse mode: {mode}")
    if not text or not text.strip():
        return ParsedAnswer(False, None, None, "none", "empty answer")
    if _NON_FINITE_RE.search(text):
        return ParsedAnswer(False, None, None, "none", "non-finite number")

    strict_attempts = (
        ("answer_tag", _ANSWER_TAG_RE.findall(text)),
        ("hash_marker", _HASH_ANSWER_RE.findall(text)),
        ("boxed", _BOXED_RE.findall(text)),
    )
    for parser, matches in strict_attempts:
        parsed = _parse_candidates(matches, parser=parser, confidence="strict")
        if parsed is not None:
            return parsed

    if mode == "lenient":
        answer_is = _ANSWER_IS_RE.findall(text)
        parsed = _parse_candidates(answer_is, parser="answer_is", confidence="lenient")
        if parsed is not None:
            return parsed

        numbers = _NUMBER_RE.findall(text)
        if numbers:
            parsed = _parse_candidates(
                [numbers[-1]],
                parser="last_number",
                confidence="lenient",
            )
            if parsed is not None:
                return parsed

    return ParsedAnswer(False, None, None, "none", "no parseable answer")


def verify_answer(predicted: str, gold: str) -> VerificationResult:
    """Exact numeric equivalence after normalization. Fail closed on ambiguity."""

    verifier = "gsm8k_numeric_v1"
    try:
        gold_value = _number_value(gold)
        gold_canonical = normalize_numeric_answer(gold)
    except ValueError as exc:
        return VerificationResult(False, None, "", verifier, f"invalid gold answer: {exc}")

    if not predicted or not predicted.strip():
        return VerificationResult(False, None, gold_canonical, verifier, "empty answer")

    try:
        predicted_value = _number_value(predicted)
        predicted_canonical = normalize_numeric_answer(predicted)
    except ValueError as exc:
        return VerificationResult(False, None, gold_canonical, verifier, str(exc))

    return VerificationResult(
        correct=predicted_value == gold_value,
        predicted_canonical=predicted_canonical,
        gold_canonical=gold_canonical,
        verifier=verifier,
    )


def format_prompt(question: str, prompt_style: PromptStyle) -> str:
    if prompt_style == "plain":
        return (
            f"Problem:\n{question}\n\n"
            "Solve the problem. Put only the final answer inside <answer>...</answer>.\n"
            "Assistant:\n"
        )
    if prompt_style == "chat":
        return (
            "<|im_start|>user\n"
            f"{question}\n\nPut only the final answer inside <answer>...</answer>."
            "\n<|im_end|>\n<|im_start|>assistant\n"
        )
    if prompt_style == "thinking_tags":
        return (
            f"Problem:\n{question}\n\n"
            "Think through the problem, then put only the final answer inside "
            "<answer>...</answer>.\nAssistant:\n"
        )
    raise ValueError(f"unknown prompt_style: {prompt_style}")


def make_gsm8k_splits(
    train_examples: list[GSM8KExample],
    seed: int,
    train_pool_size: int = 6000,
    dev_promotion_size: int = 1000,
    dev_diagnostics_size: int | None = None,
    *,
    official_examples: list[GSM8KExample] | None = None,
    dataset: str = "openai/gsm8k",
    dataset_revision: str = "unknown",
) -> GSM8KSplits:
    """Create deterministic train/dev split IDs.

    Official test examples are represented only under ``official_val`` and are
    never eligible for train-pool or curriculum selection.
    """

    if train_pool_size < 0 or dev_promotion_size < 0:
        raise ValueError("split sizes must be non-negative")
    if dev_diagnostics_size is not None and dev_diagnostics_size < 0:
        raise ValueError("dev_diagnostics_size must be non-negative")
    official_examples = official_examples or []
    _assert_unique_ids(train_examples)
    _assert_unique_ids(official_examples)
    _assert_disjoint_official(train_examples, official_examples)

    ids = [example.id for example in train_examples]
    required = train_pool_size + dev_promotion_size
    if dev_diagnostics_size is not None:
        required += dev_diagnostics_size
    if required > len(ids):
        raise ValueError(
            f"requested {required} train split examples but only {len(ids)} are available"
        )
    shuffled = ids[:]
    random.Random(seed).shuffle(shuffled)

    train_pool = shuffled[:train_pool_size]
    promotion_start = train_pool_size
    promotion_end = promotion_start + dev_promotion_size
    dev_promotion = shuffled[promotion_start:promotion_end]
    remaining = shuffled[promotion_end:]
    dev_diagnostics = (
        remaining if dev_diagnostics_size is None else remaining[:dev_diagnostics_size]
    )
    splits = {
        "train_pool": train_pool,
        "dev_promotion": dev_promotion,
        "dev_diagnostics": dev_diagnostics,
        "official_val": [example.id for example in official_examples],
    }
    body = {
        "dataset": dataset,
        "dataset_revision": dataset_revision,
        "seed": seed,
        "splits": splits,
    }
    split_hash = "sha256:" + hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return GSM8KSplits(
        dataset=dataset,
        dataset_revision=dataset_revision,
        seed=seed,
        splits=splits,
        split_hash=split_hash,
    )


def _parse_candidates(
    matches: list[str],
    *,
    parser: str,
    confidence: Literal["strict", "lenient"],
) -> ParsedAnswer | None:
    cleaned = [match.strip() for match in matches if match and match.strip()]
    if not cleaned:
        return None
    canonical: list[str] = []
    for candidate in cleaned:
        if _NON_FINITE_RE.search(candidate):
            return ParsedAnswer(False, None, parser, "none", "non-finite number")
        try:
            canonical.append(normalize_numeric_answer(candidate))
        except ValueError:
            return ParsedAnswer(False, None, parser, "none", "parser_issue")
    if len(set(canonical)) > 1:
        return ParsedAnswer(False, None, parser, "none", "multiple conflicting answers")
    return ParsedAnswer(True, canonical[0], parser, confidence)


def _number_value(text: str) -> Fraction:
    cleaned = _clean_number_text(text)
    if not cleaned:
        raise ValueError("empty numeric answer")
    if _NON_FINITE_RE.fullmatch(cleaned):
        raise ValueError("non-finite number")
    if "/" in cleaned:
        numerator, denominator = [part.strip() for part in cleaned.split("/", 1)]
        if not numerator or not denominator:
            raise ValueError("ambiguous fraction")
        try:
            value = Fraction(Decimal(numerator)) / Fraction(Decimal(denominator))
        except (InvalidOperation, ZeroDivisionError) as exc:
            raise ValueError("invalid fraction") from exc
    else:
        try:
            value = Fraction(Decimal(cleaned))
        except InvalidOperation as exc:
            raise ValueError(f"invalid numeric answer: {text!r}") from exc
    return value


def _clean_number_text(text: str) -> str:
    cleaned = str(text).strip()
    if not cleaned:
        return ""
    if _NON_FINITE_RE.search(cleaned):
        return cleaned
    cleaned = _CURRENCY_RE.sub("", cleaned)
    cleaned = cleaned.replace(",", "")
    cleaned = re.sub(r"\s+", "", cleaned)
    cleaned = cleaned.rstrip(".")
    if not re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:/[-+]?\d+(?:\.\d*)?)?", cleaned):
        raise ValueError(f"ambiguous numeric answer: {text!r}")
    return cleaned


def _finite_decimal_string(value: Fraction) -> str | None:
    denominator = value.denominator
    while denominator % 2 == 0:
        denominator //= 2
    while denominator % 5 == 0:
        denominator //= 5
    if denominator != 1:
        return None
    decimal = Decimal(value.numerator) / Decimal(value.denominator)
    normalized = decimal.normalize()
    if normalized == normalized.to_integral():
        return str(normalized.quantize(Decimal(1)))
    return format(normalized, "f").rstrip("0").rstrip(".")


def _assert_disjoint_official(
    train_examples: list[GSM8KExample],
    official_examples: list[GSM8KExample],
) -> None:
    train_ids = {example.id for example in train_examples}
    official_ids = {example.id for example in official_examples}
    overlap = train_ids & official_ids
    if overlap:
        first = sorted(overlap)[0]
        raise ValueError(f"official examples overlap train examples: {first}")


def _assert_unique_ids(examples: list[GSM8KExample]) -> None:
    ids = [example.id for example in examples]
    if len(ids) == len(set(ids)):
        return
    seen: set[str] = set()
    for example_id in ids:
        if example_id in seen:
            raise ValueError(f"duplicate GSM8K example id: {example_id}")
        seen.add(example_id)
