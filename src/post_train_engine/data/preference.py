"""Preference-pair dataset and collation for DPO-style training."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from post_train_engine.core.masking import IGNORE_INDEX
from post_train_engine.data.sft_dataset import (
    _tokenize_to_ids,
    serialize_prompt,
    serialize_response,
)


@dataclass(frozen=True)
class PreferenceExample:
    prompt: str
    chosen: str
    rejected: str
    source: str
    prompt_id: str | None = None
    chosen_grade: dict[str, Any] | None = None
    rejected_grade: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.weight <= 0.0:
            raise ValueError("preference weight must be strictly positive")


@dataclass(frozen=True)
class TokenizedPreferenceExample:
    chosen_input_ids: torch.Tensor
    chosen_attention_mask: torch.Tensor
    chosen_labels: torch.Tensor
    rejected_input_ids: torch.Tensor
    rejected_attention_mask: torch.Tensor
    rejected_labels: torch.Tensor
    source: str
    prompt_id: str | None = None
    ref_chosen_logp: float | None = None
    ref_rejected_logp: float | None = None
    weight: float = 1.0


def load_preference_pairs(path: str | Path) -> list[PreferenceExample]:
    path = Path(path)
    examples: list[PreferenceExample] = []
    with path.open("r", encoding="utf-8") as fp:
        for line_no, line in enumerate(fp, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            raw = json.loads(stripped)
            try:
                metadata = raw.get("metadata") or {}
                examples.append(
                    PreferenceExample(
                        prompt=raw["prompt"],
                        chosen=raw["chosen"],
                        rejected=raw["rejected"],
                        source=raw["source"],
                        prompt_id=raw.get("prompt_id"),
                        chosen_grade=raw.get("chosen_grade"),
                        rejected_grade=raw.get("rejected_grade"),
                        metadata=metadata,
                        weight=float(raw.get("weight", metadata.get("weight", 1.0))),
                    )
                )
            except KeyError as exc:
                raise ValueError(f"{path}:{line_no} missing required field {exc}") from exc
    if not examples:
        raise ValueError(f"no preference pairs found in {path}")
    return examples


class PreferenceDataset(Dataset):
    def __init__(self, pairs_path: str | Path) -> None:
        self.pairs_path = Path(pairs_path)
        self.examples = load_preference_pairs(self.pairs_path)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> PreferenceExample:
        return self.examples[idx]


def tokenize_preference_pairs(
    examples: list[PreferenceExample],
    *,
    tokenizer: Any,
    max_seq_len: int,
) -> list[TokenizedPreferenceExample]:
    return [
        TokenizedPreferenceExample(
            **_encode_pair(example, tokenizer=tokenizer, max_seq_len=max_seq_len),
            source=example.source,
            prompt_id=example.prompt_id,
            weight=example.weight,
        )
        for example in examples
    ]


def _encode_pair(
    example: PreferenceExample,
    *,
    tokenizer: Any,
    max_seq_len: int,
) -> dict[str, torch.Tensor]:
    chosen = _encode_side(
        tokenizer=tokenizer,
        prompt=example.prompt,
        response=example.chosen,
        max_seq_len=max_seq_len,
    )
    rejected = _encode_side(
        tokenizer=tokenizer,
        prompt=example.prompt,
        response=example.rejected,
        max_seq_len=max_seq_len,
    )
    return {
        "chosen_input_ids": chosen["input_ids"],
        "chosen_attention_mask": chosen["attention_mask"],
        "chosen_labels": chosen["labels"],
        "rejected_input_ids": rejected["input_ids"],
        "rejected_attention_mask": rejected["attention_mask"],
        "rejected_labels": rejected["labels"],
    }


def _encode_side(
    *,
    tokenizer: Any,
    prompt: str,
    response: str,
    max_seq_len: int,
) -> dict[str, torch.Tensor]:
    prompt_ids = _tokenize_to_ids(tokenizer, serialize_prompt(prompt))
    response_ids = _tokenize_to_ids(tokenizer, serialize_response(response))
    input_ids = (prompt_ids + response_ids)[:max_seq_len]
    prompt_length = min(len(prompt_ids), len(input_ids))
    if len(input_ids) <= prompt_length:
        raise ValueError("preference example has no response tokens after truncation")
    input_tensor = torch.tensor(input_ids, dtype=torch.long)
    labels = input_tensor.clone()
    labels[:prompt_length] = IGNORE_INDEX
    if (labels[1:] != IGNORE_INDEX).sum().item() == 0:
        raise ValueError("preference example has no shifted response labels")
    return {
        "input_ids": input_tensor,
        "attention_mask": torch.ones_like(input_tensor),
        "labels": labels,
    }


class PreferenceCollator:
    def __init__(self, *, pad_token_id: int = 0) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, examples: list[TokenizedPreferenceExample]) -> dict[str, Any]:
        chosen = self._pad(
            [
                {
                    "input_ids": ex.chosen_input_ids,
                    "attention_mask": ex.chosen_attention_mask,
                    "labels": ex.chosen_labels,
                }
                for ex in examples
            ]
        )
        rejected = self._pad(
            [
                {
                    "input_ids": ex.rejected_input_ids,
                    "attention_mask": ex.rejected_attention_mask,
                    "labels": ex.rejected_labels,
                }
                for ex in examples
            ]
        )
        return {
            "chosen_input_ids": chosen["input_ids"],
            "chosen_attention_mask": chosen["attention_mask"],
            "chosen_labels": chosen["labels"],
            "rejected_input_ids": rejected["input_ids"],
            "rejected_attention_mask": rejected["attention_mask"],
            "rejected_labels": rejected["labels"],
            "prompt_ids": [ex.prompt_id for ex in examples],
            "sources": [ex.source for ex in examples],
            "sample_weights": torch.tensor(
                [float(ex.weight) for ex in examples],
                dtype=torch.float32,
            ),
        }

    def _pad(self, rows: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        width = max(len(row["input_ids"]) for row in rows)
        input_rows: list[torch.Tensor] = []
        attention_rows: list[torch.Tensor] = []
        label_rows: list[torch.Tensor] = []
        for row in rows:
            pad_len = width - len(row["input_ids"])
            input_rows.append(
                torch.cat(
                    [
                        row["input_ids"],
                        torch.full((pad_len,), self.pad_token_id, dtype=torch.long),
                    ]
                )
            )
            attention_rows.append(
                torch.cat([row["attention_mask"], torch.zeros((pad_len,), dtype=torch.long)])
            )
            label_rows.append(
                torch.cat(
                    [
                        row["labels"],
                        torch.full((pad_len,), IGNORE_INDEX, dtype=torch.long),
                    ]
                )
            )
        return {
            "input_ids": torch.stack(input_rows),
            "attention_mask": torch.stack(attention_rows),
            "labels": torch.stack(label_rows),
        }
