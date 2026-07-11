"""Dataset and sequence-packing primitives for supervised fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset

from post_train_engine.core.masking import IGNORE_INDEX, mask_prompt_tokens
from post_train_engine.tasks.schema import Example


@dataclass(frozen=True)
class TokenizedSFTExample:
    input_ids: torch.Tensor
    prompt_length: int
    source: str
    example_id: str | None = None


def serialize_prompt(prompt: str) -> str:
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


def serialize_response(response: str) -> str:
    return f"{response}<|im_end|>\n"


def _tokenize_to_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    input_ids = encoded["input_ids"]
    if isinstance(input_ids, torch.Tensor):
        return input_ids.flatten().tolist()
    return list(input_ids)


class SFTDataset(Dataset):
    """Torch dataset over normalized examples."""

    def __init__(self, examples: list[Example], tokenizer: Any) -> None:
        if not examples:
            raise ValueError("SFTDataset requires at least one example")
        self.examples = examples
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> TokenizedSFTExample:
        example = self.examples[idx]
        if example.response is None:
            raise ValueError(f"example {example.id} has no response")
        prompt_ids = _tokenize_to_ids(self.tokenizer, serialize_prompt(example.prompt))
        response_ids = _tokenize_to_ids(
            self.tokenizer,
            serialize_response(example.response),
        )
        return TokenizedSFTExample(
            input_ids=torch.tensor(prompt_ids + response_ids, dtype=torch.long),
            prompt_length=len(prompt_ids),
            source=example.source,
            example_id=example.id,
        )


class PackingCollator:
    """Greedily pack tokenized SFT documents into rows up to max_seq_len."""

    def __init__(
        self,
        *,
        max_seq_len: int,
        eos_token_id: int,
        isolate_documents: bool,
        pad_token_id: int = 0,
    ) -> None:
        if max_seq_len <= 1:
            raise ValueError("max_seq_len must be greater than 1")
        self.max_seq_len = max_seq_len
        self.eos_token_id = eos_token_id
        self.isolate_documents = isolate_documents
        self.pad_token_id = pad_token_id

    def __call__(self, examples: list[TokenizedSFTExample]) -> dict[str, Any]:
        if not examples:
            raise ValueError("PackingCollator requires at least one example")
        queue = [self._truncate(example) for example in examples]
        rows: list[dict[str, Any]] = []
        idx = 0
        while idx < len(queue):
            row, idx = self._pack_one_row(queue, idx)
            rows.append(row)

        width = max(len(row["input_ids"]) for row in rows)
        input_rows: list[list[int]] = []
        label_rows: list[list[int]] = []
        position_rows: list[list[int]] = []
        padding_masks: list[list[int]] = []
        boundaries: list[list[dict[str, Any]]] = []
        for row in rows:
            pad = width - len(row["input_ids"])
            input_rows.append(row["input_ids"] + [self.pad_token_id] * pad)
            label_rows.append(row["labels"] + [IGNORE_INDEX] * pad)
            position_rows.append(row["position_ids"] + [0] * pad)
            padding_masks.append([1] * len(row["input_ids"]) + [0] * pad)
            boundaries.append(row["document_boundaries"])

        batch: dict[str, Any] = {
            "input_ids": torch.tensor(input_rows, dtype=torch.long),
            "labels": torch.tensor(label_rows, dtype=torch.long),
            "position_ids": torch.tensor(position_rows, dtype=torch.long),
            "document_boundaries": boundaries,
        }
        if self.isolate_documents:
            batch["attention_mask"] = self._isolated_attention_mask(rows, width)
        else:
            batch["attention_mask"] = torch.tensor(padding_masks, dtype=torch.long)
        return batch

    def _pack_one_row(
        self,
        examples: list[TokenizedSFTExample],
        start_idx: int,
    ) -> tuple[dict[str, Any], int]:
        input_ids: list[int] = []
        labels: list[int] = []
        position_ids: list[int] = []
        doc_ids: list[int] = []
        boundaries: list[dict[str, Any]] = []
        idx = start_idx
        while idx < len(examples):
            example = examples[idx]
            needs_eos = bool(input_ids)
            additional = len(example.input_ids) + (1 if needs_eos else 0)
            if input_ids and len(input_ids) + additional > self.max_seq_len:
                break
            if needs_eos:
                input_ids.append(self.eos_token_id)
                labels.append(IGNORE_INDEX)
                position_ids.append(0)
                doc_ids.append(-1)

            ids = example.input_ids.tolist()
            doc_start = len(input_ids)
            input_ids.extend(ids)
            prompt_length = min(example.prompt_length, len(ids))
            doc_labels = mask_prompt_tokens(
                torch.tensor([ids], dtype=torch.long),
                torch.tensor([prompt_length], dtype=torch.long),
            )[0].tolist()
            labels.extend(doc_labels)
            position_ids.extend(range(len(ids)))
            doc_index = len(boundaries)
            doc_ids.extend([doc_index] * len(ids))
            boundaries.append(
                {
                    "start": doc_start,
                    "end": doc_start + len(ids),
                    "prompt_length": prompt_length,
                    "source": example.source,
                    "example_id": example.example_id,
                }
            )
            idx += 1

        return {
            "input_ids": input_ids,
            "labels": labels,
            "position_ids": position_ids,
            "doc_ids": doc_ids,
            "document_boundaries": boundaries,
        }, idx

    def _truncate(self, example: TokenizedSFTExample) -> TokenizedSFTExample:
        if len(example.input_ids) <= self.max_seq_len:
            truncated = example
        else:
            truncated = TokenizedSFTExample(
                input_ids=example.input_ids[: self.max_seq_len],
                prompt_length=min(example.prompt_length, self.max_seq_len),
                source=example.source,
                example_id=example.example_id,
            )
        if not _has_shifted_response_labels(
            input_length=len(truncated.input_ids),
            prompt_length=truncated.prompt_length,
        ):
            label = truncated.example_id or "<unknown>"
            raise ValueError(
                f"example {label} has no shifted response labels after truncation"
            )
        return truncated

    def _isolated_attention_mask(self, rows: list[dict[str, Any]], width: int) -> torch.Tensor:
        causal = torch.tril(torch.ones((width, width), dtype=torch.long))
        masks: list[torch.Tensor] = []
        for row in rows:
            doc_ids = row["doc_ids"] + [-2] * (width - len(row["doc_ids"]))
            doc_tensor = torch.tensor(doc_ids, dtype=torch.long)
            same_document = (doc_tensor.unsqueeze(0) == doc_tensor.unsqueeze(1)).long()
            real_document = (doc_tensor.unsqueeze(0) >= 0).long()
            mask = same_document * real_document * causal
            empty_queries = mask.sum(dim=-1) == 0
            if empty_queries.any():
                indices = torch.arange(width)
                mask[empty_queries, indices[empty_queries]] = 1
            masks.append(mask)
        return torch.stack(masks, dim=0).unsqueeze(1)


def _has_shifted_response_labels(*, input_length: int, prompt_length: int) -> bool:
    return input_length > max(prompt_length, 1)
