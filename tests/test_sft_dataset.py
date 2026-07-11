from __future__ import annotations

import pytest
import torch

from post_train_engine.data.sft_dataset import PackingCollator, TokenizedSFTExample


def test_packing_collator_rejects_empty_batches() -> None:
    collator = PackingCollator(
        max_seq_len=8,
        eos_token_id=0,
        isolate_documents=False,
    )

    with pytest.raises(ValueError, match="PackingCollator requires at least one example"):
        collator([])


def test_packing_collator_rejects_prompt_only_rows_after_truncation() -> None:
    example = TokenizedSFTExample(
        input_ids=torch.tensor([101, 102, 103, 104], dtype=torch.long),
        prompt_length=4,
        source="toy",
        example_id="ex-1",
    )
    collator = PackingCollator(
        max_seq_len=3,
        eos_token_id=0,
        isolate_documents=False,
    )

    with pytest.raises(ValueError, match="no shifted response labels"):
        collator([example])


def test_packing_collator_keeps_valid_response_labels() -> None:
    example = TokenizedSFTExample(
        input_ids=torch.tensor([101, 102, 201, 202], dtype=torch.long),
        prompt_length=2,
        source="toy",
        example_id="ex-1",
    )
    collator = PackingCollator(
        max_seq_len=8,
        eos_token_id=0,
        isolate_documents=False,
    )

    batch = collator([example])

    assert batch["labels"].tolist() == [[-100, -100, 201, 202]]
