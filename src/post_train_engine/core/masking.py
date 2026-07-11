"""Prompt-token masking for Supervised Fine-Tuning.

In SFT, the training data is ``(prompt, response)`` pairs concatenated
into a single token sequence. We want the model to learn to predict
the *response* tokens conditioned on the prompt, but not to learn to
predict the prompt tokens themselves (those are inputs, not targets).

This is implemented via the ``labels`` tensor passed alongside
``input_ids``. Cross-entropy ignores positions whose label equals
``-100`` (PyTorch's standard convention for the ignore index). So we
build ``labels`` as a copy of ``input_ids`` with prompt positions
replaced by ``-100``. The loss then only flows through response
positions.

Padding tokens, when present, are also masked so we don't compute
loss on padding. The caller passes the tokenizer's ``attention_mask``
to opt into padding masking; without it, only the prompt is masked.
"""

from __future__ import annotations

import torch

# PyTorch's standard "ignore this position" sentinel for the targets
# of nn.CrossEntropyLoss / F.cross_entropy. Hard-coded into the C++
# kernels; do not change.
IGNORE_INDEX = -100


def mask_prompt_tokens(
    input_ids: torch.Tensor,
    prompt_lengths: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build a ``labels`` tensor with prompt (and optionally padding)
    positions set to ``IGNORE_INDEX``, response positions kept as their
    token IDs.

    Parameters
    ----------
    input_ids
        Shape ``(batch, seq_len)``. The full ``prompt + response``
        token IDs for each example in the batch.
    prompt_lengths
        Shape ``(batch,)``. The length of the prompt portion for each
        example. ``labels[i, :prompt_lengths[i]]`` will be set to
        ``IGNORE_INDEX``; the remainder keeps the original token IDs.
    attention_mask
        Optional shape ``(batch, seq_len)``. If provided, positions
        where ``attention_mask == 0`` are also set to ``IGNORE_INDEX``
        (those are padding tokens we don't want to learn to predict).

    Returns
    -------
    A new tensor (``input_ids`` is not mutated) of shape
    ``(batch, seq_len)`` containing token IDs at response positions and
    ``IGNORE_INDEX`` everywhere else.
    """
    # clone() so the caller's input_ids is untouched. The function is
    # supposed to return a fresh labels tensor; mutating input_ids
    # in place would be a footgun.
    labels = input_ids.clone()

    # Vectorized prompt mask:
    #   positions:        [[0, 1, 2, ..., seq_len-1]]                    shape (1, seq_len)
    #   prompt_lengths:   [[L0], [L1], ..., [L_{batch-1}]]                shape (batch, 1)
    #   broadcast compare:positions < prompt_lengths is True for the
    #                     prompt portion of each example                  shape (batch, seq_len)
    seq_len = input_ids.size(1)
    positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
    is_prompt = positions < prompt_lengths.unsqueeze(1)
    labels[is_prompt] = IGNORE_INDEX

    if attention_mask is not None:
        # Padding tokens: set wherever attention_mask is 0.
        labels[attention_mask == 0] = IGNORE_INDEX

    return labels
