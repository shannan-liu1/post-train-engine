"""Token log-probabilities for causal language models.

This module exists to hold one helper - ``sequence_log_probs``
- and the long explanation of *why* it is written the way it is. Both
post-training objectives need the same quantity: for each
response position ``t`` in a causal-LM batch, the log-probability the
current model assigns to the next ground-truth token.

The "obvious" way to compute that quantity, and the "fast" way, are
numerically identical but allocate very different amounts of VRAM at
the scales we care about. The fast way uses ``F.cross_entropy`` as a
fused kernel. Reading the comment block below once should make the
fast version intelligible - after that, both call sites can stay
short and the maths only needs to be re-derived if the helper is ever
rewritten.


The math: token log-probability via fused cross-entropy
========================================================

At each response position ``t``, a causal LM emits a logits vector
``z[t] in R^V`` (one real number per vocabulary token). The
conditional probability the model assigns to producing token ``y``
given the prefix up to position ``t`` is

    p(y | prefix) = softmax(z[t])[y]
                  = exp(z[t][y]) / sum_w exp(z[t][w]).

Taking the natural log:

    log p(y | prefix) = z[t][y] - log sum_w exp(z[t][w])         (1)
                      = log_softmax(z[t])[y].

The right-hand side of (1) is the ``y``-th entry of the log-softmax
of the logits at position ``t``. That is the per-token log-probability
both DPO and GRPO need.


The naive implementation
------------------------

The most direct PyTorch translation of (1) is:

    full = F.log_softmax(logits, dim=-1)                # (B, S, V)
    token_logps = full.gather(-1, labels.unsqueeze(-1)) # (B, S, 1)
    token_logps = token_logps.squeeze(-1)               # (B, S)

This works and is mathematically correct. It also allocates the
intermediate tensor ``full``, which has exactly the same shape as
``logits``. For Qwen3-4B (vocab about 152k) at batch 4 x seq 1024 in
bfloat16 that is

    4 * 1024 * 152_000 * 2 bytes ~= 1.24 GB

of VRAM that gets allocated, gathered from, and freed every forward
pass. At Qwen2.5-1.5B (vocab about 152k) it is the same per-token cost,
just multiplied by a smaller per-step batch.

There is a second nuisance with the naive form: ``gather`` does not
have an ``ignore_index`` parameter. So if ``labels`` contains
``IGNORE_INDEX = -100`` for prompt/padding positions (which it does
in this codebase), the gather call fails - ``-100`` is out of bounds
for a vocab of size V. The standard workaround is

    safe = labels.masked_fill(labels == IGNORE_INDEX, 0)
    token_logps = full.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    token_logps = token_logps * (labels != IGNORE_INDEX)

- which clones ``labels`` and adds two extra ops.


The fused implementation
------------------------

PyTorch's ``F.cross_entropy`` computes, by definition,

    cross_entropy(z, y) = - log p(y | prefix)                    (2)
                        = - log_softmax(z)[y].

So the per-token log-probability is just ``-F.cross_entropy(...)``.

What makes this useful is *how* the kernel computes it. Rather than
forming the full ``log_softmax`` tensor and then gathering from it,
``cross_entropy`` streams through the vocabulary dimension twice:
once to find the maximum logit (for numerical stability), once to
accumulate the log-sum-exp. It then subtracts the target logit from
the log-sum-exp in a single pass. Peak intermediate memory is the
per-position scalar normalizer, not a full (B, S, V) tensor. Memory
usage drops from O(B*S*V) to O(B*S).

The kernel also takes an ``ignore_index`` parameter directly. With
``reduction='none'`` and ``ignore_index=IGNORE_INDEX``, positions
whose target equals the ignore index contribute exactly ``0`` to the
output. That replaces the ``masked_fill`` + post-multiply dance from
the naive form: we can feed labels containing ``-100`` directly into
the kernel and get a result tensor that is already zero at those
positions.

Net effect of the swap: same numerical answer (within float
precision), much less peak VRAM at large vocab, a tiny throughput
win from fewer kernel launches, and slightly less code.


A worked check at small scale
-----------------------------

For ``logits = [[2.0, 1.0]]`` and ``labels = [[1]]``:

    log_softmax([2.0, 1.0])
      = [2.0 - log(e^2 + e^1), 1.0 - log(e^2 + e^1)]
      ~= [-0.3133, -1.3133]

    naive form:   gather index 1 -> -1.3133
    fused form:  -cross_entropy([2.0, 1.0], 1)
                    = -1.3133

Same answer. The fused form simply never builds the [-0.3133,
-1.3133] vector explicitly.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from post_train_engine.core.masking import IGNORE_INDEX


def sequence_log_probs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return causal-LM per-token log-probabilities and the response mask.

    See this module's docstring for the derivation. The short version:
    ``-F.cross_entropy(logits, labels, reduction='none')`` is the
    per-token log-probability, computed in a fused kernel that never
    materializes the full ``log_softmax`` tensor.

    Parameters
    ----------
    logits
        Shape ``(batch, seq, vocab)``. Raw model output for each
        position. The function shifts internally; pass the *full*
        logits, not pre-shifted.
    labels
        Shape ``(batch, seq)``. Ground-truth token IDs at response
        positions, ``IGNORE_INDEX`` at prompt and padding positions.
        Same convention as the SFT trainer.
    ignore_index
        Positions where ``labels == ignore_index`` are excluded from
        the output (contribute ``0.0`` to ``token_logps`` and
        ``False`` to ``response_mask``).

    Returns
    -------
    token_logps : torch.Tensor
        Shape ``(batch, seq - 1)``. ``token_logps[i, t]`` is
        ``log p(labels[i, t + 1] | prefix[i, : t])`` at response
        positions, exactly ``0.0`` at ignored positions.
    response_mask : torch.Tensor
        Shape ``(batch, seq - 1)`` of bool. ``True`` at response
        positions, ``False`` at ignored positions. Same shape as
        ``token_logps``, returned so callers can do per-row reductions
        without recomputing it.
    """
    if logits.ndim != 3:
        raise ValueError(
            f"logits must have shape (batch, seq, vocab); got {tuple(logits.shape)}"
        )
    if labels.ndim != 2:
        raise ValueError(f"labels must have shape (batch, seq); got {tuple(labels.shape)}")
    if logits.shape[:2] != labels.shape:
        raise ValueError(
            "logits and labels batch/sequence dimensions must match: "
            f"{tuple(logits.shape[:2])} vs {tuple(labels.shape)}"
        )
    if logits.size(1) < 2:
        raise ValueError(
            "sequence_log_probs requires sequence length >= 2"
        )

    # Causal-LM shift: the model predicts token at position t+1 from
    # logits at position t. So position-t logits pair with
    # position-(t+1) labels. Both tensors lose one position; both end
    # up with sequence length (seq - 1).
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    response_mask = shifted_labels != ignore_index

    # F.cross_entropy expects a 2-D input of shape (N, C) where C is
    # the vocab dim, paired with a 1-D target of shape (N,). We reshape
    # (batch, seq, vocab) -> (batch * seq, vocab) for the call and back
    # again for the output. ``reshape`` is a memory-free view here
    # because the slice ``[:, :-1, :]`` of a contiguous tensor stays
    # contiguous in the standard sense (it trims the END of dim 1, not
    # the start, so the per-row vocab stride is unchanged).
    batch_size = shifted_logits.size(0)
    seq_len_minus_1 = shifted_logits.size(1)
    vocab_size = shifted_logits.size(2)
    nll = F.cross_entropy(
        shifted_logits.reshape(-1, vocab_size),
        shifted_labels.reshape(-1),
        reduction="none",
        ignore_index=ignore_index,
    ).reshape(batch_size, seq_len_minus_1)

    # Cross-entropy is -log p; per-token log-probability is its
    # negation. At positions where labels == ignore_index the kernel
    # returns 0, so ``token_logps`` is exactly 0 there - no separate
    # masked_fill is needed.
    token_logps = -nll
    return token_logps, response_mask
