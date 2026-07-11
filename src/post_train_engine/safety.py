"""Safe wrappers around external loaders.

The functions in this module enforce defensive defaults, primarily disabling
execution of remote loader scripts and pickle-format model weights. Always call
these instead of the underlying library APIs directly. Defaults can be
overridden by passing explicit keyword arguments, but the override should be
deliberate and recorded in the change that introduces it.

See SECURITY.md for the full public policy.
"""

from __future__ import annotations

from typing import Any

# Imported under private aliases so callers cannot accidentally bypass
# the wrappers by importing the underlying functions directly.
from datasets import load_dataset as _hf_load_dataset
from transformers import AutoModelForCausalLM as _AutoModelForCausalLM
from transformers import AutoTokenizer as _AutoTokenizer


def safe_load_dataset(
    path: str,
    name: str | None = None,
    *,
    split: str | None = None,
    revision: str | None = None,
    trust_remote_code: bool = False,
    **kwargs: Any,
):
    """Load a Hugging Face dataset with defensive defaults.

    Parameters
    ----------
    path
        The dataset identifier on the Hugging Face Hub.
    name
        The dataset configuration name when the dataset requires one.
    split
        The split to load ("train", "test", or None for all).
    revision
        Git ref to pin against. For datasets that ship a Python loader
        script, pass "refs/convert/parquet" to use the auto-converted
        parquet branch instead of the script-based main branch.
    trust_remote_code
        Defaults to False; script-based datasets fail to load. Set to
        True only after auditing the loader script. Passing True must
        be deliberate; a justification belongs in the PRD or commit.
    **kwargs
        Forwarded unchanged to ``datasets.load_dataset``.

    Notes
    -----
    Both ``datasets >= 2.16`` defaults ``trust_remote_code`` to False,
    but we still pass it explicitly here. Defense in depth: if the
    upstream default ever changes, our policy is unaffected.
    """
    return _hf_load_dataset(
        path=path,
        name=name,
        split=split,
        revision=revision,
        trust_remote_code=trust_remote_code,
        **kwargs,
    )


def safe_load_model(
    pretrained_model_name_or_path: str,
    *,
    trust_remote_code: bool = False,
    use_safetensors: bool = True,
    attn_implementation: str | None = None,
    **kwargs: Any,
):
    """Load a Hugging Face causal-LM model with defensive defaults.

    Defaults disable remote-code execution and require safetensors weights
    (which cannot execute arbitrary pickle on load). Defaults can be
    overridden; the override must be deliberate.

    A small number of test models (e.g. sshleifer/tiny-gpt2) ship only
    .bin (pickle) weights. Set use_safetensors=False for those, with a
    written justification.

    Parameters
    ----------
    pretrained_model_name_or_path
        Local directory or Hugging Face Hub model identifier.
    trust_remote_code
        Defaults to False; model code execution is disallowed. Set to
        True only after auditing the model's remote code. Passing True
        must be deliberate; a justification belongs in the PRD or commit.
    use_safetensors
        Defaults to True; refuse to load pickle-format .bin weights.
        Safetensors weights cannot execute arbitrary code on load.
        Set to False only for models that ship no safetensors variant
        (e.g. test models such as sshleifer/tiny-gpt2).
    **kwargs
        Forwarded unchanged to ``AutoModelForCausalLM.from_pretrained``.

    Notes
    -----
    Both ``transformers >= 4.26`` defaults ``trust_remote_code`` to False,
    but we still pass it explicitly here. Defense in depth: if the
    upstream default ever changes, our policy is unaffected.

    ``attn_implementation`` is forwarded only when explicitly set, so the
    Hugging Face default (typically ``sdpa`` on recent transformers) is
    preserved when callers don't opt in. Set to ``"flash_attention_2"``
    in a CUDA environment with ``flash-attn`` installed for the ~1.5-2x throughput
    win on long sequences; we fall back to the HF default if the requested
    backend isn't available so a misconfigured dev box can still load.
    """
    if attn_implementation is not None:
        try:
            return _AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=trust_remote_code,
                use_safetensors=use_safetensors,
                attn_implementation=attn_implementation,
                **kwargs,
            )
        except (ImportError, ValueError) as exc:
            # flash-attn missing, or backend unsupported for this model.
            # Fall through to the default attention implementation so the
            # load still succeeds; the caller's logs will show this.
            import warnings

            warnings.warn(
                f"attn_implementation={attn_implementation!r} unavailable "
                f"({exc!r}); falling back to the HF default.",
                stacklevel=2,
            )
    return _AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path,
        trust_remote_code=trust_remote_code,
        use_safetensors=use_safetensors,
        **kwargs,
    )


def safe_load_tokenizer(
    pretrained_model_name_or_path: str,
    *,
    trust_remote_code: bool = False,
    **kwargs: Any,
):
    """Load a Hugging Face tokenizer with defensive defaults.

    Defaults disable remote-code execution. Tokenizers do not load
    weight tensors, so there is no safetensors concern, but
    trust_remote_code still matters for models with custom tokenizer
    classes that ship Python code.

    Parameters
    ----------
    pretrained_model_name_or_path
        Local directory or Hugging Face Hub model identifier.
    trust_remote_code
        Defaults to False. Set to True only after auditing the model's
        tokenizer code. Passing True must be deliberate.
    **kwargs
        Forwarded unchanged to ``AutoTokenizer.from_pretrained``.

    Notes
    -----
    Both ``transformers >= 4.26`` defaults ``trust_remote_code`` to False,
    but we still pass it explicitly here. Defense in depth: if the
    upstream default ever changes, our policy is unaffected.
    """
    return _AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        trust_remote_code=trust_remote_code,
        **kwargs,
    )
