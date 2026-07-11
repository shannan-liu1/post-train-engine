"""Resolve mutable Hugging Face references to immutable commits."""

from __future__ import annotations

import re
from typing import Any, Literal

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def resolve_huggingface_revision(
    repo_id: str,
    *,
    kind: Literal["model", "dataset"],
    requested_revision: str,
    api: Any | None = None,
) -> str:
    if not repo_id or not requested_revision:
        raise ValueError("Hub repo_id and requested_revision must be non-empty")
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    info = (
        api.model_info(repo_id, revision=requested_revision)
        if kind == "model"
        else api.dataset_info(repo_id, revision=requested_revision)
    )
    sha = str(getattr(info, "sha", "") or "").lower()
    if not _COMMIT_RE.fullmatch(sha):
        raise ValueError(
            f"Hugging Face {kind} {repo_id!r} did not resolve to an immutable commit SHA"
        )
    return sha


def is_huggingface_commit(value: str | None) -> bool:
    return bool(value and _COMMIT_RE.fullmatch(value.lower()))


__all__ = ["is_huggingface_commit", "resolve_huggingface_revision"]
