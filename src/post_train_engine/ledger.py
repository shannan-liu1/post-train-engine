"""Run ledger helpers for reproducible climb artifacts."""

from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from post_train_engine.jsonl import append_jsonl
from post_train_engine.tasks.gsm8k import verify_answer


@dataclass(frozen=True)
class RunLedgerEntry:
    run_id: str
    parent_run_id: str | None
    git_sha: str
    command: str
    base_model: str | None
    base_model_revision: str | None
    checkpoint: str | None
    dataset: str | None
    dataset_revision: str | None
    split_hash: str | None
    prompt_template_hash: str
    verifier_hash: str
    reward_config_hash: str
    trainer_config_hash: str
    seeds: tuple[int, ...]
    artifact_paths: Mapping[str, str] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        body = asdict(self)
        body["seeds"] = list(self.seeds)
        body["artifact_paths"] = dict(self.artifact_paths)
        return body


def make_run_ledger_entry(
    *,
    run_id: str,
    command: Sequence[str],
    artifact_paths: Mapping[str, str],
    parent_run_id: str | None = None,
    base_model: str | None = None,
    base_model_revision: str | None = None,
    checkpoint: str | None = None,
    dataset: str | None = None,
    dataset_revision: str | None = None,
    split_hash: str | None = None,
    prompt_template: str = "",
    reward_config: Mapping[str, Any] | None = None,
    trainer_config: Mapping[str, Any] | None = None,
    seeds: Sequence[int] = (),
) -> RunLedgerEntry:
    return RunLedgerEntry(
        run_id=run_id,
        parent_run_id=parent_run_id,
        git_sha=current_git_sha(),
        command=" ".join(command),
        base_model=base_model,
        base_model_revision=base_model_revision,
        checkpoint=checkpoint,
        dataset=dataset,
        dataset_revision=dataset_revision,
        split_hash=split_hash,
        prompt_template_hash=stable_hash(prompt_template),
        verifier_hash=stable_hash(inspect.getsource(verify_answer)),
        reward_config_hash=stable_hash(reward_config or {}),
        trainer_config_hash=stable_hash(trainer_config or {}),
        seeds=tuple(seeds),
        artifact_paths=dict(artifact_paths),
    )


def write_ledger_entry(path: str | Path, entry: RunLedgerEntry) -> None:
    append_jsonl(path, entry.to_json())


def stable_hash(value: Any) -> str:
    if isinstance(value, str):
        payload = value.encode("utf-8")
    else:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def current_git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip() or "unknown"
