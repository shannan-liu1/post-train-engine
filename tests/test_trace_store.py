from __future__ import annotations

from pathlib import Path

import pytest

from post_train_engine.traces import JsonlTraceStore, TraceRecord, stable_prompt_hash


def test_jsonl_trace_store_creates_parent_dirs_and_roundtrips_typed_records(
    tmp_path: Path,
) -> None:
    store = JsonlTraceStore(tmp_path / "nested" / "traces.jsonl")
    trace = TraceRecord(
        trace_id="trace-001",
        run_id="run-001",
        task_id="gsm8k",
        example_id="example-001",
        split_role="train",
        prompt_hash=stable_prompt_hash("What is 2 + 2?"),
        source_checkpoint="checkpoints/seed",
        policy_version="seed-v1",
        policy_step=0,
        policy_step_evidence="static",
        rollout_group_id="group-1",
        generation_backend="local",
        sampling_config={"temperature": 0.0},
        verifier_id="gsm8k_numeric_v1",
        reward_components={"exact": 1.0},
        token_counts={"prompt": 6, "completion": 1},
    )

    store.append(trace)

    assert (tmp_path / "nested" / "traces.jsonl").is_file()
    assert store.read_all() == [trace]


def test_trace_record_rejects_malformed_provenance_and_numeric_fields() -> None:
    base = _trace_payload()

    with pytest.raises(ValueError, match="at least 1 character"):
        TraceRecord(**{**base, "trace_id": ""})
    with pytest.raises(ValueError):
        TraceRecord(**{**base, "split_role": "test"})
    with pytest.raises(ValueError, match="reward_components"):
        TraceRecord(**{**base, "reward_components": {"exact": True}})
    with pytest.raises(ValueError, match="reward_components"):
        TraceRecord(**{**base, "reward_components": []})
    with pytest.raises(ValueError, match="reward_components"):
        TraceRecord(**{**base, "reward_components": {"exact": float("nan")}})
    with pytest.raises(ValueError, match="token_counts"):
        TraceRecord(**{**base, "token_counts": {"prompt": -1}})
    with pytest.raises(ValueError, match="token_counts"):
        TraceRecord(**{**base, "token_counts": []})
    with pytest.raises(ValueError, match="token_counts"):
        TraceRecord(**{**base, "token_counts": {"prompt": False}})


def test_stable_prompt_hash_is_deterministic_sha256() -> None:
    digest = stable_prompt_hash("What is 2 + 2?")

    assert digest == stable_prompt_hash("What is 2 + 2?")
    assert digest != stable_prompt_hash("What is 2 + 3?")
    assert digest.startswith("sha256:")
    assert len(digest) == len("sha256:") + 64


def test_training_eligible_query_excludes_protected_evaluation_roles(
    tmp_path: Path,
) -> None:
    store = JsonlTraceStore(tmp_path / "traces.jsonl")
    for trace_id, role in (
        ("train-trace", "train"),
        ("replay-trace", "replay"),
        ("promotion-trace", "promotion"),
        ("canary-trace", "canary"),
    ):
        store.append(
            TraceRecord(
                **{
                    **_trace_payload(),
                    "trace_id": trace_id,
                    "split_role": role,
                }
            )
        )

    assert [
        trace.trace_id for trace in store.training_eligible(task_id="gsm8k")
    ] == ["train-trace", "replay-trace"]


def _trace_payload() -> dict[str, object]:
    return {
        "trace_id": "trace-001",
        "run_id": "run-001",
        "task_id": "gsm8k",
        "example_id": "example-001",
        "split_role": "train",
        "prompt_hash": stable_prompt_hash("What is 2 + 2?"),
        "source_checkpoint": "checkpoints/seed",
        "policy_version": "seed-v1",
        "policy_step": 0,
        "policy_step_evidence": "static",
        "rollout_group_id": "group-1",
        "generation_backend": "local",
        "sampling_config": {"temperature": 0.0},
        "verifier_id": "gsm8k_numeric_v1",
    }
