from __future__ import annotations

import json
from pathlib import Path

import yaml

from post_train_engine.cli.main import main
from post_train_engine.engine import CANONICAL_STAGE_ORDER
from post_train_engine.jsonl import read_jsonl


def test_local_gsm8k_smoke_run_writes_complete_rejected_bundle(
    tmp_path: Path,
) -> None:
    config = tmp_path / "gsm8k_smoke.yaml"
    run_dir = tmp_path / "runs" / "gsm8k-smoke"
    config.write_text(
        yaml.safe_dump(
            {
                "kind": "gsm8k_local_smoke",
                "run_id": "gsm8k-smoke",
                "out_dir": str(run_dir),
                "seed": 123,
                "model_id": "local-deterministic-gsm8k",
                "prompt_style": "thinking_tags",
                "rollouts": 4,
                "early_rollouts": 2,
                "max_new_tokens": 64,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    main(["run", "--config", str(config)])

    manifest_path = run_dir / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "post_train_run_v1"
    assert manifest["status"] == "rejected"
    assert manifest["model_id"] == "local-deterministic-gsm8k"
    assert manifest["metadata"]["generation_backend"] == "deterministic_fixture"
    assert manifest["metadata"]["train_and_promotion_overlap_count"] == 0
    assert manifest["metadata"]["stage_order"] == list(CANONICAL_STAGE_ORDER)
    assert {
        name
        for name in manifest["artifacts"]
        if name.startswith("stage_receipt_")
    } == {f"stage_receipt_{stage}" for stage in CANONICAL_STAGE_ORDER}

    artifact_paths = {
        name: run_dir / ref["path"]
        for name, ref in manifest["artifacts"].items()
        if ref["required"]
    }
    assert artifact_paths
    for path in artifact_paths.values():
        assert path.exists(), path

    traces = read_jsonl(artifact_paths["traces"])
    rollout_groups = read_jsonl(artifact_paths["rollout_groups"])
    assert traces
    assert rollout_groups
    assert {row["split_role"] for row in traces} == {"probe"}
    assert all(row["prompt_hash"].startswith("sha256:") for row in traces)
    assert all(group["group_size"] == len(group["trace_ids"]) for group in rollout_groups)

    grpo_view = json.loads(
        artifact_paths["grpo_rollout_view"].read_text(encoding="utf-8"),
    )
    easy_view = json.loads(
        artifact_paths["easy_regression_view"].read_text(encoding="utf-8"),
    )
    opsd_view = json.loads(
        artifact_paths["opsd_view"].read_text(encoding="utf-8"),
    )
    assert grpo_view["view_type"] == "grpo_rollout"
    assert grpo_view["method_compatibility"] == ["grpo"]
    assert grpo_view["source_trace_ids"]
    assert grpo_view["source_split_roles"] == ["probe"]
    assert grpo_view["privileged_visibility"] == "none"
    assert easy_view["view_type"] == "sft"
    assert opsd_view["view_type"] == "opsd"
    assert opsd_view["privileged_visibility"] == "gold_answer"

    splits = json.loads(artifact_paths["splits"].read_text(encoding="utf-8"))
    train_ids = set(splits["splits"]["train_pool"])
    promotion_ids = set(splits["splits"]["dev_promotion"])
    assert train_ids
    assert promotion_ids
    assert train_ids.isdisjoint(promotion_ids)

    curriculum_rows = []
    for name in ("grpo_frontier", "easy_regression", "opsd_hard"):
        curriculum_rows.extend(read_jsonl(artifact_paths[name]))
    assert curriculum_rows
    assert {
        str(row["example_id"])
        for row in curriculum_rows
    }.isdisjoint(promotion_ids)

    promotion = json.loads(
        artifact_paths["promotion_decision"].read_text(encoding="utf-8"),
    )
    assert promotion["decision"] == "reject"
    assert promotion["rejection_reasons"]

    registry_rows = read_jsonl(artifact_paths["checkpoint_registry"])
    assert [(row["candidate_id"], row["promoted"]) for row in registry_rows] == [
        ("seed", True),
        ("candidate-dry-run", False),
    ]
    assert registry_rows[-1]["local_state"] == "available"

    lifecycle = json.loads(artifact_paths["lifecycle"].read_text(encoding="utf-8"))
    assert lifecycle["decision"] == "reject"
    assert lifecycle["promotion_decision"]["decision"] == "reject"
    assert lifecycle["data_overlap_report"]["train_and_promotion_overlap_count"] == 0


def test_report_and_diagnose_read_local_gsm8k_bundle(tmp_path: Path) -> None:
    run_dir = _write_and_run_smoke_config(tmp_path)

    main(["report", "--run", str(run_dir)])
    main(["diagnose", "--run", str(run_dir)])

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    diagnostics = json.loads((run_dir / "diagnostics.json").read_text(encoding="utf-8"))

    assert summary["run_id"] == "gsm8k-smoke"
    assert summary["status"] == "rejected"
    assert summary["promotion_decision"] == "reject"
    assert summary["artifact_status"] == "ok"
    assert summary["artifact_failures"] == []
    assert summary["required_artifact_count"] == summary["ok_artifact_count"]
    assert "grpo_rollout_view" in summary["artifacts"]
    assert "candidate-dry-run" in report
    assert diagnostics["primary_category"] == "under_min_delta"
    assert diagnostics["artifact_status"] == "ok"
    assert diagnostics["artifact_refs"]["promotion_decision"].endswith(
        "promotion_decision.json",
    )
    assert diagnostics["artifact_refs"]["artifact_status"].endswith(
        "artifact_status.json",
    )


def test_local_gsm8k_resume_does_not_duplicate_stage_side_effects(
    tmp_path: Path,
) -> None:
    run_dir = _write_and_run_smoke_config(tmp_path)
    trace_count = len(read_jsonl(run_dir / "traces.jsonl"))

    (run_dir / "manifest.json").unlink()
    resume_from = CANONICAL_STAGE_ORDER.index("evidence")
    for stage in CANONICAL_STAGE_ORDER[resume_from:]:
        (run_dir / "state" / f"{stage}.json").unlink()

    main(["run", "--config", str(tmp_path / "gsm8k_smoke.yaml")])

    assert len(read_jsonl(run_dir / "traces.jsonl")) == trace_count
    assert len(read_jsonl(run_dir / "checkpoint_registry.jsonl")) == 2
    assert len(read_jsonl(run_dir / "ledger.jsonl")) == 1


def _write_and_run_smoke_config(tmp_path: Path) -> Path:
    config = tmp_path / "gsm8k_smoke.yaml"
    run_dir = tmp_path / "runs" / "gsm8k-smoke"
    config.write_text(
        yaml.safe_dump(
            {
                "kind": "gsm8k_local_smoke",
                "run_id": "gsm8k-smoke",
                "out_dir": str(run_dir),
                "seed": 123,
                "model_id": "local-deterministic-gsm8k",
                "prompt_style": "thinking_tags",
                "rollouts": 4,
                "early_rollouts": 2,
                "max_new_tokens": 64,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    main(["run", "--config", str(config)])
    return run_dir
