from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from post_train_engine.cli.main import main


def test_push_hf_dry_run_writes_evidence_upload_plan(tmp_path: Path) -> None:
    run_dir = _write_and_run_smoke_config(tmp_path)

    main(
        [
            "push-hf",
            "--run",
            str(run_dir),
            "--repo-id",
            "user/post-train-gsm8k",
            "--dry-run",
        ]
    )

    plan = json.loads((run_dir / "push_plan.json").read_text(encoding="utf-8"))
    assert plan["dry_run"] is True
    assert plan["repo_id"] == "user/post-train-gsm8k"
    assert plan["run_id"] == "gsm8k-smoke"
    assert plan["candidate_id"] == "candidate-dry-run"
    assert plan["artifact_status"] == "ok"

    targets = {target["role"]: target for target in plan["targets"]}
    assert targets["evidence"]["will_upload"] is True
    assert targets["evidence"]["path_in_repo"].endswith(
        "tasks/gsm8k/checkpoints/2026-06-16/candidate-dry-run/evidence",
    )
    assert targets["checkpoint"]["will_upload"] is False
    assert targets["checkpoint"]["reason"] == "candidate_not_promoted"


def test_push_hf_dry_run_refuses_missing_promotion_evidence(tmp_path: Path) -> None:
    run_dir = _write_and_run_smoke_config(tmp_path)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    (run_dir / manifest["artifacts"]["promotion_decision"]["path"]).unlink()

    with pytest.raises(ValueError, match="promotion_decision:missing"):
        main(
            [
                "push-hf",
                "--run",
                str(run_dir),
                "--repo-id",
                "user/post-train-gsm8k",
                "--dry-run",
            ]
        )

    assert not (run_dir / "push_plan.json").exists()


def test_push_hf_dry_run_refuses_malformed_lifecycle_promoted_flag(
    tmp_path: Path,
) -> None:
    run_dir = _write_and_run_smoke_config(tmp_path)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    lifecycle_path = run_dir / manifest["artifacts"]["lifecycle"]["path"]
    lifecycle = json.loads(lifecycle_path.read_text(encoding="utf-8"))
    lifecycle["promoted"] = "false"
    lifecycle_path.write_text(
        json.dumps(lifecycle, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest["artifacts"]["lifecycle"]["sha256"] = _sha256(lifecycle_path)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="promotion_consistency:semantic_failed"):
        main(
            [
                "push-hf",
                "--run",
                str(run_dir),
                "--repo-id",
                "user/post-train-gsm8k",
                "--dry-run",
            ]
        )

    assert not (run_dir / "push_plan.json").exists()


def test_push_hf_dry_run_refuses_promoted_lifecycle_without_promote_decision(
    tmp_path: Path,
) -> None:
    run_dir = _write_and_run_smoke_config(tmp_path)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    lifecycle_path = run_dir / manifest["artifacts"]["lifecycle"]["path"]
    lifecycle = json.loads(lifecycle_path.read_text(encoding="utf-8"))
    lifecycle["promoted"] = True
    lifecycle_path.write_text(
        json.dumps(lifecycle, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest["artifacts"]["lifecycle"]["sha256"] = _sha256(lifecycle_path)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="promotion_consistency:semantic_failed"):
        main(
            [
                "push-hf",
                "--run",
                str(run_dir),
                "--repo-id",
                "user/post-train-gsm8k",
                "--dry-run",
            ]
        )

    assert not (run_dir / "push_plan.json").exists()


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


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
