from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from post_train_engine.cli.main import main


def test_artifacts_validate_writes_ok_status_for_smoke_bundle(tmp_path: Path) -> None:
    run_dir = _write_and_run_smoke_config(tmp_path)

    main(["artifacts", "validate", "--run", str(run_dir)])

    status = json.loads((run_dir / "artifact_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "ok"
    assert status["run_id"] == "gsm8k-smoke"
    assert status["required_count"] == status["ok_count"]
    assert status["failures"] == []
    assert {artifact["name"] for artifact in status["artifacts"]} >= {
        "manifest",
        "promotion_decision",
        "lifecycle",
        "traces",
        "rollout_groups",
    }


def test_artifacts_validate_fails_closed_on_hash_mismatch(tmp_path: Path) -> None:
    run_dir = _write_and_run_smoke_config(tmp_path)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    traces_path = run_dir / manifest["artifacts"]["traces"]["path"]
    traces_path.write_text(
        traces_path.read_text(encoding="utf-8") + "{}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="traces:hash_mismatch"):
        main(["artifacts", "validate", "--run", str(run_dir)])

    status = json.loads((run_dir / "artifact_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert status["failure_count"] == 1
    assert status["failures"][0]["name"] == "traces"
    assert status["failures"][0]["status"] == "hash_mismatch"


def test_artifacts_validate_fails_closed_on_malformed_required_flag(
    tmp_path: Path,
) -> None:
    run_dir = _write_and_run_smoke_config(tmp_path)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["traces"]["required"] = "false"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="required"):
        main(["artifacts", "validate", "--run", str(run_dir)])

    status = json.loads((run_dir / "artifact_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert "required" in status["failures"][0]["message"]


def test_artifacts_validate_fails_closed_on_promotion_consistency_conflict(
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
        main(["artifacts", "validate", "--run", str(run_dir)])

    status = json.loads((run_dir / "artifact_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert status["failures"][0]["name"] == "promotion_consistency"


def test_artifacts_validate_fails_closed_when_training_view_uses_promotion_split(
    tmp_path: Path,
) -> None:
    run_dir = _write_and_run_smoke_config(tmp_path)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    view_path = run_dir / manifest["artifacts"]["grpo_rollout_view"]["path"]
    view = json.loads(view_path.read_text(encoding="utf-8"))
    view["source_split_roles"] = ["promotion"]
    view_path.write_text(
        json.dumps(view, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest["artifacts"]["grpo_rollout_view"]["sha256"] = _sha256(view_path)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="training_view_leakage:semantic_failed"):
        main(["artifacts", "validate", "--run", str(run_dir)])

    status = json.loads((run_dir / "artifact_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert status["failures"][0]["name"] == "training_view_leakage"


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
