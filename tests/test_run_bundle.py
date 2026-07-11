from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from post_train_engine.artifacts import validate_run_bundle
from post_train_engine.run_bundle import (
    ArtifactRef,
    RunBundle,
    RunManifest,
    capture_source_identity,
    write_manifest_atomic,
)


def test_run_bundle_validates_after_relocation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    original = tmp_path / "original" / "run-001"
    original.mkdir(parents=True)
    decision_path = original / "promotion_decision.json"
    _write_json(
        decision_path,
        {
            "decision": "reject",
            "primary_metric": "accuracy",
            "primary_delta": 0.0,
            "rejection_reasons": ["no improvement"],
        },
    )
    _write_json(
        original / "manifest.json",
        {
            "schema_version": "post_train_run_v1",
            "run_id": "run-001",
            "candidate_id": "candidate-001",
            "parent_candidate_id": "baseline",
            "task_name": "gsm8k",
            "model_id": "fake-model",
            "status": "rejected",
            "inputs": {
                "model": {
                    "kind": "model",
                    "requested_id": "fake-model",
                    "resolved_id": "fake-model",
                    "resolution_state": "exact",
                    "fingerprint": "sha256:" + "1" * 64,
                },
                "dataset": {
                    "kind": "dataset",
                    "requested_id": "fixture-data",
                    "resolved_id": "fixture-data",
                    "resolution_state": "exact",
                    "resolved_revision": "fixture-v1",
                },
            },
            "source": {
                "commit_sha": "abc123",
                "state": "clean",
            },
            "artifacts": {
                "promotion_decision": {
                    "path": "promotion_decision.json",
                    "kind": "promotion_decision",
                    "sha256": _sha256(decision_path),
                    "required": True,
                },
            },
        },
    )

    relocated = tmp_path / "relocated" / "run-001"
    shutil.copytree(original, relocated)
    unrelated_cwd = tmp_path / "unrelated"
    unrelated_cwd.mkdir()
    monkeypatch.chdir(unrelated_cwd)

    status = RunBundle.load(relocated).validate()

    assert status["status"] == "ok"
    assert status["failure_count"] == 0
    assert status["run_id"] == "run-001"

    compatibility_status = validate_run_bundle(relocated, write=False)
    assert compatibility_status["status"] == "ok"
    assert compatibility_status["run_id"] == "run-001"


def test_source_identity_distinguishes_untracked_source(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    (repo / "tracked.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "add", "tracked.py")
    _git(
        repo,
        "-c",
        "user.name=Test User",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-m",
        "initial",
    )

    clean = capture_source_identity(repo)
    assert clean.state == "clean"
    assert clean.commit_sha
    assert clean.untracked_sha256 is None

    (repo / "new_source.py").write_text("NEW_VALUE = 2\n", encoding="utf-8")

    dirty = capture_source_identity(repo)
    assert dirty.state == "untracked_dirty"
    assert dirty.commit_sha == clean.commit_sha
    assert dirty.untracked_sha256 is not None


def test_finalized_manifest_rejects_unresolved_inputs() -> None:
    from pydantic import ValidationError

    from post_train_engine.run_bundle import RunManifest

    with pytest.raises(ValidationError, match="finalized Run inputs must be resolved"):
        RunManifest.model_validate(
            {
                "run_id": "run-1",
                "candidate_id": "candidate-1",
                "task_name": "fixture",
                "model_id": "model-1",
                "status": "rejected",
                "source": {"state": "unknown"},
                "inputs": {
                    "model": {
                        "kind": "model",
                        "requested_id": "model-1",
                        "resolution_state": "unresolved",
                        "non_certifying_reason": "provider omitted a revision",
                    },
                    "dataset": {
                        "kind": "dataset",
                        "requested_id": "data-1",
                        "resolved_id": "data-1",
                        "resolved_revision": "v1",
                        "resolution_state": "exact",
                    },
                },
            }
        )


def test_artifact_reference_rejects_path_traversal() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="remain inside the run bundle"):
        ArtifactRef(
            path="../outside.json",
            kind="evidence",
            sha256="sha256:" + "0" * 64,
        )


def test_manifest_finalization_resumes_after_interrupted_temporary_write(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / ".manifest.json.tmp").write_text("partial", encoding="utf-8")
    manifest = RunManifest.model_validate(
        {
            "run_id": "run-1",
            "candidate_id": "candidate-1",
            "task_name": "fixture",
            "model_id": "model-1",
            "status": "failed",
            "source": {"state": "unknown"},
            "inputs": {
                "model": {
                    "kind": "model",
                    "requested_id": "model-1",
                    "resolution_state": "unresolved",
                    "non_certifying_reason": "run failed before model resolution",
                },
                "dataset": {
                    "kind": "dataset",
                    "requested_id": "data-1",
                    "resolution_state": "unresolved",
                    "non_certifying_reason": "run failed before data resolution",
                },
            },
        }
    )

    first = write_manifest_atomic(run_dir, manifest)
    first_body = first.read_text(encoding="utf-8")
    second = write_manifest_atomic(run_dir, manifest)

    assert second == first
    assert second.read_text(encoding="utf-8") == first_body
    assert not (run_dir / ".manifest.json.tmp").exists()


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, body: dict[str, object]) -> None:
    path.write_text(json.dumps(body, indent=2, sort_keys=True), encoding="utf-8")
