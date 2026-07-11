from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from post_train_engine.lifecycle import (
    CheckpointLifecycleInput,
    CheckpointLifecycleManager,
    HuggingFaceLifecycleConfig,
    ModelLifecycleConfig,
)


class FakeUploader:
    def __init__(self, *, fail_on_path_suffix: str | None = None) -> None:
        self.fail_on_path_suffix = fail_on_path_suffix
        self.uploads: list[dict[str, Any]] = []
        self.repos: list[dict[str, Any]] = []

    def ensure_repo(
        self,
        *,
        repo_id: str,
        repo_type: str,
        private: bool,
    ) -> None:
        self.repos.append(
            {"repo_id": repo_id, "repo_type": repo_type, "private": private},
        )

    def upload_folder(
        self,
        *,
        local_dir: Path,
        repo_id: str,
        path_in_repo: str,
        repo_type: str,
        commit_message: str,
    ) -> str:
        if self.fail_on_path_suffix and path_in_repo.endswith(
            self.fail_on_path_suffix,
        ):
            raise RuntimeError("upload failed")
        self.uploads.append(
            {
                "local_dir": local_dir,
                "repo_id": repo_id,
                "path_in_repo": path_in_repo,
                "repo_type": repo_type,
                "commit_message": commit_message,
            }
        )
        return f"hf://{repo_id}/{path_in_repo}"


def test_lifecycle_refuses_to_delete_checkpoint_outside_managed_root(
    tmp_path: Path,
) -> None:
    managed_root = tmp_path / "managed-checkpoints"
    managed_root.mkdir()
    external = tmp_path / "external" / "candidate"
    external.mkdir(parents=True)
    (external / "model.safetensors").write_text("keep", encoding="utf-8")
    manager = CheckpointLifecycleManager(
        ModelLifecycleConfig(
            artifact_dir=tmp_path / "lifecycle",
            managed_checkpoint_root=managed_root,
            hf=HuggingFaceLifecycleConfig(enabled=False),
        ),
        date_provider=lambda: "2026-06-15",
    )

    with pytest.raises(ValueError, match="outside managed checkpoint root"):
        manager.finalize(
            CheckpointLifecycleInput(
                candidate_id="candidate",
                checkpoint_ref=str(external),
                task_name="gsm8k",
                parent_candidate_id="seed",
                parent_checkpoint_ref="seed",
                previous_incumbent_candidate_id="seed",
                previous_incumbent_checkpoint_ref="seed",
                previous_incumbent_remote_ref=None,
                promoted=False,
                score=0.0,
                incumbent_score=1.0,
                metrics={"accuracy": 0.0},
                evaluation_artifacts={},
                evaluation_metadata={},
                train_artifacts={},
                train_metrics={},
                train_metadata={},
                promotion_gate={"decision": "reject"},
            )
        )

    assert external.is_dir()


def test_lifecycle_uploads_promoted_checkpoint_and_discards_rejected_bytes(
    tmp_path: Path,
) -> None:
    promoted_dir = tmp_path / "checkpoints" / "candidate-1"
    rejected_dir = tmp_path / "checkpoints" / "candidate-2"
    promoted_dir.mkdir(parents=True)
    rejected_dir.mkdir(parents=True)
    (promoted_dir / "model.safetensors").write_text("promoted", encoding="utf-8")
    (rejected_dir / "model.safetensors").write_text("rejected", encoding="utf-8")
    uploader = FakeUploader()
    manager = CheckpointLifecycleManager(
        ModelLifecycleConfig(
            artifact_dir=tmp_path / "lifecycle",
            managed_checkpoint_root=tmp_path / "checkpoints",
            hf=HuggingFaceLifecycleConfig(
                enabled=True,
                repo_id="user/post-train-gsm8k",
                private=True,
            ),
        ),
        uploader=uploader,
        date_provider=lambda: "2026-06-15",
    )

    promoted = manager.finalize(
        CheckpointLifecycleInput(
            candidate_id="candidate-1",
            checkpoint_ref=str(promoted_dir),
            task_name="gsm8k",
            parent_candidate_id="seed",
            parent_checkpoint_ref="hf://user/post-train-gsm8k/tasks/gsm8k/checkpoints/2026-06-14/seed",
            previous_incumbent_candidate_id="seed",
            previous_incumbent_checkpoint_ref="seed",
            previous_incumbent_remote_ref=None,
            promoted=True,
            score=0.74,
            incumbent_score=0.69,
            metrics={"accuracy": 0.74},
            evaluation_artifacts={"eval_report": "evals/candidate-1.json"},
            evaluation_metadata={"split": "dev_promotion", "split_hash": "sha256:abc"},
            train_artifacts={"trainer_log": "logs/candidate-1.jsonl"},
            train_metrics={"train_loss": 0.12},
            train_metadata={"costs": {"runpod_usd": 3.21}},
            promotion_gate={
                "decision": "promote",
                "reasons": ["score beat incumbent and no constraints failed"],
            },
        )
    )

    rejected = manager.finalize(
        CheckpointLifecycleInput(
            candidate_id="candidate-2",
            checkpoint_ref=str(rejected_dir),
            task_name="gsm8k",
            parent_candidate_id="candidate-1",
            parent_checkpoint_ref=str(promoted_dir),
            previous_incumbent_candidate_id="candidate-1",
            previous_incumbent_checkpoint_ref=str(promoted_dir),
            previous_incumbent_remote_ref=promoted.remote_ref,
            promoted=False,
            score=0.70,
            incumbent_score=0.74,
            metrics={"accuracy": 0.70},
            evaluation_artifacts={"eval_report": "evals/candidate-2.json"},
            evaluation_metadata={"split": "dev_promotion", "split_hash": "sha256:abc"},
            train_artifacts={"trainer_log": "logs/candidate-2.jsonl"},
            train_metrics={"train_loss": 0.16},
            train_metadata={"costs": {"runpod_usd": 2.11}},
            promotion_gate={
                "decision": "reject",
                "reasons": ["score did not beat incumbent"],
            },
            rejection_reason="score did not beat incumbent",
        )
    )

    assert promoted.remote_ref == (
        "hf://user/post-train-gsm8k/tasks/gsm8k/checkpoints/"
        "2026-06-15/candidate-1"
    )
    assert promoted.local_state == "available"
    assert rejected.remote_ref is None
    assert rejected.local_state == "discarded"
    assert promoted_dir.exists()
    assert not rejected_dir.exists()
    assert [upload["path_in_repo"] for upload in uploader.uploads] == [
        "tasks/gsm8k/checkpoints/2026-06-15/candidate-1/checkpoint",
        "tasks/gsm8k/checkpoints/2026-06-15/candidate-1/evidence",
        "tasks/gsm8k/checkpoints/2026-06-15/candidate-2/evidence",
        "tasks/gsm8k/checkpoints/2026-06-15/candidate-2/evidence",
    ]

    evidence = json.loads(
        (tmp_path / "lifecycle" / "gsm8k" / "2026-06-15" / "candidate-1" / "lifecycle.json").read_text(
            encoding="utf-8",
        )
    )
    assert evidence["eval_performance"]["metrics"] == {"accuracy": 0.74}
    assert evidence["costs"] == {"runpod_usd": 3.21}
    assert evidence["promotion_gate"]["decision"] == "promote"
    assert evidence["lineage"]["parent_candidate_id"] == "seed"
    assert evidence["lineage"]["previous_incumbent_candidate_id"] == "seed"


def test_lifecycle_does_not_discard_when_hf_evidence_upload_fails(
    tmp_path: Path,
) -> None:
    rejected_dir = tmp_path / "checkpoints" / "candidate-2"
    rejected_dir.mkdir(parents=True)
    (rejected_dir / "model.safetensors").write_text("rejected", encoding="utf-8")
    manager = CheckpointLifecycleManager(
        ModelLifecycleConfig(
            artifact_dir=tmp_path / "lifecycle",
            managed_checkpoint_root=tmp_path / "checkpoints",
            hf=HuggingFaceLifecycleConfig(
                enabled=True,
                repo_id="user/post-train-gsm8k",
            ),
        ),
        uploader=FakeUploader(fail_on_path_suffix="candidate-2/evidence"),
        date_provider=lambda: "2026-06-15",
    )

    with pytest.raises(RuntimeError, match="upload failed"):
        manager.finalize(
            CheckpointLifecycleInput(
                candidate_id="candidate-2",
                checkpoint_ref=str(rejected_dir),
                task_name="gsm8k",
                parent_candidate_id="candidate-1",
                parent_checkpoint_ref="candidate-1",
                previous_incumbent_candidate_id="candidate-1",
                previous_incumbent_checkpoint_ref="candidate-1",
                previous_incumbent_remote_ref="hf://repo/candidate-1",
                promoted=False,
                score=0.70,
                incumbent_score=0.74,
                metrics={"accuracy": 0.70},
                evaluation_artifacts={},
                evaluation_metadata={"split": "dev_promotion"},
                train_artifacts={},
                train_metrics={},
                train_metadata={"costs": {"runpod_usd": 2.11}},
                promotion_gate={
                    "decision": "reject",
                    "reasons": ["score did not beat incumbent"],
                },
                rejection_reason="score did not beat incumbent",
            )
        )

    assert rejected_dir.exists()


def test_lifecycle_refuses_to_discard_non_checkpoint_directory(tmp_path: Path) -> None:
    rejected_dir = tmp_path / "not-a-checkpoint" / "candidate-2"
    rejected_dir.mkdir(parents=True)
    (rejected_dir / "notes.txt").write_text("not model bytes", encoding="utf-8")
    manager = CheckpointLifecycleManager(
        ModelLifecycleConfig(
            artifact_dir=tmp_path / "lifecycle",
            managed_checkpoint_root=tmp_path / "not-a-checkpoint",
            hf=HuggingFaceLifecycleConfig(
                enabled=True,
                repo_id="user/post-train-gsm8k",
            ),
        ),
        uploader=FakeUploader(),
        date_provider=lambda: "2026-06-15",
    )

    with pytest.raises(ValueError, match="non-checkpoint directory"):
        manager.finalize(
            CheckpointLifecycleInput(
                candidate_id="candidate-2",
                checkpoint_ref=str(rejected_dir),
                task_name="gsm8k",
                parent_candidate_id="candidate-1",
                parent_checkpoint_ref="candidate-1",
                previous_incumbent_candidate_id="candidate-1",
                previous_incumbent_checkpoint_ref="candidate-1",
                previous_incumbent_remote_ref="hf://repo/candidate-1",
                promoted=False,
                score=0.70,
                incumbent_score=0.74,
                metrics={"accuracy": 0.70},
                evaluation_artifacts={},
                evaluation_metadata={"split": "dev_promotion"},
                train_artifacts={},
                train_metrics={},
                train_metadata={"costs": {"runpod_usd": 2.11}},
                promotion_gate={
                    "decision": "reject",
                    "reasons": ["score did not beat incumbent"],
                },
                rejection_reason="score did not beat incumbent",
            )
        )

    assert rejected_dir.exists()


def test_lifecycle_refuses_to_upload_non_checkpoint_directory(tmp_path: Path) -> None:
    promoted_dir = tmp_path / "not-a-checkpoint" / "candidate-1"
    promoted_dir.mkdir(parents=True)
    (promoted_dir / "notes.txt").write_text("not model bytes", encoding="utf-8")
    uploader = FakeUploader()
    manager = CheckpointLifecycleManager(
        ModelLifecycleConfig(
            artifact_dir=tmp_path / "lifecycle",
            hf=HuggingFaceLifecycleConfig(
                enabled=True,
                repo_id="user/post-train-gsm8k",
            ),
        ),
        uploader=uploader,
        date_provider=lambda: "2026-06-15",
    )

    with pytest.raises(ValueError, match="non-checkpoint directory"):
        manager.finalize(
            CheckpointLifecycleInput(
                candidate_id="candidate-1",
                checkpoint_ref=str(promoted_dir),
                task_name="gsm8k",
                parent_candidate_id="seed",
                parent_checkpoint_ref="seed",
                previous_incumbent_candidate_id="seed",
                previous_incumbent_checkpoint_ref="seed",
                previous_incumbent_remote_ref=None,
                promoted=True,
                score=0.78,
                incumbent_score=0.74,
                metrics={"accuracy": 0.78},
                evaluation_artifacts={},
                evaluation_metadata={"split": "dev_promotion"},
                train_artifacts={},
                train_metrics={},
                train_metadata={"costs": {"runpod_usd": 2.11}},
                promotion_gate={
                    "decision": "promote",
                    "reasons": ["score beat incumbent and no constraints failed"],
                },
            )
        )

    assert uploader.uploads == []


def test_lifecycle_prunes_previous_promoted_checkpoint_after_new_best_upload(
    tmp_path: Path,
) -> None:
    previous_dir = tmp_path / "checkpoints" / "candidate-1"
    current_dir = tmp_path / "checkpoints" / "candidate-2"
    previous_dir.mkdir(parents=True)
    current_dir.mkdir(parents=True)
    (previous_dir / "model.safetensors").write_text("previous", encoding="utf-8")
    (current_dir / "model.safetensors").write_text("current", encoding="utf-8")
    uploader = FakeUploader()
    manager = CheckpointLifecycleManager(
        ModelLifecycleConfig(
            artifact_dir=tmp_path / "lifecycle",
            managed_checkpoint_root=tmp_path / "checkpoints",
            hf=HuggingFaceLifecycleConfig(
                enabled=True,
                repo_id="user/post-train-gsm8k",
            ),
        ),
        uploader=uploader,
        date_provider=lambda: "2026-06-15",
    )

    outcome = manager.finalize(
        CheckpointLifecycleInput(
            candidate_id="candidate-2",
            checkpoint_ref=str(current_dir),
            task_name="gsm8k",
            parent_candidate_id="candidate-1",
            parent_checkpoint_ref=str(previous_dir),
            previous_incumbent_candidate_id="candidate-1",
            previous_incumbent_checkpoint_ref=str(previous_dir),
            previous_incumbent_remote_ref=(
                "hf://user/post-train-gsm8k/tasks/gsm8k/checkpoints/"
                "2026-06-15/candidate-1"
            ),
            promoted=True,
            score=0.78,
            incumbent_score=0.74,
            metrics={"accuracy": 0.78},
            evaluation_artifacts={"eval_report": "evals/candidate-2.json"},
            evaluation_metadata={"split": "dev_promotion", "split_hash": "sha256:abc"},
            train_artifacts={"trainer_log": "logs/candidate-2.jsonl"},
            train_metrics={"train_loss": 0.09},
            train_metadata={"costs": {"runpod_usd": 3.44}},
            promotion_gate={
                "decision": "promote",
                "reasons": ["score beat incumbent and no constraints failed"],
            },
        )
    )

    assert current_dir.exists()
    assert not previous_dir.exists()
    assert outcome.local_state == "available"
    assert outcome.discarded_paths == (str(previous_dir),)
    evidence = json.loads(outcome.evidence_path.read_text(encoding="utf-8"))
    assert evidence["discarded_paths"] == [str(previous_dir)]
    assert [upload["path_in_repo"] for upload in uploader.uploads] == [
        "tasks/gsm8k/checkpoints/2026-06-15/candidate-2/checkpoint",
        "tasks/gsm8k/checkpoints/2026-06-15/candidate-2/evidence",
        "tasks/gsm8k/checkpoints/2026-06-15/candidate-2/evidence",
    ]


def test_hf_lifecycle_path_template_requires_task_date_and_candidate() -> None:
    with pytest.raises(ValueError, match=r"\{date\}"):
        HuggingFaceLifecycleConfig(
            enabled=True,
            repo_id="user/post-train-gsm8k",
            path_template="tasks/{task}/checkpoints/{candidate_id}",
        )


def test_lifecycle_writes_strict_promotion_evidence_bundle(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints" / "candidate-1"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "model.safetensors").write_text("model", encoding="utf-8")
    manager = CheckpointLifecycleManager(
        ModelLifecycleConfig(
            artifact_dir=tmp_path / "lifecycle",
            hf=HuggingFaceLifecycleConfig(enabled=False),
        ),
        date_provider=lambda: "2026-06-15",
    )

    outcome = manager.finalize(
        CheckpointLifecycleInput(
            candidate_id="candidate-1",
            checkpoint_ref=str(checkpoint_dir),
            task_name="gsm8k",
            parent_candidate_id="seed",
            parent_checkpoint_ref="seed",
            previous_incumbent_candidate_id="seed",
            previous_incumbent_checkpoint_ref="seed",
            previous_incumbent_remote_ref=None,
            promoted=True,
            score=0.74,
            incumbent_score=0.69,
            metrics={"accuracy": 0.74},
            evaluation_artifacts={},
            evaluation_metadata={"split": "dev_promotion"},
            train_artifacts={},
            train_metrics={"train_loss": 0.12},
            train_metadata={"train_and_promotion_overlap_count": 0},
            promotion_gate={"decision": "promote", "reasons": []},
            promotion_decision={"decision": "promote", "rejection_reasons": []},
            suite_state={
                "suite_id": "gsm8k-promotion",
                "suite_version": "2026-06-a",
                "num_times_suite_tested": 1,
                "num_candidates_evaluated": 1,
                "accepted_promotion_count": 1,
                "train_and_promotion_overlap_count": 0,
            },
            data_overlap_report={"train_and_promotion_overlap_count": 0},
            severity_summary={"critical": 0, "high": 0, "medium": 0, "low": 0},
            canary_decision={"decision": "pass", "failed_examples": []},
        )
    )

    assert set(outcome.local_artifacts) >= {
        "promotion_decision",
        "suite_state",
        "data_overlap_report",
        "severity_summary",
        "canary_decision",
    }
    promotion_decision = json.loads(
        Path(outcome.local_artifacts["promotion_decision"]).read_text(
            encoding="utf-8",
        )
    )
    suite_state = json.loads(
        Path(outcome.local_artifacts["suite_state"]).read_text(encoding="utf-8")
    )
    assert promotion_decision["decision"] == "promote"
    assert suite_state["train_and_promotion_overlap_count"] == 0
