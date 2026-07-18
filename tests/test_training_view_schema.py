from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from post_train_engine.training_views.builders import build_training_view_artifact
from post_train_engine.training_views.schema import (
    TrainingDataRef,
    TrainingViewArtifact,
)


def test_training_view_artifact_records_method_evidence(tmp_path: Path) -> None:
    data_path = tmp_path / "view.jsonl"
    data_path.write_text(
        json.dumps({"example_id": "ex1", "source_trace_ids": ["t1"]}) + "\n",
        encoding="utf-8",
    )

    view = TrainingViewArtifact(
        view_id="run1:grpo",
        run_id="run1",
        task_id="gsm8k",
        view_type="grpo_rollout",
        method_compatibility=("grpo",),
        data_artifact=TrainingDataRef.from_path(
            data_path, kind="grpo_frontier", artifact_root=tmp_path
        ),
        source_trace_ids=("t1",),
        source_split_roles=("probe",),
        privileged_visibility="none",
    )

    assert view.view_type == "grpo_rollout"
    assert view.source_trace_ids == ("t1",)


def test_training_view_rejects_mutated_data_at_consumption(tmp_path: Path) -> None:
    data_path = tmp_path / "view.jsonl"
    data_path.write_text('{"example_id":"ex1"}\n', encoding="utf-8")
    view = TrainingViewArtifact(
        view_id="run1:grpo",
        run_id="run1",
        task_id="gsm8k",
        view_type="grpo_rollout",
        method_compatibility=("grpo",),
        data_artifact=TrainingDataRef.from_path(
            data_path,
            kind="grpo_frontier",
            artifact_root=tmp_path,
        ),
        source_trace_ids=("t1",),
        source_split_roles=("probe",),
        privileged_visibility="none",
    )
    data_path.write_text('{"example_id":"changed"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="training data sha256 mismatch"):
        view.require_data_integrity(tmp_path)


def test_training_view_builder_writes_relocatable_data_reference(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    data_path = run_dir / "evidence" / "view.jsonl"
    data_path.parent.mkdir(parents=True)
    data_path.write_text(
        json.dumps(
            {
                "example_id": "ex1",
                "source_trace_ids": ["t1"],
                "source_split_roles": ["probe"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    view = build_training_view_artifact(
        view_id="run1:grpo",
        run_id="run1",
        task_id="gsm8k",
        view_type="grpo_rollout",
        method_compatibility=("grpo",),
        data_path=data_path,
        artifact_root=run_dir,
        data_kind="grpo_frontier",
        privileged_visibility="none",
    )

    relocated = tmp_path / "relocated"
    shutil.copytree(run_dir, relocated)
    assert view.data_artifact.path == "evidence/view.jsonl"
    assert view.resolve_data_path(relocated).read_text(encoding="utf-8") == (
        data_path.read_text(encoding="utf-8")
    )


def test_training_view_rejects_missing_source_trace_ids(tmp_path: Path) -> None:
    data_path = tmp_path / "view.jsonl"
    data_path.write_text("[]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="source_trace_ids"):
        TrainingViewArtifact(
            view_id="run1:grpo",
            run_id="run1",
            task_id="gsm8k",
            view_type="grpo_rollout",
            method_compatibility=("grpo",),
            data_artifact=TrainingDataRef.from_path(
                data_path, kind="grpo_frontier", artifact_root=tmp_path
            ),
            source_trace_ids=(),
            source_split_roles=("probe",),
            privileged_visibility="none",
        )


def test_training_view_rejects_promotion_split_sources(tmp_path: Path) -> None:
    data_path = tmp_path / "view.jsonl"
    data_path.write_text("[]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="protected evaluation split"):
        TrainingViewArtifact(
            view_id="run1:grpo",
            run_id="run1",
            task_id="gsm8k",
            view_type="grpo_rollout",
            method_compatibility=("grpo",),
            data_artifact=TrainingDataRef.from_path(
                data_path, kind="grpo_frontier", artifact_root=tmp_path
            ),
            source_trace_ids=("t1",),
            source_split_roles=("promotion",),
            privileged_visibility="none",
        )


def test_training_view_rejects_diagnostic_split_sources(tmp_path: Path) -> None:
    data_path = tmp_path / "view.jsonl"
    data_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="protected evaluation split"):
        TrainingViewArtifact(
            view_id="run1:grpo",
            run_id="run1",
            task_id="gsm8k",
            view_type="grpo_rollout",
            method_compatibility=("grpo",),
            data_artifact=TrainingDataRef.from_path(
                data_path, kind="grpo_frontier", artifact_root=tmp_path
            ),
            source_trace_ids=("t1",),
            source_split_roles=("diagnostic",),
            privileged_visibility="none",
        )


def test_training_view_builder_rejects_malformed_source_trace_ids(
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "view.jsonl"
    data_path.write_text(
        json.dumps({"example_id": "ex1", "source_trace_ids": [""]}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="source_trace_ids"):
        build_training_view_artifact(
            view_id="run1:grpo",
            run_id="run1",
            task_id="gsm8k",
            view_type="grpo_rollout",
            method_compatibility=("grpo",),
            data_path=data_path,
            artifact_root=tmp_path,
            data_kind="grpo_frontier",
            privileged_visibility="none",
        )


def test_training_view_builder_requires_provenance_on_every_row(
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "view.jsonl"
    data_path.write_text(
        json.dumps(
            {
                "example_id": "ex1",
                "source_trace_ids": ["t1"],
                "source_split_roles": ["train"],
            }
        )
        + "\n"
        + json.dumps({"example_id": "ex2"})
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="every training row requires"):
        build_training_view_artifact(
            view_id="run1:grpo",
            run_id="run1",
            task_id="gsm8k",
            view_type="grpo_rollout",
            method_compatibility=("grpo",),
            data_path=data_path,
            artifact_root=tmp_path,
            data_kind="grpo_frontier",
            privileged_visibility="none",
        )


def test_training_view_builder_rejects_scalar_row_provenance(tmp_path: Path) -> None:
    data_path = tmp_path / "view.jsonl"
    data_path.write_text(
        json.dumps(
            {
                "example_id": "ex1",
                "source_trace_ids": "t1",
                "source_split_roles": "train",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="source_trace_ids must be a list"):
        build_training_view_artifact(
            view_id="run1:grpo",
            run_id="run1",
            task_id="gsm8k",
            view_type="grpo_rollout",
            method_compatibility=("grpo",),
            data_path=data_path,
            artifact_root=tmp_path,
            data_kind="grpo_frontier",
            privileged_visibility="none",
        )


def test_training_view_builder_derives_roles_from_hashed_data(tmp_path: Path) -> None:
    data_path = tmp_path / "view.jsonl"
    data_path.write_text(
        json.dumps(
            {
                "source_trace_ids": ["promotion-trace"],
                "source_split_roles": ["promotion"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="protected evaluation split"):
        build_training_view_artifact(
            view_id="run1:grpo",
            run_id="run1",
            task_id="gsm8k",
            view_type="grpo_rollout",
            method_compatibility=("grpo",),
            data_path=data_path,
            artifact_root=tmp_path,
            data_kind="grpo_frontier",
            privileged_visibility="none",
        )


def test_training_view_rejects_string_source_trace_ids(tmp_path: Path) -> None:
    data_path = tmp_path / "view.jsonl"
    data_path.write_text("[]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="source_trace_ids"):
        TrainingViewArtifact(
            view_id="run1:grpo",
            run_id="run1",
            task_id="gsm8k",
            view_type="grpo_rollout",
            method_compatibility=("grpo",),
            data_artifact=TrainingDataRef.from_path(
                data_path, kind="grpo_frontier", artifact_root=tmp_path
            ),
            source_trace_ids="t1",  # type: ignore[arg-type]
            source_split_roles=("probe",),
            privileged_visibility="none",
        )


def test_training_data_ref_rejects_absolute_paths(tmp_path: Path) -> None:
    data_path = tmp_path / "view.jsonl"
    data_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="relative to artifact_root"):
        TrainingDataRef.from_path(data_path, kind="grpo_frontier")
