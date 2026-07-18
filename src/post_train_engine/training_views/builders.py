"""Helpers for building training-view manifests around existing data artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from post_train_engine.traces.schema import SplitRole
from post_train_engine.training_views.schema import (
    PrivilegedVisibility,
    TrainingDataRef,
    TrainingViewArtifact,
    TrainingViewType,
)


def build_training_view_artifact(
    *,
    view_id: str,
    run_id: str,
    task_id: str,
    view_type: TrainingViewType,
    method_compatibility: Sequence[str],
    data_path: str | Path,
    artifact_root: str | Path,
    data_kind: str,
    privileged_visibility: PrivilegedVisibility,
    metadata: Mapping[str, Any] | None = None,
) -> TrainingViewArtifact:
    """Build a typed view manifest from row-level trace provenance."""

    data_path = Path(data_path)
    trace_ids, split_roles = _read_jsonl_provenance(data_path)
    return TrainingViewArtifact(
        view_id=view_id,
        run_id=run_id,
        task_id=task_id,
        view_type=view_type,
        method_compatibility=tuple(method_compatibility),
        data_artifact=TrainingDataRef.from_path(
            data_path,
            kind=data_kind,
            artifact_root=artifact_root,
        ),
        source_trace_ids=trace_ids,
        source_split_roles=split_roles,
        privileged_visibility=privileged_visibility,
        metadata=dict(metadata or {}),
    )


def _read_jsonl_provenance(
    path: Path,
) -> tuple[tuple[str, ...], tuple[SplitRole, ...]]:
    trace_ids: set[str] = set()
    roles: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(
                    f"training data row {line_number} must be a JSON object"
                )
            _collect_row_provenance(row, trace_ids, roles)
    return (
        tuple(sorted(trace_ids)),
        cast(tuple[SplitRole, ...], tuple(sorted(roles))),
    )


def _collect_row_provenance(
    row: Mapping[str, Any],
    trace_ids: set[str],
    roles: set[str],
) -> None:
    raw_ids = row.get("source_trace_ids", ())
    if not isinstance(raw_ids, list | tuple):
        raise ValueError("source_trace_ids must be a list of strings")
    if not raw_ids:
        raise ValueError("every training row requires source_trace_ids")
    for trace_id in raw_ids:
        if not isinstance(trace_id, str) or not trace_id:
            raise ValueError("source_trace_ids entries must be non-empty strings")
        trace_ids.add(trace_id)
    raw_roles = row.get("source_split_roles", ())
    if not isinstance(raw_roles, list | tuple):
        raise ValueError("source_split_roles must be a list of strings")
    if not raw_roles:
        raise ValueError("every training row requires source_split_roles")
    for role in raw_roles:
        if not isinstance(role, str) or not role:
            raise ValueError("source_split_roles entries must be non-empty strings")
        roles.add(role)


__all__ = ["build_training_view_artifact"]
