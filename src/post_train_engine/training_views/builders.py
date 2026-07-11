"""Helpers for building training-view manifests around existing data artifacts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
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
    rows: Iterable[Mapping[str, Any]],
    privileged_visibility: PrivilegedVisibility,
    metadata: Mapping[str, Any] | None = None,
) -> TrainingViewArtifact:
    """Build a typed view manifest from row-level trace provenance."""

    row_list = tuple(rows)
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
        source_trace_ids=_collect_source_trace_ids(row_list),
        source_split_roles=_collect_source_split_roles(row_list),
        privileged_visibility=privileged_visibility,
        metadata=dict(metadata or {}),
    )


def _collect_source_trace_ids(rows: Iterable[Mapping[str, Any]]) -> tuple[str, ...]:
    trace_ids: set[str] = set()
    for row in rows:
        raw_ids = row.get("source_trace_ids", ())
        if isinstance(raw_ids, str):
            raw_ids = (raw_ids,)
        for trace_id in raw_ids:
            if not isinstance(trace_id, str) or not trace_id:
                raise ValueError("source_trace_ids entries must be non-empty strings")
            trace_ids.add(trace_id)
    return tuple(sorted(trace_ids))


def _collect_source_split_roles(rows: Iterable[Mapping[str, Any]]) -> tuple[SplitRole, ...]:
    roles: set[str] = set()
    for row in rows:
        raw_roles = row.get("source_split_roles", ())
        if isinstance(raw_roles, str):
            raw_roles = (raw_roles,)
        for role in raw_roles:
            if not isinstance(role, str) or not role:
                raise ValueError("source_split_roles entries must be non-empty strings")
            roles.add(role)
    return cast(tuple[SplitRole, ...], tuple(sorted(roles)))


__all__ = ["build_training_view_artifact"]
