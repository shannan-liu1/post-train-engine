"""OPSD dataset helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def filter_opsd_hard_rows(
    rows: Sequence[Mapping[str, Any]],
    buckets: set[str] | None = None,
) -> tuple[dict[str, Any], ...]:
    allowed = buckets or {"hard_solved", "unsolved_parseable"}
    return tuple(dict(row) for row in rows if str(row.get("bucket")) in allowed)
