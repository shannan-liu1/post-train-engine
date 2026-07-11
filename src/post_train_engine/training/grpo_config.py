"""GRPO configuration contract helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

MODIFIED_GRPO_KNOBS = ("adaptive_entropy", "outer_ratio_clip")


def configured_modified_grpo_knobs(parameters: Mapping[str, Any]) -> tuple[str, ...]:
    """Return modified-GRPO knobs that must not be silently ignored."""

    return tuple(
        knob
        for knob in MODIFIED_GRPO_KNOBS
        if modified_grpo_knob_enabled(parameters.get(knob))
    )


def modified_grpo_knob_enabled(value: Any) -> bool:
    if isinstance(value, Mapping):
        if "enabled" in value:
            return value["enabled"] is not False
        return bool(value)
    return bool(value)
