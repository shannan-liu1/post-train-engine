from __future__ import annotations

from post_train_engine.training.grpo_config import configured_modified_grpo_knobs


def test_modified_grpo_knobs_detect_truthy_scalar_values() -> None:
    assert configured_modified_grpo_knobs({"adaptive_entropy": True}) == (
        "adaptive_entropy",
    )


def test_modified_grpo_knobs_detect_non_empty_mapping_without_enabled_flag() -> None:
    assert configured_modified_grpo_knobs({"outer_ratio_clip": {"r_max": 50.0}}) == (
        "outer_ratio_clip",
    )


def test_modified_grpo_knobs_ignore_absent_empty_and_disabled_values() -> None:
    assert (
        configured_modified_grpo_knobs(
            {
                "adaptive_entropy": {},
                "outer_ratio_clip": {"enabled": False, "r_max": 50.0},
            }
        )
        == ()
    )


def test_modified_grpo_knobs_fail_closed_on_malformed_enabled_value() -> None:
    assert configured_modified_grpo_knobs({"adaptive_entropy": {"enabled": None}}) == (
        "adaptive_entropy",
    )
