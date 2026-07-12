from __future__ import annotations

import hashlib
import re
import tomllib
from pathlib import Path

import pytest

from post_train_engine.cli.main import main

ROOT = Path(__file__).resolve().parents[1]


def test_only_canonical_promotion_module_exposes_decision_authority() -> None:
    from post_train_engine import api_hillclimb
    from post_train_engine.evals import harness

    assert not hasattr(api_hillclimb, "decide_api_promotion")
    assert not hasattr(harness, "PromotionPolicy")
    assert not hasattr(harness, "PromotionReport")
    assert not hasattr(harness, "write_promotion_report")


def test_run_bundle_owns_the_only_run_manifest_and_bundle_artifact_ref() -> None:
    from post_train_engine import flywheel

    assert not hasattr(flywheel, "ExperimentEvidenceRecord")
    assert not hasattr(flywheel, "OptimizerAblationReport")
    assert not hasattr(flywheel, "ArtifactRef")


def test_campaign_exposes_one_atomic_outcome_and_incumbent_mutation() -> None:
    from post_train_engine.campaign import CampaignStore

    assert not hasattr(CampaignStore, "record_outcome")
    assert not hasattr(CampaignStore, "promote_incumbent")
    assert not hasattr(CampaignStore, "reserve_budget")


def test_retired_standalone_promotion_command_is_not_registered(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["promote"])

    assert excinfo.value.code == 2
    assert "invalid choice: 'promote'" in capsys.readouterr().err


def test_runpod_dependencies_are_frozen_without_replacing_image_torch() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    base = {_requirement_name(value) for value in project["project"]["dependencies"]}
    rlvr = {
        _requirement_name(value)
        for value in project["project"]["optional-dependencies"]["rlvr"]
    }

    assert {"accelerate", "peft", "trl"} <= rlvr
    assert base.isdisjoint(rlvr)
    assert "wandb" not in base | rlvr

    lock = ROOT / "uv.lock"
    constraints = (ROOT / "requirements" / "runpod.txt").read_text(encoding="utf-8")
    expected_hash = hashlib.sha256(
        lock.read_text(encoding="utf-8").encode("utf-8")
    ).hexdigest()
    assert f"# uv-lock-sha256: {expected_hash}" in constraints.splitlines()[:3]
    assert not any(
        line.lower().startswith("torch==") for line in constraints.splitlines()
    )


def _requirement_name(value: str) -> str:
    return re.split(r"[ <=>@\[]", value, maxsplit=1)[0].lower()
