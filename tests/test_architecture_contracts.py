from __future__ import annotations

import hashlib
import inspect
import re
import tomllib
from pathlib import Path

import pytest

from post_train_engine.cli.main import main

ROOT = Path(__file__).resolve().parents[1]


def test_only_canonical_promotion_module_exposes_decision_authority() -> None:
    import post_train_engine
    from post_train_engine import api_hillclimb
    from post_train_engine.evals import harness

    assert not hasattr(post_train_engine, "decide_promotion")
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


def test_task_utility_cli_is_not_registered(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["gsm8k"])

    assert excinfo.value.code == 2
    assert "invalid choice: 'gsm8k'" in capsys.readouterr().err
    assert not (ROOT / "src" / "post_train_engine" / "cli" / "gsm8k.py").exists()


def test_package_exposes_only_the_canonical_console_script() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert project["project"]["scripts"] == {
        "pte": "post_train_engine.cli.main:main"
    }


def test_unknown_unknown_register_ids_are_unique_and_sequential() -> None:
    spec = (ROOT / "docs" / "specs" / "canonical-research-flywheel.md").read_text(
        encoding="utf-8"
    )
    identifiers = [int(value) for value in re.findall(r"^\| U-(\d{3}) \|", spec, re.M)]

    assert identifiers == list(range(1, len(identifiers) + 1))


def test_run_engine_exposes_one_resolver_first_execution_interface() -> None:
    from post_train_engine.engine import RunEngine

    assert not hasattr(RunEngine, "resolve_and_execute")
    assert list(inspect.signature(RunEngine.execute).parameters) == [
        "self",
        "resolver",
        "coordinator",
        "resolution_timeout_seconds",
    ]


def test_r4_has_no_standalone_mutable_script() -> None:
    assert not (ROOT / "scripts" / "benchmark_runpod_eval.py").exists()
    runtime_config = ROOT / "configs" / "gsm8k_runpod_r4.yaml"
    assert runtime_config.is_file()
    assert "kind: runpod_runtime_certification" in runtime_config.read_text(
        encoding="utf-8"
    )


def test_dead_shadow_execution_and_superseded_docs_stay_deleted() -> None:
    import post_train_engine

    assert post_train_engine.__all__ == []
    assert not (ROOT / "src" / "post_train_engine" / "generation.py").exists()
    assert not (ROOT / "src" / "post_train_engine" / "evals" / "run_eval.py").exists()
    assert not (ROOT / "docs" / "lean_flywheel_plan.md").exists()
    for retired_config_family in ("probe", "tasks", "evals"):
        assert not any((ROOT / "configs" / retired_config_family).rglob("*"))


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
    assert {"pandas", "scikit-learn", "scipy", "tqdm"}.isdisjoint(base)

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
