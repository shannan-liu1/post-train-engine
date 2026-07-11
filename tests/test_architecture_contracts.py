from __future__ import annotations

import pytest

from post_train_engine.cli.main import main


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
