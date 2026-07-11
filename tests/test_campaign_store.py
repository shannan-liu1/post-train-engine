from __future__ import annotations

from pathlib import Path

import pytest

from post_train_engine.campaign import (
    CampaignStore,
    ExperimentProposal,
    ProposalOutcome,
)


def test_proposal_outcome_rejects_status_decision_mismatch() -> None:
    with pytest.raises(ValueError, match="status and decision"):
        ProposalOutcome(
            run_id="run-1",
            status="promoted",
            decision="reject",
            failure_category=None,
            cost_usd=1.0,
            primary_delta=0.1,
        )


def test_campaign_store_preserves_negative_results_and_blocks_duplicates(
    tmp_path: Path,
) -> None:
    store = CampaignStore(tmp_path / "campaign.sqlite")
    store.create_campaign("campaign-1", quality_budget=3, cost_budget_usd=10.0)
    store.add_hypothesis("campaign-1", "hypothesis-1", "Longer context repairs parsing")
    proposal = ExperimentProposal(
        campaign_id="campaign-1",
        hypothesis_id="hypothesis-1",
        method="sft",
        task_name="gsm8k",
        parent_candidate_id="seed",
        comparison_group_id="context-ablation",
        comparison_axis="context_length",
        parameters={"context_length": 2048},
    )

    proposal_id = store.submit_proposal(proposal)
    with pytest.raises(ValueError, match="duplicate proposal"):
        store.submit_proposal(proposal)

    lease = store.claim_proposal(proposal_id, worker_id="worker-1", ttl_seconds=30)
    assert lease.worker_id == "worker-1"
    assert store.claim_proposal(proposal_id, worker_id="worker-2", ttl_seconds=30) is None

    store.finalize_run(
        campaign_id="campaign-1",
        proposal_id=proposal_id,
        worker_id="worker-1",
        expected_incumbent_candidate_id="seed",
        candidate_id="candidate-1",
        outcome=ProposalOutcome(
            run_id="run-1",
            status="rejected",
            decision="reject",
            failure_category="parse_regression",
            cost_usd=1.25,
            primary_delta=-0.1,
        ),
    )

    assert store.get_outcome(proposal_id).status == "rejected"
    assert store.campaign_usage("campaign-1")["completed_runs"] == 1


def test_incumbent_update_is_compare_and_swap(tmp_path: Path) -> None:
    store = CampaignStore(tmp_path / "campaign.sqlite")
    store.create_campaign("campaign-1", quality_budget=3, cost_budget_usd=10.0)
    store.add_hypothesis("campaign-1", "hypothesis-1", "candidate improves")
    store.initialize_incumbent("campaign-1", candidate_id="seed", run_id="seed-run")
    proposal_a = store.submit_proposal(
        ExperimentProposal(
            campaign_id="campaign-1",
            hypothesis_id="hypothesis-1",
            method="sft",
            task_name="gsm8k",
            parent_candidate_id="seed",
            comparison_group_id="candidate-ablation",
            comparison_axis="candidate",
            parameters={"candidate": "a"},
        )
    )
    proposal_b = store.submit_proposal(
        ExperimentProposal(
            campaign_id="campaign-1",
            hypothesis_id="hypothesis-1",
            method="sft",
            task_name="gsm8k",
            parent_candidate_id="seed",
            comparison_group_id="candidate-ablation",
            comparison_axis="candidate",
            parameters={"candidate": "b"},
        )
    )
    store.claim_proposal(proposal_a, worker_id="worker-a", ttl_seconds=30)
    store.claim_proposal(proposal_b, worker_id="worker-b", ttl_seconds=30)

    store.finalize_run(
        campaign_id="campaign-1",
        proposal_id=proposal_a,
        worker_id="worker-a",
        expected_incumbent_candidate_id="seed",
        candidate_id="candidate-a",
        outcome=ProposalOutcome(
                run_id="run-a",
                status="promoted",
                decision="promote",
                failure_category=None,
                cost_usd=1.0,
                primary_delta=0.1,
        ),
    )

    assert store.current_incumbent("campaign-1")["generation"] == 1
    with pytest.raises(ValueError, match="incumbent changed"):
        store.finalize_run(
            campaign_id="campaign-1",
            proposal_id=proposal_b,
            worker_id="worker-b",
            expected_incumbent_candidate_id="seed",
            candidate_id="candidate-b",
            outcome=ProposalOutcome(
                run_id="run-b",
                status="promoted",
                decision="promote",
                failure_category=None,
                cost_usd=1.0,
                primary_delta=0.2,
            ),
        )


def test_mechanism_composition_is_a_new_controlled_proposal(tmp_path: Path) -> None:
    store = CampaignStore(tmp_path / "campaign.sqlite")
    store.create_campaign("campaign-1", quality_budget=3, cost_budget_usd=10.0)
    store.add_hypothesis("campaign-1", "compose-1", "SFT repair and RLVR compose")

    proposal_id = store.submit_proposal(
        ExperimentProposal(
            campaign_id="campaign-1",
            hypothesis_id="compose-1",
            method="compose",
            task_name="gsm8k",
            parent_candidate_id="seed",
            comparison_group_id="composition-ablation",
            comparison_axis="composition_strategy",
            parameters={"composition_strategy": "sft_then_rlvr"},
            mechanism_parent_ids=("sft-run-1", "rlvr-run-1"),
        )
    )

    assert proposal_id.startswith("proposal-")


def test_planner_uses_failures_budget_and_exposure(tmp_path: Path) -> None:
    store = CampaignStore(tmp_path / "campaign.sqlite")
    store.create_campaign("campaign-1", quality_budget=2, cost_budget_usd=2.0)
    store.add_hypothesis("campaign-1", "hypothesis-1", "candidate improves")
    store.record_suite_exposure(
        "campaign-1",
        suite_id="promotion-a",
        suite_version="v1",
        candidate_id="candidate-a",
    )
    recommendation = store.recommend_next_experiment(
        "campaign-1",
        failure_category="verifier_disagreement",
    )
    assert recommendation.category == "evaluation_repair"

    proposal_a = store.submit_proposal(
        ExperimentProposal(
            campaign_id="campaign-1",
            hypothesis_id="hypothesis-1",
            method="sft",
            task_name="gsm8k",
            parent_candidate_id="seed",
            comparison_group_id="budget-ablation",
            comparison_axis="candidate",
            parameters={"candidate": "a"},
            estimated_cost_usd=1.5,
        )
    )
    proposal_b = store.submit_proposal(
        ExperimentProposal(
            campaign_id="campaign-1",
            hypothesis_id="hypothesis-1",
            method="sft",
            task_name="gsm8k",
            parent_candidate_id="seed",
            comparison_group_id="budget-ablation",
            comparison_axis="candidate",
            parameters={"candidate": "b"},
            estimated_cost_usd=1.0,
        )
    )
    assert store.claim_proposal(
        proposal_a,
        worker_id="worker-a",
        ttl_seconds=30,
    ) is not None
    with pytest.raises(ValueError, match="cost budget exceeded"):
        store.claim_proposal(
            proposal_b,
            worker_id="worker-b",
            ttl_seconds=30,
        )

    exposure = store.suite_exposure("campaign-1", "promotion-a", "v1")
    assert exposure == 1
    with pytest.raises(ValueError, match="rotation required"):
        store.require_suite_available(
            "campaign-1",
            suite_id="promotion-a",
            suite_version="v1",
            max_exposures=1,
        )


def test_expired_lease_releases_campaign_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from post_train_engine import campaign as campaign_module

    now = {"value": 100.0}
    monkeypatch.setattr(campaign_module.time, "time", lambda: now["value"])
    store = CampaignStore(tmp_path / "campaign.sqlite")
    store.create_campaign("campaign-1", quality_budget=1, cost_budget_usd=1.0)
    store.add_hypothesis("campaign-1", "hypothesis-1", "candidate improves")

    def proposal(candidate: str) -> ExperimentProposal:
        return ExperimentProposal(
            campaign_id="campaign-1",
            hypothesis_id="hypothesis-1",
            method="sft",
            task_name="gsm8k",
            parent_candidate_id="seed",
            comparison_group_id="lease-budget",
            comparison_axis="candidate",
            parameters={"candidate": candidate},
            estimated_cost_usd=1.0,
        )

    proposal_a = store.submit_proposal(proposal("a"))
    proposal_b = store.submit_proposal(proposal("b"))
    assert store.claim_proposal(
        proposal_a,
        worker_id="worker-a",
        ttl_seconds=10,
    ) is not None
    with pytest.raises(ValueError, match="quality budget exceeded"):
        store.claim_proposal(proposal_b, worker_id="worker-b", ttl_seconds=10)

    now["value"] = 111.0
    assert store.claim_proposal(
        proposal_b,
        worker_id="worker-b",
        ttl_seconds=10,
    ) is not None
