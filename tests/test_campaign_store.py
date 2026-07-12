from __future__ import annotations

from pathlib import Path

import pytest

from post_train_engine.campaign import (
    CampaignStore,
    ExperimentExplanation,
    ExperimentProposal,
    ProposalOutcome,
)


def _explanation() -> ExperimentExplanation:
    return ExperimentExplanation(
        problem_statement="The incumbent fails a measured capability slice.",
        causal_explanation="The declared intervention repairs that failure mechanism.",
        hard_to_vary_details=("intervention", "affected_slice"),
        risky_prediction="The affected slice improves without protected regression.",
        rejection_criterion="Reject if the affected slice does not improve or any gate fails.",
    )


def test_proposal_requires_a_falsifiable_hard_to_vary_explanation() -> None:
    with pytest.raises(ValueError, match="hard_to_vary_details"):
        ExperimentExplanation(
            problem_statement="A measured failure exists.",
            causal_explanation="A causal mechanism may repair it.",
            hard_to_vary_details=(),
            risky_prediction="A protected metric will improve.",
            rejection_criterion="Reject if it does not improve.",
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
        explanation=_explanation(),
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
    assert store.get_proposal_explanation(proposal_id) == _explanation()
    with pytest.raises(ValueError, match="duplicate proposal"):
        store.submit_proposal(proposal)

    lease = store.claim_proposal(proposal_id, worker_id="worker-1", ttl_seconds=30)
    assert lease.worker_id == "worker-1"
    assert store.claim_proposal(proposal_id, worker_id="worker-2", ttl_seconds=30) is None

    store.finalize_run(
        campaign_id="campaign-1",
        proposal_id=proposal_id,
        worker_id="worker-1",
        fence_token=lease.fence_token,
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
            explanation=_explanation(),
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
            explanation=_explanation(),
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
    lease_a = store.claim_proposal(proposal_a, worker_id="worker-a", ttl_seconds=30)
    lease_b = store.claim_proposal(proposal_b, worker_id="worker-b", ttl_seconds=30)
    assert lease_a is not None
    assert lease_b is not None

    store.finalize_run(
        campaign_id="campaign-1",
        proposal_id=proposal_a,
        worker_id="worker-a",
        fence_token=lease_a.fence_token,
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
            fence_token=lease_b.fence_token,
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
            explanation=_explanation(),
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
    assert recommendation.problem_statement
    assert recommendation.causal_explanation
    assert recommendation.hard_to_vary_details
    assert recommendation.risky_prediction
    assert recommendation.rejection_criterion

    proposal_a = store.submit_proposal(
        ExperimentProposal(
            explanation=_explanation(),
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
            explanation=_explanation(),
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
            explanation=_explanation(),
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


def test_reacquired_lease_fences_stale_worker_with_same_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from post_train_engine import campaign as campaign_module

    now = {"value": 100.0}
    monkeypatch.setattr(campaign_module.time, "time", lambda: now["value"])
    store = CampaignStore(tmp_path / "campaign.sqlite")
    store.create_campaign("campaign-1", quality_budget=2, cost_budget_usd=2.0)
    store.add_hypothesis("campaign-1", "hypothesis-1", "candidate improves")
    store.initialize_incumbent("campaign-1", candidate_id="seed", run_id="seed-run")
    proposal_id = store.submit_proposal(
        ExperimentProposal(
            explanation=_explanation(),
            campaign_id="campaign-1",
            hypothesis_id="hypothesis-1",
            method="sft",
            task_name="gsm8k",
            parent_candidate_id="seed",
            comparison_group_id="fencing",
            comparison_axis="candidate",
            parameters={"candidate": "a"},
            estimated_cost_usd=0.5,
        )
    )
    stale = store.claim_proposal(
        proposal_id,
        worker_id="worker-1",
        ttl_seconds=10,
    )
    assert stale is not None
    now["value"] = 111.0
    current = store.claim_proposal(
        proposal_id,
        worker_id="worker-1",
        ttl_seconds=10,
    )
    assert current is not None
    assert current.fence_token > stale.fence_token

    with pytest.raises(ValueError, match="fencing token"):
        store.finalize_run(
            campaign_id="campaign-1",
            proposal_id=proposal_id,
            worker_id="worker-1",
            fence_token=stale.fence_token,
            expected_incumbent_candidate_id="seed",
            candidate_id="candidate-a",
            outcome=ProposalOutcome(
                run_id="run-a",
                status="promoted",
                decision="promote",
                failure_category=None,
                cost_usd=0.5,
                primary_delta=0.1,
            ),
        )


def test_lease_renewal_preserves_fence_and_rejects_stale_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from post_train_engine import campaign as campaign_module

    now = {"value": 100.0}
    monkeypatch.setattr(campaign_module.time, "time", lambda: now["value"])
    store = CampaignStore(tmp_path / "campaign.sqlite")
    store.create_campaign("campaign-1", quality_budget=1, cost_budget_usd=1.0)
    store.add_hypothesis("campaign-1", "hypothesis-1", "candidate improves")
    proposal_id = store.submit_proposal(
        ExperimentProposal(
            explanation=_explanation(),
            campaign_id="campaign-1",
            hypothesis_id="hypothesis-1",
            method="sft",
            task_name="gsm8k",
            parent_candidate_id="seed",
            comparison_group_id="renewal",
            comparison_axis="candidate",
            parameters={"candidate": "a"},
            estimated_cost_usd=0.5,
        )
    )
    lease = store.claim_proposal(
        proposal_id,
        worker_id="worker-1",
        ttl_seconds=10,
    )
    assert lease is not None

    now["value"] = 105.0
    renewed = store.renew_lease(
        proposal_id,
        worker_id="worker-1",
        fence_token=lease.fence_token,
        ttl_seconds=20,
    )
    assert renewed.fence_token == lease.fence_token
    assert renewed.expires_at == 125.0

    with pytest.raises(ValueError, match="fencing token"):
        store.renew_lease(
            proposal_id,
            worker_id="worker-1",
            fence_token=lease.fence_token - 1,
            ttl_seconds=20,
        )


def test_finalization_rejects_measured_cost_above_campaign_budget(
    tmp_path: Path,
) -> None:
    store = CampaignStore(tmp_path / "campaign.sqlite")
    store.create_campaign("campaign-1", quality_budget=1, cost_budget_usd=1.0)
    store.add_hypothesis("campaign-1", "hypothesis-1", "candidate improves")
    store.initialize_incumbent("campaign-1", candidate_id="seed", run_id="seed-run")
    proposal_id = store.submit_proposal(
        ExperimentProposal(
            explanation=_explanation(),
            campaign_id="campaign-1",
            hypothesis_id="hypothesis-1",
            method="sft",
            task_name="gsm8k",
            parent_candidate_id="seed",
            comparison_group_id="settlement",
            comparison_axis="candidate",
            parameters={"candidate": "a"},
            estimated_cost_usd=0.5,
        )
    )
    lease = store.claim_proposal(
        proposal_id,
        worker_id="worker-1",
        ttl_seconds=30,
    )
    assert lease is not None

    with pytest.raises(ValueError, match="measured cost budget exceeded"):
        store.finalize_run(
            campaign_id="campaign-1",
            proposal_id=proposal_id,
            worker_id="worker-1",
            fence_token=lease.fence_token,
            expected_incumbent_candidate_id="seed",
            candidate_id="candidate-a",
            outcome=ProposalOutcome(
                run_id="run-a",
                status="promoted",
                decision="promote",
                failure_category=None,
                cost_usd=1.01,
                primary_delta=0.1,
            ),
        )

    assert store.current_incumbent("campaign-1")["candidate_id"] == "seed"


def test_provider_billing_settlement_cannot_promote_before_receipt(
    tmp_path: Path,
) -> None:
    store = CampaignStore(tmp_path / "campaign.sqlite")
    store.create_campaign("campaign-1", quality_budget=1, cost_budget_usd=1.5)
    store.add_hypothesis("campaign-1", "hypothesis-1", "candidate improves")
    store.initialize_incumbent("campaign-1", candidate_id="seed", run_id="seed-run")
    proposal_id = store.submit_proposal(
        ExperimentProposal(
            explanation=_explanation(),
            campaign_id="campaign-1",
            hypothesis_id="hypothesis-1",
            method="grpo",
            task_name="gsm8k",
            parent_candidate_id="seed",
            comparison_group_id="runpod",
            comparison_axis="method",
            parameters={"method": "grpo"},
            estimated_cost_usd=0.75,
        )
    )
    lease = store.claim_proposal(proposal_id, worker_id="worker-1", ttl_seconds=30)
    assert lease is not None

    store.stage_run_settlement(
        campaign_id="campaign-1",
        proposal_id=proposal_id,
        worker_id="worker-1",
        fence_token=lease.fence_token,
        expected_incumbent_candidate_id="seed",
        candidate_id="candidate-a",
        provider_resource_id="pod-1",
        outcome=ProposalOutcome(
            run_id="run-a",
            status="promoted",
            decision="promote",
            failure_category=None,
            cost_usd=None,
            primary_delta=0.2,
        ),
        promotion_suite_id="suite-1",
        promotion_suite_version="v1",
        max_suite_exposures=2,
    )

    assert store.current_incumbent("campaign-1")["candidate_id"] == "seed"
    assert store.campaign_usage("campaign-1")["pending_settlements"] == 1
    assert store.suite_exposure("campaign-1", "suite-1", "v1") == 1

    with pytest.raises(ValueError, match="differs from Run evidence"):
        store.stage_run_settlement(
            campaign_id="campaign-1",
            proposal_id=proposal_id,
            worker_id="worker-1",
            fence_token=lease.fence_token,
            expected_incumbent_candidate_id="seed",
            candidate_id="candidate-a",
            provider_resource_id="pod-1",
            outcome=ProposalOutcome(
                run_id="run-a",
                status="promoted",
                decision="promote",
                failure_category=None,
                cost_usd=None,
                primary_delta=0.3,
            ),
            promotion_suite_id="suite-1",
            promotion_suite_version="v1",
            max_suite_exposures=2,
        )

    store.settle_run(
        proposal_id,
        provider_resource_id="pod-1",
        billed_cost_usd=0.44,
        billing_receipt_sha256="sha256:" + "a" * 64,
    )

    assert store.current_incumbent("campaign-1")["candidate_id"] == "candidate-a"
    assert store.get_outcome(proposal_id).cost_usd == 0.44
    assert store.campaign_usage("campaign-1")["pending_settlements"] == 0


def test_provider_billing_over_budget_leaves_incumbent_unchanged(
    tmp_path: Path,
) -> None:
    store = CampaignStore(tmp_path / "campaign.sqlite")
    store.create_campaign("campaign-1", quality_budget=1, cost_budget_usd=1.0)
    store.add_hypothesis("campaign-1", "hypothesis-1", "candidate improves")
    store.initialize_incumbent("campaign-1", candidate_id="seed", run_id="seed-run")
    proposal_id = store.submit_proposal(
        ExperimentProposal(
            explanation=_explanation(),
            campaign_id="campaign-1",
            hypothesis_id="hypothesis-1",
            method="grpo",
            task_name="gsm8k",
            parent_candidate_id="seed",
            comparison_group_id="runpod",
            comparison_axis="method",
            parameters={"method": "grpo"},
            estimated_cost_usd=0.5,
        )
    )
    lease = store.claim_proposal(proposal_id, worker_id="worker-1", ttl_seconds=30)
    assert lease is not None
    store.stage_run_settlement(
        campaign_id="campaign-1",
        proposal_id=proposal_id,
        worker_id="worker-1",
        fence_token=lease.fence_token,
        expected_incumbent_candidate_id="seed",
        candidate_id="candidate-a",
        provider_resource_id="pod-1",
        outcome=ProposalOutcome(
            run_id="run-a",
            status="promoted",
            decision="promote",
            failure_category=None,
            cost_usd=None,
            primary_delta=0.2,
        ),
    )

    with pytest.raises(ValueError, match="measured cost budget exceeded"):
        store.settle_run(
            proposal_id,
            provider_resource_id="pod-1",
            billed_cost_usd=1.01,
            billing_receipt_sha256="sha256:" + "b" * 64,
        )

    assert store.current_incumbent("campaign-1")["candidate_id"] == "seed"
