from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Lock

import pytest

from post_train_engine.engine import (
    CANONICAL_STAGE_ORDER,
    RunEngine,
    RunPlan,
    StageOutput,
)
from post_train_engine.evals.contract import EvalContract
from post_train_engine.evidence_safety import (
    ContentSeparationCertificate,
    VerifierSeparation,
)
from post_train_engine.run_bundle import RunBundle
from post_train_engine.campaign import (
    CampaignStore,
    ExperimentProposal,
    ProposalOutcome,
)
from post_train_engine.diagnostics import write_run_diagnostics
from post_train_engine.reports import write_run_report


class FakeStageAdapter:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def execute_stage(
        self,
        stage: str,
        plan: RunPlan,
        prior: dict[str, StageOutput],
    ) -> StageOutput:
        self.calls.append(stage)
        artifact = Path(plan.output_dir) / "artifacts" / f"{stage}.json"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        body: dict[str, object] = {"stage": stage}
        artifacts = {stage: str(artifact)}
        if stage == "evaluate":
            evaluation = {
                "primary_metric": "accuracy",
                "evaluation_contract_hash": plan.evaluation_contract.contract_hash,
                "metrics": {"accuracy": 0.5, "mean_tokens": 1.0},
                "slices": {"easy_stable": {"accuracy": 0.5}},
                "examples": [
                    {
                        "example_id": "eval-1",
                        "correct": True,
                        "parse_ok": True,
                        "tokens": 1,
                        "bucket": "easy_stable",
                    },
                    {
                        "example_id": "eval-2",
                        "correct": False,
                        "parse_ok": True,
                        "tokens": 1,
                        "bucket": "easy_stable",
                    },
                ],
            }
            baseline = artifact.parent / "baseline_eval.json"
            candidate = artifact.parent / "candidate_eval.json"
            baseline.write_text(
                json.dumps({**evaluation, "artifact_id": "baseline"}),
                encoding="utf-8",
            )
            candidate.write_text(
                json.dumps({**evaluation, "artifact_id": "candidate"}),
                encoding="utf-8",
            )
            artifacts = {
                "baseline_eval": str(baseline),
                "candidate_eval": str(candidate),
            }
        artifact.write_text(json.dumps(body), encoding="utf-8")
        return StageOutput(
            artifacts=artifacts,
            values={},
            cost_usd=0.0,
        )


def _eval_contract(
    example_ids: tuple[str, ...] = ("eval-1", "eval-2"),
    *,
    suite_id: str = "fixture-promotion",
    suite_version: str = "v1",
) -> EvalContract:
    return EvalContract.from_components(
        suite_id=suite_id,
        suite_version=suite_version,
        example_ids=example_ids,
        example_content=[{"id": value, "prompt": f"prompt:{value}"} for value in example_ids],
        prompt_contract={"template": "fixture-v1"},
        verifier_contract={"verifier": "exact-match-v1"},
        generation_contract={"temperature": 0.0},
        primary_metric="accuracy",
    )


def test_eval_contract_binds_protected_row_content() -> None:
    left = _eval_contract()
    right = EvalContract.from_components(
        suite_id=left.suite_id,
        suite_version=left.suite_version,
        example_ids=("eval-1", "eval-2"),
        example_content=[
            {"id": "eval-1", "prompt": "changed"},
            {"id": "eval-2", "prompt": "prompt:eval-2"},
        ],
        prompt_contract={"template": "fixture-v1"},
        verifier_contract={"verifier": "exact-match-v1"},
        generation_contract={"temperature": 0.0},
        primary_metric="accuracy",
    )

    assert left.example_content_sha256 != right.example_content_sha256
    assert left.contract_hash != right.contract_hash


def _content_separation(
    *,
    training_count: int = 0,
    protected_count: int = 2,
) -> ContentSeparationCertificate:
    return ContentSeparationCertificate(
        training_count=training_count,
        protected_count=protected_count,
        ngram_size=3,
        max_allowed_jaccard=0.8,
        observed_max_jaccard=0.0,
    )


def _verifier_separation() -> VerifierSeparation:
    return VerifierSeparation(
        verifier_kind="executable_ground_truth",
        training_verifier_id="exact-match-v1",
        promotion_verifier_id="exact-match-v1",
    )


def test_run_plan_rejects_promotion_rows_outside_eval_contract(tmp_path: Path) -> None:
    raw = _fixture_plan(tmp_path / "contract-mismatch").model_dump(mode="json")
    raw["promotion_example_ids"] = ["different-row"]

    with pytest.raises(ValueError, match="evaluation contract example IDs"):
        RunPlan.model_validate(raw)


def test_run_plan_rejects_separation_certificate_count_mismatch(
    tmp_path: Path,
) -> None:
    raw = _fixture_plan(tmp_path / "separation-count-mismatch").model_dump(
        mode="json"
    )
    raw["content_separation"] = ContentSeparationCertificate(
        training_count=1,
        protected_count=2,
        ngram_size=3,
        max_allowed_jaccard=0.8,
        observed_max_jaccard=0.0,
    ).model_dump(mode="json")
    raw["verifier_separation"] = VerifierSeparation(
        verifier_kind="executable_ground_truth",
        training_verifier_id="exact-match-v1",
        promotion_verifier_id="exact-match-v1",
    ).model_dump(mode="json")

    with pytest.raises(ValueError, match="content separation counts"):
        RunPlan.model_validate(raw)


class ThreadCoordinator:
    def __init__(
        self,
        *,
        is_main_process: bool,
        rank: int,
        barrier: Barrier,
        errors: dict[int, str],
        lock: Lock,
    ) -> None:
        self.is_main_process = is_main_process
        self._rank = rank
        self._barrier = barrier
        self._errors = errors
        self._lock = lock

    def wait(self) -> None:
        self._barrier.wait(timeout=10)

    def collect_errors(self, error: str | None) -> tuple[str, ...]:
        if error is not None:
            with self._lock:
                self._errors[self._rank] = error
        self.wait()
        with self._lock:
            result = tuple(self._errors[index] for index in sorted(self._errors))
        self.wait()
        if self.is_main_process:
            with self._lock:
                self._errors.clear()
        self.wait()
        return result


class WorkerStageAdapter:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def execute_stage(
        self,
        stage: str,
        _plan: RunPlan,
        _prior: dict[str, StageOutput],
    ) -> StageOutput:
        self.calls.append(stage)
        return StageOutput(cost_usd=0.0)


class PromotingStageAdapter(FakeStageAdapter):
    def execute_stage(
        self,
        stage: str,
        plan: RunPlan,
        prior: dict[str, StageOutput],
    ) -> StageOutput:
        output = super().execute_stage(stage, plan, prior)
        if stage == "evaluate":
            candidate_path = Path(output.artifacts["candidate_eval"])
            body = json.loads(candidate_path.read_text(encoding="utf-8"))
            body["examples"][1]["correct"] = True
            body["metrics"]["accuracy"] = 1.0
            body["slices"]["easy_stable"]["accuracy"] = 1.0
            candidate_path.write_text(json.dumps(body), encoding="utf-8")
        return output


class SubstitutedEvaluationRowsAdapter(FakeStageAdapter):
    def execute_stage(
        self,
        stage: str,
        plan: RunPlan,
        prior: dict[str, StageOutput],
    ) -> StageOutput:
        output = super().execute_stage(stage, plan, prior)
        if stage == "evaluate":
            for name in ("baseline_eval", "candidate_eval"):
                path = Path(output.artifacts[name])
                body = json.loads(path.read_text(encoding="utf-8"))
                body["examples"][0]["example_id"] = "substituted-row"
                path.write_text(json.dumps(body), encoding="utf-8")
        return output


class MissingCostStageAdapter(FakeStageAdapter):
    def execute_stage(
        self,
        stage: str,
        plan: RunPlan,
        prior: dict[str, StageOutput],
    ) -> StageOutput:
        output = super().execute_stage(stage, plan, prior)
        if stage == "prepare":
            return StageOutput(
                artifacts=output.artifacts,
                cost_missing_reason="price unavailable",
            )
        return output


class FailingStageAdapter(FakeStageAdapter):
    def execute_stage(
        self,
        stage: str,
        plan: RunPlan,
        prior: dict[str, StageOutput],
    ) -> StageOutput:
        if stage == "train":
            raise RuntimeError("terminal fixture failure")
        return super().execute_stage(stage, plan, prior)


class PrepareFailingAdapter(FakeStageAdapter):
    def execute_stage(
        self,
        stage: str,
        plan: RunPlan,
        prior: dict[str, StageOutput],
    ) -> StageOutput:
        if stage == "prepare":
            raise RuntimeError("prepare fixture failure")
        return super().execute_stage(stage, plan, prior)


class ExternalArtifactAdapter(FakeStageAdapter):
    def __init__(self, external_path: Path) -> None:
        super().__init__()
        self.external_path = external_path

    def execute_stage(
        self,
        stage: str,
        plan: RunPlan,
        prior: dict[str, StageOutput],
    ) -> StageOutput:
        if stage == "prepare":
            self.external_path.write_text("external", encoding="utf-8")
            return StageOutput(
                artifacts={"external": str(self.external_path)},
                cost_usd=0.0,
            )
        return super().execute_stage(stage, plan, prior)


def test_run_plan_rejects_campaign_parent_mismatch(tmp_path: Path) -> None:
    raw = _fixture_plan(tmp_path / "parent-mismatch").model_dump(mode="json")
    raw["campaign"] = {
        "database_path": str(tmp_path / "campaign.sqlite"),
        "campaign_id": "campaign-1",
        "proposal_id": "proposal-1",
        "worker_id": "worker-1",
        "expected_incumbent_candidate_id": "different-parent",
    }

    with pytest.raises(ValueError, match="campaign expected Incumbent"):
        RunPlan.model_validate(raw)


def test_run_engine_owns_stage_order_manifest_and_resume(tmp_path: Path) -> None:
    plan = RunPlan(
        run_id="run-1",
        candidate_id="candidate-1",
        parent_candidate_id="seed",
        task_name="fixture",
        model_id="model-1",
        output_dir=str(tmp_path / "run-1"),
        promotion_example_ids=("eval-1", "eval-2"),
        evaluation_contract=_eval_contract(),
        content_separation=_content_separation(),
        verifier_separation=_verifier_separation(),
        inputs={
            "model": {
                "kind": "model",
                "requested_id": "model-1",
                "resolved_id": "model-1",
                "resolved_revision": "v1",
                "resolution_state": "exact",
            },
            "dataset": {
                "kind": "dataset",
                "requested_id": "data-1",
                "resolved_id": "data-1",
                "resolved_revision": "v1",
                "resolution_state": "exact",
            },
        },
    )
    adapter = FakeStageAdapter()

    first = RunEngine().execute(plan, adapter)
    second_adapter = FakeStageAdapter()
    second = RunEngine().execute(plan, second_adapter)

    assert adapter.calls == [
        stage for stage in CANONICAL_STAGE_ORDER if stage != "promote"
    ]
    assert second_adapter.calls == []
    assert first.manifest == second.manifest
    assert first.manifest.status == "rejected"
    assert "next_experiment" in first.manifest.artifacts
    recommendation_path = (
        Path(plan.output_dir) / first.manifest.artifacts["next_experiment"].path
    )
    recommendation = json.loads(recommendation_path.read_text(encoding="utf-8"))
    assert recommendation["category"] == "data_repair"
    assert RunBundle.load(plan.output_dir).validate()["status"] == "ok"
    assert len(first.stage_receipts) == len(CANONICAL_STAGE_ORDER)
    assert all(receipt.duration_seconds >= 0 for receipt in first.stage_receipts)


def test_run_engine_rejects_stage_artifacts_outside_run_root(tmp_path: Path) -> None:
    plan = _fixture_plan(tmp_path / "contained-run")
    adapter = ExternalArtifactAdapter(tmp_path / "external.txt")

    with pytest.raises(ValueError, match="outside run directory"):
        RunEngine().execute(plan, adapter)

    assert adapter.calls == []


def test_run_engine_seals_promotion_rows_from_normal_surfaces(tmp_path: Path) -> None:
    plan = _fixture_plan(tmp_path / "sealed-evaluation")
    execution = RunEngine().execute(plan, FakeStageAdapter())
    bundle = RunBundle.load(plan.output_dir)

    for name in ("baseline_eval", "candidate_eval"):
        assert execution.manifest.artifacts[name].visibility == "sealed"
        with pytest.raises(ValueError, match="sealed artifact"):
            bundle.artifact_path(name)
        assert bundle.artifact_path(name, allow_sealed=True).is_file()

    summary = write_run_report(plan.output_dir)
    assert "baseline_eval" not in summary["artifacts"]
    assert "candidate_eval" not in summary["artifacts"]
    assert "stage_receipt_evaluate" not in summary["artifacts"]


def test_run_engine_rejects_substituted_evaluation_rows(tmp_path: Path) -> None:
    plan = _fixture_plan(tmp_path / "substituted-evaluation-rows")

    with pytest.raises(ValueError, match="rows do not match"):
        RunEngine().execute(plan, SubstitutedEvaluationRowsAdapter())


def test_report_rejects_mutated_artifact_instead_of_consuming_it(tmp_path: Path) -> None:
    plan = _fixture_plan(tmp_path / "mutated-report-artifact")
    execution = RunEngine().execute(plan, FakeStageAdapter())
    decision = Path(plan.output_dir) / execution.manifest.artifacts[
        "promotion_decision"
    ].path
    decision.write_text('{"decision":"promote"}', encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        write_run_report(plan.output_dir)


def test_run_plan_rejects_promotion_data_in_training_view(tmp_path: Path) -> None:
    from pydantic import ValidationError
    import pytest

    with pytest.raises(ValidationError, match="protected evaluation"):
        RunPlan(
            run_id="run-1",
            candidate_id="candidate-1",
            task_name="fixture",
            model_id="model-1",
            output_dir=str(tmp_path / "run-1"),
            training_example_ids=("promotion-1",),
            selection_example_ids=("selection-1",),
            promotion_example_ids=("promotion-1",),
            evaluation_contract=_eval_contract(("promotion-1",)),
            content_separation=_content_separation(
                training_count=1,
                protected_count=2,
            ),
            verifier_separation=_verifier_separation(),
            inputs={
                "model": {
                    "kind": "model",
                    "requested_id": "model-1",
                    "resolved_id": "model-1",
                    "resolved_revision": "v1",
                    "resolution_state": "exact",
                },
                "dataset": {
                    "kind": "dataset",
                    "requested_id": "data-1",
                    "resolved_id": "data-1",
                    "resolved_revision": "v1",
                    "resolution_state": "exact",
                },
            },
        )


def test_run_engine_fails_closed_when_budgeted_stage_cost_is_unknown(
    tmp_path: Path,
) -> None:
    plan = _fixture_plan(tmp_path / "unknown-cost", max_cost_usd=1.0)

    with pytest.raises(ValueError, match="cannot certify cost budget"):
        RunEngine().execute(plan, MissingCostStageAdapter())


def test_run_engine_persists_terminal_stage_failure_bundle(tmp_path: Path) -> None:
    plan = _fixture_plan(tmp_path / "failed-run")

    with pytest.raises(RuntimeError, match="terminal fixture failure"):
        RunEngine().execute(plan, FailingStageAdapter())

    bundle = RunBundle.load(plan.output_dir)
    assert bundle.manifest.status == "failed"
    assert "failure" in bundle.manifest.artifacts
    assert bundle.validate()["status"] == "ok"
    summary = write_run_report(plan.output_dir)
    diagnostics = write_run_diagnostics(plan.output_dir)
    assert summary["promotion_decision"] == "failed"
    assert diagnostics["primary_category"] == "stage_failure"
    resumed = RunEngine().execute(plan, FailingStageAdapter())
    assert resumed.manifest.status == "failed"


def test_run_engine_rejects_mutated_receipt_artifact_on_resume(tmp_path: Path) -> None:
    plan = _fixture_plan(tmp_path / "mutated-receipt")
    RunEngine().execute(plan, FakeStageAdapter())
    (Path(plan.output_dir) / "manifest.json").unlink()

    prepare_artifact = Path(plan.output_dir) / "artifacts" / "prepare.json"
    prepare_artifact.write_text('{"stage":"tampered"}', encoding="utf-8")

    with pytest.raises(ValueError, match="artifact hash mismatch"):
        RunEngine().execute(plan, FailingStageAdapter())


def test_run_engine_coordinates_distributed_stages_with_one_writer(
    tmp_path: Path,
) -> None:
    plan = RunPlan(
        run_id="distributed-run",
        candidate_id="candidate-1",
        parent_candidate_id="seed",
        task_name="fixture",
        model_id="model-1",
        output_dir=str(tmp_path / "distributed-run"),
        promotion_example_ids=("eval-1", "eval-2"),
        evaluation_contract=_eval_contract(),
        content_separation=_content_separation(),
        verifier_separation=_verifier_separation(),
        inputs={
            "model": {
                "kind": "model",
                "requested_id": "model-1",
                "resolved_id": "model-1",
                "resolved_revision": "v1",
                "resolution_state": "exact",
            },
            "dataset": {
                "kind": "dataset",
                "requested_id": "data-1",
                "resolved_id": "data-1",
                "resolved_revision": "v1",
                "resolution_state": "exact",
            },
        },
    )
    barrier = Barrier(2)
    errors: dict[int, str] = {}
    lock = Lock()
    main_adapter = FakeStageAdapter()
    worker_adapter = WorkerStageAdapter()

    with ThreadPoolExecutor(max_workers=2) as pool:
        main_future = pool.submit(
            RunEngine().execute,
            plan,
            main_adapter,
            coordinator=ThreadCoordinator(
                is_main_process=True,
                rank=0,
                barrier=barrier,
                errors=errors,
                lock=lock,
            ),
        )
        worker_future = pool.submit(
            RunEngine().execute,
            plan,
            worker_adapter,
            coordinator=ThreadCoordinator(
                is_main_process=False,
                rank=1,
                barrier=barrier,
                errors=errors,
                lock=lock,
            ),
        )
        main_execution = main_future.result(timeout=20)
        worker_execution = worker_future.result(timeout=20)

    expected_calls = [stage for stage in CANONICAL_STAGE_ORDER if stage != "promote"]
    assert main_adapter.calls == expected_calls
    assert worker_adapter.calls == expected_calls
    assert worker_execution.manifest == main_execution.manifest
    assert RunBundle.load(plan.output_dir).validate()["status"] == "ok"


def test_run_engine_propagates_distributed_stage_failure_without_deadlock(
    tmp_path: Path,
) -> None:
    plan = _fixture_plan(tmp_path / "distributed-failure")
    barrier = Barrier(2)
    errors: dict[int, str] = {}
    lock = Lock()
    main_coordinator = ThreadCoordinator(
        is_main_process=True,
        rank=0,
        barrier=barrier,
        errors=errors,
        lock=lock,
    )
    worker_coordinator = ThreadCoordinator(
        is_main_process=False,
        rank=1,
        barrier=barrier,
        errors=errors,
        lock=lock,
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        main_future = pool.submit(
            RunEngine().execute,
            plan,
            FailingStageAdapter(),
            coordinator=main_coordinator,
        )
        worker_future = pool.submit(
            RunEngine().execute,
            plan,
            WorkerStageAdapter(),
            coordinator=worker_coordinator,
        )
        with pytest.raises(RuntimeError, match="terminal fixture failure"):
            main_future.result(timeout=20)
        with pytest.raises(RuntimeError, match="distributed stage train failed"):
            worker_future.result(timeout=20)

    bundle = RunBundle.load(plan.output_dir)
    assert bundle.manifest.status == "failed"
    assert bundle.validate()["status"] == "ok"


def test_run_engine_atomically_finalizes_campaign_and_reconciles_rerun(
    tmp_path: Path,
) -> None:
    campaign = CampaignStore(tmp_path / "campaign.sqlite")
    campaign.create_campaign("campaign-1", quality_budget=3, cost_budget_usd=10.0)
    campaign.add_hypothesis("campaign-1", "hypothesis-1", "candidate improves accuracy")
    proposal_id = campaign.submit_proposal(
        ExperimentProposal(
            campaign_id="campaign-1",
            hypothesis_id="hypothesis-1",
            method="fixture",
            task_name="fixture",
            parent_candidate_id="seed",
            comparison_group_id="group-1",
            comparison_axis="learning_rate",
            parameters={"learning_rate": 1e-5},
        )
    )
    campaign.initialize_incumbent(
        "campaign-1",
        candidate_id="seed",
        run_id="seed-run",
    )
    assert campaign.claim_proposal(
        proposal_id,
        worker_id="worker-1",
        ttl_seconds=60,
    ) is not None
    plan = RunPlan(
        run_id="campaign-run",
        candidate_id="candidate-1",
        parent_candidate_id="seed",
        task_name="fixture",
        model_id="model-1",
        output_dir=str(tmp_path / "campaign-run"),
        promotion_example_ids=("eval-1", "eval-2"),
        evaluation_contract=_eval_contract(
            suite_id="suite-1",
            suite_version="v1",
        ),
        content_separation=_content_separation(),
        verifier_separation=_verifier_separation(),
        inputs={
            "model": {
                "kind": "model",
                "requested_id": "model-1",
                "resolved_id": "model-1",
                "resolved_revision": "v1",
                "resolution_state": "exact",
            },
            "dataset": {
                "kind": "dataset",
                "requested_id": "data-1",
                "resolved_id": "data-1",
                "resolved_revision": "v1",
                "resolution_state": "exact",
            },
        },
        promotion_gate={
            "min_examples": 2,
            "min_primary_delta": 0.1,
            "min_primary_ci_low": -1.0,
            "max_mcnemar_p": 1.0,
            "max_parse_regression": 0.0,
            "max_easy_regression": 0.0,
            "max_token_increase_ratio": 2.0,
        },
        campaign={
            "database_path": str(campaign.path),
            "campaign_id": "campaign-1",
            "proposal_id": proposal_id,
            "worker_id": "worker-1",
            "expected_incumbent_candidate_id": "seed",
            "promotion_suite_id": "suite-1",
            "promotion_suite_version": "v1",
            "max_suite_exposures": 3,
        },
    )

    first = RunEngine().execute(plan, PromotingStageAdapter())
    second = RunEngine().execute(plan, PromotingStageAdapter())

    assert first.manifest.status == "promoted"
    assert second.manifest == first.manifest
    assert campaign.get_outcome(proposal_id).status == "promoted"
    assert campaign.current_incumbent("campaign-1")["candidate_id"] == "candidate-1"
    assert campaign.suite_exposure("campaign-1", "suite-1", "v1") == 1
    campaign.add_hypothesis("campaign-1", "hypothesis-2", "later candidate improves")
    later_proposal_id = campaign.submit_proposal(
        ExperimentProposal(
            campaign_id="campaign-1",
            hypothesis_id="hypothesis-2",
            method="sft",
            task_name="fixture",
            parent_candidate_id="candidate-1",
            comparison_group_id="later-candidate",
            comparison_axis="candidate",
            parameters={"candidate": "candidate-2"},
        )
    )
    campaign.claim_proposal(later_proposal_id, worker_id="worker-2", ttl_seconds=30)
    campaign.finalize_run(
        campaign_id="campaign-1",
        proposal_id=later_proposal_id,
        worker_id="worker-2",
        expected_incumbent_candidate_id="candidate-1",
        candidate_id="candidate-2",
        outcome=ProposalOutcome(
            run_id="later-run",
            status="promoted",
            decision="promote",
            failure_category=None,
            cost_usd=1.0,
            primary_delta=0.2,
        ),
    )
    third = RunEngine().execute(plan, PromotingStageAdapter())
    assert third.manifest == first.manifest


def test_campaign_prepare_failure_does_not_consume_suite_exposure(
    tmp_path: Path,
) -> None:
    campaign = CampaignStore(tmp_path / "campaign.sqlite")
    campaign.create_campaign("campaign-1", quality_budget=3, cost_budget_usd=10.0)
    campaign.add_hypothesis("campaign-1", "hypothesis-1", "candidate improves")
    proposal_id = campaign.submit_proposal(
        ExperimentProposal(
            campaign_id="campaign-1",
            hypothesis_id="hypothesis-1",
            method="fixture",
            task_name="fixture",
            parent_candidate_id="seed",
            comparison_group_id="group-1",
            comparison_axis="method",
            parameters={"method": "fixture"},
        )
    )
    campaign.initialize_incumbent("campaign-1", candidate_id="seed", run_id="seed-run")
    campaign.claim_proposal(proposal_id, worker_id="worker-1", ttl_seconds=60)
    raw = _fixture_plan(tmp_path / "prepare-failure").model_dump(mode="json")
    raw["evaluation_contract"] = _eval_contract(
        suite_id="suite-1",
        suite_version="v1",
    ).model_dump(mode="json")
    raw["campaign"] = {
        "database_path": str(campaign.path),
        "campaign_id": "campaign-1",
        "proposal_id": proposal_id,
        "worker_id": "worker-1",
        "expected_incumbent_candidate_id": "seed",
        "promotion_suite_id": "suite-1",
        "promotion_suite_version": "v1",
        "max_suite_exposures": 3,
    }
    plan = RunPlan.model_validate(raw)

    with pytest.raises(RuntimeError, match="prepare fixture failure"):
        RunEngine().execute(plan, PrepareFailingAdapter())

    assert campaign.get_outcome(proposal_id).status == "failed"
    assert campaign.suite_exposure("campaign-1", "suite-1", "v1") == 0


def _fixture_plan(output_dir: Path, *, max_cost_usd: float | None = None) -> RunPlan:
    return RunPlan(
        run_id=output_dir.name,
        candidate_id="candidate-1",
        parent_candidate_id="seed",
        task_name="fixture",
        model_id="model-1",
        output_dir=str(output_dir),
        max_cost_usd=max_cost_usd,
        promotion_example_ids=("eval-1", "eval-2"),
        evaluation_contract=_eval_contract(),
        content_separation=_content_separation(),
        verifier_separation=_verifier_separation(),
        inputs={
            "model": {
                "kind": "model",
                "requested_id": "model-1",
                "resolved_id": "model-1",
                "resolved_revision": "v1",
                "resolution_state": "exact",
            },
            "dataset": {
                "kind": "dataset",
                "requested_id": "data-1",
                "resolved_id": "data-1",
                "resolved_revision": "v1",
                "resolution_state": "exact",
            },
        },
    )
