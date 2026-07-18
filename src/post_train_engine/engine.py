"""Canonical stage machine for post-training Runs."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from post_train_engine.artifacts import require_valid_run_bundle
from post_train_engine.campaign import (
    CampaignStore,
    ProposalOutcome,
    recommend_from_run,
)
from post_train_engine.evaluation_roles import EvaluationRoles
from post_train_engine.evidence_safety import (
    ContentSeparationCertificate,
    VerifierSeparation,
)
from post_train_engine.evals.contract import EvalContract, hash_example_ids
from post_train_engine.evals.promotion import (
    PromotionGateConfig,
    canary_decision_from_artifact,
    decide_promotion,
    load_eval_artifact,
    write_promotion_decision,
)
from post_train_engine.run_bundle import (
    ResolvedInput,
    RunBundle,
    RunManifest,
    SourceIdentity,
    capture_source_identity,
    make_artifact_ref,
    write_manifest_atomic,
)

RunStage = Literal[
    "resolve",
    "prepare",
    "data",
    "evidence",
    "train",
    "select",
    "evaluate",
    "promote",
    "finalize",
]
CANONICAL_STAGE_ORDER: tuple[RunStage, ...] = (
    "resolve",
    "prepare",
    "data",
    "evidence",
    "train",
    "select",
    "evaluate",
    "promote",
    "finalize",
)
ADAPTER_STAGE_ORDER: tuple[RunStage, ...] = tuple(
    stage for stage in CANONICAL_STAGE_ORDER if stage not in {"resolve", "promote"}
)
_FROZEN_FORBID = ConfigDict(frozen=True, extra="forbid")


class CampaignBinding(BaseModel):
    model_config = _FROZEN_FORBID

    database_path: str = Field(..., min_length=1)
    campaign_id: str = Field(..., min_length=1)
    proposal_id: str = Field(..., min_length=1)
    worker_id: str = Field(..., min_length=1)
    lease_fence_token: int = Field(..., gt=0)
    expected_incumbent_candidate_id: str = Field(..., min_length=1)
    settlement_mode: Literal["stage_measured", "provider_billing"] = "stage_measured"
    provider_resource_id: str | None = None
    promotion_suite_id: str | None = None
    promotion_suite_version: str | None = None
    max_suite_exposures: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _suite_binding_is_all_or_none(self) -> CampaignBinding:
        values = (
            self.promotion_suite_id,
            self.promotion_suite_version,
            self.max_suite_exposures,
        )
        if any(value is not None for value in values) and not all(
            value is not None for value in values
        ):
            raise ValueError(
                "campaign promotion suite requires id, version, and max exposures"
            )
        if self.settlement_mode == "provider_billing" and not self.provider_resource_id:
            raise ValueError(
                "provider billing settlement requires provider_resource_id"
            )
        if (
            self.settlement_mode == "stage_measured"
            and self.provider_resource_id is not None
        ):
            raise ValueError(
                "stage-measured settlement cannot name provider_resource_id"
            )
        return self


class RunPlan(BaseModel):
    """Immutable compiled plan consumed by the RunEngine."""

    model_config = _FROZEN_FORBID

    certification_mode: Literal["non_certifying_smoke", "certifying"]
    run_id: str = Field(..., min_length=1)
    candidate_id: str = Field(..., min_length=1)
    parent_candidate_id: str | None = None
    task_name: str = Field(..., min_length=1)
    model_id: str = Field(..., min_length=1)
    output_dir: str = Field(..., min_length=1)
    source_root: str | None = None
    source_identity: SourceIdentity | None = None
    inputs: dict[str, ResolvedInput]
    training_example_ids: tuple[str, ...] = ()
    selection_example_ids: tuple[str, ...] = ()
    diagnostic_example_ids: tuple[str, ...] = ()
    promotion_example_ids: tuple[str, ...] = ()
    canary_example_ids: tuple[str, ...] = ()
    unseen_example_ids: tuple[str, ...] = ()
    evaluation_contract: EvalContract
    canary_evaluation_contract: EvalContract | None = None
    content_separation: ContentSeparationCertificate
    verifier_separation: VerifierSeparation
    max_cost_usd: float | None = Field(default=None, ge=0.0)
    distributed_timeout_seconds: float = Field(default=120.0, gt=0.0)
    promotion_gate: dict[str, Any] = Field(default_factory=dict)
    campaign: CampaignBinding | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_scientific_boundaries(self) -> RunPlan:
        roles = EvaluationRoles(
            selection_example_ids=self.selection_example_ids,
            diagnostic_example_ids=self.diagnostic_example_ids,
            promotion_example_ids=self.promotion_example_ids,
            canary_example_ids=self.canary_example_ids,
            unseen_example_ids=self.unseen_example_ids,
        )
        roles.require_training_eligible(self.training_example_ids)
        if not self.promotion_example_ids:
            raise ValueError("RunPlan requires promotion example IDs")
        if self.evaluation_contract.role != "promotion":
            raise ValueError(
                "RunPlan promotion evaluation contract requires promotion role"
            )
        if self.evaluation_contract.example_ids_sha256 != hash_example_ids(
            self.promotion_example_ids
        ):
            raise ValueError(
                "RunPlan promotion example IDs do not match the evaluation contract example IDs"
            )
        if bool(self.canary_example_ids) != bool(self.canary_evaluation_contract):
            raise ValueError(
                "RunPlan canary IDs and canary evaluation contract must be declared together"
            )
        if (
            self.canary_evaluation_contract is not None
            and self.canary_evaluation_contract.role != "canary"
        ):
            raise ValueError("RunPlan canary evaluation contract requires canary role")
        if (
            self.canary_evaluation_contract is not None
            and self.canary_evaluation_contract.example_ids_sha256
            != hash_example_ids(self.canary_example_ids)
        ):
            raise ValueError(
                "RunPlan canary example IDs do not match the canary evaluation contract"
            )
        expected_protected_count = len(roles.protected_example_ids)
        if (
            self.content_separation.training_count != len(self.training_example_ids)
            or self.content_separation.protected_count != expected_protected_count
        ):
            raise ValueError(
                "RunPlan content separation counts do not match training and protected roles"
            )
        missing = {"model", "dataset"}.difference(self.inputs)
        if missing:
            raise ValueError(
                "RunPlan missing core inputs: " + ", ".join(sorted(missing))
            )
        if self.certification_mode == "certifying" and self.campaign is None:
            raise ValueError("certifying RunPlan requires a campaign binding")
        non_exact = sorted(
            name
            for name, identity in self.inputs.items()
            if identity.resolution_state != "exact"
        )
        if self.certification_mode == "certifying" and non_exact:
            raise ValueError(
                "certifying RunPlan requires exact input identities: "
                + ", ".join(non_exact)
            )
        if (
            self.certification_mode == "non_certifying_smoke"
            and self.campaign is not None
        ):
            raise ValueError("non-certifying smoke RunPlan cannot bind a campaign")
        if (
            self.campaign is not None
            and self.parent_candidate_id
            != self.campaign.expected_incumbent_candidate_id
        ):
            raise ValueError(
                "RunPlan parent_candidate_id must match the campaign expected Incumbent"
            )
        if self.campaign is not None and self.campaign.promotion_suite_id is not None:
            if (
                self.campaign.promotion_suite_id != self.evaluation_contract.suite_id
                or self.campaign.promotion_suite_version
                != self.evaluation_contract.suite_version
            ):
                raise ValueError(
                    "campaign promotion suite must match the evaluation contract"
                )
        return self

    @property
    def plan_hash(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


class StageOutput(BaseModel):
    """Serializable output from one adapter stage."""

    model_config = _FROZEN_FORBID

    artifacts: dict[str, str] = Field(default_factory=dict)
    values: dict[str, Any] = Field(default_factory=dict)
    cost_usd: float | None = Field(default=None, ge=0.0)
    cost_missing_reason: str | None = None

    @model_validator(mode="after")
    def _cost_evidence_must_be_explicit(self) -> StageOutput:
        if self.cost_usd is None and not self.cost_missing_reason:
            raise ValueError(
                "stage output requires measured cost_usd or cost_missing_reason"
            )
        return self


class StageReceipt(BaseModel):
    model_config = _FROZEN_FORBID

    stage: RunStage
    plan_hash: str
    duration_seconds: float = Field(..., ge=0.0)
    output: StageOutput
    artifact_sha256: dict[str, str]


class StageIntent(BaseModel):
    model_config = _FROZEN_FORBID

    stage: RunStage
    plan_hash: str
    execute: bool


class RunExecution(BaseModel):
    model_config = _FROZEN_FORBID

    manifest: RunManifest
    stage_receipts: tuple[StageReceipt, ...]


@dataclass(frozen=True)
class RunResolution:
    """Resolved immutable plan and adapter produced inside RunEngine."""

    plan: RunPlan
    adapter: RunStageAdapter
    output: StageOutput


class RunStageAdapter(Protocol):
    def execute_stage(
        self,
        stage: RunStage,
        plan: RunPlan,
        prior: dict[str, StageOutput],
    ) -> StageOutput:
        """Execute adapter-specific work without owning promotion mutation."""


class RunCoordinator(Protocol):
    is_main_process: bool

    def wait(self, timeout_seconds: float) -> None:
        """Synchronize all ranks at an engine-owned stage transition."""

    def collect_errors(
        self,
        error: str | None,
        timeout_seconds: float,
    ) -> tuple[str, ...]:
        """Collect rank-local error summaries in rank order."""


class RunEngine:
    """Own canonical stage order, receipts, budgets, and Run finalization."""

    def execute(
        self,
        resolver: Callable[[], RunResolution],
        *,
        coordinator: RunCoordinator | None = None,
        resolution_timeout_seconds: float = 120.0,
    ) -> RunExecution:
        """Resolve consequential inputs, freeze the RunPlan, and execute it."""

        started = time.perf_counter()
        resolution: RunResolution | None = None
        local_error: Exception | None = None
        try:
            resolution = resolver()
        except Exception as exc:
            local_error = exc
        distributed_errors = (
            ()
            if coordinator is None
            else coordinator.collect_errors(
                None
                if local_error is None
                else f"{type(local_error).__name__}: {local_error}",
                resolution_timeout_seconds,
            )
        )
        if local_error is not None:
            raise local_error
        if distributed_errors:
            raise RuntimeError(
                "distributed resolution failed: " + "; ".join(distributed_errors)
            )
        if resolution is None:
            raise RuntimeError("resolver produced no RunResolution")
        return self._execute(
            resolution.plan,
            resolution.adapter,
            coordinator=coordinator,
            resolution_output=resolution.output,
            resolution_duration_seconds=time.perf_counter() - started,
        )

    def _execute(
        self,
        plan: RunPlan,
        adapter: RunStageAdapter,
        *,
        coordinator: RunCoordinator | None,
        resolution_output: StageOutput,
        resolution_duration_seconds: float,
    ) -> RunExecution:
        run_dir = Path(plan.output_dir).resolve()
        manifest_path = run_dir / "manifest.json"
        is_writer = coordinator is None or coordinator.is_main_process
        if is_writer:
            run_dir.mkdir(parents=True, exist_ok=True)
            if plan.source_identity is None:
                plan = plan.model_copy(
                    update={
                        "source_identity": capture_source_identity(
                            plan.source_root or Path.cwd()
                        )
                    }
                )
            frozen_plan_path = run_dir / "state" / "run_plan.json"
            if frozen_plan_path.is_file():
                frozen_plan = RunPlan.model_validate_json(
                    frozen_plan_path.read_text(encoding="utf-8")
                )
                if frozen_plan.plan_hash != plan.plan_hash:
                    raise ValueError("frozen RunPlan does not match resolved plan")
            else:
                _write_model_atomic(frozen_plan_path, plan)
        if coordinator is not None:
            coordinator.wait(plan.distributed_timeout_seconds)
            if not is_writer:
                plan = RunPlan.model_validate_json(
                    (run_dir / "state" / "run_plan.json").read_text(encoding="utf-8")
                )
        if manifest_path.is_file():
            bundle = RunBundle.load(run_dir)
            require_valid_run_bundle(run_dir)
            campaign_error: Exception | None = None
            if is_writer:
                try:
                    self._finalize_campaign(plan, bundle.manifest)
                except Exception as exc:
                    campaign_error = exc
            distributed_errors = (
                ()
                if coordinator is None
                else coordinator.collect_errors(
                    None
                    if campaign_error is None
                    else f"{type(campaign_error).__name__}: {campaign_error}",
                    plan.distributed_timeout_seconds,
                )
            )
            if campaign_error is not None:
                raise campaign_error
            if distributed_errors:
                raise RuntimeError(
                    "distributed campaign reconciliation failed: "
                    + "; ".join(distributed_errors)
                )
            return RunExecution(
                manifest=bundle.manifest,
                stage_receipts=self._load_receipts(
                    run_dir,
                    plan,
                    require_complete=bundle.manifest.status
                    in {"pending_settlement", "promoted", "rejected"},
                ),
            )
        campaign_error = None
        if is_writer:
            try:
                self._require_campaign_available(plan)
            except Exception as exc:
                campaign_error = exc
        distributed_errors = (
            ()
            if coordinator is None
            else coordinator.collect_errors(
                None
                if campaign_error is None
                else f"{type(campaign_error).__name__}: {campaign_error}",
                plan.distributed_timeout_seconds,
            )
        )
        if campaign_error is not None:
            raise campaign_error
        if distributed_errors:
            raise RuntimeError(
                "distributed campaign validation failed: "
                + "; ".join(distributed_errors)
            )

        receipts: list[StageReceipt] = []
        prior: dict[str, StageOutput] = {}
        spent = 0.0
        for stage in CANONICAL_STAGE_ORDER:
            receipt_path = run_dir / "state" / f"{stage}.json"
            receipt: StageReceipt | None = None
            execute_stage = False
            if is_writer:
                if receipt_path.is_file():
                    receipt = self._read_receipt(receipt_path, plan, stage)
                else:
                    execute_stage = True
                if coordinator is not None:
                    _write_model_atomic(
                        run_dir / "state" / f"{stage}.intent.json",
                        StageIntent(
                            stage=stage,
                            plan_hash=plan.plan_hash,
                            execute=execute_stage,
                        ),
                    )
            if coordinator is not None:
                coordinator.wait(plan.distributed_timeout_seconds)
                if not is_writer:
                    intent_path = run_dir / "state" / f"{stage}.intent.json"
                    intent = StageIntent.model_validate_json(
                        intent_path.read_text(encoding="utf-8")
                    )
                    if intent.plan_hash != plan.plan_hash or intent.stage != stage:
                        raise ValueError(
                            f"stage intent does not match RunPlan: {intent_path}"
                        )
                    execute_stage = intent.execute
                    if not execute_stage:
                        receipt = self._read_receipt(receipt_path, plan, stage)

            local_error: Exception | None = None
            output: StageOutput | None = None
            if execute_stage:
                started = time.perf_counter()
                try:
                    if stage == "resolve":
                        output = resolution_output if is_writer else None
                    elif stage == "promote":
                        output = (
                            self._promote(run_dir, plan, prior) if is_writer else None
                        )
                    else:
                        output = adapter.execute_stage(stage, plan, dict(prior))
                except Exception as exc:
                    local_error = exc
            distributed_errors = (
                ()
                if coordinator is None
                else coordinator.collect_errors(
                    None
                    if local_error is None
                    else f"{type(local_error).__name__}: {local_error}",
                    plan.distributed_timeout_seconds,
                )
            )
            if local_error is not None or distributed_errors:
                failure = local_error or RuntimeError(
                    f"distributed stage {stage} failed: {'; '.join(distributed_errors)}"
                )
                if is_writer:
                    self._write_failed_run(
                        run_dir=run_dir,
                        plan=plan,
                        receipts=receipts,
                        failed_stage=stage,
                        error=failure,
                        spent=spent,
                    )
                if local_error is not None:
                    raise local_error
                raise failure
            persistence_error: Exception | None = None
            if execute_stage and is_writer:
                try:
                    if output is None:
                        raise ValueError(f"writer produced no output for stage {stage}")
                    receipt = StageReceipt(
                        stage=stage,
                        plan_hash=plan.plan_hash,
                        duration_seconds=(
                            resolution_duration_seconds
                            if stage == "resolve"
                            else time.perf_counter() - started
                        ),
                        output=output,
                        artifact_sha256=_hash_stage_artifacts(run_dir, output),
                    )
                    _write_model_atomic(receipt_path, receipt)
                except Exception as exc:
                    persistence_error = exc
            persistence_errors = (
                ()
                if coordinator is None
                else coordinator.collect_errors(
                    None
                    if persistence_error is None
                    else f"{type(persistence_error).__name__}: {persistence_error}",
                    plan.distributed_timeout_seconds,
                )
            )
            if persistence_error is not None:
                raise persistence_error
            if persistence_errors:
                raise RuntimeError(
                    f"distributed stage {stage} receipt persistence failed: "
                    + "; ".join(persistence_errors)
                )
            if coordinator is not None and not is_writer:
                receipt = self._read_receipt(receipt_path, plan, stage)
            if receipt is None:
                raise ValueError(
                    f"missing stage receipt after execution: {receipt_path}"
                )
            if receipt.output.cost_usd is not None:
                spent += receipt.output.cost_usd
            if plan.max_cost_usd is not None and receipt.output.cost_usd is None:
                error = ValueError(
                    f"cannot certify cost budget after {stage}: "
                    f"{receipt.output.cost_missing_reason}"
                )
                if is_writer:
                    self._write_failed_run(
                        run_dir=run_dir,
                        plan=plan,
                        receipts=[*receipts, receipt],
                        failed_stage=stage,
                        error=error,
                        spent=spent,
                    )
                raise error
            if plan.max_cost_usd is not None and spent > plan.max_cost_usd:
                error = ValueError(
                    f"Run cost budget exceeded after {stage}: {spent} > {plan.max_cost_usd}"
                )
                if is_writer:
                    self._write_failed_run(
                        run_dir=run_dir,
                        plan=plan,
                        receipts=[*receipts, receipt],
                        failed_stage=stage,
                        error=error,
                        spent=spent,
                    )
                raise error
            prior[stage] = receipt.output
            receipts.append(receipt)

        execution: RunExecution | None = None
        finalization_error: Exception | None = None
        if is_writer:
            try:
                execution = self._finalize_run(
                    run_dir=run_dir,
                    plan=plan,
                    receipts=receipts,
                    prior=prior,
                    spent=spent,
                )
            except Exception as exc:
                finalization_error = exc
        distributed_errors = (
            ()
            if coordinator is None
            else coordinator.collect_errors(
                None
                if finalization_error is None
                else f"{type(finalization_error).__name__}: {finalization_error}",
                plan.distributed_timeout_seconds,
            )
        )
        if finalization_error is not None:
            raise finalization_error
        if distributed_errors:
            raise RuntimeError(
                "distributed finalization failed: " + "; ".join(distributed_errors)
            )
        if execution is None:
            bundle = RunBundle.load(run_dir)
            require_valid_run_bundle(run_dir)
            execution = RunExecution(
                manifest=bundle.manifest,
                stage_receipts=tuple(receipts),
            )
        return execution

    def settle_provider_billing(
        self,
        plan: RunPlan,
        *,
        receipt: Any,
        receipt_path: str | Path,
    ) -> RunManifest:
        """Settle one provider-billed campaign Run and finish its bundle."""

        from post_train_engine.runpod_control_plane import PodBillingReceipt

        binding = plan.campaign
        if binding is None or binding.settlement_mode != "provider_billing":
            raise ValueError("RunPlan is not bound to provider billing settlement")
        parsed_receipt = PodBillingReceipt.model_validate(receipt)
        if (
            parsed_receipt.settlement_state != "settled"
            or parsed_receipt.amount_usd is None
        ):
            raise ValueError("provider billing receipt is not settled")
        if parsed_receipt.pod_id != binding.provider_resource_id:
            raise ValueError("provider billing receipt belongs to a different resource")
        run_dir = Path(plan.output_dir).resolve()
        frozen_plan_path = run_dir / "state" / "run_plan.json"
        if not frozen_plan_path.is_file():
            raise ValueError("provider settlement requires the frozen RunPlan")
        frozen_plan = RunPlan.model_validate_json(
            frozen_plan_path.read_text(encoding="utf-8")
        )
        comparable_frozen = frozen_plan.model_copy(
            update={"source_identity": plan.source_identity}
        )
        if comparable_frozen.plan_hash != plan.plan_hash:
            raise ValueError("settlement RunPlan does not match the frozen RunPlan")
        plan = frozen_plan
        binding = plan.campaign
        if binding is None or binding.settlement_mode != "provider_billing":
            raise ValueError(
                "frozen RunPlan is not bound to provider billing settlement"
            )
        bundle = RunBundle.load(run_dir)
        require_valid_run_bundle(run_dir)
        if bundle.manifest.metadata.get("plan_hash") != plan.plan_hash:
            raise ValueError("Run bundle does not match the settlement RunPlan")
        path = Path(receipt_path).resolve()
        receipt_from_disk = PodBillingReceipt.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        if receipt_from_disk != parsed_receipt:
            raise ValueError(
                "provider billing receipt file differs from supplied receipt"
            )
        receipt_ref = make_artifact_ref(
            run_dir,
            path,
            kind="provider_billing_receipt",
        )
        if bundle.manifest.status in {"promoted", "rejected"}:
            if bundle.manifest.metadata.get("campaign_settlement") != "settled":
                raise ValueError(
                    "terminal provider-billed manifest lacks settlement evidence"
                )
            if (
                bundle.manifest.metadata.get("provider_resource_id")
                != parsed_receipt.pod_id
                or float(bundle.manifest.metadata.get("cost_usd", -1.0))
                != parsed_receipt.amount_usd
            ):
                raise ValueError("provider billing receipt differs from settled bundle")
            if bundle.manifest.artifacts.get("provider_billing_receipt") != receipt_ref:
                raise ValueError("provider billing receipt differs from settled bundle")
            return bundle.manifest
        if bundle.manifest.status != "pending_settlement":
            raise ValueError("provider-billed Run is not pending settlement")
        CampaignStore(binding.database_path).settle_run(
            binding.proposal_id,
            provider_resource_id=str(binding.provider_resource_id),
            billed_cost_usd=parsed_receipt.amount_usd,
            billing_receipt_sha256=receipt_ref.sha256,
        )
        decision_ref = bundle.manifest.artifacts.get("promotion_decision")
        if decision_ref is None:
            raise ValueError("pending settlement bundle lacks promotion decision")
        decision = json.loads((run_dir / decision_ref.path).read_text(encoding="utf-8"))
        terminal_status = (
            "promoted" if decision.get("decision") == "promote" else "rejected"
        )
        manifest = bundle.manifest.model_copy(
            update={
                "status": terminal_status,
                "artifacts": {
                    **bundle.manifest.artifacts,
                    "provider_billing_receipt": receipt_ref,
                },
                "metadata": {
                    **bundle.manifest.metadata,
                    "campaign_settlement": "settled",
                    "cost_usd": parsed_receipt.amount_usd,
                    "cost_certifying": True,
                    "provider_resource_id": parsed_receipt.pod_id,
                },
            }
        )
        write_manifest_atomic(run_dir, manifest)
        require_valid_run_bundle(run_dir)
        return manifest

    def _read_receipt(
        self,
        path: Path,
        plan: RunPlan,
        stage: RunStage,
    ) -> StageReceipt:
        try:
            receipt = StageReceipt.model_validate_json(path.read_text(encoding="utf-8"))
        except ValidationError as exc:
            raise ValueError(
                "stage receipt is incompatible with the current evidence schema; "
                f"do not backfill historical evidence, use a new run_id: {path}"
            ) from exc
        if receipt.plan_hash != plan.plan_hash or receipt.stage != stage:
            raise ValueError(f"stage receipt does not match RunPlan: {path}")
        run_dir = path.parent.parent.resolve()
        if _hash_stage_artifacts(run_dir, receipt.output) != receipt.artifact_sha256:
            raise ValueError(f"stage receipt artifact hash mismatch: {path}")
        return receipt

    def _finalize_run(
        self,
        *,
        run_dir: Path,
        plan: RunPlan,
        receipts: list[StageReceipt],
        prior: dict[str, StageOutput],
        spent: float,
    ) -> RunExecution:
        for receipt in receipts:
            persisted = self._read_receipt(
                run_dir / "state" / f"{receipt.stage}.json",
                plan,
                receipt.stage,
            )
            if persisted != receipt:
                raise ValueError(
                    f"stage receipt changed before finalization: {receipt.stage}"
                )
        promotion = prior["promote"]
        decision = promotion.values.get("decision")
        if decision not in {"promote", "reject"}:
            raise ValueError("promote stage must return decision=promote or reject")
        artifacts: dict[str, Any] = {}
        for receipt in receipts:
            for name, raw_path in receipt.output.artifacts.items():
                if name in artifacts:
                    raise ValueError(f"duplicate artifact name across stages: {name}")
                artifacts[name] = make_artifact_ref(
                    run_dir,
                    raw_path,
                    kind=name,
                    visibility=(
                        "sealed"
                        if name
                        in {
                            "baseline_eval",
                            "candidate_eval",
                            "canary_eval",
                            "unseen_eval",
                        }
                        else "standard"
                    ),
                )
            receipt_path = run_dir / "state" / f"{receipt.stage}.json"
            artifacts[f"stage_receipt_{receipt.stage}"] = make_artifact_ref(
                run_dir,
                receipt_path,
                kind="stage_receipt",
                visibility="sealed" if receipt.stage == "evaluate" else "standard",
            )
        if "promotion_decision" not in artifacts:
            raise ValueError("promote stage must emit promotion_decision artifact")

        settlement_pending = (
            plan.campaign is not None
            and plan.campaign.settlement_mode == "provider_billing"
        )
        manifest = RunManifest(
            run_id=plan.run_id,
            candidate_id=plan.candidate_id,
            parent_candidate_id=plan.parent_candidate_id,
            task_name=plan.task_name,
            model_id=plan.model_id,
            status=(
                "pending_settlement"
                if settlement_pending
                else ("promoted" if decision == "promote" else "rejected")
            ),
            source=plan.source_identity
            or capture_source_identity(plan.source_root or Path.cwd()),
            inputs=plan.inputs,
            artifacts=artifacts,
            metadata={
                **plan.metadata,
                **(
                    {}
                    if plan.campaign is None
                    else {
                        "campaign": {
                            "campaign_id": plan.campaign.campaign_id,
                            "proposal_id": plan.campaign.proposal_id,
                        }
                    }
                ),
                "plan_hash": plan.plan_hash,
                "certification_mode": plan.certification_mode,
                **(
                    {}
                    if plan.campaign is None
                    else {
                        "campaign_settlement": (
                            "pending" if settlement_pending else "settled"
                        )
                    }
                ),
                "evaluation_contract": {
                    **plan.evaluation_contract.model_dump(mode="json"),
                    "contract_hash": plan.evaluation_contract.contract_hash,
                },
                "content_separation": plan.content_separation.model_dump(mode="json"),
                "verifier_separation": plan.verifier_separation.model_dump(mode="json"),
                "stage_order": list(CANONICAL_STAGE_ORDER),
                "cost_usd": spent,
                **_cost_certification_metadata(receipts),
            },
        )
        write_manifest_atomic(run_dir, manifest)
        require_valid_run_bundle(run_dir)
        self._finalize_campaign(plan, manifest)
        return RunExecution(manifest=manifest, stage_receipts=tuple(receipts))

    def _write_failed_run(
        self,
        *,
        run_dir: Path,
        plan: RunPlan,
        receipts: list[StageReceipt],
        failed_stage: RunStage,
        error: Exception,
        spent: float,
    ) -> None:
        failure_path = run_dir / "failure.json"
        _write_json_atomic(
            failure_path,
            {
                "stage": failed_stage,
                "error_type": type(error).__name__,
                "message": "stage failed; the original exception was re-raised to the caller",
            },
        )
        artifacts: dict[str, Any] = {
            "failure": make_artifact_ref(run_dir, failure_path, kind="failure")
        }
        for receipt in receipts:
            for name, raw_path in receipt.output.artifacts.items():
                if name not in artifacts:
                    artifacts[name] = make_artifact_ref(run_dir, raw_path, kind=name)
            receipt_path = run_dir / "state" / f"{receipt.stage}.json"
            if receipt_path.is_file():
                artifacts[f"stage_receipt_{receipt.stage}"] = make_artifact_ref(
                    run_dir,
                    receipt_path,
                    kind="stage_receipt",
                    visibility="sealed" if receipt.stage == "evaluate" else "standard",
                )
        manifest = RunManifest(
            run_id=plan.run_id,
            candidate_id=plan.candidate_id,
            parent_candidate_id=plan.parent_candidate_id,
            task_name=plan.task_name,
            model_id=plan.model_id,
            status="failed",
            source=plan.source_identity
            or capture_source_identity(plan.source_root or Path.cwd()),
            inputs=plan.inputs,
            artifacts=artifacts,
            metadata={
                **plan.metadata,
                "plan_hash": plan.plan_hash,
                "certification_mode": plan.certification_mode,
                "evaluation_contract": {
                    **plan.evaluation_contract.model_dump(mode="json"),
                    "contract_hash": plan.evaluation_contract.contract_hash,
                },
                "content_separation": plan.content_separation.model_dump(mode="json"),
                "verifier_separation": plan.verifier_separation.model_dump(mode="json"),
                "stage_order": list(CANONICAL_STAGE_ORDER),
                "failed_stage": failed_stage,
                "cost_usd": spent,
                **_cost_certification_metadata(receipts),
            },
        )
        write_manifest_atomic(run_dir, manifest)
        require_valid_run_bundle(run_dir)
        self._finalize_campaign(plan, manifest)

    def _require_campaign_available(self, plan: RunPlan) -> None:
        binding = plan.campaign
        if binding is None or binding.promotion_suite_id is None:
            return
        CampaignStore(binding.database_path).require_suite_available(
            binding.campaign_id,
            suite_id=binding.promotion_suite_id,
            suite_version=str(binding.promotion_suite_version),
            max_exposures=int(binding.max_suite_exposures),
        )

    def _finalize_campaign(self, plan: RunPlan, manifest: RunManifest) -> None:
        binding = plan.campaign
        if binding is None:
            return
        status = manifest.status
        if status == "failed":
            failure_ref = manifest.artifacts.get("failure")
            failure = (
                {}
                if failure_ref is None
                else json.loads(
                    (Path(plan.output_dir).resolve() / failure_ref.path).read_text(
                        encoding="utf-8"
                    )
                )
            )
            decision_name = "failed"
            failure_category = str(failure.get("error_type") or "stage_failure")
            primary_delta = None
        else:
            decision_ref = manifest.artifacts.get("promotion_decision")
            if decision_ref is None:
                raise ValueError(
                    "campaign finalization requires promotion_decision evidence"
                )
            decision_path = Path(plan.output_dir).resolve() / decision_ref.path
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            if not isinstance(decision, dict):
                raise ValueError("promotion_decision must be a JSON object")
            rejection_reasons = decision.get("rejection_reasons", [])
            failure_category = (
                str(rejection_reasons[0]).split(":", 1)[0]
                if isinstance(rejection_reasons, list) and rejection_reasons
                else None
            )
            raw_decision = str(decision.get("decision"))
            if raw_decision not in {"promote", "reject"}:
                raise ValueError("promotion_decision has invalid decision")
            decision_name = raw_decision
            primary_delta = float(decision["primary_delta"])
        suite_was_exposed = status != "failed" or str(
            manifest.metadata.get("failed_stage")
        ) in {"evaluate", "promote", "finalize"}
        outcome_status = (
            "failed"
            if decision_name == "failed"
            else ("promoted" if decision_name == "promote" else "rejected")
        )
        outcome = ProposalOutcome(
            run_id=manifest.run_id,
            status=outcome_status,
            decision=decision_name,
            failure_category=failure_category,
            cost_usd=(
                float(manifest.metadata.get("cost_usd", 0.0))
                if manifest.metadata.get("cost_certifying") is True
                and binding.settlement_mode == "stage_measured"
                else None
            ),
            primary_delta=primary_delta,
        )
        store = CampaignStore(binding.database_path)
        if binding.settlement_mode == "provider_billing":
            store.stage_run_settlement(
                campaign_id=binding.campaign_id,
                proposal_id=binding.proposal_id,
                worker_id=binding.worker_id,
                fence_token=binding.lease_fence_token,
                expected_incumbent_candidate_id=binding.expected_incumbent_candidate_id,
                candidate_id=manifest.candidate_id,
                provider_resource_id=str(binding.provider_resource_id),
                outcome=outcome,
                promotion_suite_id=(
                    binding.promotion_suite_id if suite_was_exposed else None
                ),
                promotion_suite_version=(
                    binding.promotion_suite_version if suite_was_exposed else None
                ),
                max_suite_exposures=(
                    binding.max_suite_exposures if suite_was_exposed else None
                ),
            )
            return
        store.finalize_run(
            campaign_id=binding.campaign_id,
            proposal_id=binding.proposal_id,
            worker_id=binding.worker_id,
            fence_token=binding.lease_fence_token,
            expected_incumbent_candidate_id=binding.expected_incumbent_candidate_id,
            candidate_id=manifest.candidate_id,
            outcome=outcome,
            promotion_suite_id=(
                binding.promotion_suite_id if suite_was_exposed else None
            ),
            promotion_suite_version=(
                binding.promotion_suite_version if suite_was_exposed else None
            ),
            max_suite_exposures=(
                binding.max_suite_exposures if suite_was_exposed else None
            ),
        )

    def _promote(
        self,
        run_dir: Path,
        plan: RunPlan,
        prior: dict[str, StageOutput],
    ) -> StageOutput:
        evaluation = prior.get("evaluate")
        if evaluation is None:
            raise ValueError("promotion requires the evaluate stage")
        try:
            baseline_path = evaluation.artifacts["baseline_eval"]
            candidate_path = evaluation.artifacts["candidate_eval"]
        except KeyError as exc:
            raise ValueError(
                "evaluate stage must emit baseline_eval and candidate_eval artifacts"
            ) from exc
        baseline_eval = load_eval_artifact(baseline_path)
        candidate_eval = load_eval_artifact(candidate_path)
        expected_contract_hash = plan.evaluation_contract.contract_hash
        if (
            baseline_eval.evaluation_contract_hash != expected_contract_hash
            or candidate_eval.evaluation_contract_hash != expected_contract_hash
        ):
            raise ValueError(
                "promotion artifacts do not match the RunPlan evaluation contract"
            )
        if (
            baseline_eval.primary_metric != plan.evaluation_contract.primary_metric
            or candidate_eval.primary_metric != plan.evaluation_contract.primary_metric
        ):
            raise ValueError(
                "promotion artifacts do not match the evaluation contract primary metric"
            )
        expected_example_ids_hash = plan.evaluation_contract.example_ids_sha256
        if any(
            hash_example_ids(tuple(row.example_id for row in artifact.examples))
            != expected_example_ids_hash
            for artifact in (baseline_eval, candidate_eval)
        ):
            raise ValueError(
                "promotion artifact rows do not match the evaluation contract"
            )
        canary_decision = None
        if plan.canary_example_ids:
            try:
                canary_path = evaluation.artifacts["canary_eval"]
            except KeyError as exc:
                raise ValueError(
                    "evaluate stage must emit canary_eval when the RunPlan declares canaries"
                ) from exc
            canary_eval = load_eval_artifact(canary_path)
            canary_contract = plan.canary_evaluation_contract
            if canary_contract is None:
                raise ValueError("RunPlan omitted the canary evaluation contract")
            if (
                canary_eval.evaluation_contract_hash != canary_contract.contract_hash
                or canary_eval.primary_metric != canary_contract.primary_metric
                or hash_example_ids(
                    tuple(row.example_id for row in canary_eval.examples)
                )
                != canary_contract.example_ids_sha256
            ):
                raise ValueError("canary_eval rows do not match the RunPlan canary IDs")
            canary_decision = canary_decision_from_artifact(canary_eval)
        decision = decide_promotion(
            baseline_eval,
            candidate_eval,
            PromotionGateConfig(**plan.promotion_gate),
            canary_decision=canary_decision,
        )
        if plan.certification_mode == "non_certifying_smoke":
            decision = replace(
                decision,
                decision="reject",
                gates={**decision.gates, "certification_mode": "fail"},
                rejection_reasons=(
                    *decision.rejection_reasons,
                    "non_certifying_smoke",
                ),
            )
        path = run_dir / "promotion_decision.json"
        write_promotion_decision(decision, path)
        evidence = prior.get("evidence")
        recommendation = recommend_from_run(
            decision.to_dict(),
            evidence_signal=(
                None
                if evidence is None
                else evidence.values.get("next_experiment_signal")
            ),
            training_eligible=(
                None if evidence is None else evidence.values.get("training_eligible")
            ),
        )
        recommendation_path = run_dir / "next_experiment.json"
        _write_json_atomic(recommendation_path, recommendation.to_dict())
        return StageOutput(
            artifacts={
                "promotion_decision": str(path),
                "next_experiment": str(recommendation_path),
            },
            values={"decision": decision.decision},
            cost_usd=0.0,
        )

    def _load_receipts(
        self,
        run_dir: Path,
        plan: RunPlan,
        *,
        require_complete: bool,
    ) -> tuple[StageReceipt, ...]:
        receipts: list[StageReceipt] = []
        for stage in CANONICAL_STAGE_ORDER:
            path = run_dir / "state" / f"{stage}.json"
            if not path.is_file() and not require_complete:
                break
            receipt = self._read_receipt(path, plan, stage)
            receipts.append(receipt)
        return tuple(receipts)


def require_nonfailed_manifest(
    manifest: RunManifest,
    run_dir: str | Path,
) -> None:
    """Make command entry points fail loudly when replay finds a failed Run."""

    if manifest.status != "failed":
        return
    failed_stage = str(manifest.metadata.get("failed_stage", "unknown"))
    failure_ref = manifest.artifacts.get("failure")
    failure_path = (
        Path(run_dir) / failure_ref.path
        if failure_ref is not None
        else Path(run_dir) / "failure.json"
    )
    raise RuntimeError(
        f"Run {manifest.run_id!r} is terminally failed at stage {failed_stage!r}; "
        f"inspect {failure_path}"
    )


def _write_model_atomic(path: Path, model: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(
        json.dumps(model.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _hash_stage_artifacts(run_dir: Path, output: StageOutput) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name, raw_path in sorted(output.artifacts.items()):
        path = Path(raw_path).resolve()
        try:
            path.relative_to(run_dir.resolve())
        except ValueError as exc:
            raise ValueError(
                f"stage artifact is outside run directory: {name}={path}"
            ) from exc
        if not path.is_file():
            raise ValueError(f"stage artifact is not a file: {name}={path}")
        hashes[name] = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def _write_json_atomic(path: Path, body: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(
        json.dumps(body, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _cost_certification_metadata(
    receipts: list[StageReceipt],
) -> dict[str, Any]:
    missing = [receipt.stage for receipt in receipts if receipt.output.cost_usd is None]
    return {
        "cost_certifying": not missing,
        "missing_cost_stages": missing,
    }


__all__ = [
    "ADAPTER_STAGE_ORDER",
    "CANONICAL_STAGE_ORDER",
    "RunEngine",
    "RunCoordinator",
    "RunExecution",
    "RunPlan",
    "RunResolution",
    "RunStage",
    "RunStageAdapter",
    "require_nonfailed_manifest",
    "StageOutput",
    "StageReceipt",
]
