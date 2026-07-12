"""API-first GSM8K hill-climb orchestration."""

from __future__ import annotations

import json
import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from post_train_engine.api_schemas import (
    Candidate,
    EvalExampleRecord,
    EvalResult,
    HillClimbConfig,
    JobRequest,
    JobResult,
    ProviderSpec,
    is_chat_completions_provider_type,
)
from post_train_engine.artifact_store import ArtifactStore
from post_train_engine.engine import (
    RunEngine,
    RunPlan,
    RunStage,
    StageOutput,
    require_nonfailed_manifest,
)
from post_train_engine.env import EnvResolver, load_env_file
from post_train_engine.evals.promotion import (
    EvalArtifact,
    EvalExampleResult,
    PromotionDecision,
    load_eval_artifact,
)
from post_train_engine.evals.contract import EvalContract
from post_train_engine.evidence_safety import (
    VerifierSeparation,
    certify_content_separation,
)
from post_train_engine.providers.base import RemoteProvider
from post_train_engine.providers.fake import FakeInferenceProvider, FakePromptAdapterProvider
from post_train_engine.providers.openai_compatible import OpenAICompatibleProvider
from post_train_engine.provider_operations import execute_provider_operation
from post_train_engine.traces import TraceRecord, build_rollout_group, stable_prompt_hash
from post_train_engine.training_views import (
    TrainingViewArtifact,
    build_training_view_artifact,
)
from post_train_engine.tasks.gsm8k import (
    GSM8KExample,
    embedded_gsm8k_examples,
    format_prompt,
    load_gsm8k,
    parse_model_answer,
    verify_answer,
)


@dataclass(frozen=True)
class ResolvedProvider:
    spec: ProviderSpec
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None

    def redacted_json(self) -> dict[str, Any]:
        body = self.spec.model_dump(mode="json")
        body["resolved"] = {
            "base_url": self.base_url,
            "api_key": "[REDACTED]" if self.api_key else None,
            "model": self.model,
        }
        return body


@dataclass(frozen=True)
class ResolvedHillClimbConfig:
    config: HillClimbConfig
    baseline: Candidate
    inference_provider: ResolvedProvider
    training_provider: ResolvedProvider
    env_redacted: dict[str, Any]

    def redacted_json(self) -> dict[str, Any]:
        body = self.config.model_dump(mode="json")
        body["baseline"]["model_id"] = self.baseline.model_id
        body["baseline"].pop("model_id_env", None)
        body["providers"] = {
            "inference": self.inference_provider.redacted_json(),
            "training": self.training_provider.redacted_json(),
        }
        return body


def load_hillclimb_config(
    path: str | Path,
    *,
    env_path: str | Path | None = ".env",
) -> ResolvedHillClimbConfig:
    """Load, validate, and secret-resolve an API-first hill-climb config."""

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("hillclimb config root must be a mapping")
    config = HillClimbConfig.model_validate(raw)
    env_names = _required_env_names(config)
    resolver = EnvResolver(load_env_file(env_path) if env_names else {})
    inference = _resolve_provider(config.providers.inference, resolver)
    training = _resolve_provider(config.providers.training, resolver)
    model_id = config.baseline.model_id
    if not model_id and config.baseline.model_id_env:
        model_id = resolver.require(config.baseline.model_id_env, secret=False)
    if not model_id:
        raise ValueError("baseline model_id could not be resolved")
    baseline = Candidate(
        candidate_id=config.baseline.candidate_id,
        model_id=model_id,
        system_prompt=config.baseline.system_prompt,
        prompt_prefix=config.baseline.prompt_prefix,
        prompt_suffix=config.baseline.prompt_suffix,
        adapter_kind="base",
        metadata={"source": "baseline_config"},
    )
    return ResolvedHillClimbConfig(
        config=config,
        baseline=baseline,
        inference_provider=inference,
        training_provider=training,
        env_redacted=resolver.redacted_provider_env(
            [config.providers.inference, config.providers.training],
        ),
    )


def _required_env_names(config: HillClimbConfig) -> set[str]:
    names: set[str] = set()
    if config.baseline.model_id_env:
        names.add(config.baseline.model_id_env)
    for spec in (config.providers.inference, config.providers.training):
        for name in (spec.base_url_env, spec.api_key_env, spec.model_env):
            if name:
                names.add(name)
    return names


def run_hillclimb(
    config_path: str | Path,
    *,
    env_path: str | Path | None = ".env",
) -> dict[str, Any]:
    resolved = load_hillclimb_config(config_path, env_path=env_path)
    plan, adapter = _compile_api_hillclimb(
        resolved,
        config_path=Path(config_path),
    )
    execution = RunEngine().execute(plan, adapter)
    require_nonfailed_manifest(execution.manifest, plan.output_dir)
    report = json.loads(
        (Path(plan.output_dir) / "final_report.json").read_text(encoding="utf-8")
    )
    if not isinstance(report, dict):
        raise ValueError("final report must be a JSON object")
    return report


def _compile_api_hillclimb(
    resolved: ResolvedHillClimbConfig,
    *,
    config_path: Path,
) -> tuple[RunPlan, _APIHillClimbAdapter]:
    cfg = resolved.config
    train_examples, eval_examples = _load_and_split_dataset(cfg)
    candidate = Candidate(
        candidate_id=f"{resolved.baseline.candidate_id}-{cfg.run.run_id}-candidate",
        model_id=resolved.baseline.model_id,
        parent_id=resolved.baseline.candidate_id,
        adapter_kind="planned",
        metadata={"source": "run_plan"},
    )
    run_dir = Path(cfg.run.output_dir)
    resumable = run_dir.is_dir() and (
        (run_dir / "manifest.json").is_file() or (run_dir / "state").is_dir()
    )
    store = (
        ArtifactStore(run_dir, resume=True)
        if resumable
        else ArtifactStore(run_dir, overwrite=cfg.run.overwrite)
    )
    promotion_gate = {
        "min_examples": cfg.promotion.min_eval_examples,
        "min_primary_delta": max(cfg.promotion.min_accuracy_delta, 1e-12),
        "min_primary_ci_low": cfg.promotion.min_paired_delta_ci_low,
        "max_mcnemar_p": cfg.promotion.max_mcnemar_p,
        "max_parse_regression": cfg.promotion.max_parse_regression,
        "max_easy_regression": cfg.promotion.max_easy_regression,
        "max_token_increase_ratio": cfg.promotion.max_token_increase_ratio,
    }
    plan = RunPlan(
        certification_mode=cfg.run.certification_mode,
        run_id=cfg.run.run_id,
        candidate_id=candidate.candidate_id,
        parent_candidate_id=resolved.baseline.candidate_id,
        task_name=cfg.dataset.name,
        model_id=candidate.model_id,
        output_dir=str(store.run_dir),
        source_root=str(Path(__file__).resolve().parents[2]),
        inputs=_api_input_identities(
            cfg=cfg,
            resolved=resolved,
            candidate=candidate,
            train_examples=train_examples,
        ),
        training_example_ids=tuple(example.id for example in train_examples),
        promotion_example_ids=tuple(example.id for example in eval_examples),
        evaluation_contract=EvalContract.from_components(
            suite_id=f"{cfg.dataset.name}-promotion",
            suite_version=(
                f"{cfg.dataset.dataset_name}:"
                f"{cfg.dataset.dataset_revision or cfg.dataset.source}:"
                f"seed={cfg.dataset.split_seed}"
            ),
            example_ids=tuple(example.id for example in eval_examples),
            example_content=tuple(
                {
                    "id": example.id,
                    "question": example.question,
                    "gold_answer": example.gold_answer,
                }
                for example in eval_examples
            ),
            prompt_contract={"prompt_style": cfg.dataset.prompt_style},
            verifier_contract={"task": cfg.dataset.name, "verifier": "exact-answer-v1"},
            generation_contract=cfg.eval.model_dump(mode="json"),
            primary_metric="accuracy",
        ),
        content_separation=certify_content_separation(
            training_texts=tuple(example.question for example in train_examples),
            protected_texts=tuple(example.question for example in eval_examples),
        ),
        verifier_separation=VerifierSeparation(
            verifier_kind="executable_ground_truth",
            training_verifier_id="gsm8k-exact-answer-v1",
            promotion_verifier_id="gsm8k-exact-answer-v1",
        ),
        promotion_gate=promotion_gate,
        campaign=cfg.run.campaign,
        metadata={
            "execution_mode": "api_hillclimb",
            "dataset": {
                "source": cfg.dataset.source,
                "split_seed": cfg.dataset.split_seed,
                "train_ids": [example.id for example in train_examples],
                "eval_ids": [example.id for example in eval_examples],
            },
            "providers": {
                "inference": _provider_report(resolved.inference_provider),
                "training": _provider_report(resolved.training_provider),
            },
        },
    )
    return plan, _APIHillClimbAdapter(
        resolved=resolved,
        config_path=config_path,
        store=store,
        train_examples=train_examples,
        eval_examples=eval_examples,
    )


class _APIHillClimbAdapter:
    """Provider adapter that leaves stage order and promotion to RunEngine."""

    def __init__(
        self,
        *,
        resolved: ResolvedHillClimbConfig,
        config_path: Path,
        store: ArtifactStore,
        train_examples: list[GSM8KExample],
        eval_examples: list[GSM8KExample],
    ) -> None:
        self.resolved = resolved
        self.config_path = config_path
        self.store = store
        self.train_examples = train_examples
        self.eval_examples = eval_examples

    def execute_stage(
        self,
        stage: RunStage,
        plan: RunPlan,
        prior: dict[str, StageOutput],
    ) -> StageOutput:
        if stage == "promote":
            raise ValueError("RunEngine owns the promote stage")
        handlers = {
            "prepare": self._prepare,
            "data": self._data,
            "evidence": self._evidence,
            "train": self._train,
            "select": self._select,
            "evaluate": self._evaluate,
            "finalize": self._finalize,
        }
        return handlers[stage](plan, prior)

    def _prepare(
        self,
        _plan: RunPlan,
        _prior: dict[str, StageOutput],
    ) -> StageOutput:
        self.store.copy_file(self.config_path, "config.raw.yaml")
        self.store.write_json("config.resolved.json", self.resolved.redacted_json())
        self.store.write_json("env.redacted.json", self.resolved.env_redacted)
        self.store.write_json("candidates/baseline.json", self.resolved.baseline.to_json())
        return _api_stage_output(
            self.store,
            artifacts={
                "config_raw": "config.raw.yaml",
                "config_resolved": "config.resolved.json",
                "env_redacted": "env.redacted.json",
                "baseline_candidate": "candidates/baseline.json",
            },
        )

    def _data(
        self,
        _plan: RunPlan,
        _prior: dict[str, StageOutput],
    ) -> StageOutput:
        _write_dataset_artifacts(
            self.store,
            self.resolved.config,
            self.train_examples,
            self.eval_examples,
        )
        return _api_stage_output(
            self.store,
            artifacts={
                "dataset_splits": "datasets/splits.json",
                "train_examples": "datasets/train.jsonl",
                "eval_examples": "datasets/eval.jsonl",
            },
        )

    def _evidence(
        self,
        plan: RunPlan,
        _prior: dict[str, StageOutput],
    ) -> StageOutput:
        cfg = self.resolved.config
        result = _run_provider_job(
            provider=_build_provider(self.resolved.inference_provider),
            store=self.store,
            request=_inference_request(
                run_id=plan.run_id,
                suffix="baseline-train-rollouts",
                job_type="rollout_generation",
                provider_id=self.resolved.inference_provider.spec.provider_id,
                candidate=self.resolved.baseline,
                examples=self.train_examples,
                prompt_style=cfg.dataset.prompt_style,
                samples_per_example=cfg.rollout.samples_per_example,
                split_role="train",
                generation=cfg.rollout.model_dump(mode="json"),
            ),
        )
        rows = _grade_generations(
            candidate=self.resolved.baseline,
            examples=self.train_examples,
            generation_result=result,
            prompt_style=cfg.dataset.prompt_style,
        )
        self.store.write_jsonl(
            "rollouts/baseline_train_rollouts.jsonl",
            [row.model_dump(mode="json") for row in rows],
        )
        evidence_artifacts, evidence_values = _write_api_training_evidence(
            store=self.store,
            cfg=cfg,
            baseline=self.resolved.baseline,
            provider=self.resolved.inference_provider,
            rows=rows,
        )
        return _api_stage_output(
            self.store,
            artifacts={
                "train_rollouts": "rollouts/baseline_train_rollouts.jsonl",
                **evidence_artifacts,
            },
            values=evidence_values,
            cost_missing_reason="provider did not report rollout cost",
        )

    def _train(
        self,
        plan: RunPlan,
        prior: dict[str, StageOutput],
    ) -> StageOutput:
        if not bool(prior["evidence"].values["training_eligible"]):
            candidate = Candidate(
                candidate_id=plan.candidate_id,
                model_id=self.resolved.baseline.model_id,
                parent_id=self.resolved.baseline.candidate_id,
                system_prompt=self.resolved.baseline.system_prompt,
                prompt_prefix=self.resolved.baseline.prompt_prefix,
                prompt_suffix=self.resolved.baseline.prompt_suffix,
                adapter_kind="no_training_outcome",
                metadata={
                    "reason": prior["evidence"].values["non_training_reason"],
                },
            )
            self.store.write_json("candidates/candidate.json", candidate.to_json())
            return _api_stage_output(
                self.store,
                artifacts={"candidate": "candidates/candidate.json"},
                values={"training_outcome": "skipped_no_learnable_evidence"},
            )
        view = TrainingViewArtifact.model_validate(
            _read_json(prior["evidence"].artifacts["prompt_adaptation_view"])
        )
        if "prompt_adaptation" not in view.method_compatibility:
            raise ValueError("TrainingView is not compatible with prompt_adaptation")
        training_rows = _read_jsonl(
            prior["evidence"].artifacts["prompt_adaptation_rows"]
        )
        result = _run_provider_job(
            provider=_build_provider(self.resolved.training_provider),
            store=self.store,
            request=JobRequest(
                job_id=f"{plan.run_id}:candidate-adaptation",
                job_type="candidate_adaptation",
                provider_id=self.resolved.training_provider.spec.provider_id,
                payload={
                    "candidate_id": plan.candidate_id,
                    "baseline_candidate": self.resolved.baseline.to_json(),
                    "training_view": view.to_json(),
                    "training_rows": training_rows,
                    "method": "prompt_adaptation",
                },
            ),
        )
        payload = result.payload.get("candidate")
        if not isinstance(payload, dict):
            raise ValueError("adaptation provider result missing candidate object")
        candidate = Candidate.model_validate(payload)
        _validate_adapted_candidate(candidate, self.resolved.baseline)
        if candidate.candidate_id != plan.candidate_id:
            raise ValueError("adaptation provider changed the RunPlan candidate_id")
        self.store.write_json("candidates/candidate.json", candidate.to_json())
        return _api_stage_output(
            self.store,
            artifacts={"candidate": "candidates/candidate.json"},
            cost_missing_reason="provider did not report adaptation cost",
        )

    def _select(
        self,
        plan: RunPlan,
        _prior: dict[str, StageOutput],
    ) -> StageOutput:
        self.store.write_json(
            "checkpoint_selection.json",
            {
                "candidate_id": plan.candidate_id,
                "rule": "single_provider_candidate",
                "selected": True,
            },
        )
        return _api_stage_output(
            self.store,
            artifacts={"checkpoint_selection": "checkpoint_selection.json"},
        )

    def _evaluate(
        self,
        plan: RunPlan,
        prior: dict[str, StageOutput],
    ) -> StageOutput:
        cfg = self.resolved.config
        candidate = Candidate.model_validate(
            _read_json(prior["train"].artifacts["candidate"])
        )
        provider = _build_provider(self.resolved.inference_provider)
        baseline = _evaluate_candidate(
            cfg=cfg,
            store=self.store,
            provider=provider,
            provider_id=self.resolved.inference_provider.spec.provider_id,
            candidate=self.resolved.baseline,
            examples=self.eval_examples,
            name="baseline",
        )
        candidate_result = _evaluate_candidate(
            cfg=cfg,
            store=self.store,
            provider=provider,
            provider_id=self.resolved.inference_provider.spec.provider_id,
            candidate=candidate,
            examples=self.eval_examples,
            name="candidate",
        )
        self.store.write_json(
            "evals/baseline.json",
            _api_promotion_artifact(
                baseline,
                evaluation_contract_hash=plan.evaluation_contract.contract_hash,
            ).to_dict(),
        )
        self.store.write_json(
            "evals/candidate.json",
            _api_promotion_artifact(
                candidate_result,
                evaluation_contract_hash=plan.evaluation_contract.contract_hash,
            ).to_dict(),
        )
        return _api_stage_output(
            self.store,
            artifacts={
                "baseline_eval": "evals/baseline.json",
                "candidate_eval": "evals/candidate.json",
            },
            cost_missing_reason="provider did not report evaluation cost",
        )

    def _finalize(
        self,
        _plan: RunPlan,
        prior: dict[str, StageOutput],
    ) -> StageOutput:
        candidate = Candidate.model_validate(
            _read_json(prior["train"].artifacts["candidate"])
        )
        baseline_eval = load_eval_artifact(prior["evaluate"].artifacts["baseline_eval"])
        candidate_eval = load_eval_artifact(prior["evaluate"].artifacts["candidate_eval"])
        decision = _read_json(prior["promote"].artifacts["promotion_decision"])
        next_experiment = _read_json(
            prior["promote"].artifacts["next_experiment"]
        )
        report = _final_report(
            cfg=self.resolved.config,
            resolved=self.resolved,
            baseline=self.resolved.baseline,
            candidate=candidate,
            train_examples=self.train_examples,
            eval_examples=self.eval_examples,
            baseline_eval=baseline_eval,
            candidate_eval=candidate_eval,
            decision=decision,
            next_experiment=next_experiment,
        )
        self.store.write_json("final_report.json", report)
        self.store.write_text("final_report.md", _markdown_report(report))
        return _api_stage_output(
            self.store,
            artifacts={
                "final_report_json": "final_report.json",
                "final_report_md": "final_report.md",
                "provider_requests": "provider_requests.jsonl",
                "provider_responses": "provider_responses.jsonl",
                "provider_operations": "provider_operations.jsonl",
            },
            values={"decision": decision["decision"]},
        )


def _write_api_training_evidence(
    *,
    store: ArtifactStore,
    cfg: HillClimbConfig,
    baseline: Candidate,
    provider: ResolvedProvider,
    rows: list[EvalExampleRecord],
) -> tuple[dict[str, str], dict[str, Any]]:
    grouped: dict[str, list[EvalExampleRecord]] = {}
    for row in rows:
        grouped.setdefault(row.example_id, []).append(row)

    traces: list[TraceRecord] = []
    trace_by_key: dict[tuple[str, int], TraceRecord] = {}
    for example_id, group_rows in sorted(grouped.items()):
        group_id = f"{cfg.run.run_id}:train:{example_id}"
        for row in sorted(group_rows, key=lambda item: item.sample_index):
            trace = TraceRecord(
                trace_id=f"{group_id}:{row.sample_index}",
                run_id=cfg.run.run_id,
                task_id=cfg.dataset.name,
                example_id=example_id,
                split_role="train",
                prompt_hash=stable_prompt_hash(row.prompt),
                source_checkpoint=baseline.candidate_id,
                policy_version=baseline.candidate_id,
                policy_step=0,
                policy_step_evidence="static",
                rollout_group_id=group_id,
                generation_backend=provider.spec.type,
                sampling_config=cfg.rollout.model_dump(mode="json"),
                verifier_id="gsm8k_numeric_v1",
                prompt=row.prompt,
                completion=row.completion,
                parsed_answer=row.parsed_answer,
                parser_status={"parse_ok": row.parse_ok},
                verifier_result={"correct": row.correct},
                reward_components={"correct": float(row.correct)},
                token_counts={"completion": float(row.completion_tokens)},
                privileged_visibility="none",
            )
            traces.append(trace)
            trace_by_key[(example_id, row.sample_index)] = trace
    groups = [
        build_rollout_group(
            group_id=f"{cfg.run.run_id}:train:{example_id}",
            traces=(
                trace_by_key[(example_id, row.sample_index)]
                for row in sorted(group_rows, key=lambda item: item.sample_index)
            ),
            rewards=(
                float(row.correct)
                for row in sorted(group_rows, key=lambda item: item.sample_index)
            ),
        )
        for example_id, group_rows in sorted(grouped.items())
    ]
    store.write_jsonl(
        "traces/baseline_train.jsonl",
        [trace.model_dump(mode="json") for trace in traces],
    )
    store.write_jsonl(
        "rollout_groups/baseline_train.jsonl",
        [group.model_dump(mode="json") for group in groups],
    )

    frontier_ids = {
        example_id
        for example_id, group_rows in grouped.items()
        if 0.0
        < sum(float(row.correct) for row in group_rows) / len(group_rows)
        < 1.0
    }
    common_artifacts = {
        "traces": "traces/baseline_train.jsonl",
        "rollout_groups": "rollout_groups/baseline_train.jsonl",
    }
    if not frontier_ids:
        store.write_json(
            "non_training_outcome.json",
            {
                "outcome": "no_learnable_evidence",
                "selection_policy": "parent_success_rate_frontier",
                "measured_example_count": len(grouped),
            },
        )
        return (
            {**common_artifacts, "non_training_outcome": "non_training_outcome.json"},
            {
                "training_eligible": False,
                "non_training_reason": "no parent-policy success rate was strictly between zero and one",
            },
        )

    selected_rows = []
    for row in rows:
        if row.example_id not in frontier_ids:
            continue
        trace = trace_by_key[(row.example_id, row.sample_index)]
        selected_rows.append(
            {
                **row.model_dump(mode="json"),
                "source_trace_ids": [trace.trace_id],
                "source_split_roles": [trace.split_role],
            }
        )
    data_relative = "training_views/prompt_adaptation_rows.jsonl"
    view_relative = "training_views/prompt_adaptation.json"
    store.write_jsonl(data_relative, selected_rows)
    view = build_training_view_artifact(
        view_id=f"{cfg.run.run_id}:prompt_adaptation",
        run_id=cfg.run.run_id,
        task_id=cfg.dataset.name,
        view_type="sft",
        method_compatibility=("prompt_adaptation",),
        data_path=store.run_dir / data_relative,
        artifact_root=store.run_dir,
        data_kind="prompt_adaptation_rows",
        rows=selected_rows,
        privileged_visibility="gold_answer",
        metadata={
            "selection_policy": "parent_success_rate_frontier",
            "selection_evidence": "measured_parent_success_rate",
            "selected_example_ids": sorted(frontier_ids),
            "parent_policy_version": baseline.candidate_id,
        },
    )
    store.write_json(view_relative, view.to_json())
    return (
        {
            **common_artifacts,
            "prompt_adaptation_rows": data_relative,
            "prompt_adaptation_view": view_relative,
        },
        {"training_eligible": True},
    )


def _api_stage_output(
    store: ArtifactStore,
    *,
    artifacts: Mapping[str, str],
    values: Mapping[str, Any] | None = None,
    cost_missing_reason: str | None = None,
) -> StageOutput:
    return StageOutput(
        artifacts={
            name: str(store.run_dir / relative)
            for name, relative in artifacts.items()
        },
        values=dict(values or {}),
        cost_usd=0.0 if cost_missing_reason is None else None,
        cost_missing_reason=cost_missing_reason,
    )


def _read_json(path: str | Path) -> dict[str, Any]:
    body = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(body, dict):
        raise ValueError(f"expected JSON object: {path}")
    return body


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"expected JSONL objects: {path}")
        rows.append(row)
    return rows
def _api_promotion_artifact(
    result: EvalResult,
    *,
    evaluation_contract_hash: str,
) -> EvalArtifact:
    return EvalArtifact(
        artifact_id=result.candidate_id,
        primary_metric="accuracy",
        evaluation_contract_hash=evaluation_contract_hash,
        examples=tuple(
            EvalExampleResult(
                example_id=row.example_id,
                correct=row.correct,
                parse_ok=row.parse_ok,
                tokens=row.completion_tokens,
                bucket="easy_stable",
            )
            for row in result.examples
        ),
        metrics=dict(result.metrics),
        slices={"easy_stable": {"accuracy": result.metrics["accuracy"]}},
    )


def _resolve_provider(spec: ProviderSpec, resolver: EnvResolver) -> ResolvedProvider:
    if spec.type in {"fake", "fake_prompt_adapter"}:
        return ResolvedProvider(spec=spec, model=spec.model)
    base_url = spec.base_url or resolver.require(str(spec.base_url_env), secret=False)
    api_key = resolver.require(str(spec.api_key_env), secret=True)
    model = spec.model or resolver.require(str(spec.model_env), secret=False)
    return ResolvedProvider(spec=spec, base_url=base_url, api_key=api_key, model=model)


def _build_provider(provider: ResolvedProvider) -> RemoteProvider:
    spec = provider.spec
    if spec.type == "fake":
        return FakeInferenceProvider(provider_id=spec.provider_id)
    if spec.type == "fake_prompt_adapter":
        return FakePromptAdapterProvider(provider_id=spec.provider_id)
    if is_chat_completions_provider_type(spec.type):
        if provider.base_url is None or provider.api_key is None or provider.model is None:
            raise ValueError(f"provider {spec.provider_id} is not fully resolved")
        return OpenAICompatibleProvider(
            provider_id=spec.provider_id,
            provider_type=spec.type,
            base_url=provider.base_url,
            api_key=provider.api_key,
            model=provider.model,
            timeout_seconds=spec.timeout_seconds,
            max_tokens_field=spec.max_tokens_field,
        )
    raise ValueError(f"unsupported provider type: {spec.type}")


def _run_provider_job(
    *,
    provider: RemoteProvider,
    store: ArtifactStore,
    request: JobRequest,
    timeout_seconds: float = 300.0,
    poll_interval_seconds: float = 0.05,
) -> JobResult:
    return execute_provider_operation(
        provider=provider,
        store=store,
        request=request,
        timeout_seconds=timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
    )


def _validate_adapted_candidate(candidate: Candidate, baseline: Candidate) -> None:
    if candidate.parent_id != baseline.candidate_id:
        raise ValueError(
            "adapted candidate parent_id must match baseline candidate_id "
            f"{baseline.candidate_id!r}",
        )
    if candidate.candidate_id == baseline.candidate_id:
        raise ValueError("adapted candidate_id must differ from baseline candidate_id")


def _load_and_split_dataset(
    cfg: HillClimbConfig,
) -> tuple[list[GSM8KExample], list[GSM8KExample]]:
    if cfg.dataset.source == "embedded_gsm8k_tiny":
        examples = embedded_gsm8k_examples()
        dataset_revision = "embedded-gsm8k-tiny-v1"
    elif cfg.dataset.source == "huggingface_gsm8k":
        examples = load_gsm8k("train", cfg.dataset.dataset_name)
        dataset_revision = cfg.dataset.dataset_revision or str(
            examples[0].metadata.get("dataset_revision", "unknown")
            if examples
            else "unknown"
        )
        examples = [
            GSM8KExample(
                id=example.id,
                split=example.split,
                question=example.question,
                gold_solution=example.gold_solution,
                gold_answer=example.gold_answer,
                source=example.source,
                metadata={**dict(example.metadata), "dataset_revision": dataset_revision},
            )
            for example in examples
        ]
    else:
        raise ValueError(f"unsupported dataset source: {cfg.dataset.source}")

    required = cfg.dataset.train_size + cfg.dataset.eval_size
    if required > len(examples):
        raise ValueError(f"requested {required} examples but dataset has {len(examples)}")
    shuffled = list(examples)
    random.Random(cfg.dataset.split_seed).shuffle(shuffled)
    train = shuffled[: cfg.dataset.train_size]
    eval_rows = shuffled[cfg.dataset.train_size : required]
    return train, eval_rows


def _write_dataset_artifacts(
    store: ArtifactStore,
    cfg: HillClimbConfig,
    train_examples: list[GSM8KExample],
    eval_examples: list[GSM8KExample],
) -> None:
    store.write_json(
        "datasets/splits.json",
        {
            "dataset": cfg.dataset.name,
            "source": cfg.dataset.source,
            "split_seed": cfg.dataset.split_seed,
            "prompt_style": cfg.dataset.prompt_style,
            "splits": {
                "train": [example.id for example in train_examples],
                "eval": [example.id for example in eval_examples],
            },
        },
    )
    store.write_jsonl(
        "datasets/train.jsonl",
        [_example_json(example, cfg.dataset.prompt_style, "train") for example in train_examples],
    )
    store.write_jsonl(
        "datasets/eval.jsonl",
        [_example_json(example, cfg.dataset.prompt_style, "eval") for example in eval_examples],
    )


def _inference_request(
    *,
    run_id: str,
    suffix: str,
    job_type: str,
    provider_id: str,
    candidate: Candidate,
    examples: list[GSM8KExample],
    prompt_style: str,
    samples_per_example: int,
    split_role: str,
    generation: dict[str, Any],
) -> JobRequest:
    prompts = []
    for example in examples:
        prompt = format_prompt(example.question, prompt_style)  # type: ignore[arg-type]
        for sample_index in range(samples_per_example):
            prompts.append(
                {
                    "example_id": example.id,
                    "prompt": prompt,
                    "sample_index": sample_index,
                    "split_role": split_role,
                }
            )
    return JobRequest(
        job_id=f"{run_id}:{suffix}",
        job_type=job_type,  # type: ignore[arg-type]
        provider_id=provider_id,
        payload={
            "candidate": candidate.to_json(),
            "prompts": prompts,
            "generation": generation,
        },
    )


def _evaluate_candidate(
    *,
    cfg: HillClimbConfig,
    store: ArtifactStore,
    provider: RemoteProvider,
    provider_id: str,
    candidate: Candidate,
    examples: list[GSM8KExample],
    name: str,
) -> EvalResult:
    result = _run_provider_job(
        provider=provider,
        store=store,
        request=_inference_request(
            run_id=cfg.run.run_id,
            suffix=f"{name}-eval",
            job_type="evaluation",
            provider_id=provider_id,
            candidate=candidate,
            examples=examples,
            prompt_style=cfg.dataset.prompt_style,
            samples_per_example=cfg.eval.samples_per_example,
            split_role="eval",
            generation=cfg.eval.model_dump(mode="json"),
        ),
    )
    rows = _grade_generations(
        candidate=candidate,
        examples=examples,
        generation_result=result,
        prompt_style=cfg.dataset.prompt_style,
    )
    eval_result = _eval_result(
        candidate.candidate_id,
        rows,
        expected_example_ids={example.id for example in examples},
    )
    store.write_json(f"evals/{name}.json", eval_result.to_json())
    return eval_result


def _grade_generations(
    *,
    candidate: Candidate,
    examples: list[GSM8KExample],
    generation_result: JobResult,
    prompt_style: str,
) -> list[EvalExampleRecord]:
    examples_by_id = {example.id: example for example in examples}
    rows: list[EvalExampleRecord] = []
    generations = generation_result.payload.get("generations")
    if not isinstance(generations, list):
        raise ValueError("provider result generations must be a list")
    for index, generation in enumerate(generations):
        if not isinstance(generation, dict):
            raise ValueError(f"provider generation row {index} must be an object")
        example_id = generation.get("example_id")
        if not isinstance(example_id, str) or not example_id:
            raise ValueError(f"provider generation row {index} missing example_id")
        if example_id not in examples_by_id:
            raise ValueError(f"provider returned unknown example_id: {example_id}")
        example = examples_by_id[example_id]
        completion_value = generation.get("completion")
        if not isinstance(completion_value, str):
            raise ValueError(f"provider generation row {index} missing completion string")
        completion = completion_value
        sample_index = _nonnegative_int(
            generation.get("sample_index", 0),
            field=f"provider generation row {index} sample_index",
        )
        completion_tokens = _nonnegative_int(
            generation.get("completion_tokens", len(completion.split())),
            field=f"provider generation row {index} completion_tokens",
        )
        parsed = parse_model_answer(completion, mode="strict")
        verification = (
            verify_answer(parsed.answer, example.gold_answer)
            if parsed.parse_ok and parsed.answer is not None
            else None
        )
        rows.append(
            EvalExampleRecord(
                example_id=example.id,
                prompt=format_prompt(example.question, prompt_style),  # type: ignore[arg-type]
                completion=completion,
                parsed_answer=parsed.answer,
                gold_answer=example.gold_answer,
                correct=bool(verification and verification.correct),
                parse_ok=parsed.parse_ok,
                completion_tokens=completion_tokens,
                finish_reason=(
                    None
                    if generation.get("finish_reason") is None
                    else str(generation.get("finish_reason"))
                ),
                provider_job_id=(
                    None
                    if generation.get("provider_job_id") is None
                    else str(generation.get("provider_job_id"))
                ),
                sample_index=sample_index,
            )
        )
    if not rows:
        raise ValueError(f"provider returned no generations for candidate {candidate.candidate_id}")
    return rows


def _nonnegative_int(value: Any, *, field: str) -> int:
    if type(value) is bool or not isinstance(value, int | float):
        raise ValueError(f"{field} must be a non-negative integer")
    converted = int(value)
    if converted != value or converted < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return converted


def _eval_result(
    candidate_id: str,
    rows: list[EvalExampleRecord],
    *,
    expected_example_ids: set[str],
) -> EvalResult:
    first_samples = [row for row in rows if row.sample_index == 0]
    if not first_samples:
        raise ValueError("eval requires sample_index=0 rows")
    seen_ids = [row.example_id for row in first_samples]
    duplicate_ids = sorted({example_id for example_id in seen_ids if seen_ids.count(example_id) > 1})
    if duplicate_ids:
        raise ValueError(f"eval has duplicate sample_index=0 rows: {duplicate_ids}")
    missing = sorted(expected_example_ids.difference(seen_ids))
    unexpected = sorted(set(seen_ids).difference(expected_example_ids))
    if missing or unexpected:
        raise ValueError(
            "eval first-sample coverage mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    accuracy = sum(row.correct for row in first_samples) / len(first_samples)
    parse_rate = sum(row.parse_ok for row in first_samples) / len(first_samples)
    mean_tokens = sum(float(row.completion_tokens) for row in first_samples) / len(first_samples)
    return EvalResult(
        candidate_id=candidate_id,
        metrics={
            "accuracy": accuracy,
            "parse_success_rate": parse_rate,
            "mean_tokens": mean_tokens,
        },
        examples=tuple(first_samples),
    )


def _final_report(
    *,
    cfg: HillClimbConfig,
    resolved: ResolvedHillClimbConfig,
    baseline: Candidate,
    candidate: Candidate,
    train_examples: list[GSM8KExample],
    eval_examples: list[GSM8KExample],
    baseline_eval: EvalResult | EvalArtifact,
    candidate_eval: EvalResult | EvalArtifact,
    decision: PromotionDecision | Mapping[str, Any],
    next_experiment: Mapping[str, Any],
) -> dict[str, Any]:
    decision_body = (
        decision.to_dict()
        if isinstance(decision, PromotionDecision)
        else dict(decision)
    )
    return {
        "run_id": cfg.run.run_id,
        "status": decision_body["decision"],
        "providers": {
            "inference": _provider_report(resolved.inference_provider),
            "training": _provider_report(resolved.training_provider),
        },
        "baseline": baseline.to_json(),
        "candidate": candidate.to_json(),
        "data": {
            "dataset": cfg.dataset.name,
            "source": cfg.dataset.source,
            "train_examples": len(train_examples),
            "eval_examples": len(eval_examples),
            "split_seed": cfg.dataset.split_seed,
            "prompt_style": cfg.dataset.prompt_style,
            "train_ids": [example.id for example in train_examples],
            "eval_ids": [example.id for example in eval_examples],
        },
        "prompts": {
            "raw_prompt_artifacts": [
                "datasets/train.jsonl",
                "datasets/eval.jsonl",
                "provider_requests.jsonl",
            ],
            "sample_eval_prompt": (
                format_prompt(eval_examples[0].question, cfg.dataset.prompt_style)  # type: ignore[arg-type]
                if eval_examples
                else None
            ),
        },
        "raw_outputs": {
            "train_rollouts": "rollouts/baseline_train_rollouts.jsonl",
            "baseline_eval": "evals/baseline.json",
            "candidate_eval": "evals/candidate.json",
            "provider_responses": "provider_responses.jsonl",
        },
        "metrics": {
            "baseline": dict(baseline_eval.metrics),
            "candidate": dict(candidate_eval.metrics),
            "delta": decision_body["primary_delta"],
            "paired_ci95": list(decision_body["primary_ci95"]),
            "mcnemar_p": decision_body["mcnemar_p"],
        },
        "promotion": decision_body,
        "cost": {"available": False, "reason": "provider cost not reported"},
        "failures": list(decision_body["rejection_reasons"]),
        "next_experiment": dict(next_experiment),
    }


def _provider_report(provider: ResolvedProvider) -> dict[str, Any]:
    return {
        "provider_id": provider.spec.provider_id,
        "type": provider.spec.type,
        "base_url": provider.base_url,
        "model": provider.model,
    }


def _api_input_identities(
    *,
    cfg: HillClimbConfig,
    resolved: ResolvedHillClimbConfig,
    candidate: Candidate,
    train_examples: list[GSM8KExample],
) -> dict[str, Any]:
    dataset_revision = (
        str(train_examples[0].metadata.get("dataset_revision", "unknown"))
        if train_examples
        else "unknown"
    )
    dataset_exact = dataset_revision != "unknown"
    provider_types = {
        resolved.inference_provider.spec.type,
        resolved.training_provider.spec.type,
    }
    model_exact = provider_types <= {"fake", "fake_prompt_adapter"}
    return {
        "model": {
            "kind": "model",
            "requested_id": candidate.model_id,
            "resolved_id": candidate.model_id,
            "resolved_revision": candidate.model_id if model_exact else None,
            "resolution_state": "exact" if model_exact else "provider_managed",
            "non_certifying_reason": (
                None
                if model_exact
                else "remote provider did not expose an immutable weight revision"
            ),
        },
        "dataset": {
            "kind": "dataset",
            "requested_id": cfg.dataset.dataset_name,
            "requested_revision": cfg.dataset.dataset_revision,
            "resolved_id": cfg.dataset.dataset_name,
            "resolved_revision": dataset_revision if dataset_exact else None,
            "resolution_state": "exact" if dataset_exact else "provider_managed",
            "non_certifying_reason": (
                None
                if dataset_exact
                else "dataset loader did not expose an immutable revision"
            ),
        },
    }


def _markdown_report(report: dict[str, Any]) -> str:
    failures = report["failures"] or ["none"]
    failure_lines = "\n".join(f"- {failure}" for failure in failures)
    return "\n".join(
        [
            f"# Hillclimb Run {report['run_id']}",
            "",
            f"- Status: {report['status']}",
            f"- Baseline: {report['baseline']['candidate_id']}",
            f"- Candidate: {report['candidate']['candidate_id']}",
            f"- Inference provider: {report['providers']['inference']['type']}",
            f"- Training provider: {report['providers']['training']['type']}",
            f"- Dataset: {report['data']['dataset']} ({report['data']['source']})",
            f"- Accuracy delta: {report['metrics']['delta']}",
            f"- CI95: {report['metrics']['paired_ci95']}",
            f"- Promotion: {report['promotion']['decision']}",
            "",
            "## Failures",
            "",
            failure_lines,
            "",
            "## Next Experiment",
            "",
            f"- {report['next_experiment']['category']}: {report['next_experiment']['rationale']}",
            "",
        ]
    )


def _example_json(example: GSM8KExample, prompt_style: str, split_role: str) -> dict[str, Any]:
    return {
        "example_id": example.id,
        "source": example.source,
        "split_role": split_role,
        "question": example.question,
        "gold_solution": example.gold_solution,
        "gold_answer": example.gold_answer,
        "prompt": format_prompt(example.question, prompt_style),  # type: ignore[arg-type]
        "metadata": dict(example.metadata),
    }


__all__ = [
    "ResolvedHillClimbConfig",
    "load_hillclimb_config",
    "run_hillclimb",
]
