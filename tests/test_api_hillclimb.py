from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from post_train_engine.cli.main import main
from post_train_engine.engine import CANONICAL_STAGE_ORDER


def test_hillclimb_dryrun_writes_full_api_first_run_bundle(tmp_path: Path) -> None:
    config = _write_dryrun_config(tmp_path)
    run_dir = tmp_path / "runs" / "gsm8k-dryrun"

    main(["hillclimb", "--config", str(config)])

    expected_files = [
        "config.raw.yaml",
        "config.resolved.json",
        "env.redacted.json",
        "provider_requests.jsonl",
        "provider_responses.jsonl",
        "candidates/baseline.json",
        "candidates/candidate.json",
        "datasets/splits.json",
        "datasets/train.jsonl",
        "datasets/eval.jsonl",
        "rollouts/baseline_train_rollouts.jsonl",
        "evals/baseline.json",
        "evals/candidate.json",
        "promotion_decision.json",
        "final_report.json",
        "final_report.md",
        "manifest.json",
    ]
    for relative in expected_files:
        assert (run_dir / relative).is_file(), relative
    assert (run_dir / "artifact_status.json").is_file()
    assert not (run_dir / "promotion_decisions.jsonl").exists()
    assert not (run_dir / "logs" / "events.jsonl").exists()

    report = json.loads((run_dir / "final_report.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert report["run_id"] == "gsm8k-dryrun"
    assert manifest["schema_version"] == "post_train_run_v1"
    assert manifest["status"] == "rejected"
    assert manifest["metadata"]["certification_mode"] == "non_certifying_smoke"
    assert manifest["candidate_id"] == report["candidate"]["candidate_id"]
    assert manifest["parent_candidate_id"] == "baseline"
    assert manifest["model_id"] == "fake-gsm8k"
    assert manifest["metadata"]["stage_order"] == list(CANONICAL_STAGE_ORDER)
    assert {
        name
        for name in manifest["artifacts"]
        if name.startswith("stage_receipt_")
    } == {f"stage_receipt_{stage}" for stage in CANONICAL_STAGE_ORDER}
    assert manifest["source"]["state"] in {
        "clean",
        "tracked_dirty",
        "untracked_dirty",
        "mixed",
    }
    assert set(manifest["artifacts"]) >= {
        "config_raw",
        "config_resolved",
        "env_redacted",
        "provider_requests",
        "provider_responses",
        "provider_operations",
        "baseline_candidate",
        "candidate",
        "dataset_splits",
        "train_examples",
        "eval_examples",
        "train_rollouts",
        "traces",
        "rollout_groups",
        "prompt_adaptation_view",
        "baseline_eval",
        "candidate_eval",
        "promotion_decision",
        "final_report_json",
        "final_report_md",
    }
    assert all(
        ref["sha256"].startswith("sha256:")
        for ref in manifest["artifacts"].values()
    )
    assert all(
        not Path(ref["path"]).is_absolute()
        for ref in manifest["artifacts"].values()
    )
    assert report["providers"]["inference"]["type"] == "fake"
    assert report["providers"]["training"]["type"] == "fake_prompt_adapter"
    assert report["baseline"]["candidate_id"] == "baseline"
    assert report["candidate"]["parent_id"] == "baseline"
    assert report["data"]["train_examples"] == 3
    assert report["data"]["eval_examples"] == 2
    assert report["promotion"]["decision"] == "reject"
    assert "non_certifying_smoke" in report["promotion"]["rejection_reasons"]
    assert report["promotion"]["primary_metric"] == "accuracy"
    assert report["promotion"]["stats"]["n"] == 2
    assert report["next_experiment"]["category"]

    requests = _read_jsonl(run_dir / "provider_requests.jsonl")
    responses = _read_jsonl(run_dir / "provider_responses.jsonl")
    assert {row["job_type"] for row in requests} == {
        "rollout_generation",
        "candidate_adaptation",
        "evaluation",
    }
    assert len(responses) == len(requests)
    assert all(row["status"]["state"] == "succeeded" for row in responses)
    assert any("Problem:" in json.dumps(row) for row in requests)
    assert any("<answer>" in json.dumps(row) for row in responses)
    adaptation_request = next(
        row for row in requests if row["job_type"] == "candidate_adaptation"
    )
    assert "train_rollouts" not in adaptation_request["payload"]
    assert adaptation_request["payload"]["training_view"]["method_compatibility"] == [
        "prompt_adaptation"
    ]
    view = json.loads(
        (run_dir / "training_views" / "prompt_adaptation.json").read_text(
            encoding="utf-8"
        )
    )
    assert view["source_trace_ids"]
    assert view["source_split_roles"] == ["train"]
    assert view["metadata"]["selection_evidence"] == "measured_parent_success_rate"
    serialized_responses = json.dumps(responses)
    assert '"completion_tokens": "[REDACTED]"' not in serialized_responses
    assert '"max_output_tokens": "[REDACTED]"' not in json.dumps(requests)

    main(["hillclimb", "--config", str(config)])
    rerun_requests = _read_jsonl(run_dir / "provider_requests.jsonl")
    assert len(rerun_requests) == len(requests)

    main(["artifacts", "validate", "--run", str(run_dir)])
    status = json.loads((run_dir / "artifact_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "ok"
    assert status["run_id"] == "gsm8k-dryrun"


def test_hillclimb_rerun_preserves_finalized_run_even_when_legacy_overwrite_is_true(
    tmp_path: Path,
) -> None:
    config = _write_dryrun_config(tmp_path)
    run_dir = tmp_path / "runs" / "gsm8k-dryrun"

    main(["hillclimb", "--config", str(config)])
    manifest_before = (run_dir / "manifest.json").read_bytes()
    stale_file = run_dir / "evals" / "stale.json"
    stale_file.write_text('{"from":"previous-run"}', encoding="utf-8")

    main(["hillclimb", "--config", str(config)])

    assert stale_file.exists()
    assert (run_dir / "manifest.json").read_bytes() == manifest_before
    status = json.loads((run_dir / "artifact_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "ok"


def test_api_hillclimb_resume_does_not_duplicate_provider_side_effects(
    tmp_path: Path,
) -> None:
    config = _write_dryrun_config(tmp_path)
    run_dir = tmp_path / "runs" / "gsm8k-dryrun"
    main(["hillclimb", "--config", str(config)])
    request_count = len(_read_jsonl(run_dir / "provider_requests.jsonl"))
    response_count = len(_read_jsonl(run_dir / "provider_responses.jsonl"))

    (run_dir / "manifest.json").unlink()
    resume_from = CANONICAL_STAGE_ORDER.index("evidence")
    for stage in CANONICAL_STAGE_ORDER[resume_from:]:
        (run_dir / "state" / f"{stage}.json").unlink()

    main(["hillclimb", "--config", str(config)])

    assert len(_read_jsonl(run_dir / "provider_requests.jsonl")) == request_count
    assert len(_read_jsonl(run_dir / "provider_responses.jsonl")) == response_count


def test_api_hillclimb_emits_non_training_outcome_without_measured_frontier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from post_train_engine.providers import fake as fake_provider

    monkeypatch.setattr(fake_provider, "_fake_correctness", lambda **_kwargs: True)
    config = _write_dryrun_config(tmp_path)
    run_dir = tmp_path / "runs" / "gsm8k-dryrun"

    main(["hillclimb", "--config", str(config)])

    requests = _read_jsonl(run_dir / "provider_requests.jsonl")
    assert "candidate_adaptation" not in {row["job_type"] for row in requests}
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "rejected"
    assert "non_training_outcome" in manifest["artifacts"]
    candidate = json.loads(
        (run_dir / "candidates" / "candidate.json").read_text(encoding="utf-8")
    )
    assert candidate["adapter_kind"] == "no_training_outcome"


def test_artifact_store_overwrite_refuses_unmarked_existing_directory(tmp_path: Path) -> None:
    from post_train_engine.artifact_store import ArtifactStore

    existing = tmp_path / "src"
    existing.mkdir()
    (existing / "important.py").write_text("print('keep me')\n", encoding="utf-8")

    with pytest.raises(ValueError, match="refusing to overwrite unmarked run directory"):
        ArtifactStore(existing, overwrite=True)

    assert (existing / "important.py").is_file()


def test_artifact_store_overwrite_refuses_unmarked_directory_under_runs(
    tmp_path: Path,
) -> None:
    from post_train_engine.artifact_store import ArtifactStore

    existing = tmp_path / "runs" / "unowned"
    existing.mkdir(parents=True)
    (existing / "important.txt").write_text("keep me", encoding="utf-8")

    with pytest.raises(ValueError, match="refusing to overwrite unmarked run directory"):
        ArtifactStore(existing, overwrite=True)

    assert (existing / "important.txt").is_file()


def test_artifact_store_overwrite_requires_owned_run_marker(tmp_path: Path) -> None:
    from post_train_engine.artifact_store import ArtifactStore

    existing = tmp_path / "runs" / "unowned"
    existing.mkdir(parents=True)
    (existing / "manifest.json").write_text("{}", encoding="utf-8")
    (existing / "important.txt").write_text("keep me", encoding="utf-8")

    with pytest.raises(ValueError, match="refusing to overwrite unmarked run directory"):
        ArtifactStore(existing, overwrite=True)

    assert (existing / "important.txt").is_file()


def test_artifact_store_overwrite_refuses_git_repository_root(tmp_path: Path) -> None:
    from post_train_engine.artifact_store import ArtifactStore

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / "manifest.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="refusing to overwrite repository root"):
        ArtifactStore(repo, overwrite=True)

    assert (repo / "manifest.json").is_file()


def test_artifact_store_overwrite_refuses_managed_junction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from post_train_engine.artifact_store import ArtifactStore

    existing = tmp_path / "run"
    ArtifactStore(existing).write_text("important.txt", "keep me")
    monkeypatch.setattr(
        Path,
        "is_junction",
        lambda path: path == existing,
    )

    with pytest.raises(ValueError, match="non-directory run path"):
        ArtifactStore(existing, overwrite=True)

    assert (existing / "important.txt").is_file()


def test_artifact_store_rejects_relative_path_escape(tmp_path: Path) -> None:
    from post_train_engine.artifact_store import ArtifactStore

    store = ArtifactStore(tmp_path / "run")
    escaped = tmp_path / "outside.txt"

    with pytest.raises(ValueError, match="outside run directory"):
        store.write_text("../outside.txt", "do not write")

    assert not escaped.exists()


def test_hillclimb_rejects_underpowered_eval_without_silent_pass(
    tmp_path: Path,
) -> None:
    config = _write_dryrun_config(
        tmp_path,
        promotion_overrides={"min_eval_examples": 3},
    )
    run_dir = tmp_path / "runs" / "gsm8k-dryrun"

    main(["hillclimb", "--config", str(config)])

    decision = json.loads((run_dir / "promotion_decision.json").read_text(encoding="utf-8"))
    report = json.loads((run_dir / "final_report.json").read_text(encoding="utf-8"))
    assert decision["decision"] == "reject"
    assert any("underpowered_eval" in reason for reason in decision["rejection_reasons"])
    assert report["promotion"]["decision"] == "reject"


def test_api_manifest_validation_fails_on_malformed_final_report(
    tmp_path: Path,
) -> None:
    config = _write_dryrun_config(tmp_path)
    run_dir = tmp_path / "runs" / "gsm8k-dryrun"

    main(["hillclimb", "--config", str(config)])

    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report_path = run_dir / "final_report.json"
    report_path.write_text("[]", encoding="utf-8")
    manifest["artifacts"]["final_report_json"]["sha256"] = _sha256(report_path)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="promotion_consistency:semantic_failed"):
        main(["artifacts", "validate", "--run", str(run_dir)])


def test_api_hillclimb_config_requires_missing_secret_for_real_provider(
    tmp_path: Path,
) -> None:
    from post_train_engine.api_hillclimb import load_hillclimb_config

    config = _write_api_config(tmp_path)
    env_path = tmp_path / ".env"
    env_path.write_text("PTE_REMOTE_BASE_URL=https://api.example.test/v1\n")

    with pytest.raises(ValueError, match="missing required secret env"):
        load_hillclimb_config(config, env_path=env_path)


def test_api_hillclimb_cli_reports_missing_secret_without_traceback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _write_api_config(tmp_path)
    env_path = tmp_path / ".env"
    env_path.write_text("PTE_REMOTE_BASE_URL=https://api.example.test/v1\n")

    with pytest.raises(SystemExit) as excinfo:
        main(["hillclimb", "--config", str(config), "--env", str(env_path)])

    captured = capsys.readouterr()
    assert excinfo.value.code == 2
    assert "pte hillclimb: error: missing required secret env" in captured.err
    assert "Traceback" not in captured.err


def test_hillclimb_dryrun_does_not_read_dotenv_without_env_refs(tmp_path: Path) -> None:
    from post_train_engine.api_hillclimb import load_hillclimb_config

    config = _write_dryrun_config(tmp_path)
    malformed_env = tmp_path / ".env"
    malformed_env.write_text("not-a-valid-dotenv-line\n", encoding="utf-8")

    resolved = load_hillclimb_config(config, env_path=malformed_env)

    assert resolved.env_redacted == {}
    assert resolved.baseline.model_id == "fake-gsm8k"


def test_hillclimb_cli_no_env_runs_without_reading_dotenv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _write_dryrun_config(tmp_path)
    (tmp_path / ".env").write_text("not-a-valid-dotenv-line\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    main(["hillclimb", "--config", str(config), "--no-env"])

    assert (tmp_path / "runs" / "gsm8k-dryrun" / "final_report.json").is_file()


def test_hillclimb_load_config_no_env_uses_process_env_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _write_api_config(tmp_path)
    malformed_env = tmp_path / ".env"
    malformed_env.write_text("not-a-valid-dotenv-line\n", encoding="utf-8")
    monkeypatch.setenv("PTE_REMOTE_BASE_URL", "https://api.example.test/v1")
    monkeypatch.setenv("PTE_REMOTE_API_KEY", "secret-value")
    monkeypatch.setenv("PTE_MODEL", "test-model")

    from post_train_engine.api_hillclimb import load_hillclimb_config

    resolved = load_hillclimb_config(config, env_path=None)

    assert resolved.inference_provider.base_url == "https://api.example.test/v1"
    assert resolved.inference_provider.api_key == "secret-value"
    assert resolved.baseline.model_id == "test-model"


def test_hyperbolic_api_smoke_config_uses_named_secret_env(tmp_path: Path) -> None:
    from post_train_engine.api_hillclimb import load_hillclimb_config

    env_path = tmp_path / ".env"
    env_path.write_text("PTE_REMOTE_HYPERBOLIC_GPU=secret-value\n", encoding="utf-8")

    resolved = load_hillclimb_config(
        "configs/gsm8k_hyperbolic_api_smoke.yaml",
        env_path=env_path,
    )

    assert resolved.inference_provider.base_url == "https://api.hyperbolic.xyz/v1"
    assert resolved.inference_provider.api_key == "secret-value"
    assert resolved.inference_provider.model == "meta-llama/Llama-3.3-70B-Instruct"
    assert resolved.training_provider.spec.api_key_env == "PTE_REMOTE_HYPERBOLIC_GPU"
    assert resolved.env_redacted["PTE_REMOTE_HYPERBOLIC_GPU"]["secret"] is True
    assert resolved.env_redacted["PTE_REMOTE_HYPERBOLIC_GPU"]["value"] == "[REDACTED]"


def test_legacy_openai_compatible_provider_type_is_rejected() -> None:
    from post_train_engine.api_schemas import ProviderSpec

    with pytest.raises(ValueError, match="type"):
        ProviderSpec(
            type="openai_compatible",
            provider_id="legacy-chat-completions",
            base_url_env="PTE_REMOTE_BASE_URL",
            api_key_env="PTE_REMOTE_API_KEY",
            model_env="PTE_REMOTE_MODEL",
        )


def test_eval_result_requires_complete_first_sample_coverage() -> None:
    from post_train_engine.api_hillclimb import _eval_result
    from post_train_engine.api_schemas import EvalExampleRecord

    rows = [
        EvalExampleRecord(
            example_id="gsm8k/train/000001",
            prompt="Problem:\n1+1?\nAssistant:\n",
            completion="<answer>2</answer>",
            parsed_answer="2",
            gold_answer="2",
            correct=True,
            parse_ok=True,
            completion_tokens=1,
            sample_index=0,
        )
    ]

    with pytest.raises(ValueError, match="eval first-sample coverage mismatch"):
        _eval_result(
            "candidate",
            rows,
            expected_example_ids={"gsm8k/train/000001", "gsm8k/train/000002"},
        )


def test_grade_generations_fails_closed_on_unknown_provider_example_id() -> None:
    from post_train_engine.api_hillclimb import _grade_generations
    from post_train_engine.tasks.gsm8k import embedded_gsm8k_examples
    from post_train_engine.api_schemas import Candidate, JobHandle, JobResult, JobStatus

    handle = JobHandle(
        job_id="job-1",
        job_type="evaluation",
        provider_id="provider",
        provider_job_id="provider-job-1",
    )
    result = JobResult(
        handle=handle,
        status=JobStatus(state="succeeded"),
        payload={
            "generations": [
                {
                    "example_id": "not-in-request",
                    "completion": "<answer>1</answer>",
                    "sample_index": 0,
                    "completion_tokens": 1,
                }
            ]
        },
    )

    with pytest.raises(ValueError, match="provider returned unknown example_id"):
        _grade_generations(
            candidate=Candidate(candidate_id="candidate", model_id="fake"),
            examples=embedded_gsm8k_examples()[:1],
            generation_result=result,
            prompt_style="thinking_tags",
        )


def test_chat_completions_provider_lifecycle_parses_chat_completion() -> None:
    from post_train_engine.api_schemas import Candidate, JobRequest
    from post_train_engine.providers.openai_compatible import OpenAICompatibleProvider

    calls: list[dict[str, Any]] = []

    def transport(
        *,
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        calls.append(
            {
                "url": url,
                "headers": headers,
                "body": body,
                "timeout_seconds": timeout_seconds,
            }
        )
        return {
            "id": "chatcmpl-test",
            "model": body["model"],
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "<answer>4</answer>"},
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
        }

    provider = OpenAICompatibleProvider(
        provider_id="chat-completions-test",
        base_url="https://api.example.test/v1",
        api_key="secret-value",
        model="test-model",
        transport=transport,
    )
    request = JobRequest(
        job_id="job-1",
        job_type="evaluation",
        provider_id="chat-completions-test",
        payload={
            "candidate": Candidate(
                candidate_id="baseline",
                model_id="test-model",
                system_prompt="Return only <answer>...</answer>.",
            ).to_json(),
            "prompts": [
                {
                    "example_id": "gsm8k/train/000000",
                    "prompt": "Problem:\n2+2?\nAssistant:\n",
                    "sample_index": 0,
                    "split_role": "eval",
                }
            ],
            "generation": {"temperature": 0.0, "max_output_tokens": 32},
        },
    )

    handle = provider.submit_job(request)
    status = provider.poll_job(handle)
    result = provider.fetch_result(handle)

    assert status.state == "succeeded"
    assert result.status.state == "succeeded"
    assert result.payload["generations"][0]["completion"] == "<answer>4</answer>"
    assert result.payload["generations"][0]["completion_tokens"] == 3
    assert calls[0]["url"] == "https://api.example.test/v1/chat/completions"
    assert calls[0]["headers"]["Authorization"] == "Bearer secret-value"
    assert "secret-value" not in json.dumps(result.to_redacted_json())


def test_chat_completions_provider_fails_closed_on_malformed_response() -> None:
    from post_train_engine.api_schemas import Candidate, JobRequest
    from post_train_engine.providers.openai_compatible import OpenAICompatibleProvider

    provider = OpenAICompatibleProvider(
        provider_id="chat-completions-test",
        base_url="https://api.example.test/v1",
        api_key="secret-value",
        model="test-model",
        transport=lambda **_kwargs: {"id": "bad", "choices": []},
    )
    request = JobRequest(
        job_id="job-1",
        job_type="evaluation",
        provider_id="chat-completions-test",
        payload={
            "candidate": Candidate(
                candidate_id="baseline",
                model_id="test-model",
            ).to_json(),
            "prompts": [
                {
                    "example_id": "gsm8k/train/000000",
                    "prompt": "Problem:\n2+2?\nAssistant:\n",
                    "sample_index": 0,
                    "split_role": "eval",
                }
            ],
            "generation": {"temperature": 0.0, "max_output_tokens": 32},
        },
    )

    with pytest.raises(RuntimeError, match="malformed chat completion response"):
        provider.submit_job(request)


def test_provider_submit_failure_writes_redacted_error_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from post_train_engine import api_hillclimb
    from post_train_engine.api_schemas import JobRequest

    config = _write_dryrun_config(tmp_path)
    run_dir = tmp_path / "runs" / "gsm8k-dryrun"

    class FailingProvider:
        provider_id = "fake-inference"
        provider_type = "fake"
        recovery_policy = "replay_safe"

        def reconcile_job(self, request, handle):  # type: ignore[no-untyped-def]
            return None

        def submit_job(self, request: JobRequest):  # type: ignore[no-untyped-def]
            raise RuntimeError("provider exploded with api_key=secret-value")

        def poll_job(self, handle):  # type: ignore[no-untyped-def]
            raise AssertionError("submit failed before polling")

        def fetch_result(self, handle):  # type: ignore[no-untyped-def]
            raise AssertionError("submit failed before fetching")

    def build_provider(resolved):  # type: ignore[no-untyped-def]
        if resolved.spec.provider_id == "fake-inference":
            return FailingProvider()
        return api_hillclimb.FakePromptAdapterProvider(provider_id="fake-trainer")

    monkeypatch.setattr(api_hillclimb, "_build_provider", build_provider)

    with pytest.raises(SystemExit) as excinfo:
        main(["hillclimb", "--config", str(config)])

    captured = capsys.readouterr()
    assert excinfo.value.code == 2
    assert "pte hillclimb: error: provider job failed closed" in captured.err
    assert "secret-value" not in captured.err
    error_rows = _read_jsonl(run_dir / "provider_errors.jsonl")
    assert len(error_rows) == 1
    serialized = json.dumps(error_rows)
    assert "provider exploded" in serialized
    assert "secret-value" not in serialized
    assert "api_key" not in serialized

    with pytest.raises(SystemExit) as replay_excinfo:
        main(["hillclimb", "--config", str(config)])

    replay_error = capsys.readouterr().err
    assert replay_excinfo.value.code == 2
    assert "terminally failed" in replay_error
    assert "failure.json" in replay_error


def test_provider_result_handle_mismatch_fails_closed(tmp_path: Path) -> None:
    from post_train_engine.api_hillclimb import _run_provider_job
    from post_train_engine.api_schemas import JobHandle, JobRequest, JobResult, JobStatus
    from post_train_engine.artifact_store import ArtifactStore

    class MismatchedResultProvider:
        provider_id = "provider"
        provider_type = "fake"
        recovery_policy = "replay_safe"

        def reconcile_job(self, request, handle):  # type: ignore[no-untyped-def]
            return None

        def submit_job(self, request: JobRequest) -> JobHandle:
            return JobHandle(
                job_id=request.job_id,
                job_type=request.job_type,
                provider_id=request.provider_id,
                provider_job_id="provider-job",
            )

        def poll_job(self, handle: JobHandle) -> JobStatus:
            return JobStatus(state="succeeded")

        def fetch_result(self, handle: JobHandle) -> JobResult:
            mismatched = JobHandle(
                job_id="other-job",
                job_type=handle.job_type,
                provider_id=handle.provider_id,
                provider_job_id=handle.provider_job_id,
            )
            return JobResult(
                handle=mismatched,
                status=JobStatus(state="succeeded"),
                payload={"generations": []},
            )

    store = ArtifactStore(tmp_path / "run")
    request = JobRequest(
        job_id="job-1",
        job_type="evaluation",
        provider_id="provider",
        payload={},
    )

    with pytest.raises(RuntimeError, match="provider result handle mismatch"):
        _run_provider_job(
            provider=MismatchedResultProvider(),
            store=store,
            request=request,
        )
    error_rows = _read_jsonl(tmp_path / "run" / "provider_errors.jsonl")
    assert error_rows[0]["job_id"] == "job-1"


def test_adapted_candidate_must_link_to_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from post_train_engine import api_hillclimb
    from post_train_engine.api_schemas import Candidate, JobHandle, JobRequest, JobResult, JobStatus

    config = _write_dryrun_config(tmp_path)

    class BadLineageAdapter:
        provider_id = "fake-trainer"
        provider_type = "fake_prompt_adapter"
        recovery_policy = "replay_safe"

        def reconcile_job(self, request, handle):  # type: ignore[no-untyped-def]
            return None

        def submit_job(self, request: JobRequest) -> JobHandle:
            handle = JobHandle(
                job_id=request.job_id,
                job_type=request.job_type,
                provider_id=request.provider_id,
                provider_job_id="bad-lineage",
            )
            self.result = JobResult(
                handle=handle,
                status=JobStatus(state="succeeded"),
                payload={
                    "candidate": Candidate(
                        candidate_id="candidate",
                        model_id="fake-gsm8k",
                        parent_id="not-baseline",
                    ).to_json()
                },
            )
            return handle

        def poll_job(self, handle: JobHandle) -> JobStatus:
            return JobStatus(state="succeeded")

        def fetch_result(self, handle: JobHandle) -> JobResult:
            return self.result

    def build_provider(resolved):  # type: ignore[no-untyped-def]
        if resolved.spec.provider_id == "fake-trainer":
            return BadLineageAdapter()
        return api_hillclimb.FakeInferenceProvider(provider_id="fake-inference")

    monkeypatch.setattr(api_hillclimb, "_build_provider", build_provider)

    with pytest.raises(SystemExit) as excinfo:
        main(["hillclimb", "--config", str(config)])

    assert excinfo.value.code == 2


def _write_dryrun_config(
    tmp_path: Path,
    *,
    promotion_overrides: dict[str, Any] | None = None,
) -> Path:
    run_dir = tmp_path / "runs" / "gsm8k-dryrun"
    promotion = {
        "primary_metric": "accuracy",
        "min_accuracy_delta": 0.0,
        "min_paired_delta_ci_low": -1.0,
        "min_eval_examples": 2,
        "max_parse_regression": 0.0,
    }
    promotion.update(promotion_overrides or {})
    body = {
        "run": {
            "certification_mode": "non_certifying_smoke",
            "run_id": "gsm8k-dryrun",
            "output_dir": str(run_dir),
            "seed": 123,
            "overwrite": True,
        },
        "dataset": {
            "name": "gsm8k",
            "source": "embedded_gsm8k_tiny",
            "train_size": 3,
            "eval_size": 2,
            "split_seed": 123,
            "prompt_style": "thinking_tags",
        },
        "baseline": {
            "candidate_id": "baseline",
            "model_id": "fake-gsm8k",
            "system_prompt": "Return only the final answer inside <answer> tags.",
        },
        "providers": {
            "inference": {"type": "fake", "provider_id": "fake-inference"},
            "training": {"type": "fake_prompt_adapter", "provider_id": "fake-trainer"},
        },
        "rollout": {
            "samples_per_example": 2,
            "temperature": 0.0,
            "max_output_tokens": 64,
        },
        "eval": {
            "samples_per_example": 1,
            "temperature": 0.0,
            "max_output_tokens": 64,
        },
        "promotion": promotion,
    }
    config = tmp_path / "gsm8k_tiny_dryrun.yaml"
    config.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    return config


def _write_api_config(tmp_path: Path) -> Path:
    body = {
        "run": {
            "certification_mode": "non_certifying_smoke",
            "run_id": "gsm8k-api-smoke",
            "output_dir": str(tmp_path / "runs" / "gsm8k-api-smoke"),
            "seed": 123,
        },
        "dataset": {
            "name": "gsm8k",
            "source": "embedded_gsm8k_tiny",
            "train_size": 3,
            "eval_size": 2,
            "split_seed": 123,
            "prompt_style": "thinking_tags",
        },
        "baseline": {"candidate_id": "baseline", "model_id_env": "PTE_MODEL"},
        "providers": {
            "inference": {
                "type": "chat_completions",
                "provider_id": "chat-completions",
                "base_url_env": "PTE_REMOTE_BASE_URL",
                "api_key_env": "PTE_REMOTE_API_KEY",
                "model_env": "PTE_MODEL",
            },
            "training": {
                "type": "chat_completions_prompt_adapter",
                "provider_id": "chat-completions-adapter",
                "base_url_env": "PTE_REMOTE_BASE_URL",
                "api_key_env": "PTE_REMOTE_API_KEY",
                "model_env": "PTE_MODEL",
            },
        },
        "rollout": {"samples_per_example": 1, "max_output_tokens": 64},
        "eval": {"samples_per_example": 1, "max_output_tokens": 64},
        "promotion": {
            "primary_metric": "accuracy",
            "min_accuracy_delta": 0.0,
            "min_paired_delta_ci_low": -1.0,
            "min_eval_examples": 2,
        },
    }
    config = tmp_path / "gsm8k_api_smoke.yaml"
    config.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    return config


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fp:
        return [json.loads(line) for line in fp if line.strip()]


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
