from __future__ import annotations

import hashlib
import json
import re
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import yaml

from post_train_engine.cli.main import main
from post_train_engine.artifact_store import ArtifactStore
from post_train_engine.engine import CANONICAL_STAGE_ORDER
from post_train_engine.run_bundle import RunBundle
from post_train_engine.runpod_grpo import (
    DistributedContext,
    RunPodGRPOConfig,
    _align_model_with_tokenizer,
    _as_batch,
    _candidate_checkpoint_refs,
    _checkpoint_score,
    _filter_trl_config_kwargs,
    _grpo_config_kwargs,
    _evaluate_hf_model,
    _gsm8k_reward_func,
    _load_and_split_dataset,
    _select_checkpoint,
    _shard_sequence,
    _validate_hf_upload_config,
    _validate_launch_topology,
    _validate_grpo_runtime_shape,
    _write_dataset_artifacts,
    _write_measured_training_view,
    is_runpod_grpo_config,
    load_runpod_grpo_config,
)
from post_train_engine.tasks.gsm8k import GSM8KExample


def test_batched_runpod_eval_matches_scalar_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch
    import post_train_engine.runpod_grpo as runpod_module

    class Batch(dict):
        def to(self, _device):
            return self

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 99
        bos_token_id = None
        pad_token = "<pad>"
        padding_side = "left"

        def __len__(self) -> int:
            return 100

        def __call__(self, prompts, **_kwargs):
            values = [
                int(re.search(r"Problem:\n(\d+)", prompt).group(1))
                for prompt in prompts
            ]
            return Batch(input_ids=torch.tensor([[0, value] for value in values]))

        def decode(self, token_ids, *, skip_special_tokens: bool):
            assert skip_special_tokens is True
            return f"<answer>{int(token_ids[0])}</answer>"

    class Config:
        pad_token_id = None
        eos_token_id = None
        bos_token_id = None

    class FakeModel:
        config = Config()
        generation_config = Config()

        def eval(self):
            return self

        def get_input_embeddings(self):
            return types.SimpleNamespace(num_embeddings=100)

        def to(self, _device):
            return self

        def generate(self, **kwargs):
            input_ids = kwargs["input_ids"]
            return torch.cat((input_ids, input_ids[:, -1:]), dim=1)

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return FakeModel()

    monkeypatch.setattr(runpod_module, "_load_tokenizer", lambda *_args, **_kwargs: FakeTokenizer())
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoModelForCausalLM=FakeAutoModel),
    )
    cfg = load_runpod_grpo_config(_write_config(tmp_path))
    examples = [
        GSM8KExample(
            id=f"ex-{value}",
            split="train",
            question=str(value),
            gold_solution=f"#### {value}",
            gold_answer=str(value),
            source="fixture",
        )
        for value in (2, 4, 6, 8)
    ]
    scalar = cfg.model_copy(update={"eval": cfg.eval.model_copy(update={"batch_size": 1})})
    batched = cfg.model_copy(update={"eval": cfg.eval.model_copy(update={"batch_size": 3})})

    scalar_rows = _evaluate_hf_model(
        cfg=scalar,
        model_ref=cfg.model.base_model_id,
        examples=examples,
        dist=DistributedContext(),
    )
    batched_rows = _evaluate_hf_model(
        cfg=batched,
        model_ref=cfg.model.base_model_id,
        examples=examples,
        dist=DistributedContext(),
    )

    assert batched_rows == scalar_rows


def test_runpod_grpo_builds_policy_lineage_training_view(tmp_path: Path) -> None:
    cfg = load_runpod_grpo_config(_write_config(tmp_path))
    train, selection, promotion = _load_and_split_dataset(cfg)
    store = ArtifactStore(cfg.run.output_dir)
    _write_dataset_artifacts(store, cfg, train, selection, promotion)
    probe_rows = [
        _eval_row(train[0].id, correct=False, sample_index=0),
        _eval_row(train[0].id, correct=True, sample_index=1),
        _eval_row(train[1].id, correct=True, sample_index=0),
        _eval_row(train[1].id, correct=True, sample_index=1),
        _eval_row(train[2].id, correct=False, sample_index=0),
        _eval_row(train[2].id, correct=False, sample_index=1),
        *[
            _eval_row(example.id, correct=True, sample_index=sample_index)
            for example in train[3:]
            for sample_index in (0, 1)
        ],
    ]

    view = _write_measured_training_view(store, cfg, train, probe_rows)

    assert view is not None
    assert view.method_compatibility == ("grpo",)
    assert view.source_trace_ids
    assert view.source_split_roles == ("train",)
    rows = [
        json.loads(line)
        for line in (store.run_dir / "evidence" / "input_traces.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {row["policy_version"] for row in rows} == {cfg.model.base_model_id}
    assert all(row["rollout_group_id"] for row in rows)
    assert view.metadata["selection_evidence"] == "measured_parent_success_rate"
    assert view.metadata["selected_example_ids"] == [train[0].id]


def test_runpod_grpo_emits_non_training_outcome_without_parent_frontier(
    tmp_path: Path,
) -> None:
    cfg = load_runpod_grpo_config(_write_config(tmp_path))
    train, selection, promotion = _load_and_split_dataset(cfg)
    store = ArtifactStore(cfg.run.output_dir)
    _write_dataset_artifacts(store, cfg, train, selection, promotion)
    probe_rows = [
        _eval_row(example.id, correct=True, sample_index=sample_index)
        for example in train
        for sample_index in range(cfg.training.num_generations)
    ]

    view = _write_measured_training_view(store, cfg, train, probe_rows)

    assert view is None
    outcome = json.loads(
        (store.run_dir / "evidence" / "non_training_outcome.json").read_text(
            encoding="utf-8"
        )
    )
    assert outcome["outcome"] == "no_learnable_evidence"


def test_runpod_lifecycle_fails_closed_on_ambiguous_remote_transaction(
    tmp_path: Path,
) -> None:
    from post_train_engine.runpod_grpo import _finalize_lifecycle_if_configured

    cfg = load_runpod_grpo_config(
        _write_config(
            tmp_path,
            {
                "hf_upload": {
                    "enabled": True,
                    "repo_id": "owner/repo",
                    "token_env": "HF_TOKEN",
                }
            },
        )
    )
    store = ArtifactStore(cfg.run.output_dir)
    store.write_json("lifecycle/transaction.json", {"state": "started"})

    with pytest.raises(ValueError, match="remote lifecycle transaction is ambiguous"):
        _finalize_lifecycle_if_configured(
            cfg=cfg,
            store=store,
            candidate={"candidate_id": "candidate", "model_id": "checkpoint"},
            train_result={"metrics": {}},
            decision={"decision": "reject"},
        )


def test_runpod_grpo_configs_parse_and_are_dispatched() -> None:
    smoke = load_runpod_grpo_config("configs/gsm8k_runpod_smoke.yaml")

    assert is_runpod_grpo_config("configs/gsm8k_runpod_smoke.yaml")
    assert smoke.execution.provider == "runpod"
    assert smoke.execution.gpu_count == 2
    assert smoke.execution.cuda_version == "12.8"
    assert smoke.training.max_steps == 1
    assert smoke.run.certification_mode == "non_certifying_smoke"
    assert smoke.checkpoint_selection.enabled is True
    assert smoke.trace_capture.enabled is True


def test_runpod_config_derives_cuda_filter_from_image(
    tmp_path: Path,
) -> None:
    config = _write_config(
        tmp_path,
        {
            "execution": {
                "container_image": (
                    "runpod/pytorch:2.8.0-py3.11-cuda12.4.1-"
                    "cudnn-devel-ubuntu22.04"
                )
            }
        },
    )

    assert load_runpod_grpo_config(config).execution.cuda_version == "12.4"


def test_runpod_config_rejects_image_without_parseable_cuda_filter(
    tmp_path: Path,
) -> None:
    config = _write_config(
        tmp_path,
        {"execution": {"container_image": "example.invalid/pytorch:latest"}},
    )

    with pytest.raises(ValueError, match="parseable cudaMAJOR.MINOR"):
        load_runpod_grpo_config(config)


def test_runpod_config_round_trips_without_serializing_derived_cuda_filter() -> None:
    config = load_runpod_grpo_config("configs/gsm8k_runpod_smoke.yaml")

    restored = RunPodGRPOConfig.model_validate(config.model_dump(mode="json"))

    assert restored.execution.cuda_version == "12.8"


def test_runpod_grpo_config_rejects_save_steps_after_training(tmp_path: Path) -> None:
    config = _write_config(tmp_path, {"training": {"max_steps": 1, "save_steps": 2}})

    with pytest.raises(ValueError, match="save_steps"):
        load_runpod_grpo_config(config)


def test_runpod_grpo_runtime_shape_requires_batch_divisible_by_generations() -> None:
    cfg = load_runpod_grpo_config("configs/gsm8k_runpod_smoke.yaml")
    _validate_grpo_runtime_shape(cfg.training, world_size=2)

    bad = cfg.model_copy(
        update={
            "training": cfg.training.model_copy(
                update={"per_device_train_batch_size": 1, "num_generations": 4}
            )
        }
    )
    with pytest.raises(ValueError, match="divisible by num_generations"):
        _validate_grpo_runtime_shape(bad.training, world_size=2)


def test_runpod_grpo_launch_topology_requires_accelerate_for_multi_gpu() -> None:
    cfg = load_runpod_grpo_config("configs/gsm8k_runpod_smoke.yaml")

    with pytest.raises(ValueError, match="requires accelerate launch"):
        _validate_launch_topology(cfg, DistributedContext(world_size=1, rank=0, local_rank=0))

    _validate_launch_topology(cfg, DistributedContext(world_size=2, rank=0, local_rank=0))


def test_distributed_context_from_env_identifies_rank_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("LOCAL_RANK", "1")

    ctx = DistributedContext.from_env()

    assert ctx.world_size == 4
    assert ctx.rank == 2
    assert ctx.local_rank == 1
    assert not ctx.is_main_process
    assert ctx.is_distributed


def test_eval_sharding_covers_examples_once_across_ranks() -> None:
    rows = list(range(17))
    shards = [
        _shard_sequence(rows, DistributedContext(world_size=4, rank=rank, local_rank=rank))
        for rank in range(4)
    ]

    assert sorted(item for shard in shards for item in shard) == rows
    assert sum(len(shard) for shard in shards) == len(rows)
    assert all(len(set(shard)) == len(shard) for shard in shards)


def test_grpo_config_kwargs_filter_optional_trl_fields() -> None:
    cfg = load_runpod_grpo_config("configs/gsm8k_runpod_smoke.yaml")

    class MinimalGRPOConfig:
        def __init__(
            self,
            *,
            output_dir: str,
            max_steps: int,
            learning_rate: float,
            per_device_train_batch_size: int,
            gradient_accumulation_steps: int,
            bf16: bool,
            fp16: bool,
            tf32: bool,
            gradient_checkpointing: bool,
            logging_steps: int,
            save_steps: int,
            save_total_limit: int,
            report_to: str,
            run_name: str,
            remove_unused_columns: bool,
            seed: int,
            num_generations: int,
            max_completion_length: int,
            temperature: float,
            top_p: float,
            beta: float,
        ) -> None:
            pass

    kwargs = _filter_trl_config_kwargs(_grpo_config_kwargs(cfg), MinimalGRPOConfig)

    assert kwargs["num_generations"] == 2
    assert "generation_kwargs" not in kwargs
    assert "use_vllm" not in kwargs
    assert "vllm_gpu_memory_utilization" not in kwargs


def test_gsm8k_reward_batch_aliases_and_repeated_answers() -> None:
    assert _as_batch(["5", "6"], n=4, field_name="answer") == ["5", "5", "6", "6"]

    with pytest.raises(ValueError, match="does not match completions"):
        _as_batch(["1", "2", "3"], n=4, field_name="answer")


def test_grpo_reward_function_captures_training_rollout_traces(tmp_path: Path) -> None:
    class FakeTokenizer:
        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            assert add_special_tokens is False
            return text.split()

    trace_path = tmp_path / "traces" / "rank0.jsonl"
    reward = _gsm8k_reward_func(
        FakeTokenizer(),
        config=load_runpod_grpo_config("configs/gsm8k_runpod_smoke.yaml").training_reward_config(),
        trace_path=trace_path,
        run_id="run-001",
        source_checkpoint="Qwen/Qwen2.5-0.5B-Instruct",
        rank=0,
    )

    rewards = reward(
        completions=["reasoning <answer>4</answer>", "bad <answer>5</answer>"],
        answer=["4", "4"],
        prompt=["What is 2+2?", "What is 2+2?"],
        example_id=["ex-1", "ex-1"],
    )

    assert len(rewards) == 2
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert [row["example_id"] for row in rows] == ["ex-1", "ex-1"]
    assert rows[0]["completion"] == "reasoning <answer>4</answer>"
    assert rows[0]["reward_components"]["task_reward"] == 1.0
    assert rows[0]["policy_version"].startswith("run-001:rank0:reward-batch:")
    assert rows[0]["policy_step_evidence"] == "inferred_batch"
    assert rows[0]["rollout_group_id"] == rows[1]["rollout_group_id"]
    assert rows[1]["verifier_result"]["correct"] is False


def test_candidate_checkpoint_refs_include_sorted_checkpoints_and_final(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    for name in ("checkpoint-10", "checkpoint-2", "final"):
        path = train_dir / name
        path.mkdir(parents=True)
        (path / "model.safetensors").write_text("x", encoding="utf-8")

    refs = _candidate_checkpoint_refs(tmp_path)

    assert [ref["checkpoint_id"] for ref in refs] == ["checkpoint-2", "checkpoint-10", "final"]


def test_select_checkpoint_uses_metric_then_token_tie_breaker() -> None:
    rows_a = [
        _eval_row("a", correct=True, completion_tokens=9),
        _eval_row("b", correct=False, completion_tokens=9),
    ]
    rows_b = [
        _eval_row("a", correct=True, completion_tokens=4),
        _eval_row("b", correct=False, completion_tokens=4),
    ]

    selection = _select_checkpoint(
        [
            {"checkpoint_id": "checkpoint-1", "path": "train/checkpoint-1", "rows": rows_a},
            {"checkpoint_id": "checkpoint-2", "path": "train/checkpoint-2", "rows": rows_b},
        ],
        metric="accuracy",
    )

    assert selection["selected_checkpoint_id"] == "checkpoint-2"
    assert selection["selection_reason"] == "max_accuracy_then_min_mean_tokens"


def test_checkpoint_score_rejects_unknown_selection_metric() -> None:
    with pytest.raises(ValueError, match="unsupported checkpoint selection metric"):
        _checkpoint_score([_eval_row("a", correct=True)], "loss")


def test_hf_upload_config_requires_token_only_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = load_runpod_grpo_config("configs/gsm8k_runpod_smoke.yaml")
    _validate_hf_upload_config(cfg)
    enabled = cfg.model_copy(
        update={
            "hf_upload": cfg.hf_upload.model_copy(
                update={"enabled": True, "repo_id": "user/post-train-gsm8k"}
            )
        }
    )

    monkeypatch.delenv("PTE_REMOTE_HF_WRITE", raising=False)
    with pytest.raises(ValueError, match="missing required HF token env"):
        _validate_hf_upload_config(enabled)

    monkeypatch.setenv("PTE_REMOTE_HF_WRITE", "hf_fake")
    _validate_hf_upload_config(enabled)


def test_runpod_hillclimb_fails_closed_without_cuda_but_writes_preflight_artifacts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _write_config(tmp_path)
    run_dir = tmp_path / "runs" / "gsm8k-runpod-test"

    with pytest.raises(SystemExit) as excinfo:
        main(["hillclimb", "--config", str(config)])

    captured = capsys.readouterr()
    assert excinfo.value.code == 2
    assert "requires CUDA" in captured.err
    assert (run_dir / "config.raw.yaml").is_file()
    assert (run_dir / "config.resolved.json").is_file()
    assert (run_dir / "environment.json").is_file()


def test_runpod_compatibility_command_executes_canonical_engine_with_fakes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import post_train_engine.runpod_grpo as runpod_module

    config = _write_config(tmp_path)
    cfg = load_runpod_grpo_config(config)

    monkeypatch.setattr(runpod_module, "_require_cuda", lambda _cfg: None)
    monkeypatch.setattr(
        runpod_module,
        "_resolve_hub_revisions",
        lambda value: value.model_copy(
            update={
                "model": value.model.model_copy(
                    update={"resolved_revision": "a" * 40}
                )
            }
        ),
    )

    def fake_evaluate_hf_model(
        *,
        cfg: Any,
        model_ref: str,
        examples: list[GSM8KExample],
        dist: DistributedContext,
        mode: str = "evaluation",
        samples_per_example: int = 1,
    ) -> list[Any]:
        del dist
        rows = []
        for example_index, example in enumerate(examples):
            for sample_index in range(samples_per_example):
                correct = (
                    sample_index % 2 == 1
                    if mode == "training_probe"
                    else model_ref != cfg.model.base_model_id or example_index == 0
                )
                rows.append(
                    _eval_row(
                        example.id,
                        correct=correct,
                        sample_index=sample_index,
                    )
                )
        return rows

    def fake_train_grpo(
        cfg: Any,
        _view: Any,
        store: ArtifactStore | None,
        *,
        dist: DistributedContext,
    ) -> dict[str, Any]:
        assert store is not None
        checkpoint = Path(cfg.run.output_dir) / "train" / "final"
        checkpoint.mkdir(parents=True, exist_ok=True)
        (checkpoint / "model.safetensors").write_bytes(b"fixture")
        return {
            "status": "trained",
            "metrics": {"train_loss": 0.0},
            "distributed": dist.to_json(),
            "trace_capture": {},
        }

    monkeypatch.setattr(runpod_module, "_evaluate_hf_model", fake_evaluate_hf_model)
    monkeypatch.setattr(runpod_module, "_train_grpo", fake_train_grpo)

    main(["hillclimb", "--config", str(config)])

    run_dir = Path(cfg.run.output_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["metadata"]["stage_order"] == list(CANONICAL_STAGE_ORDER)
    assert {
        name
        for name in manifest["artifacts"]
        if name.startswith("stage_receipt_")
    } == {f"stage_receipt_{stage}" for stage in CANONICAL_STAGE_ORDER}
    assert manifest["status"] == "rejected"
    assert manifest["metadata"]["certification_mode"] == "non_certifying_smoke"
    assert manifest["metadata"]["cost_certifying"] is False
    assert set(manifest["metadata"]["missing_cost_stages"]) >= {
        "evidence",
        "train",
        "select",
        "evaluate",
    }
    assert RunBundle.load(run_dir).validate()["status"] == "ok"

    trace_path = run_dir / "evidence" / "input_traces.jsonl"
    trace_count = len(trace_path.read_text(encoding="utf-8").splitlines())
    (run_dir / "manifest.json").unlink()
    resume_from = CANONICAL_STAGE_ORDER.index("evidence")
    for stage in CANONICAL_STAGE_ORDER[resume_from:]:
        (run_dir / "state" / f"{stage}.json").unlink()

    main(["hillclimb", "--config", str(config)])

    assert len(trace_path.read_text(encoding="utf-8").splitlines()) == trace_count
    events = [
        json.loads(line)["event"]
        for line in (run_dir / "logs" / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(events) == len(set(events))
    assert RunBundle.load(run_dir).validate()["status"] == "ok"
    assert (run_dir / "config.raw.yaml").is_file()
    assert (run_dir / "config.resolved.json").is_file()
    assert (run_dir / "environment.json").is_file()
    assert (run_dir / "command.txt").is_file()
    env = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
    assert "cuda" in env

    view = json.loads(
        (run_dir / "evidence" / "method_training_view.json").read_text(
            encoding="utf-8"
        )
    )
    missing_trace_id = view["source_trace_ids"][0]
    retained = [
        line
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["trace_id"] != missing_trace_id
    ]
    trace_path.write_text("\n".join(retained) + "\n", encoding="utf-8")
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["input_traces"]["sha256"] = (
        "sha256:" + hashlib.sha256(trace_path.read_bytes()).hexdigest()
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    validation = RunBundle.load(run_dir).validate()
    assert validation["status"] == "failed"
    assert any(
        failure["name"] == "grpo_reward_evidence"
        for failure in validation["failures"]
    )


def test_runpod_multi_gpu_config_refuses_direct_python_without_accelerate(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["hillclimb", "--config", "configs/gsm8k_runpod_smoke.yaml"])

    captured = capsys.readouterr()
    assert excinfo.value.code == 2
    assert "requires accelerate launch" in captured.err
    assert "WORLD_SIZE=1" in captured.err


def test_align_model_with_tokenizer_matches_generation_ids() -> None:
    class FakeEmbeddings:
        num_embeddings = 3

    class FakeConfig:
        pad_token_id = None
        eos_token_id = None
        bos_token_id = None

    class FakeModel:
        def __init__(self) -> None:
            self.config = FakeConfig()
            self.generation_config = FakeConfig()
            self.resized_to = None

        def get_input_embeddings(self) -> FakeEmbeddings:
            return FakeEmbeddings()

        def resize_token_embeddings(self, token_count: int) -> None:
            self.resized_to = token_count

    class FakeTokenizer:
        padding_side = "right"
        pad_token_id = 10
        eos_token_id = 11
        bos_token_id = None

        def __len__(self) -> int:
            return 12

    model = FakeModel()
    tokenizer = FakeTokenizer()

    _align_model_with_tokenizer(model, tokenizer)

    assert tokenizer.padding_side == "left"
    assert model.resized_to == 12
    assert model.config.pad_token_id == 10
    assert model.generation_config.eos_token_id == 11


def _eval_row(
    example_id: str,
    *,
    correct: bool,
    completion_tokens: int = 1,
    sample_index: int = 0,
):
    from post_train_engine.runpod_grpo import EvalRow

    return EvalRow(
        example_id=example_id,
        prompt=f"prompt {example_id}",
        completion=f"completion {example_id}",
        parsed_answer="1",
        gold_answer="1",
        correct=correct,
        parse_ok=True,
        completion_tokens=completion_tokens,
        sample_index=sample_index,
    )


def _write_config(
    tmp_path: Path,
    overrides: dict[str, Any] | None = None,
) -> Path:
    run_dir = tmp_path / "runs" / "gsm8k-runpod-test"
    body: dict[str, Any] = {
        "schema_version": "runpod_grpo_hillclimb_v1",
        "execution": {
            "mode": "runpod_manual_grpo",
            "provider": "runpod",
            "gpu_type": "A40",
            "gpu_count": 1,
            "container_image": (
                "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-"
                "cudnn-devel-ubuntu22.04"
            ),
            "disk_gb": 100,
            "volume_gb": 150,
        },
        "run": {
            "certification_mode": "non_certifying_smoke",
            "run_id": "gsm8k-runpod-test",
            "output_dir": str(run_dir),
            "seed": 42,
            "overwrite": True,
        },
        "model": {"base_model_id": "Qwen/Qwen2.5-0.5B-Instruct"},
        "dataset": {
            "source": "embedded_gsm8k_tiny",
            "train_size": 4,
            "selection_size": 2,
            "eval_size": 2,
            "split_seed": 42,
        },
        "training": {
            "method": "grpo",
            "max_steps": 1,
            "learning_rate": 5.0e-7,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 2,
            "num_generations": 2,
            "max_completion_length": 64,
            "save_steps": 1,
            "logging_steps": 1,
        },
        "promotion": {"min_eval_examples": 2},
    }
    _deep_update(body, overrides or {})
    path = tmp_path / "gsm8k_runpod_test.yaml"
    path.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    return path


def _deep_update(target: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
