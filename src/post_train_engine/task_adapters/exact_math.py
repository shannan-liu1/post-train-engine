"""Executable-verifier exact-math task for RunEngine contract tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from post_train_engine.agentic.environments import run_exact_math_tool_episode
from post_train_engine.engine import RunPlan, RunStage, StageOutput


class ExactMathRunAdapter:
    """Build evidence and candidates for a deterministic calculator task."""

    def execute_stage(
        self,
        stage: RunStage,
        plan: RunPlan,
        prior: dict[str, StageOutput],
    ) -> StageOutput:
        root = Path(plan.output_dir)
        handlers = {
            "prepare": self._prepare,
            "data": self._data,
            "evidence": self._evidence,
            "train": self._train,
            "select": self._select,
            "evaluate": self._evaluate,
            "finalize": self._finalize,
        }
        try:
            artifacts = handlers[stage](root, plan, prior)
        except KeyError as exc:
            raise ValueError(f"ExactMathRunAdapter cannot execute engine-owned stage {stage}") from exc
        return StageOutput(artifacts=artifacts, cost_usd=0.0)

    def _prepare(
        self,
        root: Path,
        plan: RunPlan,
        _prior: dict[str, StageOutput],
    ) -> dict[str, str]:
        path = _write_json(root / "config.json", {"task": plan.task_name, "method": "sft"})
        return {"config": str(path)}

    def _data(
        self,
        root: Path,
        _plan: RunPlan,
        _prior: dict[str, StageOutput],
    ) -> dict[str, str]:
        rows = [
            {"example_id": f"math-{index}", "expression": expression, "answer": answer}
            for index, (expression, answer) in enumerate(
                (("2 + 2", "4"), ("3 * 5", "15"), ("12 - 7", "5"), ("8 / 2", "4"))
            )
        ]
        path = _write_json(root / "data" / "examples.json", rows)
        return {"dataset": str(path)}

    def _evidence(
        self,
        root: Path,
        plan: RunPlan,
        _prior: dict[str, StageOutput],
    ) -> dict[str, str]:
        traces = []
        for index, (expression, answer) in enumerate(
            (("2 + 2", "4"), ("3 * 5", "15"), ("12 - 7", "5"), ("8 / 2", "4"))
        ):
            trajectory = run_exact_math_tool_episode(
                run_id=plan.run_id,
                example_id=f"math-{index}",
                prompt=f"Evaluate {expression}",
                expression=expression,
                expected_answer=answer,
            )
            traces.append(trajectory.to_trace_record().model_dump(mode="json"))
        trace_path = _write_jsonl(root / "evidence" / "traces.jsonl", traces)
        view_path = _write_json(
            root / "evidence" / "sft_view.json",
            {
                "view_id": f"{plan.run_id}:exact-math-sft",
                "method_compatibility": ["sft"],
                "source_trace_ids": [trace["trace_id"] for trace in traces],
                "source_split_roles": ["train"],
                "privileged_visibility": "environment",
            },
        )
        return {"traces": str(trace_path), "sft_view": str(view_path)}

    def _train(
        self,
        root: Path,
        plan: RunPlan,
        _prior: dict[str, StageOutput],
    ) -> dict[str, str]:
        path = _write_json(
            root / "candidate" / "checkpoint_manifest.json",
            {
                "candidate_id": plan.candidate_id,
                "method": "sft",
                "training_view_id": f"{plan.run_id}:exact-math-sft",
                "implementation": "deterministic_task_adapter_fixture",
            },
        )
        return {"checkpoint_manifest": str(path)}

    def _select(
        self,
        root: Path,
        plan: RunPlan,
        _prior: dict[str, StageOutput],
    ) -> dict[str, str]:
        path = _write_json(
            root / "selection.json",
            {"candidate_id": plan.candidate_id, "reason": "only eligible candidate"},
        )
        return {"checkpoint_selection": str(path)}

    def _evaluate(
        self,
        root: Path,
        _plan: RunPlan,
        _prior: dict[str, StageOutput],
    ) -> dict[str, str]:
        baseline = _eval_artifact("baseline", correct=(False, False, False, False))
        candidate = _eval_artifact("candidate", correct=(True, True, True, True))
        baseline_path = _write_json(root / "eval" / "baseline.json", baseline)
        candidate_path = _write_json(root / "eval" / "candidate.json", candidate)
        return {"baseline_eval": str(baseline_path), "candidate_eval": str(candidate_path)}

    def _finalize(
        self,
        root: Path,
        _plan: RunPlan,
        prior: dict[str, StageOutput],
    ) -> dict[str, str]:
        path = _write_json(
            root / "final_report.json",
            {"promotion": {"decision": prior["promote"].values["decision"]}},
        )
        return {"final_report_json": str(path)}


def _eval_artifact(artifact_id: str, *, correct: tuple[bool, ...]) -> dict[str, Any]:
    accuracy = sum(correct) / len(correct)
    return {
        "artifact_id": artifact_id,
        "primary_metric": "accuracy",
        "metrics": {"accuracy": accuracy, "mean_tokens": 1.0},
        "slices": {"easy_stable": {"accuracy": accuracy}},
        "examples": [
            {
                "example_id": f"promotion-{index}",
                "correct": value,
                "parse_ok": True,
                "tokens": 1,
                "bucket": "easy_stable",
            }
            for index, value in enumerate(correct)
        ],
    }


def _write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


__all__ = ["ExactMathRunAdapter"]
