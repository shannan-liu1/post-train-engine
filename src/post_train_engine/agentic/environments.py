"""Minimal tool-use environment primitives for agentic RL evidence."""

from __future__ import annotations

import ast
import hashlib
import math
import operator
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from post_train_engine.traces.schema import TraceRecord, stable_prompt_hash

_FROZEN_FORBID = ConfigDict(frozen=True, extra="forbid")


class ToolEnvironmentStep(BaseModel):
    """One action/observation/reward step from a tool-use environment."""

    model_config = _FROZEN_FORBID

    step_id: str = Field(..., min_length=1)
    action: str = Field(..., min_length=1)
    action_input: str = ""
    observation: str = ""
    reward: float
    done: bool
    verifier: dict[str, Any] = Field(default_factory=dict)

    @field_validator("reward")
    @classmethod
    def _reward_finite(cls, value: float) -> float:
        if type(value) is bool or not isinstance(value, int | float) or not math.isfinite(float(value)):
            raise ValueError("reward must be finite")
        return float(value)


class ToolUseTrajectory(BaseModel):
    """Serializable trajectory that can become a common TraceRecord."""

    model_config = _FROZEN_FORBID

    trajectory_id: str = Field(..., min_length=1)
    run_id: str = Field(..., min_length=1)
    task_id: str = Field(..., min_length=1)
    example_id: str = Field(..., min_length=1)
    prompt: str = Field(..., min_length=1)
    steps: tuple[ToolEnvironmentStep, ...] = Field(..., min_length=1)
    total_reward: float
    verified: bool

    @field_validator("total_reward")
    @classmethod
    def _total_reward_finite(cls, value: float) -> float:
        if type(value) is bool or not isinstance(value, int | float) or not math.isfinite(float(value)):
            raise ValueError("total_reward must be finite")
        return float(value)

    @model_validator(mode="after")
    def _done_final_step(self) -> ToolUseTrajectory:
        if not self.steps[-1].done:
            raise ValueError("final environment step must be done")
        return self

    def completion_text(self) -> str:
        lines: list[str] = []
        for step in self.steps:
            lines.append(f"{step.action}({step.action_input})")
            lines.append(step.observation)
        return "\n".join(lines)

    def to_trace_record(self) -> TraceRecord:
        return TraceRecord(
            trace_id=self.trajectory_id,
            run_id=self.run_id,
            task_id=self.task_id,
            example_id=self.example_id,
            split_role="train",
            prompt_hash=stable_prompt_hash(self.prompt),
            source_checkpoint="environment_policy",
            policy_version="environment_policy:v1",
            policy_step=0,
            policy_step_evidence="static",
            rollout_group_id=self.trajectory_id,
            generation_backend="tool_environment",
            sampling_config={"mode": "deterministic_tool_episode"},
            verifier_id="exact_math_tool_v1",
            prompt=self.prompt,
            completion=self.completion_text(),
            verifier_result={"verified": self.verified},
            reward_components={"reward": self.total_reward},
            token_counts={"steps": float(len(self.steps))},
            privileged_visibility="environment",
        )


def run_exact_math_tool_episode(
    *,
    run_id: str,
    example_id: str,
    prompt: str,
    expression: str,
    expected_answer: str,
) -> ToolUseTrajectory:
    """Run a deterministic calculator episode for smoke-testing agentic traces."""

    actual = _safe_eval_arithmetic(expression)
    actual_text = _format_number(actual)
    correct = actual_text == expected_answer.strip()
    reward = 1.0 if correct else 0.0
    return ToolUseTrajectory(
        trajectory_id=f"{run_id}:{example_id}:trajectory-0000",
        run_id=run_id,
        task_id="exact_math_tool",
        example_id=example_id,
        prompt=prompt,
        steps=(
            ToolEnvironmentStep(
                step_id=f"{example_id}:step-0000",
                action="calculator.evaluate",
                action_input=expression,
                observation=actual_text,
                reward=reward,
                done=True,
                verifier={
                    "verifier": "exact_string_v1",
                    "expected": expected_answer.strip(),
                    "actual": actual_text,
                    "correct": correct,
                },
            ),
        ),
        total_reward=reward,
        verified=correct,
    )


_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
}


def _safe_eval_arithmetic(expression: str) -> float:
    node = ast.parse(expression, mode="eval")
    return float(_eval_node(node.body))


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return float(_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right)))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return float(_OPS[type(node.op)](_eval_node(node.operand)))
    raise ValueError("calculator only supports numeric arithmetic expressions")


def _format_number(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("calculator result must be finite")
    if value == int(value):
        return str(int(value))
    return str(value)


def _hash_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ToolObservationRef:
    text_hash: str

    @classmethod
    def from_text(cls, text: str) -> ToolObservationRef:
        return cls(text_hash=_hash_text(text))


__all__ = [
    "ToolEnvironmentStep",
    "ToolObservationRef",
    "ToolUseTrajectory",
    "run_exact_math_tool_episode",
]
