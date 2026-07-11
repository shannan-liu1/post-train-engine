from __future__ import annotations

import pytest

from post_train_engine.agentic.environments import (
    ToolEnvironmentStep,
    ToolUseTrajectory,
    run_exact_math_tool_episode,
)
from post_train_engine.methods.ppo.schema import PPOTransition, compute_gae
from post_train_engine.traces.spans import TraceSpan, build_loss_mask


def test_echo_trace_spans_distinguish_actions_and_observations() -> None:
    spans = [
        TraceSpan(
            span_id="action-1",
            role="assistant_action",
            start_token=0,
            end_token=4,
            text_hash="sha256:" + "a" * 64,
            loss_mask_kind="policy",
        ),
        TraceSpan(
            span_id="obs-1",
            role="environment_observation",
            start_token=4,
            end_token=7,
            text_hash="sha256:" + "b" * 64,
            tool_name="calculator",
            stdout_hash="sha256:" + "c" * 64,
            loss_mask_kind="observation_aux",
        ),
    ]

    assert build_loss_mask(spans, total_tokens=7, include_kinds={"policy"}) == [
        1,
        1,
        1,
        1,
        0,
        0,
        0,
    ]
    assert build_loss_mask(spans, total_tokens=7, include_kinds={"observation_aux"}) == [
        0,
        0,
        0,
        0,
        1,
        1,
        1,
    ]


def test_agentic_tool_environment_preserves_verifier_and_step_evidence() -> None:
    trajectory = run_exact_math_tool_episode(
        run_id="run-1",
        example_id="math-1",
        prompt="Use calculator: 2 + 2",
        expression="2 + 2",
        expected_answer="4",
    )

    assert isinstance(trajectory, ToolUseTrajectory)
    assert trajectory.verified is True
    assert trajectory.total_reward == 1.0
    assert trajectory.steps == (
        ToolEnvironmentStep(
            step_id="math-1:step-0000",
            action="calculator.evaluate",
            action_input="2 + 2",
            observation="4",
            reward=1.0,
            done=True,
            verifier={"verifier": "exact_string_v1", "expected": "4", "actual": "4", "correct": True},
        ),
    )
    assert trajectory.to_trace_record().completion == "calculator.evaluate(2 + 2)\n4"


def test_ppo_transition_and_gae_fail_closed_on_nonfinite_values() -> None:
    transitions = [
        PPOTransition(
            trace_id="trace-1",
            reward=1.0,
            value=0.2,
            logprob=-0.1,
            done=False,
        ),
        PPOTransition(
            trace_id="trace-2",
            reward=0.5,
            value=0.1,
            logprob=-0.2,
            done=True,
        ),
    ]

    result = compute_gae(transitions, gamma=1.0, lam=1.0)

    assert result.advantages == pytest.approx((1.3, 0.4))
    assert result.returns == pytest.approx((1.5, 0.5))

    with pytest.raises(ValueError, match="finite"):
        PPOTransition(trace_id="bad", reward=float("nan"), value=0.0, logprob=0.0, done=True)
