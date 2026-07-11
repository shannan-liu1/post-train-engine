"""Typed trace records and rollout grouping primitives."""

from post_train_engine.traces.rollouts import RolloutGroup, build_rollout_group
from post_train_engine.traces.schema import SplitRole, TraceRecord, stable_prompt_hash
from post_train_engine.traces.spans import LossMaskKind, SpanRole, TraceSpan, build_loss_mask
from post_train_engine.traces.store import JsonlTraceStore

__all__ = [
    "JsonlTraceStore",
    "LossMaskKind",
    "RolloutGroup",
    "SplitRole",
    "SpanRole",
    "TraceRecord",
    "TraceSpan",
    "build_rollout_group",
    "build_loss_mask",
    "stable_prompt_hash",
]
