"""Eval-driven post-training optimization primitives."""

from post_train_engine.difficulty import (
    DifficultyBand,
    DifficultyBucket,
    DifficultyBucketConfig,
    DifficultyBucketRecord,
    DifficultyMap,
    DifficultyRecord,
)
from post_train_engine.evals.promotion import (
    CanaryDecision,
    EvalArtifact,
    PromotionDecision,
    PromotionGateConfig,
    SliceGateConfig,
    canary_decision_from_artifact,
    eval_report_to_artifact,
)
from post_train_engine.evals.suites import (
    PromotionSuiteState,
    SuiteRotationPolicy,
    SuiteRotationRecommendation,
)
from post_train_engine.evals.harness import (
    EvalReport,
    EvalSpec,
    MetricResult,
    SliceReport,
    evaluate_model,
    write_eval_report,
)
from post_train_engine.flywheel import (
    CostRecord,
    PriceSnapshot,
    ResourceTopology,
    ShardInfo,
    ShardPlan,
)
from post_train_engine.lifecycle import (
    CheckpointLifecycleInput,
    CheckpointLifecycleManager,
    CheckpointLifecycleOutcome,
    HuggingFaceCheckpointUploader,
    HuggingFaceLifecycleConfig,
    ModelLifecycleConfig,
)
from post_train_engine.probe import ProbeResult, ProbeRunner, ProbeTrace
from post_train_engine.rewards.gsm8k import (
    GSM8KReward,
    GSM8KRewardConfig,
    compute_gsm8k_reward,
)

__all__ = [
    "CanaryDecision",
    "CostRecord",
    "DifficultyBand",
    "DifficultyBucket",
    "DifficultyBucketConfig",
    "DifficultyBucketRecord",
    "DifficultyMap",
    "DifficultyRecord",
    "EvalArtifact",
    "EvalReport",
    "EvalSpec",
    "MetricResult",
    "PriceSnapshot",
    "ProbeResult",
    "ProbeRunner",
    "ProbeTrace",
    "PromotionDecision",
    "PromotionGateConfig",
    "PromotionSuiteState",
    "ResourceTopology",
    "ShardInfo",
    "ShardPlan",
    "SliceReport",
    "SliceGateConfig",
    "SuiteRotationPolicy",
    "SuiteRotationRecommendation",
    "GSM8KReward",
    "GSM8KRewardConfig",
    "CheckpointLifecycleInput",
    "CheckpointLifecycleManager",
    "CheckpointLifecycleOutcome",
    "HuggingFaceCheckpointUploader",
    "HuggingFaceLifecycleConfig",
    "ModelLifecycleConfig",
    "canary_decision_from_artifact",
    "compute_gsm8k_reward",
    "eval_report_to_artifact",
    "evaluate_model",
    "write_eval_report",
]
