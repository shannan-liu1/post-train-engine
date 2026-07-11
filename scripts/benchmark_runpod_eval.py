"""Measure RunPod evaluation batching and model reuse with output parity."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from post_train_engine.runpod_grpo import (
    DistributedContext,
    _distributed_state,
    _evaluate_hf_model,
    _gather_eval_rows,
    _load_and_split_dataset,
    _require_cuda,
    _resolve_hub_revisions,
    _shard_sequence,
    _validate_launch_topology,
    _wait_for_everyone,
    load_runpod_grpo_config,
    runtime_environment,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = load_runpod_grpo_config(args.config)
    dist = DistributedContext.from_env()
    _validate_launch_topology(cfg, dist)
    state = _distributed_state(dist)
    runtime_attestation = _require_cuda(cfg)
    cfg = _resolve_hub_revisions(cfg)
    _train, _selection, promotion = _load_and_split_dataset(cfg)
    local_examples = _shard_sequence(promotion, dist)

    _wait_for_everyone(state)
    scalar_started = time.perf_counter()
    scalar_local = []
    for example in local_examples:
        scalar_local.extend(
            _evaluate_hf_model(
                cfg=cfg.model_copy(
                    update={"eval": cfg.eval.model_copy(update={"batch_size": 1})}
                ),
                model_ref=cfg.model.base_model_id,
                examples=[example],
                dist=dist,
            )
        )
    scalar_seconds = _max_rank_seconds(time.perf_counter() - scalar_started, dist)
    scalar_rows = _gather_eval_rows(scalar_local, dist)

    _wait_for_everyone(state)
    optimized_started = time.perf_counter()
    optimized_local = _evaluate_hf_model(
        cfg=cfg,
        model_ref=cfg.model.base_model_id,
        examples=local_examples,
        dist=dist,
    )
    optimized_seconds = _max_rank_seconds(
        time.perf_counter() - optimized_started,
        dist,
    )
    optimized_rows = _gather_eval_rows(optimized_local, dist)

    if not dist.is_main_process:
        return 0
    scalar_payload = [row.to_json() for row in scalar_rows]
    optimized_payload = [row.to_json() for row in optimized_rows]
    parity = scalar_payload == optimized_payload
    speedup = scalar_seconds / optimized_seconds if optimized_seconds > 0 else None
    result: dict[str, Any] = {
        "schema_version": "runpod_eval_runtime_benchmark_v1",
        "certifying": bool(parity and speedup is not None and speedup > 1.0),
        "config": str(Path(args.config)),
        "model_id": cfg.model.base_model_id,
        "model_revision": cfg.model.resolved_revision,
        "dataset_revision": (
            "embedded-gsm8k-tiny-v1"
            if cfg.dataset.source == "embedded_gsm8k_tiny"
            else cfg.dataset.resolved_revision
        ),
        "topology": dist.to_json(),
        "environment": runtime_environment(dist),
        "runtime_attestation": runtime_attestation,
        "example_count": len(promotion),
        "baseline": {
            "strategy": "one_model_load_per_example",
            "model_load_count_per_rank": len(local_examples),
            "batch_size": 1,
            "max_rank_wall_seconds": scalar_seconds,
        },
        "optimized": {
            "strategy": "one_model_load_per_shard_with_batching",
            "model_load_count_per_rank": 1,
            "batch_size": cfg.eval.batch_size,
            "max_rank_wall_seconds": optimized_seconds,
        },
        "speedup": speedup,
        "output_parity": parity,
        "output_sha256": _stable_hash(optimized_payload),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    if not parity:
        raise RuntimeError("batched evaluation output drifted from scalar evaluation")
    if speedup is None or speedup <= 1.0:
        raise RuntimeError(
            "batched model reuse did not improve max-rank wall time; inspect benchmark artifact"
        )
    return 0


def _max_rank_seconds(seconds: float, dist: DistributedContext) -> float:
    if not dist.is_distributed:
        return seconds
    import torch
    import torch.distributed as torch_dist

    value = torch.tensor(seconds, device=f"cuda:{dist.local_rank}")
    torch_dist.all_reduce(value, op=torch_dist.ReduceOp.MAX)
    return float(value.item())


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
