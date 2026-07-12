"""Run the canonical RunPod evaluation runtime benchmark."""

from __future__ import annotations

import argparse

from post_train_engine.runpod_grpo import run_runpod_eval_benchmark


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run_runpod_eval_benchmark(args.config, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
