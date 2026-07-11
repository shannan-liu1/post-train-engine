"""API-first hill-climb CLI."""

from __future__ import annotations

import argparse

from post_train_engine.cli.run import execute_run_config


def register_hillclimb_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser("hillclimb")
    parser.add_argument("--config", required=True)
    env_group = parser.add_mutually_exclusive_group()
    env_group.add_argument("--env", default=".env")
    env_group.add_argument(
        "--no-env",
        action="store_true",
        help="Do not read a dotenv file; resolve env vars from the process only.",
    )
    parser.set_defaults(func=lambda args: cmd_hillclimb(args, parser))


def cmd_hillclimb(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> None:
    try:
        execute_run_config(
            args.config,
            env_path=None if args.no_env else args.env,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"pte hillclimb: error: {exc}\n")
