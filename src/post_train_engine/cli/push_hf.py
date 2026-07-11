"""Hugging Face push planning CLI."""

from __future__ import annotations

import argparse

from post_train_engine.push_hf import write_hf_push_plan


def register_push_hf_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser("push-hf")
    parser.add_argument("--run", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--repo-type", default="model")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.set_defaults(func=cmd_push_hf)


def cmd_push_hf(args: argparse.Namespace) -> None:
    write_hf_push_plan(
        args.run,
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        private=args.private,
        dry_run=args.dry_run,
    )
