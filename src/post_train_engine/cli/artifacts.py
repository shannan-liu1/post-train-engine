"""Artifact bundle inspection CLI."""

from __future__ import annotations

import argparse

from post_train_engine.artifacts import require_valid_run_bundle, validate_run_bundle


def register_artifacts_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser("artifacts")
    artifact_subparsers = parser.add_subparsers(dest="artifacts_command", required=True)

    validate_parser = artifact_subparsers.add_parser("validate")
    validate_parser.add_argument("--run", required=True)
    validate_parser.set_defaults(func=cmd_artifacts_validate)

    list_parser = artifact_subparsers.add_parser("list")
    list_parser.add_argument("--run", required=True)
    list_parser.set_defaults(func=cmd_artifacts_list)


def cmd_artifacts_validate(args: argparse.Namespace) -> None:
    require_valid_run_bundle(args.run)


def cmd_artifacts_list(args: argparse.Namespace) -> None:
    validate_run_bundle(args.run, write=True)
