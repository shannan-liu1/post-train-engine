"""Run diagnostics CLI."""

from __future__ import annotations

import argparse

from post_train_engine.diagnostics import write_run_diagnostics


def register_diagnose_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser("diagnose")
    parser.add_argument("--run", required=True)
    parser.set_defaults(func=cmd_diagnose)


def cmd_diagnose(args: argparse.Namespace) -> None:
    write_run_diagnostics(args.run)
