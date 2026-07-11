"""Run report CLI."""

from __future__ import annotations

import argparse

from post_train_engine.reports import write_run_report


def register_report_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser("report")
    parser.add_argument("--run", required=True)
    parser.set_defaults(func=cmd_report)


def cmd_report(args: argparse.Namespace) -> None:
    write_run_report(args.run)
