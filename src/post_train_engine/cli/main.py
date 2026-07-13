"""Top-level ``pte`` command."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from post_train_engine.cli.artifacts import register_artifacts_parser
from post_train_engine.cli.diagnose import register_diagnose_parser
from post_train_engine.cli.hillclimb import register_hillclimb_parser
from post_train_engine.cli.push_hf import register_push_hf_parser
from post_train_engine.cli.report import register_report_parser
from post_train_engine.cli.run import register_run_parser
from post_train_engine.cli.runpod import register_runpod_parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="pte")
    subparsers = parser.add_subparsers(dest="command", required=True)
    register_artifacts_parser(subparsers)
    register_diagnose_parser(subparsers)
    register_hillclimb_parser(subparsers)
    register_push_hf_parser(subparsers)
    register_report_parser(subparsers)
    register_run_parser(subparsers)
    register_runpod_parser(subparsers)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
