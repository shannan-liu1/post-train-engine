"""RunPod remote execution planning CLI."""

from __future__ import annotations

import argparse

from post_train_engine.runpod import write_runpod_plan


def register_runpod_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser("runpod")
    runpod_subparsers = parser.add_subparsers(dest="runpod_command", required=True)

    plan_parser = runpod_subparsers.add_parser("plan")
    plan_parser.add_argument("--run")
    plan_parser.add_argument("--config")
    plan_parser.add_argument("--out")
    plan_parser.add_argument("--image")
    plan_parser.add_argument("--gpu-type")
    plan_parser.add_argument("--gpu-count", type=int)
    plan_parser.add_argument("--command", required=True)
    plan_parser.add_argument("--repo-root", default=".")
    plan_parser.add_argument("--remote-workdir", default="/workspace/post-train-engine")
    plan_parser.add_argument("--setup-command", action="append", default=[])
    plan_parser.add_argument("--env", action="append", default=[])
    plan_parser.add_argument("--secret-env", action="append", default=[])
    plan_parser.add_argument("--container-disk-gb", type=int)
    plan_parser.add_argument("--volume-gb", type=int)
    plan_parser.add_argument("--dry-run", action="store_true")
    plan_parser.set_defaults(func=cmd_runpod_plan)


def cmd_runpod_plan(args: argparse.Namespace) -> None:
    write_runpod_plan(
        run_dir=args.run,
        config_path=args.config,
        out_path=args.out,
        image=args.image,
        gpu_type=args.gpu_type,
        command=args.command,
        repo_root=args.repo_root,
        remote_workdir=args.remote_workdir,
        setup_commands=tuple(args.setup_command),
        env=tuple(args.env),
        secret_env=tuple(args.secret_env),
        gpu_count=args.gpu_count,
        container_disk_gb=args.container_disk_gb,
        volume_gb=args.volume_gb,
        dry_run=args.dry_run,
    )
