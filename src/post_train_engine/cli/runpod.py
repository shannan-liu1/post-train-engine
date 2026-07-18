"""RunPod remote execution planning CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from post_train_engine.env import EnvResolver, load_env_file
from post_train_engine.runpod import write_runpod_plan
from post_train_engine.runpod_attempt import (
    GRPO_DOWNSTREAM_RESERVE_SECONDS,
    RunPodAttemptRunner,
    SSHRunPodRemoteExecutor,
    load_runpod_attempt,
    prepare_runpod_attempt,
    settle_runpod_billing,
    verify_runpod_attempt_source,
)
from post_train_engine.runpod_control_plane import (
    RunPodControlPlane,
    RunPodProviderTransport,
    RunPodRESTTransport,
)
from post_train_engine.runpod_watchdog import (
    RUNPOD_API_KEY_ENV,
    launch_local_deletion_watchdog,
)


def register_runpod_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser("runpod")
    runpod_subparsers = parser.add_subparsers(dest="runpod_command", required=True)

    plan_parser = runpod_subparsers.add_parser("plan")
    plan_parser.add_argument("--config", required=True)
    plan_parser.add_argument("--out", required=True)
    plan_parser.add_argument("--command", required=True)
    plan_parser.add_argument("--remote-workdir", default="/workspace/post-train-engine")
    plan_parser.add_argument("--dry-run", action="store_true")
    plan_parser.set_defaults(func=cmd_runpod_plan)

    watchdog_parser = runpod_subparsers.add_parser("watchdog")
    watchdog_parser.add_argument("--journal", required=True)
    watchdog_parser.add_argument("--receipt", required=True)
    watchdog_parser.add_argument("--log", required=True)
    watchdog_parser.add_argument("--env", default=".env")
    watchdog_parser.add_argument("--no-env", action="store_true")
    watchdog_parser.set_defaults(func=cmd_runpod_watchdog)

    attempt_parser = runpod_subparsers.add_parser("attempt")
    attempt_subparsers = attempt_parser.add_subparsers(
        dest="runpod_attempt_command", required=True
    )

    prepare_parser = attempt_subparsers.add_parser("prepare")
    prepare_parser.add_argument("--plan", required=True)
    prepare_parser.add_argument("--attempt-dir", required=True)
    prepare_parser.add_argument("--repo-url", required=True)
    prepare_parser.add_argument("--commit-sha", required=True)
    prepare_parser.add_argument("--target-spend-usd", required=True, type=float)
    prepare_parser.add_argument("--settled-spend-usd", required=True, type=float)
    prepare_parser.add_argument("--reserve-usd", type=float, default=0.15)
    prepare_parser.add_argument("--max-runtime-seconds", type=int, default=1200)
    prepare_parser.add_argument(
        "--minimum-grpo-remaining-seconds",
        type=int,
        default=GRPO_DOWNSTREAM_RESERVE_SECONDS,
    )
    prepare_parser.set_defaults(func=cmd_runpod_attempt_prepare)

    execute_parser = attempt_subparsers.add_parser("execute")
    execute_parser.add_argument("--attempt", required=True)
    execute_parser.add_argument("--ssh-private-key", required=True)
    execute_parser.add_argument("--confirm-spend-cap-usd", required=True, type=float)
    execute_parser.add_argument("--env", default=".env")
    execute_parser.add_argument("--no-env", action="store_true")
    execute_parser.set_defaults(func=cmd_runpod_attempt_execute)

    settle_parser = attempt_subparsers.add_parser("settle")
    settle_parser.add_argument("--journal", required=True)
    settle_parser.add_argument("--pod-id", required=True)
    settle_parser.add_argument("--start-time", required=True)
    settle_parser.add_argument("--end-time")
    settle_parser.add_argument("--final", action="store_true")
    settle_parser.add_argument("--out", required=True)
    settle_parser.add_argument("--env", default=".env")
    settle_parser.add_argument("--no-env", action="store_true")
    settle_parser.set_defaults(func=cmd_runpod_attempt_settle)


def cmd_runpod_plan(args: argparse.Namespace) -> None:
    write_runpod_plan(
        config_path=args.config,
        out_path=args.out,
        image=None,
        gpu_type=None,
        command=args.command,
        remote_workdir=args.remote_workdir,
        dry_run=args.dry_run,
    )


def cmd_runpod_watchdog(args: argparse.Namespace) -> None:
    resolver = EnvResolver(load_env_file(None if args.no_env else args.env))
    receipt = launch_local_deletion_watchdog(
        journal_path=args.journal,
        receipt_path=args.receipt,
        log_path=args.log,
        api_key=resolver.require_unambiguous(RUNPOD_API_KEY_ENV, secret=True),
    )
    identity = (
        f"pod_id={receipt['pod_id']}"
        if receipt.get("pod_id")
        else f"pod_name={receipt['pod_name']}"
    )
    if receipt["state"] == "armed":
        print(
            f"armed local RunPod deletion watchdog pid={receipt['pid']} "
            f"{identity} deadline={receipt['hard_deadline_seconds']}s"
        )
    else:
        print(
            f"RunPod watchdog resolved expired target {identity} "
            f"state={receipt['state']}"
        )


def cmd_runpod_attempt_prepare(args: argparse.Namespace) -> None:
    spec = prepare_runpod_attempt(
        plan_path=args.plan,
        attempt_dir=args.attempt_dir,
        repo_url=args.repo_url,
        commit_sha=args.commit_sha,
        target_spend_usd=args.target_spend_usd,
        settled_spend_usd=args.settled_spend_usd,
        reserve_usd=args.reserve_usd,
        max_runtime_seconds=args.max_runtime_seconds,
        minimum_grpo_remaining_seconds=args.minimum_grpo_remaining_seconds,
    )
    print(Path(spec.attempt_dir) / "attempt.json")


def cmd_runpod_attempt_execute(args: argparse.Namespace) -> None:
    spec = load_runpod_attempt(args.attempt)
    if args.confirm_spend_cap_usd != spec.budget.target_spend_usd:
        raise ValueError("confirmed spend cap does not match the prepared attempt")
    verify_runpod_attempt_source(spec)
    resolver = EnvResolver(load_env_file(None if args.no_env else args.env))
    api_key = resolver.require_unambiguous(RUNPOD_API_KEY_ENV, secret=True)
    attempt_dir = Path(spec.attempt_dir)
    journal = attempt_dir / "runpod_operation.json"
    control = RunPodControlPlane(RunPodProviderTransport(api_key), journal)
    remote = SSHRunPodRemoteExecutor(
        control=control,
        ssh_private_key=args.ssh_private_key,
    )

    def arm_watchdog(_spec, _receipt):
        return launch_local_deletion_watchdog(
            journal_path=journal,
            receipt_path=attempt_dir / "watchdog.json",
            log_path=attempt_dir / "watchdog.log",
            api_key=api_key,
        )

    result = RunPodAttemptRunner(
        control=control,
        remote=remote,
        launch_watchdog=arm_watchdog,
    ).execute(spec)
    print(json.dumps(result.model_dump(mode="json"), sort_keys=True))


def cmd_runpod_attempt_settle(args: argparse.Namespace) -> None:
    if args.final and args.end_time is None:
        raise ValueError("final billing settlement requires --end-time")
    resolver = EnvResolver(load_env_file(None if args.no_env else args.env))
    api_key = resolver.require_unambiguous(RUNPOD_API_KEY_ENV, secret=True)
    control = RunPodControlPlane(
        RunPodRESTTransport(api_key),
        args.journal,
    )
    receipt = settle_runpod_billing(
        control=control,
        pod_id=args.pod_id,
        start_time=args.start_time,
        end_time=args.end_time,
        final=args.final,
        out=args.out,
    )
    print(json.dumps(receipt.model_dump(mode="json"), sort_keys=True))
