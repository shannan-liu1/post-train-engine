from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from post_train_engine import runpod_preflight


def test_preflight_records_timeout_and_stops_before_next_gate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[list[str]] = []

    def time_out(command, **kwargs):
        calls.append(command)
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(runpod_preflight.subprocess, "run", time_out)
    out = tmp_path / "preflight.json"

    report = runpod_preflight.run_preflight(
        out=out,
        checks=[("first", ["first"]), ("second", ["second"])],
        command_timeout_sec=10,
        total_timeout_sec=20,
        cuda_required=False,
    )

    assert report["ok"] is False
    assert report["results"][0]["timed_out"] is True
    assert [result["name"] for result in report["results"]] == ["first"]
    assert calls == [["first"]]
    assert json.loads(out.read_text(encoding="utf-8")) == report


def test_preflight_never_gives_a_command_more_than_aggregate_time_remaining(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monotonic_values = iter([100.0, 100.75])
    observed_timeout: list[float] = []

    monkeypatch.setattr(runpod_preflight.time, "monotonic", lambda: next(monotonic_values))

    def succeed(command, **kwargs):
        observed_timeout.append(kwargs["timeout"])
        return subprocess.CompletedProcess(command, 0, "ok", "")

    monkeypatch.setattr(runpod_preflight.subprocess, "run", succeed)

    report = runpod_preflight.run_preflight(
        out=tmp_path / "preflight.json",
        checks=[("only", ["only"])],
        command_timeout_sec=10,
        total_timeout_sec=1,
        cuda_required=False,
    )

    assert report["ok"] is True
    assert observed_timeout[0] <= 0.25


def test_preflight_verifies_normalized_lock_hash_before_remote_checks(
    tmp_path: Path,
) -> None:
    lock_text = "version = 1\r\npackage = []\r\n"
    normalized = lock_text.replace("\r\n", "\n").encode("utf-8")
    digest = hashlib.sha256(normalized).hexdigest()
    (tmp_path / "uv.lock").write_text(lock_text, encoding="utf-8", newline="")
    requirements = tmp_path / "requirements"
    requirements.mkdir()
    (requirements / "runpod.txt").write_text(
        f"# uv-lock-sha256: {digest}\naccelerate==1.0\n",
        encoding="utf-8",
    )

    runpod_preflight.verify_runpod_constraints(tmp_path)

    assert runpod_preflight.command_specs(require_cuda=True)[0][0] == "dependency_lock"

    (requirements / "runpod.txt").write_text(
        f"# uv-lock-sha256: {digest}\ntorch>=2\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="image-provided Torch"):
        runpod_preflight.verify_runpod_constraints(tmp_path)


def test_preflight_records_missing_executable_as_a_failed_gate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def missing(_command, **_kwargs):
        raise FileNotFoundError("missing-command")

    monkeypatch.setattr(runpod_preflight.subprocess, "run", missing)

    report = runpod_preflight.run_preflight(
        out=tmp_path / "preflight.json",
        checks=[("missing", ["missing"])],
        command_timeout_sec=10,
        total_timeout_sec=20,
        cuda_required=False,
    )

    assert report["ok"] is False
    assert report["results"][0]["error_type"] == "FileNotFoundError"


def test_constraints_can_run_before_project_dependencies_are_installed(
    tmp_path: Path,
) -> None:
    lock_text = "version = 1\npackage = []\n"
    digest = hashlib.sha256(lock_text.encode("utf-8")).hexdigest()
    (tmp_path / "uv.lock").write_text(lock_text, encoding="utf-8")
    requirements = tmp_path / "requirements"
    requirements.mkdir()
    (requirements / "runpod.txt").write_text(
        f"# uv-lock-sha256: {digest}\naccelerate==1.0\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-S",
            str(Path(runpod_preflight.__file__)),
            "--constraints-only",
            "--root",
            str(tmp_path),
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "RunPod constraints OK"
