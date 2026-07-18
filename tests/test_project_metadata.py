from __future__ import annotations

import tomllib
from pathlib import Path


MUON_COMMIT = "f98f1cacc0263b04290753e32be8d498c1efc806"


def test_optimizer_extras_use_muon_distribution_name() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    optional = project["project"]["optional-dependencies"]

    for extra in ("optimizer", "gpu"):
        assert (
            "muon-optimizer @ "
            f"git+https://github.com/KellerJordan/Muon@{MUON_COMMIT}" in optional[extra]
        )
        assert all(
            not dependency.startswith("muon @") for dependency in optional[extra]
        )


def test_repository_commits_dependency_lock() -> None:
    assert Path("uv.lock").is_file()


def test_declared_python_floor_matches_runtime_syntax() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert project["project"]["requires-python"] == ">=3.11"


def test_ci_enforces_reproducible_quality_and_secret_gates() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "permissions:\n  contents: read" in workflow
    assert "actions/checkout@df4cb1c069e1874edd31b4311f1884172cec0e10" in workflow
    assert "astral-sh/setup-uv@08807647e7069bb48b6ef5acd8ec9567f424441b" in workflow
    assert (
        "gitleaks/gitleaks-action@ff98106e4c7b2bc287b24eaf42907196329070c7" in workflow
    )
    for command in (
        "uv lock --check",
        "uv sync --frozen --extra dev",
        "uv run --frozen ruff check .",
        "uv run --frozen pytest -q",
        "uv build",
    ):
        assert command in workflow
