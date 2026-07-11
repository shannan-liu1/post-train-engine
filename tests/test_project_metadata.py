from __future__ import annotations

import tomllib
from pathlib import Path


def test_optimizer_extras_use_muon_distribution_name() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    optional = project["project"]["optional-dependencies"]

    for extra in ("optimizer", "gpu"):
        assert (
            "muon-optimizer @ git+https://github.com/KellerJordan/Muon"
            in optional[extra]
        )
        assert all(
            not dependency.startswith("muon @")
            for dependency in optional[extra]
        )
