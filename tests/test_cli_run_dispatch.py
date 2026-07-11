from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from post_train_engine.cli.main import main
from post_train_engine.engine import CANONICAL_STAGE_ORDER


def test_run_cli_reports_supported_runner_kinds(tmp_path: Path) -> None:
    config = tmp_path / "unknown.yaml"
    config.write_text(
        yaml.safe_dump({"kind": "unknown_runner"}, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="supported run kinds: gsm8k_local_smoke"):
        main(["run", "--config", str(config)])


def test_run_cli_accepts_api_hillclimb_config_through_canonical_dispatch(
    tmp_path: Path,
) -> None:
    raw = yaml.safe_load(Path("configs/gsm8k_tiny_dryrun.yaml").read_text(encoding="utf-8"))
    run_dir = tmp_path / "runs" / "api-through-run"
    raw["run"]["run_id"] = "api-through-run"
    raw["run"]["output_dir"] = str(run_dir)
    config = tmp_path / "api.yaml"
    config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    main(["run", "--config", str(config)])

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["metadata"]["stage_order"] == list(CANONICAL_STAGE_ORDER)
    assert manifest["metadata"]["execution_mode"] == "api_hillclimb"


def test_top_level_cli_has_no_independent_train_execution_command(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["train"])

    assert excinfo.value.code == 2
    assert "invalid choice: 'train'" in capsys.readouterr().err
