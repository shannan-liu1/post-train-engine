from __future__ import annotations

from pathlib import Path

import pytest

from post_train_engine.training.checkpoint import apply_retention_policy


def test_retention_policy_ignores_incomplete_tmp_step_dirs(tmp_path) -> None:
    for name in [
        "step-00000001",
        "step-00000002",
        "step-00000003",
        "step-00000004.tmp",
    ]:
        (tmp_path / name).mkdir()

    apply_retention_policy(tmp_path, last_n=1, best_so_far=None)

    assert not (tmp_path / "step-00000001").exists()
    assert not (tmp_path / "step-00000002").exists()
    assert (tmp_path / "step-00000003").exists()
    assert (tmp_path / "step-00000004.tmp").exists()


def test_retention_policy_rejects_negative_last_n(tmp_path) -> None:
    with pytest.raises(ValueError, match="last_n must be non-negative"):
        apply_retention_policy(tmp_path, last_n=-1, best_so_far=None)


def test_retention_policy_refuses_linked_checkpoint_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    linked = tmp_path / "step-00000001"
    linked.mkdir()
    (tmp_path / "step-00000002").mkdir()
    monkeypatch.setattr(
        Path,
        "is_junction",
        lambda path: path == linked,
        raising=False,
    )

    with pytest.raises(ValueError, match="linked checkpoint directory"):
        apply_retention_policy(tmp_path, last_n=1, best_so_far=None)

    assert linked.is_dir()
