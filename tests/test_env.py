from __future__ import annotations

import pytest

from post_train_engine.env import EnvResolver


def test_unambiguous_secret_rejects_conflicting_process_and_file_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PTE_REMOTE_RUNPOD_ALL", "process-account")
    resolver = EnvResolver({"PTE_REMOTE_RUNPOD_ALL": "file-account"})

    with pytest.raises(ValueError, match="conflicting secret env"):
        resolver.require_unambiguous("PTE_REMOTE_RUNPOD_ALL", secret=True)
