from __future__ import annotations

import pytest

from post_train_engine.hub_identity import resolve_huggingface_revision


class FakeInfo:
    sha = "a" * 40


class FakeApi:
    def model_info(self, repo_id: str, *, revision: str):
        assert repo_id == "org/model"
        assert revision == "main"
        return FakeInfo()

    def dataset_info(self, repo_id: str, *, revision: str):
        assert repo_id == "org/data"
        assert revision == "v1"
        return FakeInfo()


def test_hub_revision_resolves_model_and_dataset_to_commit() -> None:
    api = FakeApi()
    assert resolve_huggingface_revision(
        "org/model", kind="model", requested_revision="main", api=api
    ) == "a" * 40
    assert resolve_huggingface_revision(
        "org/data", kind="dataset", requested_revision="v1", api=api
    ) == "a" * 40


def test_hub_revision_fails_closed_without_commit_sha() -> None:
    class BadApi(FakeApi):
        def model_info(self, repo_id: str, *, revision: str):
            return type("Info", (), {"sha": None})()

    with pytest.raises(ValueError, match="immutable commit SHA"):
        resolve_huggingface_revision(
            "org/model", kind="model", requested_revision="main", api=BadApi()
        )
