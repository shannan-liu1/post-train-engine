from __future__ import annotations

from pathlib import Path

import pytest
import torch

from post_train_engine.training.checkpoint import load_checkpoint, save_checkpoint


def _write_marker(path: str) -> None:
    Path(path).write_text("executed", encoding="utf-8")


class _UnsafePayload:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self):  # type: ignore[no-untyped-def]
        return (_write_marker, (str(self.marker),))


def test_checkpoint_load_rejects_pickle_code_execution(tmp_path: Path) -> None:
    marker = tmp_path / "executed.txt"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    torch.save(
        {
            "step": 1,
            "optimizer": _UnsafePayload(marker),
            "scheduler": {},
            "rng_states": {},
            "config": {},
        },
        checkpoint / "state.pt",
    )

    with pytest.raises(Exception, match="Weights only load failed"):
        load_checkpoint(checkpoint)

    assert not marker.exists()


def test_safe_checkpoint_round_trip_remains_loadable(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    loss = model(torch.ones(1, 2)).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()

    checkpoint = save_checkpoint(
        directory=tmp_path,
        step=3,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        rng_states={"torch": torch.get_rng_state()},
        config={"learning_rate": 0.1},
    )
    loaded = load_checkpoint(checkpoint)

    assert loaded.step == 3
    assert loaded.config == {"learning_rate": 0.1}
    assert torch.equal(loaded.rng_states["torch"], torch.get_rng_state())
    assert loaded.optimizer_state_dict["state"]
    assert loaded.scheduler_state_dict["last_epoch"] == 1
    assert set(loaded.model_state_dict) == {"bias", "weight"}
