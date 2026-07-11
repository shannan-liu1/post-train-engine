"""Task plugin registry."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from post_train_engine.tasks.schema import Example


@dataclass(frozen=True)
class TaskSpec:
    """A task plugin that exposes normalized train/eval examples."""

    name: str
    load_train: Callable[[], list[Example]]
    load_eval: Callable[[], list[Example]]
    description: str = ""
    default_metric: str = "accuracy"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("task name must be non-empty")


class TaskRegistry:
    """In-memory registry for task plugins."""

    def __init__(self) -> None:
        self._tasks: dict[str, TaskSpec] = {}

    def register(self, task: TaskSpec) -> None:
        if task.name in self._tasks:
            raise ValueError(f"task already registered: {task.name}")
        self._tasks[task.name] = task

    def get(self, name: str) -> TaskSpec:
        try:
            return self._tasks[name]
        except KeyError as exc:
            raise KeyError(f"unknown task: {name}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tasks))
