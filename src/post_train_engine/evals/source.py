"""Evaluation source contract."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from post_train_engine.evals.grades import Grade
from post_train_engine.tasks.schema import Example


@dataclass(frozen=True)
class EvalSource:
    """A verifier-backed evaluation source."""

    name: str
    load_examples: Callable[[], list[Example]]
    extract_answer: Callable[[str], str | None]
    score: Callable[[str | None, Example], Grade]
    default_max_new_tokens: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("eval source name must be non-empty")
        if self.default_max_new_tokens <= 0:
            raise ValueError("default_max_new_tokens must be positive")

    def grade(self, generation: str, example: Example) -> Grade:
        return self.score(self.extract_answer(generation), example)


class EvalRegistry:
    """In-memory registry for eval sources."""

    def __init__(self) -> None:
        self._sources: dict[str, EvalSource] = {}

    def register(self, source: EvalSource) -> None:
        if source.name in self._sources:
            raise ValueError(f"eval source already registered: {source.name}")
        self._sources[source.name] = source

    def get(self, name: str) -> EvalSource:
        try:
            return self._sources[name]
        except KeyError as exc:
            raise KeyError(f"unknown eval source: {name}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._sources))
