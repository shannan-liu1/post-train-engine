"""Environment and secret resolution for API-first runs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from post_train_engine.api_schemas import ProviderSpec


def load_env_file(path: str | Path | None) -> dict[str, str]:
    """Load a simple dotenv file without mutating ``os.environ``."""

    if path is None:
        return {}
    env_path = Path(path)
    if not env_path.exists():
        return {}
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(env_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            raise ValueError(f"{env_path}:{line_number} must be KEY=VALUE")
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"{env_path}:{line_number} has an empty env key")
        values[key] = _unquote(value.strip())
    return values


class EnvResolver:
    """Resolve config values from process env plus an optional dotenv file."""

    def __init__(self, file_values: dict[str, str] | None = None) -> None:
        self.file_values = dict(file_values or {})

    def get(self, name: str | None) -> str | None:
        if not name:
            return None
        if name in os.environ:
            return os.environ[name]
        return self.file_values.get(name)

    def require(self, name: str, *, secret: bool) -> str:
        value = self.get(name)
        if value is None or value == "":
            kind = "secret env" if secret else "env"
            raise ValueError(f"missing required {kind}: {name}")
        return value

    def require_unambiguous(self, name: str, *, secret: bool) -> str:
        process_value = os.environ.get(name)
        file_value = self.file_values.get(name)
        if (
            process_value not in {None, ""}
            and file_value not in {None, ""}
            and process_value != file_value
        ):
            kind = "secret env" if secret else "env"
            raise ValueError(f"conflicting {kind} sources: {name}")
        return self.require(name, secret=secret)

    def redacted_provider_env(self, specs: list[ProviderSpec]) -> dict[str, Any]:
        env_names: set[str] = set()
        secret_names: set[str] = set()
        for spec in specs:
            for name in (spec.base_url_env, spec.model_env):
                if name:
                    env_names.add(name)
            if spec.api_key_env:
                env_names.add(spec.api_key_env)
                secret_names.add(spec.api_key_env)
        resolved: dict[str, Any] = {}
        for name in sorted(env_names):
            value = self.get(name)
            resolved[name] = {
                "present": value is not None and value != "",
                "secret": name in secret_names,
                "value": "[REDACTED]" if name in secret_names and value else value,
            }
        return resolved


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


__all__ = ["EnvResolver", "load_env_file"]
