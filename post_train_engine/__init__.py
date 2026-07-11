"""Source-tree import shim for ``python -m post_train_engine.cli``.

The installable package lives under ``src/post_train_engine``. This shim makes
the user-facing ``python -m post_train_engine.cli`` command work from an
uninstalled checkout while preserving the real package exports.
"""

from __future__ import annotations

from pathlib import Path

_SRC_PACKAGE = Path(__file__).resolve().parent.parent / "src" / "post_train_engine"
if not _SRC_PACKAGE.is_dir():
    raise ImportError(f"cannot find source package: {_SRC_PACKAGE}")

__path__.append(str(_SRC_PACKAGE))  # type: ignore[name-defined]

_SRC_INIT = _SRC_PACKAGE / "__init__.py"
exec(compile(_SRC_INIT.read_text(encoding="utf-8"), str(_SRC_INIT), "exec"), globals())
