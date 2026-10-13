"""Строка PATH в профиле оболочки. Не вызывается, пока пользователь не подтвердил добавление."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from lama.errors import LamaError
from lama.i18n import t
from lama.platform.pathfix import append_export, merge_path, profile_file


def persist_user_path(directory: Path, *, home: Path | None = None, shell: str | None = None) -> bool:
    selected_home = Path.home() if home is None else home
    selected_shell = os.environ.get("SHELL", "") if shell is None else shell
    config_home = _config_home()
    path = profile_file(selected_home, selected_shell, config_home=config_home)
    text = _read_profile(path)
    updated = append_export(text, str(directory), shell=selected_shell)
    changed = False
    if updated is not None:
        _replace_profile(path, updated)
        changed = True
    merged = merge_path(os.environ.get("PATH", ""), str(directory))
    if merged is not None:
        os.environ["PATH"] = merged
    return changed


def _config_home() -> Path | None:
    raw = os.environ.get("XDG_CONFIG_HOME", "").strip()
    if not raw:
        return None
    return Path(raw)


def _read_profile(path: Path) -> str:
    if not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise LamaError(t("Could not update the shell profile {path}: {error}", path=path, error=exc)) from exc
    except OSError as exc:
        raise LamaError(t("Could not update the shell profile {path}: {error}", path=path, error=exc)) from exc


def _replace_profile(path: Path, text: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        )
        temporary = Path(handle.name)
        try:
            with handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except (OSError, UnicodeError):
            temporary.unlink(missing_ok=True)
            raise
    except (OSError, UnicodeError) as exc:
        raise LamaError(t("Could not update the shell profile {path}: {error}", path=path, error=exc)) from exc
