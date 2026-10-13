"""Склейка PATH и строка профиля. Не пишет в реестр и не открывает домашний каталог."""

from __future__ import annotations

import os
from pathlib import Path


def merge_path(path: str, directory: str) -> str | None:
    """Вернуть PATH с каталогом в начале. None, если каталог уже есть."""
    if _contains(path, directory):
        return None
    if path == "":
        return directory
    return directory + os.pathsep + path


def shell_export(directory: str) -> str:
    quoted = directory.replace("'", "'\\''")
    return f"export PATH='{quoted}':\"$PATH\""


def append_export(profile: str, directory: str) -> str | None:
    """Добавить одну строку экспорта. None, если такая строка уже есть."""
    line = shell_export(directory)
    if line in profile.splitlines():
        return None
    if profile == "":
        return line + "\n"
    if profile.endswith("\n"):
        return profile + line + "\n"
    return profile + "\n" + line + "\n"


def profile_file(home: Path, shell: str) -> Path:
    name = Path(shell).name
    if name in {"zsh", "zsh.exe"}:
        return home / ".zshrc"
    if name in {"bash", "bash.exe"}:
        return home / ".bashrc"
    return home / ".profile"


def _contains(path: str, directory: str) -> bool:
    target = os.path.normcase(os.path.normpath(directory))
    for part in path.split(os.pathsep):
        if part and os.path.normcase(os.path.normpath(part)) == target:
            return True
    return False
