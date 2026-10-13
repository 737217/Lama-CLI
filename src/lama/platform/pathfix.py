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
    return f"export PATH={_quote(directory)}:\"$PATH\""


def profile_line(shell: str, directory: str) -> str:
    """Строка, которую выбранная оболочка выполнит при следующем запуске."""
    if Path(shell).name in {"fish", "fish.exe"}:
        return f"fish_add_path --prepend --global {_quote(directory)}"
    return shell_export(directory)


def append_export(profile: str, directory: str, *, shell: str = "") -> str | None:
    """Добавить одну строку экспорта. None, если такая строка уже есть."""
    line = profile_line(shell, directory)
    if line in profile.splitlines():
        return None
    if profile == "":
        return line + "\n"
    if profile.endswith("\n"):
        return profile + line + "\n"
    return profile + "\n" + line + "\n"


def profile_file(home: Path, shell: str, *, config_home: Path | None = None) -> Path:
    name = Path(shell).name
    if name in {"fish", "fish.exe"}:
        base = home / ".config" if config_home is None else config_home
        return base / "fish" / "config.fish"
    if name in {"zsh", "zsh.exe"}:
        return home / ".zshrc"
    if name in {"bash", "bash.exe"}:
        return home / ".bashrc"
    return home / ".profile"


def _quote(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"


def _contains(path: str, directory: str) -> bool:
    target = os.path.normcase(os.path.normpath(directory))
    for part in path.split(os.pathsep):
        if part and os.path.normcase(os.path.normpath(part)) == target:
            return True
    return False
