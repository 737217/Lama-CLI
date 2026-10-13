"""Где лежит команда ollama: PATH, затем известные каталоги этой ОС."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from lama.platform.model import PlatformKind, assert_never


@dataclass(frozen=True)
class BinaryLocation:
    path: Path
    on_path: bool


def candidate_paths(kind: PlatformKind, environ: Mapping[str, str]) -> tuple[Path, ...]:
    if kind == "windows":
        local = environ.get("LOCALAPPDATA", "")
        if not local:
            return ()
        return (Path(local) / "Programs" / "Ollama" / "ollama.exe",)
    unix = (
        Path("/usr/local/bin/ollama"),
        Path("/usr/bin/ollama"),
    )
    if kind == "linux":
        return unix
    if kind == "darwin":
        return (
            *unix,
            Path("/opt/homebrew/bin/ollama"),
            Path("/Applications/Ollama.app/Contents/Resources/ollama"),
        )
    return assert_never(kind)


def locate(
    kind: PlatformKind,
    *,
    environ: Mapping[str, str],
    which: Callable[[str], str | None],
    is_file: Callable[[Path], bool],
) -> BinaryLocation | None:
    found = which("ollama")
    if found:
        return BinaryLocation(Path(found), True)
    path = environ.get("PATH", "")
    for candidate in candidate_paths(kind, environ):
        if not is_file(candidate):
            continue
        return BinaryLocation(candidate, _directory_on_path(candidate.parent, path))
    return None


def directory_on_path(directory: Path, path: str) -> bool:
    return _directory_on_path(directory, path)


def _directory_on_path(directory: Path, path: str) -> bool:
    target = os.path.normcase(os.path.normpath(str(directory)))
    for part in path.split(os.pathsep):
        if not part:
            continue
        if os.path.normcase(os.path.normpath(part)) == target:
            return True
    return False
