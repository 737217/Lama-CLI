"""Текущая ОС и действия, которые от неё зависят."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path

from lama.platform.locate import BinaryLocation, locate
from lama.platform.model import PlatformKind, assert_never

_PROC_ROOT = Path("/proc")


def current_kind() -> PlatformKind:
    if sys.platform == "win32":
        return "windows"
    if sys.platform == "darwin":
        return "darwin"
    return "linux"


def locate_ollama(
    *,
    kind: PlatformKind | None = None,
    environ: Mapping[str, str] | None = None,
    which: Callable[[str], str | None] | None = None,
    is_file: Callable[[Path], bool] | None = None,
) -> BinaryLocation | None:
    selected = current_kind() if kind is None else kind
    return locate(
        selected,
        environ=os.environ if environ is None else environ,
        which=shutil.which if which is None else which,
        is_file=Path.is_file if is_file is None else is_file,
    )


def spawn_options(kind: PlatformKind | None = None) -> dict[str, object]:
    selected = current_kind() if kind is None else kind
    if selected == "windows":
        creationflags = 0
        for name in ("CREATE_NEW_PROCESS_GROUP", "CREATE_NO_WINDOW"):
            value = getattr(subprocess, name, 0)
            if isinstance(value, int):
                creationflags |= value
        return {"creationflags": creationflags}
    if selected == "linux" or selected == "darwin":
        return {"start_new_session": True}
    return assert_never(selected)


def uses_native_process_table(proc_root: Path, *, kind: PlatformKind | None = None) -> bool:
    """Windows и macOS не читают /proc. Поддельный корень в тестах остаётся Linux-путём."""
    selected = current_kind() if kind is None else kind
    if proc_root != _PROC_ROOT:
        return False
    if selected == "windows" or selected == "darwin":
        return True
    if selected == "linux":
        return False
    return assert_never(selected)


def open_catalog(url: str) -> None:
    kind = current_kind()
    if kind == "windows":
        startfile = getattr(os, "startfile", None)
        if startfile is None:
            return
        try:
            startfile(url)
        except OSError:
            return
        return
    if kind == "darwin":
        command = "open"
    elif kind == "linux":
        command = "xdg-open"
    else:
        assert_never(kind)
        return
    if shutil.which(command) is None:
        return
    subprocess.run(
        [command, url],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
