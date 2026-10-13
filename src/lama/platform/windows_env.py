"""Пользовательский PATH Windows. Модуль импортируется только на Windows."""

from __future__ import annotations

import ctypes
import os
import winreg
from pathlib import Path

from lama.platform.pathfix import merge_path

_ENVIRONMENT = r"Environment"
_HWND_BROADCAST = 0xFFFF
_WM_SETTINGCHANGE = 0x001A
_SMTO_ABORTIFHUNG = 0x0002


def persist_user_path(directory: Path) -> bool:
    current = _read()
    merged = merge_path(current, str(directory))
    changed = False
    if merged is not None:
        _write(merged)
        _notify()
        changed = True
    _sync_process(directory)
    return changed


def _read() -> str:
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _ENVIRONMENT) as key:
            value, _kind = winreg.QueryValueEx(key, "Path")
    except OSError:
        return ""
    return value if isinstance(value, str) else ""


def _write(value: str) -> None:
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, _ENVIRONMENT) as key:
        winreg.SetValueEx(key, "Path", 0, winreg.REG_EXPAND_SZ, value)


def _notify() -> None:
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.SendMessageTimeoutW.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.c_wchar_p,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    user32.SendMessageTimeoutW.restype = ctypes.c_void_p
    result = ctypes.c_size_t()
    user32.SendMessageTimeoutW(
        _HWND_BROADCAST,
        _WM_SETTINGCHANGE,
        0,
        "Environment",
        _SMTO_ABORTIFHUNG,
        1000,
        ctypes.byref(result),
    )


def _sync_process(directory: Path) -> None:
    merged = merge_path(os.environ.get("PATH", ""), str(directory))
    if merged is not None:
        os.environ["PATH"] = merged
