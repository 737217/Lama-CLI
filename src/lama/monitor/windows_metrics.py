"""CPU и RAM Windows через kernel32. Модуль не вызывает API при импорте."""

from __future__ import annotations

import ctypes
from ctypes import wintypes


class _FileTime(ctypes.Structure):
    _fields_ = (
        ("dwLowDateTime", wintypes.DWORD),
        ("dwHighDateTime", wintypes.DWORD),
    )


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = (
        ("dwLength", wintypes.DWORD),
        ("dwMemoryLoad", wintypes.DWORD),
        ("ullTotalPhys", ctypes.c_uint64),
        ("ullAvailPhys", ctypes.c_uint64),
        ("ullTotalPageFile", ctypes.c_uint64),
        ("ullAvailPageFile", ctypes.c_uint64),
        ("ullTotalVirtual", ctypes.c_uint64),
        ("ullAvailVirtual", ctypes.c_uint64),
        ("ullAvailExtendedVirtual", ctypes.c_uint64),
    )


def cpu_times_from_counters(idle: int, kernel: int, user: int) -> tuple[int, int]:
    """Kernel time уже включает idle. Вернуть занятые и общие такты."""
    idle_ticks = min(max(idle, 0), max(kernel, 0))
    kernel_ticks = max(kernel, 0)
    user_ticks = max(user, 0)
    busy = (kernel_ticks - idle_ticks) + user_ticks
    total = kernel_ticks + user_ticks
    return busy, total


def _ticks(value: _FileTime) -> int:
    return (int(value.dwHighDateTime) << 32) | int(value.dwLowDateTime)


def read_windows_cpu() -> tuple[int, int] | None:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetSystemTimes.argtypes = [
        ctypes.POINTER(_FileTime),
        ctypes.POINTER(_FileTime),
        ctypes.POINTER(_FileTime),
    ]
    kernel32.GetSystemTimes.restype = wintypes.BOOL
    idle = _FileTime()
    kernel = _FileTime()
    user = _FileTime()
    if not kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
        return None
    return cpu_times_from_counters(_ticks(idle), _ticks(kernel), _ticks(user))


def read_windows_memory() -> tuple[int, int]:
    """Общая и доступная физическая память в байтах."""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(_MemoryStatusEx)]
    kernel32.GlobalMemoryStatusEx.restype = wintypes.BOOL
    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(status)
    if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError(ctypes.get_last_error())
    return int(status.ullTotalPhys), int(status.ullAvailPhys)
