"""Чтение процесса Ollama на Windows без /proc.

Идентичность для файла состояния: PID, время создания, полный путь образа
и владелец (хеш SID). Командная строка нужна, чтобы отличить ``ollama serve``
от другого запуска того же ``ollama.exe``. Чужой процесс по этим полям
не завершается.
"""

from __future__ import annotations

import ctypes
import hashlib
from ctypes import wintypes
from dataclasses import dataclass

_PROCESS_QUERY_INFORMATION = 0x0400
_PROCESS_VM_READ = 0x0010
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_PROCESS_ACCESS = _PROCESS_QUERY_INFORMATION | _PROCESS_VM_READ | _PROCESS_QUERY_LIMITED_INFORMATION
_ERROR_INVALID_PARAMETER = 87
_TOKEN_QUERY = 0x0008
_TOKEN_USER = 1
_PEB_PROCESS_PARAMETERS = 0x20
_PARAMETERS_COMMAND_LINE = 0x70
_MAX_COMMAND_BYTES = 65534


@dataclass(frozen=True)
class WindowsProcessRead:
    """``gone`` — PID нет. ``unreadable`` — процесс есть, но поля не собраны. ``ok`` — поля полные."""

    status: str
    pid: int = 0
    start_time: int = 0
    executable: str = ""
    uid: int = 0
    command: tuple[str, ...] = ()


class _FileTime(ctypes.Structure):
    _fields_ = (("dwLowDateTime", wintypes.DWORD), ("dwHighDateTime", wintypes.DWORD))


class _ProcessBasicInformation(ctypes.Structure):
    _fields_ = (
        ("Reserved1", ctypes.c_void_p),
        ("PebBaseAddress", ctypes.c_void_p),
        ("Reserved2", ctypes.c_void_p * 2),
        ("UniqueProcessId", ctypes.c_void_p),
        ("Reserved3", ctypes.c_void_p),
    )


class _SidAndAttributes(ctypes.Structure):
    _fields_ = (("Sid", ctypes.c_void_p), ("Attributes", wintypes.DWORD))


class _TokenUser(ctypes.Structure):
    _fields_ = (("User", _SidAndAttributes),)


def sid_owner_id(sid: str) -> int:
    """Устойчивый неотрицательный идентификатор владельца по строке SID."""
    digest = hashlib.sha256(sid.encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF


def split_command_line(command: str) -> list[str] | None:
    """Разобрать командную строку Windows так же, как это делает CreateProcess."""
    if command == "":
        return None
    shell32 = ctypes.WinDLL("shell32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    shell32.CommandLineToArgvW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_int)]
    shell32.CommandLineToArgvW.restype = ctypes.POINTER(wintypes.LPWSTR)
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p
    count = ctypes.c_int()
    argv = shell32.CommandLineToArgvW(command, ctypes.byref(count))
    if not argv:
        return None
    try:
        return [argv[index] for index in range(count.value)]
    finally:
        kernel32.LocalFree(argv)


def read_windows_process(pid: int) -> WindowsProcessRead:
    """Собрать поля процесса. Ошибка чтения не бросается наружу: сигнала без сверки не будет."""
    if pid <= 0 or pid > 0xFFFFFFFF:
        return WindowsProcessRead("gone")
    if ctypes.sizeof(ctypes.c_void_p) != 8:
        return WindowsProcessRead("unreadable")
    try:
        opened = _open_process(pid)
    except (OSError, ValueError, OverflowError):
        return WindowsProcessRead("unreadable")
    if isinstance(opened, str):
        return WindowsProcessRead(opened)
    kernel32 = _kernel32()
    try:
        executable = _image_path(kernel32, opened)
        start_time = _creation_time(kernel32, opened)
        uid = _owner_id(opened)
        command = _command_line(kernel32, opened)
    except (OSError, ValueError, OverflowError):
        return WindowsProcessRead("unreadable")
    finally:
        kernel32.CloseHandle(opened)
    if not executable or start_time <= 0 or uid is None or not command:
        return WindowsProcessRead("unreadable")
    return WindowsProcessRead(
        "ok",
        pid=pid,
        start_time=start_time,
        executable=executable,
        uid=uid,
        command=tuple(command),
    )


def _kernel32() -> ctypes.WinDLL:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    return kernel32


def _open_process(pid: int) -> ctypes.wintypes.HANDLE | str:
    kernel32 = _kernel32()
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    handle = kernel32.OpenProcess(_PROCESS_ACCESS, False, pid)
    if handle:
        return handle
    if ctypes.get_last_error() == _ERROR_INVALID_PARAMETER:
        return "gone"
    return "unreadable"


def _image_path(kernel32: ctypes.WinDLL, handle: ctypes.wintypes.HANDLE) -> str:
    kernel32.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD),
    ]
    kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
    length = wintypes.DWORD(32768)
    buffer = ctypes.create_unicode_buffer(length.value)
    if not kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(length)):
        return ""
    return buffer.value


def _creation_time(kernel32: ctypes.WinDLL, handle: ctypes.wintypes.HANDLE) -> int:
    kernel32.GetProcessTimes.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_FileTime),
        ctypes.POINTER(_FileTime),
        ctypes.POINTER(_FileTime),
        ctypes.POINTER(_FileTime),
    ]
    kernel32.GetProcessTimes.restype = wintypes.BOOL
    created = _FileTime()
    exited = _FileTime()
    kernel = _FileTime()
    user = _FileTime()
    if not kernel32.GetProcessTimes(
        handle,
        ctypes.byref(created),
        ctypes.byref(exited),
        ctypes.byref(kernel),
        ctypes.byref(user),
    ):
        return 0
    return (int(created.dwHighDateTime) << 32) | int(created.dwLowDateTime)


def _owner_id(handle: ctypes.wintypes.HANDLE) -> int | None:
    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32 = _kernel32()
    advapi32.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
    advapi32.OpenProcessToken.restype = wintypes.BOOL
    token = wintypes.HANDLE()
    if not advapi32.OpenProcessToken(handle, _TOKEN_QUERY, ctypes.byref(token)):
        return None
    try:
        return _token_user_id(advapi32, kernel32, token)
    finally:
        kernel32.CloseHandle(token)


def _token_user_id(advapi32: ctypes.WinDLL, kernel32: ctypes.WinDLL, token: ctypes.wintypes.HANDLE) -> int | None:
    advapi32.GetTokenInformation.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    advapi32.GetTokenInformation.restype = wintypes.BOOL
    needed = wintypes.DWORD()
    advapi32.GetTokenInformation(token, _TOKEN_USER, None, 0, ctypes.byref(needed))
    if needed.value == 0:
        return None
    buffer = ctypes.create_string_buffer(needed.value)
    if not advapi32.GetTokenInformation(token, _TOKEN_USER, buffer, needed.value, ctypes.byref(needed)):
        return None
    user = ctypes.cast(buffer, ctypes.POINTER(_TokenUser)).contents
    advapi32.ConvertSidToStringSidW.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_wchar_p)]
    advapi32.ConvertSidToStringSidW.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p
    sid_pointer = ctypes.c_wchar_p()
    if not advapi32.ConvertSidToStringSidW(user.User.Sid, ctypes.byref(sid_pointer)):
        return None
    try:
        sid = sid_pointer.value
    finally:
        kernel32.LocalFree(sid_pointer)
    if not sid:
        return None
    return sid_owner_id(sid)


def _command_line(kernel32: ctypes.WinDLL, handle: ctypes.wintypes.HANDLE) -> list[str] | None:
    ntdll = ctypes.WinDLL("ntdll", use_last_error=True)
    ntdll.NtQueryInformationProcess.argtypes = [
        wintypes.HANDLE,
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.POINTER(ctypes.c_ulong),
    ]
    ntdll.NtQueryInformationProcess.restype = ctypes.c_long
    info = _ProcessBasicInformation()
    status = ntdll.NtQueryInformationProcess(handle, 0, ctypes.byref(info), ctypes.sizeof(info), None)
    if status != 0 or not info.PebBaseAddress:
        return None
    parameters = _read_pointer(kernel32, handle, info.PebBaseAddress + _PEB_PROCESS_PARAMETERS)
    if parameters is None:
        return None
    raw = _read_memory(kernel32, handle, parameters + _PARAMETERS_COMMAND_LINE, 16)
    if raw is None:
        return None
    length = int.from_bytes(raw[0:2], "little")
    buffer = int.from_bytes(raw[8:16], "little")
    if length <= 0 or length > _MAX_COMMAND_BYTES or length % 2 != 0 or buffer <= 0:
        return None
    text = _read_memory(kernel32, handle, buffer, length)
    if text is None:
        return None
    return split_command_line(text.decode("utf-16-le", errors="replace"))


def _read_pointer(kernel32: ctypes.WinDLL, handle: ctypes.wintypes.HANDLE, address: int) -> int | None:
    raw = _read_memory(kernel32, handle, address, ctypes.sizeof(ctypes.c_void_p))
    if raw is None:
        return None
    value = int.from_bytes(raw, "little")
    if value <= 0:
        return None
    return value


def _read_memory(kernel32: ctypes.WinDLL, handle: ctypes.wintypes.HANDLE, address: int, size: int) -> bytes | None:
    if address <= 0 or size <= 0 or size > _MAX_COMMAND_BYTES:
        return None
    kernel32.ReadProcessMemory.argtypes = [
        wintypes.HANDLE,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    kernel32.ReadProcessMemory.restype = wintypes.BOOL
    buffer = ctypes.create_string_buffer(size)
    read = ctypes.c_size_t()
    ok = kernel32.ReadProcessMemory(handle, ctypes.c_void_p(address), buffer, size, ctypes.byref(read))
    if not ok or read.value != size:
        return None
    return buffer.raw
