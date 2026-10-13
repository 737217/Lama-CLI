"""Личность процесса на macOS: путь, владелец, время старта и командная строка."""

from __future__ import annotations

import ctypes
import sys
from ctypes import c_char, c_int, c_int32, c_uint32, c_uint64
from dataclasses import dataclass

# Flavor для буфера proc_bsdinfo. Значение 2 — proc_taskallinfo, и ядро отвергает меньший буфер.
_PROC_PIDTBSDINFO = 3
_PROC_PIDPATHINFO_MAXSIZE = 4096
_CTL_KERN = 1
_KERN_PROCARGS2 = 49
_ESRCH = 3


@dataclass(frozen=True)
class DarwinProcessRead:
    """``gone`` — PID нет. ``unreadable`` — поля не собраны. ``ok`` — поля полные."""

    status: str
    pid: int = 0
    start_time: int = 0
    executable: str = ""
    uid: int = 0
    command: tuple[str, ...] = ()


class _ProcBsdInfo(ctypes.Structure):
    _fields_ = (
        ("pbi_flags", c_uint32),
        ("pbi_status", c_uint32),
        ("pbi_xstatus", c_uint32),
        ("pbi_pid", c_uint32),
        ("pbi_ppid", c_uint32),
        ("pbi_uid", c_uint32),
        ("pbi_gid", c_uint32),
        ("pbi_ruid", c_uint32),
        ("pbi_rgid", c_uint32),
        ("pbi_svuid", c_uint32),
        ("pbi_svgid", c_uint32),
        ("rfu_1", c_uint32),
        ("pbi_comm", c_char * 16),
        ("pbi_name", c_char * 32),
        ("pbi_nfiles", c_uint32),
        ("pbi_pgid", c_uint32),
        ("pbi_pjobc", c_uint32),
        ("e_tdev", c_uint32),
        ("e_tpgid", c_uint32),
        ("pbi_nice", c_int32),
        ("pbi_start_tvsec", c_uint64),
        ("pbi_start_tvusec", c_uint64),
    )


def parse_kern_procargs2(data: bytes) -> list[str] | None:
    """Разобрать блок KERN_PROCARGS2: число аргументов, путь, затем argv."""
    if len(data) < 4:
        return None
    argc = int.from_bytes(data[:4], sys.byteorder)
    if argc < 1 or argc > 4096:
        return None
    raw = data[4:]
    path_end = raw.find(b"\x00")
    if path_end < 0:
        return None
    cursor = path_end
    while cursor < len(raw) and raw[cursor] == 0:
        cursor += 1
    arguments: list[str] = []
    while len(arguments) < argc and cursor < len(raw):
        end = raw.find(b"\x00", cursor)
        if end < 0:
            break
        arguments.append(raw[cursor:end].decode("utf-8", errors="replace"))
        cursor = end + 1
    if len(arguments) < argc:
        return None
    return arguments


def read_darwin_process(pid: int) -> DarwinProcessRead:
    if pid <= 0:
        return DarwinProcessRead("gone")
    if sys.platform != "darwin":
        return DarwinProcessRead("unreadable")
    try:
        return _read_open(pid)
    except (OSError, ValueError):
        return DarwinProcessRead("unreadable")


def _read_open(pid: int) -> DarwinProcessRead:
    libc = ctypes.CDLL("libc.dylib", use_errno=True)
    libproc = ctypes.CDLL("libproc.dylib", use_errno=True)
    info = _ProcBsdInfo()
    libproc.proc_pidinfo.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint64,
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    libproc.proc_pidinfo.restype = ctypes.c_int
    written = libproc.proc_pidinfo(pid, _PROC_PIDTBSDINFO, 0, ctypes.byref(info), ctypes.sizeof(info))
    if written <= 0:
        if ctypes.get_errno() == _ESRCH:
            return DarwinProcessRead("gone")
        return DarwinProcessRead("unreadable")
    executable = _image_path(libproc, pid)
    command = _command_line(libc, pid)
    if not executable or int(info.pbi_start_tvsec) <= 0 or command is None:
        return DarwinProcessRead("unreadable")
    return DarwinProcessRead(
        "ok",
        pid=pid,
        start_time=int(info.pbi_start_tvsec),
        executable=executable,
        uid=int(info.pbi_uid),
        command=tuple(command),
    )


def _image_path(libproc: ctypes.CDLL, pid: int) -> str:
    libproc.proc_pidpath.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
    libproc.proc_pidpath.restype = ctypes.c_int
    buffer = ctypes.create_string_buffer(_PROC_PIDPATHINFO_MAXSIZE)
    written = libproc.proc_pidpath(pid, buffer, _PROC_PIDPATHINFO_MAXSIZE)
    if written <= 0:
        return ""
    return buffer.value.decode("utf-8", errors="replace")


def _command_line(libc: ctypes.CDLL, pid: int) -> list[str] | None:
    libc.sysctl.argtypes = [
        ctypes.POINTER(c_int),
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    libc.sysctl.restype = ctypes.c_int
    mib = (c_int * 3)(_CTL_KERN, _KERN_PROCARGS2, pid)
    size = ctypes.c_size_t(0)
    if libc.sysctl(mib, 3, None, ctypes.byref(size), None, 0) != 0 or size.value == 0:
        return None
    buffer = ctypes.create_string_buffer(size.value)
    if libc.sysctl(mib, 3, buffer, ctypes.byref(size), None, 0) != 0:
        return None
    return parse_kern_procargs2(buffer.raw[: size.value])
