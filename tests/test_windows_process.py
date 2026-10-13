"""Чтение процесса Windows. На других системах проверяется только устойчивый идентификатор владельца."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from lama.server.windows import read_windows_process, sid_owner_id, split_command_line

pytestmark_windows = pytest.mark.skipif(sys.platform != "win32", reason="нужен Windows API")


def test_sid_owner_id_is_stable_and_non_negative() -> None:
    first = sid_owner_id("S-1-5-21-1-2-3-1001")
    assert first == sid_owner_id("S-1-5-21-1-2-3-1001")
    assert first != sid_owner_id("S-1-5-21-1-2-3-1002")
    assert first >= 0


@pytestmark_windows
def test_split_command_line_keeps_quoted_path() -> None:
    assert split_command_line(r'"C:\Program Files\Ollama\ollama.exe" serve') == [
        r"C:\Program Files\Ollama\ollama.exe",
        "serve",
    ]


@pytestmark_windows
def test_read_windows_process_sees_this_interpreter() -> None:
    read = read_windows_process(os.getpid())
    assert read.status == "ok"
    assert read.pid == os.getpid()
    assert read.start_time > 0
    assert read.uid >= 0
    assert read.executable
    assert read.command
    assert Path(read.executable).name.lower() == Path(read.command[0]).name.lower()


@pytestmark_windows
def test_missing_windows_pid_is_gone() -> None:
    assert read_windows_process(0xFFFFFFFE).status == "gone"
