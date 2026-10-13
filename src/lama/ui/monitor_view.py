"""Автообновление мониторинга. Выход по q, Esc или Ctrl+C."""

from __future__ import annotations

import os
import sys
import time

from rich.live import Live

from lama.monitor.resources import collect_snapshot
from lama.ollama.client import OllamaClient
from lama.ui.display import console, render_snapshot

if sys.platform == "win32":
    import msvcrt
else:
    import select
    import termios
    import tty


def run_monitor(client: OllamaClient) -> None:
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        console.print(render_snapshot(collect_snapshot(client, settle_seconds=0.2)))
        return
    if sys.platform == "win32":
        _run_monitor_windows(client)
        return
    _run_monitor_posix(client)


def _run_monitor_posix(client: OllamaClient) -> None:
    fd = sys.stdin.fileno()
    previous = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        snapshot = collect_snapshot(client, settle_seconds=0.2)
        with Live(render_snapshot(snapshot, live=True), console=console, auto_refresh=False) as live:
            while True:
                live.update(render_snapshot(snapshot, live=True), refresh=True)
                ready, _, _ = select.select([fd], [], [], 2.0)
                if ready:
                    key = os.read(fd, 1)
                    if key in {b"q", b"Q", b"\x1b", b"\x03"}:
                        return
                snapshot = collect_snapshot(client, previous_cpu=snapshot.cpu_times)
    except KeyboardInterrupt:
        return
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, previous)


def _run_monitor_windows(client: OllamaClient) -> None:
    snapshot = collect_snapshot(client, settle_seconds=0.2)
    try:
        with Live(render_snapshot(snapshot, live=True), console=console, auto_refresh=False) as live:
            while True:
                live.update(render_snapshot(snapshot, live=True), refresh=True)
                if _windows_quit_key(2.0):
                    return
                snapshot = collect_snapshot(client, previous_cpu=snapshot.cpu_times)
    except KeyboardInterrupt:
        return


def _windows_quit_key(timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if msvcrt.kbhit():
            key = msvcrt.getwch()
            if key in {"q", "Q", "\x1b", "\x03"}:
                return True
            if key in {"\x00", "\xe0"}:
                msvcrt.getwch()
        time.sleep(0.05)
    return False
