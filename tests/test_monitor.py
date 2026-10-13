import sys
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from lama.formatting import format_percent
from lama.monitor.resources import (
    CpuInfo,
    CpuTimes,
    GpuInfo,
    MemoryInfo,
    Snapshot,
    collect_snapshot,
    cpu_percent,
    parse_loadavg,
    parse_meminfo,
    parse_nvidia_csv,
    parse_proc_stat,
    query_nvidia,
)
from lama.monitor.windows_metrics import cpu_times_from_counters
from lama.ui.display import _console_size, render_snapshot


def test_parse_meminfo() -> None:
    info = parse_meminfo("MemTotal:       1000 kB\nMemAvailable:    400 kB\n")
    assert info.total == 1000 * 1024
    assert info.available == 400 * 1024
    assert info.used == 600 * 1024


def test_parse_nvidia_csv() -> None:
    gpus = parse_nvidia_csv("NVIDIA GeForce, 24564, 1024, 23540, 37\n")
    assert gpus[0].name == "NVIDIA GeForce"
    assert gpus[0].memory_total == 24564 * 1024 * 1024
    assert gpus[0].memory_free == 23540 * 1024 * 1024
    assert gpus[0].utilization == 37


def test_parse_nvidia_csv_without_utilization() -> None:
    gpus = parse_nvidia_csv("NVIDIA GeForce, 24564, 1024, 23540, [N/A]\n")
    assert gpus[0].utilization is None
    assert gpus[0].memory_utilization is None


def test_parse_nvidia_memory_controller() -> None:
    gpus = parse_nvidia_csv("NVIDIA GeForce RTX 5070, 12282, 3790, 8492, 0, 17\n")
    assert gpus[0].utilization == 0
    assert gpus[0].memory_utilization == 17


def test_windows_cpu_counters_treat_kernel_time_as_including_idle() -> None:
    assert cpu_times_from_counters(10, 40, 30) == (60, 70)


def test_console_size_follows_columns_only_when_stdout_is_not_a_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COLUMNS", "100")
    monkeypatch.delenv("LINES", raising=False)

    class _Stream:
        def __init__(self, tty: bool) -> None:
            self._tty = tty

        def isatty(self) -> bool:
            return self._tty

    monkeypatch.setattr(sys, "stdout", _Stream(True))
    assert _console_size() is None
    monkeypatch.setattr(sys, "stdout", _Stream(False))
    assert _console_size() == (100, 25)


def test_cpu_load_between_samples() -> None:
    previous = parse_proc_stat("cpu  10 0 10 80 0 0 0 0 0 0\n")
    current = parse_proc_stat("cpu  20 0 20 160 0 0 0 0 0 0\n")
    assert previous == CpuTimes(busy=20, total=100)
    assert cpu_percent(previous, current) == 20.0
    assert parse_loadavg("1.50 0.80 0.40 1/200 10\n") == (1.5, 0.8, 0.4)
    assert format_percent(20.4) == "20%"
    assert format_percent(None) == "—"


def test_nvidia_missing() -> None:
    gpus, note = query_nvidia(which=lambda _name: None)
    assert gpus == []
    assert note is not None
    assert "nvidia-smi" in note


def test_snapshot_includes_cpu_and_gpu_load(tmp_path: Path) -> None:
    stat = tmp_path / "stat"
    loadavg = tmp_path / "loadavg"
    stat.write_text("cpu  10 0 10 80 0 0 0 0 0 0\n", encoding="utf-8")
    loadavg.write_text("0.25 0.20 0.10 1/10 1\n", encoding="utf-8")

    class Idle:
        def list_running(self) -> list[object]:
            return []

    snapshot = collect_snapshot(
        Idle(),  # type: ignore[arg-type]
        meminfo_path=tmp_path / "missing-meminfo",
        stat_path=stat,
        loadavg_path=loadavg,
        previous_cpu=CpuTimes(busy=0, total=0),
        nvidia=lambda: (
            [GpuInfo(name="RTX", memory_total=1024, memory_used=256, memory_free=768, utilization=64)],
            None,
        ),
    )
    assert snapshot.cpu == CpuInfo(percent=20.0, load_1=0.25, load_5=0.2, load_15=0.1)
    assert snapshot.gpus[0].utilization == 64
    buffer = StringIO()
    Console(file=buffer, force_terminal=False, width=100, color_system=None).print(render_snapshot(snapshot))
    text = buffer.getvalue()
    assert "20%" in text
    assert "64%" in text
    assert "0.25" in text
    assert "Resource" in text
    assert "No models loaded" in text


def test_snapshot_without_gpu_and_without_api(tmp_path: Path) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal: 10 kB\nMemAvailable: 4 kB\n", encoding="utf-8")
    stat = tmp_path / "stat"
    stat.write_text("нет строки\n", encoding="utf-8")

    from lama.errors import OllamaUnavailable

    class Down:
        def list_running(self) -> list[object]:
            raise OllamaUnavailable("Сервер Ollama недоступен.")

    snapshot = collect_snapshot(
        Down(),  # type: ignore[arg-type]
        meminfo_path=meminfo,
        stat_path=stat,
        loadavg_path=tmp_path / "missing-loadavg",
        nvidia=lambda: ([], "nvidia-smi не найден. Показатели NVIDIA недоступны."),
    )
    assert snapshot.memory is not None
    assert snapshot.cpu is None
    assert snapshot.cpu_note is not None
    assert snapshot.gpus == []
    assert snapshot.gpu_note is not None
    assert snapshot.running == []
    assert snapshot.running_note is not None


def test_snapshot_view_shows_load_bars() -> None:
    gib = 1024**3
    snapshot = Snapshot(
        cpu=CpuInfo(percent=14, load_1=0.4, load_5=0.3, load_15=0.2),
        memory=MemoryInfo(total=30 * gib, available=20 * gib, used=10 * gib),
        gpus=[
            GpuInfo(
                name="RTX 5070",
                memory_total=12 * gib,
                memory_used=4 * gib,
                memory_free=8 * gib,
                utilization=0,
                memory_utilization=17,
            )
        ],
    )
    buffer = StringIO()
    Console(file=buffer, force_terminal=False, width=120, height=40, color_system=None).print(render_snapshot(snapshot))
    text = buffer.getvalue()
    assert "GPU 0: RTX 5070" in text
    assert "10.0 GB / 30.0 GB" in text
    assert "free 20.0 GB" in text
    assert "4.0 GB / 12.0 GB" in text
    assert "memory controller" in text
    assert "17%" in text
    assert "0%" in text
    assert "q or Esc" not in text
    columns = []
    for needle in ("14%", "10.0 GB", "0%", "4.0 GB"):
        line = next(item for item in text.splitlines() if needle in item)
        columns.append(line.index(needle))
    assert columns[0] == columns[1] == columns[2] == columns[3]

    live = StringIO()
    Console(file=live, force_terminal=False, width=120, color_system=None).print(render_snapshot(snapshot, live=True))
    assert "q or Esc" in live.getvalue()
