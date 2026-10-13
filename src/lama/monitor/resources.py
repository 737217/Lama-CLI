"""Снимок CPU, RAM, NVIDIA GPU и загруженных моделей. Недоступные метрики не подменяются."""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from lama.errors import LamaError
from lama.i18n import t
from lama.monitor.windows_metrics import read_windows_cpu, read_windows_memory
from lama.ollama.client import OllamaClient
from lama.ollama.types import RunningModel


@dataclass(frozen=True)
class CpuTimes:
    busy: int
    total: int


@dataclass(frozen=True)
class CpuInfo:
    percent: float | None
    load_1: float | None
    load_5: float | None
    load_15: float | None


@dataclass(frozen=True)
class MemoryInfo:
    total: int
    available: int
    used: int


@dataclass(frozen=True)
class GpuInfo:
    name: str
    memory_total: int
    memory_used: int
    memory_free: int
    utilization: int | None = None
    memory_utilization: int | None = None


@dataclass
class Snapshot:
    cpu: CpuInfo | None = None
    cpu_note: str | None = None
    cpu_times: CpuTimes | None = None
    memory: MemoryInfo | None = None
    memory_note: str | None = None
    gpus: list[GpuInfo] = field(default_factory=list)
    gpu_note: str | None = None
    running: list[RunningModel] = field(default_factory=list)
    running_note: str | None = None


def parse_meminfo(text: str) -> MemoryInfo:
    values: dict[str, int] = {}
    for line in text.splitlines():
        key, separator, rest = line.partition(":")
        if not separator:
            continue
        number = rest.strip().split()[0]
        values[key] = int(number) * 1024
    if "MemTotal" not in values or "MemAvailable" not in values:
        raise ValueError(t("/proc/meminfo has no MemTotal or MemAvailable."))
    total = values["MemTotal"]
    available = values["MemAvailable"]
    return MemoryInfo(total=total, available=available, used=max(total - available, 0))


def _stat_field(values: list[int], index: int) -> int:
    return values[index] if index < len(values) else 0


def parse_proc_stat(text: str) -> CpuTimes:
    for line in text.splitlines():
        if not line.startswith("cpu "):
            continue
        parts = line.split()[1:]
        if len(parts) < 4:
            break
        values = [int(part) for part in parts]
        guest = _stat_field(values, 8) + _stat_field(values, 9)
        busy = (
            _stat_field(values, 0)
            + _stat_field(values, 1)
            + _stat_field(values, 2)
            + _stat_field(values, 5)
            + _stat_field(values, 6)
            + _stat_field(values, 7)
            - guest
        )
        total = busy + _stat_field(values, 3) + _stat_field(values, 4)
        return CpuTimes(busy=max(busy, 0), total=max(total, 0))
    raise ValueError(t("/proc/stat has no cpu line."))


def cpu_percent(previous: CpuTimes, current: CpuTimes) -> float | None:
    busy = current.busy - previous.busy
    total = current.total - previous.total
    if total <= 0:
        return None
    return max(0.0, min(100.0, 100.0 * busy / total))


def parse_loadavg(text: str) -> tuple[float, float, float]:
    parts = text.split()
    if len(parts) < 3:
        raise ValueError(t("/proc/loadavg does not contain three averages."))
    return float(parts[0]), float(parts[1]), float(parts[2])


def _optional_percent(text: str) -> int | None:
    try:
        value = float(text)
    except ValueError:
        return None
    return max(0, min(100, round(value)))


def parse_nvidia_csv(text: str) -> list[GpuInfo]:
    gpus: list[GpuInfo] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            raise ValueError(t("Unexpected nvidia-smi line: {line}", line=line))
        name, total, used, free = parts[0], parts[1], parts[2], parts[3]
        utilization = _optional_percent(parts[4]) if len(parts) > 4 else None
        memory_utilization = _optional_percent(parts[5]) if len(parts) > 5 else None
        gpus.append(
            GpuInfo(
                name=name,
                memory_total=int(float(total)) * 1024 * 1024,
                memory_used=int(float(used)) * 1024 * 1024,
                memory_free=int(float(free)) * 1024 * 1024,
                utilization=utilization,
                memory_utilization=memory_utilization,
            )
        )
    return gpus


def query_nvidia(*, which: Callable[[str], str | None] | None = None) -> tuple[list[GpuInfo], str | None]:
    finder = which or shutil.which
    if finder("nvidia-smi") is None:
        return [], t("nvidia-smi was not found. NVIDIA metrics are unavailable.")
    argv = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(argv, check=False, capture_output=True, text=True)
    except FileNotFoundError:
        return [], t("nvidia-smi was not found. NVIDIA metrics are unavailable.")
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip().splitlines()
        reason = detail[0] if detail else t("exit code {code}", code=completed.returncode)
        return [], t("nvidia-smi returned no data: {reason}", reason=reason)
    try:
        return parse_nvidia_csv(completed.stdout), None
    except ValueError:
        return [], t("nvidia-smi returned an unexpected format.")


def read_cpu(
    stat_path: Path,
    loadavg_path: Path,
    previous: CpuTimes | None = None,
) -> tuple[CpuInfo | None, CpuTimes | None, str | None]:
    times: CpuTimes | None = None
    stat_error: str | None = None
    try:
        times = parse_proc_stat(stat_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        stat_error = str(exc)
    loads: tuple[float, float, float] | None = None
    try:
        loads = parse_loadavg(loadavg_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        loads = None
    if times is None and loads is None:
        detail = stat_error or t("no /proc/stat or /proc/loadavg")
        return None, None, t("CPU load is unavailable: {detail}", detail=detail)
    percent = cpu_percent(previous, times) if previous is not None and times is not None else None
    return (
        CpuInfo(
            percent=percent,
            load_1=None if loads is None else loads[0],
            load_5=None if loads is None else loads[1],
            load_15=None if loads is None else loads[2],
        ),
        times,
        None,
    )


_LINUX_MEMINFO = Path("/proc/meminfo")
_LINUX_STAT = Path("/proc/stat")


def _windows_counters(meminfo_path: Path, stat_path: Path) -> bool:
    return sys.platform == "win32" and meminfo_path == _LINUX_MEMINFO and stat_path == _LINUX_STAT


def collect_snapshot(
    client: OllamaClient,
    *,
    meminfo_path: Path = Path("/proc/meminfo"),
    stat_path: Path = Path("/proc/stat"),
    loadavg_path: Path = Path("/proc/loadavg"),
    previous_cpu: CpuTimes | None = None,
    settle_seconds: float = 0,
    nvidia: Callable[[], tuple[list[GpuInfo], str | None]] | None = None,
) -> Snapshot:
    snapshot = Snapshot()
    if _windows_counters(meminfo_path, stat_path):
        _fill_windows_host(snapshot, previous_cpu=previous_cpu, settle_seconds=settle_seconds)
    else:
        if settle_seconds > 0 and previous_cpu is None:
            _, baseline, _ = read_cpu(stat_path, loadavg_path)
            if baseline is not None:
                time.sleep(settle_seconds)
                previous_cpu = baseline
        snapshot.cpu, snapshot.cpu_times, snapshot.cpu_note = read_cpu(stat_path, loadavg_path, previous_cpu)
        try:
            snapshot.memory = parse_meminfo(meminfo_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            snapshot.memory_note = t("Memory is unavailable: {error}", error=exc)
    gpu_query = nvidia or query_nvidia
    snapshot.gpus, snapshot.gpu_note = gpu_query()
    try:
        snapshot.running = client.list_running()
    except LamaError as exc:
        snapshot.running_note = str(exc)
    return snapshot


def _fill_windows_host(snapshot: Snapshot, *, previous_cpu: CpuTimes | None, settle_seconds: float) -> None:
    if settle_seconds > 0 and previous_cpu is None:
        baseline = read_windows_cpu()
        if baseline is not None:
            time.sleep(settle_seconds)
            previous_cpu = CpuTimes(busy=baseline[0], total=baseline[1])
    current = read_windows_cpu()
    if current is None:
        snapshot.cpu_note = t("CPU load is unavailable: {detail}", detail=t("Windows CPU counters did not respond."))
    else:
        times = CpuTimes(busy=current[0], total=current[1])
        percent = cpu_percent(previous_cpu, times) if previous_cpu is not None else None
        snapshot.cpu = CpuInfo(percent=percent, load_1=None, load_5=None, load_15=None)
        snapshot.cpu_times = times
    try:
        total, available = read_windows_memory()
    except OSError as exc:
        snapshot.memory_note = t("Memory is unavailable: {error}", error=exc)
        return
    snapshot.memory = MemoryInfo(total=total, available=available, used=max(total - available, 0))
