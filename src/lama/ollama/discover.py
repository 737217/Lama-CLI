"""Поиск установленных команд ollama.

Один и тот же файл, открытый разными путями или симлинками, считается одной
установкой. Сначала идут каталоги PATH слева направо, затем домашние и
обычные каталоги установки, затем уже запущенные ``ollama serve`` и пути,
которые lama уже видела.
"""

from __future__ import annotations

import ipaddress
import os
import re
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from lama.config import state_dir

_VERSION_RE = re.compile(r"(\d+\.\d+\.\d+\S*)")
_VERSION_TIMEOUT = 3.0


_DEFAULT_LISTEN = "127.0.0.1:11434"
_EXECSTART_PATH = re.compile(r"\bpath=(\S+)")


@dataclass(frozen=True)
class OllamaInstall:
    path: str
    version: str | None
    listen: str = _DEFAULT_LISTEN


def discover_ollama_installs(
    *,
    path: str | None = None,
    home: Path | None = None,
    directories: Sequence[Path] | None = None,
    version_of: Callable[[str], str | None] | None = None,
    proc_root: Path | None = None,
    remembered: Sequence[str] | None = None,
    unit_records: Sequence[tuple[str, str]] | None = None,
) -> list[OllamaInstall]:
    """Вернуть уникальные исполняемые файлы ollama в порядке обнаружения.

    Адрес берётся из живого процесса, иначе из службы systemd, иначе из
    запомненного запуска. Если ничего этого нет, остаётся ``127.0.0.1:11434``.
    """
    reader = version_of or read_ollama_version
    search_dirs = _path_directories(os.environ.get("PATH", "") if path is None else path)
    if directories is None:
        search_dirs.extend(_default_directories(Path.home() if home is None else home))
    else:
        search_dirs.extend(directories)
    listens: dict[str, str] = {}
    if remembered is None:
        remembered_pairs = _remembered_pairs()
    else:
        remembered_pairs = [(item, None) for item in remembered]
    for executable, listen in remembered_pairs:
        _note_listen(listens, executable, listen)
    records = lookup_unit_listens() if unit_records is None else unit_records
    for executable, listen in records:
        _note_listen(listens, executable, listen)
    root = Path("/proc") if proc_root is None else proc_root
    running = running_serve_listens(root)
    for executable, listen in running:
        _note_listen(listens, executable, listen)
    installs: list[OllamaInstall] = []
    seen: set[str] = set()
    for directory in search_dirs:
        _consider(directory / "ollama", installs, seen, reader, listens)
    for executable, _listen in records:
        _consider(Path(executable), installs, seen, reader, listens)
    for executable, _listen in running:
        _consider(Path(executable), installs, seen, reader, listens)
    for executable, _listen in remembered_pairs:
        _consider(Path(executable), installs, seen, reader, listens)
    return installs


def read_ollama_version(binary: str) -> str | None:
    """Прочитать `ollama --version`, не запуская сервер."""
    try:
        completed = subprocess.run(
            [binary, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=_VERSION_TIMEOUT,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = (completed.stdout or completed.stderr).strip()
    if not text:
        return None
    for line in text.splitlines():
        match = _VERSION_RE.search(line)
        if match:
            return match.group(1)
    first = text.splitlines()[0].strip()
    return first or None


def remembered_executables() -> list[str]:
    """Пути ``ollama serve``, которые lama уже наблюдала."""
    return [path for path, _listen in _remembered_pairs()]


def remember_executable(path: str, listen: str | None = None) -> None:
    """Запомнить файл и его адрес, чтобы выбор остался после остановки процесса."""
    try:
        resolved = os.path.realpath(path)
    except OSError:
        return
    if not _is_executable(Path(resolved)):
        return
    pairs = _remembered_pairs()
    lines: list[str] = []
    updated = False
    for existing, host in pairs:
        try:
            current = os.path.realpath(existing)
        except OSError:
            current = existing
        if current == resolved:
            lines.append(_memory_line(resolved, listen or host))
            updated = True
        else:
            lines.append(_memory_line(existing, host))
    if not updated:
        lines.append(_memory_line(resolved, listen))
    file = _memory_path()
    try:
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError:
        return


def lookup_unit_listens() -> list[tuple[str, str]]:
    """Адрес ``OLLAMA_HOST`` загруженных служб systemd ollama."""
    found: list[tuple[str, str]] = []
    for user in (False, True):
        argv = ["systemctl"]
        if user:
            argv.append("--user")
        argv.extend(["show", "ollama.service", "-p", "LoadState", "-p", "ExecStart", "-p", "Environment", "--no-pager"])
        try:
            completed = subprocess.run(argv, check=False, capture_output=True, text=True, timeout=_VERSION_TIMEOUT)
        except (OSError, subprocess.TimeoutExpired):
            continue
        text = completed.stdout
        if "LoadState=loaded" not in text:
            continue
        executable = _execstart_path(text)
        if not executable:
            continue
        found.append((executable, _environment_value(text, "OLLAMA_HOST") or _DEFAULT_LISTEN))
    return found


def running_serve_listens(proc_root: Path) -> list[tuple[str, str | None]]:
    """Файлы ``ollama serve`` и их ``OLLAMA_HOST``, если он есть в окружении процесса."""
    try:
        entries = list(proc_root.iterdir())
    except OSError:
        return []
    found: list[tuple[str, str | None]] = []
    for entry in entries:
        if not entry.name.isdigit():
            continue
        command = _read_cmdline(entry / "cmdline")
        if not _is_serve_command(command):
            continue
        try:
            executable = os.path.realpath(entry / "exe")
        except OSError:
            continue
        found.append((executable, _listen_from_environ(entry)))
    return found


def serve_listen_host(proc_root: Path, pid: int) -> str | None:
    """``OLLAMA_HOST`` процесса, если окружение можно прочитать."""
    return _listen_from_environ(proc_root / str(pid))


def running_serve_executables(proc_root: Path) -> list[str]:
    """Файлы процессов, у которых командная строка — ``ollama serve``."""
    return [executable for executable, _listen in running_serve_listens(proc_root)]


def client_base_for(listen: str) -> str:
    """Адрес, к которому можно подключиться. ``0.0.0.0`` и ``::`` становятся loopback."""
    from lama.validation import normalize_host

    base, _env = normalize_host(listen)
    parsed_host, port = _split_host_port(base)
    if parsed_host is None:
        return base
    try:
        address = ipaddress.ip_address(parsed_host)
    except ValueError:
        return base
    if not address.is_unspecified:
        return base
    loopback = "::1" if address.version == 6 else "127.0.0.1"
    if address.version == 6:
        return f"http://[{loopback}]:{port}"
    return f"http://{loopback}:{port}"


def server_env_for(listen: str) -> str:
    """Значение ``OLLAMA_HOST`` для запуска выбранной версии."""
    from lama.validation import normalize_host

    _base, env = normalize_host(listen)
    return env


def _consider(
    candidate: Path,
    installs: list[OllamaInstall],
    seen: set[str],
    reader: Callable[[str], str | None],
    listens: dict[str, str],
) -> None:
    try:
        resolved = os.path.realpath(candidate)
    except OSError:
        return
    if resolved in seen:
        return
    seen.add(resolved)
    if not _is_executable(Path(resolved)):
        return
    installs.append(
        OllamaInstall(path=resolved, version=reader(resolved), listen=listens.get(resolved, _DEFAULT_LISTEN))
    )


def _memory_path() -> Path:
    return state_dir() / "ollama-binaries.txt"


def _remembered_pairs() -> list[tuple[str, str | None]]:
    try:
        text = _memory_path().read_text(encoding="utf-8")
    except OSError:
        return []
    pairs: list[tuple[str, str | None]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        path, separator, host = stripped.partition("\t")
        pairs.append((path, host.strip() if separator and host.strip() else None))
    return pairs


def _memory_line(path: str, listen: str | None) -> str:
    if listen:
        return f"{path}\t{listen}"
    return path


def _note_listen(listens: dict[str, str], path: str, listen: str | None) -> None:
    if not listen:
        return
    try:
        resolved = os.path.realpath(path)
    except OSError:
        return
    listens[resolved] = listen


def _execstart_path(text: str) -> str | None:
    match = _EXECSTART_PATH.search(text)
    if match is None:
        return None
    raw = match.group(1).rstrip(";")
    if not raw or raw.startswith("["):
        return None
    return raw


def _environment_value(text: str, key: str) -> str | None:
    prefix = key + "="
    for line in text.splitlines():
        body = line[len("Environment=") :] if line.startswith("Environment=") else line
        for token in body.split():
            if token.startswith(prefix):
                return token[len(prefix) :].strip('"')
    return None


def _listen_from_environ(proc_dir: Path) -> str | None:
    try:
        raw = (proc_dir / "environ").read_bytes()
    except OSError:
        return None
    prefix = b"OLLAMA_HOST="
    for item in raw.split(b"\0"):
        if item.startswith(prefix):
            value = item[len(prefix) :].decode(errors="replace").strip()
            return value or None
    return None


def _split_host_port(base: str) -> tuple[str | None, int]:
    parsed = urlparse(base)
    return parsed.hostname, parsed.port or 11434


def _read_cmdline(path: Path) -> list[str] | None:
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    parts = [part.decode(errors="replace") for part in raw.split(b"\0") if part]
    return parts or None


def _is_serve_command(command: list[str] | None) -> bool:
    return bool(command) and len(command) >= 2 and Path(command[0]).name == "ollama" and command[1] == "serve"


def _path_directories(path: str) -> list[Path]:
    directories: list[Path] = []
    for entry in path.split(os.pathsep):
        if not entry:
            continue
        directories.append(Path(entry))
    return directories


def _default_directories(home: Path) -> list[Path]:
    return [
        home / ".local" / "bin",
        home / "bin",
        Path("/usr/local/bin"),
        Path("/usr/bin"),
        Path("/bin"),
        Path("/snap/bin"),
        Path("/opt/ollama/bin"),
        Path("/opt/ollama"),
    ]


def _is_executable(path: Path) -> bool:
    try:
        return path.is_file() and os.access(path, os.X_OK)
    except OSError:
        return False
