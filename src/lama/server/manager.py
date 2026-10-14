"""Определение способа управления сервером Ollama и безопасный start/stop.

Классификация адреса
--------------------
Управление процессами и systemd относится только к этой машине, поэтому
сначала определяется область endpoint. DNS не используется: имя само по себе
не доказывает, что это локальный компьютер.

- Локальный: имя ``localhost`` (без других меток, точка в конце допустима),
  любой IPv4 из 127.0.0.0/8, IPv6 ``::1`` и IPv4-mapped loopback.
- Wildcard: ``0.0.0.0`` и ``::``. Это адреса «слушать всё», а не конкретный
  сервер, поэтому локальные start/stop/restart и локальные журналы недоступны.
- Удалённый: любой другой IP или hostname, включая частные и link-local адреса
  и имена, которые могли бы разрешиться в loopback. Для них доступны только
  проверка API, версия и сам адрес.

Выбор локального backend
-------------------------
Для локального endpoint опрашиваются обе области systemd и файл состояния
собственного процесса. Факты разделены: служба установлена, служба активна
(``ActiveState=active``), API отвечает, этот backend связан с endpoint.

Ответ API не делает установленную службу владельцем сервера. Активная
user-служба выбирается вместо неактивной system-службы. Если активны
несколько способов сразу (system, user и/или подтверждённый процесс lama),
разрушительные действия блокируются.

Когда ни один способ не активен и API молчит, запуск выбирается так:
установленная system-служба, иначе user-служба, иначе свой ``ollama serve``.
Отсутствующий systemctl, недоступная user-шина и ручной процесс обрабатываются
без sudo и без угадывания по имени процесса.

Если пользователь выбрал конкретный файл ollama, а он не совпадает с
``ExecStart`` неактивной службы, запускается этот файл, а служба остаётся
остановленной. Совпадение запускает службу.

Идентичность своего процесса
-----------------------------
Файл состояния хранит PID, время старта из ``/proc/<pid>/stat``, путь
программы, uid и выбранный endpoint. Перед сигналом и ещё раз перед SIGKILL
эта запись сверяется с живым процессом. Совпадения PID и argv ``ollama serve``
недостаточно: другое время старта, другая программа или другой владелец
запрещают сигнал. Старый файл только с PID и повреждённая запись не дают
права завершать процесс.
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
import signal
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from urllib.parse import urlparse

from lama.config import Settings, state_dir
from lama.errors import LamaError, OllamaUnavailable
from lama.i18n import t
from lama.ollama.client import OllamaClient
from lama.ollama.discover import remember_executable, serve_listen_host
from lama.ollama.runner import OllamaRunner, build_serve_argv
from lama.validation import normalize_host

_STATE_VERSION = 1
_WAIT_ATTEMPTS = 20
_WAIT_INTERVAL = 0.25
_IDENTITY_ATTEMPTS = 10
_IDENTITY_INTERVAL = 0.05
_EXECSTART_PATH = re.compile(r"\bpath=(\S+)")


class ControlMode(StrEnum):
    SYSTEMD_SYSTEM = "systemd-system"
    SYSTEMD_USER = "systemd-user"
    MANAGED = "managed"
    LISTENER = "listener"
    EXTERNAL = "external"
    STOPPED = "stopped"
    REMOTE = "remote"
    WILDCARD = "wildcard"
    AMBIGUOUS = "ambiguous"


class EndpointScope(StrEnum):
    LOCAL = "local"
    WILDCARD = "wildcard"
    REMOTE = "remote"


class StateKind(StrEnum):
    ABSENT = "absent"
    READY = "ready"
    LEGACY = "legacy"
    CORRUPT = "corrupt"
    UNREADABLE = "unreadable"


@dataclass(frozen=True)
class CommandCapture:
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class UnitState:
    loaded: bool
    active: bool


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    start_time: int
    executable: str
    uid: int
    endpoint: str


@dataclass(frozen=True)
class StateRecord:
    kind: StateKind
    identity: ProcessIdentity | None = None
    legacy_pid: int | None = None


@dataclass(frozen=True)
class ProcessProbe:
    """Результат чтения /proc без сигнала процессу.

    ``ready`` — cmdline это ``ollama serve`` и идентичность собрана.
    ``other`` — процесс прочитан, но это не ``ollama serve``.
    ``gone`` — каталога процесса нет.
    ``unreadable`` — прочитать или разобрать идентичность нельзя.
    """

    status: str
    identity: ProcessIdentity | None = None


@dataclass
class ServerStatus:
    api_up: bool
    version: str | None
    host: str
    mode: ControlMode
    can_start: bool
    can_stop: bool
    stop_reason: str | None
    pid: int | None
    api_error: str | None = None


Executor = Callable[[list[str]], CommandCapture]
PopenFactory = Callable[..., subprocess.Popen[bytes]]
Sleeper = Callable[[float], None]

_BLOCKED_MODES = frozenset({ControlMode.REMOTE, ControlMode.WILDCARD, ControlMode.AMBIGUOUS})

EXTERNAL_STOP_REASON = (
    "The API is responding, but lama does not control this process: there is no single active systemd ollama "
    "service and no confirmed process started by this utility. Another process will not be stopped."
)
REMOTE_STOP_REASON = (
    "The selected address {host} is remote. An API check does not control the local machine: "
    "start, stop, restart, systemctl, ollama serve, and process signals are unavailable. "
    "Local logs do not apply to this address."
)
WILDCARD_STOP_REASON = (
    "The address {host} is a wildcard (0.0.0.0 or ::). It does not name a specific local server, "
    "so start, stop, restart, and local logs are unavailable. "
    "To control this machine, use a loopback address: 127.0.0.1, localhost, or [::1]."
)
AMBIGUOUS_STOP_REASON = (
    "Several local control methods are active at once ({backends}). "
    "The link to the selected address is ambiguous, so start, stop, and restart are blocked "
    "until only one option remains."
)
LEGACY_LIVE_REASON = (
    "The state file contains only a PID, without start time, program, owner, and address. "
    "That is not enough to stop a process, so no signal will be sent. "
    "Start is also blocked while this PID exists."
)
OTHER_ENDPOINT_REASON = (
    "The state file confirms a lama process for a different address ({endpoint}). "
    "It will not be stopped, and the record will not be replaced by a start for {host}."
)
LEGACY_DEAD_NOTE = "The previous state file stored only a PID and does not allow stopping a process."
CORRUPT_STATE_NOTE = "The state file is damaged or incomplete and does not allow stopping a process."
UNREADABLE_STATE_NOTE = "The state file could not be read. The process will not be stopped from it."
STALE_MISMATCH_NOTE = "The state file record does not match the live process and does not allow stopping it."


class ServerManager:
    def __init__(
        self,
        settings: Settings,
        client: OllamaClient,
        runner: OllamaRunner,
        *,
        executor: Executor | None = None,
        popen: PopenFactory | None = None,
        proc_root: Path = Path("/proc"),
        sleeper: Sleeper = time.sleep,
        killer: Callable[[int, int], None] | None = None,
        pid_path: Path | None = None,
        log_path: Path | None = None,
    ) -> None:
        self.settings = settings
        self.client = client
        self.runner = runner
        self._executor = executor or _subprocess_capture
        self._popen = popen or subprocess.Popen
        self._proc_root = proc_root
        self._sleep = sleeper
        self._kill = killer or os.kill
        self.pid_path = pid_path or (state_dir() / "ollama.pid")
        self.log_path = log_path or (state_dir() / "ollama.log")

    def status(self) -> ServerStatus:
        scope = classify_endpoint(self.settings.host)
        api_up, version, api_error = self._probe_api()
        if scope is not EndpointScope.LOCAL:
            return self._unreachable_status(scope, api_up, version, api_error)
        system_unit = self._unit("system")
        user_unit = self._unit("user")
        record = self._load_state()
        return self._local_status(system_unit, user_unit, record, api_up, version, api_error)

    def start(self, *, prefer_binary: str | None = None) -> str:
        current = self.status()
        self._refuse_blocked(current)
        if current.api_up:
            return t("Server is already responding on {host}.", host=current.host)
        if not current.can_start:
            raise LamaError(current.stop_reason or t("Server start is unavailable."))
        if prefer_binary and self._unit_matches_binary(current.mode, prefer_binary):
            try:
                if current.mode is ControlMode.SYSTEMD_SYSTEM:
                    self._require(_systemctl("start"))
                    self._wait_api()
                    return t("systemd ollama service started.")
                if current.mode is ControlMode.SYSTEMD_USER:
                    self._require(_systemctl("start", user=True))
                    self._wait_api()
                    return t("systemd user ollama service started.")
            except LamaError:
                up, _version, _error = self._probe_api()
                if up:
                    return t("Server is already responding on {host}.", host=current.host)
            self.runner.use_binary(prefer_binary)
            self._start_managed()
            return t("Started ollama serve.")
        if prefer_binary and not self._unit_matches_binary(current.mode, prefer_binary):
            self.runner.use_binary(prefer_binary)
            if current.mode in {ControlMode.SYSTEMD_SYSTEM, ControlMode.SYSTEMD_USER, ControlMode.STOPPED}:
                self._start_managed()
                return t("Started ollama serve.")
            raise LamaError(current.stop_reason or t("Server start is unavailable."))
        if current.mode is ControlMode.SYSTEMD_SYSTEM:
            self._require(_systemctl("start"))
            self._wait_api()
            return t("systemd ollama service started.")
        if current.mode is ControlMode.SYSTEMD_USER:
            self._require(_systemctl("start", user=True))
            self._wait_api()
            return t("systemd user ollama service started.")
        if current.mode is ControlMode.STOPPED:
            self._start_managed()
            return t("Started ollama serve.")
        raise LamaError(current.stop_reason or t("Server start is unavailable."))

    def stop(self) -> str:
        current = self.status()
        self._refuse_blocked(current)
        if current.mode is ControlMode.EXTERNAL or not current.can_stop:
            raise LamaError(current.stop_reason or t("Server stop is unavailable."))
        if current.mode is ControlMode.SYSTEMD_SYSTEM:
            self._require(_systemctl("stop"))
            return t("systemd ollama service stopped.")
        if current.mode is ControlMode.SYSTEMD_USER:
            self._require(_systemctl("stop", user=True))
            return t("systemd user ollama service stopped.")
        if current.mode is ControlMode.MANAGED:
            return self._stop_owned()
        if current.mode is ControlMode.LISTENER:
            return self._stop_listener()
        raise LamaError(current.stop_reason or t("Server stop is unavailable."))

    def restart(self) -> str:
        current = self.status()
        self._refuse_blocked(current)
        if current.mode is ControlMode.EXTERNAL or not (current.can_stop or current.can_start):
            raise LamaError(current.stop_reason or t(EXTERNAL_STOP_REASON))
        if current.mode is ControlMode.SYSTEMD_SYSTEM:
            self._require(_systemctl("restart"))
            self._wait_api()
            return t("systemd ollama service restarted.")
        if current.mode is ControlMode.SYSTEMD_USER:
            self._require(_systemctl("restart", user=True))
            self._wait_api()
            return t("systemd user ollama service restarted.")
        if current.can_stop:
            self.stop()
        message = self.start()
        return t("Server restarted. {message}", message=message)

    def logs(self) -> str:
        current = self.status()
        if current.mode in {ControlMode.REMOTE, ControlMode.WILDCARD, ControlMode.AMBIGUOUS}:
            return current.stop_reason or t("The local log does not apply to the selected address.")
        if current.mode is ControlMode.SYSTEMD_SYSTEM:
            return self._journal(user=False)
        if current.mode is ControlMode.SYSTEMD_USER:
            return self._journal(user=True)
        if current.mode is ControlMode.EXTERNAL:
            return t(
                "This API's log is unavailable: lama does not control the process and will not substitute "
                "local journalctl or ollama.log."
            )
        if self.log_path.is_file() and current.mode in {ControlMode.MANAGED, ControlMode.STOPPED}:
            try:
                content = self.log_path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                raise LamaError(t("Could not read the log {path}: {error}", path=self.log_path, error=exc)) from exc
            return content[-20000:] or t("The log file is empty.")
        return t(
            "No log is available. A log appears when the server is started by a systemd service "
            "or by an ollama serve process that lama started."
        )

    def _unreachable_status(
        self,
        scope: EndpointScope,
        api_up: bool,
        version: str | None,
        api_error: str | None,
    ) -> ServerStatus:
        if scope is EndpointScope.WILDCARD:
            mode = ControlMode.WILDCARD
            reason = t(WILDCARD_STOP_REASON, host=self.settings.host)
        else:
            mode = ControlMode.REMOTE
            reason = t(REMOTE_STOP_REASON, host=self.settings.host)
        return ServerStatus(
            api_up=api_up,
            version=version if api_up else None,
            host=self.settings.host,
            mode=mode,
            can_start=False,
            can_stop=False,
            stop_reason=reason,
            pid=None,
            api_error=api_error,
        )

    def _local_status(
        self,
        system_unit: UnitState | None,
        user_unit: UnitState | None,
        record: StateRecord,
        api_up: bool,
        version: str | None,
        api_error: str | None,
    ) -> ServerStatus:
        system_installed = system_unit is not None and system_unit.loaded
        user_installed = user_unit is not None and user_unit.loaded
        system_active = bool(system_installed and system_unit is not None and system_unit.active)
        user_active = bool(user_installed and user_unit is not None and user_unit.active)
        owned = self._matching_identity(record)
        state_note = self._state_note(record, owned)

        active: list[str] = []
        if system_active:
            active.append("systemd system")
        if user_active:
            active.append("systemd user")
        if owned is not None:
            active.append(t("lama process"))
        if len(active) > 1:
            return self._status(
                mode=ControlMode.AMBIGUOUS,
                api_up=api_up,
                version=version,
                api_error=api_error,
                can_start=False,
                can_stop=False,
                stop_reason=_join(t(AMBIGUOUS_STOP_REASON, backends=", ".join(active)), state_note),
                pid=owned.pid if owned is not None else None,
            )
        unit_note = None if state_note == t(LEGACY_LIVE_REASON) else state_note
        if system_active:
            return self._active_unit_status(ControlMode.SYSTEMD_SYSTEM, api_up, version, api_error, unit_note)
        if user_active:
            return self._active_unit_status(ControlMode.SYSTEMD_USER, api_up, version, api_error, unit_note)
        if owned is not None:
            reason = state_note
            if not api_up:
                reason = _join(
                    t("ollama serve was started by lama, but the API is not responding. Stop it and start it again."),
                    state_note,
                )
            return self._status(
                mode=ControlMode.MANAGED,
                api_up=api_up,
                version=version,
                api_error=api_error,
                can_start=False,
                can_stop=True,
                stop_reason=reason,
                pid=owned.pid,
            )
        if self._legacy_pid_alive(record):
            return self._status(
                mode=ControlMode.EXTERNAL,
                api_up=api_up,
                version=version,
                api_error=api_error,
                can_start=False,
                can_stop=False,
                stop_reason=_join(t(LEGACY_LIVE_REASON), state_note if state_note != t(LEGACY_LIVE_REASON) else None),
                pid=None,
            )
        if not system_active and not user_active and owned is None:
            listener = self._listener_identity()
            if listener is not None:
                reason = None
                if not api_up:
                    reason = t(
                        "ollama serve is listening on {host} (pid {pid}), but the API is not responding. "
                        "Stop it and start it again.",
                        host=self.settings.host,
                        pid=listener.pid,
                    )
                return self._status(
                    mode=ControlMode.LISTENER,
                    api_up=api_up,
                    version=version,
                    api_error=api_error,
                    can_start=False,
                    can_stop=True,
                    stop_reason=_join(reason, state_note),
                    pid=listener.pid,
                )
        if api_up:
            return self._status(
                mode=ControlMode.EXTERNAL,
                api_up=True,
                version=version,
                api_error=None,
                can_start=False,
                can_stop=False,
                stop_reason=_join(t(EXTERNAL_STOP_REASON), state_note),
                pid=None,
            )
        if system_installed and self._unit_serves_selected("system"):
            return self._start_target_status(ControlMode.SYSTEMD_SYSTEM, api_error, state_note)
        if user_installed and self._unit_serves_selected("user"):
            return self._start_target_status(ControlMode.SYSTEMD_USER, api_error, state_note)
        other = self._confirmed_other_endpoint(record)
        if other is not None:
            return self._status(
                mode=ControlMode.EXTERNAL,
                api_up=False,
                version=None,
                api_error=api_error,
                can_start=False,
                can_stop=False,
                stop_reason=t(OTHER_ENDPOINT_REASON, endpoint=other.endpoint, host=self.settings.host),
                pid=None,
            )
        if self._current_endpoint_unconfirmed(record):
            return self._status(
                mode=ControlMode.EXTERNAL,
                api_up=False,
                version=None,
                api_error=api_error,
                can_start=False,
                can_stop=False,
                stop_reason=state_note or t(STALE_MISMATCH_NOTE),
                pid=None,
            )
        return self._status(
            mode=ControlMode.STOPPED,
            api_up=False,
            version=None,
            api_error=api_error,
            can_start=True,
            can_stop=False,
            stop_reason=_join(t("The server is not running."), state_note),
            pid=None,
        )

    def _active_unit_status(
        self,
        mode: ControlMode,
        api_up: bool,
        version: str | None,
        api_error: str | None,
        state_note: str | None,
    ) -> ServerStatus:
        reason = state_note
        if not api_up:
            reason = _join(
                t(
                    "The service is active, but the API is not responding. "
                    "Stop or restart it instead of starting a second server."
                ),
                state_note,
            )
        return self._status(
            mode=mode,
            api_up=api_up,
            version=version,
            api_error=api_error,
            can_start=False,
            can_stop=True,
            stop_reason=reason,
            pid=None,
        )

    def _start_target_status(self, mode: ControlMode, api_error: str | None, state_note: str | None) -> ServerStatus:
        return self._status(
            mode=mode,
            api_up=False,
            version=None,
            api_error=api_error,
            can_start=True,
            can_stop=False,
            stop_reason=_join(t("The service is installed but not running."), state_note),
            pid=None,
        )

    def _status(
        self,
        *,
        mode: ControlMode,
        api_up: bool,
        version: str | None,
        api_error: str | None,
        can_start: bool,
        can_stop: bool,
        stop_reason: str | None,
        pid: int | None,
    ) -> ServerStatus:
        return ServerStatus(
            api_up=api_up,
            version=version,
            host=self.settings.host,
            mode=mode,
            can_start=can_start,
            can_stop=can_stop,
            stop_reason=stop_reason,
            pid=pid,
            api_error=api_error,
        )

    def _state_note(self, record: StateRecord, owned: ProcessIdentity | None) -> str | None:
        if owned is not None:
            return None
        if record.kind is StateKind.LEGACY:
            if self._legacy_pid_alive(record):
                return t(LEGACY_LIVE_REASON)
            return t(LEGACY_DEAD_NOTE)
        if record.kind is StateKind.CORRUPT:
            return t(CORRUPT_STATE_NOTE)
        if record.kind is StateKind.UNREADABLE:
            return t(UNREADABLE_STATE_NOTE)
        if record.kind is StateKind.READY and record.identity is not None:
            if record.identity.endpoint != self.settings.host:
                return None
            observation = classify_observation(record.identity, self._proc_root, endpoint=self.settings.host)
            if observation == "gone":
                return t("The state file points at a process that has already exited. No signal is sent.")
            if observation == "unreadable":
                return t("The process identity from the state file cannot be read right now. No signal is sent.")
            if observation == "mismatch":
                return t(STALE_MISMATCH_NOTE)
        return None

    def _matching_identity(self, record: StateRecord) -> ProcessIdentity | None:
        identity = record.identity
        if record.kind is not StateKind.READY or identity is None:
            return None
        if identity.endpoint != self.settings.host:
            return None
        if classify_observation(identity, self._proc_root, endpoint=self.settings.host) != "match":
            return None
        return identity

    def _legacy_pid_alive(self, record: StateRecord) -> bool:
        if record.kind is not StateKind.LEGACY or record.legacy_pid is None:
            return False
        try:
            return (self._proc_root / str(record.legacy_pid)).exists()
        except OSError:
            return True

    def _current_endpoint_unconfirmed(self, record: StateRecord) -> bool:
        identity = record.identity
        if record.kind is not StateKind.READY or identity is None or identity.endpoint != self.settings.host:
            return False
        observation = classify_observation(identity, self._proc_root, endpoint=self.settings.host)
        return observation in {"mismatch", "unreadable"}

    def _confirmed_other_endpoint(self, record: StateRecord) -> ProcessIdentity | None:
        identity = record.identity
        if record.kind is not StateKind.READY or identity is None or identity.endpoint == self.settings.host:
            return None
        if classify_observation(identity, self._proc_root, endpoint=identity.endpoint) != "match":
            return None
        return identity

    def _refuse_blocked(self, current: ServerStatus) -> None:
        if current.mode in _BLOCKED_MODES:
            raise LamaError(current.stop_reason or t("Server control is unavailable for the selected address."))

    def _probe_api(self) -> tuple[bool, str | None, str | None]:
        try:
            return True, self.client.version(), None
        except OllamaUnavailable as exc:
            return False, None, str(exc)
        except LamaError as exc:
            return False, None, str(exc)

    def _listener_identity(self) -> ProcessIdentity | None:
        target = endpoint_port_and_addresses(self.settings.host)
        if target is None:
            return None
        port, addresses = target
        found = listening_serve_identities(
            self._proc_root,
            port=port,
            addresses=addresses,
            endpoint=self.settings.host,
        )
        if len(found) != 1:
            return None
        remember_executable(found[0].executable, serve_listen_host(self._proc_root, found[0].pid))
        return found[0]

    def stop_other_than(self, executable: str) -> None:
        """Остановить другие локальные ``ollama serve`` и чужие активные службы."""
        try:
            chosen = os.path.realpath(executable)
        except OSError:
            chosen = executable
        for user, scope in ((False, "system"), (True, "user")):
            unit = self._unit(scope)
            if unit is None or not unit.active:
                continue
            binary = self._unit_executable(scope)
            if binary is not None and os.path.realpath(binary) == chosen:
                continue
            self._require(_systemctl("stop", user=user))
        for identity in serve_identities(self._proc_root, endpoint=self.settings.host):
            try:
                exe = os.path.realpath(identity.executable)
            except OSError:
                exe = identity.executable
            if exe == chosen:
                continue
            observation = classify_observation(identity, self._proc_root, endpoint=self.settings.host)
            if observation != "match":
                continue
            self._stop_confirmed_process(identity)

    def _unit_serves_selected(self, scope: str) -> bool:
        """Служба подходит для запуска, только если её OLLAMA_HOST обслуживает выбранный адрес."""
        argv = ["systemctl", "show", "ollama.service", "-p", "Environment", "--no-pager"]
        if scope == "user":
            argv = ["systemctl", "--user", "show", "ollama.service", "-p", "Environment", "--no-pager"]
        try:
            result = self._executor(argv)
        except FileNotFoundError:
            return False
        host = parse_environment_value(result.stdout, "OLLAMA_HOST") or "127.0.0.1:11434"
        try:
            return endpoint_served_by(host, self.settings.host)
        except (LamaError, ValueError):
            return False

    def _stop_listener(self) -> str:
        identity = self._listener_identity()
        if identity is None:
            raise LamaError(
                t(
                    "The ollama serve process listening on {host} is no longer there. No signal was sent.",
                    host=self.settings.host,
                )
            )
        return self._stop_confirmed_process(identity)

    def _stop_confirmed_process(self, identity: ProcessIdentity) -> str:
        self._signal_confirmed(identity, signal.SIGTERM)
        outcome = self._wait_until_changed(identity)
        if outcome == "gone":
            return t("The ollama serve process listening on {host} has been stopped.", host=self.settings.host)
        if outcome != "match":
            raise LamaError(t("Process identity changed. Another process was not stopped."))
        self._signal_confirmed(identity, signal.SIGKILL)
        outcome = self._wait_until_changed(identity)
        if outcome == "gone":
            return t("The ollama serve process listening on {host} has been stopped.", host=self.settings.host)
        raise LamaError(
            t("The process did not exit in time. The state file was kept, and no further signal was sent.")
        )

    def _signal_confirmed(self, identity: ProcessIdentity, sig: int) -> None:
        observation = classify_observation(identity, self._proc_root, endpoint=self.settings.host)
        if observation == "gone":
            raise LamaError(t("The process from the state file has already exited. No signal was sent."))
        if observation == "mismatch":
            raise LamaError(t("Process identity changed. Another process was not stopped."))
        if observation != "match":
            raise LamaError(
                t("Could not read /proc before the signal. No signal was sent, and the state file was kept.")
            )
        self._kill(identity.pid, sig)

    def service_binary(self, mode: ControlMode) -> str | None:
        """Путь ExecStart неактивной или активной службы, если она есть в этом режиме."""
        if mode is ControlMode.SYSTEMD_SYSTEM:
            return self._unit_executable("system")
        if mode is ControlMode.SYSTEMD_USER:
            return self._unit_executable("user")
        return None

    def _unit_matches_binary(self, mode: ControlMode, binary: str) -> bool:
        unit = self.service_binary(mode)
        if unit is None:
            return False
        return unit == os.path.realpath(binary)

    def _unit_executable(self, scope: str) -> str | None:
        argv = ["systemctl", "show", "ollama.service", "-p", "ExecStart", "--no-pager"]
        if scope == "user":
            argv = ["systemctl", "--user", "show", "ollama.service", "-p", "ExecStart", "--no-pager"]
        try:
            result = self._executor(argv)
        except FileNotFoundError:
            return None
        found = parse_execstart_path(result.stdout)
        if not found:
            return None
        return os.path.realpath(found)

    def _unit(self, scope: str) -> UnitState | None:
        argv = ["systemctl", "show", "ollama.service", "-p", "LoadState", "-p", "ActiveState"]
        if scope == "user":
            argv.insert(1, "--user")
        argv.append("--no-pager")
        try:
            result = self._executor(argv)
        except FileNotFoundError:
            return None
        if "LoadState=" not in result.stdout:
            return None
        values: dict[str, str] = {}
        for line in result.stdout.splitlines():
            key, separator, value = line.partition("=")
            if separator:
                values[key.strip()] = value.strip()
        return UnitState(loaded=values.get("LoadState") == "loaded", active=values.get("ActiveState") == "active")

    def _load_state(self) -> StateRecord:
        try:
            if not self.pid_path.exists():
                return StateRecord(StateKind.ABSENT)
            text = self.pid_path.read_text(encoding="utf-8")
        except OSError:
            return StateRecord(StateKind.UNREADABLE)
        except UnicodeDecodeError:
            return StateRecord(StateKind.CORRUPT)
        return parse_state_text(text)

    def _start_managed(self) -> None:
        try:
            self.pid_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise LamaError(
                t("Could not create directory {path}: {error}", path=self.pid_path.parent, error=exc)
            ) from exc
        other = self._confirmed_other_endpoint(self._load_state())
        if other is not None:
            raise LamaError(t(OTHER_ENDPOINT_REASON, endpoint=other.endpoint, host=self.settings.host))
        binary = self.runner.binary()
        try:
            log_handle = self.log_path.open("ab")
        except OSError as exc:
            raise LamaError(t("Could not open the log {path}: {error}", path=self.log_path, error=exc)) from exc
        try:
            process = self._popen(
                build_serve_argv(binary),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                env=self.runner.environment(),
            )
        finally:
            log_handle.close()
        identity = self._capture_identity(process.pid)
        if identity is None:
            _terminate_untracked(process)
            raise LamaError(
                t(
                    "Could not confirm the identity of process {pid} through /proc. "
                    "The process was stopped and was not recorded as permission to stop it.",
                    pid=process.pid,
                )
            )
        try:
            self.pid_path.write_text(dump_process_state(identity), encoding="utf-8")
        except OSError as exc:
            _terminate_untracked(process)
            raise LamaError(t("Could not write the state file {path}: {error}", path=self.pid_path, error=exc)) from exc
        try:
            self._wait_api()
        except LamaError as exc:
            raise LamaError(t("ollama serve is running (pid {pid}), but {error}", pid=process.pid, error=exc)) from exc

    def _capture_identity(self, pid: int) -> ProcessIdentity | None:
        for attempt in range(_IDENTITY_ATTEMPTS):
            if attempt:
                self._sleep(_IDENTITY_INTERVAL)
            probe = probe_process(pid, self._proc_root, endpoint=self.settings.host)
            if probe.status == "ready" and probe.identity is not None:
                return probe.identity
        return None

    def _stop_owned(self) -> str:
        record = self._load_state()
        identity = self._matching_identity(record)
        if identity is None:
            observation = _observation_for_record(record, self._proc_root, endpoint=self.settings.host)
            if observation == "gone" and record.kind is StateKind.READY:
                self._clear_state(record.identity)
                return t("The process from the state file has already exited. No signal was sent.")
            raise LamaError(
                self._state_note(record, None)
                or t("The process from the state file is not confirmed and will not be stopped.")
            )
        self._signal_if_match(identity, signal.SIGTERM)
        outcome = self._wait_until_changed(identity)
        if outcome == "gone":
            self._clear_state(identity)
            return t("The ollama serve process started by lama has been stopped.")
        if outcome == "unreadable":
            raise LamaError(t("Could not read /proc again. SIGKILL was not sent, and the state file was kept."))
        if outcome == "mismatch":
            self._clear_state(identity)
            raise LamaError(t("Process identity changed. Another process was not stopped with SIGKILL."))
        self._signal_if_match(identity, signal.SIGKILL)
        outcome = self._wait_until_changed(identity)
        if outcome == "gone":
            self._clear_state(identity)
            return t("The ollama serve process started by lama has been stopped.")
        if outcome == "mismatch":
            self._clear_state(identity)
            raise LamaError(t("After SIGKILL the PID already belongs to another process. No further signal was sent."))
        raise LamaError(
            t("The process did not exit in time. The state file was kept, and no further signal was sent.")
        )

    def _signal_if_match(self, identity: ProcessIdentity, sig: int) -> None:
        observation = classify_observation(identity, self._proc_root, endpoint=self.settings.host)
        if observation == "gone":
            self._clear_state(identity)
            raise LamaError(t("The process from the state file has already exited. No signal was sent."))
        if observation == "mismatch":
            self._clear_state(identity)
            raise LamaError(t("Process identity changed. Another process was not stopped."))
        if observation != "match":
            raise LamaError(
                t("Could not read /proc before the signal. No signal was sent, and the state file was kept.")
            )
        record = self._load_state()
        if record.identity != identity:
            raise LamaError(t("The state file changed before the signal. No signal was sent."))
        self._kill(identity.pid, sig)

    def _wait_until_changed(self, identity: ProcessIdentity) -> str:
        for _ in range(_WAIT_ATTEMPTS):
            observation = classify_observation(identity, self._proc_root, endpoint=self.settings.host)
            if observation != "match":
                return observation
            self._sleep(_WAIT_INTERVAL)
        return classify_observation(identity, self._proc_root, endpoint=self.settings.host)

    def _clear_state(self, identity: ProcessIdentity | None) -> None:
        if identity is None:
            return
        record = self._load_state()
        if record.kind is not StateKind.READY or record.identity != identity:
            return
        try:
            self.pid_path.unlink(missing_ok=True)
        except OSError as exc:
            raise LamaError(
                t(
                    "The process is no longer confirmed, but {path} could not be deleted: {error}",
                    path=self.pid_path,
                    error=exc,
                )
            ) from exc

    def _wait_api(self) -> None:
        last_error = t("The API did not respond.")
        for _ in range(_WAIT_ATTEMPTS):
            try:
                self.client.version()
                return
            except LamaError as exc:
                last_error = str(exc)
                self._sleep(_WAIT_INTERVAL)
        raise LamaError(t("The API did not respond: {error}", error=last_error))

    def _journal(self, *, user: bool) -> str:
        argv = ["journalctl", "-u", "ollama.service", "-n", "100", "--no-pager"]
        if user:
            argv = ["journalctl", "--user", "-u", "ollama.service", "-n", "100", "--no-pager"]
        result = self._executor(argv)
        text = (result.stdout or result.stderr).strip()
        if result.returncode != 0:
            raise LamaError(text or t("Could not read journalctl."))
        return text or t("The systemd journal is empty.")

    def _require(self, argv: list[str]) -> None:
        result = self._executor(argv)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise LamaError(
                detail
                or t("Command {command} exited with code {code}.", command=" ".join(argv), code=result.returncode)
            )


def parse_environment_value(text: str, key: str) -> str | None:
    """Достать ``KEY=value`` из вывода ``systemctl show -p Environment``."""
    prefix = key + "="
    for line in text.splitlines():
        body = line[len("Environment=") :] if line.startswith("Environment=") else line
        for token in body.split():
            if token.startswith(prefix):
                return token[len(prefix) :].strip('"')
    return None


def endpoint_served_by(unit_host: str, selected_base: str) -> bool:
    """True, если адрес службы принимает соединения на выбранный endpoint.

    Совпадают порт и хост. ``0.0.0.0`` и ``::`` обслуживают loopback на том же порту.
    """
    unit_base, _env = normalize_host(unit_host)
    unit = urlparse(unit_base)
    selected = urlparse(selected_base)
    if (unit.port or 11434) != (selected.port or 11434):
        return False
    if classify_endpoint(unit_base) is EndpointScope.WILDCARD:
        return classify_endpoint(selected_base) is EndpointScope.LOCAL
    if _same_loopback(unit.hostname, selected.hostname):
        return True
    return _host_text(unit.hostname) == _host_text(selected.hostname)


def endpoint_port_and_addresses(base_url: str) -> tuple[int, set[str]] | None:
    """Порт и локальные адреса, которые обслуживают выбранный endpoint."""
    parsed = urlparse(base_url)
    host = parsed.hostname
    if not host:
        return None
    port = parsed.port or 11434
    name = host.rstrip(".").lower()
    if name == "localhost":
        return port, {"127.0.0.1", "::1", "0.0.0.0", "::"}
    try:
        address = ipaddress.ip_address(name)
    except ValueError:
        return None
    if address.is_loopback or address.is_unspecified:
        return port, {"127.0.0.1", "::1", "0.0.0.0", "::"}
    return port, {address.compressed, "0.0.0.0", "::"}


def parse_listening_inodes(text: str, *, port: int, addresses: set[str]) -> set[int]:
    """Inode слушающих сокетов из ``/proc/net/tcp`` или ``tcp6``."""
    wanted = {_normalize_ip(item) for item in addresses}
    inodes: set[int] = set()
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 10 or ":" not in parts[1]:
            continue
        if parts[3] != "0A":
            continue
        local, separator, raw_port = parts[1].rpartition(":")
        if not separator:
            continue
        try:
            local_port = int(raw_port, 16)
            inode = int(parts[9])
        except ValueError:
            continue
        if local_port != port or inode <= 0:
            continue
        decoded = _decode_proc_address(local)
        if decoded is None or _normalize_ip(decoded) not in wanted:
            continue
        inodes.add(inode)
    return inodes


def serve_identities(proc_root: Path, *, endpoint: str) -> list[ProcessIdentity]:
    """Все процессы ``ollama serve``, которые можно прочитать в /proc."""
    try:
        entries = list(proc_root.iterdir())
    except OSError:
        return []
    found: list[ProcessIdentity] = []
    for entry in entries:
        if not entry.name.isdigit():
            continue
        command = read_cmdline(int(entry.name), proc_root)
        if not _is_serve_command(command or []):
            continue
        probe = probe_process(int(entry.name), proc_root, endpoint=endpoint)
        if probe.status == "ready" and probe.identity is not None:
            found.append(probe.identity)
    return found


def listening_serve_identities(
    proc_root: Path,
    *,
    port: int,
    addresses: set[str],
    endpoint: str,
) -> list[ProcessIdentity]:
    """Процессы ``ollama serve``, которые слушают выбранный порт."""
    inodes: set[int] = set()
    for name in ("tcp", "tcp6"):
        path = proc_root / "net" / name
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        inodes.update(parse_listening_inodes(text, port=port, addresses=addresses))
    if not inodes:
        return []
    try:
        entries = list(proc_root.iterdir())
    except OSError:
        return []
    found: list[ProcessIdentity] = []
    for entry in entries:
        if not entry.name.isdigit() or not _process_has_socket(entry, inodes):
            continue
        probe = probe_process(int(entry.name), proc_root, endpoint=endpoint)
        if probe.status == "ready" and probe.identity is not None:
            found.append(probe.identity)
    return found


def parse_execstart_path(text: str) -> str | None:
    """Достать ``path=`` из вывода ``systemctl show -p ExecStart``."""
    match = _EXECSTART_PATH.search(text)
    if not match:
        return None
    raw = match.group(1).rstrip(";")
    if not raw or raw.startswith("["):
        return None
    return raw


def classify_endpoint(base_url: str) -> EndpointScope:
    """Отнести нормализованный базовый URL к локальному, wildcard или удалённому адресу.

    Разбор идёт через URL и ``ipaddress``. Произвольный hostname не считается
    локальным и не резолвится.
    """
    host = urlparse(base_url).hostname
    if not host:
        return EndpointScope.REMOTE
    name = host.rstrip(".").lower()
    if name == "localhost":
        return EndpointScope.LOCAL
    try:
        address = ipaddress.ip_address(name)
    except ValueError:
        return EndpointScope.REMOTE
    if address.is_loopback:
        return EndpointScope.LOCAL
    if address.is_unspecified:
        return EndpointScope.WILDCARD
    return EndpointScope.REMOTE


def parse_stat_start_time(stat_text: str) -> int | None:
    """Достать starttime (поле 22) из ``/proc/<pid>/stat``.

    Поле comm стоит в скобках и может содержать пробелы и скобки, поэтому
    хвост берётся после последней ``)``.
    """
    text = stat_text.strip()
    open_paren = text.find("(")
    close_paren = text.rfind(")")
    if open_paren < 0 or close_paren <= open_paren:
        return None
    fields = text[close_paren + 1 :].split()
    if len(fields) <= 19:
        return None
    try:
        start_time = int(fields[19])
    except ValueError:
        return None
    if start_time < 0:
        return None
    return start_time


def dump_process_state(identity: ProcessIdentity) -> str:
    payload = {
        "version": _STATE_VERSION,
        "pid": identity.pid,
        "start_time": identity.start_time,
        "executable": identity.executable,
        "uid": identity.uid,
        "endpoint": identity.endpoint,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def parse_state_text(text: str) -> StateRecord:
    stripped = text.strip()
    if not stripped:
        return StateRecord(StateKind.CORRUPT)
    if stripped.isdigit():
        pid = int(stripped)
        if pid <= 0:
            return StateRecord(StateKind.CORRUPT)
        return StateRecord(StateKind.LEGACY, legacy_pid=pid)
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        return StateRecord(StateKind.CORRUPT)
    identity = _identity_from_json(data)
    if identity is None:
        return StateRecord(StateKind.CORRUPT)
    return StateRecord(StateKind.READY, identity=identity)


def probe_process(pid: int, proc_root: Path, *, endpoint: str) -> ProcessProbe:
    directory = proc_root / str(pid)
    try:
        if not directory.exists():
            return ProcessProbe("gone")
    except OSError:
        return ProcessProbe("unreadable")
    try:
        command = read_cmdline(pid, proc_root)
        stat_text = (directory / "stat").read_text(encoding="utf-8", errors="replace")
        executable = os.readlink(directory / "exe")
        uid = _read_uid(directory / "status")
    except OSError:
        return ProcessProbe("unreadable")
    start_time = parse_stat_start_time(stat_text)
    if command is None or start_time is None or not executable or uid is None:
        return ProcessProbe("unreadable")
    identity = ProcessIdentity(
        pid=pid,
        start_time=start_time,
        executable=executable,
        uid=uid,
        endpoint=endpoint,
    )
    if not _is_serve_command(command):
        return ProcessProbe("other", identity)
    return ProcessProbe("ready", identity)


def classify_observation(identity: ProcessIdentity, proc_root: Path, *, endpoint: str) -> str:
    """Сравнить сохранённую идентичность с /proc: match, gone, mismatch или unreadable."""
    if identity.endpoint != endpoint:
        return "mismatch"
    probe = probe_process(identity.pid, proc_root, endpoint=endpoint)
    if probe.status in {"gone", "unreadable"}:
        return probe.status
    if probe.status != "ready" or probe.identity != identity:
        return "mismatch"
    return "match"


def is_owned_serve(identity: ProcessIdentity, proc_root: Path = Path("/proc")) -> bool:
    """Процесс принадлежит lama, только если живая идентичность совпала с записью."""
    return classify_observation(identity, proc_root, endpoint=identity.endpoint) == "match"


def read_cmdline(pid: int, proc_root: Path = Path("/proc")) -> list[str] | None:
    path = proc_root / str(pid) / "cmdline"
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if not raw:
        return None
    parts = [part.decode(errors="replace") for part in raw.split(b"\0") if part]
    return parts or None


def _identity_from_json(data: object) -> ProcessIdentity | None:
    if not isinstance(data, dict):
        return None
    if data.get("version") != _STATE_VERSION:
        return None
    pid = _strict_int(data.get("pid"))
    start_time = _strict_int(data.get("start_time"))
    uid = _strict_int(data.get("uid"))
    executable = data.get("executable")
    endpoint = data.get("endpoint")
    if pid is None or pid <= 0 or start_time is None or start_time < 0 or uid is None or uid < 0:
        return None
    if not isinstance(executable, str) or not executable or not isinstance(endpoint, str) or not endpoint:
        return None
    return ProcessIdentity(pid=pid, start_time=start_time, executable=executable, uid=uid, endpoint=endpoint)


def _strict_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _read_uid(path: Path) -> int | None:
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        if not line.startswith("Uid:"):
            continue
        parts = line.split()
        if len(parts) < 2:
            return None
        try:
            uid = int(parts[1])
        except ValueError:
            return None
        if uid < 0:
            return None
        return uid
    return None


def _host_text(host: str | None) -> str:
    if not host:
        return ""
    return host.rstrip(".").lower()


def _same_loopback(left: str | None, right: str | None) -> bool:
    return _is_loopback_name(left) and _is_loopback_name(right)


def _is_loopback_name(host: str | None) -> bool:
    name = _host_text(host)
    if name == "localhost":
        return True
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False


def _normalize_ip(value: str) -> str:
    try:
        return ipaddress.ip_address(value).compressed
    except ValueError:
        return value


def _decode_proc_address(hex_ip: str) -> str | None:
    try:
        raw = bytes.fromhex(hex_ip)
    except ValueError:
        return None
    if len(raw) == 4:
        return ".".join(str(byte) for byte in reversed(raw))
    if len(raw) != 16:
        return None
    words = [raw[index : index + 4][::-1] for index in range(0, 16, 4)]
    try:
        return str(ipaddress.ip_address(b"".join(words)))
    except ValueError:
        return None


def _process_has_socket(proc_dir: Path, inodes: set[int]) -> bool:
    fd_dir = proc_dir / "fd"
    try:
        entries = list(fd_dir.iterdir())
    except OSError:
        return False
    for entry in entries:
        try:
            target = os.readlink(entry)
        except OSError:
            continue
        if not target.startswith("socket:[") or not target.endswith("]"):
            continue
        inode = target[len("socket:[") : -1]
        if inode.isdigit() and int(inode) in inodes:
            return True
    return False


def _is_serve_command(command: list[str]) -> bool:
    return len(command) >= 2 and Path(command[0]).name == "ollama" and command[1] == "serve"


def _observation_for_record(record: StateRecord, proc_root: Path, *, endpoint: str) -> str:
    if record.kind is not StateKind.READY or record.identity is None:
        return "mismatch"
    return classify_observation(record.identity, proc_root, endpoint=endpoint)


def _terminate_untracked(process: subprocess.Popen[bytes]) -> None:
    try:
        process.kill()
    except OSError:
        return
    try:
        process.wait(timeout=1)
    except OSError:
        return


def _join(first: str | None, second: str | None) -> str | None:
    parts = [part for part in (first, second) if part]
    if not parts:
        return None
    return " ".join(parts)


def _systemctl(action: str, *, user: bool = False) -> list[str]:
    argv = ["systemctl"]
    if user:
        argv.append("--user")
    argv.extend([action, "ollama.service"])
    return argv


def _subprocess_capture(argv: list[str]) -> CommandCapture:
    completed = subprocess.run(argv, check=False, capture_output=True, text=True)
    return CommandCapture(completed.returncode, completed.stdout, completed.stderr)
