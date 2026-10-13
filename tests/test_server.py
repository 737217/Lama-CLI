import json
import shutil
import signal
import socket
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from lama.config import Settings
from lama.errors import LamaError, OllamaUnavailable
from lama.ollama.runner import CommandResult, OllamaRunner
from lama.server.manager import (
    CommandCapture,
    ControlMode,
    EndpointScope,
    ProcessIdentity,
    ServerManager,
    ServerStatus,
    classify_endpoint,
    dump_process_state,
    is_owned_serve,
    parse_stat_start_time,
)
from lama.services import stop_model
from lama.ui import display

SHOW_SYSTEM = ("systemctl", "show", "ollama.service", "-p", "LoadState", "-p", "ActiveState", "--no-pager")
SHOW_USER = ("systemctl", "--user", "show", "ollama.service", "-p", "LoadState", "-p", "ActiveState", "--no-pager")
LOCAL_HOST = "http://127.0.0.1:11434"


class FakeClient:
    def __init__(self, *, up: bool = False, version: str = "0.32.15") -> None:
        self.up = up
        self._version = version
        self.calls = 0

    def version(self) -> str:
        self.calls += 1
        if not self.up:
            raise OllamaUnavailable("Сервер Ollama недоступен по адресу http://127.0.0.1:11434.")
        return self._version


class FakePopen:
    def __init__(self, pid: int, killed: list[int]) -> None:
        self.pid = pid
        self._killed = killed

    def kill(self) -> None:
        self._killed.append(self.pid)

    def wait(self, timeout: int | None = None) -> int:
        return 0


def _settings(host: str = LOCAL_HOST, env_host: str = "127.0.0.1:11434") -> Settings:
    return Settings(host=host, env_host=env_host)


def _unit(loaded: bool, active: bool = False) -> CommandCapture:
    if not loaded:
        return CommandCapture(1, "LoadState=not-found\nActiveState=inactive\n", "")
    state = "active" if active else "inactive"
    return CommandCapture(0, f"LoadState=loaded\nActiveState={state}\n", "")


def _mutations(calls: list[tuple[str, ...]]) -> list[tuple[str, ...]]:
    verbs = {"start", "stop", "restart"}
    return [item for item in calls if any(part in verbs for part in item)]


def _identity(
    *,
    pid: int = 4242,
    start_time: int = 100,
    executable: str = "/usr/bin/ollama",
    uid: int = 1000,
    endpoint: str = LOCAL_HOST,
) -> ProcessIdentity:
    return ProcessIdentity(pid=pid, start_time=start_time, executable=executable, uid=uid, endpoint=endpoint)


def _stat_line(pid: int, comm: str, start_time: int) -> str:
    fields = ["S", *(["0"] * 18), str(start_time)]
    return f"{pid} ({comm}) {' '.join(fields)}\n"


def _install_proc(
    proc_root: Path,
    identity: ProcessIdentity,
    *,
    command: bytes | None = None,
    comm: str = "ollama",
    start_time: int | None = None,
    executable: str | None = None,
    uid: int | None = None,
) -> Path:
    proc = proc_root / str(identity.pid)
    proc.mkdir(parents=True, exist_ok=True)
    program = executable if executable is not None else identity.executable
    payload = command if command is not None else program.encode() + b"\0serve\0"
    (proc / "cmdline").write_bytes(payload)
    (proc / "stat").write_text(
        _stat_line(identity.pid, comm, identity.start_time if start_time is None else start_time),
        encoding="utf-8",
    )
    owner = identity.uid if uid is None else uid
    (proc / "status").write_text(f"Uid:\t{owner}\t{owner}\t{owner}\t{owner}\n", encoding="utf-8")
    link = proc / "exe"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(program)
    return proc


def _manager(
    tmp_path: Path,
    client: FakeClient,
    responses: dict[tuple[str, ...], CommandCapture] | None = None,
    *,
    settings: Settings | None = None,
    killer=None,
    popen=None,
    sleeper=None,
    executor=None,
) -> tuple[ServerManager, list[tuple[str, ...]], list[tuple[int, int]]]:
    calls: list[tuple[str, ...]] = []
    killed: list[tuple[int, int]] = []

    def default_executor(argv: list[str]) -> CommandCapture:
        calls.append(tuple(argv))
        if responses is not None and tuple(argv) in responses:
            return responses[tuple(argv)]
        if argv and argv[0] in {"systemctl", "journalctl"}:
            return CommandCapture(0, "", "")
        return CommandCapture(1, "", "not found")

    def default_killer(pid: int, sig: int) -> None:
        killed.append((pid, sig))

    chosen = settings or _settings()
    manager = ServerManager(
        chosen,
        client,  # type: ignore[arg-type]
        OllamaRunner(chosen, binary="/usr/bin/ollama"),
        executor=executor or default_executor,
        popen=popen,
        proc_root=tmp_path / "proc",
        sleeper=sleeper or (lambda _: None),
        killer=killer or default_killer,
        pid_path=tmp_path / "ollama.pid",
        log_path=tmp_path / "ollama.log",
    )
    return manager, calls, killed


def _none() -> dict[tuple[str, ...], CommandCapture]:
    return {SHOW_SYSTEM: _unit(False), SHOW_USER: _unit(False)}


def test_parse_stat_allows_spaces_and_parentheses_in_comm() -> None:
    assert parse_stat_start_time(_stat_line(5, "ollama serve", 42)) == 42
    assert parse_stat_start_time(_stat_line(5, "ollama (main)", 99)) == 99
    assert parse_stat_start_time("1 (broken") is None
    assert parse_stat_start_time("1 (ollama) S 0\n") is None


def test_endpoint_classification_uses_addresses_not_dns(monkeypatch) -> None:
    def fail_lookup(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("классификация не должна резолвить DNS")

    monkeypatch.setattr(socket, "getaddrinfo", fail_lookup)
    monkeypatch.setattr(socket, "gethostbyname", fail_lookup)
    assert classify_endpoint("http://127.0.0.1:11434") is EndpointScope.LOCAL
    assert classify_endpoint("http://127.0.0.2:11434") is EndpointScope.LOCAL
    assert classify_endpoint("http://localhost:11434") is EndpointScope.LOCAL
    assert classify_endpoint("http://LocalHost.:11434") is EndpointScope.LOCAL
    assert classify_endpoint("http://[::1]:11434") is EndpointScope.LOCAL
    assert classify_endpoint("http://[0:0:0:0:0:0:0:1]:11434") is EndpointScope.LOCAL
    assert classify_endpoint("http://[::ffff:127.0.0.1]:11434") is EndpointScope.LOCAL
    assert classify_endpoint("http://0.0.0.0:11434") is EndpointScope.WILDCARD
    assert classify_endpoint("http://[::]:11434") is EndpointScope.WILDCARD
    assert classify_endpoint("http://remote.example:11434") is EndpointScope.REMOTE
    assert classify_endpoint("http://localhost.example:11434") is EndpointScope.REMOTE
    assert classify_endpoint("http://10.1.2.3:11434") is EndpointScope.REMOTE
    assert classify_endpoint("http://[2001:db8::5]:11434") is EndpointScope.REMOTE
    assert classify_endpoint("http://my-pc:11434") is EndpointScope.REMOTE


@pytest.mark.parametrize(
    ("host", "env_host", "mode"),
    [
        ("http://remote.example:11434", "remote.example:11434", ControlMode.REMOTE),
        ("http://10.1.2.3:11434", "10.1.2.3:11434", ControlMode.REMOTE),
        ("http://[2001:db8::5]:11434", "[2001:db8::5]:11434", ControlMode.REMOTE),
        ("http://localhost.example:11434", "localhost.example:11434", ControlMode.REMOTE),
        ("http://0.0.0.0:11434", "0.0.0.0:11434", ControlMode.WILDCARD),
        ("http://[::]:11434", "[::]:11434", ControlMode.WILDCARD),
    ],
)
@pytest.mark.parametrize("api_up", [True, False])
def test_non_local_endpoint_does_not_touch_local_server(
    tmp_path: Path,
    host: str,
    env_host: str,
    mode: ControlMode,
    api_up: bool,
) -> None:
    identity = _identity()
    _install_proc(tmp_path / "proc", identity)
    (tmp_path / "ollama.pid").write_text(dump_process_state(identity), encoding="utf-8")
    (tmp_path / "ollama.log").write_text("LOCAL-LOG-MARKER", encoding="utf-8")
    popped: list[list[str]] = []
    client = FakeClient(up=api_up)
    manager, calls, killed = _manager(
        tmp_path,
        client,
        {
            SHOW_SYSTEM: _unit(True, True),
            SHOW_USER: _unit(True, True),
            ("journalctl", "-u", "ollama.service", "-n", "100", "--no-pager"): CommandCapture(0, "local-journal", ""),
        },
        settings=_settings(host, env_host),
        popen=lambda argv, **_kwargs: popped.append(argv),
    )
    status = manager.status()
    assert status.mode is mode
    assert status.api_up is api_up
    assert status.host == host
    assert status.can_start is False
    assert status.can_stop is False
    assert status.pid is None
    if api_up:
        assert status.version == "0.32.15"
    else:
        assert status.version is None
        assert status.api_error
    assert host in (status.stop_reason or "")
    for action in (manager.start, manager.stop, manager.restart):
        with pytest.raises(LamaError, match="unavailable|wildcard|remote"):
            action()
    logs = manager.logs()
    assert "LOCAL-LOG-MARKER" not in logs
    assert "local-journal" not in logs
    assert calls == []
    assert killed == []
    assert popped == []
    assert manager.runner.environment()["OLLAMA_HOST"] == env_host


@pytest.mark.parametrize(
    "host",
    [
        "http://127.0.0.1:11434",
        "http://localhost:11434",
        "http://127.0.0.2:11434",
        "http://[::1]:11434",
    ],
)
def test_loopback_uses_the_active_local_backend(tmp_path: Path, host: str) -> None:
    client = FakeClient(up=True)
    manager, calls, killed = _manager(
        tmp_path,
        client,
        {SHOW_SYSTEM: _unit(True, True), SHOW_USER: _unit(True, False)},
        settings=_settings(host, host.removeprefix("http://")),
    )
    status = manager.status()
    assert status.mode is ControlMode.SYSTEMD_SYSTEM
    assert SHOW_SYSTEM in calls
    assert SHOW_USER in calls
    assert manager.stop()
    assert ("systemctl", "stop", "ollama.service") in calls
    assert ("systemctl", "--user", "stop", "ollama.service") not in calls
    assert killed == []


def test_stopped_when_nothing_manages_the_server(tmp_path: Path) -> None:
    manager, calls, _killed = _manager(tmp_path, FakeClient(up=False), _none())
    status = manager.status()
    assert status.mode is ControlMode.STOPPED
    assert status.can_start is True
    assert status.can_stop is False
    assert SHOW_SYSTEM in calls
    assert SHOW_USER in calls


def test_external_api_is_not_stopped(tmp_path: Path) -> None:
    manager, calls, killed = _manager(tmp_path, FakeClient(up=True), _none())
    status = manager.status()
    assert status.mode is ControlMode.EXTERNAL
    assert status.api_up is True
    assert status.version == "0.32.15"
    assert status.can_stop is False
    with pytest.raises(LamaError, match="Another process"):
        manager.stop()
    assert _mutations(calls) == []
    assert killed == []


def test_inactive_system_unit_does_not_hide_active_user_unit(tmp_path: Path) -> None:
    client = FakeClient(up=False)
    manager, calls, killed = _manager(
        tmp_path,
        client,
        {SHOW_SYSTEM: _unit(True, False), SHOW_USER: _unit(True, True)},
    )
    status = manager.status()
    assert status.mode is ControlMode.SYSTEMD_USER
    assert status.can_stop is True
    assert status.can_start is False
    assert manager.stop() == "systemd user ollama service stopped."
    assert ("systemctl", "--user", "stop", "ollama.service") in calls
    assert ("systemctl", "stop", "ollama.service") not in calls
    assert killed == []


def test_active_system_unit_is_used_when_user_is_inactive(tmp_path: Path) -> None:
    client = FakeClient(up=True)
    manager, calls, _killed = _manager(
        tmp_path,
        client,
        {SHOW_SYSTEM: _unit(True, True), SHOW_USER: _unit(True, False)},
    )
    assert manager.status().mode is ControlMode.SYSTEMD_SYSTEM
    assert manager.restart()
    assert ("systemctl", "restart", "ollama.service") in calls
    assert ("systemctl", "--user", "restart", "ollama.service") not in calls
    assert not any(item[0] == "journalctl" and "--user" in item for item in calls)
    (tmp_path / "ollama.log").write_text("LOCAL-LOG-MARKER", encoding="utf-8")
    calls.clear()
    manager, calls, _killed = _manager(
        tmp_path,
        FakeClient(up=True),
        {
            SHOW_SYSTEM: _unit(True, True),
            SHOW_USER: _unit(True, False),
            ("journalctl", "-u", "ollama.service", "-n", "100", "--no-pager"): CommandCapture(0, "system-journal", ""),
        },
    )
    assert manager.logs() == "system-journal"
    assert ("journalctl", "-u", "ollama.service", "-n", "100", "--no-pager") in calls


def test_both_inactive_units_prefer_system_only_as_start_method(tmp_path: Path) -> None:
    client = FakeClient(up=False)

    def executor(argv: list[str]) -> CommandCapture:
        calls.append(tuple(argv))
        if tuple(argv) == SHOW_SYSTEM:
            return _unit(True, False)
        if tuple(argv) == SHOW_USER:
            return _unit(True, False)
        if argv[:3] == ["systemctl", "start", "ollama.service"]:
            client.up = True
            return CommandCapture(0, "", "")
        return CommandCapture(1, "", "not found")

    calls: list[tuple[str, ...]] = []
    manager, _ignored, killed = _manager(tmp_path, client, executor=executor)
    status = manager.status()
    assert status.mode is ControlMode.SYSTEMD_SYSTEM
    assert status.can_start is True
    assert status.can_stop is False
    assert SHOW_SYSTEM in calls
    assert SHOW_USER in calls
    assert "started" in manager.start()
    assert ("systemctl", "start", "ollama.service") in calls
    assert ("systemctl", "--user", "start", "ollama.service") not in calls
    assert killed == []


def test_user_unit_is_the_start_method_when_system_is_absent(tmp_path: Path) -> None:
    client = FakeClient(up=False)

    def executor(argv: list[str]) -> CommandCapture:
        calls.append(tuple(argv))
        if tuple(argv) == SHOW_SYSTEM:
            return _unit(False)
        if tuple(argv) == SHOW_USER:
            return _unit(True, False)
        if argv[:4] == ["systemctl", "--user", "restart", "ollama.service"]:
            client.up = True
            return CommandCapture(0, "", "")
        return CommandCapture(1, "", "not found")

    calls: list[tuple[str, ...]] = []
    manager, _ignored, _killed = _manager(tmp_path, client, executor=executor)
    assert manager.status().mode is ControlMode.SYSTEMD_USER
    assert "restarted" in manager.restart()
    assert ("systemctl", "--user", "restart", "ollama.service") in calls
    assert ("systemctl", "restart", "ollama.service") not in calls


def test_both_active_units_block_destructive_actions(tmp_path: Path) -> None:
    manager, calls, killed = _manager(
        tmp_path,
        FakeClient(up=True),
        {SHOW_SYSTEM: _unit(True, True), SHOW_USER: _unit(True, True)},
    )
    (tmp_path / "ollama.log").write_text("LOCAL-LOG-MARKER", encoding="utf-8")
    status = manager.status()
    assert status.mode is ControlMode.AMBIGUOUS
    assert status.can_start is False
    assert status.can_stop is False
    assert "systemd system" in (status.stop_reason or "")
    assert "systemd user" in (status.stop_reason or "")
    for action in (manager.start, manager.stop, manager.restart):
        with pytest.raises(LamaError, match="ambiguous"):
            action()
    logs = manager.logs()
    assert "LOCAL-LOG-MARKER" not in logs
    assert _mutations(calls) == []
    assert not any(item[0] == "journalctl" for item in calls)
    assert killed == []


def test_inactive_installed_units_do_not_own_external_api(tmp_path: Path) -> None:
    manager, calls, killed = _manager(
        tmp_path,
        FakeClient(up=True),
        {SHOW_SYSTEM: _unit(True, False), SHOW_USER: _unit(True, False)},
    )
    status = manager.status()
    assert status.mode is ControlMode.EXTERNAL
    assert status.can_stop is False
    assert status.can_start is False
    with pytest.raises(LamaError, match="Another process"):
        manager.stop()
    with pytest.raises(LamaError):
        manager.restart()
    assert _mutations(calls) == []
    assert killed == []
    (tmp_path / "ollama.log").write_text("LOCAL-LOG-MARKER", encoding="utf-8")
    assert "LOCAL-LOG-MARKER" not in manager.logs()


def test_owned_process_is_chosen_over_inactive_installed_units(tmp_path: Path) -> None:
    identity = _identity()
    proc = _install_proc(tmp_path / "proc", identity)
    (tmp_path / "ollama.pid").write_text(dump_process_state(identity), encoding="utf-8")

    def killer(pid: int, sig: int) -> None:
        killed.append((pid, sig))
        if sig == signal.SIGTERM:
            shutil.rmtree(proc)

    killed: list[tuple[int, int]] = []
    manager, calls, _ignored = _manager(
        tmp_path,
        FakeClient(up=True),
        {SHOW_SYSTEM: _unit(True, False), SHOW_USER: _unit(True, False)},
        killer=killer,
    )
    status = manager.status()
    assert status.mode is ControlMode.MANAGED
    assert status.pid == identity.pid
    assert status.can_stop is True
    assert "stopped" in manager.stop()
    assert killed == [(identity.pid, signal.SIGTERM)]
    assert _mutations(calls) == []
    assert not (tmp_path / "ollama.pid").exists()


def test_owned_process_plus_active_unit_is_ambiguous(tmp_path: Path) -> None:
    identity = _identity()
    _install_proc(tmp_path / "proc", identity)
    (tmp_path / "ollama.pid").write_text(dump_process_state(identity), encoding="utf-8")
    manager, calls, killed = _manager(
        tmp_path,
        FakeClient(up=True),
        {SHOW_SYSTEM: _unit(True, True), SHOW_USER: _unit(False)},
    )
    assert manager.status().mode is ControlMode.AMBIGUOUS
    with pytest.raises(LamaError, match="ambiguous"):
        manager.stop()
    assert _mutations(calls) == []
    assert killed == []


def test_missing_systemctl_falls_back_to_stopped(tmp_path: Path) -> None:
    calls: list[tuple[str, ...]] = []

    def executor(argv: list[str]) -> CommandCapture:
        calls.append(tuple(argv))
        raise FileNotFoundError(argv[0])

    manager, _ignored, _killed = _manager(tmp_path, FakeClient(up=False), executor=executor)
    status = manager.status()
    assert status.mode is ControlMode.STOPPED
    assert status.can_start is True
    assert SHOW_SYSTEM in calls
    assert SHOW_USER in calls


def test_unavailable_user_bus_keeps_active_system_unit(tmp_path: Path) -> None:
    def executor(argv: list[str]) -> CommandCapture:
        calls.append(tuple(argv))
        if tuple(argv) == SHOW_SYSTEM:
            return _unit(True, True)
        if tuple(argv) == SHOW_USER:
            return CommandCapture(1, "", "Failed to connect to bus")
        return CommandCapture(0, "", "")

    calls: list[tuple[str, ...]] = []
    manager, _ignored, killed = _manager(tmp_path, FakeClient(up=False), executor=executor)
    assert manager.status().mode is ControlMode.SYSTEMD_SYSTEM
    assert manager.stop()
    assert ("systemctl", "stop", "ollama.service") in calls
    assert ("systemctl", "--user", "stop", "ollama.service") not in calls
    assert killed == []


def test_owned_stop_requires_full_identity(tmp_path: Path) -> None:
    identity = _identity()
    proc = _install_proc(tmp_path / "proc", identity, comm="ollama serve")
    (tmp_path / "ollama.pid").write_text(dump_process_state(identity), encoding="utf-8")
    assert is_owned_serve(identity, tmp_path / "proc")

    def killer(pid: int, sig: int) -> None:
        killed.append((pid, sig))
        if sig == signal.SIGTERM:
            shutil.rmtree(proc)

    killed: list[tuple[int, int]] = []
    manager, _calls, _ignored = _manager(tmp_path, FakeClient(up=True), _none(), killer=killer)
    assert manager.status().mode is ControlMode.MANAGED
    assert "lama" in manager.stop()
    assert killed == [(identity.pid, signal.SIGTERM)]
    assert not (tmp_path / "ollama.pid").exists()


@pytest.mark.parametrize(
    ("start_time", "executable", "uid"),
    [
        (101, "/usr/bin/ollama", 1000),
        (100, "/usr/local/bin/ollama", 1000),
        (100, "/usr/bin/ollama", 1001),
    ],
)
def test_reused_or_foreign_process_is_not_killed(
    tmp_path: Path,
    start_time: int,
    executable: str,
    uid: int,
) -> None:
    identity = _identity()
    _install_proc(tmp_path / "proc", identity, start_time=start_time, executable=executable, uid=uid)
    original = dump_process_state(identity)
    (tmp_path / "ollama.pid").write_text(original, encoding="utf-8")
    assert is_owned_serve(identity, tmp_path / "proc") is False
    manager, calls, killed = _manager(tmp_path, FakeClient(up=False), _none())
    status = manager.status()
    assert status.mode is not ControlMode.MANAGED
    assert status.can_stop is False
    with pytest.raises(LamaError):
        manager.stop()
    with pytest.raises(LamaError):
        manager.start()
    assert killed == []
    assert _mutations(calls) == []
    assert (tmp_path / "ollama.pid").read_text(encoding="utf-8") == original


def test_foreign_cmdline_is_not_killed(tmp_path: Path) -> None:
    identity = _identity(pid=7)
    _install_proc(tmp_path / "proc", identity, command=b"/usr/bin/python\0ollama\0serve\0")
    (tmp_path / "ollama.pid").write_text(dump_process_state(identity), encoding="utf-8")
    assert is_owned_serve(identity, tmp_path / "proc") is False
    manager, calls, killed = _manager(tmp_path, FakeClient(up=True), _none())
    assert manager.status().mode is ControlMode.EXTERNAL
    with pytest.raises(LamaError, match="Another process"):
        manager.stop()
    assert killed == []
    assert _mutations(calls) == []


def test_legacy_pid_file_does_not_authorize_kill(tmp_path: Path) -> None:
    identity = _identity()
    _install_proc(tmp_path / "proc", identity)
    original = "4242\n"
    (tmp_path / "ollama.pid").write_text(original, encoding="utf-8")
    manager, calls, killed = _manager(tmp_path, FakeClient(up=True), _none())
    status = manager.status()
    assert status.mode is not ControlMode.MANAGED
    assert status.can_stop is False
    assert status.can_start is False
    with pytest.raises(LamaError, match="only a PID"):
        manager.stop()
    assert killed == []
    assert _mutations(calls) == []
    assert (tmp_path / "ollama.pid").read_text(encoding="utf-8") == original


def test_legacy_pid_file_without_process_does_not_kill(tmp_path: Path) -> None:
    (tmp_path / "ollama.pid").write_text("4242\n", encoding="utf-8")
    manager, _calls, killed = _manager(tmp_path, FakeClient(up=False), _none())
    status = manager.status()
    assert status.mode is ControlMode.STOPPED
    assert status.can_start is True
    assert "only a PID" in (status.stop_reason or "")
    assert killed == []


def test_corrupt_state_file_does_not_traceback_or_kill(tmp_path: Path) -> None:
    original = "{не json\n"
    (tmp_path / "ollama.pid").write_text(original, encoding="utf-8")
    manager, calls, killed = _manager(tmp_path, FakeClient(up=False), _none())
    status = manager.status()
    assert status.mode is not ControlMode.MANAGED
    assert "damaged" in (status.stop_reason or "")
    with pytest.raises(LamaError):
        manager.stop()
    assert killed == []
    assert _mutations(calls) == []
    assert (tmp_path / "ollama.pid").read_text(encoding="utf-8") == original


def test_unreadable_proc_does_not_kill(tmp_path: Path) -> None:
    identity = _identity()
    proc = _install_proc(tmp_path / "proc", identity)
    (proc / "stat").unlink()
    original = dump_process_state(identity)
    (tmp_path / "ollama.pid").write_text(original, encoding="utf-8")
    manager, _calls, killed = _manager(tmp_path, FakeClient(up=False), _none())
    assert manager.status().can_stop is False
    with pytest.raises(LamaError):
        manager.stop()
    assert killed == []
    assert (tmp_path / "ollama.pid").read_text(encoding="utf-8") == original


def test_process_gone_between_status_and_signal_is_not_killed(tmp_path: Path) -> None:
    identity = _identity()
    proc = _install_proc(tmp_path / "proc", identity)
    (tmp_path / "ollama.pid").write_text(dump_process_state(identity), encoding="utf-8")
    manager, _calls, killed = _manager(tmp_path, FakeClient(up=True), _none())

    class Dropping(ServerManager):
        def status(self):  # type: ignore[override]
            current = super().status()
            shutil.rmtree(proc)
            return current

    manager.__class__ = Dropping
    message = manager.stop()
    assert "No signal was sent" in message
    assert killed == []
    assert not (tmp_path / "ollama.pid").exists()


def test_identity_change_before_sigkill_spares_the_new_process(tmp_path: Path) -> None:
    identity = _identity()
    _install_proc(tmp_path / "proc", identity)
    (tmp_path / "ollama.pid").write_text(dump_process_state(identity), encoding="utf-8")

    def sleeper(_delay: float) -> None:
        _install_proc(tmp_path / "proc", identity, start_time=identity.start_time + 50)

    manager, _calls, killed = _manager(tmp_path, FakeClient(up=True), _none(), sleeper=sleeper)
    with pytest.raises(LamaError, match="SIGKILL"):
        manager.stop()
    assert killed == [(identity.pid, signal.SIGTERM)]
    assert not (tmp_path / "ollama.pid").exists()


def test_sigkill_is_sent_only_while_identity_matches_and_file_stays_if_process_survives(tmp_path: Path) -> None:
    identity = _identity()
    _install_proc(tmp_path / "proc", identity)
    original = dump_process_state(identity)
    (tmp_path / "ollama.pid").write_text(original, encoding="utf-8")
    manager, _calls, killed = _manager(tmp_path, FakeClient(up=True), _none())
    with pytest.raises(LamaError, match="kept"):
        manager.stop()
    assert killed == [(identity.pid, signal.SIGTERM), (identity.pid, signal.SIGKILL)]
    assert (tmp_path / "ollama.pid").read_text(encoding="utf-8") == original


def test_sigkill_clears_state_only_after_the_process_exits(tmp_path: Path) -> None:
    identity = _identity()
    proc = _install_proc(tmp_path / "proc", identity)
    (tmp_path / "ollama.pid").write_text(dump_process_state(identity), encoding="utf-8")

    def killer(pid: int, sig: int) -> None:
        killed.append((pid, sig))
        if sig == signal.SIGKILL:
            shutil.rmtree(proc)

    killed: list[tuple[int, int]] = []
    manager, _calls, _ignored = _manager(tmp_path, FakeClient(up=True), _none(), killer=killer)
    assert "stopped" in manager.stop()
    assert killed == [(identity.pid, signal.SIGTERM), (identity.pid, signal.SIGKILL)]
    assert not (tmp_path / "ollama.pid").exists()


def test_other_endpoint_state_is_not_killed_or_replaced(tmp_path: Path) -> None:
    identity = _identity(endpoint="http://127.0.0.1:9")
    _install_proc(tmp_path / "proc", identity)
    original = dump_process_state(identity)
    (tmp_path / "ollama.pid").write_text(original, encoding="utf-8")
    popped: list[list[str]] = []
    manager, calls, killed = _manager(
        tmp_path,
        FakeClient(up=False),
        _none(),
        popen=lambda argv, **_kwargs: popped.append(argv),
    )
    status = manager.status()
    assert status.can_start is False
    assert status.can_stop is False
    assert "127.0.0.1:9" in (status.stop_reason or "")
    with pytest.raises(LamaError):
        manager.start()
    assert killed == []
    assert popped == []
    assert _mutations(calls) == []
    assert (tmp_path / "ollama.pid").read_text(encoding="utf-8") == original


def test_managed_start_records_identity(tmp_path: Path) -> None:
    identity = _identity(pid=77)
    _install_proc(tmp_path / "proc", identity)
    client = FakeClient(up=False)
    popped: list[list[str]] = []

    def popen(argv: list[str], **_kwargs: object) -> FakePopen:
        popped.append(argv)
        client.up = True
        return FakePopen(identity.pid, [])

    manager, calls, killed = _manager(tmp_path, client, _none(), popen=popen)
    assert "ollama serve" in manager.start()
    saved = json.loads((tmp_path / "ollama.pid").read_text(encoding="utf-8"))
    assert saved == {
        "version": 1,
        "pid": 77,
        "start_time": identity.start_time,
        "executable": identity.executable,
        "uid": identity.uid,
        "endpoint": LOCAL_HOST,
    }
    assert popped == [["/usr/bin/ollama", "serve"]]
    assert killed == []
    assert _mutations(calls) == []


def test_managed_start_without_proc_identity_does_not_keep_a_killable_record(tmp_path: Path) -> None:
    child: list[int] = []

    def popen(_argv: list[str], **_kwargs: object) -> FakePopen:
        return FakePopen(77, child)

    manager, _calls, killed = _manager(tmp_path, FakeClient(up=False), _none(), popen=popen)
    with pytest.raises(LamaError, match="identity"):
        manager.start()
    assert child == [77]
    assert killed == []
    assert not (tmp_path / "ollama.pid").exists()


def test_remote_model_stop_uses_selected_host_not_local_server(tmp_path: Path) -> None:
    settings = _settings("http://remote.example:11434", "remote.example:11434")
    runner = OllamaRunner(settings, binary="/usr/bin/ollama")
    seen: dict[str, object] = {}

    def execute(argv: list[str], inherit: bool = False) -> CommandResult:
        seen["argv"] = argv
        seen["host"] = runner.environment()["OLLAMA_HOST"]
        return CommandResult(0, "", "", argv)

    runner.execute = execute  # type: ignore[method-assign]
    assert stop_model(runner, FakeClient(up=True), "gemma4:12b") == "cli"  # type: ignore[arg-type]
    assert seen == {"argv": ["/usr/bin/ollama", "stop", "gemma4:12b"], "host": "remote.example:11434"}
    manager, calls, killed = _manager(
        tmp_path,
        FakeClient(up=True),
        {SHOW_SYSTEM: _unit(True, True)},
        settings=settings,
    )
    with pytest.raises(LamaError):
        manager.stop()
    assert calls == []
    assert killed == []


def test_remote_status_panel_does_not_claim_a_local_backend(monkeypatch) -> None:
    buffer = StringIO()
    monkeypatch.setattr(display, "console", Console(file=buffer, force_terminal=False, width=200))
    display.show_server(
        ServerStatus(
            api_up=True,
            version="0.32.15",
            host="http://remote.example:11434",
            mode=ControlMode.REMOTE,
            can_start=False,
            can_stop=False,
            stop_reason="Выбран удалённый адрес http://remote.example:11434.",
            pid=None,
        )
    )
    text = buffer.getvalue()
    assert "remote API" in text
    assert "remote.example:11434" in text
    assert "0.32.15" in text
    assert "systemd" not in text
