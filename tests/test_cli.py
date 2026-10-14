import os
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from lama.cli import app, typer_app
from lama.errors import LamaError, OllamaUnavailable
from lama.ollama.discover import OllamaInstall
from lama.server.manager import ControlMode, ServerStatus

runner = CliRunner()


def test_help_lists_commands() -> None:
    result = runner.invoke(typer_app, ["--help"])
    assert result.exit_code == 0
    for name in ("run", "stop", "pull", "list", "ps", "create", "show", "rm", "server", "monitor"):
        assert name in result.stdout


def test_list_empty(monkeypatch) -> None:
    class Client:
        def version(self) -> str:
            return "0.32.15"

        def list_models(self) -> list[object]:
            return []

        def close(self) -> None:
            return None

    monkeypatch.setattr("lama.ui.actions.build_client", lambda _settings: Client())
    monkeypatch.setattr("lama.ui.actions.build_runner", lambda _settings: object())
    result = runner.invoke(typer_app, ["list"])
    assert result.exit_code == 0
    assert "No models are installed." in result.stdout


def test_ps_empty(monkeypatch) -> None:
    class Client:
        def version(self) -> str:
            return "0.32.15"

        def list_running(self) -> list[object]:
            return []

        def close(self) -> None:
            return None

    monkeypatch.setattr("lama.ui.actions.build_client", lambda _settings: Client())
    monkeypatch.setattr("lama.ui.actions.build_runner", lambda _settings: object())
    result = runner.invoke(typer_app, ["ps"])
    assert result.exit_code == 0
    assert "No models are loaded in memory." in result.stdout


def test_list_when_server_is_down(monkeypatch) -> None:
    class Client:
        def version(self) -> str:
            raise OllamaUnavailable("Ollama is unreachable at http://127.0.0.1:11434.")

        def close(self) -> None:
            return None

    class Manager:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            return None

        def status(self) -> ServerStatus:
            return ServerStatus(
                api_up=False,
                version=None,
                host="http://127.0.0.1:11434",
                mode=ControlMode.REMOTE,
                can_start=False,
                can_stop=False,
                stop_reason="remote",
                pid=None,
            )

    monkeypatch.setattr("lama.ui.actions.ServerManager", Manager)
    monkeypatch.setattr("lama.ui.actions.build_client", lambda _settings: Client())
    monkeypatch.setattr("lama.ui.actions.build_runner", lambda _settings: object())
    result = runner.invoke(typer_app, ["list"])
    assert result.exit_code == 1
    assert "unreachable" in result.stdout


class _DownClient:
    def __init__(self) -> None:
        self.up = False

    def version(self) -> str:
        if not self.up:
            raise OllamaUnavailable("Ollama is unreachable at http://127.0.0.1:11434.")
        return "0.32.15"

    def list_models(self) -> list[object]:
        return []

    def close(self) -> None:
        return None


class _StartManager:
    def __init__(self, client: _DownClient, started: list[str | None]) -> None:
        self.client = client
        self.started = started

    def status(self) -> ServerStatus:
        return ServerStatus(
            api_up=self.client.up,
            version="0.32.15" if self.client.up else None,
            host="http://127.0.0.1:11434",
            mode=ControlMode.STOPPED,
            can_start=not self.client.up,
            can_stop=False,
            stop_reason="The server is not running.",
            pid=None,
        )

    def start(self, *, prefer_binary: str | None = None) -> str:
        self.started.append(prefer_binary)
        self.client.up = True
        return "Started ollama serve."

    def service_binary(self, _mode: ControlMode) -> str | None:
        return None

    def stop_other_than(self, _executable: str) -> None:
        return None


class _BinaryRunner:
    def __init__(self) -> None:
        self.path: str | None = None

    def use_binary(self, path: str) -> None:
        self.path = path


def test_list_starts_a_stopped_local_server(monkeypatch) -> None:
    client = _DownClient()
    started: list[str | None] = []
    monkeypatch.setattr("lama.ui.actions.ServerManager", lambda *_args, **_kwargs: _StartManager(client, started))
    monkeypatch.setattr(
        "lama.ui.actions.discover_ollama_installs",
        lambda: [OllamaInstall("/usr/local/bin/ollama", "0.11.4")],
    )
    monkeypatch.setattr("lama.ui.actions.build_client", lambda _settings: client)
    monkeypatch.setattr("lama.ui.actions.build_runner", lambda _settings: _BinaryRunner())
    result = runner.invoke(typer_app, ["list"])
    assert result.exit_code == 0
    assert started == [None]
    assert "Started ollama serve." in result.stdout
    assert "No models are installed." in result.stdout


def test_list_asks_which_ollama_to_start(monkeypatch, tmp_path: Path) -> None:
    system = tmp_path / "usr" / "bin" / "ollama"
    local = tmp_path / "usr" / "local" / "bin" / "ollama"
    system.parent.mkdir(parents=True)
    local.parent.mkdir(parents=True)
    system.write_text("system", encoding="utf-8")
    local.write_text("local", encoding="utf-8")
    client = _DownClient()
    started: list[str | None] = []
    seen: list[str] = []

    class Prompter:
        def select(self, message: str, choices: list[tuple[str, str]], search: bool = False) -> str:
            assert message == "Which Ollama should be started?"
            seen.extend(title for title, _value in choices)
            return str(local)

    monkeypatch.setattr("lama.ui.actions._prompt_available", lambda: True)
    monkeypatch.setattr("lama.ui.actions.prompts.prompter", Prompter())
    monkeypatch.setattr(
        "lama.ui.actions.discover_ollama_installs",
        lambda: [
            OllamaInstall(str(system), "0.6.2", "0.0.0.0:11434"),
            OllamaInstall(str(local), "0.11.4", "127.0.0.1:11435"),
        ],
    )
    monkeypatch.setattr("lama.ui.actions.save_settings", lambda _settings: None)
    monkeypatch.setattr("lama.ui.actions.running_serve_executables", lambda _root: [])

    class Manager(_StartManager):
        def service_binary(self, _mode: ControlMode) -> str | None:
            return os.path.realpath(system)

    monkeypatch.setattr("lama.ui.actions.ServerManager", lambda *_args, **_kwargs: Manager(client, started))
    monkeypatch.setattr("lama.ui.actions.build_client", lambda _settings: client)
    binary = _BinaryRunner()
    monkeypatch.setattr("lama.ui.actions.build_runner", lambda _settings: binary)
    result = runner.invoke(typer_app, ["list"])
    assert result.exit_code == 0
    assert started == [str(local)]
    assert binary.path == str(local)
    assert any("0.6.2" in title and "0.0.0.0:11434" in title for title in seen)
    assert any("0.11.4" in title and "127.0.0.1:11435" in title for title in seen)


def test_list_without_a_terminal_does_not_guess_among_versions(monkeypatch, tmp_path: Path) -> None:
    first = tmp_path / "a" / "ollama"
    second = tmp_path / "b" / "ollama"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text("a", encoding="utf-8")
    second.write_text("b", encoding="utf-8")
    client = _DownClient()
    started: list[str | None] = []
    monkeypatch.setattr("lama.ui.actions._prompt_available", lambda: False)
    monkeypatch.setattr("lama.ui.actions.ServerManager", lambda *_args, **_kwargs: _StartManager(client, started))
    monkeypatch.setattr(
        "lama.ui.actions.discover_ollama_installs",
        lambda: [OllamaInstall(str(first), "0.6.2"), OllamaInstall(str(second), "0.11.4")],
    )
    monkeypatch.setattr("lama.ui.actions.build_client", lambda _settings: client)
    monkeypatch.setattr("lama.ui.actions.build_runner", lambda _settings: _BinaryRunner())
    result = runner.invoke(typer_app, ["list"])
    assert result.exit_code == 1
    assert "Several Ollama versions" in result.stdout
    assert started == []


def test_server_status(monkeypatch) -> None:
    class Manager:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            return None

        def status(self) -> ServerStatus:
            return ServerStatus(
                api_up=False,
                version=None,
                host="http://127.0.0.1:11434",
                mode=ControlMode.STOPPED,
                can_start=True,
                can_stop=False,
                stop_reason="Сервер не запущен.",
                pid=None,
                api_error="Сервер Ollama недоступен.",
            )

    class Client:
        def close(self) -> None:
            return None

    monkeypatch.setattr("lama.ui.actions.ServerManager", Manager)
    monkeypatch.setattr("lama.ui.actions.build_client", lambda _settings: Client())
    monkeypatch.setattr("lama.ui.actions.build_runner", lambda _settings: object())
    result = runner.invoke(typer_app, ["server", "status"])
    assert result.exit_code == 0
    assert "Stopped" in result.stdout
    assert "127.0.0.1:11434" in result.stdout


def _corrupt_config(monkeypatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.delenv("LAMA_DEBUG", raising=False)
    path = tmp_path / "lama" / "config.toml"
    path.parent.mkdir()
    path.write_text("host = [\n", encoding="utf-8")
    return path


def _compact(text: str) -> str:
    ignored = "│╭╮╰╯─┬┴┼"
    return "".join(char for char in text if not char.isspace() and char not in ignored)


def test_list_reports_corrupt_config_without_traceback(monkeypatch, tmp_path: Path) -> None:
    path = _corrupt_config(monkeypatch, tmp_path)
    before = path.read_bytes()

    def fail_client(_settings: object) -> None:
        raise AssertionError("клиент не должен создаваться при ошибочной конфигурации")

    monkeypatch.setattr("lama.ui.actions.build_client", fail_client)
    result = runner.invoke(typer_app, ["--host", "127.0.0.1:9", "list"])
    output = result.stdout + result.stderr
    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)
    if isinstance(result.exception, SystemExit):
        assert result.exception.code == 1
    assert "Traceback" not in output
    assert "TOMLDecodeError" not in output
    assert "TOML syntax error" in output
    assert _compact(str(path)) in _compact(output)
    assert "No models are installed." not in output
    assert path.read_bytes() == before


def test_debug_config_error_is_reraised(monkeypatch, tmp_path: Path) -> None:
    _corrupt_config(monkeypatch, tmp_path)
    monkeypatch.setenv("LAMA_DEBUG", "1")
    monkeypatch.setattr(sys, "argv", ["lama", "list"])
    with pytest.raises(LamaError, match="TOML syntax error"):
        app()


def test_module_entry_exits_nonzero_without_traceback(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "lama" / "config.toml"
    path.parent.mkdir()
    path.write_text("host = [\n", encoding="utf-8")
    env = os.environ.copy()
    env["XDG_CONFIG_HOME"] = str(tmp_path)
    env["COLUMNS"] = "240"
    env.pop("OLLAMA_HOST", None)
    env.pop("LAMA_DEBUG", None)
    completed = subprocess.run(
        [sys.executable, "-m", "lama", "list"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode != 0
    assert "Traceback" not in output
    assert "TOMLDecodeError" not in output
    assert "TOML syntax error" in output
    assert _compact(str(path)) in _compact(output)
    assert path.read_text(encoding="utf-8") == "host = [\n"
