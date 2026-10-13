import os
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from lama.cli import app, typer_app
from lama.errors import LamaError, OllamaUnavailable
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

    monkeypatch.setattr("lama.ui.actions.build_client", lambda _settings: Client())
    monkeypatch.setattr("lama.ui.actions.build_runner", lambda _settings: object())
    result = runner.invoke(typer_app, ["list"])
    assert result.exit_code == 1
    assert "unreachable" in result.stdout


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
