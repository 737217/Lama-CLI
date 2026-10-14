import pytest

from lama.config import Settings
from lama.errors import LamaError, OllamaCommandError, OllamaUnavailable, UserCancelled
from lama.ollama.discover import OllamaInstall
from lama.ollama.types import InstalledModel, ModelShow
from lama.server.manager import ControlMode, ServerStatus
from lama.services import stop_model
from lama.ui.actions import Session, configure_flow, delete_flow, run_flow
from lama.ui.app import run_main_menu


def _settings() -> Settings:
    return Settings(host="http://127.0.0.1:11434", env_host="127.0.0.1:11434")


def _model() -> InstalledModel:
    return InstalledModel(name="gemma4:12b", size=100, modified_at=None)


class Scripted:
    def __init__(self, answers: list[object]) -> None:
        self.answers = list(answers)

    def _next(self) -> object:
        return self.answers.pop(0)

    def select(self, message: str, choices: list[tuple[str, str]], search: bool = False) -> object:
        return self._next()

    def text(self, message: str, default: str = "", hint: str = "") -> object:
        return self._next()

    def confirm(self, message: str, default: bool = False) -> object:
        return self._next()

    def path(self, message: str) -> object:
        return self._next()


class Runner:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def run_chat(self, model: str, keepalive: str | None = None, think: str | None = None) -> None:
        self.calls.append(model)

    def stop(self, model: str) -> None:
        raise OllamaCommandError("cli failed", returncode=1, argv=["ollama", "stop", model])


class Models:
    def __init__(self, models: list[InstalledModel] | None = None) -> None:
        self.models = models if models is not None else [_model()]
        self.deleted: list[str] = []

    def version(self) -> str:
        return "0.32.15"

    def list_models(self) -> list[InstalledModel]:
        return self.models

    def delete(self, model: str) -> None:
        self.deleted.append(model)

    def close(self) -> None:
        return None


def test_back_from_model_list_does_not_start_a_chat() -> None:
    runner = Runner()
    session = Session(_settings(), Scripted(["back"]), Models(), runner)  # type: ignore[arg-type]
    run_flow(session)
    assert runner.calls == []


def test_back_on_mode_returns_to_model_list() -> None:
    runner = Runner()
    seen: list[list[tuple[str, str]]] = []

    class Recording(Scripted):
        def select(self, message: str, choices: list[tuple[str, str]], search: bool = False) -> object:
            seen.append(choices)
            return self._next()

    session = Session(
        _settings(),
        Recording(["gemma4:12b", "back", "back"]),  # type: ignore[arg-type]
        Models(),
        runner,  # type: ignore[arg-type]
    )
    run_flow(session)
    assert runner.calls == []
    assert seen[0][-1] == ("Back", "back")
    assert seen[1][-1] == ("Back", "back")
    assert seen[2][-1] == ("Back", "back")


def test_empty_model_list_does_not_prompt() -> None:
    class Boom(Scripted):
        def select(self, message: str, choices: list[tuple[str, str]], search: bool = False) -> object:
            raise AssertionError("пустой список не должен открывать выбор")

    session = Session(_settings(), Boom([]), Models([]), Runner())  # type: ignore[arg-type]
    with pytest.raises(LamaError, match="No models are installed"):
        run_flow(session)


def test_declined_delete_does_not_call_api() -> None:
    client = Models()
    session = Session(_settings(), Scripted([False]), client, Runner())  # type: ignore[arg-type]
    with pytest.raises(UserCancelled):
        delete_flow(session, "gemma4:12b")
    assert client.deleted == []


def test_confirmed_delete_calls_api() -> None:
    client = Models()
    session = Session(_settings(), Scripted([True]), client, Runner())  # type: ignore[arg-type]
    delete_flow(session, "gemma4:12b")
    assert client.deleted == ["gemma4:12b"]


def test_confirmation_is_a_yes_no_menu() -> None:
    from lama.ui.prompts import QuestionaryPrompter

    prompter = QuestionaryPrompter()
    seen: dict[str, object] = {}

    def select(message: str, choices: list[tuple[str, str]], search: bool = False) -> str:
        seen["message"] = message
        seen["choices"] = choices
        seen["search"] = search
        return "yes"

    prompter.select = select  # type: ignore[method-assign]
    assert prompter.confirm("Delete gemma4:12b?", default=False) is True
    assert seen["choices"] == [("No", "no"), ("Yes", "yes")]
    assert prompter.confirm("Create it?", default=True) is True
    assert seen["choices"] == [("Yes", "yes"), ("No", "no")]


def test_declined_overwrite_does_not_create_model() -> None:
    class Client(Models):
        def show(self, model: str) -> ModelShow:
            return ModelShow()

    class Creating(Runner):
        def create(self, model: str, modelfile: object) -> None:
            self.calls.append(f"create:{model}")

    answers: list[object] = ["gemma4:12b", *([""] * 9), "", "", True, False]
    runner = Creating()
    session = Session(_settings(), Scripted(answers), Client(), runner)  # type: ignore[arg-type]
    with pytest.raises(UserCancelled):
        configure_flow(session, "gemma4:12b")
    assert runner.calls == []


def test_stop_falls_back_to_api_unload() -> None:
    class Api:
        def __init__(self) -> None:
            self.unloaded = False

        def unload(self, model: str) -> None:
            self.unloaded = True

    api = Api()
    assert stop_model(Runner(), api, "gemma4:12b") == "api"  # type: ignore[arg-type]
    assert api.unloaded is True


def test_stop_reports_both_failures() -> None:
    class Api:
        def unload(self, model: str) -> None:
            raise OllamaUnavailable("Сервер Ollama недоступен.")

    with pytest.raises(LamaError, match="Unload via the API"):
        stop_model(Runner(), Api(), "gemma4:12b")  # type: ignore[arg-type]


class _Gate:
    def __init__(self) -> None:
        self.up = False

    def version(self) -> str:
        if not self.up:
            raise OllamaUnavailable("Ollama is unreachable.")
        return "0.32.15"

    def list_models(self) -> list[InstalledModel]:
        return []

    def close(self) -> None:
        return None


class _MenuManager:
    def __init__(self, client: _Gate, started: list[str | None], *, can_start: bool) -> None:
        self.client = client
        self.started = started
        self._can_start = can_start

    def status(self) -> ServerStatus:
        return ServerStatus(
            api_up=self.client.up,
            version="0.32.15" if self.client.up else None,
            host="http://127.0.0.1:11434",
            mode=ControlMode.STOPPED if self._can_start else ControlMode.REMOTE,
            can_start=self._can_start and not self.client.up,
            can_stop=False,
            stop_reason=None,
            pid=None,
        )

    def start(self, *, prefer_binary: str | None = None) -> str:
        self.started.append(prefer_binary)
        self.client.up = True
        return "Started ollama serve."

    def service_binary(self, _mode: ControlMode) -> str | None:
        return None


class _Binary:
    def use_binary(self, path: str) -> None:
        return None


def test_menu_starts_a_stopped_local_server(monkeypatch) -> None:
    client = _Gate()
    started: list[str | None] = []
    monkeypatch.setattr(
        "lama.ui.actions.ServerManager",
        lambda *_args, **_kwargs: _MenuManager(client, started, can_start=True),
    )
    monkeypatch.setattr(
        "lama.ui.actions.discover_ollama_installs",
        lambda: [OllamaInstall("/usr/local/bin/ollama", "0.11.4")],
    )
    monkeypatch.setattr("lama.ui.actions.build_client", lambda _settings: client)
    monkeypatch.setattr("lama.ui.actions.build_runner", lambda _settings: _Binary())
    run_main_menu(_settings(), prompter=Scripted(["exit"]))  # type: ignore[arg-type]
    assert started == [None]
    assert client.up is True


def test_menu_does_not_start_a_remote_server(monkeypatch) -> None:
    client = _Gate()
    started: list[str | None] = []
    monkeypatch.setattr(
        "lama.ui.actions.ServerManager",
        lambda *_args, **_kwargs: _MenuManager(client, started, can_start=False),
    )
    monkeypatch.setattr("lama.ui.actions.build_client", lambda _settings: client)
    monkeypatch.setattr("lama.ui.actions.build_runner", lambda _settings: _Binary())
    run_main_menu(_settings(), prompter=Scripted(["exit"]))  # type: ignore[arg-type]
    assert started == []
    assert client.up is False


def test_menu_leaves_the_server_stopped_when_the_choice_is_cancelled(monkeypatch) -> None:
    client = _Gate()
    started: list[str | None] = []
    monkeypatch.setattr("lama.ui.actions._prompt_available", lambda: True)
    monkeypatch.setattr(
        "lama.ui.actions.ServerManager",
        lambda *_args, **_kwargs: _MenuManager(client, started, can_start=True),
    )
    monkeypatch.setattr(
        "lama.ui.actions.discover_ollama_installs",
        lambda: [
            OllamaInstall("/var/tmp/lama-test-unit/ollama", "0.6.2", "0.0.0.0:11434"),
            OllamaInstall("/var/tmp/lama-test-other/ollama", "0.11.4", "127.0.0.1:11435"),
        ],
    )
    monkeypatch.setattr("lama.ui.actions.build_client", lambda _settings: client)
    monkeypatch.setattr("lama.ui.actions.build_runner", lambda _settings: _Binary())
    run_main_menu(_settings(), prompter=Scripted([None, "exit"]))  # type: ignore[arg-type]
    assert started == []
    assert client.up is False


def test_menu_exit_draws_header(monkeypatch) -> None:
    class Client:
        def version(self) -> str:
            return "0.32.15"

        def list_models(self) -> list[InstalledModel]:
            return []

        def close(self) -> None:
            return None

    monkeypatch.setattr("lama.ui.actions.build_client", lambda _settings: Client())
    monkeypatch.setattr("lama.ui.actions.build_runner", lambda _settings: Runner())
    run_main_menu(_settings(), prompter=Scripted(["exit"]))  # type: ignore[arg-type]
