import pytest

from lama.config import Settings
from lama.errors import LamaError, OllamaCommandError, OllamaUnavailable, UserCancelled
from lama.ollama.types import InstalledModel, ModelShow
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
