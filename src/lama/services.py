"""Операции, общие для меню и прямых команд."""

from __future__ import annotations

import tempfile
from pathlib import Path

from lama.errors import LamaError, OllamaCommandError, OllamaUnavailable
from lama.i18n import t
from lama.ollama.client import OllamaClient, ProgressCallback
from lama.ollama.modelfile import render_modelfile
from lama.ollama.runner import OllamaRunner
from lama.validation import validate_model_name


def ensure_server(client: OllamaClient) -> str:
    try:
        return client.version()
    except OllamaUnavailable as exc:
        raise OllamaUnavailable(
            t("{reason} Start the server from the Server menu or with ollama serve.", reason=exc)
        ) from exc


def stop_model(runner: OllamaRunner, client: OllamaClient, model: str) -> str:
    """Остановить модель через `ollama stop`, а при неудаче — через API keep_alive=0."""
    name = validate_model_name(model)
    cli_error: LamaError | None = None
    try:
        runner.stop(name)
        return "cli"
    except (OllamaCommandError, OllamaUnavailable) as exc:
        cli_error = exc
    try:
        client.unload(name)
    except LamaError as api_error:
        raise LamaError(
            t("{cli_error} Unload via the API also failed: {api_error}", cli_error=cli_error, api_error=api_error)
        ) from api_error
    return "api"


def pull_model(client: OllamaClient, model: str, on_progress: ProgressCallback | None = None) -> str:
    name = validate_model_name(model)
    ensure_server(client)
    client.pull(name, on_progress=on_progress)
    return name


def create_from_file(runner: OllamaRunner, name: str, modelfile: Path) -> None:
    model = validate_model_name(name)
    runner.create(model, modelfile)


def create_from_spec(
    runner: OllamaRunner,
    name: str,
    base: str,
    parameters: dict[str, int | float],
    system: str | None,
) -> str:
    model = validate_model_name(name)
    text = render_modelfile(base, parameters, system)
    with tempfile.TemporaryDirectory(prefix="lama-") as directory:
        path = Path(directory) / "Modelfile"
        path.write_text(text, encoding="utf-8")
        runner.create(model, path)
    return text


def supported_commands(help_text: str) -> dict[str, bool]:
    wanted = (
        "serve",
        "create",
        "show",
        "run",
        "stop",
        "pull",
        "push",
        "list",
        "ps",
        "cp",
        "rm",
        "signin",
    )
    tokens = set(help_text.split())
    return {name: name in tokens for name in wanted}
