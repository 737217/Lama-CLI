"""Интерактивные сценарии. Меню спрашивает, операции выполняют клиент и runner."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from rich.text import Text

from lama.config import Settings, remember_keep_alive, save_settings
from lama.context import build_client, build_runner
from lama.errors import LamaError, UserCancelled
from lama.formatting import format_bytes
from lama.i18n import set_language, t
from lama.monitor.resources import collect_snapshot
from lama.ollama.client import OllamaClient
from lama.ollama.modelfile import PARAMETER_ORDER, render_modelfile
from lama.ollama.runner import OllamaRunner
from lama.ollama.types import InstalledModel, PullEvent, RunningModel
from lama.server.manager import ServerManager
from lama.services import (
    create_from_file,
    create_from_spec,
    ensure_server,
    pull_model,
    stop_model,
    supported_commands,
)
from lama.ui import prompts
from lama.ui.display import (
    console,
    render_snapshot,
    show_error,
    show_model_info,
    show_models,
    show_running,
    show_server,
    show_success,
    show_text_block,
)
from lama.ui.monitor_view import run_monitor
from lama.ui.prompts import QuestionaryPrompter
from lama.validation import (
    PARAMETER_FIELDS,
    is_library_name,
    is_pushable_name,
    normalize_host,
    parse_parameter,
    validate_gguf_path,
    validate_keep_alive,
    validate_model_name,
    validate_modelfile_path,
    validate_system_prompt,
)

_THINK_VALUES = {"true", "false", "high", "medium", "low"}
_LIBRARY_URL = "https://ollama.com/library"


@dataclass
class Session:
    settings: Settings
    prompter: QuestionaryPrompter
    client: OllamaClient
    runner: OllamaRunner


def open_session(
    settings: Settings,
    *,
    prompter: QuestionaryPrompter | None = None,
    client: OllamaClient | None = None,
    runner: OllamaRunner | None = None,
) -> Session:
    return Session(
        settings=settings,
        prompter=prompter or prompts.prompter,
        client=client or build_client(settings),
        runner=runner or build_runner(settings),
    )


def pubkey_path() -> Path:
    return Path.home() / ".ollama" / "id_ed25519.pub"


def run_flow(session: Session, model: str | None = None, *, direct_chat: bool = False) -> bool:
    ensure_server(session.client)
    picked_here = model is None
    if model is None:
        model = _choose_installed(session, t("Choose a model"))
        if model is None:
            return False
    else:
        model = validate_model_name(model)
    if direct_chat:
        _start_chat(session, model, keepalive=session.settings.keep_alive.get(model))
        return True
    while True:
        mode = _menu(
            session,
            t("How to run"),
            [
                (t("Interactive chat"), "chat"),
                (t("Load into memory without a chat"), "load"),
                (t("Run with custom settings"), "custom"),
            ],
        )
        if mode is not None:
            break
        if not picked_here:
            return False
        model = _choose_installed(session, t("Choose a model"))
        if model is None:
            return False
    if mode == "chat":
        _start_chat(session, model, keepalive=session.settings.keep_alive.get(model))
    elif mode == "load":
        _load_model(session, model)
    else:
        _run_custom(session, model)
    return True


def stop_flow(session: Session, model: str | None = None) -> None:
    ensure_server(session.client)
    running = session.client.list_running()
    if model is not None:
        target = resolve_running(running, validate_model_name(model))
        channel = stop_model(session.runner, session.client, target)
        show_success(t("Model {name} unloaded ({channel}).", name=target, channel=_stop_channel(channel)))
        return
    if not running:
        raise LamaError(t("No models are loaded in memory."))
    choices = [(_running_title(item), item.name) for item in running]
    choices.append((t("Stop all"), "__all__"))
    picked = _menu(session, t("Loaded models"), choices, search=True)
    if picked is None:
        return
    if picked == "__all__":
        _need_confirm(session.prompter.confirm(t("Stop every loaded model?"), default=False))
        for item in running:
            stop_model(session.runner, session.client, item.name)
        show_success(t("All loaded models were unloaded."))
        return
    channel = stop_model(session.runner, session.client, picked)
    show_success(t("Model {name} unloaded ({channel}).", name=picked, channel=_stop_channel(channel)))


def pull_flow(session: Session, model: str | None = None) -> None:
    if model is None:
        model = _ask_model_to_pull(session)
        if model is None:
            return
    name = validate_model_name(model)
    ensure_server(session.client)
    tracker = _PullProgress()
    try:
        name = pull_model(session.client, name, on_progress=tracker.handle)
    finally:
        tracker.close()
    show_success(t("Model {name} downloaded.", name=name))


def create_flow(session: Session) -> None:
    ensure_server(session.client)
    while True:
        source = _menu(
            session,
            t("Source"),
            [
                (t("Installed model"), "model"),
                (t("Local GGUF"), "gguf"),
                (t("Existing Modelfile"), "file"),
            ],
        )
        if source is None:
            return
        name = validate_model_name(
            _need(session.prompter.text(t("Name of the new model"), hint=t("For example my-qwen")))
        )
        if source == "file":
            path = validate_modelfile_path(_need(session.prompter.path(t("Path to the Modelfile"))))
            _confirm_overwrite(session, name)
            create_from_file(session.runner, name, path)
            _store_optional_keep_alive(session, name)
            show_success(t("Model {name} was created from {path}.", name=name, path=path))
            return
        if source == "model":
            base = _choose_installed(session, t("Base model"))
            if base is None:
                continue
        else:
            base = str(validate_gguf_path(_need(session.prompter.path(t("Path to the GGUF file")))))
        break
    parameters = _ask_parameters(session, current=None)
    system = _ask_system(session)
    keep_alive = _ask_keep_alive(session, default="")
    preview = render_modelfile(base, parameters, system)
    show_text_block("Modelfile", preview)
    _need_confirm(session.prompter.confirm(t("Create model {name}?", name=name), default=True))
    _confirm_overwrite(session, name)
    create_from_spec(session.runner, name, base, parameters, system)
    if keep_alive:
        remember_keep_alive(session.settings, name, keep_alive)
    show_success(t("Model {name} created.", name=name))


def installed_flow(session: Session) -> None:
    ensure_server(session.client)
    while True:
        models = session.client.list_models()
        show_models(models)
        if not models:
            return
        model = _menu(session, t("Model"), _installed_choices(models), search=True)
        if model is None:
            return
        while True:
            action = _menu(
                session,
                model,
                [
                    (t("Run"), "run"),
                    (t("Information"), "show"),
                    (t("Copy"), "copy"),
                    (t("Configure"), "configure"),
                    (t("Delete"), "delete"),
                ],
            )
            if action is None:
                break
            if action == "run":
                if not run_flow(session, model):
                    continue
            elif action == "show":
                show_flow(session, model)
            elif action == "copy":
                copy_flow(session, model)
            elif action == "configure":
                configure_flow(session, model)
            else:
                delete_flow(session, model)
            break


def running_flow(session: Session) -> None:
    ensure_server(session.client)
    show_running(session.client.list_running())


def configure_flow(session: Session, model: str | None = None) -> None:
    ensure_server(session.client)
    if model is None:
        chosen = _choose_installed(session, t("Which model to configure"))
        if chosen is None:
            return
        model = chosen
    else:
        model = resolve_installed(session.client.list_models(), validate_model_name(model))
    shown = session.client.show(model)
    notice = t(
        "A new model will be created. The original stays unchanged until you explicitly confirm the same name."
    )
    console.print(f"[dim]{notice}[/dim]")
    new_name = validate_model_name(
        _need(
            session.prompter.text(
                t("New model name"),
                default=_variant_name(model),
                hint=t("Another name creates a variant. The same name overwrites the model after confirmation."),
            )
        )
    )
    parameters = _ask_parameters(session, current=shown.parameter_map)
    system = _ask_system(session, current=shown.system)
    keep_alive = _ask_keep_alive(session, default=session.settings.keep_alive.get(model, ""))
    if not parameters and system is None:
        unchanged = t("Modelfile parameters were not changed. A copy of the base model will be created.")
        console.print(f"[dim]{unchanged}[/dim]")
    preview = render_modelfile(model, parameters, system)
    show_text_block("Modelfile", preview)
    _need_confirm(
        session.prompter.confirm(t("Create {name} based on {base}?", name=new_name, base=model), default=True)
    )
    _confirm_overwrite(session, new_name)
    create_from_spec(session.runner, new_name, model, parameters, system)
    if keep_alive:
        remember_keep_alive(session.settings, new_name, keep_alive)
    if new_name != model:
        show_success(t("Model {name} created. {base} was not changed.", name=new_name, base=model))
    else:
        show_success(t("Model {name} was overwritten.", name=new_name))


def copy_flow(
    session: Session,
    source: str | None = None,
    destination: str | None = None,
) -> None:
    ensure_server(session.client)
    models = session.client.list_models()
    if source is None:
        chosen = _choose_installed(session, t("Which model to copy"))
        if chosen is None:
            return
        source = chosen
    else:
        source = resolve_installed(models, validate_model_name(source))
    if destination is None:
        destination = validate_model_name(
            _need(session.prompter.text(t("Name of the copy"), hint=t("For example my-copy or user/model")))
        )
    else:
        destination = validate_model_name(destination)
    if _model_exists(models, destination):
        _need_confirm(
            session.prompter.confirm(t("{name} already exists. Continue copying?", name=destination), default=False)
        )
    session.client.copy(source, destination)
    show_success(t("{source} copied to {destination}.", source=source, destination=destination))


def delete_flow(session: Session, model: str | None = None, *, yes: bool = False) -> None:
    ensure_server(session.client)
    models = session.client.list_models()
    if model is None:
        if not models:
            raise LamaError(t("No models are installed."))
        chosen = _menu(session, t("Which model to delete"), _installed_choices(models), search=True)
        if chosen is None:
            return
        model = chosen
    else:
        model = resolve_installed(models, validate_model_name(model))
    if not yes:
        _need_confirm(session.prompter.confirm(t("Delete {name}? This cannot be undone.", name=model), default=False))
    session.client.delete(model)
    show_success(t("Model {name} deleted.", name=model))


def show_flow(session: Session, model: str | None = None) -> None:
    ensure_server(session.client)
    models = session.client.list_models()
    if model is None:
        chosen = _choose_installed(session, t("Which model to describe"))
        if chosen is None:
            return
        model = chosen
    else:
        model = resolve_installed(models, validate_model_name(model))
    show_model_info(model, session.client.show(model))


def push_flow(session: Session, model: str | None = None) -> None:
    ensure_server(session.client)
    if model is None:
        chosen = _choose_installed(session, t("Which model to publish"))
        if chosen is None:
            return
        model = chosen
    else:
        model = validate_model_name(model)
    if not is_pushable_name(model):
        raise LamaError(
            t(
                "ollama push requires a name like username/model. "
                "Copy the model to such a name and publish again."
            )
        )
    key = pubkey_path()
    if not key.is_file():
        console.print(
            "[yellow]"
            + t(
                "Key ~/.ollama/id_ed25519.pub was not found. "
                "Publishing will not work without signing in to ollama.com."
            )
            + "[/yellow]"
        )
        _need_confirm(session.prompter.confirm(t("Run ollama signin now?"), default=True))
        session.runner.signin()
        if not key.is_file():
            raise LamaError(
                t("After sign-in, ~/.ollama/id_ed25519.pub did not appear. Publishing was cancelled.")
            )
    _need_confirm(session.prompter.confirm(t("Publish {name}?", name=model), default=False))
    session.runner.push(model)
    show_success(t("Model {name} was pushed to the registry.", name=model))


def server_menu(session: Session) -> None:
    while True:
        manager = ServerManager(session.settings, session.client, session.runner)
        show_server(manager.status())
        choice = _menu(
            session,
            "Ollama Server",
            [
                (t("Start"), "start"),
                (t("Stop"), "stop"),
                (t("Restart"), "restart"),
                (t("Journal"), "logs"),
                (t("Connection address"), "host"),
            ],
        )
        if choice is None:
            return
        try:
            if choice == "start":
                show_success(manager.start())
            elif choice == "stop":
                show_success(manager.stop())
            elif choice == "restart":
                show_success(manager.restart())
            elif choice == "logs":
                show_text_block(t("Journal"), manager.logs())
            elif choice == "host":
                connection_flow(session)
        except UserCancelled:
            console.print(f"[dim]{t('Cancelled')}[/dim]")
        except LamaError as exc:
            show_error(str(exc))


def server_action(session: Session, action: str) -> None:
    manager = ServerManager(session.settings, session.client, session.runner)
    if action == "status":
        show_server(manager.status())
        return
    if action == "start":
        show_success(manager.start())
        return
    if action == "stop":
        show_success(manager.stop())
        return
    if action == "restart":
        show_success(manager.restart())
        return
    if action == "logs":
        show_text_block(t("Journal"), manager.logs())
        return
    raise LamaError(t("Unknown server action: {action}", action=action))


def connection_flow(session: Session) -> None:
    entered = _need(
        session.prompter.text(
            t("Ollama address"),
            default=session.settings.host,
            hint=t("127.0.0.1:11434 or http://127.0.0.1:11434"),
        )
    )
    base, env_host = normalize_host(entered)
    session.settings.host = base
    session.settings.env_host = env_host
    save_settings(session.settings)
    session.client.close()
    session.client = build_client(session.settings)
    session.runner = build_runner(session.settings)
    if os.environ.get("OLLAMA_HOST"):
        console.print(
            "[yellow]"
            + t("OLLAMA_HOST is set and will override the saved address on the next launch.")
            + "[/yellow]"
        )
    show_success(t("Address saved: {host}", host=base))


def language_flow(session: Session) -> None:
    picked = _menu(session, t("Language"), [("English", "en"), ("Русский", "ru")])
    if picked is None:
        return
    session.settings.lang = picked
    save_settings(session.settings)
    set_language(picked)
    if os.environ.get("LAMA_LANG"):
        console.print(
            "[yellow]"
            + t("LAMA_LANG is set and will override the saved language on the next launch.")
            + "[/yellow]"
        )
    show_success(t("Language saved."))


def system_info(session: Session) -> None:
    _open_resources(session, include_runtime=True)


def monitor_screen(session: Session) -> None:
    _open_resources(session, include_runtime=False)


def _open_resources(session: Session, *, include_runtime: bool) -> None:
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        _show_static_resources(session, include_runtime=include_runtime)
        return
    choice = _menu(
        session,
        t("System resources"),
        [(t("Live refresh"), "monitor"), (t("Snapshot"), "static")],
    )
    if choice == "monitor":
        run_monitor(session.client)
    elif choice == "static":
        _show_static_resources(session, include_runtime=include_runtime)


def _show_static_resources(session: Session, *, include_runtime: bool) -> None:
    if include_runtime:
        lines = [t("Address: {host}", host=session.settings.host)]
        try:
            lines.append(f"API: {session.client.version()}")
        except LamaError as exc:
            lines.append(t("API: unavailable ({error})", error=exc))
        try:
            binary = session.runner.binary()
            lines.append(t("Command: {binary}", binary=binary))
            lines.append(session.runner.version_text() or t("Client version was not received."))
            help_text = session.runner.help_text()
        except LamaError as exc:
            help_text = ""
            lines.append(str(exc))
        show_text_block("Ollama", "\n".join(lines))
        if help_text:
            available = supported_commands(help_text)
            rows = "\n".join(f"{'●' if enabled else '○'} {name}" for name, enabled in available.items())
            show_text_block(t("Commands in this ollama version"), rows)
    console.print(render_snapshot(collect_snapshot(session.client, settle_seconds=0.2)))


def print_installed(session: Session) -> None:
    ensure_server(session.client)
    show_models(session.client.list_models())


def print_running(session: Session) -> None:
    ensure_server(session.client)
    show_running(session.client.list_running())


def resolve_installed(models: list[InstalledModel], name: str) -> str:
    names = [item.name for item in models]
    if name in names:
        return name
    if ":" not in name:
        matches = [item for item in names if item.split(":", 1)[0] == name]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise LamaError(
                t("Several models match {name}: {matches}. Specify a tag.", name=name, matches=", ".join(matches))
            )
    raise LamaError(t("Model {name} is not installed.", name=name))


def resolve_running(models: list[RunningModel], name: str) -> str:
    names = [item.name for item in models]
    if name in names:
        return name
    if ":" not in name:
        matches = [item for item in names if item.split(":", 1)[0] == name]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise LamaError(
                t(
                    "Several loaded variants match {name}: {matches}. Specify a tag.",
                    name=name,
                    matches=", ".join(matches),
                )
            )
    raise LamaError(t("Model {name} is not loaded in memory.", name=name))


def open_library() -> None:
    console.print(Text(_LIBRARY_URL))
    if shutil.which("xdg-open") is None:
        return
    subprocess.run(
        ["xdg-open", _LIBRARY_URL],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _start_chat(session: Session, model: str, *, keepalive: str | None) -> None:
    console.print(f"[dim]{t('Ollama interactive chat. After you leave it, you return to the menu.')}[/dim]")
    session.runner.run_chat(model, keepalive=keepalive)


def _load_model(session: Session, model: str) -> None:
    default = session.settings.keep_alive.get(model, "5m")
    keep_alive = _ask_keep_alive(session, default=default) or default
    session.client.load(model, keep_alive)
    remember_keep_alive(session.settings, model, keep_alive)
    show_success(t("Model {name} loaded into memory for {duration}.", name=model, duration=keep_alive))


def _run_custom(session: Session, model: str) -> None:
    shown = session.client.show(model)
    parameters = _ask_parameters(session, current=shown.parameter_map)
    system = _ask_system(session, current=shown.system)
    keep_alive = _ask_keep_alive(session, default=session.settings.keep_alive.get(model, ""))
    think_raw = _need(
        session.prompter.text(
            "Thinking",
            hint=t("true, false, high, medium, or low. Empty — do not pass --think."),
        )
    ).strip()
    think = think_raw or None
    if think is not None and think not in _THINK_VALUES:
        raise LamaError(t("Thinking accepts true, false, high, medium, or low."))
    target = model
    if parameters or system is not None:
        target = validate_model_name(
            _need(
                session.prompter.text(
                    t("New model name"),
                    default=_variant_name(model),
                    hint=t("Modelfile settings are written to a new model. The original is not changed."),
                )
            )
        )
        preview = render_modelfile(model, parameters, system)
        show_text_block("Modelfile", preview)
        _need_confirm(session.prompter.confirm(t("Create {name} and start a chat?", name=target), default=True))
        _confirm_overwrite(session, target)
        create_from_spec(session.runner, target, model, parameters, system)
    if keep_alive:
        remember_keep_alive(session.settings, target, keep_alive)
    _start_chat(session, target, keepalive=keep_alive)


def _ask_parameters(session: Session, current: dict[str, str] | None) -> dict[str, int | float]:
    values: dict[str, int | float] = {}
    for name in PARAMETER_ORDER:
        hint = t(str(PARAMETER_FIELDS[name]["hint"]))
        if current is None:
            message = name
            hint = f"{hint} {t('Empty — do not write it to the Modelfile.')}"
        else:
            shown = current.get(name) or t("not set")
            message = f"{name} [{shown}]"
            hint = f"{hint} {t('Empty — keep the inherited value.')}"
        raw = _need(session.prompter.text(message, hint=hint))
        parsed = parse_parameter(name, raw)
        if parsed is not None:
            values[name] = parsed
    return values


def _ask_system(session: Session, current: str = "") -> str | None:
    hint = t("Empty — do not set one.")
    if current:
        hint = t("Current: {text}. Empty — do not change.", text=current)
    raw = _need(session.prompter.text("System prompt", hint=hint))
    return validate_system_prompt(raw)


def _ask_keep_alive(session: Session, *, default: str) -> str | None:
    raw = _need(
        session.prompter.text(
            "Keep Alive",
            default=default,
            hint=t("10m, 30s, 1h, a number of seconds, or -1. Not written to the Modelfile. Empty — skip."),
        )
    )
    if raw.strip() == "":
        return None
    return validate_keep_alive(raw)


def _store_optional_keep_alive(session: Session, model: str) -> None:
    keep_alive = _ask_keep_alive(session, default="")
    if keep_alive:
        remember_keep_alive(session.settings, model, keep_alive)


def _confirm_overwrite(session: Session, name: str) -> None:
    if _model_exists(session.client.list_models(), name):
        _need_confirm(
            session.prompter.confirm(
                t("Model {name} already exists. Overwrite it?", name=name),
                default=False,
            )
        )


def _ask_model_to_pull(session: Session) -> str | None:
    while True:
        open_catalog = session.prompter.confirm(
            t("Open the catalog {url}?", url=_LIBRARY_URL),
            default=False,
        )
        if open_catalog is None:
            raise UserCancelled()
        if open_catalog:
            open_library()
        entered = _need(
            session.prompter.text(
                t("Model name"),
                hint=t("For example qwen3.5:9b or hf.co/user/repo:Q4_K_M"),
            )
        ).strip()
        if not entered:
            raise LamaError(t("Model name cannot be empty."))
        resolved = _maybe_pick_tag(session, entered)
        if resolved is not None:
            return resolved


def _maybe_pick_tag(session: Session, name: str) -> str | None:
    if ":" in name or not is_library_name(name):
        return validate_model_name(name)
    tags = session.client.library_tags(name)
    if not tags:
        console.print(f"[dim]{t('The registry tag list is unavailable. The name you entered will be used.')}[/dim]")
        return validate_model_name(name)
    choices = [(tag, tag) for tag in tags]
    choices.append((t("Enter a tag manually"), "__manual__"))
    picked = _menu(session, t("Choose a tag"), choices, search=True)
    if picked is None:
        return None
    if picked == "__manual__":
        manual = _need(session.prompter.text(t("Tag"), hint=t("For example 9b or Q4_K_M"))).strip()
        if not manual:
            raise LamaError(t("Tag cannot be empty."))
        return validate_model_name(f"{name}:{manual}")
    return validate_model_name(f"{name}:{picked}")


class _PullProgress:
    def __init__(self) -> None:
        from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        )
        self._task = self._progress.add_task(t("Downloading"), total=None)
        self._layers: dict[str, tuple[int, int]] = {}
        self._progress.start()

    def handle(self, event: PullEvent) -> None:
        if event.digest and event.total:
            self._layers[event.digest] = (event.completed or 0, event.total)
            total = sum(item[1] for item in self._layers.values())
            done = sum(item[0] for item in self._layers.values())
            self._progress.update(self._task, total=total, completed=done, description=event.status)
            return
        self._progress.update(self._task, description=event.status)

    def close(self) -> None:
        self._progress.stop()


def _choose_installed(session: Session, message: str) -> str | None:
    models = session.client.list_models()
    if not models:
        raise LamaError(t("No models are installed."))
    return _menu(session, message, _installed_choices(models), search=True)


def _menu(
    session: Session,
    message: str,
    choices: list[tuple[str, str]],
    *,
    search: bool = False,
) -> str | None:
    picked = session.prompter.select(message, [*choices, (t("Back"), "back")], search=search)
    if picked is None or picked == "back":
        return None
    return str(picked)


def _installed_choices(models: list[InstalledModel]) -> list[tuple[str, str]]:
    return [(f"{item.name}   {format_bytes(item.size)}", item.name) for item in models]


def _running_title(model: RunningModel) -> str:
    return f"{model.name}   {format_bytes(model.size)}"


def _model_exists(models: list[InstalledModel], name: str) -> bool:
    names = {item.name for item in models}
    if name in names:
        return True
    return ":" not in name and f"{name}:latest" in names


def _variant_name(name: str) -> str:
    base, separator, tag = name.partition(":")
    variant = f"{base}-custom"
    if separator:
        return f"{variant}:{tag}"
    return variant


def _need(value: str | None) -> str:
    if value is None:
        raise UserCancelled()
    return value


def _need_confirm(value: bool | None) -> None:
    if not value:
        raise UserCancelled()


def _stop_channel(channel: str) -> str:
    if channel == "cli":
        return "ollama stop"
    return "API keep_alive=0"
