"""Прямые команды lama и точка входа интерактивного меню."""

from __future__ import annotations

import os
from collections.abc import Callable
from functools import wraps
from typing import Annotated

import typer
from typer._click.exceptions import ClickException

from lama.config import Settings, load_settings, peek_language
from lama.errors import LamaError, UserCancelled
from lama.i18n import set_language, t
from lama.ui.actions import (
    copy_flow,
    create_flow,
    delete_flow,
    monitor_screen,
    open_session,
    print_installed,
    print_running,
    pull_flow,
    push_flow,
    run_flow,
    server_action,
    server_menu,
    show_flow,
    stop_flow,
    system_info,
)
from lama.ui.app import run_main_menu
from lama.ui.display import console, show_error


def _activate_language() -> None:
    try:
        set_language(peek_language())
    except LamaError as exc:
        show_error(str(exc))
        raise SystemExit(1) from None


_activate_language()

typer_app = typer.Typer(
    name="lama",
    help=t("Interactive manager for a local Ollama."),
    no_args_is_help=False,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)
server_app = typer.Typer(
    help=t("Manage the Ollama server."),
    no_args_is_help=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)
typer_app.add_typer(server_app, name="server")


def guarded(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args: object, **kwargs: object) -> object:
        try:
            return func(*args, **kwargs)
        except UserCancelled:
            console.print(f"[dim]{t('Cancelled')}[/dim]")
            raise typer.Exit(0) from None
        except LamaError as exc:
            show_error(str(exc))
            if os.environ.get("LAMA_DEBUG"):
                raise
            raise typer.Exit(1) from None

    return wrapper


def current_settings(ctx: typer.Context) -> Settings:
    current: typer.Context | None = ctx
    while current is not None:
        if isinstance(current.obj, Settings):
            return current.obj
        current = current.parent
    return load_settings(None)


@typer_app.callback(
    invoke_without_command=True,
    help=t("Open the menu. Without a subcommand, lama runs interactively."),
)
@guarded
def main(
    ctx: typer.Context,
    host: Annotated[
        str | None,
        typer.Option("--host", help=t("Ollama address. Overrides OLLAMA_HOST and the config file.")),
    ] = None,
) -> None:
    """Открыть меню. Без подкоманды lama работает интерактивно."""
    ctx.obj = load_settings(host)
    if ctx.invoked_subcommand is None:
        run_main_menu(ctx.obj)


@typer_app.command(help=t("Run a model: chat, load into memory, or custom settings."))
@guarded
def run(
    ctx: typer.Context,
    model: Annotated[str | None, typer.Argument(help=t("Name of an installed model."))] = None,
) -> None:
    """Запустить модель: чат, загрузка в память или свои настройки."""
    session = open_session(current_settings(ctx))
    try:
        run_flow(session, model, direct_chat=model is not None)
    finally:
        session.client.close()


@typer_app.command(help=t("Unload a model from memory. Without a name, a picker opens."))
@guarded
def stop(
    ctx: typer.Context,
    model: Annotated[str | None, typer.Argument(help=t("Name of a loaded model."))] = None,
) -> None:
    """Выгрузить модель из памяти. Без имени откроется выбор."""
    session = open_session(current_settings(ctx))
    try:
        stop_flow(session, model)
    finally:
        session.client.close()


@typer_app.command(help=t("Download a model and show progress."))
@guarded
def pull(
    ctx: typer.Context,
    model: Annotated[str | None, typer.Argument(help=t("Model name, tag, or an hf.co reference."))] = None,
) -> None:
    """Скачать модель и показать прогресс."""
    session = open_session(current_settings(ctx))
    try:
        pull_flow(session, model)
    finally:
        session.client.close()


@typer_app.command("list", help=t("Show installed models."))
@guarded
def list_models(ctx: typer.Context) -> None:
    """Показать установленные модели."""
    session = open_session(current_settings(ctx))
    try:
        print_installed(session)
    finally:
        session.client.close()


@typer_app.command(help=t("Show models loaded in memory."))
@guarded
def ps(ctx: typer.Context) -> None:
    """Показать модели, загруженные в память."""
    session = open_session(current_settings(ctx))
    try:
        print_running(session)
    finally:
        session.client.close()


@typer_app.command(help=t("Create a model from a Modelfile, a GGUF file, or an existing model."))
@guarded
def create(ctx: typer.Context) -> None:
    """Создать модель мастером Modelfile, GGUF или готового файла."""
    session = open_session(current_settings(ctx))
    try:
        create_flow(session)
    finally:
        session.client.close()


@typer_app.command(help=t("Show model details and its Modelfile."))
@guarded
def show(
    ctx: typer.Context,
    model: Annotated[str | None, typer.Argument(help=t("Model name."))] = None,
) -> None:
    """Показать сведения и Modelfile модели."""
    session = open_session(current_settings(ctx))
    try:
        show_flow(session, model)
    finally:
        session.client.close()


@typer_app.command(help=t("Delete a model after confirmation."))
@guarded
def rm(
    ctx: typer.Context,
    model: Annotated[str | None, typer.Argument(help=t("Model name."))] = None,
    yes: Annotated[bool, typer.Option("--yes", "-y", help=t("Delete without confirmation."))] = False,
) -> None:
    """Удалить модель после подтверждения."""
    session = open_session(current_settings(ctx))
    try:
        delete_flow(session, model, yes=yes)
    finally:
        session.client.close()


@typer_app.command(help=t("Copy a model."))
@guarded
def cp(
    ctx: typer.Context,
    source: Annotated[str | None, typer.Argument(help=t("Source model."))] = None,
    destination: Annotated[str | None, typer.Argument(help=t("Name of the copy."))] = None,
) -> None:
    """Скопировать модель."""
    session = open_session(current_settings(ctx))
    try:
        copy_flow(session, source, destination)
    finally:
        session.client.close()


@typer_app.command(help=t("Publish a model with ollama push."))
@guarded
def push(
    ctx: typer.Context,
    model: Annotated[str | None, typer.Argument(help=t("Name of the form username/model."))] = None,
) -> None:
    """Опубликовать модель через ollama push."""
    session = open_session(current_settings(ctx))
    try:
        push_flow(session, model)
    finally:
        session.client.close()


@typer_app.command(help=t("Ask for a mode and show RAM, GPU, and loaded models."))
@guarded
def monitor(ctx: typer.Context) -> None:
    """Спросить режим и показать RAM, GPU и загруженные модели."""
    session = open_session(current_settings(ctx))
    try:
        monitor_screen(session)
    finally:
        session.client.close()


@server_app.callback(invoke_without_command=True, help=t("Open the server menu when no action is given."))
@guarded
def server_root(ctx: typer.Context) -> None:
    """Открыть меню сервера, если действие не указано."""
    if ctx.invoked_subcommand is None:
        session = open_session(current_settings(ctx))
        try:
            server_menu(session)
        finally:
            session.client.close()


@server_app.command("status", help=t("Check the API, version, and how the server is controlled."))
@guarded
def server_status(ctx: typer.Context) -> None:
    """Проверить API, версию и способ управления сервером."""
    _server(ctx, "status")


@server_app.command("start", help=t("Start the server using the available method."))
@guarded
def server_start(ctx: typer.Context) -> None:
    """Запустить сервер доступным способом."""
    _server(ctx, "start")


@server_app.command("stop", help=t("Stop the systemd service or the process started by lama."))
@guarded
def server_stop(ctx: typer.Context) -> None:
    """Остановить службу systemd или процесс, запущенный lama."""
    _server(ctx, "stop")


@server_app.command("restart", help=t("Restart the managed server."))
@guarded
def server_restart(ctx: typer.Context) -> None:
    """Перезапустить управляемый сервер."""
    _server(ctx, "restart")


@server_app.command("logs", help=t("Show the systemd journal or the lama process log."))
@guarded
def server_logs(ctx: typer.Context) -> None:
    """Показать журнал systemd или файл процесса lama."""
    _server(ctx, "logs")


@typer_app.command("info", help=t("Ollama version, command compatibility, and a resource snapshot."))
@guarded
def info(ctx: typer.Context) -> None:
    """Версия Ollama, совместимость команд и снимок ресурсов."""
    session = open_session(current_settings(ctx))
    try:
        system_info(session)
    finally:
        session.client.close()


def _server(ctx: typer.Context, action: str) -> None:
    session = open_session(current_settings(ctx))
    try:
        server_action(session, action)
    finally:
        session.client.close()


def app() -> None:
    try:
        code = typer_app(standalone_mode=False)
    except ClickException as exc:
        exc.show()
        raise SystemExit(exc.exit_code) from None
    except UserCancelled:
        console.print(f"[dim]{t('Cancelled')}[/dim]")
        raise SystemExit(0) from None
    except LamaError as exc:
        show_error(str(exc))
        if os.environ.get("LAMA_DEBUG"):
            raise
        raise SystemExit(1) from None
    if code == 130:
        console.print(f"\n[dim]{t('Interrupted')}[/dim]")
    if code:
        raise SystemExit(code)
