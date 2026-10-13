"""Главное меню. Цикл, а не рекурсия: после действия управление возвращается сюда."""

from __future__ import annotations

from lama.config import Settings
from lama.errors import LamaError, UserCancelled
from lama.i18n import t
from lama.ollama.client import OllamaClient
from lama.ui.actions import (
    Session,
    configure_flow,
    connection_flow,
    copy_flow,
    create_flow,
    delete_flow,
    installed_flow,
    language_flow,
    open_session,
    pull_flow,
    push_flow,
    run_flow,
    running_flow,
    server_menu,
    show_flow,
    stop_flow,
    system_info,
)
from lama.ui.display import console, show_error, show_header
from lama.ui.prompts import QuestionaryPrompter


def main_menu() -> list[tuple[str, str]]:
    return [
        (f"▶  {t('Run a model')}", "run"),
        (f"■  {t('Stop a model')}", "stop"),
        (f"↓  {t('Pull a model')}", "pull"),
        (f"+  {t('Create a model')}", "create"),
        (f"☷  {t('Installed models')}", "list"),
        (f"◉  {t('Loaded models')}", "ps"),
        (f"⚙  {t('Configure a model')}", "configure"),
        (f"⧉  {t('Copy a model')}", "copy"),
        (f"✕  {t('Delete a model')}", "delete"),
        (f"ⓘ  {t('Model information')}", "show"),
        (f"⇧  {t('Publish a model')}", "push"),
        (f"◈  {t('Server')}", "server"),
        (f"▣  {t('System information')}", "info"),
        (f"🌐  {t('Language')}", "language"),
        (f"↩  {t('Exit')}", "exit"),
    ]


def run_main_menu(settings: Settings, prompter: QuestionaryPrompter | None = None) -> None:
    session = open_session(settings, prompter=prompter)
    try:
        while True:
            _print_status(session.client)
            choice = session.prompter.select(t("Action"), main_menu())
            if choice is None or choice == "exit":
                return
            try:
                _dispatch(session, choice)
            except KeyboardInterrupt:
                console.print(f"\n[dim]{t('Operation interrupted')}[/dim]")
            except UserCancelled:
                console.print(f"[dim]{t('Cancelled')}[/dim]")
            except LamaError as exc:
                show_error(str(exc))
    finally:
        session.client.close()


def _print_status(client: OllamaClient) -> None:
    try:
        version = client.version()
        count = len(client.list_models())
    except LamaError:
        show_header(api_up=False, version=None, model_count=None)
        return
    show_header(api_up=True, version=version, model_count=count)


def _dispatch(session: Session, choice: str) -> None:
    if choice == "run":
        run_flow(session)
    elif choice == "stop":
        stop_flow(session)
    elif choice == "pull":
        pull_flow(session)
    elif choice == "create":
        create_flow(session)
    elif choice == "list":
        installed_flow(session)
    elif choice == "ps":
        running_flow(session)
    elif choice == "configure":
        configure_flow(session)
    elif choice == "copy":
        copy_flow(session)
    elif choice == "delete":
        delete_flow(session)
    elif choice == "show":
        show_flow(session)
    elif choice == "push":
        push_flow(session)
    elif choice == "server":
        server_menu(session)
    elif choice == "info":
        system_info(session)
    elif choice == "language":
        language_flow(session)
    elif choice == "host":
        connection_flow(session)
