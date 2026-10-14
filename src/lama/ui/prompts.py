"""Обёртки questionary: стрелки, Enter, поиск и Esc как отмена."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import questionary
from prompt_toolkit.key_binding import KeyBindings

from lama.i18n import t

Choice = tuple[str, str]

_STYLE = questionary.Style(
    [
        ("qmark", "fg:cyan"),
        ("question", "bold"),
        ("pointer", "fg:cyan bold"),
        ("highlighted", "fg:cyan"),
        ("answer", "fg:cyan"),
        ("instruction", "fg:ansibrightblack"),
    ]
)


class QuestionaryPrompter:
    def select(
        self,
        message: str,
        choices: list[Choice],
        *,
        search: bool = False,
    ) -> str | None:
        instruction = t("↑↓ Enter · Esc back")
        if search:
            instruction = t("↑↓ Enter · Esc back · type to search")
        options = [questionary.Choice(title=title, value=value) for title, value in choices]
        kwargs: dict[str, Any] = {
            "qmark": " ",
            "pointer": "❯",
            "style": _STYLE,
            "instruction": instruction,
            "use_shortcuts": False,
            "use_jk_keys": False,
        }
        if search:
            kwargs["use_search_filter"] = True
        question = _select(message, options, kwargs)
        result = _ask(question)
        if result is None:
            return None
        return str(result)

    def text(self, message: str, *, default: str = "", hint: str = "") -> str | None:
        question = questionary.text(
            message,
            default=default,
            instruction=hint or t("Esc — back"),
            qmark="❯",
            style=_STYLE,
        )
        result = _ask(question)
        if result is None:
            return None
        return str(result)

    def confirm(self, message: str, *, default: bool = False) -> bool | None:
        """Да или нет тем же списком, что и меню: стрелки, Enter и Esc.

        questionary.confirm ждёт клавиши y/n и по Enter берёт значение по умолчанию.
        Для удаления это «нет», поэтому Enter молча отменял операцию.
        """
        yes = (t("Yes"), "yes")
        no = (t("No"), "no")
        picked = self.select(message, [yes, no] if default else [no, yes])
        if picked is None:
            return None
        return picked == "yes"

    def path(self, message: str) -> str | None:
        question = questionary.path(message, qmark="❯", style=_STYLE)
        result = _ask(question)
        if result is None:
            return None
        return str(result)


def _select(message: str, options: list[questionary.Choice], kwargs: dict[str, Any]) -> Any:
    try:
        return questionary.select(message, choices=options, **kwargs)
    except TypeError:
        kwargs.pop("use_search_filter", None)
        return questionary.select(message, choices=options, **kwargs)


def _ask(question: Any) -> Any:
    _bind_escape(question)
    unsafe = getattr(question, "unsafe_ask", None)
    if isinstance(unsafe, Callable):
        return unsafe()
    return question.ask()


def _bind_escape(question: Any) -> None:
    application = getattr(question, "application", None)
    bindings = getattr(application, "key_bindings", None)
    target = bindings if isinstance(bindings, KeyBindings) else None
    if target is None:
        for registry in getattr(bindings, "registries", ()):
            if isinstance(registry, KeyBindings):
                target = registry
                break
    if target is None:
        return

    @target.add("escape", eager=True)
    def _escape(event: Any) -> None:
        event.app.exit(result=None)


prompter = QuestionaryPrompter()
