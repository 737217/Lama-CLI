"""Ожидаемые ошибки утилиты. Их текст показывается пользователю без traceback."""


class LamaError(Exception):
    """Ошибка, которую можно объяснить пользователю."""


class UserCancelled(LamaError):
    """Пользователь отменил шаг клавишей Esc или отказался от подтверждения."""


class ValidationError(LamaError):
    """Некорректный ввод."""


class OllamaUnavailable(LamaError):
    """Сервер Ollama не отвечает."""


class OllamaAPIError(LamaError):
    """Ollama ответила ошибкой API."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class OllamaCommandError(LamaError):
    """Команда ollama завершилась с ненулевым кодом."""

    def __init__(self, message: str, *, returncode: int, argv: list[str]) -> None:
        super().__init__(message)
        self.returncode = returncode
        self.argv = argv
