"""Конфигурация: --host, OLLAMA_HOST, затем ~/.config/lama/config.toml.

Если файл существует, он читается и проверяется целиком до выбора адреса.
Синтаксическая ошибка TOML, ошибка чтения и неверный тип или значение известных
полей (``lang``, ``host``, таблица ``keep_alive``, длительности) останавливают запуск:
lama не подставляет адрес по умолчанию и не перезаписывает повреждённый файл.
Это же правило действует, когда адрес задан флагом ``--host`` или переменной
``OLLAMA_HOST``: приоритет этих источников работает только для корректного файла.
Отсутствующий файл не ошибка, остаются обычные значения по умолчанию.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from lama.errors import LamaError, ValidationError
from lama.i18n import set_language, t
from lama.validation import normalize_host, validate_keep_alive

DEFAULT_BASE_URL = "http://127.0.0.1:11434"
LANGS = frozenset({"en", "ru"})


@dataclass
class Settings:
    host: str
    env_host: str
    keep_alive: dict[str, str] = field(default_factory=dict)
    lang: str = "en"


def config_path() -> Path:
    base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "lama" / "config.toml"


def state_dir() -> Path:
    base = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    return base / "lama"


def peek_language() -> str:
    """Язык для справки до полной проверки файла. Битый TOML не читается как язык."""
    raw = _env_lang()
    if raw is not None:
        return _require_lang(raw, source="LAMA_LANG")
    data = _peek_data()
    if data is None or "lang" not in data:
        return "en"
    return _require_lang(data.get("lang"), source="file", path=config_path())


def load_settings(host_override: str | None = None) -> Settings:
    """Собрать настройки. Повреждённый файл конфигурации не молча пропускается."""
    _apply_env_language()
    stored = _read_file()
    if _env_lang() is None:
        set_language(stored.lang)
    if host_override:
        raw = host_override
    elif os.environ.get("OLLAMA_HOST"):
        raw = os.environ["OLLAMA_HOST"]
    elif stored.host:
        raw = stored.host
    else:
        raw = DEFAULT_BASE_URL
    try:
        base, env_host = normalize_host(raw)
    except ValidationError as exc:
        if host_override:
            source = t("the --host flag")
        elif os.environ.get("OLLAMA_HOST"):
            source = "OLLAMA_HOST"
        else:
            source = t("the configuration")
        raise LamaError(t("Invalid address from {source}: {error}", source=source, error=exc)) from exc
    return Settings(host=base, env_host=env_host, keep_alive=stored.keep_alive, lang=stored.lang)


def save_settings(settings: Settings) -> Path:
    path = config_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_dump(settings), encoding="utf-8")
    except OSError as exc:
        raise LamaError(t("Could not save configuration {path}: {error}", path=path, error=exc)) from exc
    return path


def remember_keep_alive(settings: Settings, model: str, duration: str) -> None:
    settings.keep_alive[model] = duration
    save_settings(settings)


def _read_file() -> Settings:
    path = config_path()
    if not path.is_file():
        return Settings(host="", env_host="", keep_alive={}, lang="en")
    data = _load_toml(path)
    lang = _parse_lang(path, data.get("lang", "en"))
    if _env_lang() is None:
        set_language(lang)
    return Settings(
        host=_parse_host(path, data.get("host", "")),
        env_host="",
        keep_alive=_parse_keep_alive(path, data),
        lang=lang,
    )


def _parse_host(path: Path, host: object) -> str:
    if host is None:
        return ""
    if not isinstance(host, str):
        raise LamaError(
            t("In {path} the host field must be a string, not {kind}.", path=path, kind=type(host).__name__)
        )
    if host != "":
        try:
            normalize_host(host)
        except ValidationError as exc:
            raise LamaError(t("Invalid host value in {path}: {error}", path=path, error=exc)) from exc
    return host


def _parse_keep_alive(path: Path, data: dict[str, object]) -> dict[str, str]:
    if "keep_alive" not in data:
        return {}
    keep_raw = data["keep_alive"]
    if not isinstance(keep_raw, dict):
        raise LamaError(
            t("In {path} keep_alive must be a TOML table, not {kind}.", path=path, kind=type(keep_raw).__name__)
        )
    keep_alive: dict[str, str] = {}
    for key, value in keep_raw.items():
        if not isinstance(key, str):
            raise LamaError(t("In {path} a keep_alive key must be a string.", path=path))
        if not isinstance(value, str):
            raise LamaError(
                t(
                    "In {path} the keep_alive value for {key} must be a duration string, not {kind}.",
                    path=path,
                    key=key,
                    kind=type(value).__name__,
                )
            )
        try:
            keep_alive[key] = validate_keep_alive(value)
        except ValidationError as exc:
            raise LamaError(
                t("In {path} the keep_alive duration for {key} is invalid: {error}", path=path, key=key, error=exc)
            ) from exc
    return keep_alive


def _dump(settings: Settings) -> str:
    lines = [f"lang = {_quote(settings.lang)}", f"host = {_quote(settings.host)}", "", "[keep_alive]"]
    for name in sorted(settings.keep_alive):
        lines.append(f"{_quote(name)} = {_quote(settings.keep_alive[name])}")
    lines.append("")
    return "\n".join(lines)


def _env_lang() -> str | None:
    raw = os.environ.get("LAMA_LANG")
    if raw is None or raw.strip() == "":
        return None
    return raw.strip()


def _apply_env_language() -> None:
    raw = _env_lang()
    if raw is None:
        return
    set_language(_require_lang(raw, source="LAMA_LANG"))


def _peek_data() -> dict[str, object] | None:
    path = config_path()
    if not path.is_file():
        return None
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def _load_toml(path: Path) -> dict[str, object]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise LamaError(
            t("Could not read configuration {path}: the file is not UTF-8 ({error})", path=path, error=exc)
        ) from exc
    except OSError as exc:
        raise LamaError(t("Could not read configuration {path}: {error}", path=path, error=exc)) from exc
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise LamaError(
            t("Could not parse configuration {path}: TOML syntax error ({error})", path=path, error=exc)
        ) from exc
    if not isinstance(data, dict):
        raise LamaError(t("Could not parse configuration {path}: the file root must be a TOML table.", path=path))
    return data


def _parse_lang(path: Path, value: object) -> str:
    if value is None:
        return "en"
    return _require_lang(value, source="file", path=path)


def _require_lang(value: object, *, source: str, path: Path | None = None) -> str:
    if isinstance(value, str) and value in LANGS:
        return value
    shown = value if isinstance(value, str) else type(value).__name__
    if source == "LAMA_LANG":
        raise LamaError(t("LAMA_LANG must be en or ru, not {value}.", value=shown))
    raise LamaError(t("In {path} lang must be en or ru, not {value}.", path=path, value=shown))


def _quote(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'
