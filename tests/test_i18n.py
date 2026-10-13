import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from lama.config import load_settings, save_settings
from lama.errors import LamaError
from lama.formatting import format_modified
from lama.i18n import get_language, set_language, t, tn
from lama.server.manager import REMOTE_STOP_REASON
from lama.ui.actions import Session, language_flow
from lama.ui.app import main_menu


def _days(count: int) -> str:
    return tn(count, en=("{n} day ago", "{n} days ago"), ru=("{n} day ago", "{n} days ago", "{n} days ago"))


def test_missing_lang_stays_english_and_does_not_rewrite_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    path = tmp_path / "lama" / "config.toml"
    path.parent.mkdir()
    original = 'host = "http://10.0.0.2:11434"\n'
    path.write_text(original, encoding="utf-8")
    settings = load_settings(None)
    assert settings.lang == "en"
    assert get_language() == "en"
    assert path.read_text(encoding="utf-8") == original


def test_lang_roundtrip_keeps_host_and_keep_alive(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    path = tmp_path / "lama" / "config.toml"
    path.parent.mkdir()
    path.write_text(
        'lang = "ru"\nhost = "http://10.0.0.2:11434"\n\n[keep_alive]\n"qwen:9b" = "10m"\n',
        encoding="utf-8",
    )
    settings = load_settings(None)
    assert settings.lang == "ru"
    assert settings.host == "http://10.0.0.2:11434"
    assert settings.keep_alive == {"qwen:9b": "10m"}
    assert get_language() == "ru"
    save_settings(settings)
    text = path.read_text(encoding="utf-8")
    assert 'lang = "ru"' in text
    assert 'host = "http://10.0.0.2:11434"' in text
    assert '"qwen:9b" = "10m"' in text


def test_invalid_lang_and_env_do_not_rewrite_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    path = tmp_path / "lama" / "config.toml"
    path.parent.mkdir()
    original = 'lang = "de"\nhost = "http://127.0.0.1:11434"\n'
    path.write_text(original, encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(LamaError, match="lang must be en or ru"):
        load_settings(None)
    assert path.read_bytes() == before
    monkeypatch.setenv("LAMA_LANG", "de")
    with pytest.raises(LamaError, match="LAMA_LANG must be en or ru"):
        load_settings(None)
    assert path.read_bytes() == before


def test_lama_lang_overrides_file_without_changing_saved_lang(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    path = tmp_path / "lama" / "config.toml"
    path.parent.mkdir()
    original = 'lang = "ru"\nhost = "http://127.0.0.1:11434"\n'
    path.write_text(original, encoding="utf-8")
    monkeypatch.setenv("LAMA_LANG", "en")
    settings = load_settings(None)
    assert settings.lang == "ru"
    assert get_language() == "en"
    assert path.read_text(encoding="utf-8") == original


def test_russian_menu_messages_and_plurals() -> None:
    set_language("ru")
    titles = [title for title, value in main_menu()]
    assert any("Запустить модель" in title for title in titles)
    assert any(title.endswith("Язык") for title in titles)
    assert t("No models are installed.") == "Нет установленных моделей."
    assert t("Cancelled") == "Отменено"
    assert _days(1) == "1 день назад"
    assert _days(2) == "2 дня назад"
    assert _days(5) == "5 дней назад"
    assert _days(11) == "11 дней назад"
    assert _days(21) == "21 день назад"
    now = datetime(2026, 10, 12, 12, tzinfo=UTC)
    assert format_modified(datetime(2026, 10, 12, 8, tzinfo=UTC), now=now) == "сегодня"
    assert "удалённый" in t(REMOTE_STOP_REASON, host="http://example")


def test_language_flow_saves_and_switches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("OLLAMA_HOST", raising=False)

    class Scripted:
        def select(self, message: str, choices: list[tuple[str, str]], search: bool = False) -> str:
            assert ("English", "en") in choices
            assert ("Русский", "ru") in choices
            return "ru"

    settings = load_settings(None)
    session = Session(settings, Scripted(), object(), object())  # type: ignore[arg-type]
    language_flow(session)
    assert settings.lang == "ru"
    assert get_language() == "ru"
    assert t("Exit") == "Выход"
    text = (tmp_path / "lama" / "config.toml").read_text(encoding="utf-8")
    assert 'lang = "ru"' in text


def test_saved_lang_localizes_the_next_config_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    path = tmp_path / "lama" / "config.toml"
    path.parent.mkdir()
    path.write_text('lang = "ru"\nhost = ["nope"]\n', encoding="utf-8")
    with pytest.raises(LamaError, match="поле host должно быть строкой"):
        load_settings(None)


def test_toml_error_follows_lama_lang(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.setenv("LAMA_LANG", "ru")
    path = tmp_path / "lama" / "config.toml"
    path.parent.mkdir()
    path.write_text("host = [\n", encoding="utf-8")
    with pytest.raises(LamaError, match="синтаксическая ошибка TOML"):
        load_settings(None)


def test_help_uses_lama_lang(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["LAMA_LANG"] = "ru"
    env["XDG_CONFIG_HOME"] = str(tmp_path)
    env.pop("OLLAMA_HOST", None)
    completed = subprocess.run(
        [sys.executable, "-m", "lama", "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "Показать установленные модели" in completed.stdout


def test_invalid_lama_lang_stops_startup(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["LAMA_LANG"] = "de"
    env["XDG_CONFIG_HOME"] = str(tmp_path)
    completed = subprocess.run(
        [sys.executable, "-m", "lama", "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode != 0
    assert "LAMA_LANG" in output
    assert "Traceback" not in output
