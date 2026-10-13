from pathlib import Path

import pytest

from lama.config import DEFAULT_BASE_URL, Settings, load_settings, save_settings
from lama.errors import LamaError


def test_host_priority(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    stored = Settings(host="http://10.0.0.2:11434", env_host="10.0.0.2:11434", keep_alive={"qwen:9b": "10m"})
    save_settings(stored)

    from_file = load_settings(None)
    assert from_file.host == "http://10.0.0.2:11434"
    assert from_file.env_host == "10.0.0.2:11434"
    assert from_file.keep_alive["qwen:9b"] == "10m"

    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:11435")
    from_env = load_settings(None)
    assert from_env.host == "http://127.0.0.1:11435"
    assert from_env.keep_alive["qwen:9b"] == "10m"

    from_flag = load_settings("http://127.0.0.1:9")
    assert from_flag.host == "http://127.0.0.1:9"
    assert from_flag.env_host == "127.0.0.1:9"


def _config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "lama" / "config.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_missing_file_uses_defaults(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    settings = load_settings(None)
    assert settings.host == DEFAULT_BASE_URL
    assert settings.keep_alive == {}
    assert not (tmp_path / "lama" / "config.toml").exists()


def test_valid_file_is_not_rewritten(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    original = '# comment\nhost = "http://10.0.0.2:11434"\n\n[keep_alive]\n"qwen:9b" = "10m"\n'
    path = _config(tmp_path, original)
    settings = load_settings(None)
    assert settings.host == "http://10.0.0.2:11434"
    assert settings.keep_alive == {"qwen:9b": "10m"}
    assert path.read_text(encoding="utf-8") == original


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("host = [\n", "TOML syntax error"),
        ('host = ["http://127.0.0.1:11434"]\n', "host field must be a string"),
        ('host = "http://127.0.0.1:11434/extra"\n', "Invalid host value"),
        ('host = "http://127.0.0.1:11434"\nkeep_alive = "10m"\n', "keep_alive must be a TOML table"),
        (
            'host = "http://127.0.0.1:11434"\n\n[keep_alive]\n"qwen:9b" = 10\n',
            "must be a duration string",
        ),
        (
            'host = "http://127.0.0.1:11434"\n\n[keep_alive]\n"qwen:9b" = "forever"\n',
            "is invalid",
        ),
    ],
)
def test_invalid_config_is_reported_and_not_replaced(monkeypatch, tmp_path: Path, text: str, match: str) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    path = _config(tmp_path, text)
    before = path.read_bytes()
    with pytest.raises(LamaError, match=match) as caught:
        load_settings("127.0.0.1:9")
    assert str(path) in str(caught.value)
    assert path.read_bytes() == before
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:11435")
    with pytest.raises(LamaError, match=match):
        load_settings(None)
    assert path.read_bytes() == before


def test_oserror_while_reading_config(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    path = _config(tmp_path, 'host = "http://127.0.0.1:11434"\n')
    before = path.read_bytes()
    original = Path.read_text

    def wrapped(self: Path, *args: object, **kwargs: object) -> str:
        if self == path:
            raise OSError(13, "Permission denied")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", wrapped)
    with pytest.raises(LamaError, match="Permission denied") as caught:
        load_settings("127.0.0.1:9")
    assert str(path) in str(caught.value)
    assert path.read_bytes() == before
