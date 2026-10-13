import pytest

from lama.errors import ValidationError
from lama.ollama.modelfile import render_modelfile


def test_modelfile_contains_only_changed_parameters() -> None:
    text = render_modelfile(
        "qwen3.5:9b",
        {"temperature": 0.6, "num_ctx": 16384},
        "Ты помощник",
    )
    assert text == ('FROM qwen3.5:9b\nPARAMETER num_ctx 16384\nPARAMETER temperature 0.6\nSYSTEM """Ты помощник"""\n')
    assert "keep_alive" not in text
    assert "num_gpu" not in text


def test_modelfile_gguf_path_is_not_a_shell_command() -> None:
    path = "/home/alex/models/my model.gguf"
    text = render_modelfile(path, {"num_gpu": -1}, None)
    assert text.splitlines()[0] == f"FROM {path}"
    assert "sh -c" not in text
    assert "$(" not in text
    assert ";" not in text


def test_keep_alive_is_rejected() -> None:
    with pytest.raises(ValidationError, match="keep_alive"):
        render_modelfile("qwen", {"keep_alive": 10}, None)  # type: ignore[dict-item]


def test_unknown_parameter_is_rejected() -> None:
    with pytest.raises(ValidationError, match="mirostat"):
        render_modelfile("qwen", {"mirostat": 2}, None)
