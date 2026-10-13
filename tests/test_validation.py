import pytest

from lama.errors import ValidationError
from lama.validation import (
    is_pushable_name,
    normalize_host,
    parse_parameter,
    validate_keep_alive,
    validate_model_name,
    validate_system_prompt,
)


@pytest.mark.parametrize(
    "name",
    [
        "gemma4:12b",
        "qwen3.5:9b",
        "my-qwen",
        "user/my-model:latest",
        "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF",
        "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M",
        "huggingface.co/user/repo:Q5_K_M",
    ],
)
def test_valid_model_names(name: str) -> None:
    assert validate_model_name(f"  {name}  ") == name


@pytest.mark.parametrize(
    "name",
    ["", "   ", "model name", "model;rm", "model && ls", "$(whoami)", "../secrets", "model`id`"],
)
def test_rejects_unsafe_model_names(name: str) -> None:
    with pytest.raises(ValidationError):
        validate_model_name(name)


@pytest.mark.parametrize("value", ["-1", "0", "10m", "30s", "1h", "20"])
def test_keep_alive_values(value: str) -> None:
    assert validate_keep_alive(value) == value


@pytest.mark.parametrize("value", ["", "10 minutes", "5m30s", "-2", "m10", "drop table"])
def test_keep_alive_rejects_garbage(value: str) -> None:
    with pytest.raises(ValidationError):
        validate_keep_alive(value)


def test_parameter_ranges_and_blank() -> None:
    assert parse_parameter("num_ctx", "") is None
    assert parse_parameter("num_ctx", "16384") == 16384
    assert parse_parameter("num_gpu", "-1") == -1
    assert parse_parameter("temperature", "0.6") == 0.6
    assert parse_parameter("top_p", "1") == 1.0
    assert parse_parameter("num_predict", "-1") == -1
    with pytest.raises(ValidationError):
        parse_parameter("num_ctx", "0")
    with pytest.raises(ValidationError):
        parse_parameter("top_p", "1.2")
    with pytest.raises(ValidationError):
        parse_parameter("num_gpu", "-2")
    with pytest.raises(ValidationError):
        parse_parameter("temperature", "hot")


def test_system_prompt_quotes() -> None:
    assert validate_system_prompt("  ") is None
    assert validate_system_prompt("Будь краток") == "Будь краток"
    with pytest.raises(ValidationError):
        validate_system_prompt('текст """ внутри')


def test_host_normalization() -> None:
    assert normalize_host("127.0.0.1:11434") == ("http://127.0.0.1:11434", "127.0.0.1:11434")
    assert normalize_host("http://localhost:11434/") == ("http://localhost:11434", "localhost:11434")
    assert normalize_host("https://ollama.internal") == (
        "https://ollama.internal:11434",
        "https://ollama.internal:11434",
    )
    with pytest.raises(ValidationError):
        normalize_host("http://bad host")
    with pytest.raises(ValidationError):
        normalize_host("")


def test_pushable_name() -> None:
    assert is_pushable_name("alex/my-qwen")
    assert is_pushable_name("alex/my-qwen:latest")
    assert not is_pushable_name("my-qwen")
    assert not is_pushable_name("hf.co/alex/model")
