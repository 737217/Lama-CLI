from datetime import UTC, datetime
from io import StringIO

from rich.console import Console

from lama.formatting import (
    format_bytes,
    format_context,
    format_expiry,
    format_modified,
    format_options,
    format_temperature,
)
from lama.ollama.types import InstalledModel
from lama.ui import display


def test_format_bytes_matches_table_style() -> None:
    assert format_bytes(None) == "—"
    assert format_bytes(8_697_040_896) == "8.1 GB"
    assert format_bytes(0) == "0 B"


def test_format_modified_today() -> None:
    now = datetime(2026, 10, 12, 12, tzinfo=UTC)
    assert format_modified(datetime(2026, 10, 12, 8, tzinfo=UTC), now=now) == "today"
    assert format_modified(datetime(2026, 10, 10, 8, tzinfo=UTC), now=now) == "2 days ago"
    assert format_modified(None) == "—"


def test_format_context_prefers_configured_num_ctx() -> None:
    assert format_context({"num_ctx": "65536"}, 262144) == "65536"
    assert format_context({}, 262144) == "max 262144"
    assert format_context({}, None) == "—"
    assert format_temperature({"temperature": "0.6"}) == "0.6"
    assert format_temperature({}) == "—"
    assert format_options({"top_p": "0.9", "num_gpu": "-1", "num_ctx": "65536"}) == "top_p 0.9 · num_gpu -1"


def test_long_model_name_wraps_inside_the_table(monkeypatch) -> None:
    buffer = StringIO()
    monkeypatch.setattr(
        display,
        "console",
        Console(file=buffer, force_terminal=True, width=100, height=30, color_system=None),
    )
    name = "hf.co/byteshape/Qwen3-Coder-30B-A3B-Instruct-GGUF:Qwen3-Coder-30B-A3B-Instruct-Q3_K_M-2.69bpw.gguf"
    display.show_models(
        [
            InstalledModel(
                name=name,
                size=10_000_000_000,
                modified_at=None,
                parameter_size="30.5B",
                quantization="Q3_K_M",
                parameters={"temperature": "0.7", "top_p": "0.8"},
                max_context=262144,
            )
        ]
    )
    lines = [line for line in buffer.getvalue().splitlines() if line.strip()]
    assert lines
    assert max(len(line) for line in lines) <= 100
    assert len(set(map(len, lines))) == 1
    assert not any(name in line for line in lines)
    assert "30.5B" in buffer.getvalue()
    assert "0.7" in buffer.getvalue()


def test_model_table_shows_context_temperature_and_options(monkeypatch) -> None:
    buffer = StringIO()
    monkeypatch.setattr(display, "console", Console(file=buffer, force_terminal=False, width=160))
    display.show_models(
        [
            InstalledModel(
                name="qwen3.8:27b-ctx-64k",
                size=100,
                modified_at=None,
                parameter_size="26.9B",
                quantization="IQ2_S",
                parameters={"num_ctx": "65536", "temperature": "0.6", "top_p": "0.9"},
                max_context=262144,
            )
        ]
    )
    text = buffer.getvalue()
    assert "65536" in text
    assert "0.6" in text
    assert "26.9B" in text
    assert "IQ2_S" in text
    assert "top_p 0.9" in text


def test_format_expiry_hides_missing_and_forever() -> None:
    assert format_expiry(None) == "—"
    assert format_expiry(datetime(1, 1, 1, tzinfo=UTC)) == "does not unload"
