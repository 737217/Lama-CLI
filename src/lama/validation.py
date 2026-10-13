"""Проверка пользовательского ввода до обращения к Ollama."""

from __future__ import annotations

import re
from pathlib import Path

from lama.errors import ValidationError
from lama.i18n import t

_MODEL_NAME = re.compile(
    r"^(?:hf\.co/|huggingface\.co/)?"
    r"[A-Za-z0-9][A-Za-z0-9._+-]*"
    r"(?:/[A-Za-z0-9][A-Za-z0-9._+-]*)*"
    r"(?::[A-Za-z0-9][A-Za-z0-9._+-]*)?$"
)
_KEEP_ALIVE = re.compile(r"^(-1|\d+[smh]?)$")
_HOST_WITH_PORT = re.compile(r"^[A-Za-z0-9._-]+:\d{1,5}$")

PARAMETER_FIELDS: dict[str, dict[str, object]] = {
    "num_ctx": {
        "kind": "int",
        "minimum": 1,
        "maximum": None,
        "hint": "Context length in tokens. If unset, the model keeps its own value or the Ollama default (2048).",
    },
    "num_batch": {
        "kind": "int",
        "minimum": 1,
        "maximum": None,
        "hint": "Prompt batch size. The Ollama runner default is 512.",
    },
    "num_gpu": {
        "kind": "int",
        "minimum": -1,
        "maximum": None,
        "hint": "How many model layers to send to the GPU. -1 — auto, 0 — CPU only. This is not the number of GPUs.",
    },
    "temperature": {
        "kind": "float",
        "minimum": 0,
        "maximum": None,
        "hint": "Sampling temperature. Higher is more varied, lower is more stable. Documented default: 0.8.",
    },
    "top_p": {
        "kind": "float",
        "minimum": 0,
        "maximum": 1,
        "hint": "Nucleus sampling. Allowed from 0 to 1. Documented default: 0.9.",
    },
    "top_k": {
        "kind": "int",
        "minimum": 0,
        "maximum": None,
        "hint": "How many of the most likely tokens to consider. Documented default: 40.",
    },
    "repeat_penalty": {
        "kind": "float",
        "minimum": 0,
        "maximum": None,
        "hint": "Repeat penalty. Documented Modelfile default: 1.0 (off).",
    },
    "num_predict": {
        "kind": "int",
        "minimum": -1,
        "maximum": None,
        "hint": "How many tokens to generate. -1 — no limit.",
    },
    "seed": {
        "kind": "int",
        "minimum": None,
        "maximum": None,
        "hint": "Random seed. The same seed and prompt give a more stable result.",
    },
}


def validate_model_name(name: str) -> str:
    """Принять имя модели Ollama, включая тег, namespace и ссылку Hugging Face."""
    cleaned = name.strip()
    if not cleaned:
        raise ValidationError(t("Model name cannot be empty."))
    if any(char.isspace() for char in cleaned) or any(ord(char) < 32 for char in cleaned):
        raise ValidationError(t("Model name must not contain spaces or control characters."))
    if ".." in cleaned or not _MODEL_NAME.fullmatch(cleaned):
        raise ValidationError(
            t("Invalid model name. Example: qwen3.5:9b, user/model:tag, or hf.co/user/repo:Q4_K_M.")
        )
    return cleaned


def validate_keep_alive(value: str) -> str:
    """Проверить длительность keep_alive: 10m, 30s, 1h, секунды числом, -1."""
    cleaned = value.strip()
    if not _KEEP_ALIVE.fullmatch(cleaned):
        raise ValidationError(
            t("Invalid duration. Use 10m, 30s, 1h, a number of seconds, or -1 (keep until the server stops).")
        )
    return cleaned


def parse_parameter(name: str, raw: str) -> int | float | None:
    """Разобрать значение параметра Modelfile. Пустая строка означает «не менять»."""
    if name not in PARAMETER_FIELDS:
        raise ValidationError(t("Parameter {name} is not supported.", name=name))
    text = raw.strip()
    if text == "":
        return None
    spec = PARAMETER_FIELDS[name]
    kind = spec["kind"]
    try:
        if kind == "int":
            if any(char in text for char in ".eE"):
                raise ValueError
            number: int | float = int(text, 10)
        else:
            number = float(text)
    except ValueError as exc:
        raise ValidationError(t("{name}: a number is expected.", name=name)) from exc
    minimum = spec["minimum"]
    maximum = spec["maximum"]
    if isinstance(minimum, (int, float)) and number < minimum:
        raise ValidationError(t("{name}: value cannot be less than {minimum}.", name=name, minimum=minimum))
    if isinstance(maximum, (int, float)) and number > maximum:
        raise ValidationError(t("{name}: value cannot be greater than {maximum}.", name=name, maximum=maximum))
    return number


def validate_gguf_path(path: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise ValidationError(t("GGUF file not found: {path}", path=candidate))
    if ".gguf" not in candidate.name.lower():
        raise ValidationError(t("Expected a local file with the .gguf extension."))
    return candidate


def validate_modelfile_path(path: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise ValidationError(t("Modelfile not found: {path}", path=candidate))
    return candidate


def validate_system_prompt(text: str) -> str | None:
    cleaned = text.strip()
    if cleaned == "":
        return None
    if '"""' in cleaned:
        raise ValidationError(t('System prompt must not contain the sequence """.'))
    return cleaned


def is_library_name(name: str) -> bool:
    """Имя каталога Ollama без namespace и без явного тега."""
    if name.startswith(("hf.co/", "huggingface.co/")):
        return False
    base = name.split(":", 1)[0]
    return "/" not in base


def is_pushable_name(name: str) -> bool:
    """Имя для ollama push: namespace/model[:tag], без хоста Hugging Face."""
    if name.startswith(("hf.co/", "huggingface.co/")):
        return False
    base = name.split(":", 1)[0]
    namespace, separator, model = base.partition("/")
    return bool(separator) and "/" not in model and bool(namespace) and bool(model)


def normalize_host(value: str) -> tuple[str, str]:
    """Вернуть базовый URL для HTTP и значение OLLAMA_HOST для процесса ollama."""
    raw = value.strip().rstrip("/")
    if not raw or any(char.isspace() for char in raw):
        raise ValidationError(t("Address must look like 127.0.0.1:11434 or http://127.0.0.1:11434."))
    if "://" in raw:
        scheme, separator, rest = raw.partition("://")
        if scheme not in {"http", "https"} or not separator or not rest or "/" in rest:
            raise ValidationError(t("Address must look like 127.0.0.1:11434 or http://127.0.0.1:11434."))
        hostport = rest
        if ":" not in hostport:
            hostport = f"{hostport}:11434"
        base = f"{scheme}://{hostport}"
        env_value = base if scheme == "https" else hostport
        return base, env_value
    hostport = raw if ":" in raw else f"{raw}:11434"
    if not _HOST_WITH_PORT.fullmatch(hostport):
        raise ValidationError(t("Address must look like 127.0.0.1:11434 or http://127.0.0.1:11434."))
    return f"http://{hostport}", hostport
