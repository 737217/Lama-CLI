"""Генерация Modelfile из параметров, которые реально принимает Ollama."""

from __future__ import annotations

from lama.errors import ValidationError
from lama.i18n import t
from lama.validation import PARAMETER_FIELDS

PARAMETER_ORDER: tuple[str, ...] = (
    "num_ctx",
    "num_batch",
    "num_gpu",
    "temperature",
    "top_p",
    "top_k",
    "repeat_penalty",
    "num_predict",
    "seed",
)


def render_modelfile(
    base: str,
    parameters: dict[str, int | float],
    system: str | None = None,
) -> str:
    """Собрать Modelfile. keep_alive сюда не входит: это параметр запуска, не файла."""
    if "keep_alive" in parameters:
        raise ValidationError(t("keep_alive cannot be written to a Modelfile. It is set at launch."))
    unknown = [name for name in parameters if name not in PARAMETER_FIELDS]
    if unknown:
        raise ValidationError(t("Unsupported Modelfile parameter: {names}.", names=", ".join(unknown)))
    source = base.strip()
    if not source or "\n" in source or "\r" in source:
        raise ValidationError(t("Base model or GGUF path is invalid."))
    lines = [f"FROM {source}"]
    for name in PARAMETER_ORDER:
        if name not in parameters:
            continue
        lines.append(f"PARAMETER {name} {_format_value(name, parameters[name])}")
    if system:
        if '"""' in system:
            raise ValidationError(t('System prompt must not contain the sequence """.'))
        lines.append(f'SYSTEM """{system}"""')
    return "\n".join(lines) + "\n"


def _format_value(name: str, value: int | float) -> str:
    if PARAMETER_FIELDS[name]["kind"] == "int":
        return str(int(value))
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return text or "0"
