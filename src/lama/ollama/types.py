"""Структуры ответов Ollama API. Отсутствующие поля остаются None."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class InstalledModel:
    name: str
    size: int
    modified_at: datetime | None
    digest: str = ""
    family: str = ""
    parameter_size: str = ""
    quantization: str = ""


@dataclass(frozen=True)
class RunningModel:
    name: str
    size: int | None = None
    size_vram: int | None = None
    expires_at: datetime | None = None
    context_length: int | None = None
    digest: str = ""

    @property
    def cpu_bytes(self) -> int | None:
        if self.size is None or self.size_vram is None:
            return None
        return max(self.size - self.size_vram, 0)


@dataclass(frozen=True)
class ModelShow:
    modelfile: str = ""
    parameters: str = ""
    system: str = ""
    template: str = ""
    details: dict[str, object] = field(default_factory=dict)
    capabilities: tuple[str, ...] = ()
    model_info: dict[str, object] = field(default_factory=dict)
    parameter_map: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PullEvent:
    status: str
    digest: str | None = None
    total: int | None = None
    completed: int | None = None
