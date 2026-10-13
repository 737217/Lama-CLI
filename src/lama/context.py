"""Фабрики клиента и runner, чтобы команды и тесты собирали их в одном месте."""

from __future__ import annotations

from lama.config import Settings
from lama.ollama.client import OllamaClient
from lama.ollama.runner import OllamaRunner


def build_client(settings: Settings) -> OllamaClient:
    return OllamaClient(settings.host)


def build_runner(settings: Settings) -> OllamaRunner:
    return OllamaRunner(settings)
