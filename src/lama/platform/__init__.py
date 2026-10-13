"""Выбор поведения по текущей ОС: поиск Ollama, установка и PATH."""

from lama.platform.host import (
    current_kind,
    locate_ollama,
    open_catalog,
    spawn_options,
    uses_native_process_table,
)
from lama.platform.install import Installer, installer_for, run_installer
from lama.platform.locate import BinaryLocation
from lama.platform.persist import publish_path

__all__ = [
    "BinaryLocation",
    "Installer",
    "current_kind",
    "installer_for",
    "locate_ollama",
    "open_catalog",
    "publish_path",
    "run_installer",
    "spawn_options",
    "uses_native_process_table",
]
