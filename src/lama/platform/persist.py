"""Запись PATH в профиль текущей ОС после подтверждения пользователя."""

from __future__ import annotations

import sys
from pathlib import Path

if sys.platform == "win32":
    from lama.platform.windows_env import persist_user_path as _persist_user_path
else:
    from lama.platform.unix_env import persist_user_path as _persist_user_path


def publish_path(directory: Path) -> bool:
    """Добавить каталог в PATH этого процесса и в постоянный PATH пользователя."""
    return _persist_user_path(directory)
