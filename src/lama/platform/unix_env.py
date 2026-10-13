"""Строка PATH в профиле оболочки. Не вызывается, пока пользователь не подтвердил добавление."""

from __future__ import annotations

import os
from pathlib import Path

from lama.platform.pathfix import append_export, merge_path, profile_file


def persist_user_path(directory: Path) -> bool:
    path = profile_file(Path.home(), os.environ.get("SHELL", ""))
    text = ""
    if path.is_file():
        text = path.read_text(encoding="utf-8")
    updated = append_export(text, str(directory))
    changed = False
    if updated is not None:
        path.write_text(updated, encoding="utf-8")
        changed = True
    merged = merge_path(os.environ.get("PATH", ""), str(directory))
    if merged is not None:
        os.environ["PATH"] = merged
    return changed
