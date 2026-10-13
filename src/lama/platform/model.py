"""Вид операционной системы, для которой выбирается поведение."""

from __future__ import annotations

from typing import Literal, Never

PlatformKind = Literal["windows", "linux", "darwin"]


def assert_never(kind: Never) -> Never:
    raise AssertionError(kind)
