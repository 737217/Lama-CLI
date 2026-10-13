"""Официальный установщик Ollama. Адрес фиксирован и не берётся из ввода пользователя."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx

from lama.errors import LamaError
from lama.i18n import t
from lama.platform.model import PlatformKind, assert_never

_WINDOWS_URL = "https://ollama.com/install.ps1"
_UNIX_URL = "https://ollama.com/install.sh"


@dataclass(frozen=True)
class Installer:
    url: str
    filename: str

    def argv(self, script: Path) -> list[str]:
        if self.filename.endswith(".ps1"):
            return ["powershell", "-NoProfile", "-File", str(script)]
        return ["sh", str(script)]


def installer_for(kind: PlatformKind) -> Installer:
    if kind == "windows":
        return Installer(_WINDOWS_URL, "install.ps1")
    if kind == "linux" or kind == "darwin":
        return Installer(_UNIX_URL, "install.sh")
    return assert_never(kind)


def download_installer(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    timeout = httpx.Timeout(30.0, read=None)
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=timeout) as response:
            response.raise_for_status()
            with target.open("wb") as handle:
                for chunk in response.iter_bytes():
                    handle.write(chunk)
    except (httpx.HTTPError, OSError) as exc:
        raise LamaError(t("Could not download the Ollama installer: {error}", error=exc)) from exc


def run_installer(
    kind: PlatformKind,
    directory: Path,
    *,
    download: Callable[[str, Path], None] = download_installer,
    execute: Callable[[list[str]], int],
) -> None:
    spec = installer_for(kind)
    script = directory / spec.filename
    download(spec.url, script)
    code = execute(spec.argv(script))
    if code != 0:
        raise LamaError(t("The official installer exited with code {code}.", code=code))
