"""Запуск штатных команд ollama списком аргументов, без shell."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from lama.config import Settings
from lama.errors import OllamaCommandError, OllamaUnavailable
from lama.i18n import t
from lama.platform.host import locate_ollama


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str
    argv: list[str]


def build_run_argv(
    binary: str,
    model: str,
    *,
    keepalive: str | None = None,
    think: str | None = None,
    hidethinking: bool = False,
    verbose: bool = False,
) -> list[str]:
    argv = [binary, "run", model]
    if keepalive is not None:
        argv.extend(["--keepalive", keepalive])
    if think is not None:
        argv.extend(["--think", think])
    if hidethinking:
        argv.append("--hidethinking")
    if verbose:
        argv.append("--verbose")
    return argv


def build_stop_argv(binary: str, model: str) -> list[str]:
    return [binary, "stop", model]


def build_create_argv(binary: str, model: str, modelfile: str | Path) -> list[str]:
    return [binary, "create", model, "-f", str(modelfile)]


def build_push_argv(binary: str, model: str) -> list[str]:
    return [binary, "push", model]


def build_show_argv(binary: str, model: str, *, modelfile: bool = False) -> list[str]:
    argv = [binary, "show", model]
    if modelfile:
        argv.append("--modelfile")
    return argv


def build_cp_argv(binary: str, source: str, destination: str) -> list[str]:
    return [binary, "cp", source, destination]


def build_rm_argv(binary: str, model: str) -> list[str]:
    return [binary, "rm", model]


def build_signin_argv(binary: str) -> list[str]:
    return [binary, "signin"]


def build_serve_argv(binary: str) -> list[str]:
    return [binary, "serve"]


class OllamaRunner:
    def __init__(self, settings: Settings, *, binary: str | None = None) -> None:
        self.settings = settings
        self._binary = binary

    def pin(self, binary: str) -> None:
        self._binary = binary

    def binary(self) -> str:
        if self._binary:
            return self._binary
        located = locate_ollama()
        if located is None:
            raise OllamaUnavailable(
                t("The ollama command was not found. Install it from the Server menu or with lama server install.")
            )
        self._binary = str(located.path)
        return self._binary

    def environment(self) -> dict[str, str]:
        env = os.environ.copy()
        env["OLLAMA_HOST"] = self.settings.env_host
        return env

    def run_chat(
        self,
        model: str,
        *,
        keepalive: str | None = None,
        think: str | None = None,
    ) -> None:
        argv = build_run_argv(self.binary(), model, keepalive=keepalive, think=think)
        self.execute(argv, inherit=True)

    def stop(self, model: str) -> None:
        self.execute(build_stop_argv(self.binary(), model))

    def create(self, model: str, modelfile: str | Path) -> None:
        self.execute(build_create_argv(self.binary(), model, modelfile), inherit=True)

    def push(self, model: str) -> None:
        self.execute(build_push_argv(self.binary(), model), inherit=True)

    def signin(self) -> None:
        self.execute(build_signin_argv(self.binary()), inherit=True)

    def help_text(self) -> str:
        result = self.capture([self.binary(), "--help"])
        return result.stdout or result.stderr

    def version_text(self) -> str:
        result = self.capture([self.binary(), "--version"])
        return (result.stdout or result.stderr).strip()

    def execute(self, argv: list[str], *, inherit: bool = False) -> CommandResult:
        if inherit:
            completed = subprocess.run(argv, env=self.environment(), check=False)
            result = CommandResult(completed.returncode, "", "", argv)
        else:
            completed = subprocess.run(
                argv,
                env=self.environment(),
                check=False,
                capture_output=True,
                text=True,
            )
            result = CommandResult(
                completed.returncode,
                completed.stdout,
                completed.stderr,
                argv,
            )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            message = detail or t("The ollama command exited with code {code}.", code=result.returncode)
            raise OllamaCommandError(message, returncode=result.returncode, argv=argv)
        return result

    def capture(self, argv: list[str]) -> CommandResult:
        completed = subprocess.run(
            argv,
            env=self.environment(),
            check=False,
            capture_output=True,
            text=True,
        )
        return CommandResult(completed.returncode, completed.stdout, completed.stderr, argv)
