"""Поиск Ollama, склейка PATH и выбор установщика без сети и без записи профиля."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from lama.config import Settings
from lama.errors import LamaError, OllamaUnavailable
from lama.ollama.runner import OllamaRunner
from lama.platform.darwin import parse_kern_procargs2, read_darwin_process
from lama.platform.host import spawn_options, uses_native_process_table
from lama.platform.install import installer_for, run_installer
from lama.platform.locate import BinaryLocation, candidate_paths, locate
from lama.platform.pathfix import append_export, merge_path, profile_file, shell_export


def test_windows_candidates_use_localappdata() -> None:
    paths = candidate_paths("windows", {"LOCALAPPDATA": r"C:\Users\me\AppData\Local"})
    assert paths == (Path(r"C:\Users\me\AppData\Local") / "Programs" / "Ollama" / "ollama.exe",)
    assert candidate_paths("windows", {}) == ()


def test_linux_and_macos_candidates() -> None:
    linux = candidate_paths("linux", {})
    assert linux == (Path("/usr/local/bin/ollama"), Path("/usr/bin/ollama"))
    darwin = candidate_paths("darwin", {})
    assert Path("/opt/homebrew/bin/ollama") in darwin
    assert Path("/Applications/Ollama.app/Contents/Resources/ollama") in darwin
    assert linux[0] in darwin


def test_locate_prefers_path_over_known_directories() -> None:
    found = locate(
        "linux",
        environ={"PATH": "/usr/bin"},
        which=lambda name: "/usr/bin/ollama" if name == "ollama" else None,
        is_file=lambda _path: True,
    )
    assert found is not None
    assert found.on_path is True
    assert found.path == Path("/usr/bin/ollama")


def test_locate_known_file_reports_whether_directory_is_on_path() -> None:
    binary = Path(r"C:\Users\me\AppData\Local\Programs\Ollama\ollama.exe")
    environ = {
        "LOCALAPPDATA": r"C:\Users\me\AppData\Local",
        "PATH": r"C:\Windows\System32",
    }
    found = locate(
        "windows",
        environ=environ,
        which=lambda _name: None,
        is_file=lambda path: path == binary,
    )
    assert found is not None
    assert found.path == binary
    assert found.on_path is False

    listed = locate(
        "windows",
        environ={**environ, "PATH": str(binary.parent)},
        which=lambda _name: None,
        is_file=lambda path: path == binary,
    )
    assert listed is not None
    assert listed.on_path is True


def test_locate_returns_none_when_nothing_exists() -> None:
    assert locate("linux", environ={}, which=lambda _name: None, is_file=lambda _path: False) is None


def test_merge_path_prepends_once_and_ignores_case() -> None:
    assert merge_path("", "/opt/bin") == "/opt/bin"
    joined = "/usr/bin" + os.pathsep + "/bin"
    assert merge_path(joined, "/opt/bin") == "/opt/bin" + os.pathsep + joined
    assert merge_path("/opt/bin" + os.pathsep + "/usr/bin", "/opt/bin") is None
    assert merge_path(r"C:\Windows", r"c:\windows") is None


def test_shell_export_and_profile_line_are_idempotent() -> None:
    line = shell_export("/tmp/it's")
    assert "it'\\''s" in line
    assert append_export("", "/opt/bin") == shell_export("/opt/bin") + "\n"
    existing = "export PATH=/usr/bin\n"
    updated = append_export(existing, "/opt/bin")
    assert updated == existing + shell_export("/opt/bin") + "\n"
    assert append_export("note", "/opt/bin") == "note\n" + shell_export("/opt/bin") + "\n"
    assert updated is not None
    assert append_export(updated, "/opt/bin") is None


def test_profile_file_follows_the_shell_name() -> None:
    home = Path("/home/me")
    assert profile_file(home, "/bin/zsh") == home / ".zshrc"
    assert profile_file(home, "bash") == home / ".bashrc"
    assert profile_file(home, "/bin/sh") == home / ".profile"


def test_installer_urls_and_commands_are_fixed() -> None:
    windows = installer_for("windows")
    linux = installer_for("linux")
    darwin = installer_for("darwin")
    assert windows.url == "https://ollama.com/install.ps1"
    assert windows.argv(Path("install.ps1")) == ["powershell", "-NoProfile", "-File", "install.ps1"]
    assert linux.url == darwin.url == "https://ollama.com/install.sh"
    assert linux.argv(Path("install.sh")) == ["sh", "install.sh"]
    assert darwin.argv(Path("install.sh")) == ["sh", "install.sh"]


def test_run_installer_uses_injected_download_and_rejects_failure(tmp_path: Path) -> None:
    seen: list[object] = []

    def download(url: str, target: Path) -> None:
        seen.append(url)
        target.write_text("echo ok\n", encoding="utf-8")

    def execute(argv: list[str]) -> int:
        seen.append(argv)
        return 0

    run_installer("linux", tmp_path, download=download, execute=execute)
    assert seen[0] == "https://ollama.com/install.sh"
    assert seen[1] == ["sh", str(tmp_path / "install.sh")]

    def fail(argv: list[str]) -> int:
        return 3

    with pytest.raises(LamaError, match="exited with code 3"):
        run_installer("windows", tmp_path, download=download, execute=fail)


def test_fake_proc_root_stays_on_the_linux_probe() -> None:
    fake = Path("/tmp/lama-proc")
    assert uses_native_process_table(fake, kind="windows") is False
    assert uses_native_process_table(fake, kind="darwin") is False
    assert uses_native_process_table(Path("/proc"), kind="linux") is False
    assert uses_native_process_table(Path("/proc"), kind="windows") is True
    assert uses_native_process_table(Path("/proc"), kind="darwin") is True


def test_spawn_options_match_the_operating_system() -> None:
    windows = spawn_options("windows")
    assert "creationflags" in windows
    assert isinstance(windows["creationflags"], int)
    assert spawn_options("linux") == {"start_new_session": True}
    assert spawn_options("darwin") == {"start_new_session": True}


def test_parse_kern_procargs2_reads_argv() -> None:
    body = b"/opt/homebrew/bin/ollama\x00\x00ollama\x00serve\x00"
    data = (2).to_bytes(4, sys.byteorder) + body
    assert parse_kern_procargs2(data) == ["ollama", "serve"]
    assert parse_kern_procargs2(b"\x00\x00") is None


def test_darwin_reader_does_not_touch_proc_on_other_systems() -> None:
    if sys.platform == "darwin":
        return
    assert read_darwin_process(1).status == "unreadable"
    assert read_darwin_process(0).status == "gone"


def test_runner_pins_the_located_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "lama.ollama.runner.locate_ollama",
        lambda: BinaryLocation(Path("/opt/homebrew/bin/ollama"), False),
    )
    runner = OllamaRunner(Settings(host="http://127.0.0.1:11434", env_host="127.0.0.1:11434"))
    assert runner.binary() == str(Path("/opt/homebrew/bin/ollama"))


def test_runner_keeps_an_injected_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden() -> None:
        raise AssertionError("locate should not run")

    monkeypatch.setattr("lama.ollama.runner.locate_ollama", forbidden)
    runner = OllamaRunner(
        Settings(host="http://127.0.0.1:11434", env_host="127.0.0.1:11434"),
        binary="/usr/bin/ollama",
    )
    assert runner.binary() == "/usr/bin/ollama"


def test_runner_reports_a_missing_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lama.ollama.runner.locate_ollama", lambda: None)
    runner = OllamaRunner(Settings(host="http://127.0.0.1:11434", env_host="127.0.0.1:11434"))
    with pytest.raises(OllamaUnavailable, match="lama server install"):
        runner.binary()
