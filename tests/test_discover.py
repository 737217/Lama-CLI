import os
from pathlib import Path

from lama.ollama.discover import client_base_for, discover_ollama_installs, read_ollama_version, server_env_for


def _executable(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    path.chmod(0o755)


def test_discover_keeps_path_order_and_skips_duplicates(tmp_path: Path) -> None:
    first = tmp_path / "first" / "ollama"
    second = tmp_path / "second" / "ollama"
    _executable(first)
    _executable(second)
    linked = tmp_path / "linked"
    linked.mkdir()
    (linked / "ollama").symlink_to(first)
    versions = {str(first.resolve()): "0.6.2", str(second.resolve()): "0.11.4"}
    found = discover_ollama_installs(
        path=os.pathsep.join((str(first.parent), str(linked), str(second.parent))),
        directories=[],
        version_of=versions.get,
        proc_root=tmp_path / "no-proc",
        remembered=[],
        unit_records=[],
    )
    assert [(item.path, item.version) for item in found] == [
        (str(first.resolve()), "0.6.2"),
        (str(second.resolve()), "0.11.4"),
    ]


def test_discover_skips_non_executable_and_adds_extra_directories(tmp_path: Path) -> None:
    blocked = tmp_path / "path" / "ollama"
    blocked.parent.mkdir()
    blocked.write_text("not executable", encoding="utf-8")
    blocked.chmod(0o644)
    extra = tmp_path / "opt" / "ollama"
    _executable(extra)
    found = discover_ollama_installs(
        path=str(blocked.parent),
        directories=[extra.parent],
        version_of=lambda _path: None,
        proc_root=tmp_path / "no-proc",
        remembered=[],
        unit_records=[],
    )
    assert [item.path for item in found] == [str(extra.resolve())]
    assert found[0].version is None


def test_discover_includes_a_running_serve_binary(tmp_path: Path) -> None:
    binary = tmp_path / "patched" / "bin" / "ollama"
    _executable(binary)
    proc = tmp_path / "proc" / "77"
    proc.mkdir(parents=True)
    (proc / "cmdline").write_bytes(str(binary).encode() + b"\0serve\0")
    (proc / "exe").symlink_to(binary)
    found = discover_ollama_installs(
        path="",
        directories=[],
        proc_root=tmp_path / "proc",
        remembered=[],
        unit_records=[],
        version_of=lambda _path: "patched",
    )
    assert [(item.path, item.version) for item in found] == [(str(binary.resolve()), "patched")]


def test_ports_come_from_the_unit_and_the_running_process(tmp_path: Path) -> None:
    stock = tmp_path / "stock" / "ollama"
    patched = tmp_path / "patched" / "bin" / "ollama"
    _executable(stock)
    _executable(patched)
    proc = tmp_path / "proc" / "77"
    proc.mkdir(parents=True)
    (proc / "cmdline").write_bytes(str(patched).encode() + b"\0serve\0")
    (proc / "exe").symlink_to(patched)
    (proc / "environ").write_bytes(b"PATH=/usr/bin\0OLLAMA_HOST=127.0.0.1:11435\0")
    found = discover_ollama_installs(
        path=str(stock.parent),
        directories=[],
        proc_root=tmp_path / "proc",
        remembered=[],
        unit_records=[(str(stock), "0.0.0.0:11434"), (str(patched), "127.0.0.1:11434")],
        version_of=lambda path: "stock" if path == str(stock.resolve()) else "patched",
    )
    by_path = {item.path: item for item in found}
    assert by_path[str(stock.resolve())].listen == "0.0.0.0:11434"
    assert by_path[str(patched.resolve())].listen == "127.0.0.1:11435"
    assert client_base_for("0.0.0.0:11434") == "http://127.0.0.1:11434"
    assert server_env_for("0.0.0.0:11434") == "0.0.0.0:11434"
    assert server_env_for("127.0.0.1:11435") == "127.0.0.1:11435"


def test_read_ollama_version_extracts_the_number(tmp_path: Path) -> None:
    binary = tmp_path / "ollama"
    binary.write_text("#!/bin/sh\nprintf '%s\\n' 'ollama version is 0.11.4'\n", encoding="utf-8")
    binary.chmod(0o755)
    assert read_ollama_version(str(binary)) == "0.11.4"
    assert read_ollama_version(str(tmp_path / "missing")) is None
