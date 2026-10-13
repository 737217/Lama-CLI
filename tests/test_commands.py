from lama.ollama.runner import (
    build_create_argv,
    build_push_argv,
    build_run_argv,
    build_stop_argv,
)


def test_run_argv_is_a_list_without_shell() -> None:
    argv = build_run_argv("/usr/bin/ollama", "gemma4:12b", keepalive="10m", think="false")
    assert argv == [
        "/usr/bin/ollama",
        "run",
        "gemma4:12b",
        "--keepalive",
        "10m",
        "--think",
        "false",
    ]
    assert all(isinstance(part, str) for part in argv)


def test_stop_create_and_push_argv() -> None:
    assert build_stop_argv("/usr/bin/ollama", "gemma4:12b") == [
        "/usr/bin/ollama",
        "stop",
        "gemma4:12b",
    ]
    assert build_create_argv("/usr/bin/ollama", "my-qwen", "/tmp/Modelfile") == [
        "/usr/bin/ollama",
        "create",
        "my-qwen",
        "-f",
        "/tmp/Modelfile",
    ]
    assert build_push_argv("/usr/bin/ollama", "alex/my-qwen") == [
        "/usr/bin/ollama",
        "push",
        "alex/my-qwen",
    ]
