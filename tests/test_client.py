import json

import httpx
import pytest

from lama.errors import OllamaAPIError, OllamaUnavailable
from lama.ollama.client import OllamaClient, context_length_from_info
from lama.ollama.types import PullEvent


def _client(handler) -> OllamaClient:
    return OllamaClient("http://ollama.test", transport=httpx.MockTransport(handler))


def test_empty_model_lists() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": []})
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": []})
        raise AssertionError(request.url.path)

    client = _client(handler)
    assert client.list_models() == []
    assert client.list_running() == []


def test_parses_installed_and_running_models() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "name": "gemma4:12b",
                            "size": 8697040896,
                            "modified_at": "2026-10-10T12:00:00Z",
                            "digest": "abc",
                            "details": {
                                "family": "gemma",
                                "parameter_size": "12B",
                                "quantization_level": "Q4_K_M",
                            },
                        }
                    ]
                },
            )
        if request.url.path == "/api/ps":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "name": "gemma4:12b",
                            "size": 9000,
                            "size_vram": 7000,
                            "expires_at": "2026-10-12T13:00:00Z",
                            "context_length": 4096,
                        },
                        {"name": "partial:latest"},
                    ]
                },
            )
        raise AssertionError(request.url.path)

    client = _client(handler)
    installed = client.list_models()
    assert installed[0].name == "gemma4:12b"
    assert installed[0].size == 8697040896
    assert installed[0].family == "gemma"
    assert installed[0].modified_at is not None

    running = client.list_running()
    assert running[0].size_vram == 7000
    assert running[0].cpu_bytes == 2000
    assert running[0].context_length == 4096
    assert running[1].size is None
    assert running[1].size_vram is None
    assert running[1].cpu_bytes is None
    assert running[1].context_length is None


def test_api_error_uses_server_message() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model 'missing' not found"})

    client = _client(handler)
    with pytest.raises(OllamaAPIError, match="not found") as caught:
        client.show("missing")
    assert caught.value.status_code == 404


def test_unavailable_server() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client = _client(handler)
    with pytest.raises(OllamaUnavailable, match="unreachable"):
        client.version()


def test_pull_stream_and_error_inside_stream() -> None:
    events: list[PullEvent] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["model"] == "qwen3.5:9b"
        body = "\n".join(
            [
                json.dumps({"status": "pulling manifest"}),
                json.dumps({"status": "downloading", "digest": "sha", "total": 100, "completed": 40}),
                json.dumps({"status": "success"}),
            ]
        )
        return httpx.Response(200, text=body)

    client = _client(handler)
    client.pull("qwen3.5:9b", on_progress=events.append)
    assert [event.status for event in events] == ["pulling manifest", "downloading", "success"]
    assert events[1].completed == 40
    assert events[1].total == 100


def test_pull_error_payload() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=json.dumps({"error": "file does not exist"}))

    client = _client(handler)
    with pytest.raises(OllamaAPIError, match="does not exist"):
        client.pull("nope:latest")


def test_delete_accepts_null_success_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "DELETE"
        assert json.loads(request.content) == {"model": "gemma4:12b"}
        return httpx.Response(200, content=b"null")

    _client(handler).delete("gemma4:12b")


def test_list_models_with_parameters_reads_show_without_verbose() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "name": "qwen3.8:27b-ctx-64k",
                            "size": 100,
                            "details": {"parameter_size": "26.9B", "quantization_level": "IQ2_S"},
                        },
                        {"name": "broken:latest", "size": 10, "details": {}},
                    ]
                },
            )
        if request.url.path == "/api/show":
            body = json.loads(request.content)
            assert "verbose" not in body
            if body["model"] == "broken:latest":
                return httpx.Response(500, json={"error": "show failed"})
            return httpx.Response(
                200,
                json={
                    "parameters": "num_ctx 65536\ntemperature 0.6\ntop_p 0.9\n",
                    "model_info": {"qwen35.context_length": 262144},
                },
            )
        raise AssertionError(request.url.path)

    models = _client(handler).list_models_with_parameters()
    assert models[0].parameters == {"num_ctx": "65536", "temperature": "0.6", "top_p": "0.9"}
    assert models[0].max_context == 262144
    assert models[0].parameter_size == "26.9B"
    assert models[1].name == "broken:latest"
    assert models[1].parameters == {}
    assert models[1].max_context is None


def test_context_length_from_info_uses_the_largest_architecture_value() -> None:
    assert context_length_from_info({"qwen35.context_length": 262144, "other": 1}) == 262144
    assert context_length_from_info({"note": "none"}) is None


def test_unload_sends_keep_alive_zero() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"done": True})

    client = _client(handler)
    client.unload("gemma4:12b")
    assert seen["body"] == {
        "model": "gemma4:12b",
        "prompt": "",
        "stream": False,
        "keep_alive": 0,
    }


def test_unload_targets_the_selected_remote_host() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "remote.example"
        assert request.url.port == 11434
        return httpx.Response(200, json={"done": True})

    client = OllamaClient("http://remote.example:11434", transport=httpx.MockTransport(handler))
    client.unload("gemma4:12b")
