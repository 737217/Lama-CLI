"""HTTP-клиент локального API Ollama."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from datetime import datetime

import httpx

from lama.errors import OllamaAPIError, OllamaUnavailable
from lama.i18n import t
from lama.ollama.types import InstalledModel, ModelShow, PullEvent, RunningModel

ProgressCallback = Callable[[PullEvent], None]


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        *,
        transport: httpx.BaseTransport | None = None,
        timeout: httpx.Timeout | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout or httpx.Timeout(connect=5.0, read=60.0, write=30.0, pool=5.0)
        self._client = httpx.Client(
            base_url=self.base_url,
            transport=transport,
            timeout=self._timeout,
        )

    def close(self) -> None:
        self._client.close()

    def version(self) -> str:
        payload = self._request("GET", "/api/version")
        version = payload.get("version")
        if not isinstance(version, str) or not version:
            raise OllamaAPIError(t("Ollama did not return a server version."))
        return version

    def list_models(self) -> list[InstalledModel]:
        payload = self._request("GET", "/api/tags")
        items = payload.get("models") or []
        if not isinstance(items, list):
            raise OllamaAPIError(t("Unexpected response from /api/tags."))
        models: list[InstalledModel] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            details = item.get("details") if isinstance(item.get("details"), dict) else {}
            name = _text(item.get("name")) or _text(item.get("model"))
            if not name:
                continue
            models.append(
                InstalledModel(
                    name=name,
                    size=_required_int(item.get("size")),
                    modified_at=_parse_datetime(item.get("modified_at")),
                    digest=_text(item.get("digest")),
                    family=_text(details.get("family")),
                    parameter_size=_text(details.get("parameter_size")),
                    quantization=_text(details.get("quantization_level")),
                )
            )
        return models

    def list_running(self) -> list[RunningModel]:
        payload = self._request("GET", "/api/ps")
        items = payload.get("models") or []
        if not isinstance(items, list):
            raise OllamaAPIError(t("Unexpected response from /api/ps."))
        running: list[RunningModel] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            name = _text(item.get("name")) or _text(item.get("model"))
            if not name:
                continue
            running.append(
                RunningModel(
                    name=name,
                    size=_optional_int(item, "size"),
                    size_vram=_optional_int(item, "size_vram"),
                    expires_at=_parse_datetime(item.get("expires_at")) if "expires_at" in item else None,
                    context_length=_optional_int(item, "context_length"),
                    digest=_text(item.get("digest")),
                )
            )
        return running

    def show(self, model: str) -> ModelShow:
        payload = self._request("POST", "/api/show", json={"model": model, "verbose": True})
        parameters = _text(payload.get("parameters"))
        capabilities = payload.get("capabilities") or []
        if not isinstance(capabilities, list):
            capabilities = []
        details = payload.get("details") if isinstance(payload.get("details"), dict) else {}
        model_info = payload.get("model_info") if isinstance(payload.get("model_info"), dict) else {}
        return ModelShow(
            modelfile=_text(payload.get("modelfile")),
            parameters=parameters,
            system=_text(payload.get("system")),
            template=_text(payload.get("template")),
            details=details,
            capabilities=tuple(str(item) for item in capabilities),
            model_info=model_info,
            parameter_map=parse_parameter_block(parameters),
        )

    def pull(self, model: str, on_progress: ProgressCallback | None = None) -> None:
        timeout = httpx.Timeout(connect=5.0, read=None, write=30.0, pool=5.0)
        try:
            with self._client.stream(
                "POST",
                "/api/pull",
                json={"model": model, "stream": True},
                timeout=timeout,
            ) as response:
                if response.status_code >= 400:
                    response.read()
                _raise_for_status(response)
                for line in response.iter_lines():
                    if not line:
                        continue
                    payload = json.loads(line)
                    if not isinstance(payload, dict):
                        continue
                    if payload.get("error"):
                        raise OllamaAPIError(str(payload["error"]))
                    event = PullEvent(
                        status=_text(payload.get("status")) or t("downloading"),
                        digest=_text(payload.get("digest")) or None,
                        total=_optional_int(payload, "total"),
                        completed=_optional_int(payload, "completed"),
                    )
                    if on_progress is not None:
                        on_progress(event)
        except httpx.ConnectError as exc:
            raise OllamaUnavailable(t("Ollama is unreachable at {url}.", url=self.base_url)) from exc
        except httpx.TimeoutException as exc:
            raise OllamaUnavailable(t("Ollama did not respond in time ({url}).", url=self.base_url)) from exc
        except json.JSONDecodeError as exc:
            raise OllamaAPIError(t("Ollama returned an invalid download progress stream.")) from exc

    def load(self, model: str, keep_alive: str, options: dict[str, int | float] | None = None) -> None:
        body: dict[str, object] = {
            "model": model,
            "prompt": "",
            "stream": False,
            "keep_alive": keep_alive,
        }
        if options:
            body["options"] = options
        timeout = httpx.Timeout(connect=5.0, read=600.0, write=30.0, pool=5.0)
        self._request("POST", "/api/generate", json=body, timeout=timeout)

    def unload(self, model: str) -> None:
        """Выгрузить модель через документированный keep_alive=0, не останавливая сервер."""
        timeout = httpx.Timeout(connect=5.0, read=120.0, write=30.0, pool=5.0)
        self._request(
            "POST",
            "/api/generate",
            json={"model": model, "prompt": "", "stream": False, "keep_alive": 0},
            timeout=timeout,
        )

    def copy(self, source: str, destination: str) -> None:
        self._request(
            "POST",
            "/api/copy",
            json={"source": source, "destination": destination},
        )

    def delete(self, model: str) -> None:
        self._request("DELETE", "/api/delete", json={"model": model})

    def library_tags(self, name: str) -> list[str]:
        """Теги модели из реестра Ollama. Пустой список, если реестр недоступен."""
        base = name.split(":", 1)[0]
        if base.startswith(("hf.co/", "huggingface.co/")) or not base:
            return []
        if "/" in base:
            url = f"https://registry.ollama.ai/v2/{base}/tags/list"
        else:
            url = f"https://registry.ollama.ai/v2/library/{base}/tags/list"
        try:
            response = httpx.get(url, timeout=8.0)
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, json.JSONDecodeError, ValueError):
            return []
        tags = payload.get("tags") if isinstance(payload, dict) else None
        if not isinstance(tags, list):
            return []
        return [str(tag) for tag in tags if isinstance(tag, str) and tag]

    def _request(self, method: str, path: str, **kwargs: object) -> dict[str, object]:
        try:
            response = self._client.request(method, path, **kwargs)  # type: ignore[arg-type]
        except httpx.ConnectError as exc:
            raise OllamaUnavailable(t("Ollama is unreachable at {url}.", url=self.base_url)) from exc
        except httpx.TimeoutException as exc:
            raise OllamaUnavailable(t("Ollama did not respond in time ({url}).", url=self.base_url)) from exc
        _raise_for_status(response)
        if not response.content:
            return {}
        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise OllamaAPIError(t("Ollama returned a response that is not JSON.")) from exc
        if not isinstance(payload, dict):
            raise OllamaAPIError(t("Ollama returned unexpected JSON."))
        if payload.get("error"):
            raise OllamaAPIError(str(payload["error"]))
        return payload


def parse_parameter_block(text: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, separator, value = stripped.partition(" ")
        if not separator:
            continue
        key = key.strip()
        if key in parsed:
            continue
        parsed[key] = value.strip().strip('"')
    return parsed


def iter_pull_events(lines: Iterator[str]) -> Iterator[PullEvent]:
    for line in lines:
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict) and payload.get("error"):
            raise OllamaAPIError(str(payload["error"]))
        if not isinstance(payload, dict):
            continue
        yield PullEvent(
            status=_text(payload.get("status")) or t("downloading"),
            digest=_text(payload.get("digest")) or None,
            total=_optional_int(payload, "total"),
            completed=_optional_int(payload, "completed"),
        )


def _raise_for_status(response: httpx.Response) -> None:
    if response.status_code < 400:
        return
    message = t("Ollama returned HTTP {code}.", code=response.status_code)
    try:
        payload = response.json()
    except (json.JSONDecodeError, ValueError):
        payload = None
    if isinstance(payload, dict) and payload.get("error"):
        message = str(payload["error"])
    elif response.text:
        message = response.text.strip() or message
    raise OllamaAPIError(message, status_code=response.status_code)


def _parse_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    text = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _optional_int(payload: dict[str, object], key: str) -> int | None:
    if key not in payload or payload[key] is None:
        return None
    try:
        return int(payload[key])  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _required_int(value: object) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def _text(value: object) -> str:
    return value if isinstance(value, str) else ""
