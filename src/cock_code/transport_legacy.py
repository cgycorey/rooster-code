from __future__ import annotations

from dataclasses import dataclass
import os
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlparse

import httpx


class TransportHTTPError(Exception):
    def __init__(self, status_code: int, body: str):
        super().__init__(body)
        self.status_code = status_code


@dataclass
class RawUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


@dataclass
class RawMessageResponse:
    content: list[Any]
    model: str
    stop_reason: str
    usage: RawUsage


def _messages_url(base_url: str) -> str:
    stripped = base_url.rstrip("/")
    path = urlparse(stripped).path.rstrip("/")
    if path.endswith("/messages"):
        return stripped
    if path.endswith("/v1"):
        return f"{stripped}/messages"
    return f"{stripped}/v1/messages"


def _to_block(block: dict[str, Any]) -> Any:
    return SimpleNamespace(**block)


def _to_usage(payload: dict[str, Any]) -> RawUsage:
    return RawUsage(
        input_tokens=payload.get("input_tokens", 0),
        output_tokens=payload.get("output_tokens", 0),
        cache_creation_input_tokens=payload.get("cache_creation_input_tokens", 0),
        cache_read_input_tokens=payload.get("cache_read_input_tokens", 0),
    )


class _RawMessagesAPI:
    def __init__(self, client: "RawAnthropicHTTPClient"):
        self._client = client

    async def create(self, **kwargs: Any) -> RawMessageResponse:
        try:
            response = await self._client._http_client.post(
                _messages_url(self._client.base_url),
                headers=self._client._headers,
                json=kwargs,
            )
        except httpx.TimeoutException as exc:
            raise TransportHTTPError(504, f"Model request timed out: {exc}") from exc

        if response.status_code >= 400:
            raise TransportHTTPError(response.status_code, response.text)

        payload = response.json()
        return RawMessageResponse(
            content=[_to_block(block) for block in payload.get("content", [])],
            model=payload.get("model", kwargs.get("model", "")),
            stop_reason=payload.get("stop_reason", ""),
            usage=_to_usage(payload.get("usage", {})),
        )


class RawAnthropicHTTPClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str,
        default_headers: dict[str, str] | None = None,
        http_client: httpx.AsyncClient | None = None,
    ):
        self.base_url = base_url
        timeout_seconds = float(os.environ.get("COCK_CODE_HTTP_TIMEOUT_SECONDS", "45"))
        self._http_client = http_client or httpx.AsyncClient(timeout=timeout_seconds)
        self._headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        if api_key:
            self._headers["x-api-key"] = api_key
            self._headers["authorization"] = f"Bearer {api_key}"
        if default_headers:
            self._headers.update(default_headers)
        self.messages = _RawMessagesAPI(self)

    async def close(self) -> None:
        await self._http_client.aclose()
