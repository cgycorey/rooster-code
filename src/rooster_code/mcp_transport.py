"""HTTP / SSE MCP transport client.

The SDK's connect_mcp_server only handles stdio — SSE and HTTP are stubs
that return tools=[]. This module fills the gap so SSE/HTTP MCP servers work
in our daemon and CLI without touching the SDK.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import Any
from urllib.parse import urlparse

import httpx
from open_agent_sdk.types import BaseTool, ToolInputSchema, ToolResult

log = logging.getLogger("rooster.mcp_transport")


class McpToolWrapper(BaseTool):

    def __init__(self, name: str, description: str, input_schema_dict: dict[str, Any],
                 server_name: str, tool_name: str, transport_client: SseClient):
        self._name = name
        self._description = description
        self._input_schema = ToolInputSchema(properties=input_schema_dict.get("properties", {}),
                                             required=input_schema_dict.get("required", []))
        self.server_name = server_name
        self.tool_name = tool_name
        self._client = transport_client

    def is_read_only(self, input: dict[str, Any] | None = None) -> bool:
        return False

    def is_concurrency_safe(self, input: dict[str, Any] | None = None) -> bool:
        return False

    async def call(self, input: dict[str, Any], context: Any) -> Any:
        request_id = self._client.next_request_id()
        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "tools/call",
            "params": {"name": self.tool_name, "arguments": input},
        }
        result = await self._client.send_request(request)
        content = result.get("content", [])
        texts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
        return ToolResult(tool_use_id="", content="\n".join(texts) if texts else json.dumps(result))


class SseClient:
    """MCP SSE transport client.

    Uses a single persistent GET stream to /sse for both endpoint
    discovery and response reading. Requests are POSTed to the
    messages endpoint; responses arrive on the GET stream
    correlated by request ID.
    """

    def __init__(self, url: str, timeout: float = 30.0):
        self._sse_url = url.rstrip("/")
        self._timeout = timeout
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(timeout))
        self._next_id = 1
        self._messages_url: str | None = None
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._reader_task: asyncio.Task[None] | None = None
        self._endpoint_discovered = asyncio.Event()


    def next_request_id(self) -> int:
        rid = self._next_id
        self._next_id += 1
        return rid

    async def _sse_stream_reader(self) -> None:
        """Persistent GET /sse that discovers the endpoint then reads responses.
        Reconnects automatically on transient stream failures with backoff."""
        retry = 0
        _MAX_RETRIES = 3
        while True:
            try:
                async with self._http.stream("GET", self._sse_url, headers={"Accept": "text/event-stream"}) as resp:
                    resp.raise_for_status()
                    retry = 0  # reset on successful connection
                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        payload = line[6:].strip()

                        if self._messages_url is None:
                            path = payload
                            if path.startswith("/"):
                                parsed = urlparse(self._sse_url)
                                self._messages_url = f"{parsed.scheme}://{parsed.netloc}{path}"
                            else:
                                self._messages_url = path
                            self._endpoint_discovered.set()
                            log.debug("SSE messages endpoint: %s", self._messages_url)
                            continue

                        try:
                            event = json.loads(payload)
                        except json.JSONDecodeError:
                            continue

                        req_id = event.get("id")
                        if req_id is not None and req_id in self._pending:
                            fut = self._pending.pop(req_id)
                            if not fut.done():
                                if "result" in event:
                                    fut.set_result(event["result"])
                                elif "error" in event:
                                    fut.set_exception(RuntimeError(f"MCP error: {event['error']}"))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                retry += 1
                if retry > _MAX_RETRIES:
                    log.error("SSE reader failed after %d retries: %s", _MAX_RETRIES, exc, exc_info=True)
                    if not self._endpoint_discovered.is_set():
                        self._endpoint_discovered.set()
                    for fut in self._pending.values():
                        if not fut.done():
                            fut.set_exception(exc)
                    self._pending.clear()
                    return
                log.warning("SSE reader disconnected (retry %d/%d): %s", retry, _MAX_RETRIES, exc)
                # Fail orphaned pending requests — the response was lost
                for fut in self._pending.values():
                    if not fut.done():
                        fut.set_exception(exc)
                self._pending.clear()
                self._messages_url = None
                await asyncio.sleep(min(2 ** retry, 10))

    async def initialize(self) -> dict[str, Any]:
        self._reader_task = asyncio.create_task(self._sse_stream_reader())
        await asyncio.wait_for(self._endpoint_discovered.wait(), timeout=self._timeout)
        init_request = {
            "jsonrpc": "2.0",
            "id": self.next_request_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "rooster-code", "version": "1.0"},
            },
        }
        result = await self.send_request(init_request)
        notif = {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}
        await self.send_notification(notif)
        return result
    async def list_tools(self) -> list[dict[str, Any]]:
        req = {"jsonrpc": "2.0", "id": self.next_request_id(), "method": "tools/list", "params": {}}
        result = await self.send_request(req)
        return result.get("tools", [])

    async def send_request(self, request: dict[str, Any]) -> dict[str, Any]:
        if self._messages_url is None:
            raise ConnectionError("MCP transport not connected")
        req_id = request.get("id")
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[dict[str, Any]] = loop.create_future()
        if req_id is not None:
            self._pending[req_id] = fut
        try:
            resp = await self._http.post(self._messages_url, json=request, headers={"Content-Type": "application/json"})
        except Exception:
            if req_id is not None and req_id in self._pending:
                del self._pending[req_id]
            raise
        if resp.status_code >= 400:
            if req_id is not None and req_id in self._pending:
                del self._pending[req_id]
            resp.raise_for_status()
        if resp.status_code == 200:
            content_type = resp.headers.get("content-type", "")
            if "text/event-stream" in content_type:
                try:
                    return self._parse_inline_sse(resp)
                finally:
                    if req_id is not None and req_id in self._pending:
                        del self._pending[req_id]
            body = resp.json()
            if req_id is not None and req_id in self._pending:
                del self._pending[req_id]
            return body.get("result", body) if isinstance(body, dict) else body
        try:
            return await asyncio.wait_for(fut, timeout=self._timeout)
        except asyncio.TimeoutError:
            if req_id is not None and req_id in self._pending:
                del self._pending[req_id]
            raise

    async def send_notification(self, notification: dict[str, Any]) -> None:
        if self._messages_url is None:
            raise ConnectionError("MCP transport not connected")
        await self._http.post(self._messages_url, json=notification, headers={"Content-Type": "application/json"})

    def _parse_inline_sse(self, resp: httpx.Response) -> dict[str, Any]:
        for line in resp.text.split("\n"):
            if line.startswith("data: "):
                try:
                    event = json.loads(line[6:])
                    if "result" in event:
                        return event["result"]
                    if "error" in event:
                        raise RuntimeError(f"MCP error: {event['error']}")
                except json.JSONDecodeError:
                    continue
        return {}

    async def close(self) -> None:
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader_task
        await self._http.aclose()


async def connect_http_mcp(server_name: str, config: dict[str, Any]) -> list[Any]:
    url = config.get("url", "")
    if not url:
        log.warning("MCP %s: no url, skipping", server_name)
        return []

    client = SseClient(url)
    try:
        await client.initialize()
        mcp_tools = await client.list_tools()
    except Exception as exc:
        log.warning("MCP %s: connection failed: %s", server_name, exc)
        await client.close()
        return []

    tools: list[Any] = []
    for mt in mcp_tools:
        tool_name = mt.get("name", "")
        tool_desc = mt.get("description", "")
        tool_schema = mt.get("inputSchema", {})
        wrapper_name = f"mcp__{server_name}__{tool_name}"
        wrapper = McpToolWrapper(
            name=wrapper_name,
            description=tool_desc,
            input_schema_dict=tool_schema,
            server_name=server_name,
            tool_name=tool_name,
            transport_client=client,
        )
        tools.append(wrapper)

    log.info("MCP %s: connected, %d tools", server_name, len(tools))
    return tools


def split_mcp_servers(mcp_servers: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    stdio: dict[str, Any] = {}
    remote: dict[str, Any] = {}
    for name, cfg in mcp_servers.items():
        if not isinstance(cfg, dict):
            stdio[name] = cfg
            continue
        transport = cfg.get("type", "stdio")
        if transport == "stdio":
            stdio[name] = cfg
        else:
            remote[name] = cfg
    return stdio, remote
