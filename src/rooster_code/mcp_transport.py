"""HTTP / SSE MCP transport client.

The SDK's connect_mcp_server only handles stdio — SSE and HTTP are stubs
that return tools=[]. This module fills the gap so SSE/HTTP MCP servers work
in our daemon and CLI without touching the SDK.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx
from open_agent_sdk.types import BaseTool, ToolInputSchema, ToolResult

log = logging.getLogger("rooster.mcp_transport")


class McpToolWrapper(BaseTool):
    """Thin tool wrapper compatible with SDK BaseTool protocol."""

    def __init__(self, name: str, description: str, input_schema_dict: dict[str, Any],
                 server_name: str, tool_name: str, url: str, transport: str,
                 sse_client: "SseClient | None" = None):
        self._name = name
        self._description = description
        self._input_schema = ToolInputSchema(properties=input_schema_dict.get("properties", {}),
                                             required=input_schema_dict.get("required", []))
        self.server_name = server_name
        self.tool_name = tool_name
        self.url = url
        self.transport = transport
        self._sse_client = sse_client
        self._next_id = 1

    def is_read_only(self, input: dict[str, Any] | None = None) -> bool:
        return False

    def is_concurrency_safe(self, input: dict[str, Any] | None = None) -> bool:
        return False

    async def call(self, input: dict[str, Any], context: Any) -> Any:
        request_id = self._next_id
        self._next_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "tools/call",
            "params": {"name": self.tool_name, "arguments": input},
        }
        client = self._sse_client or SseClient(self.url)
        result = await client.send_request(request)
        content = result.get("content", [])
        texts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
        return ToolResult(tool_use_id="", content="\n".join(texts) if texts else json.dumps(result))


class SseClient:
    """Minimal JSON-RPC over HTTP + SSE client for MCP."""

    def __init__(self, url: str, timeout: float = 30.0):
        self._url = url.rstrip("/")
        self._timeout = timeout
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(timeout))

    async def initialize(self) -> dict[str, Any]:
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "rooster-code", "version": "1.0"},
            },
        }
        result = await self.send_request(init_request)
        # Send initialized notification
        notif = {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}
        await self.send_notification(notif)
        return result

    async def list_tools(self) -> list[dict[str, Any]]:
        req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
        result = await self.send_request(req)
        return result.get("tools", [])

    async def send_request(self, request: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        resp = await self._http.post(self._url, json=request, headers=headers)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")

        if "text/event-stream" in content_type:
            return await self._parse_sse_response(resp)
        return resp.json().get("result", {})

    async def send_notification(self, notification: dict[str, Any]) -> None:
        headers = {"Content-Type": "application/json"}
        await self._http.post(self._url, json=notification, headers=headers)

    async def _parse_sse_response(self, resp: httpx.Response) -> dict[str, Any]:
        async for line in resp.aiter_lines():
            if line.startswith("data: "):
                data = line[6:]
                try:
                    event = json.loads(data)
                    if "result" in event:
                        return event["result"]
                    if "error" in event:
                        raise RuntimeError(f"MCP error: {event['error']}")
                except json.JSONDecodeError:
                    continue
        return {}

    async def close(self) -> None:
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
            url=url,
            transport="sse" if "sse" in config.get("type", "") else "http",
            sse_client=client,
        )
        tools.append(wrapper)

    log.info("MCP %s: connected, %d tools", server_name, len(tools))
    return tools


def split_mcp_servers(mcp_servers: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split MCP config into stdio (handled by SDK) and http/sse (handled by us)."""
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
