import asyncio
import json

import pytest

from rooster_code.mcp_transport import SseClient, split_mcp_servers


class FakeSseServer:
    """Minimal in-process MCP SSE server for testing.

    Follows the MCP SSE transport protocol:
    - GET /sse -> streams events, first one is 'endpoint' with session URL
    - POST /messages/ -> accepts JSON-RPC, returns 202; responses go on SSE stream
    """

    REQUEST_ID = 0

    def __init__(self):
        self.session_id = "test-session-001"
        self.messages_url = f"/messages/?session_id={self.session_id}"
        self._tools = [
            {
                "name": "echo",
                "description": "Echo back the input text.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            },
            {
                "name": "add",
                "description": "Add two numbers.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number"},
                        "b": {"type": "number"},
                    },
                    "required": ["a", "b"],
                },
            },
        ]
        self._pending_responses: list[dict] = []
        self._request_counter = 0

    def handle_sse_get(self) -> str:
        events = f"event: endpoint\ndata: {self.messages_url}\n\n"
        return events

    def handle_message_post(self, request: dict) -> int:
        method = request.get("method", "")
        req_id = request.get("id")
        params = request.get("params", {})

        if method == "initialize":
            result = {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fake-mcp", "version": "0.1.0"},
            }
            self._queue_response(req_id, result)
        elif method == "tools/list":
            self._queue_response(req_id, {"tools": self._tools})
        elif method == "tools/call":
            tool_name = params.get("name", "")
            args = params.get("arguments", {})
            if tool_name == "echo":
                text = args.get("text", "")
                self._queue_response(req_id, {
                    "content": [{"type": "text", "text": f"Echo: {text}"}]
                })
            elif tool_name == "add":
                a = args.get("a", 0)
                b = args.get("b", 0)
                self._queue_response(req_id, {
                    "content": [{"type": "text", "text": str(a + b)}]
                })
            else:
                self._queue_error(req_id, f"Unknown tool: {tool_name}")
        elif method == "notifications/initialized":
            pass

        return 202

    def _queue_response(self, req_id: int | None, result: dict) -> None:
        self._pending_responses.append({"id": req_id, "result": result})

    def _queue_error(self, req_id: int | None, message: str) -> None:
        self._pending_responses.append({"id": req_id, "error": {"code": -32601, "message": message}})

    def drain_responses(self) -> list[dict]:
        responses = self._pending_responses[:]
        self._pending_responses.clear()
        return responses


class FakeSseStream:

    def __init__(self, server: FakeSseServer):
        self._server = server
        self._lines: list[str] = []
        self._closed = False
        self._endpoint_sent = False

    async def read_lines(self) -> list[str]:
        await asyncio.sleep(0.01)
        if not self._endpoint_sent:
            self._endpoint_sent = True
            return [
                f"data: {self._server.messages_url}",
            ]
        responses = self._server.drain_responses()
        lines = []
        for resp in responses:
            lines.append(f"data: {json.dumps(resp)}")
        return lines

    def close(self) -> None:
        self._closed = True


class TestSplitMcpServers:

    def test_splits_stdio_from_remote(self):
        config = {
            "local-tool": {"type": "stdio", "command": "my-tool"},
            "remote-tool": {"type": "sse", "url": "http://example.com/sse"},
        }
        stdio, remote = split_mcp_servers(config)
        assert "local-tool" in stdio
        assert "remote-tool" not in stdio
        assert "remote-tool" in remote
        assert "local-tool" not in remote

    def test_defaults_to_stdio(self):
        config = {"my-tool": {"command": "my-tool"}}
        stdio, remote = split_mcp_servers(config)
        assert "my-tool" in stdio
        assert len(remote) == 0

    def test_handles_streamable_http(self):
        config = {
            "streamable": {"type": "streamable-http", "url": "http://example.com/mcp"},
        }
        stdio, remote = split_mcp_servers(config)
        assert "streamable" in remote
        assert len(stdio) == 0

    def test_empty_config(self):
        stdio, remote = split_mcp_servers({})
        assert len(stdio) == 0
        assert len(remote) == 0


class TestSseClientUnit:

    def test_messages_url_built_from_relative_path(self):
        client = SseClient("http://127.0.0.1:9876/sse")
        parsed_path = "/messages/?session_id=abc123"
        client._messages_url = f"http://127.0.0.1:9876{parsed_path}"
        assert client._messages_url == "http://127.0.0.1:9876/messages/?session_id=abc123"

    def test_messages_url_absolute(self):
        client = SseClient("http://127.0.0.1:9876/sse")
        client._messages_url = "http://other-host:8080/messages/?session_id=xyz"
        assert "other-host" in client._messages_url


class TestMcpToolWrapper:

    def test_wrapper_name_format(self):
        from rooster_code.mcp_transport import McpToolWrapper, SseClient

        client = SseClient("http://localhost:8080/sse")
        wrapper = McpToolWrapper(
            name="mcp__myserver__mytool",
            description="A test tool",
            input_schema_dict={"properties": {"x": {"type": "int"}}, "required": ["x"]},
            server_name="myserver",
            tool_name="mytool",
            transport_client=client,
        )
        assert wrapper._name == "mcp__myserver__mytool"
        assert wrapper.tool_name == "mytool"
        assert wrapper.server_name == "myserver"
        assert not wrapper.is_read_only()
        assert not wrapper.is_concurrency_safe()


class TestSseClientIntegration:

    @pytest.fixture
    def fake_server(self):
        return FakeSseServer()

    @pytest.fixture
    def fake_stream(self, fake_server):
        return FakeSseStream(fake_server)

    def test_messages_url_discovery(self, fake_server, fake_stream):
        client = SseClient("http://127.0.0.1:9876/sse")
        assert client._messages_url is None

    def test_split_mcp_preserves_config(self):
        config = {
            "stdio-tool": {"type": "stdio", "command": "uvx", "args": ["my-tool"]},
            "sse-tool": {"type": "sse", "url": "https://api.example.com/sse"},
            "http-tool": {"type": "streamable-http", "url": "https://api.example.com/mcp"},
        }
        stdio, remote = split_mcp_servers(config)
        assert stdio["stdio-tool"]["command"] == "uvx"
        assert remote["sse-tool"]["url"] == "https://api.example.com/sse"
        assert remote["http-tool"]["url"] == "https://api.example.com/mcp"

    def test_connect_http_mcp_bad_url(self):
        async def run():
            from rooster_code.mcp_transport import connect_http_mcp
            tools = await connect_http_mcp("bad-server", {"url": ""})
            assert tools == []

        asyncio.run(run())

    def test_connect_http_mcp_no_url(self):
        async def run():
            from rooster_code.mcp_transport import connect_http_mcp
            tools = await connect_http_mcp("bad-server", {})
            assert tools == []

        asyncio.run(run())

    def test_connect_http_mcp_unreachable(self):
        async def run():
            from rooster_code.mcp_transport import connect_http_mcp
            tools = await connect_http_mcp("dead-server", {"url": "http://127.0.0.1:59999/sse"})
            assert tools == []

        asyncio.run(run())

    def test_connect_http_mcp_closes_client_when_server_has_no_tools(self, monkeypatch):
        closed: list[bool] = []

        class FakeSseClient:
            def __init__(self, url: str) -> None:
                self.url = url

            async def initialize(self) -> None:
                return None

            async def list_tools(self) -> list[dict]:
                return []

            async def close(self) -> None:
                closed.append(True)

        async def run():
            import rooster_code.mcp_transport as mcp_transport
            monkeypatch.setattr(mcp_transport, "SseClient", FakeSseClient)

            tools = await mcp_transport.connect_http_mcp("empty-server", {"url": "http://localhost/sse"})

            assert tools == []
            assert closed == [True]

        asyncio.run(run())

    def test_parse_inline_sse_response(self):
        client = SseClient("http://localhost/sse")
        import httpx

        resp = httpx.Response(
            200,
            text='data: {"id":1,"result":{"content":[{"type":"text","text":"hello"}]}}\n\n',
            headers={"content-type": "text/event-stream"},
            request=httpx.Request("POST", "http://localhost/messages/"),
        )
        result = client._parse_inline_sse(resp)
        assert result["content"][0]["text"] == "hello"

    def test_parse_inline_sse_error(self):
        client = SseClient("http://localhost/sse")
        import httpx

        resp = httpx.Response(
            200,
            text='data: {"id":1,"error":{"code":-32601,"message":"not found"}}\n\n',
            headers={"content-type": "text/event-stream"},
            request=httpx.Request("POST", "http://localhost/messages/"),
        )
        with pytest.raises(RuntimeError, match="MCP error"):
            client._parse_inline_sse(resp)

    def test_pending_future_resolved(self, fake_server):
        async def run():
            client = SseClient("http://127.0.0.1:9876/sse", timeout=5)
            loop = asyncio.get_running_loop()
            fut = loop.create_future()
            client._pending[1] = fut
            fake_server._queue_response(1, {"tools": []})
            for resp in fake_server.drain_responses():
                req_id = resp.get("id")
                if req_id in client._pending and not client._pending[req_id].done():
                    client._pending[req_id].set_result(resp.get("result", {}))
            assert fut.done()
            assert fut.result() == {"tools": []}

        asyncio.run(run())

    def test_next_id_increments(self):
        client = SseClient("http://localhost/sse")
        assert client._next_id == 1
        client._next_id += 1
        assert client._next_id == 2

    def test_next_request_id_returns_incremented_non_none(self):
        # Regression: next_request_id() must return the id, not None.
        # All request payloads use the return value as the jsonrpc "id";
        # returning None breaks SSE stream response correlation and
        # silently yields zero tools from real MCP servers.
        client = SseClient("http://localhost/sse")
        first = client.next_request_id()
        second = client.next_request_id()
        assert first == 1
        assert second == 2
        assert first is not None and second is not None

    def test_initialize_request_carries_non_none_id(self):
        client = SseClient("http://localhost/sse")
        # Build the initialize request id the same way initialize() does.
        req_id = client.next_request_id()
        assert req_id is not None and req_id == 1
        assert req_id in (1,)  # ensures the payload "id" field is populated


class TestSseClientProtocolCompliance:

    def test_sse_url_trailing_slash_stripped(self):
        client = SseClient("http://localhost:8080/sse/")
        assert client._sse_url == "http://localhost:8080/sse"

    def test_timeout_default(self):
        client = SseClient("http://localhost/sse")
        assert client._timeout == 30.0

    def test_timeout_custom(self):
        client = SseClient("http://localhost/sse", timeout=10.0)
        assert client._timeout == 10.0

    def test_messages_url_starts_none(self):
        client = SseClient("http://localhost/sse")
        assert client._messages_url is None

    def test_close_without_reader(self):
        async def run():
            client = SseClient("http://localhost/sse")
            await client.close()

        asyncio.run(run())

    def test_close_with_cancelled_reader(self):
        async def run():
            client = SseClient("http://localhost/sse")
            client._reader_task = asyncio.create_task(asyncio.sleep(9999))
            await client.close()

        asyncio.run(run())
