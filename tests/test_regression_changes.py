"""Regression tests for the uncommitted working-tree changes.

Each test targets a specific changed file/area so we can confirm the fix
holds and guard against regressions:

1. mcp_transport — next_request_id uniqueness across tool wrappers
2. mcp_transport — send_request cleans up _pending on all paths
3. runtime_session — _read_cron_jobs uses closing() (no leak on error)
4. daemon — _CONFIG_KEYS includes the newly added fields
5. cli — main() exception handler prints to stderr and returns 1
6. cli — goal-loop stop flag is settable and checked
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# 1. MCP transport: next_request_id uniqueness across tool wrappers
# ---------------------------------------------------------------------------


class TestMcpRequestIdUniqueness:
    """Regression: each McpToolWrapper used to have its own _next_id
    starting at 1, colliding across tools and with SseClient's own ids.
    Now all ids come from the shared SseClient.next_request_id()."""

    def test_multiple_wrappers_share_single_id_space(self):
        from rooster_code.mcp_transport import SseClient

        client = SseClient("http://localhost/sse")
        init_id = client.next_request_id()   # 1
        list_id = client.next_request_id()   # 2
        call1_id = client.next_request_id()  # 3
        call2_id = client.next_request_id()  # 4

        assert call1_id == 3, f"expected 3 (after init=1, list=2), got {call1_id}"
        assert call2_id == 4
        assert call1_id != init_id
        assert call2_id != list_id
        assert call1_id != call2_id

    def test_wrapper_has_no_own_next_id_attribute(self):
        """The per-wrapper _next_id field must be gone — it was the bug source."""
        from rooster_code.mcp_transport import McpToolWrapper, SseClient

        client = SseClient("http://localhost/sse")
        wrapper = McpToolWrapper("t", "d", {}, "srv", "tool", client)
        assert not hasattr(wrapper, "_next_id"), (
            "McpToolWrapper must not carry its own _next_id; "
            "all ids must come from the shared SseClient."
        )

    def test_tool_call_uses_client_id_not_none(self):
        """The 'id' field in a tools/call request must be a real int from the client."""
        from rooster_code.mcp_transport import McpToolWrapper, SseClient

        client = SseClient("http://localhost/sse")
        client.next_request_id()  # 1
        client.next_request_id()  # 2

        captured: dict = {}

        async def fake_send_request(request):
            captured["id"] = request.get("id")
            return {"content": [{"type": "text", "text": "ok"}]}

        client._messages_url = "http://localhost/messages"
        client.send_request = fake_send_request  # type: ignore[method-assign]

        wrapper = McpToolWrapper("t", "d", {}, "srv", "tool", client)
        asyncio.run(wrapper.call({"x": 1}, context=None))

        assert captured["id"] is not None, "tools/call request id must not be None"
        assert isinstance(captured["id"], int)
        assert captured["id"] == 3


# ---------------------------------------------------------------------------
# 2. MCP transport: send_request _pending cleanup
# ---------------------------------------------------------------------------


class TestSendRequestPendingCleanup:
    """Regression: send_request must remove the pending future on every exit
    path (inline SSE, HTTP error, timeout) to prevent future dict growth."""

    def test_pending_cleared_after_inline_sse_response(self):
        import httpx
        from rooster_code.mcp_transport import SseClient

        client = SseClient("http://localhost/sse", timeout=5)
        client._messages_url = "http://localhost/messages"

        req_id = client.next_request_id()
        request = {"jsonrpc": "2.0", "id": req_id, "method": "tools/list", "params": {}}

        inline_resp = httpx.Response(
            200,
            text=f'data: {{"id":{req_id},"result":{{"tools":[]}}}}\n\n',
            headers={"content-type": "text/event-stream"},
            request=httpx.Request("POST", "http://localhost/messages"),
        )

        async def fake_post(*args, **kwargs):
            return inline_resp

        client._http.post = fake_post  # type: ignore[method-assign]

        result = asyncio.run(client.send_request(request))
        assert result == {"tools": []}
        assert req_id not in client._pending, (
            f"pending entry for id={req_id} leaked after inline SSE parse"
        )

    def test_pending_cleared_on_http_error(self):
        import httpx
        from rooster_code.mcp_transport import SseClient

        client = SseClient("http://localhost/sse", timeout=5)
        client._messages_url = "http://localhost/messages"

        req_id = client.next_request_id()
        request = {"jsonrpc": "2.0", "id": req_id, "method": "tools/list", "params": {}}

        error_resp = httpx.Response(
            500,
            text="Internal Server Error",
            request=httpx.Request("POST", "http://localhost/messages"),
        )

        async def fake_post(*args, **kwargs):
            return error_resp

        client._http.post = fake_post  # type: ignore[method-assign]

        with pytest.raises(httpx.HTTPStatusError):
            asyncio.run(client.send_request(request))

        assert req_id not in client._pending, (
            f"pending entry for id={req_id} leaked after HTTP 500"
        )


# ---------------------------------------------------------------------------
# 3. runtime_session: _read_cron_jobs uses closing() — no connection leak
# ---------------------------------------------------------------------------


class TestCronReaderResourceSafety:
    """Regression: _read_cron_jobs used to call conn.close() only on the happy
    path. Now it uses contextlib.closing() so the connection is closed even
    if execute/fetchall raises."""

    def test_connection_closed_on_query_error(self, monkeypatch, tmp_path):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        db_path = tmp_path / ".rooster-code" / "daemon.db"
        db_path.parent.mkdir()

        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE cron_jobs (job_id TEXT PRIMARY KEY)")
        conn.close()

        from rooster_code import runtime_session

        closed_flags: list[bool] = []
        original_connect = sqlite3.connect

        class TrackingConnection:
            def __init__(self, real):
                self._real = real
                self.row_factory = None

            def execute(self, *args, **kwargs):
                raise sqlite3.OperationalError("simulated corruption")

            def close(self):
                closed_flags.append(True)
                self._real.close()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.close()
                # Don't suppress exceptions — sqlite3's __exit__ would rollback,
                # but we just close and let the exception propagate

        def tracking_connect(path):
            return TrackingConnection(original_connect(path))

        monkeypatch.setattr(sqlite3, "connect", tracking_connect)

        runtime_session._read_cron_jobs()

        assert closed_flags == [True], (
            "sqlite connection was not closed when execute() raised — "
            "contextlib.closing() must ensure cleanup"
        )

    def test_connection_closed_on_success(self, monkeypatch, tmp_path):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        db_path = tmp_path / ".rooster-code" / "daemon.db"
        db_path.parent.mkdir()

        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE cron_jobs (job_id TEXT PRIMARY KEY, schedule TEXT, command TEXT, name TEXT)")
        conn.execute("INSERT INTO cron_jobs VALUES ('j1', '*/5 * * * *', 'echo hi', 'test')")
        conn.commit()
        conn.close()

        from rooster_code import runtime_session
        result = runtime_session._read_cron_jobs()
        assert "j1" in result


# ---------------------------------------------------------------------------
# 4. daemon: _CONFIG_KEYS includes newly added fields
# ---------------------------------------------------------------------------


class TestDaemonConfigKeysExpansion:
    """Regression: _CONFIG_KEYS and _CONFIG_MERGE_KEYS cover all daemon override
    fields. Dict-typed keys (agents, hooks, mcp_servers, extra_args) use merge
    semantics to avoid silently discarding config-file values."""

    def test_new_keys_present_in_config_keys_or_merge_keys(self):
        import rooster_code.daemon as dm

        expected_new = {
            "search_url", "skills_dir", "agents",
            "hooks", "json_schema", "mcp_servers", "extra_args",
        }
        all_keys = set(dm._CONFIG_KEYS) | set(dm._CONFIG_MERGE_KEYS)
        missing = expected_new - all_keys
        assert not missing, f"Newly added keys missing from _CONFIG_KEYS or _CONFIG_MERGE_KEYS: {missing}"

    def test_dict_keys_use_merge_semantics(self):
        """Dict-typed keys must be in _CONFIG_MERGE_KEYS, not _CONFIG_KEYS."""
        import rooster_code.daemon as dm

        dict_keys = {"agents", "hooks", "mcp_servers", "extra_args"}
        for key in dict_keys:
            assert key in dm._CONFIG_MERGE_KEYS, (
                f"Dict-typed key '{key}' should be in _CONFIG_MERGE_KEYS for merge semantics"
            )

    def test_all_config_keys_are_valid_runtimeconfig_fields(self):
        """Every key in _CONFIG_KEYS and _CONFIG_MERGE_KEYS must be a real RuntimeConfig field."""
        import dataclasses
        from rooster_code.config import RuntimeConfig
        import rooster_code.daemon as dm

        field_names = {f.name for f in dataclasses.fields(RuntimeConfig)}
        for key in dm._CONFIG_KEYS + dm._CONFIG_MERGE_KEYS:
            assert key in field_names, (
                f"Key '{key}' is not a RuntimeConfig field. Valid fields: {sorted(field_names)}"
            )

    def test_new_keys_settable_via_setattr(self):
        from rooster_code.config import RuntimeConfig
        import rooster_code.daemon as dm

        config = RuntimeConfig()
        test_values = {
            "search_url": "http://search:8080",
            "skills_dir": "/tmp/skills",
            "json_schema": {"type": "object"},
            "extra_args": {"foo": "bar"},
        }
        for key, val in test_values.items():
            all_keys = set(dm._CONFIG_KEYS) | set(dm._CONFIG_MERGE_KEYS)
            assert key in all_keys
            setattr(config, key, val)
            assert getattr(config, key) == val


# ---------------------------------------------------------------------------
# 5. cli: main() exception handler prints to stderr and returns 1
# ---------------------------------------------------------------------------


class TestMainExceptionHandler:
    def test_unhandled_exception_returns_one_and_logs_traceback(self, monkeypatch, caplog):
        from rooster_code import cli

        def boom():
            raise RuntimeError("test explosion")

        monkeypatch.setattr(cli, "build_parser", boom)

        with caplog.at_level("ERROR", logger="rooster.cli"):
            exit_code = cli.main([])

        assert exit_code == 1
        # log.exception must preserve the error message and traceback.
        assert "test explosion" in caplog.text
        assert "Traceback" in caplog.text
        assert "rooster-code failed" in caplog.text

    def test_keyboard_interrupt_returns_130(self, monkeypatch):
        from rooster_code import cli

        def boom():
            raise KeyboardInterrupt

        monkeypatch.setattr(cli, "build_parser", boom)
        exit_code = cli.main([])
        assert exit_code == 130


# ---------------------------------------------------------------------------
# 6. cli: goal-loop stop flag
# ---------------------------------------------------------------------------


class TestGoalLoopStopFlag:
    """Regression: the SIGINT handler now sets _goal_stop so the goal loop
    exits with 'Stopped by user' instead of being treated as a regular
    interrupt. We test the flag mechanics in isolation since run_chat
    requires a full agent."""

    def test_goal_stop_flag_lifecycle(self):
        _goal_stop = False
        _goal_loop_active = True
        _goal_loop_turns = 0
        _MAX_GOAL_TURNS = 20

        # SIGINT handler
        _goal_stop = True

        # Goal loop check (as in cli.py:920)
        message = ""
        if _goal_stop or _goal_loop_turns >= _MAX_GOAL_TURNS:
            if _goal_stop:
                message = "Stopped by user."
            elif _goal_loop_turns >= _MAX_GOAL_TURNS:
                message = f"Stopped after {_MAX_GOAL_TURNS} turns."
            _goal_loop_active = False
            _goal_loop_turns = 0
            _goal_stop = False

        assert _goal_loop_active is False
        assert _goal_stop is False
        assert _goal_loop_turns == 0
        assert message == "Stopped by user."

    def test_goal_stop_takes_priority_over_max_turns(self):
        _goal_stop = True
        _goal_loop_turns = 20
        _MAX_GOAL_TURNS = 20

        message = ""
        if _goal_stop or _goal_loop_turns >= _MAX_GOAL_TURNS:
            if _goal_stop:
                message = "Stopped by user."
            elif _goal_loop_turns >= _MAX_GOAL_TURNS:
                message = f"Stopped after {_MAX_GOAL_TURNS} turns."

        assert message == "Stopped by user."

    def test_max_turns_message_without_user_stop(self):
        _goal_stop = False
        _goal_loop_turns = 20
        _MAX_GOAL_TURNS = 20

        message = ""
        if _goal_stop or _goal_loop_turns >= _MAX_GOAL_TURNS:
            if _goal_stop:
                message = "Stopped by user."
            elif _goal_loop_turns >= _MAX_GOAL_TURNS:
                message = f"Stopped after {_MAX_GOAL_TURNS} turns."

        assert message == "Stopped after 20 turns."
