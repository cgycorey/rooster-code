"""Tests for rooster_code.team — AgentPool, TeamManager, TeamDispatchTool, TeamSendMessageTool, patch_tool_pool."""

import asyncio
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from open_agent_sdk.types import ToolContext
from open_agent_sdk.tools import clear_tasks, get_all_tasks

from rooster_code.config import RuntimeConfig
from rooster_code.team import (
    MAILBOX_DISPATCH_TASK,
    AgentPool,
    SDKTeamCreateBridgeTool,
    SDKTeamDeleteBridgeTool,
    TeamDispatchTool,
    TeamManager,
    TeamSendMessageTool,
    TeamStatusTool,
    patch_tool_pool,
    set_runtime_team_bridge,
)


class FakeQueryResult:
    def __init__(self, text: str = "done", messages: list | None = None):
        self.text = text
        self.messages = messages or []


class FakeAgent:
    def __init__(self, responses: list[str] | None = None):
        self._options = MagicMock()
        self._options.abort_signal = None
        self._options.append_system_prompt = ""
        self._tool_pool = []
        self._history: list[dict[str, Any]] = []
        self._responses = responses or ["done"]
        self._call_count = 0
        self._closed = False
        self._prompt_error = None
        self._last_prompt = ""

    async def prompt(self, text: str, overrides: dict[str, Any] | None = None):
        if self._prompt_error:
            raise self._prompt_error
        self._last_prompt = text
        result_text = self._responses[self._call_count] if self._call_count < len(self._responses) else self._responses[-1]
        self._call_count += 1
        return FakeQueryResult(text=result_text)

    async def _initialize(self):
        pass

    async def close(self):
        self._closed = True

    def clear(self):
        self._history.clear()


def _make_config(agents: dict[str, Any] | None = None) -> RuntimeConfig:
    return RuntimeConfig(
        model="test-model",
        api_key="test",
        base_url="https://example.test",
        agents=agents or {"reviewer": {"description": "code reviewer", "prompt": "You are a reviewer."}},
    )


def test_agent_pool_dispatch_unknown_member():
    pool = AgentPool()
    result = asyncio.run(pool.dispatch("unknown", "do something"))
    assert "Error" in result
    assert "unknown" in result


def test_agent_pool_dispatch_marks_unhealthy_on_failure():
    pool = AgentPool()
    fake_agent = FakeAgent()
    fake_agent._prompt_error = RuntimeError("API error")
    pool._members["reviewer"] = fake_agent
    pool._mailboxes["reviewer"] = asyncio.Queue()
    pool._locks["reviewer"] = asyncio.Lock()

    result = asyncio.run(pool.dispatch("reviewer", "review code"))

    assert "Error" in result
    assert "reviewer" in result
    assert "reviewer" in pool._unhealthy


def test_agent_pool_dispatch_unhealthy_member():
    pool = AgentPool()
    fake_agent = FakeAgent()
    pool._members["reviewer"] = fake_agent
    pool._mailboxes["reviewer"] = asyncio.Queue()
    pool._locks["reviewer"] = asyncio.Lock()
    pool._unhealthy.add("reviewer")

    result = asyncio.run(pool.dispatch("reviewer", "review code"))

    assert "unavailable" in result
    assert "reviewer" in result


def test_agent_pool_dispatch_success():
    pool = AgentPool()
    fake_agent = FakeAgent(responses=["LGTM"])
    pool._members["reviewer"] = fake_agent
    pool._mailboxes["reviewer"] = asyncio.Queue()
    pool._locks["reviewer"] = asyncio.Lock()

    result = asyncio.run(pool.dispatch("reviewer", "review code"))

    assert result == "LGTM"


def test_agent_pool_send_and_inject_mailbox():
    pool = AgentPool()
    fake_agent = FakeAgent(responses=["noted"])
    pool._members["reviewer"] = fake_agent
    pool._mailboxes["reviewer"] = asyncio.Queue()
    pool._locks["reviewer"] = asyncio.Lock()

    pool.send_message("reviewer", {"from": "builder", "content": "check this"})
    task = pool._inject_mailbox("reviewer", "review code")

    assert "[Message from builder]: check this" in task
    assert "review code" in task


def test_agent_pool_snapshot_mailboxes_preserves_messages_and_defaults_type():
    pool = AgentPool()
    pool._mailboxes["reviewer"] = asyncio.Queue()
    pool._locks["reviewer"] = asyncio.Lock()

    pool.send_message("reviewer", {"from": "builder", "content": "check this"})

    snapshot = pool.snapshot_mailboxes()

    assert snapshot == {
        "reviewer": [{"type": "text", "from": "builder", "content": "check this"}]
    }
    assert not pool._mailboxes["reviewer"].empty()
    assert pool._mailboxes["reviewer"].get_nowait() == {"from": "builder", "content": "check this"}


def test_agent_pool_inject_mailbox_no_messages():
    pool = AgentPool()
    pool._mailboxes["reviewer"] = asyncio.Queue()
    pool._locks["reviewer"] = asyncio.Lock()

    result = pool._inject_mailbox("reviewer", "just a task")
    assert result == "just a task"


def test_agent_pool_inject_mailbox_multiple_messages():
    pool = AgentPool()
    pool._mailboxes["reviewer"] = asyncio.Queue()
    pool._locks["reviewer"] = asyncio.Lock()

    pool.send_message("reviewer", {"from": "builder", "content": "msg1"})
    pool.send_message("reviewer", {"from": "tester", "content": "msg2"})
    task = pool._inject_mailbox("reviewer", "do work")

    assert "[Message from builder]: msg1" in task
    assert "[Message from tester]: msg2" in task
    assert "do work" in task


def test_agent_pool_send_message_unknown_recipient():
    pool = AgentPool()
    pool.send_message("unknown", {"from": "builder", "content": "hello"})
    assert pool._mailboxes.get("unknown") is None


def test_agent_pool_wake_for_messages_dispatches_mailbox_task():
    async def _run():
        pool = AgentPool()
        fake_agent = FakeAgent(responses=["noted"])
        pool._members["reviewer"] = fake_agent
        pool._mailboxes["reviewer"] = asyncio.Queue()
        pool._locks["reviewer"] = asyncio.Lock()

        pool.send_message("reviewer", {"from": "builder", "content": "hello"})
        triggered = pool.wake_for_messages("reviewer")
        assert triggered is True

        for _ in range(20):
            if fake_agent._call_count:
                break
            await asyncio.sleep(0)

        assert fake_agent._call_count == 1
        assert "[Message from builder]: hello" in fake_agent._last_prompt
        assert MAILBOX_DISPATCH_TASK in fake_agent._last_prompt
        await pool.close_all()

    asyncio.run(_run())


def test_wake_for_messages_returns_false_on_empty_mailbox():
    pool = AgentPool()
    pool._members["reviewer"] = FakeAgent()
    pool._mailboxes["reviewer"] = asyncio.Queue()
    pool._locks["reviewer"] = asyncio.Lock()
    # No messages queued
    result = pool.wake_for_messages("reviewer")
    assert result is False


def test_wake_for_messages_returns_false_when_unhealthy():
    pool = AgentPool()
    pool._members["reviewer"] = FakeAgent()
    pool._mailboxes["reviewer"] = asyncio.Queue()
    pool._locks["reviewer"] = asyncio.Lock()
    pool._unhealthy.add("reviewer")
    pool.send_message("reviewer", {"from": "builder", "content": "hello"})
    result = pool.wake_for_messages("reviewer")
    assert result is False


def test_wake_for_messages_returns_false_when_busy():
    pool = AgentPool()
    pool._members["reviewer"] = FakeAgent()
    pool._mailboxes["reviewer"] = asyncio.Queue()
    pool._locks["reviewer"] = asyncio.Lock()
    pool._busy.add("reviewer")
    pool.send_message("reviewer", {"from": "builder", "content": "hello"})
    result = pool.wake_for_messages("reviewer")
    assert result is False


def test_wake_for_messages_returns_false_unknown_member():
    pool = AgentPool()
    result = pool.wake_for_messages("unknown")
    assert result is False


def test_wake_for_messages_multiple_queued_messages():
    async def _run():
        pool = AgentPool()
        fake_agent = FakeAgent(responses=["noted"])
        pool._members["reviewer"] = fake_agent
        pool._mailboxes["reviewer"] = asyncio.Queue()
        pool._locks["reviewer"] = asyncio.Lock()

        pool.send_message("reviewer", {"from": "a", "content": "msg1"})
        pool.send_message("reviewer", {"from": "b", "content": "msg2"})
        pool.send_message("reviewer", {"from": "c", "content": "msg3"})

        pool.wake_for_messages("reviewer")

        for _ in range(20):
            if fake_agent._call_count:
                break
            await asyncio.sleep(0)

        assert "[Message from a]: msg1" in fake_agent._last_prompt
        assert "[Message from b]: msg2" in fake_agent._last_prompt
        assert "[Message from c]: msg3" in fake_agent._last_prompt

        await pool.close_all()

    asyncio.run(_run())


def test_agent_pool_close_all():
    pool = AgentPool()
    fake1 = FakeAgent()
    fake2 = FakeAgent()
    pool._members["a"] = fake1
    pool._members["b"] = fake2
    pool._mailboxes["a"] = asyncio.Queue()
    pool._mailboxes["b"] = asyncio.Queue()
    pool._locks["a"] = asyncio.Lock()
    pool._locks["b"] = asyncio.Lock()

    asyncio.run(pool.close_all())

    assert fake1._closed
    assert fake2._closed
    assert len(pool._members) == 0
    assert len(pool._mailboxes) == 0
    assert len(pool._locks) == 0
    assert len(pool._unhealthy) == 0
    assert len(pool._dispatch_tasks) == 0
    assert len(pool._message_dispatch_tasks) == 0


def test_agent_pool_close_all_cancels_in_flight_dispatch():
    async def _run():
        clear_tasks()
        pool = AgentPool()
        fake_agent = FakeAgent(responses=["LGTM"])

        pool._members["reviewer"] = fake_agent
        pool._mailboxes["reviewer"] = asyncio.Queue()
        pool._locks["reviewer"] = asyncio.Lock()

        cancelled_outputs: list[tuple[str, str]] = []

        async def _slow_dispatch():
            try:
                await asyncio.sleep(9999)
                cancelled_outputs.append(("status", "completed"))
            except asyncio.CancelledError:
                cancelled_outputs.append(("status", "cancelled"))
                raise

        import unittest.mock
        with unittest.mock.patch("rooster_code.runtime._update_background_subagent_task", new_callable=unittest.mock.AsyncMock) as mock_update:
            with unittest.mock.patch("rooster_code.runtime._track_background_task"):
                task_handle = asyncio.create_task(_slow_dispatch())
                task_handle.add_done_callback(pool._dispatch_tasks.discard)
                pool._dispatch_tasks.add(task_handle)
                pool._busy.add("reviewer")
                await asyncio.sleep(0)

                await pool.close_all()

        mock_update.assert_not_called()
        assert cancelled_outputs == [("status", "cancelled")]
        assert len(pool._dispatch_tasks) == 0
        assert len(pool._busy) == 0
        assert len(pool._members) == 0

        clear_tasks()

    asyncio.run(_run())


def test_agent_pool_clear_histories():
    pool = AgentPool()
    fake_agent = FakeAgent()
    fake_agent._history = [{"role": "user", "content": "hello"}]
    pool._members["reviewer"] = fake_agent

    asyncio.run(pool.clear_histories())

    assert len(fake_agent._history) == 0


def test_patch_tool_pool_add_and_remove():
    class FakeTool:
        name = "FakeTool"

    class RemoveMeTool:
        name = "RemoveMe"

    agent = MagicMock()
    tool_remove = RemoveMeTool()
    tool_keep = FakeTool()
    agent._tool_pool = [tool_remove, tool_keep]

    new_tool = MagicMock()
    new_tool.name = "TeamDispatch"

    patch_tool_pool(agent, add_tools=[new_tool], remove_names=["RemoveMe"])

    names = [t.name for t in agent._tool_pool]
    assert "RemoveMe" not in names
    assert "FakeTool" in names
    assert "TeamDispatch" in names


def test_patch_tool_pool_remove_only():
    class ToolA:
        name = "A"

    class ToolB:
        name = "B"

    agent = MagicMock()
    agent._tool_pool = [ToolA(), ToolB()]

    patch_tool_pool(agent, remove_names=["A"])

    names = [t.name for t in agent._tool_pool]
    assert "A" not in names
    assert "B" in names


def test_patch_tool_pool_add_only():
    agent = MagicMock()
    agent._tool_pool = []

    new_tool = MagicMock()
    new_tool.name = "NewTool"

    patch_tool_pool(agent, add_tools=[new_tool])

    assert len(agent._tool_pool) == 1
    assert agent._tool_pool[0].name == "NewTool"


def test_patch_tool_pool_none_tool_pool():
    agent = MagicMock()
    agent._tool_pool = None

    patch_tool_pool(agent, add_tools=[MagicMock(name="X")], remove_names=["Y"])

    assert agent._tool_pool is None


def test_dispatch_tool_missing_member():
    manager = TeamManager()
    manager._active = True
    manager._pool = AgentPool()

    tool = TeamDispatchTool(manager)

    async def _run():
        import unittest.mock
        with unittest.mock.patch("rooster_code.runtime._create_background_subagent_task", new_callable=unittest.mock.AsyncMock) as mock_create_task:
            mock_create_task.return_value = "test-task-err"
            result = await tool.call({"member": "unknown", "task": "do thing"}, ToolContext(cwd=".", env={}))

        assert result.is_error
        assert "unknown" in str(result.content)

    asyncio.run(_run())


def test_dispatch_tool_missing_task():
    manager = TeamManager()
    tool = TeamDispatchTool(manager)
    result = asyncio.run(tool.call({"member": "reviewer"}, ToolContext(cwd=".", env={})))
    assert result.is_error
    assert "task" in str(result.content).lower()


def test_dispatch_tool_async_dispatch():
    async def _run():
        manager = TeamManager()
        manager._active = True
        pool = AgentPool()
        fake_agent = FakeAgent(responses=["LGTM"])
        pool._members["reviewer"] = fake_agent
        pool._mailboxes["reviewer"] = asyncio.Queue()
        pool._locks["reviewer"] = asyncio.Lock()
        manager._pool = pool

        tool = TeamDispatchTool(manager)

        import unittest.mock
        with unittest.mock.patch("rooster_code.runtime._create_background_subagent_task", new_callable=unittest.mock.AsyncMock) as mock_create_task:
            mock_create_task.return_value = "test-task-123"
            result = await tool.call({"member": "reviewer", "task": "review this change"}, ToolContext(cwd=".", env={}))

        assert not result.is_error
        content = str(result.content)
        assert "Dispatched task to team member 'reviewer'" in content
        assert "test-task-123" in content
        assert "background" in content.lower()
        assert "Do not also perform the same task" in content

    asyncio.run(_run())


def test_send_message_tool_missing_to():
    manager = TeamManager()
    tool = TeamSendMessageTool(manager)
    result = asyncio.run(tool.call({"content": "hello"}, ToolContext(cwd=".", env={})))
    assert result.is_error
    assert "to" in str(result.content).lower()


def test_send_message_tool_missing_content():
    manager = TeamManager()
    tool = TeamSendMessageTool(manager)
    result = asyncio.run(tool.call({"to": "reviewer"}, ToolContext(cwd=".", env={})))
    assert result.is_error
    assert "content" in str(result.content).lower()


def test_send_message_tool_success():
    async def _run():
        clear_tasks()
        manager = TeamManager()
        manager._active = True
        pool = AgentPool()
        reviewer = FakeAgent()
        pool._members["reviewer"] = reviewer
        pool._mailboxes["reviewer"] = asyncio.Queue()
        pool._locks["reviewer"] = asyncio.Lock()
        manager._pool = pool

        tool = TeamSendMessageTool(manager)
        result = await tool.call({"to": "reviewer", "content": "check this"}, ToolContext(cwd=".", env={}))

        assert not result.is_error
        assert "reviewer" in result.content
        assert "Wake task created:" in str(result.content)
        for _ in range(20):
            if reviewer._last_prompt:
                break
            await asyncio.sleep(0)
        assert "[Message from orchestrator]: check this" in reviewer._last_prompt
        assert MAILBOX_DISPATCH_TASK in reviewer._last_prompt
        tasks = get_all_tasks()
        assert len(tasks) == 1
        task = next(iter(tasks.values()))
        assert task["subject"] == "team-mailbox-reviewer"

        clear_tasks()

    asyncio.run(_run())


def test_send_message_tool_uses_bound_sender_name():
    async def _run():
        clear_tasks()
        manager = TeamManager()
        manager._active = True
        pool = AgentPool()
        reviewer = FakeAgent()
        pool._members["reviewer"] = reviewer
        pool._mailboxes["reviewer"] = asyncio.Queue()
        pool._locks["reviewer"] = asyncio.Lock()
        manager._pool = pool

        tool = TeamSendMessageTool(manager, sender_name="builder")
        result = await tool.call({"to": "reviewer", "content": "check this"}, ToolContext(cwd=".", env={}))

        assert not result.is_error
        assert "Wake task created:" in str(result.content)
        for _ in range(20):
            if reviewer._last_prompt:
                break
            await asyncio.sleep(0)
        assert "[Message from builder]: check this" in reviewer._last_prompt

        clear_tasks()

    asyncio.run(_run())


def test_send_message_tool_supports_sdk_message_type():
    async def _run():
        clear_tasks()
        manager = TeamManager()
        manager._active = True
        pool = AgentPool()
        reviewer = FakeAgent()
        pool._members["reviewer"] = reviewer
        pool._mailboxes["reviewer"] = asyncio.Queue()
        pool._locks["reviewer"] = asyncio.Lock()
        manager._pool = pool

        tool = TeamSendMessageTool(manager, sender_name="builder")
        result = await tool.call(
            {"to": "reviewer", "content": "please stop", "type": "shutdown_request"},
            ToolContext(cwd=".", env={}),
        )

        assert not result.is_error
        assert "Wake task created:" in str(result.content)
        for _ in range(20):
            if reviewer._last_prompt:
                break
            await asyncio.sleep(0)
        assert "[Message from builder]: please stop" in reviewer._last_prompt

        clear_tasks()

    asyncio.run(_run())


def test_send_message_tool_supports_broadcast():
    async def _run():
        clear_tasks()
        manager = TeamManager()
        manager._active = True
        pool = AgentPool()
        builder = FakeAgent()
        reviewer = FakeAgent()
        pool._members["builder"] = builder
        pool._members["reviewer"] = reviewer
        pool._mailboxes["builder"] = asyncio.Queue()
        pool._mailboxes["reviewer"] = asyncio.Queue()
        pool._locks["builder"] = asyncio.Lock()
        pool._locks["reviewer"] = asyncio.Lock()
        manager._pool = pool

        tool = TeamSendMessageTool(manager, sender_name="builder")
        result = await tool.call(
            {"to": "*", "content": "heads up", "type": "plan_approval_response"},
            ToolContext(cwd=".", env={}),
        )

        assert not result.is_error
        assert str(result.content).startswith("Message broadcast to all agents.")
        assert "Wake tasks created:" in str(result.content)
        for _ in range(20):
            if builder._last_prompt and reviewer._last_prompt:
                break
            await asyncio.sleep(0)
        assert "[Message from builder]: heads up" in builder._last_prompt
        assert "[Message from builder]: heads up" in reviewer._last_prompt
        tasks = get_all_tasks()
        assert len(tasks) == 2

        clear_tasks()

    asyncio.run(_run())


def test_send_message_tool_queues_when_member_busy():
    async def _run():
        clear_tasks()
        manager = TeamManager()
        manager._active = True
        pool = AgentPool()
        reviewer = FakeAgent()
        pool._members["reviewer"] = reviewer
        pool._mailboxes["reviewer"] = asyncio.Queue()
        pool._locks["reviewer"] = asyncio.Lock()
        pool._busy.add("reviewer")
        manager._pool = pool

        tool = TeamSendMessageTool(manager)
        result = await tool.call({"to": "reviewer", "content": "check this later"}, ToolContext(cwd=".", env={}))

        assert not result.is_error
        assert "Delivery queued: queued_busy." in str(result.content)
        assert get_all_tasks() == {}

        clear_tasks()

    asyncio.run(_run())


def test_send_message_tool_errors_when_team_inactive():
    manager = TeamManager()
    tool = TeamSendMessageTool(manager)

    result = asyncio.run(tool.call({"to": "reviewer", "content": "hello"}, ToolContext(cwd=".", env={})))

    assert result.is_error
    assert "No team is active" in str(result.content)


def test_send_message_tool_errors_for_unknown_member():
    manager = TeamManager()
    manager._active = True
    manager._pool = AgentPool()
    tool = TeamSendMessageTool(manager)

    result = asyncio.run(tool.call({"to": "reviewer", "content": "hello"}, ToolContext(cwd=".", env={})))

    assert result.is_error
    assert "Unknown team member 'reviewer'" in str(result.content)


def test_sdk_team_create_bridge_tool_creates_persistent_team():
    from rooster_code.daemon import TeamSnapshotStore

    db_path = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name

    async def _run():
        manager = TeamManager()
        manager.enable_snapshot_persistence(db_path)
        config = _make_config({"reviewer": {"description": "code reviewer", "prompt": "You are a reviewer."}})
        fake_agent = FakeAgent()
        abort_signal = asyncio.Event()

        orchestrator = MagicMock()
        orchestrator._options.abort_signal = abort_signal
        orchestrator._options.append_system_prompt = "original prompt"
        orchestrator._tool_pool = []
        orchestrator._rooster_code_config = config

        async def fake_create_member(self, name, definition, config, abort_signal=None):
            self._members[name] = fake_agent
            self._mailboxes[name] = asyncio.Queue()
            self._locks[name] = asyncio.Lock()

        import unittest.mock
        with unittest.mock.patch.object(AgentPool, "create_member", fake_create_member):
            set_runtime_team_bridge(manager, orchestrator)
            try:
                tool = SDKTeamCreateBridgeTool()
                result = await tool.call({"name": "dev-team", "members": ["reviewer"]}, ToolContext(cwd=".", env={}))
            finally:
                set_runtime_team_bridge(None, None)

        assert not result.is_error
        assert str(result.content).startswith("Created team ")
        assert manager.is_active()
        assert manager.active_team_id()
        assert manager.info()["team_name"] == "dev-team"
        assert abort_signal.is_set() is False

        store = TeamSnapshotStore(db_path=db_path)
        try:
            snap = store.get_team(manager.active_team_id())
        finally:
            store.close()
        assert snap is not None
        assert snap["team_name"] == "dev-team"

    try:
        asyncio.run(_run())
    finally:
        Path(db_path).unlink(missing_ok=True)


def test_sdk_team_create_bridge_tool_materializes_missing_agent_definitions():
    async def _run():
        manager = TeamManager()
        config = RuntimeConfig(model="test-model", api_key="test", base_url="https://example.test", agents={})
        fake_agent = FakeAgent()
        abort_signal = asyncio.Event()

        orchestrator = MagicMock()
        orchestrator._options.abort_signal = abort_signal
        orchestrator._options.append_system_prompt = "original prompt"
        orchestrator._tool_pool = []
        orchestrator._rooster_code_config = config

        async def fake_create_member(self, name, definition, config, abort_signal=None):
            self._members[name] = fake_agent
            self._mailboxes[name] = asyncio.Queue()
            self._locks[name] = asyncio.Lock()

        import unittest.mock
        with unittest.mock.patch.object(AgentPool, "create_member", fake_create_member):
            set_runtime_team_bridge(manager, orchestrator)
            try:
                tool = SDKTeamCreateBridgeTool()
                result = await tool.call(
                    {"name": "review-team", "description": "Review changed files", "members": ["reviewer-1", "reviewer-2"]},
                    ToolContext(cwd=".", env={}),
                )
            finally:
                set_runtime_team_bridge(None, None)

        assert not result.is_error
        assert manager.is_active()
        assert sorted(config.agents.keys()) == ["reviewer-1", "reviewer-2"]
        assert "Review changed files" in str(config.agents["reviewer-1"]["prompt"])
        assert abort_signal.is_set() is False

    asyncio.run(_run())


def test_sdk_team_create_bridge_tool_rolls_back_materialized_agents_on_failure():
    async def _run():
        manager = TeamManager()
        config = RuntimeConfig(model="test-model", api_key="test", base_url="https://example.test", agents={})
        abort_signal = asyncio.Event()

        orchestrator = MagicMock()
        orchestrator._options.abort_signal = abort_signal
        orchestrator._options.append_system_prompt = "original prompt"
        orchestrator._tool_pool = []
        orchestrator._rooster_code_config = config

        async def fake_create_member(self, name, definition, config, abort_signal=None):
            raise RuntimeError("member failed to initialize")

        import unittest.mock
        with unittest.mock.patch.object(AgentPool, "create_member", fake_create_member):
            set_runtime_team_bridge(manager, orchestrator)
            try:
                tool = SDKTeamCreateBridgeTool()
                result = await tool.call(
                    {"name": "review-team", "description": "Review changed files", "members": ["reviewer-1"]},
                    ToolContext(cwd=".", env={}),
                )
            finally:
                set_runtime_team_bridge(None, None)

        assert result.is_error
        assert not manager.is_active()
        assert config.agents == {}

    asyncio.run(_run())


def test_sdk_team_delete_bridge_tool_deletes_persistent_team():
    async def _run():
        manager = TeamManager()
        config = _make_config({"reviewer": {"description": "code reviewer", "prompt": "You are a reviewer."}})
        fake_agent = FakeAgent()

        orchestrator = MagicMock()
        orchestrator._options.abort_signal = None
        orchestrator._options.append_system_prompt = "original prompt"
        orchestrator._tool_pool = []
        orchestrator._rooster_code_config = config

        async def fake_create_member(self, name, definition, config, abort_signal=None):
            self._members[name] = fake_agent
            self._mailboxes[name] = asyncio.Queue()
            self._locks[name] = asyncio.Lock()

        import unittest.mock
        with unittest.mock.patch.object(AgentPool, "create_member", fake_create_member):
            await manager.create_team("dev-team", ["reviewer"], config, orchestrator)

        team_id = manager.active_team_id()
        set_runtime_team_bridge(manager, orchestrator)
        try:
            tool = SDKTeamDeleteBridgeTool()
            result = await tool.call({"team_id": team_id}, ToolContext(cwd=".", env={}))
        finally:
            set_runtime_team_bridge(None, None)

        assert not result.is_error
        assert str(result.content) == f"Deleted team {team_id}"
        assert not manager.is_active()

    asyncio.run(_run())


def test_team_manager_initial_state():
    manager = TeamManager()
    assert not manager.is_active()
    info = manager.info()
    assert info["active"] is False


def test_team_manager_info_inactive():
    manager = TeamManager()
    info = manager.info()
    assert info == {"active": False}


def test_team_manager_create_team_already_active():
    manager = TeamManager()
    manager._active = True
    with pytest.raises(RuntimeError, match="already active"):
        asyncio.run(manager.create_team("team1", ["reviewer"], _make_config(), MagicMock()))


def test_team_manager_create_team_no_agents():
    manager = TeamManager()
    config = RuntimeConfig(model="m", agents={})
    with pytest.raises(RuntimeError, match="No agent definitions"):
        asyncio.run(manager.create_team("team1", ["reviewer"], config, MagicMock()))


def test_team_manager_create_team_undefined_member():
    manager = TeamManager()
    config = RuntimeConfig(model="m", agents={"reviewer": {"description": "rev"}})
    with pytest.raises(RuntimeError, match="not defined"):
        asyncio.run(manager.create_team("team1", ["builder"], config, MagicMock()))


def test_team_manager_create_team_duplicate_member():
    manager = TeamManager()
    config = RuntimeConfig(model="m", agents={"reviewer": {"description": "rev"}})
    with pytest.raises(RuntimeError, match="Duplicate"):
        asyncio.run(manager.create_team("team1", ["reviewer", "reviewer"], config, MagicMock()))


def test_team_manager_create_team_too_many_members():
    manager = TeamManager()
    agents = {f"m{i}": {"description": f"member {i}"} for i in range(6)}
    config = RuntimeConfig(model="m", agents=agents)
    with pytest.raises(RuntimeError, match="more than"):
        asyncio.run(manager.create_team("team1", list(agents.keys()), config, MagicMock()))


def test_team_manager_create_team_closes_partial_pool_on_member_creation_failure():
    async def _run():
        manager = TeamManager()
        config = _make_config({
            "reviewer": {"description": "reviews"},
            "builder": {"description": "builds"},
        })
        created_agent = FakeAgent()

        async def fake_create_member(self, name, definition, config, abort_signal=None):
            if name == "builder":
                raise RuntimeError("builder failed to initialize")
            self._members[name] = created_agent
            self._mailboxes[name] = asyncio.Queue()
            self._locks[name] = asyncio.Lock()

        import unittest.mock
        with unittest.mock.patch.object(AgentPool, "create_member", fake_create_member):
            with pytest.raises(RuntimeError, match="builder failed"):
                await manager.create_team("team1", ["reviewer", "builder"], config, MagicMock())

        assert not manager.is_active()
        assert manager._pool is None
        assert created_agent._closed is True

    asyncio.run(_run())


def test_team_manager_create_team_restores_state_when_partial_pool_cleanup_fails():
    async def _run():
        manager = TeamManager()
        config = _make_config({
            "reviewer": {"description": "reviews"},
            "builder": {"description": "builds"},
        })

        class FakeTool:
            def __init__(self, name: str):
                self._name = name

            @property
            def name(self) -> str:
                return self._name

        orchestrator = MagicMock()
        orchestrator._options.append_system_prompt = "original prompt"
        orchestrator._tool_pool = [FakeTool("Read"), FakeTool("Agent"), FakeTool("TeamCreate")]

        async def fake_create_member(self, name, definition, config, abort_signal=None):
            if name == "builder":
                raise RuntimeError("builder failed to initialize")
            self._members[name] = FakeAgent()
            self._mailboxes[name] = asyncio.Queue()
            self._locks[name] = asyncio.Lock()

        async def failing_close_all(self):
            raise RuntimeError("cleanup failed")

        import unittest.mock
        with unittest.mock.patch.object(AgentPool, "create_member", fake_create_member):
            with unittest.mock.patch.object(AgentPool, "close_all", failing_close_all):
                with pytest.raises(RuntimeError, match="builder failed"):
                    await manager.create_team("team1", ["reviewer", "builder"], config, orchestrator)

        assert not manager.is_active()
        assert manager._pool is None
        assert orchestrator._options.append_system_prompt == "original prompt"
        assert [tool.name for tool in orchestrator._tool_pool] == ["Read", "Agent", "TeamCreate"]

    asyncio.run(_run())


def test_team_manager_dispatch_no_active_team():
    manager = TeamManager()
    result = asyncio.run(manager.dispatch("reviewer", "do thing"))
    assert "No team is active" in result


def test_team_manager_send_message_no_active_team():
    manager = TeamManager()
    with pytest.raises(RuntimeError, match="No team is active"):
        asyncio.run(manager.send_message("reviewer", "hello"))


def test_team_manager_send_message_unknown_member():
    manager = TeamManager()
    manager._active = True
    manager._pool = AgentPool()

    with pytest.raises(RuntimeError, match="Unknown team member 'reviewer'"):
        asyncio.run(manager.send_message("reviewer", "hello"))


def test_team_manager_close_inactive_team():
    manager = TeamManager()
    asyncio.run(manager.close_team(MagicMock()))


def test_team_manager_info_active():
    manager = TeamManager()
    manager._active = True
    manager._team_name = "team1"
    pool = AgentPool()
    pool._members["reviewer"] = FakeAgent()
    pool._mailboxes["reviewer"] = asyncio.Queue()
    pool._locks["reviewer"] = asyncio.Lock()
    manager._pool = pool

    info = manager.info()
    assert info["active"] is True
    assert info["team_name"] == "team1"
    assert "reviewer" in info["members"]
    assert info["members"]["reviewer"] == "idle"
    assert "note" in info


def test_team_manager_sdk_team_snapshot_active():
    manager = TeamManager()
    manager._active = True
    manager._team_id = "team123"
    manager._team_name = "team1"
    manager._member_definitions = {
        "reviewer": {"description": "reviews"},
        "builder": {"description": "builds"},
    }
    pool = AgentPool()
    pool._members["reviewer"] = FakeAgent()
    pool._members["builder"] = FakeAgent()
    pool._mailboxes["reviewer"] = asyncio.Queue()
    pool._mailboxes["builder"] = asyncio.Queue()
    pool._locks["reviewer"] = asyncio.Lock()
    pool._locks["builder"] = asyncio.Lock()
    pool._busy.add("builder")
    manager._pool = pool

    snapshot = manager.sdk_team_snapshot()

    assert snapshot == {
        "team123": {
            "id": "team123",
            "name": "team1",
            "description": "",
            "members": ["reviewer", "builder"],
            "member_statuses": {"reviewer": "idle", "builder": "busy"},
            "runtime_managed": True,
        }
    }


def test_team_manager_dispatch_delegates_to_pool():
    manager = TeamManager()
    manager._active = True
    pool = AgentPool()
    fake_agent = FakeAgent(responses=["LGTM"])
    pool._members["reviewer"] = fake_agent
    pool._mailboxes["reviewer"] = asyncio.Queue()
    pool._locks["reviewer"] = asyncio.Lock()
    manager._pool = pool

    result = asyncio.run(manager.dispatch("reviewer", "review code"))
    assert result == "LGTM"


def test_team_manager_dispatch_recovers_unhealthy_member():
    async def _run():
        manager = TeamManager()
        manager._active = True
        manager._team_name = "dev-team"
        manager._config = _make_config()
        manager._member_definitions = {
            "reviewer": {"description": "code reviewer", "prompt": "You are a reviewer."}
        }
        pool = AgentPool()
        failed_agent = FakeAgent()
        failed_agent._prompt_error = RuntimeError("rate limited")
        pool._members["reviewer"] = failed_agent
        pool._mailboxes["reviewer"] = asyncio.Queue()
        pool._locks["reviewer"] = asyncio.Lock()
        pool._unhealthy.add("reviewer")
        manager._pool = pool

        async def fake_create_member(self, name, definition, config, abort_signal=None):
            recovered = FakeAgent(responses=["LGTM"])
            self._members[name] = recovered
            self._mailboxes[name] = asyncio.Queue()
            self._locks[name] = asyncio.Lock()

        import unittest.mock
        with unittest.mock.patch.object(AgentPool, "create_member", fake_create_member):
            result = await manager.dispatch("reviewer", "review code")

        assert result == "LGTM"
        assert "reviewer" not in pool._unhealthy
        assert failed_agent._closed is True

    asyncio.run(_run())


def test_team_manager_dispatch_async_recovers_unhealthy_member():
    async def _run():
        manager = TeamManager()
        manager._active = True
        manager._team_name = "dev-team"
        manager._config = _make_config()
        manager._member_definitions = {
            "reviewer": {"description": "code reviewer", "prompt": "You are a reviewer."}
        }
        pool = AgentPool()
        failed_agent = FakeAgent()
        failed_agent._prompt_error = RuntimeError("rate limited")
        pool._members["reviewer"] = failed_agent
        pool._mailboxes["reviewer"] = asyncio.Queue()
        pool._locks["reviewer"] = asyncio.Lock()
        pool._unhealthy.add("reviewer")
        manager._pool = pool

        async def fake_create_member(self, name, definition, config, abort_signal=None):
            recovered = FakeAgent(responses=["LGTM"])
            self._members[name] = recovered
            self._mailboxes[name] = asyncio.Queue()
            self._locks[name] = asyncio.Lock()

        import unittest.mock
        with unittest.mock.patch.object(AgentPool, "create_member", fake_create_member):
            with unittest.mock.patch("rooster_code.runtime._track_background_task"):
                with unittest.mock.patch("rooster_code.runtime._update_background_subagent_task", new_callable=unittest.mock.AsyncMock):
                    task_id = await manager.dispatch_async("reviewer", "review code", "task-1", ".", {})

        assert task_id == "task-1"
        assert "reviewer" not in pool._unhealthy
        assert failed_agent._closed is True

    asyncio.run(_run())


def test_team_manager_dispatch_recovery_preserves_mailbox_messages():
    async def _run():
        manager = TeamManager()
        manager._active = True
        manager._team_name = "dev-team"
        manager._config = _make_config()
        manager._member_definitions = {
            "reviewer": {"description": "code reviewer", "prompt": "You are a reviewer."}
        }
        pool = AgentPool()
        failed_agent = FakeAgent()
        failed_agent._prompt_error = RuntimeError("rate limited")
        pool._members["reviewer"] = failed_agent
        pool._mailboxes["reviewer"] = asyncio.Queue()
        pool._locks["reviewer"] = asyncio.Lock()
        pool._unhealthy.add("reviewer")
        pool.send_message("reviewer", {"from": "builder", "content": "check this first"})
        manager._pool = pool
        captured: dict[str, str] = {}

        class RecoveringAgent(FakeAgent):
            async def prompt(self, text: str, overrides: dict[str, Any] | None = None):
                captured["task"] = text
                return await super().prompt(text, overrides)

        async def fake_create_member(self, name, definition, config, abort_signal=None):
            recovered = RecoveringAgent(responses=["LGTM"])
            self._members[name] = recovered
            self._mailboxes[name] = asyncio.Queue()
            self._locks[name] = asyncio.Lock()

        import unittest.mock
        with unittest.mock.patch.object(AgentPool, "create_member", fake_create_member):
            result = await manager.dispatch("reviewer", "review code")

        assert result == "LGTM"
        assert "[Message from builder]: check this first" in captured["task"]
        assert captured["task"].endswith("review code")

    asyncio.run(_run())


def test_team_manager_dispatch_recovery_failure_keeps_member_recoverable():
    async def _run():
        manager = TeamManager()
        manager._active = True
        manager._team_name = "dev-team"
        manager._config = _make_config()
        manager._member_definitions = {
            "reviewer": {"description": "code reviewer", "prompt": "You are a reviewer."}
        }
        pool = AgentPool()
        failed_agent = FakeAgent()
        pool._members["reviewer"] = failed_agent
        pool._mailboxes["reviewer"] = asyncio.Queue()
        pool._locks["reviewer"] = asyncio.Lock()
        pool._unhealthy.add("reviewer")
        original_mailbox = pool._mailboxes["reviewer"]
        manager._pool = pool

        async def fake_create_member(self, name, definition, config, abort_signal=None):
            raise RuntimeError("still rate limited")

        import unittest.mock
        with unittest.mock.patch.object(AgentPool, "create_member", fake_create_member):
            result = await manager.dispatch("reviewer", "review code")

        assert "could not be recreated" in result
        assert "reviewer" in pool._unhealthy
        assert pool._members["reviewer"] is failed_agent
        assert pool._mailboxes["reviewer"] is original_mailbox
        assert failed_agent._closed is False

    asyncio.run(_run())


def test_team_manager_concurrent_dispatches_share_one_recovery():
    async def _run():
        manager = TeamManager()
        manager._active = True
        manager._team_name = "dev-team"
        manager._config = _make_config()
        manager._member_definitions = {
            "reviewer": {"description": "code reviewer", "prompt": "You are a reviewer."}
        }
        pool = AgentPool()
        failed_agent = FakeAgent()
        pool._members["reviewer"] = failed_agent
        pool._mailboxes["reviewer"] = asyncio.Queue()
        pool._locks["reviewer"] = asyncio.Lock()
        pool._unhealthy.add("reviewer")
        manager._pool = pool

        recovered_agent = FakeAgent(responses=["LGTM", "LGTM"])
        create_calls = 0

        async def fake_create_member(self, name, definition, config, abort_signal=None):
            nonlocal create_calls
            create_calls += 1
            await asyncio.sleep(0)
            self._members[name] = recovered_agent
            self._mailboxes[name] = asyncio.Queue()
            self._locks[name] = asyncio.Lock()

        import unittest.mock
        with unittest.mock.patch.object(AgentPool, "create_member", fake_create_member):
            results = await asyncio.gather(
                manager.dispatch("reviewer", "review code 1"),
                manager.dispatch("reviewer", "review code 2"),
            )

        assert results == ["LGTM", "LGTM"]
        assert create_calls == 1
        assert recovered_agent._call_count == 2

    asyncio.run(_run())


def test_team_manager_send_message_delegates_to_pool():
    async def _run():
        clear_tasks()
        manager = TeamManager()
        manager._active = True
        pool = AgentPool()
        fake_agent = FakeAgent()
        pool._members["reviewer"] = fake_agent
        pool._mailboxes["reviewer"] = asyncio.Queue()
        pool._locks["reviewer"] = asyncio.Lock()
        manager._pool = pool

        result = await manager.send_message("reviewer", "check this")
        assert result["results"][0]["status"] == "dispatched"
        for _ in range(20):
            if fake_agent._last_prompt:
                break
            await asyncio.sleep(0)
        assert "[Message from orchestrator]: check this" in fake_agent._last_prompt

        clear_tasks()

    asyncio.run(_run())


def test_team_manager_send_message_broadcasts_to_all_members():
    async def _run():
        clear_tasks()
        manager = TeamManager()
        manager._active = True
        pool = AgentPool()
        builder = FakeAgent()
        reviewer = FakeAgent()
        pool._members["builder"] = builder
        pool._members["reviewer"] = reviewer
        pool._mailboxes["builder"] = asyncio.Queue()
        pool._mailboxes["reviewer"] = asyncio.Queue()
        pool._locks["builder"] = asyncio.Lock()
        pool._locks["reviewer"] = asyncio.Lock()
        manager._pool = pool

        result = await manager.send_message("*", "announce", sender="builder", message_type="shutdown_response")
        assert all(entry["status"] == "dispatched" for entry in result["results"])
        for _ in range(20):
            if builder._last_prompt and reviewer._last_prompt:
                break
            await asyncio.sleep(0)
        assert "[Message from builder]: announce" in builder._last_prompt
        assert "[Message from builder]: announce" in reviewer._last_prompt

        clear_tasks()

    asyncio.run(_run())


def test_team_manager_clear_delegates_to_pool():
    manager = TeamManager()
    pool = AgentPool()
    fake_agent = FakeAgent()
    fake_agent._history = [{"role": "user", "content": "hello"}]
    pool._members["reviewer"] = fake_agent
    pool._mailboxes["reviewer"] = asyncio.Queue()
    pool._locks["reviewer"] = asyncio.Lock()
    manager._pool = pool

    asyncio.run(manager.clear())

    assert len(fake_agent._history) == 0


def test_team_manager_create_and_close_team():
    async def _run():
        manager = TeamManager()
        config = _make_config()
        fake_agent = FakeAgent()

        class FakeTool:
            def __init__(self, name: str):
                self._name = name

            @property
            def name(self) -> str:
                return self._name

        orchestrator = MagicMock()
        orchestrator._options.append_system_prompt = "original prompt"
        orchestrator._tool_pool = [FakeTool("Read"), FakeTool("Agent"), FakeTool("TeamCreate"), FakeTool("TeamDelete")]

        async def fake_create_member(self, name, definition, config, abort_signal=None):
            self._members[name] = fake_agent
            self._mailboxes[name] = asyncio.Queue()
            self._locks[name] = asyncio.Lock()

        import unittest.mock
        with unittest.mock.patch.object(AgentPool, "create_member", fake_create_member):
            await manager.create_team("dev-team", ["reviewer"], config, orchestrator)

        assert manager.is_active()
        assert manager._team_name == "dev-team"
        assert manager._pool is not None
        assert "reviewer" in manager._pool.member_names
        tool_names = [t.name for t in orchestrator._tool_pool]
        assert "TeamDispatch" in tool_names
        assert "SendMessage" in tool_names
        assert "TeamStatus" in tool_names
        assert "TeamDelete" in tool_names
        assert "Agent" not in tool_names
        assert "Read" not in tool_names
        assert tool_names.count("SendMessage") == 1
        assert "Team: dev-team" in orchestrator._options.append_system_prompt
        assert "call TeamDelete" in orchestrator._options.append_system_prompt
        assert "Do not also perform" in orchestrator._options.append_system_prompt
        member_tool_names = [t.name for t in fake_agent._tool_pool]
        assert "SendMessage" in member_tool_names, f"Member should have SendMessage, got: {member_tool_names}"
        assert "TeamCreate" not in member_tool_names, f"Member should not have TeamCreate, got: {member_tool_names}"
        assert "Agent" not in member_tool_names, f"Member should not have Agent, got: {member_tool_names}"
        assert "do not duplicate that same work" in fake_agent._options.append_system_prompt.lower()

        await manager.close_team(orchestrator)

        assert not manager.is_active()
        assert fake_agent._closed
        tool_names = [t.name for t in orchestrator._tool_pool]
        assert "TeamDispatch" not in tool_names
        assert "SendMessage" not in tool_names
        assert tool_names == ["Read", "Agent", "TeamCreate", "TeamDelete"]
        assert orchestrator._options.append_system_prompt == "original prompt"

    asyncio.run(_run())


def test_team_manager_create_team_adds_team_delete_when_missing_from_orchestrator():
    async def _run():
        manager = TeamManager()
        config = _make_config()
        fake_agent = FakeAgent()

        class FakeTool:
            def __init__(self, name: str):
                self._name = name

            @property
            def name(self) -> str:
                return self._name

        orchestrator = MagicMock()
        orchestrator._options.append_system_prompt = "original prompt"
        orchestrator._tool_pool = [FakeTool("Read"), FakeTool("Agent"), FakeTool("TeamCreate")]

        async def fake_create_member(self, name, definition, config, abort_signal=None):
            self._members[name] = fake_agent
            self._mailboxes[name] = asyncio.Queue()
            self._locks[name] = asyncio.Lock()

        import unittest.mock
        with unittest.mock.patch.object(AgentPool, "create_member", fake_create_member):
            await manager.create_team("dev-team", ["reviewer"], config, orchestrator)

        tool_names = [t.name for t in orchestrator._tool_pool]
        assert "TeamDelete" in tool_names

    asyncio.run(_run())


def test_team_manager_send_message_after_create_team_auto_dispatches():
    async def _run():
        manager = TeamManager()
        config = _make_config()
        fake_agent = FakeAgent(responses=["noted"])

        orchestrator = MagicMock()
        orchestrator._options.append_system_prompt = ""
        orchestrator._tool_pool = []

        async def fake_create_member(self, name, definition, config, abort_signal=None):
            self._members[name] = fake_agent
            self._mailboxes[name] = asyncio.Queue()
            self._locks[name] = asyncio.Lock()

        import unittest.mock
        with unittest.mock.patch.object(AgentPool, "create_member", fake_create_member):
            await manager.create_team("dev-team", ["reviewer"], config, orchestrator)

        await manager.send_message("reviewer", "please review this")

        for _ in range(20):
            if fake_agent._last_prompt:
                break
            await asyncio.sleep(0)

        assert "[Message from orchestrator]: please review this" in fake_agent._last_prompt
        assert MAILBOX_DISPATCH_TASK in fake_agent._last_prompt

        await manager.close_team(orchestrator)

    asyncio.run(_run())


def test_team_manager_prompt_removes_agent_guidance_from_original_prompt():
    async def _run():
        manager = TeamManager()
        config = _make_config()
        fake_agent = FakeAgent()

        class FakeTool:
            def __init__(self, name: str):
                self._name = name

            @property
            def name(self) -> str:
                return self._name

        orchestrator = MagicMock()
        orchestrator._options.append_system_prompt = "\n".join(
            [
                "# Tool Use Guidance",
                "If the user asks to use a named skill and it appears under Available Skills, call the Skill tool once.",
                "If work is multi-step, exploratory, or likely to benefit from parallelism, use the Agent tool with a concise description and prompt. Set run_in_background=true when the user can continue while it works.",
                "If a background task is assigned, do not duplicate the same work yourself unless it fails.",
                "If a team is active, prefer TeamDispatch for assigning work to members; use SendMessage only for coordination.",
                "# Configured Agents",
                "Use the Agent tool with the agent name when delegation is helpful.",
            ]
        )
        orchestrator._tool_pool = [FakeTool("Read"), FakeTool("Agent"), FakeTool("TeamCreate"), FakeTool("TeamDelete")]

        async def fake_create_member(self, name, definition, config, abort_signal=None):
            self._members[name] = fake_agent
            self._mailboxes[name] = asyncio.Queue()
            self._locks[name] = asyncio.Lock()

        import unittest.mock
        with unittest.mock.patch.object(AgentPool, "create_member", fake_create_member):
            await manager.create_team("dev-team", ["reviewer"], config, orchestrator)

        prompt = orchestrator._options.append_system_prompt
        assert "call the Skill tool once" in prompt
        assert "prefer TeamDispatch" in prompt
        assert "multi-step, exploratory" not in prompt
        assert "run_in_background" not in prompt
        assert "do not duplicate" not in prompt
        assert "# Configured Agents" not in prompt
        assert "Use the Agent tool with the agent name" not in prompt

    asyncio.run(_run())


def test_team_manager_ensure_orchestrator_team_state_reapplies_after_reset():
    async def _run():
        manager = TeamManager()
        config = _make_config()
        fake_agent = FakeAgent()

        class FakeTool:
            def __init__(self, name: str):
                self._name = name

            @property
            def name(self) -> str:
                return self._name

        orchestrator = MagicMock()
        orchestrator._options.append_system_prompt = "original prompt"
        orchestrator._tool_pool = [FakeTool("Read"), FakeTool("Agent"), FakeTool("SendMessage"), FakeTool("TeamDelete")]
        orchestrator._initialized = True

        async def fake_initialize():
            orchestrator._tool_pool = [FakeTool("Read"), FakeTool("Agent"), FakeTool("SendMessage"), FakeTool("TeamDelete")]
            orchestrator._initialized = True

        orchestrator._initialize = fake_initialize

        async def fake_create_member(self, name, definition, config, abort_signal=None):
            self._members[name] = fake_agent
            self._mailboxes[name] = asyncio.Queue()
            self._locks[name] = asyncio.Lock()

        import unittest.mock
        with unittest.mock.patch.object(AgentPool, "create_member", fake_create_member):
            await manager.create_team("dev-team", ["reviewer"], config, orchestrator)

        orchestrator._initialized = False
        orchestrator._tool_pool = [FakeTool("Read"), FakeTool("Agent"), FakeTool("SendMessage")]

        await manager.ensure_orchestrator_team_state(orchestrator)

        tool_names = [t.name for t in orchestrator._tool_pool]
        assert tool_names.count("SendMessage") == 1
        assert "TeamDispatch" in tool_names
        assert "TeamStatus" in tool_names
        assert "TeamDelete" in tool_names
        assert "Agent" not in tool_names
        assert "Read" not in tool_names
        assert "Team: dev-team" in orchestrator._options.append_system_prompt

    asyncio.run(_run())


def test_team_manager_create_team_syncs_active_engine_tool_map():
    async def _run():
        manager = TeamManager()
        config = _make_config()
        fake_agent = FakeAgent()

        class FakeTool:
            def __init__(self, name: str):
                self._name = name

            @property
            def name(self) -> str:
                return self._name

        orchestrator = MagicMock()
        orchestrator._options.append_system_prompt = "original prompt"
        orchestrator._tool_pool = [FakeTool("Read"), FakeTool("Agent"), FakeTool("TeamCreate"), FakeTool("TeamDelete")]
        orchestrator._engine = MagicMock()
        orchestrator._engine._config = MagicMock()
        orchestrator._engine._tool_map = {tool.name: tool for tool in orchestrator._tool_pool}

        async def fake_create_member(self, name, definition, config, abort_signal=None):
            self._members[name] = fake_agent
            self._mailboxes[name] = asyncio.Queue()
            self._locks[name] = asyncio.Lock()

        import unittest.mock
        with unittest.mock.patch.object(AgentPool, "create_member", fake_create_member):
            await manager.create_team("dev-team", ["reviewer"], config, orchestrator)

        assert "TeamDispatch" in orchestrator._engine._tool_map
        assert "TeamStatus" in orchestrator._engine._tool_map
        assert "Agent" not in orchestrator._engine._tool_map
        assert "Read" not in orchestrator._engine._tool_map
        assert orchestrator._engine._config.tools is orchestrator._tool_pool

    asyncio.run(_run())


def test_team_manager_close_team_restores_active_engine_tool_map():
    async def _run():
        manager = TeamManager()
        config = _make_config()
        fake_agent = FakeAgent()

        class FakeTool:
            def __init__(self, name: str):
                self._name = name

            @property
            def name(self) -> str:
                return self._name

        original_tools = [FakeTool("Read"), FakeTool("Agent"), FakeTool("TeamCreate"), FakeTool("TeamDelete")]
        orchestrator = MagicMock()
        orchestrator._options.append_system_prompt = "original prompt"
        orchestrator._tool_pool = list(original_tools)
        orchestrator._engine = MagicMock()
        orchestrator._engine._config = MagicMock()
        orchestrator._engine._tool_map = {tool.name: tool for tool in orchestrator._tool_pool}

        async def fake_create_member(self, name, definition, config, abort_signal=None):
            self._members[name] = fake_agent
            self._mailboxes[name] = asyncio.Queue()
            self._locks[name] = asyncio.Lock()

        import unittest.mock
        with unittest.mock.patch.object(AgentPool, "create_member", fake_create_member):
            await manager.create_team("dev-team", ["reviewer"], config, orchestrator)

        await manager.close_team(orchestrator)

        assert sorted(orchestrator._engine._tool_map) == ["Agent", "Read", "TeamCreate", "TeamDelete"]
        assert orchestrator._engine._config.tools is orchestrator._tool_pool

    asyncio.run(_run())


def test_team_manager_clear_cleared():
    async def _run():
        manager = TeamManager()
        config = _make_config()
        fake_agent = FakeAgent()

        orchestrator = MagicMock()
        orchestrator._options.append_system_prompt = ""
        orchestrator._tool_pool = []

        async def fake_create_member(self, name, definition, config, abort_signal=None):
            self._members[name] = fake_agent
            self._mailboxes[name] = asyncio.Queue()
            self._locks[name] = asyncio.Lock()

        import unittest.mock
        with unittest.mock.patch.object(AgentPool, "create_member", fake_create_member):
            await manager.create_team("team1", ["reviewer"], config, orchestrator)

        fake_agent._history = [{"role": "user", "content": "hello"}]
        await manager.clear()

        assert len(fake_agent._history) == 0

        await manager.close_team(orchestrator)

    asyncio.run(_run())


def test_agent_pool_dispatch_async_marks_busy():
    async def _run():
        import unittest.mock
        pool = AgentPool()
        fake_agent = FakeAgent(responses=["done"])
        pool._members["reviewer"] = fake_agent
        pool._mailboxes["reviewer"] = asyncio.Queue()
        pool._locks["reviewer"] = asyncio.Lock()

        with unittest.mock.patch("rooster_code.runtime._track_background_task"):
            with unittest.mock.patch("rooster_code.runtime._update_background_subagent_task", new_callable=unittest.mock.AsyncMock):
                result = await pool.dispatch_async("reviewer", "review code", "task-1", ".", {})

        assert result == "task-1"
        assert pool.is_busy("reviewer")

    asyncio.run(_run())


def test_agent_pool_dispatch_async_rejects_busy_member():
    pool = AgentPool()
    fake_agent = FakeAgent()
    pool._members["reviewer"] = fake_agent
    pool._mailboxes["reviewer"] = asyncio.Queue()
    pool._locks["reviewer"] = asyncio.Lock()
    pool._busy.add("reviewer")

    result = asyncio.run(pool.dispatch_async("reviewer", "review code", "task-1", ".", {}))
    assert "Error" in result
    assert "busy" in result.lower()


def test_agent_pool_dispatch_async_rejects_unhealthy():
    pool = AgentPool()
    fake_agent = FakeAgent()
    pool._members["reviewer"] = fake_agent
    pool._mailboxes["reviewer"] = asyncio.Queue()
    pool._locks["reviewer"] = asyncio.Lock()
    pool._unhealthy.add("reviewer")

    result = asyncio.run(pool.dispatch_async("reviewer", "review code", "task-1", ".", {}))
    assert "Error" in result
    assert "unavailable" in result.lower()


def test_agent_pool_dispatch_async_rejects_unknown():
    pool = AgentPool()
    result = asyncio.run(pool.dispatch_async("unknown", "do thing", "task-1", ".", {}))
    assert "Error" in result
    assert "unknown" in result.lower()


def test_agent_pool_dispatch_async_preserves_assistant_messages_when_text_empty():
    async def _run():
        import unittest.mock

        class MessageOnlyAgent:
            async def prompt(self, text: str, overrides: dict[str, Any] | None = None):
                return FakeQueryResult(
                    text="",
                    messages=[
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Outcome: Team review complete.\n\n## Findings\nMember found the missing completion handoff.",
                                }
                            ],
                        }
                    ],
                )

            async def close(self):
                return None

        pool = AgentPool()
        pool._members["reviewer"] = MessageOnlyAgent()
        pool._mailboxes["reviewer"] = asyncio.Queue()
        pool._locks["reviewer"] = asyncio.Lock()

        with unittest.mock.patch("rooster_code.runtime._track_background_task"):
            with unittest.mock.patch("rooster_code.runtime._update_background_subagent_task", new_callable=unittest.mock.AsyncMock) as mock_update:
                result = await pool.dispatch_async("reviewer", "review code", "task-1", ".", {})

        assert result == "task-1"

        for _ in range(50):
            if mock_update.await_count:
                break
            await asyncio.sleep(0)

        assert mock_update.await_args is not None
        output = str(mock_update.await_args.kwargs["output"])
        assert "Team review complete." in output
        assert "Member found the missing completion handoff." in output

    asyncio.run(_run())


def test_agent_pool_close_clears_busy():
    pool = AgentPool()
    fake_agent = FakeAgent()
    pool._members["a"] = fake_agent
    pool._mailboxes["a"] = asyncio.Queue()
    pool._locks["a"] = asyncio.Lock()
    pool._busy.add("a")

    asyncio.run(pool.close_all())
    assert len(pool._busy) == 0


def test_team_manager_info_shows_busy():
    manager = TeamManager()
    manager._active = True
    manager._team_name = "team1"
    pool = AgentPool()
    pool._members["reviewer"] = FakeAgent()
    pool._mailboxes["reviewer"] = asyncio.Queue()
    pool._locks["reviewer"] = asyncio.Lock()
    pool._busy.add("reviewer")
    manager._pool = pool

    info = manager.info()
    assert info["members"]["reviewer"] == "busy"


def test_team_status_tool_active_team():
    manager = TeamManager()
    manager._active = True
    manager._team_id = "abc123"
    manager._team_name = "dev-team"
    pool = AgentPool()
    pool._members["reviewer"] = FakeAgent()
    pool._mailboxes["reviewer"] = asyncio.Queue()
    pool._locks["reviewer"] = asyncio.Lock()
    manager._pool = pool

    tool = TeamStatusTool(manager)
    result = asyncio.run(tool.call({}, ToolContext(cwd=".", env={})))

    assert not result.is_error
    content = str(result.content)
    assert "dev-team" in content
    assert "reviewer" in content
    assert "idle" in content


def test_team_status_tool_no_active_team():
    manager = TeamManager()
    tool = TeamStatusTool(manager)
    result = asyncio.run(tool.call({}, ToolContext(cwd=".", env={})))
    assert "No team is active" in str(result.content)


def test_dispatch_tool_busy_member_returns_error():
    async def _run():
        manager = TeamManager()
        manager._active = True
        pool = AgentPool()
        fake_agent = FakeAgent(responses=["done"])
        pool._members["reviewer"] = fake_agent
        pool._mailboxes["reviewer"] = asyncio.Queue()
        pool._locks["reviewer"] = asyncio.Lock()
        pool._busy.add("reviewer")
        manager._pool = pool

        tool = TeamDispatchTool(manager)

        import unittest.mock
        with unittest.mock.patch("rooster_code.runtime._create_background_subagent_task", new_callable=unittest.mock.AsyncMock) as mock_create_task:
            mock_create_task.return_value = "test-task-err"
            with unittest.mock.patch("rooster_code.runtime._update_background_subagent_task", new_callable=unittest.mock.AsyncMock):
                result = await tool.call({"member": "reviewer", "task": "review this"}, ToolContext(cwd=".", env={}))

        assert result.is_error
        assert "busy" in str(result.content).lower()

    asyncio.run(_run())


def test_agent_pool_dispatch_async_clears_busy_after_completion():
    """After a normal dispatch_async run completes, the member is no longer busy."""
    async def _run():
        import unittest.mock
        pool = AgentPool()
        fake_agent = FakeAgent(responses=["all done"])
        pool._members["reviewer"] = fake_agent
        pool._mailboxes["reviewer"] = asyncio.Queue()
        pool._locks["reviewer"] = asyncio.Lock()

        with unittest.mock.patch("rooster_code.runtime._track_background_task"):
            with unittest.mock.patch("rooster_code.runtime._update_background_subagent_task", new_callable=unittest.mock.AsyncMock):
                await pool.dispatch_async("reviewer", "review code", "task-ok", ".", {})

        # The dispatch_async spawns a background task; await all of them
        for t in list(pool._dispatch_tasks):
            await t

        assert pool._busy == set()
        assert not pool.is_busy("reviewer")

    asyncio.run(_run())


def test_agent_pool_dispatch_async_clears_busy_after_failure():
    """After a dispatch_async run crashes, busy is still cleared in the finally block."""
    async def _run():
        import unittest.mock
        pool = AgentPool()
        # Agent that raises on prompt to simulate failure
        fake_agent = FakeAgent()  # no responses set → KeyError on prompt
        pool._members["reviewer"] = fake_agent
        pool._mailboxes["reviewer"] = asyncio.Queue()
        pool._locks["reviewer"] = asyncio.Lock()

        with unittest.mock.patch("rooster_code.runtime._track_background_task"):
            with unittest.mock.patch("rooster_code.runtime._update_background_subagent_task", new_callable=unittest.mock.AsyncMock):
                await pool.dispatch_async("reviewer", "review code", "task-fail", ".", {})

        for t in list(pool._dispatch_tasks):
            with __import__("contextlib").suppress(Exception):
                await t

        assert pool._busy == set()
        assert not pool.is_busy("reviewer")

    asyncio.run(_run())


def test_dispatch_tool_task_creation_failure_reports_error():
    """TeamDispatch returns an error tool result when background-task creation fails."""
    async def _run():
        import unittest.mock
        from rooster_code.team import TeamDispatchTool, TeamManager
        from open_agent_sdk import ToolContext

        manager = TeamManager()
        manager._active = True
        manager._team_id = "t123"
        manager._team_name = "test-team"
        manager._member_definitions = {"reviewer": {"description": "reviews"}}
        manager._config = _make_config()

        pool = AgentPool()
        fake_agent = FakeAgent()
        pool._members["reviewer"] = fake_agent
        pool._mailboxes["reviewer"] = asyncio.Queue()
        pool._locks["reviewer"] = asyncio.Lock()
        manager._pool = pool

        tool = TeamDispatchTool(manager)

        with unittest.mock.patch("rooster_code.runtime._create_background_subagent_task", new_callable=unittest.mock.AsyncMock) as mock_create:
            mock_create.side_effect = RuntimeError("simulated task creation failure")
            with unittest.mock.patch("rooster_code.runtime._update_background_subagent_task", new_callable=unittest.mock.AsyncMock):
                result = await tool.call({"member": "reviewer", "task": "test"}, ToolContext(cwd=".", env={}))

        assert result.is_error
        assert "simulated task creation failure" in str(result.content)

    asyncio.run(_run())


def test_wake_member_for_messages_inactive_team():
    """_wake_member_for_messages returns inactive when team is not active."""
    async def _run():
        manager = TeamManager()
        result = await manager._wake_member_for_messages("reviewer", ".", {})
        assert result == {"status": "inactive"}
    asyncio.run(_run())


def test_wake_member_for_messages_no_messages():
    """_wake_member_for_messages returns no_messages when mailbox is empty."""
    async def _run():
        manager = TeamManager()
        manager._active = True
        manager._team_id = "t1"
        manager._team_name = "test"
        manager._config = _make_config()

        pool = AgentPool()
        fake_agent = FakeAgent()
        pool._members["reviewer"] = fake_agent
        pool._mailboxes["reviewer"] = asyncio.Queue()  # empty
        pool._locks["reviewer"] = asyncio.Lock()
        manager._pool = pool

        result = await manager._wake_member_for_messages("reviewer", ".", {})
        assert result == {"status": "no_messages"}
    asyncio.run(_run())


def test_wake_member_for_messages_queued_busy():
    """_wake_member_for_messages returns queued_busy when member is busy."""
    async def _run():
        manager = TeamManager()
        manager._active = True
        manager._team_id = "t2"
        manager._team_name = "test"
        manager._config = _make_config()

        pool = AgentPool()
        fake_agent = FakeAgent()
        pool._members["reviewer"] = fake_agent
        pool._mailboxes["reviewer"] = asyncio.Queue()
        pool._mailboxes["reviewer"].put_nowait({"type": "text", "from": "alpha", "content": "hi"})
        pool._locks["reviewer"] = asyncio.Lock()
        pool._busy.add("reviewer")
        manager._pool = pool

        result = await manager._wake_member_for_messages("reviewer", ".", {})
        assert result == {"status": "queued_busy"}
    asyncio.run(_run())


def test_wake_member_for_messages_queued_unhealthy():
    """When unhealthy and recovery fails, _wake_member_for_messages returns queued_unhealthy."""
    async def _run():
        manager = TeamManager()
        manager._active = True
        manager._team_id = "t3"
        manager._team_name = "test"
        # No _member_definitions → recovery will raise RuntimeError
        manager._member_definitions = {}
        manager._config = _make_config()

        pool = AgentPool()
        fake_agent = FakeAgent()
        pool._members["reviewer"] = fake_agent
        pool._mailboxes["reviewer"] = asyncio.Queue()
        pool._mailboxes["reviewer"].put_nowait({"type": "text", "from": "alpha", "content": "hi"})
        pool._locks["reviewer"] = asyncio.Lock()
        pool._unhealthy.add("reviewer")
        manager._pool = pool

        result = await manager._wake_member_for_messages("reviewer", ".", {})
        assert result == {"status": "queued_unhealthy"}
    asyncio.run(_run())


def test_wake_member_for_messages_recovers_unhealthy():
    """When unhealthy but recovery succeeds, _wake_member_for_messages recovers and dispatches."""
    async def _run():
        import unittest.mock
        manager = TeamManager()
        manager._active = True
        manager._team_id = "t4"
        manager._team_name = "test"
        manager._member_definitions = {"reviewer": {"description": "reviews"}}
        manager._config = _make_config()

        pool = AgentPool()
        fake_agent = FakeAgent(responses=["recovered"])
        pool._members["reviewer"] = fake_agent
        pool._mailboxes["reviewer"] = asyncio.Queue()
        pool._mailboxes["reviewer"].put_nowait({"type": "text", "from": "alpha", "content": "hi"})
        pool._locks["reviewer"] = asyncio.Lock()
        pool._unhealthy.add("reviewer")
        manager._pool = pool

        with unittest.mock.patch("rooster_code.runtime._create_background_subagent_task", return_value="task-recovered"):
            with unittest.mock.patch("rooster_code.runtime._track_background_task"):
                with unittest.mock.patch("rooster_code.runtime._update_background_subagent_task", new_callable=unittest.mock.AsyncMock):
                    result = await manager._wake_member_for_messages("reviewer", ".", {})

        assert result["status"] == "dispatched"
        assert "reviewer" not in pool._unhealthy  # recovery cleared unhealthy flag
    asyncio.run(_run())


def test_send_message_maintains_fifo_under_concurrent_dispatch():
    """send_message (sync) + _inject_mailbox (sync) are atomic under asyncio cooperative model."""
    async def _run():
        import unittest.mock
        pool = AgentPool()
        prompt_started = asyncio.Event()
        release = asyncio.Event()

        class SlowAgent(FakeAgent):
            async def prompt(self, text, overrides=None):
                prompt_started.set()
                await release.wait()
                return await super().prompt(text, overrides)

        pool._members["reviewer"] = SlowAgent(responses=["done"])
        pool._mailboxes["reviewer"] = asyncio.Queue()
        pool._locks["reviewer"] = asyncio.Lock()

        pool.send_message("reviewer", {"from": "a", "content": "M1"})
        pool.send_message("reviewer", {"from": "a", "content": "M2"})

        with unittest.mock.patch("rooster_code.runtime._track_background_task"):
            with unittest.mock.patch("rooster_code.runtime._update_background_subagent_task", new_callable=unittest.mock.AsyncMock):
                asyncio.create_task(
                    pool.dispatch_async("reviewer", "check", "task-fifo", ".", {})
                )
                await prompt_started.wait()

                pool.send_message("reviewer", {"from": "b", "content": "M3"})

                release.set()
                await asyncio.gather(*list(pool._dispatch_tasks))

        drained = []
        while not pool._mailboxes["reviewer"].empty():
            drained.append(pool._mailboxes["reviewer"].get_nowait())
        assert len(drained) == 3
        assert drained[0]["content"] == "M1"
        assert drained[1]["content"] == "M2"
        assert drained[2]["content"] == "M3"

    asyncio.run(_run())


def test_wake_member_for_messages_queued_unavailable():
    """_wake_member_for_messages returns queued_unavailable when agent has no callable prompt."""
    async def _run():
        manager = TeamManager()
        manager._active = True
        manager._team_id = "t5"
        manager._team_name = "test"
        manager._config = _make_config()

        pool = AgentPool()
        agent_without_prompt = object()  # no prompt attribute
        pool._members["worker"] = agent_without_prompt
        pool._mailboxes["worker"] = asyncio.Queue()
        pool._mailboxes["worker"].put_nowait({"type": "text", "from": "alpha", "content": "hi"})
        pool._locks["worker"] = asyncio.Lock()
        manager._pool = pool

        result = await manager._wake_member_for_messages("worker", ".", {})
        assert result == {"status": "queued_unavailable"}
    asyncio.run(_run())


def test_create_team_persists_snapshot_to_daemon_db():
    """After create_team succeeds, TeamSnapshotStore has the team entry."""
    from rooster_code.daemon import TeamSnapshotStore

    db_path = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name
    _team_id = None
    try:
        async def _run():
            nonlocal _team_id
            manager = TeamManager()
            manager.enable_snapshot_persistence(db_path)
            config = _make_config()
            fake_agent = FakeAgent()
            orchestrator = MagicMock()
            orchestrator._options.append_system_prompt = "original"
            orchestrator._tool_pool = []

            import unittest.mock
            async def fake_create_member(self, name, definition, config, abort_signal=None):
                self._members[name] = fake_agent
                self._mailboxes[name] = asyncio.Queue()
                self._locks[name] = asyncio.Lock()
            with unittest.mock.patch.object(AgentPool, "create_member", fake_create_member):
                await manager.create_team("dev-team", ["reviewer"], config, orchestrator)

            assert manager.is_active()
            _team_id = manager._team_id

        asyncio.run(_run())

        assert _team_id is not None
        store = TeamSnapshotStore(db_path=db_path)
        snap = store.get_team(_team_id)
        assert snap is not None, "Expected team snapshot in daemon DB after create_team + save_team_snapshot"
        assert snap["team_name"] == "dev-team"
        store.close()
    finally:
        Path(db_path).unlink(missing_ok=True)


def test_close_team_removes_snapshot_from_daemon_db():
    """After close_team + drop_team_snapshot, TeamSnapshotStore has no entry."""
    from rooster_code.daemon import TeamSnapshotStore

    db_path = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name
    _team_id = None
    try:
        async def _run():
            nonlocal _team_id
            manager = TeamManager()
            manager.enable_snapshot_persistence(db_path)
            config = _make_config()
            fake_agent = FakeAgent()
            orchestrator = MagicMock()
            orchestrator._options.append_system_prompt = "original"
            orchestrator._tool_pool = []

            import unittest.mock
            async def fake_create_member(self, name, definition, config, abort_signal=None):
                self._members[name] = fake_agent
                self._mailboxes[name] = asyncio.Queue()
                self._locks[name] = asyncio.Lock()
            with unittest.mock.patch.object(AgentPool, "create_member", fake_create_member):
                await manager.create_team("temp-team", ["reviewer"], config, orchestrator)

            assert manager.is_active()
            _team_id = manager._team_id
            await manager.close_team(orchestrator)

        asyncio.run(_run())

        assert _team_id is not None
        store = TeamSnapshotStore(db_path=db_path)
        snap = store.get_team(_team_id)
        assert snap is None, "Expected NO team snapshot after close_team + drop_team_snapshot"
        store.close()
    finally:
        Path(db_path).unlink(missing_ok=True)


def test_read_team_snapshots_returns_persisted_teams():
    """read_team_snapshots returns all persisted team snapshots."""
    from rooster_code.daemon import save_team_snapshot, drop_team_snapshot, read_team_snapshots

    db_path = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name
    try:
        save_team_snapshot("t1", "alpha", '{"a": 1}', db_path=db_path)
        save_team_snapshot("t2", "beta", '{"b": 2}', db_path=db_path)

        snaps = read_team_snapshots(db_path=db_path)
        assert len(snaps) == 2
        names = {s["team_name"] for s in snaps}
        assert names == {"alpha", "beta"}

        drop_team_snapshot("t1", db_path=db_path)
        snaps = read_team_snapshots(db_path=db_path)
        assert len(snaps) == 1
        assert snaps[0]["team_name"] == "beta"
    finally:
        Path(db_path).unlink(missing_ok=True)


def test_dispatch_tool_skips_task_creation_for_unknown_member():
    """TeamDispatch does NOT create a background task when member is unknown."""
    async def _run():
        import unittest.mock
        from open_agent_sdk import ToolContext

        manager = TeamManager()
        manager._active = True
        manager._team_id = "t42"
        manager._pool = AgentPool()

        tool = TeamDispatchTool(manager)

        create_called = False
        async def fake_create_task(name, task_spec, cwd, env):
            nonlocal create_called
            create_called = True
            return "task-123"

        with unittest.mock.patch("rooster_code.runtime._create_background_subagent_task", fake_create_task):
            result = await tool.call({"member": "nonexistent", "task": "do stuff"}, ToolContext(cwd=".", env={}))

        assert result.is_error
        assert not create_called, "Should not have created a background task for unknown member"

    asyncio.run(_run())


def test_dispatch_tool_skips_task_creation_for_inactive_team():
    """TeamDispatch does NOT create a background task when no team is active."""
    async def _run():
        import unittest.mock
        from open_agent_sdk import ToolContext

        manager = TeamManager()

        tool = TeamDispatchTool(manager)

        create_called = False
        async def fake_create_task(name, task_spec, cwd, env):
            nonlocal create_called
            create_called = True
            return "task-123"

        with unittest.mock.patch("rooster_code.runtime._create_background_subagent_task", fake_create_task):
            result = await tool.call({"member": "reviewer", "task": "do stuff"}, ToolContext(cwd=".", env={}))

        assert result.is_error
        assert not create_called, "Should not have created a background task for inactive team"

    asyncio.run(_run())
