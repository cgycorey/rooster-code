"""Tests for rooster_code.team — AgentPool, TeamManager, TeamDispatchTool, TeamSendMessageTool, patch_tool_pool."""

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest
from open_agent_sdk.types import ToolContext

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
        for _ in range(20):
            if reviewer._last_prompt:
                break
            await asyncio.sleep(0)
        assert "[Message from orchestrator]: check this" in reviewer._last_prompt
        assert MAILBOX_DISPATCH_TASK in reviewer._last_prompt

    asyncio.run(_run())


def test_send_message_tool_uses_bound_sender_name():
    async def _run():
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
        for _ in range(20):
            if reviewer._last_prompt:
                break
            await asyncio.sleep(0)
        assert "[Message from builder]: check this" in reviewer._last_prompt

    asyncio.run(_run())


def test_send_message_tool_supports_sdk_message_type():
    async def _run():
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
        for _ in range(20):
            if reviewer._last_prompt:
                break
            await asyncio.sleep(0)
        assert "[Message from builder]: please stop" in reviewer._last_prompt

    asyncio.run(_run())


def test_send_message_tool_supports_broadcast():
    async def _run():
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
        assert str(result.content) == "Message broadcast to all agents."
        for _ in range(20):
            if builder._last_prompt and reviewer._last_prompt:
                break
            await asyncio.sleep(0)
        assert "[Message from builder]: heads up" in builder._last_prompt
        assert "[Message from builder]: heads up" in reviewer._last_prompt

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
    async def _run():
        manager = TeamManager()
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

    asyncio.run(_run())


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
        manager = TeamManager()
        manager._active = True
        pool = AgentPool()
        fake_agent = FakeAgent()
        pool._members["reviewer"] = fake_agent
        pool._mailboxes["reviewer"] = asyncio.Queue()
        pool._locks["reviewer"] = asyncio.Lock()
        manager._pool = pool

        await manager.send_message("reviewer", "check this")
        for _ in range(20):
            if fake_agent._last_prompt:
                break
            await asyncio.sleep(0)
        assert "[Message from orchestrator]: check this" in fake_agent._last_prompt

    asyncio.run(_run())


def test_team_manager_send_message_broadcasts_to_all_members():
    async def _run():
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

        await manager.send_message("*", "announce", sender="builder", message_type="shutdown_response")
        for _ in range(20):
            if builder._last_prompt and reviewer._last_prompt:
                break
            await asyncio.sleep(0)
        assert "[Message from builder]: announce" in builder._last_prompt
        assert "[Message from builder]: announce" in reviewer._last_prompt

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
        orchestrator._tool_pool = [FakeTool("Read"), FakeTool("Agent"), FakeTool("TeamCreate")]

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
        assert "Agent" not in tool_names
        assert "Read" not in tool_names
        assert tool_names.count("SendMessage") == 1
        assert "Team: dev-team" in orchestrator._options.append_system_prompt
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
        assert tool_names == ["Read", "Agent", "TeamCreate"]
        assert orchestrator._options.append_system_prompt == "original prompt"

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
        orchestrator._tool_pool = [FakeTool("Read"), FakeTool("Agent"), FakeTool("SendMessage")]
        orchestrator._initialized = True

        async def fake_initialize():
            orchestrator._tool_pool = [FakeTool("Read"), FakeTool("Agent"), FakeTool("SendMessage")]
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
        assert "Agent" not in tool_names
        assert "Read" not in tool_names
        assert "Team: dev-team" in orchestrator._options.append_system_prompt

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
