import asyncio

import cock_code.runtime as runtime
import pytest
from cock_code.config import RuntimeConfig

from cock_code.runtime import build_agent_options, create_runtime_agent
from cock_code.runtime import enforce_session_retention, find_requested_agent_name


def test_build_agent_options_uses_explicit_api_fields() -> None:
    config = RuntimeConfig(
        api_key="abc",
        base_url="https://example.test",
        model="m1",
        api_type="openai-completions",
    )

    options = build_agent_options(config)

    assert options.api_key == "abc"
    assert options.base_url == "https://example.test"
    assert options.model == "m1"
    assert options.api_type == "openai-completions"


def test_build_agent_options_carries_runtime_configuration() -> None:
    config = RuntimeConfig(
        api_key="abc",
        base_url="https://example.test",
        model="m1",
        cwd="/tmp/project",
        allowed_tools=["Read"],
        disallowed_tools=["Bash"],
        resume="sess-1",
        session_id="sess-2",
        continue_session=True,
        fork_session="sess-0",
        persist_session=False,
        permission_mode="acceptEdits",
        max_turns=12,
        max_budget_usd=4.5,
        max_tokens=900,
        thinking_budget=321,
        debug=True,
        sandbox=True,
        include_partials=True,
        env={"A": "1"},
        custom_headers={"X-Test": "1"},
        agents={"reviewer": {"description": "code reviewer"}},
        hooks={"PreToolUse": []},
        json_schema={"type": "object"},
        mcp_servers={"fs": {"type": "stdio", "command": "echo", "args": ["hi"]}},
        extra_args={"temperature": 0},
    )

    options = build_agent_options(config)

    assert options.cwd == "/tmp/project"
    assert options.allowed_tools == ["Read"]
    assert options.disallowed_tools == ["Bash"]
    assert options.resume == "sess-1"
    assert options.session_id == "sess-2"
    assert options.continue_session is True
    assert options.fork_session == "sess-0"
    assert options.persist_session is False
    assert options.permission_mode.value == "acceptEdits"
    assert options.max_turns == 12
    assert options.max_budget_usd == 4.5
    assert options.max_tokens == 900
    assert options.thinking is not None
    assert options.thinking.budget_tokens == 321
    assert options.debug is True
    assert options.sandbox is True
    assert options.include_partial_messages is True
    assert options.env == {"A": "1"}
    assert options.custom_headers == {"X-Test": "1"}
    assert options.agents == {"reviewer": {"description": "code reviewer"}}
    assert "reviewer" in options.append_system_prompt
    assert options.hooks == {"PreToolUse": []}
    assert options.json_schema == {"type": "object"}
    assert options.mcp_servers == {"fs": {"type": "stdio", "command": "echo", "args": ["hi"]}}
    assert options.extra_args == {"temperature": 0}


def test_create_runtime_agent_does_not_inject_custom_transport(monkeypatch) -> None:
    class FakeAgent:
        _client = None

    monkeypatch.setattr("cock_code.runtime.create_agent", lambda options: FakeAgent())

    agent = create_runtime_agent(
        RuntimeConfig(
            api_key="test-key",
            base_url="https://nano-gpt.com/api/v1",
            model="test-model",
            api_type="openai-completions",
        )
    )

    assert isinstance(agent, FakeAgent)
    assert agent._client is None


def test_create_runtime_agent_replaces_placeholder_agent_tool_after_initialize(monkeypatch) -> None:
    class PlaceholderAgentTool:
        name = "Agent"

    class ReadTool:
        name = "Read"

    class FakeAgent:
        def __init__(self) -> None:
            self._client = None
            self._tool_pool = []

        async def _initialize(self) -> None:
            self._tool_pool = [PlaceholderAgentTool(), ReadTool()]

    monkeypatch.setattr("cock_code.runtime.create_agent", lambda options: FakeAgent())

    agent = create_runtime_agent(
        RuntimeConfig(
            api_key="test-key",
            base_url="https://nano-gpt.com/api/v1",
            model="test-model",
            agents={"reviewer": {"description": "code reviewer"}},
        )
    )

    asyncio.run(agent._initialize())

    assert [tool.name for tool in agent._tool_pool] == ["Agent", "Read"]
    assert agent._tool_pool[0].__class__.__name__ == "RuntimeAgentTool"


def test_create_runtime_agent_replaces_read_and_edit_tools_after_initialize(monkeypatch) -> None:
    class ReadTool:
        name = "Read"

    class EditTool:
        name = "Edit"

    class OtherTool:
        name = "Bash"

    class FakeAgent:
        def __init__(self) -> None:
            self._client = None
            self._tool_pool = []

        async def _initialize(self) -> None:
            self._tool_pool = [ReadTool(), EditTool(), OtherTool()]

    monkeypatch.setattr("cock_code.runtime.create_agent", lambda options: FakeAgent())

    agent = create_runtime_agent(RuntimeConfig(api_key="test", base_url="https://nano-gpt.com/api/v1", model="m1"))

    asyncio.run(agent._initialize())

    assert [tool.name for tool in agent._tool_pool] == ["Read", "Edit", "Bash", "Agent"]
    assert agent._tool_pool[0].__class__.__name__ == "RuntimeReadTool"
    assert agent._tool_pool[1].__class__.__name__ == "RuntimeEditTool"


def test_create_runtime_agent_adds_default_task_agent_without_agents(monkeypatch) -> None:
    class ReadTool:
        name = "Read"

    class AgentTool:
        name = "Agent"

    class FakeAgent:
        def __init__(self) -> None:
            self._client = None
            self._tool_pool = []

        async def _initialize(self) -> None:
            self._tool_pool = [ReadTool(), AgentTool()]

    monkeypatch.setattr("cock_code.runtime.create_agent", lambda options: FakeAgent())

    agent = create_runtime_agent(RuntimeConfig(api_key="test", base_url="https://nano-gpt.com/api/v1", model="m1"))

    asyncio.run(agent._initialize())

    assert [tool.name for tool in agent._tool_pool] == ["Read", "Agent"]


def test_build_agent_options_includes_default_task_agent_context() -> None:
    options = build_agent_options(RuntimeConfig(api_key="a", base_url="https://example.test", model="m"))

    assert "task" in options.append_system_prompt.lower()


def test_find_requested_agent_name_uses_default_task_agent() -> None:
    result = find_requested_agent_name(
        RuntimeConfig(api_key="a", base_url="https://example.test", model="m"),
        "Use an agent to summarize this request.",
    )

    assert result == "task"


def test_enforce_session_retention_deletes_sessions_beyond_limit(monkeypatch) -> None:
    deleted: list[str] = []

    async def fake_list_sessions() -> list[dict[str, object]]:
        return [{"id": f"sess-{index}", "updatedAt": f"2026-04-{30-index:02d}T00:00:00"} for index in range(25)]

    async def fake_delete_session(session_id: str) -> bool:
        deleted.append(session_id)
        return True

    monkeypatch.setattr("cock_code.runtime.sdk_list_sessions", fake_list_sessions)
    monkeypatch.setattr("cock_code.runtime.sdk_delete_session", fake_delete_session)

    asyncio.run(enforce_session_retention(limit=20))

    assert deleted == [f"sess-{index}" for index in range(20, 25)]


def test_compact_current_session_rewrites_agent_history(monkeypatch) -> None:
    original_history = [
        {"role": "user", "content": [{"type": "text", "text": "hello"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
    ]
    compacted_history = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "[Previous conversation summary]\n\nsummary"}],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "I understand the context. Let me continue from where we left off."}],
        },
    ]

    class FakeAgent:
        def __init__(self) -> None:
            self._history = list(original_history)
            self.initialized = False

        async def _initialize(self) -> None:
            self.initialized = True

        def _ensure_client(self):
            return object()

        def _resolve_model(self) -> str:
            return "claude-sonnet-4-5"

    agent = FakeAgent()

    async def fake_compact_conversation(client, model, messages, state):
        assert client is not None
        assert model == "claude-sonnet-4-5"
        assert messages == original_history
        return {
            "compacted_messages": compacted_history,
            "summary": "summary",
            "state": {"compacted": True},
        }

    monkeypatch.setattr(runtime, "compact_conversation", fake_compact_conversation, raising=False)
    monkeypatch.setattr(runtime, "create_auto_compact_state", lambda: {"compacted": False}, raising=False)
    monkeypatch.setattr(
        runtime,
        "estimate_messages_tokens",
        lambda messages: 1200 if messages == original_history else 240,
        raising=False,
    )

    result = asyncio.run(runtime.compact_current_session(agent))

    assert agent.initialized is True
    assert agent._history == compacted_history
    assert result == {
        "compacted": True,
        "summary": "summary",
        "before_tokens": 1200,
        "after_tokens": 240,
        "reason": "",
    }


def test_compact_current_session_skips_small_history(monkeypatch) -> None:
    class FakeAgent:
        def __init__(self) -> None:
            self._history = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]

        async def _initialize(self) -> None:
            return None

    agent = FakeAgent()
    monkeypatch.setattr(runtime, "estimate_messages_tokens", lambda messages: 42, raising=False)

    result = asyncio.run(runtime.compact_current_session(agent))

    assert result == {
        "compacted": False,
        "summary": "",
        "before_tokens": 42,
        "after_tokens": 42,
        "reason": "Need at least two messages before compaction.",
    }


def test_compact_current_session_omits_private_history_blocks(monkeypatch) -> None:
    original_history = [
        {"role": "system", "content": [{"type": "text", "text": "hidden system prompt"}]},
        {"role": "user", "content": [{"type": "text", "text": "hello"}]},
        {"role": "assistant", "content": [{"type": "tool_result", "content": "SECRET=value"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
    ]

    class FakeAgent:
        def __init__(self) -> None:
            self._history = list(original_history)

        async def _initialize(self) -> None:
            return None

        def _ensure_client(self):
            return object()

        def _resolve_model(self) -> str:
            return "claude-sonnet-4-5"

    agent = FakeAgent()

    async def fake_compact_conversation(client, model, messages, state):
        assert messages == [
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
        ]
        return {
            "compacted_messages": messages,
            "summary": "visible summary only",
            "state": type("State", (), {"consecutive_failures": 0})(),
        }

    monkeypatch.setattr(runtime, "compact_conversation", fake_compact_conversation, raising=False)
    monkeypatch.setattr(runtime, "create_auto_compact_state", lambda: type("State", (), {"consecutive_failures": 0})(), raising=False)
    monkeypatch.setattr(runtime, "estimate_messages_tokens", lambda messages: len(messages), raising=False)

    result = asyncio.run(runtime.compact_current_session(agent))

    assert result["summary"] == "visible summary only"


def test_compact_current_session_raises_on_sdk_compaction_failure(monkeypatch) -> None:
    original_history = [
        {"role": "user", "content": [{"type": "text", "text": "hello"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
    ]

    class FakeAgent:
        def __init__(self) -> None:
            self._history = list(original_history)

        async def _initialize(self) -> None:
            return None

        def _ensure_client(self):
            return object()

        def _resolve_model(self) -> str:
            return "claude-sonnet-4-5"

    agent = FakeAgent()

    async def fake_compact_conversation(client, model, messages, state):
        return {
            "compacted_messages": messages,
            "summary": "",
            "state": type("State", (), {"consecutive_failures": 1})(),
        }

    monkeypatch.setattr(runtime, "compact_conversation", fake_compact_conversation, raising=False)
    monkeypatch.setattr(runtime, "create_auto_compact_state", lambda: type("State", (), {"consecutive_failures": 0})(), raising=False)
    monkeypatch.setattr(runtime, "estimate_messages_tokens", lambda messages: 42, raising=False)

    with pytest.raises(RuntimeError, match="Compaction failed"):
        asyncio.run(runtime.compact_current_session(agent))


def test_compact_current_session_rejects_non_anthropic_backends(monkeypatch) -> None:
    class FakeAgent:
        def __init__(self) -> None:
            self._history = [
                {"role": "user", "content": [{"type": "text", "text": "hello"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
            ]

        async def _initialize(self) -> None:
            return None

        def get_api_type(self) -> str:
            return "openai-completions"

        def _ensure_client(self):
            raise AssertionError("should not create client")

        def _resolve_model(self) -> str:
            return "gpt-4o-mini"

    monkeypatch.setattr(runtime, "estimate_messages_tokens", lambda messages: 42, raising=False)

    with pytest.raises(RuntimeError, match="requires an anthropic-messages backend"):
        asyncio.run(runtime.compact_current_session(FakeAgent()))
