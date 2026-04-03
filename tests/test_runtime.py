from cock_code.config import RuntimeConfig
import asyncio

from cock_code.runtime import build_agent_options, create_runtime_agent


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
