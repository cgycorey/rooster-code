import asyncio
from pathlib import Path
from typing import Coroutine, cast

import cock_code.runtime as runtime
import pytest
from open_agent_sdk import SDKMessage, SDKMessageType, ToolContext, ToolResult
from open_agent_sdk.providers import CreateMessageResponse
from open_agent_sdk.skills import clear_skills, get_skill
from cock_code.config import RuntimeConfig

from cock_code.runtime import build_agent_options, create_runtime_agent
from cock_code.runtime import enforce_session_retention, find_requested_agent_name


@pytest.fixture(autouse=True)
def reset_sdk_skills():
    clear_skills()
    import open_agent_sdk.skills.bundled as bundled_mod

    bundled_mod._initialized = False
    yield
    clear_skills()
    bundled_mod._initialized = False


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
        skills_dir="/tmp/skills",
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


def test_build_agent_options_includes_bundled_and_local_skills(tmp_path: Path) -> None:
    skill_dir = tmp_path / "skills" / "explain"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: explain
description: Explain a concept simply.
when_to_use: When the user wants a simple explanation.
---

Explain the user request clearly and simply.
""",
        encoding="utf-8",
    )

    options = build_agent_options(
        RuntimeConfig(
            api_key="a",
            base_url="https://example.test",
            model="m",
            skills_dir=str(tmp_path / "skills"),
        )
    )

    assert "- commit:" in options.append_system_prompt
    assert "- explain:" in options.append_system_prompt


def test_list_skill_names_includes_bundled_and_local_skills(tmp_path: Path) -> None:
    skill_dir = tmp_path / "skills" / "explain"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: explain
description: Explain a concept simply.
---

Explain the user request clearly and simply.
""",
        encoding="utf-8",
    )

    build_agent_options(
        RuntimeConfig(
            api_key="a",
            base_url="https://example.test",
            model="m",
            skills_dir=str(tmp_path / "skills"),
        )
    )

    names = runtime.list_skill_names()

    assert "commit" in names
    assert "explain" in names


def test_repo_plan_skill_loads_and_has_detailed_generic_prompt() -> None:
    repo_skills_dir = Path(__file__).resolve().parents[1] / "skills"

    build_agent_options(
        RuntimeConfig(
            api_key="a",
            base_url="https://example.test",
            model="m",
            skills_dir=str(repo_skills_dir),
        )
    )

    assert "plan" in runtime.list_skill_names()

    skill = get_skill("plan")
    assert skill is not None
    assert skill.description
    assert skill.get_prompt is not None
    prompt_fn = cast(Coroutine[object, object, list[dict[str, object]]], skill.get_prompt("add auth support", ToolContext()))
    blocks = asyncio.run(prompt_fn)
    text = "\n".join(str(block["text"]) for block in blocks if block.get("type") == "text")

    assert "Goal" in text
    assert "Files to inspect" in text
    assert "Files likely to change" in text
    assert "Implementation plan" in text
    assert "Tests to add or run" in text
    assert "Verification commands" in text
    assert "Do not implement yet" in text
    assert "Do not save the plan unless the user explicitly asks" in text
    assert "User request: add auth support" in text


def test_stream_skill_events_runs_inline_skills_on_current_agent(monkeypatch) -> None:
    class FakeAgent:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, object] | None]] = []

        async def query(self, prompt: str, overrides: dict[str, object] | None = None):
            self.calls.append((prompt, overrides))
            yield SDKMessage(type=SDKMessageType.SYSTEM, text="inline-start")
            yield SDKMessage(type=SDKMessageType.RESULT, text="inline-result")

    async def fake_skill_call(self, input, context):
        return ToolResult(
            tool_use_id="",
            content='{"success": true, "commandName": "plan", "status": "inline", "prompt": "Goal\\n- test", "allowedTools": ["Read"], "model": "m-inline"}',
        )

    monkeypatch.setattr(runtime.SkillTool, "call", fake_skill_call)

    agent = FakeAgent()

    async def collect_events():
        return [
            event
            async for event in runtime.stream_skill_events(
                RuntimeConfig(api_key="x", base_url="https://example.test", model="m", skills_dir="skills"),
                agent,
                "plan",
                "add auth support",
            )
        ]

    events = asyncio.run(collect_events())

    assert [event.type for event in events] == [SDKMessageType.SYSTEM, SDKMessageType.SYSTEM, SDKMessageType.RESULT]
    assert agent.calls == [("Goal\n- test", {"model": "m-inline", "allowed_tools": ["Read"]})]


def test_stream_skill_events_forks_child_agent_for_forked_skills(monkeypatch) -> None:
    class FakeChildAgent:
        async def query(self, prompt: str, overrides: dict[str, object] | None = None):
            yield SDKMessage(type=SDKMessageType.SYSTEM, text="fork-start")
            yield SDKMessage(type=SDKMessageType.RESULT, text="fork-result")

        async def close(self) -> None:
            return None

    class FakeParentAgent:
        async def query(self, prompt: str, overrides: dict[str, object] | None = None):
            raise AssertionError("forked skill should not use parent agent")
            yield

    async def fake_skill_call(self, input, context):
        return ToolResult(
            tool_use_id="",
            content='{"success": true, "commandName": "review", "status": "forked", "prompt": "Review changes", "model": "m-fork"}',
        )

    created: list[RuntimeConfig] = []

    monkeypatch.setattr(runtime.SkillTool, "call", fake_skill_call)
    monkeypatch.setattr(runtime, "_create_sdk_agent", lambda config, include_runtime_agent_tool=True, system_prompt="": created.append(config) or FakeChildAgent())

    async def collect_events():
        return [
            event
            async for event in runtime.stream_skill_events(
                RuntimeConfig(api_key="x", base_url="https://example.test", model="m", skills_dir="skills"),
                FakeParentAgent(),
                "review",
                "check code",
            )
        ]

    events = asyncio.run(collect_events())

    assert [event.type for event in events] == [SDKMessageType.SYSTEM, SDKMessageType.SYSTEM, SDKMessageType.RESULT]
    assert created[0].model == "m-fork"


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


def test_run_subagent_returns_rich_plain_text_summary(monkeypatch) -> None:
    class FakeChildAgent:
        async def prompt(self, prompt: str):
            from open_agent_sdk.types import QueryResult

            return QueryResult(
                text="implemented the change",
                messages=[
                    {"role": "user", "content": [{"type": "text", "text": "Add feature X"}]},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": "Outcome: implemented the change\nFiles: src/a.py, tests/test_a.py\nCommands: pytest tests/test_a.py -q\nOpen issues: none\nNext step: run full suite",
                            }
                        ],
                    },
                ],
            )

        async def close(self) -> None:
            return None

    monkeypatch.setattr(runtime, "_create_sdk_agent", lambda config, include_runtime_agent_tool=False, system_prompt="": FakeChildAgent())

    result = asyncio.run(
        runtime._run_subagent(
            RuntimeConfig(model="m1", agents={"builder": {"description": "build agent"}}),
            {"name": "builder", "prompt": "do work", "description": "builder"},
            ToolContext(cwd="/tmp/project", env={}),
        )
    )

    text = str(result.content)
    assert "Outcome:" in text
    assert "Outcome: implemented the change" in text
    assert "Files:" in text
    assert "src/a.py" in text
    assert "Commands:" in text
    assert "pytest tests/test_a.py -q" in text
    assert "Open issues:" in text
    assert "Next step:" in text


def test_run_subagent_summary_excludes_user_text_and_unlabeled_assistant_text(monkeypatch) -> None:
    class FakeChildAgent:
        async def prompt(self, prompt: str):
            from open_agent_sdk.types import QueryResult

            return QueryResult(
                text="implemented the change",
                messages=[
                    {"role": "user", "content": [{"type": "text", "text": "SECRET user request"}]},
                    {"role": "assistant", "content": [{"type": "text", "text": "intermediate reasoning that should not leak"}]},
                    {"role": "assistant", "content": [{"type": "text", "text": "Files: src/a.py"}]},
                    {"role": "assistant", "content": [{"type": "text", "text": "Commands: pytest tests/test_a.py -q"}]},
                    {"role": "assistant", "content": [{"type": "text", "text": "Open issues: none"}]},
                    {"role": "assistant", "content": [{"type": "text", "text": "Next step: run full suite"}]},
                ],
            )

        async def close(self) -> None:
            return None

    monkeypatch.setattr(runtime, "_create_sdk_agent", lambda config, include_runtime_agent_tool=False, system_prompt="": FakeChildAgent())

    result = asyncio.run(
        runtime._run_subagent(
            RuntimeConfig(model="m1", agents={"builder": {"description": "build agent"}}),
            {"name": "builder", "prompt": "do work", "description": "builder"},
            ToolContext(cwd="/tmp/project", env={}),
        )
    )

    text = str(result.content)
    assert "SECRET user request" not in text
    assert "intermediate reasoning that should not leak" not in text
    assert "Files: src/a.py" in text
    assert "Commands: pytest tests/test_a.py -q" in text


def test_run_subagent_summary_falls_back_to_final_text_for_outcome(monkeypatch) -> None:
    class FakeChildAgent:
        async def prompt(self, prompt: str):
            from open_agent_sdk.types import QueryResult

            return QueryResult(
                text="explored the recent changes and identified the likely cause",
                messages=[
                    {"role": "user", "content": [{"type": "text", "text": "use subagent to explore changes made"}]},
                    {"role": "assistant", "content": [{"type": "text", "text": "explored the recent changes and identified the likely cause"}]},
                ],
            )

        async def close(self) -> None:
            return None

    monkeypatch.setattr(runtime, "_create_sdk_agent", lambda config, include_runtime_agent_tool=False, system_prompt="": FakeChildAgent())

    result = asyncio.run(
        runtime._run_subagent(
            RuntimeConfig(model="m1", agents={"task": {"description": "explore agent"}}),
            {"name": "task", "prompt": "explore changes", "description": "task"},
            ToolContext(cwd="/tmp/project", env={}),
        )
    )

    text = str(result.content)
    assert "Outcome: explored the recent changes and identified the likely cause" in text
    assert "Files: None" in text
    assert "Commands: None" in text


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


def test_create_runtime_agent_attaches_activity_trace_to_tool_result(monkeypatch, tmp_path: Path) -> None:
    class ReadTool:
        name = "Read"

        async def call(self, input, context):
            return ToolResult(tool_use_id="", content="file body")

    class FakeAgent:
        def __init__(self) -> None:
            self._client = None
            self._tool_pool = []

        async def _initialize(self) -> None:
            self._tool_pool = [ReadTool()]

        async def query(self, prompt, overrides=None):
            await self._tool_pool[0].call({"file_path": "sample.txt"}, ToolContext(cwd=str(tmp_path), env={}))
            yield SDKMessage(type=SDKMessageType.TOOL_RESULT, tool_name="Read", result_content="file body")

    monkeypatch.setattr("cock_code.runtime.create_agent", lambda options: FakeAgent())

    agent = create_runtime_agent(RuntimeConfig(api_key="test", base_url="https://nano-gpt.com/api/v1", model="m1"))
    (tmp_path / "sample.txt").write_text("hello\n", encoding="utf-8")
    asyncio.run(agent._initialize())

    async def collect_events():
        return [event async for event in agent.query("read sample")]

    events = asyncio.run(collect_events())

    assert events[0].system_data["activity_trace"] == [
        {"action": "Reading file", "tool": "Read", "target": str(tmp_path / "sample.txt")}
    ]


def test_create_runtime_agent_emits_activity_system_event_before_tool_result(monkeypatch, tmp_path: Path) -> None:
    class ReadTool:
        name = "Read"

        async def call(self, input, context):
            await asyncio.sleep(0)
            return ToolResult(tool_use_id="", content="file body")

    class FakeAgent:
        def __init__(self) -> None:
            self._client = None
            self._tool_pool = []

        async def _initialize(self) -> None:
            self._tool_pool = [ReadTool()]

        async def query(self, prompt, overrides=None):
            await self._tool_pool[0].call({"file_path": "sample.txt"}, ToolContext(cwd=str(tmp_path), env={}))
            yield SDKMessage(type=SDKMessageType.TOOL_RESULT, tool_name="Read", result_content="file body")

    monkeypatch.setattr("cock_code.runtime.create_agent", lambda options: FakeAgent())

    agent = create_runtime_agent(RuntimeConfig(api_key="test", base_url="https://nano-gpt.com/api/v1", model="m1"))
    (tmp_path / "sample.txt").write_text("hello\n", encoding="utf-8")
    asyncio.run(agent._initialize())

    async def collect_events():
        return [event async for event in agent.query("read sample")]

    events = asyncio.run(collect_events())

    assert [event.type for event in events] == [SDKMessageType.SYSTEM, SDKMessageType.TOOL_RESULT]
    assert events[0].system_data["activity_trace"] == [
        {"action": "Reading file", "tool": "Read", "target": str(tmp_path / "sample.txt")}
    ]


def test_create_runtime_agent_emits_edit_activity_system_event_before_tool_result(monkeypatch, tmp_path: Path) -> None:
    class ReadTool:
        name = "Read"

        async def call(self, input, context):
            return ToolResult(tool_use_id="", content="file body")

    class EditTool:
        name = "Edit"

        async def call(self, input, context):
            target = tmp_path / "sample.txt"
            target.write_text("new\n", encoding="utf-8")
            return ToolResult(tool_use_id="", content=f"Successfully edited {target}")

    class FakeAgent:
        def __init__(self) -> None:
            self._client = None
            self._tool_pool = []

        async def _initialize(self) -> None:
            self._tool_pool = [ReadTool(), EditTool()]

        async def query(self, prompt, overrides=None):
            await self._tool_pool[0].call({"file_path": "sample.txt"}, ToolContext(cwd=str(tmp_path), env={}))
            await self._tool_pool[1].call(
                {"file_path": "sample.txt", "old_string": "old", "new_string": "new"},
                ToolContext(cwd=str(tmp_path), env={}),
            )
            yield SDKMessage(type=SDKMessageType.TOOL_RESULT, tool_name="Edit", result_content="done")

    monkeypatch.setattr("cock_code.runtime.create_agent", lambda options: FakeAgent())

    agent = create_runtime_agent(RuntimeConfig(api_key="test", base_url="https://nano-gpt.com/api/v1", model="m1"))
    (tmp_path / "sample.txt").write_text("old\n", encoding="utf-8")
    asyncio.run(agent._initialize())

    async def collect_events():
        return [event async for event in agent.query("edit sample")]

    events = asyncio.run(collect_events())

    assert [event.type for event in events] == [SDKMessageType.SYSTEM, SDKMessageType.SYSTEM, SDKMessageType.TOOL_RESULT]
    assert events[0].system_data["activity_trace"] == [
        {"action": "Reading file", "tool": "Read", "target": str(tmp_path / "sample.txt")}
    ]
    assert events[1].system_data["activity_trace"] == [
        {"action": "Editing file", "tool": "Edit", "target": str(tmp_path / "sample.txt")}
    ]


def test_create_runtime_agent_emits_generic_tool_activity_before_tool_result(monkeypatch) -> None:
    class BashTool:
        name = "Bash"

        async def call(self, input, context):
            return ToolResult(tool_use_id="", content="done")

    class FakeAgent:
        def __init__(self) -> None:
            self._client = None
            self._tool_pool = []

        async def _initialize(self) -> None:
            self._tool_pool = [BashTool()]

        async def query(self, prompt, overrides=None):
            await self._tool_pool[0].call({"command": "pytest -q"}, ToolContext(cwd="/tmp/project", env={}))
            yield SDKMessage(type=SDKMessageType.TOOL_RESULT, tool_name="Bash", result_content="done")

    monkeypatch.setattr("cock_code.runtime.create_agent", lambda options: FakeAgent())

    agent = create_runtime_agent(RuntimeConfig(api_key="test", base_url="https://nano-gpt.com/api/v1", model="m1"))
    asyncio.run(agent._initialize())

    async def collect_events():
        return [event async for event in agent.query("run tests")]

    events = asyncio.run(collect_events())

    assert [event.type for event in events] == [SDKMessageType.SYSTEM, SDKMessageType.TOOL_RESULT]
    assert events[0].system_data["activity_trace"] == [
        {"action": "Running tool", "tool": "Bash", "target": "pytest -q"}
    ]


def test_create_runtime_agent_preserves_query_exceptions(monkeypatch) -> None:
    class FakeAgent:
        def __init__(self) -> None:
            self._client = None
            self._tool_pool = []

        async def _initialize(self) -> None:
            return None

        async def query(self, prompt, overrides=None):
            raise RuntimeError("boom")
            yield

    monkeypatch.setattr("cock_code.runtime.create_agent", lambda options: FakeAgent())

    agent = create_runtime_agent(RuntimeConfig(api_key="test", base_url="https://nano-gpt.com/api/v1", model="m1"))

    async def collect_events():
        return [event async for event in agent.query("hello")]

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(collect_events())


def test_stream_named_agent_events_emits_child_tool_activity(monkeypatch, tmp_path: Path) -> None:
    class ReadTool:
        name = "Read"

        async def call(self, input, context):
            return ToolResult(tool_use_id="", content="file body")

    class FakeChildAgent:
        def __init__(self) -> None:
            self._client = None
            self._tool_pool = []

        async def _initialize(self) -> None:
            self._tool_pool = [ReadTool()]

        async def query(self, prompt, overrides=None):
            if not self._tool_pool:
                await self._initialize()
            await self._tool_pool[0].call({"file_path": "child.txt"}, ToolContext(cwd=str(tmp_path), env={}))
            yield SDKMessage(type=SDKMessageType.TOOL_RESULT, tool_name="Read", result_content="file body")

        async def close(self) -> None:
            return None

    monkeypatch.setattr("cock_code.runtime.create_agent", lambda options: FakeChildAgent())

    async def collect_events():
        return [
            event
            async for event in runtime.stream_named_agent_events(
                RuntimeConfig(
                    api_key="test",
                    base_url="https://nano-gpt.com/api/v1",
                    model="m1",
                    agents={"reviewer": {"description": "reviewer"}},
                    cwd=str(tmp_path),
                ),
                "reviewer",
                "read the child file",
            )
        ]

    events = asyncio.run(collect_events())

    assert [event.type for event in events] == [SDKMessageType.SYSTEM, SDKMessageType.TOOL_RESULT]
    assert events[0].system_data["activity_trace"] == [
        {"action": "Reading file", "tool": "Read", "target": str(tmp_path / "child.txt")}
    ]


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

        def _ensure_provider(self):
            return provider

        def _resolve_model(self) -> str:
            return "claude-sonnet-4-5"

    class FakeProvider:
        def __init__(self) -> None:
            self.params = None

        async def create_message(self, params):
            self.params = params
            return CreateMessageResponse(
                content=[{"type": "text", "text": "summary"}],
                stop_reason="end_turn",
            )

    provider = FakeProvider()
    agent = FakeAgent()
    monkeypatch.setattr(
        runtime,
        "estimate_messages_tokens",
        lambda messages: 1200 if messages == original_history else 240,
        raising=False,
    )

    result = asyncio.run(runtime.compact_current_session(agent))

    assert agent.initialized is True
    assert provider.params is not None
    assert provider.params.model == "claude-sonnet-4-5"
    assert agent._history == compacted_history
    assert result == {
        "compacted": True,
        "summary": "summary",
        "before_tokens": 1200,
        "after_tokens": 240,
        "reason": "",
    }


def test_build_manual_compaction_summary_prompt_uses_structured_handoff() -> None:
    prompt = runtime._build_manual_compaction_summary_prompt([
        {"role": "user", "content": [{"type": "text", "text": "Please add OpenAI-compatible /compact support."}]},
        {"role": "assistant", "content": [{"type": "text", "text": "I switched manual compaction to the provider abstraction."}]},
    ])

    assert "Summarize this session for immediate continuation" in prompt
    assert "## Goal" in prompt
    assert "## Current State" in prompt
    assert "## Key Decisions" in prompt
    assert "## Code/Files" in prompt
    assert "## Constraints / What to Avoid" in prompt
    assert "## Blockers / Open Questions" in prompt
    assert "## Next Step" in prompt
    assert "user: Please add OpenAI-compatible /compact support." in prompt
    assert "assistant: I switched manual compaction to the provider abstraction." in prompt


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

        def _ensure_provider(self):
            return provider

        def _resolve_model(self) -> str:
            return "claude-sonnet-4-5"

    class FakeProvider:
        def __init__(self) -> None:
            self.params = None

        async def create_message(self, params):
            self.params = params
            return CreateMessageResponse(
                content=[{"type": "text", "text": "visible summary only"}],
                stop_reason="end_turn",
            )

    provider = FakeProvider()
    agent = FakeAgent()
    monkeypatch.setattr(runtime, "estimate_messages_tokens", lambda messages: len(messages), raising=False)

    result = asyncio.run(runtime.compact_current_session(agent))

    assert provider.params is not None
    assert provider.params.messages[0]["role"] == "user"
    prompt = provider.params.messages[0]["content"]
    assert "## Goal" in prompt
    assert "## Current State" in prompt
    assert "## Key Decisions" in prompt
    assert "## Transcript" in prompt
    assert "user: hello" in prompt
    assert "assistant: hi" in prompt
    assert result["summary"] == "visible summary only"


def test_compact_current_session_raises_provider_error(monkeypatch) -> None:
    original_history = [
        {"role": "user", "content": [{"type": "text", "text": "hello"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
    ]

    class FakeAgent:
        def __init__(self) -> None:
            self._history = list(original_history)

        async def _initialize(self) -> None:
            return None

        def _ensure_provider(self):
            return provider

        def _resolve_model(self) -> str:
            return "claude-sonnet-4-5"

    class FakeProvider:
        async def create_message(self, params):
            raise RuntimeError("provider unavailable")

    provider = FakeProvider()
    agent = FakeAgent()
    monkeypatch.setattr(runtime, "estimate_messages_tokens", lambda messages: 42, raising=False)

    with pytest.raises(RuntimeError, match="provider unavailable"):
        asyncio.run(runtime.compact_current_session(agent))


def test_compact_current_session_requires_provider_message_creation(monkeypatch) -> None:
    class FakeAgent:
        def __init__(self) -> None:
            self._history = [
                {"role": "user", "content": [{"type": "text", "text": "hello"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
            ]

        async def _initialize(self) -> None:
            return None

        def _ensure_provider(self):
            return object()

        def _resolve_model(self) -> str:
            return "gpt-4o-mini"

    monkeypatch.setattr(runtime, "estimate_messages_tokens", lambda messages: 42, raising=False)

    with pytest.raises(RuntimeError, match="provider does not support message creation"):
        asyncio.run(runtime.compact_current_session(FakeAgent()))


def test_compact_current_session_supports_openai_compatible_provider(monkeypatch) -> None:
    original_history = [
        {"role": "user", "content": [{"type": "text", "text": "hello"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
    ]

    class FakeProvider:
        def __init__(self) -> None:
            self.params = None

        async def create_message(self, params):
            self.params = params
            return CreateMessageResponse(
                content=[{"type": "text", "text": "summary from openai"}],
                stop_reason="end_turn",
            )

    provider = FakeProvider()

    class FakeAgent:
        def __init__(self) -> None:
            self._history = list(original_history)

        async def _initialize(self) -> None:
            return None

        def get_api_type(self) -> str:
            return "openai-completions"

        def _ensure_provider(self):
            return provider

        def _ensure_client(self):
            raise AssertionError("openai path should not use _ensure_client")

        def _resolve_model(self) -> str:
            return "gpt-4o-mini"

    monkeypatch.setattr(runtime, "estimate_messages_tokens", lambda messages: 1200 if messages == original_history else 240, raising=False)

    result = asyncio.run(runtime.compact_current_session(FakeAgent()))

    assert provider.params is not None
    assert provider.params.model == "gpt-4o-mini"
    assert provider.params.messages[0]["role"] == "user"
    prompt = provider.params.messages[0]["content"]
    assert "## Goal" in prompt
    assert "## Code/Files" in prompt
    assert "## Next Step" in prompt
    assert "user: hello" in prompt
    assert "assistant: hi" in prompt
    assert result == {
        "compacted": True,
        "summary": "summary from openai",
        "before_tokens": 1200,
        "after_tokens": 240,
        "reason": "",
    }


def test_compact_current_session_rejects_empty_provider_summary(monkeypatch) -> None:
    original_history = [
        {"role": "user", "content": [{"type": "text", "text": "hello"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
    ]

    class FakeProvider:
        async def create_message(self, params):
            return CreateMessageResponse(
                content=[{"type": "text", "text": "   "}],
                stop_reason="end_turn",
            )

    class FakeAgent:
        def __init__(self) -> None:
            self._history = list(original_history)

        async def _initialize(self) -> None:
            return None

        def _ensure_provider(self):
            return FakeProvider()

        def _resolve_model(self) -> str:
            return "gpt-4o-mini"

    monkeypatch.setattr(runtime, "estimate_messages_tokens", lambda messages: 42, raising=False)

    agent = FakeAgent()
    with pytest.raises(RuntimeError, match="Compaction produced an empty summary"):
        asyncio.run(runtime.compact_current_session(agent))

    assert agent._history == original_history
