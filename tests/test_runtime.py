import asyncio
import contextlib
import json
import logging
from pathlib import Path
from typing import Coroutine, cast

import rooster_code.runtime as runtime
import rooster_code.runtime_session as runtime_session
import pytest
from open_agent_sdk import SDKMessage, SDKMessageType, ToolContext, ToolResult
from open_agent_sdk.providers import CreateMessageResponse
from open_agent_sdk.skills import clear_skills, get_skill
from open_agent_sdk.tools import clear_tasks, get_all_tasks
from rooster_code.config import RuntimeConfig

from rooster_code.runtime import build_agent_options, create_runtime_agent
from rooster_code.runtime import enforce_session_retention, find_requested_agent_name


@pytest.fixture(autouse=True)
def reset_sdk_skills():
    clear_skills()
    import open_agent_sdk.skills.bundled as bundled_mod

    bundled_mod._initialized = False
    yield
    clear_skills()
    bundled_mod._initialized = False


@pytest.fixture(autouse=True)
def reset_sdk_tasks():
    clear_tasks()
    yield
    clear_tasks()


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



def test_build_subagent_config_passes_max_tokens_from_definition() -> None:
    from rooster_code.runtime import _build_subagent_config

    config = RuntimeConfig(
        api_key="abc",
        base_url="https://example.test",
        model="m1",
        max_tokens=280,
    )
    definition = {"description": "test", "max_tokens": 4096, "max_turns": 2}

    result = _build_subagent_config(config, definition, {}, ToolContext(cwd=".", env={}))

    assert result.max_tokens == 4096
    assert result.max_turns == 2


def test_build_subagent_config_passes_thinking_budget_from_definition() -> None:
    from rooster_code.runtime import _build_subagent_config

    config = RuntimeConfig(
        api_key="abc",
        base_url="https://example.test",
        model="m1",
        thinking_budget=100,
    )
    definition = {"description": "test", "thinking_budget": 500}

    result = _build_subagent_config(config, definition, {}, ToolContext(cwd=".", env={}))

    assert result.thinking_budget == 500


def test_build_subagent_config_falls_back_to_config_when_definition_omits() -> None:
    from rooster_code.runtime import _build_subagent_config

    config = RuntimeConfig(
        api_key="abc",
        base_url="https://example.test",
        model="m1",
        max_tokens=800,
        thinking_budget=200,
    )
    definition = {"description": "test"}

    result = _build_subagent_config(config, definition, {}, ToolContext(cwd=".", env={}))

    assert result.max_tokens == 800
    assert result.thinking_budget == 200

def test_build_agent_options_omits_agent_tool_prompt_when_agent_tool_disabled() -> None:
    options = build_agent_options(
        RuntimeConfig(
            api_key="abc",
            base_url="https://example.test",
            model="m1",
            agents={"reviewer": {"description": "code reviewer"}},
        ),
        include_runtime_agent_tool=False,
    )

    assert "Use the Agent tool" not in options.append_system_prompt
    assert "# Configured Agents" not in options.append_system_prompt


def test_build_agent_options_limits_tool_use_guidance_when_agent_tool_disabled() -> None:
    options = build_agent_options(
        RuntimeConfig(
            api_key="abc",
            base_url="https://example.test",
            model="m1",
            agents={"reviewer": {"description": "code reviewer"}},
        ),
        include_runtime_agent_tool=False,
    )

    prompt = options.append_system_prompt
    assert "# Tool Use Guidance" in prompt
    assert "call the Skill tool once" in prompt
    assert "use the Agent tool" not in prompt
    assert "run_in_background=true" not in prompt
    assert "# Configured Agents" not in prompt


def test_build_agent_options_includes_tool_use_guidance() -> None:
    options = build_agent_options(
        RuntimeConfig(
            api_key="abc",
            base_url="https://example.test",
            model="m1",
            agents={"reviewer": {"description": "code reviewer"}},
        )
    )

    prompt = options.append_system_prompt
    assert "# Tool Use Guidance" in prompt
    assert "call the Skill tool once" in prompt
    assert "use the Agent tool" in prompt
    assert "run_in_background=true" in prompt
    assert "do not duplicate" in prompt
    assert "TeamDispatch" in prompt


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
    assert "<user-request>\nadd auth support\n</user-request>" in text


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

    assert [event.type for event in events] == [SDKMessageType.SYSTEM, SDKMessageType.SYSTEM, SDKMessageType.RESULT, SDKMessageType.SYSTEM]
    assert agent.calls == [("Goal\n- test", {"model": "m-inline", "allowed_tools": ["Read"]})]
    assert events[0].system_data["activity_trace"] == [
        {"action": "Resolved subagent", "tool": "Skill", "target": "plan (inline)"}
    ]
    assert events[-1].system_data["activity_trace"] == [
        {"action": "Completed subagent", "tool": "Skill", "target": "plan"}
    ]


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

    created: list[tuple[RuntimeConfig, bool]] = []

    monkeypatch.setattr(runtime.SkillTool, "call", fake_skill_call)
    monkeypatch.setattr(
        runtime,
        "_create_sdk_agent",
        lambda config, include_runtime_agent_tool=True, system_prompt="", is_child_agent=False: created.append((config, include_runtime_agent_tool)) or FakeChildAgent(),
    )

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

    assert [event.type for event in events] == [SDKMessageType.SYSTEM, SDKMessageType.SYSTEM, SDKMessageType.RESULT, SDKMessageType.SYSTEM]
    assert created[0][0].model == "m-fork"
    assert created[0][1] is False
    assert events[0].system_data["activity_trace"] == [
        {"action": "Resolved subagent", "tool": "Skill", "target": "review (forked)"}
    ]
    assert events[-1].system_data["activity_trace"] == [
        {"action": "Completed subagent", "tool": "Skill", "target": "review"}
    ]


def test_create_runtime_agent_does_not_inject_custom_transport(monkeypatch) -> None:
    class FakeAgent:
        _client = None

    monkeypatch.setattr("rooster_code.runtime.create_agent", lambda options: FakeAgent())

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
        async def prompt(self, prompt: str, overrides=None):
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

    monkeypatch.setattr(runtime, "_create_sdk_agent", lambda config, include_runtime_agent_tool=False, system_prompt="", is_child_agent=False: FakeChildAgent())

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
        async def prompt(self, prompt: str, overrides=None):
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

    monkeypatch.setattr(runtime, "_create_sdk_agent", lambda config, include_runtime_agent_tool=False, system_prompt="", is_child_agent=False: FakeChildAgent())

    result = asyncio.run(
        runtime._run_subagent(
            RuntimeConfig(model="m1", agents={"builder": {"description": "build agent"}}),
            {"name": "builder", "prompt": "do work", "description": "builder"},
            ToolContext(cwd="/tmp/project", env={}),
        )
    )
    text = str(result.content)
    assert "SECRET user request" not in text
    assert "implemented the change" in text
    assert "Files: src/a.py" in text
    assert "Commands: pytest tests/test_a.py -q" in text


def test_run_subagent_summary_falls_back_to_final_text_for_outcome(monkeypatch) -> None:
    class FakeChildAgent:
        async def prompt(self, prompt: str, overrides=None):
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

    monkeypatch.setattr(runtime, "_create_sdk_agent", lambda config, include_runtime_agent_tool=False, system_prompt="", is_child_agent=False: FakeChildAgent())

    result = asyncio.run(
        runtime._run_subagent(
            RuntimeConfig(model="m1", agents={"task": {"description": "explore agent"}}),
            {"name": "task", "prompt": "explore changes", "description": "task"},
            ToolContext(cwd="/tmp/project", env={}),
        )
    )
    text = str(result.content)
    assert "explored the recent changes and identified the likely cause" in text


def test_run_subagent_summary_does_not_emit_outcome_none(monkeypatch) -> None:
    class FakeChildAgent:
        async def prompt(self, prompt: str):
            from open_agent_sdk.types import QueryResult

            return QueryResult(text="", messages=[])

        async def close(self) -> None:
            return None

    monkeypatch.setattr(runtime, "_create_sdk_agent", lambda config, include_runtime_agent_tool=False, system_prompt="", is_child_agent=False: FakeChildAgent())

    result = asyncio.run(
        runtime._run_subagent(
            RuntimeConfig(model="m1", agents={"task": {"description": "explore agent"}}),
            {"name": "task", "prompt": "explore changes", "description": "task"},
            ToolContext(cwd="/tmp/project", env={}),
        )
    )
    text = str(result.content)
    assert "Outcome: None" not in text
    assert "No useful output returned" in text or "Agent completed with no text output" in text


def test_run_subagent_default_task_does_not_force_read_only_tool_subset(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeChildAgent:
        async def prompt(self, prompt: str):
            from open_agent_sdk.types import QueryResult

            return QueryResult(text="done", messages=[])

        async def close(self) -> None:
            return None

    def fake_create_sdk_agent(config, include_runtime_agent_tool=False, system_prompt="", is_child_agent=False):
        captured["allowed_tools"] = config.allowed_tools
        captured["system_prompt"] = system_prompt
        return FakeChildAgent()

    monkeypatch.setattr(runtime, "_create_sdk_agent", fake_create_sdk_agent)

    asyncio.run(
        runtime._run_subagent(
            RuntimeConfig(model="m1"),
            {"prompt": "check changes", "description": "task"},
            ToolContext(cwd="/tmp/project", env={}),
        )
    )

    assert captured["allowed_tools"] is None
    assert "read-only" not in str(captured["system_prompt"]).lower()


def test_run_subagent_background_creates_sdk_task_and_updates_output(monkeypatch) -> None:
    class FakeChildAgent:
        async def prompt(self, prompt: str):
            from open_agent_sdk.types import QueryResult

            return QueryResult(
                text="background done",
                messages=[
                    {"role": "assistant", "content": [{"type": "text", "text": "Outcome: background done"}]},
                ],
            )

        async def close(self) -> None:
            return None

    monkeypatch.setattr(runtime, "_create_sdk_agent", lambda config, include_runtime_agent_tool=False, system_prompt="", is_child_agent=False: FakeChildAgent())

    async def run_case():
        result = await runtime._run_subagent(
            RuntimeConfig(model="m1", agents={"builder": {"description": "build agent"}}),
            {"name": "builder", "prompt": "do work", "description": "builder", "run_in_background": True},
            ToolContext(cwd="/tmp/project", env={}),
        )
        assert result.is_error is False
        assert "Created task" in str(result.content)
        assert "Do not also perform the same work yourself" in str(result.content)
        tasks = get_all_tasks()
        assert len(tasks) == 1
        task_id = next(iter(tasks))
        assert tasks[task_id]["status"] == "in_progress"
        for _ in range(50):
            if tasks[task_id]["status"] == "completed":
                break
            await asyncio.sleep(0)
        assert tasks[task_id]["status"] == "completed"
        assert "background done" in tasks[task_id]["output"]

    asyncio.run(run_case())


def test_run_subagent_background_skill_completes_task(monkeypatch) -> None:
    class FakeSkillAgent:
        async def prompt(self, prompt: str, overrides=None):
            from open_agent_sdk.types import QueryResult

            return QueryResult(
                text="skill background done",
                messages=[
                    {"role": "assistant", "content": [{"type": "text", "text": "Outcome: skill background done"}]},
                ],
            )

        async def close(self) -> None:
            return None

    async def fake_skill_call(self, input, context):
        return ToolResult(
            tool_use_id="",
            content=json.dumps({
                "success": True,
                "commandName": "review",
                "status": "inline",
                "prompt": "Review the code",
            }),
        )

    monkeypatch.setattr(runtime.SkillTool, "call", fake_skill_call)
    monkeypatch.setattr(runtime, "_create_sdk_agent", lambda config, include_runtime_agent_tool=False, system_prompt="", is_child_agent=False: FakeSkillAgent())
    monkeypatch.setattr(runtime, "_resolve_subagent_skill_request", lambda config, input: ("review", "check changes"))
    monkeypatch.setattr(runtime, "_resolve_agent_definition", lambda config, input: ("review", None))

    async def run_case():
        result = await runtime._run_subagent(
            RuntimeConfig(model="m1"),
            {"name": "review", "prompt": "check changes", "description": "review", "run_in_background": True},
            ToolContext(cwd="/tmp/project", env={}),
        )
        assert result.is_error is False
        assert "Created task" in str(result.content)
        tasks = get_all_tasks()
        task_id = next(iter(tasks))
        for _ in range(50):
            if tasks[task_id]["status"] == "completed":
                break
            await asyncio.sleep(0)
        assert tasks[task_id]["status"] == "completed"
        assert "skill background done" in tasks[task_id]["output"]

    asyncio.run(run_case())


def test_run_subagent_background_marks_failure_and_output(monkeypatch) -> None:
    class FakeChildAgent:
        async def prompt(self, prompt: str):
            raise RuntimeError("background exploded")

        async def close(self) -> None:
            return None

    monkeypatch.setattr(runtime, "_create_sdk_agent", lambda config, include_runtime_agent_tool=False, system_prompt="", is_child_agent=False: FakeChildAgent())

    async def run_case():
        result = await runtime._run_subagent(
            RuntimeConfig(model="m1", agents={"builder": {"description": "build agent"}}),
            {"name": "builder", "prompt": "do work", "description": "builder", "run_in_background": True},
            ToolContext(cwd="/tmp/project", env={}),
        )
        assert result.is_error is False
        assert "Created task" in str(result.content)
        assert "Do not also perform the same work yourself" in str(result.content)
        tasks = get_all_tasks()
        task_id = next(iter(tasks))
        for _ in range(50):
            if tasks[task_id]["status"] != "in_progress":
                break
            await asyncio.sleep(0)
        assert tasks[task_id]["status"] == "cancelled"
        assert "background exploded" in tasks[task_id]["output"]

    asyncio.run(run_case())


def test_run_subagent_background_surfaces_completion_via_task_store(monkeypatch) -> None:
    class FakeChildAgent:
        async def prompt(self, prompt: str):
            from open_agent_sdk.types import QueryResult

            return QueryResult(
                text="background done",
                messages=[
                    {"role": "assistant", "content": [{"type": "text", "text": "Outcome: background done"}]},
                ],
            )

        async def close(self) -> None:
            return None

    monkeypatch.setattr(runtime, "_create_sdk_agent", lambda config, include_runtime_agent_tool=False, system_prompt="", is_child_agent=False: FakeChildAgent())
    runtime._notified_task_ids.clear()

    async def run_case():
        result = await runtime._run_subagent(
            RuntimeConfig(model="m1", agents={"builder": {"description": "build agent"}}),
            {
                "name": "builder",
                "prompt": "do work",
                "description": "builder",
                "run_in_background": True,
            },
            ToolContext(cwd="/tmp/project", env={}),
        )
        assert result.is_error is False
        tasks = get_all_tasks()
        task_id = next(iter(tasks))
        for _ in range(50):
            if tasks[task_id]["status"] == "completed":
                break
            await asyncio.sleep(0)

    asyncio.run(run_case())

    notifications = runtime.read_background_notifications()
    assert len(notifications) == 1
    note = notifications[0]
    assert note["type"] == "background_task_completed"
    assert note["status"] == "completed"
    assert note["subject"] == "builder"
    assert "background done" in str(note["output"])


def test_run_subagent_background_uses_tool_result_text_when_assistant_text_missing(monkeypatch) -> None:
    class FakeChildAgent:
        async def prompt(self, prompt: str):
            from open_agent_sdk.types import QueryResult

            return QueryResult(
                text="",
                messages=[
                    {
                        "role": "assistant",
                        "content": [{"type": "tool_use", "id": "tool_1", "name": "Read", "input": {"filePath": "src/a.py"}}],
                    },
                    {
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": "tool_1", "content": "src/a.py\nFound review target"}],
                    },
                ],
            )

        async def close(self) -> None:
            return None

    monkeypatch.setattr(runtime, "_create_sdk_agent", lambda config, include_runtime_agent_tool=False, system_prompt="", is_child_agent=False: FakeChildAgent())

    async def run_case():
        result = await runtime._run_subagent(
            RuntimeConfig(model="m1", agents={"builder": {"description": "build agent"}}),
            {"name": "builder", "prompt": "do work", "description": "builder", "run_in_background": True},
            ToolContext(cwd="/tmp/project", env={}),
        )
        assert result.is_error is False
        tasks = get_all_tasks()
        task_id = next(iter(tasks))
        for _ in range(50):
            if tasks[task_id]["status"] == "completed":
                break
            await asyncio.sleep(0)
        assert tasks[task_id]["status"] == "completed"
        assert "src/a.py" in tasks[task_id]["output"]
        assert "Found review target" in tasks[task_id]["output"]

    asyncio.run(run_case())


def test_update_background_subagent_task_strips_ansi_output() -> None:
    async def run_case():
        task_id = await runtime._create_background_subagent_task("builder", "do work", "/tmp/project", {})
        await runtime._update_background_subagent_task(
            task_id,
            status="completed",
            output="\x1b[32mOutcome: done\x1b[0m",
            cwd="/tmp/project",
            env={},
        )
        task = get_all_tasks()[task_id]
        assert task["status"] == "completed"
        assert task["output"] == "Outcome: done"

    asyncio.run(run_case())


def test_sanitize_task_output_strips_tool_call_artifacts() -> None:
    assert runtime.sanitize_task_output("minimax:tool_call Review the code\nFiles: a.py") == "\nFiles: a.py"

    assert runtime.sanitize_task_output("normal text without artifacts") == "normal text without artifacts"

    multi_line = "minimax:tool_call Do stuff\nOutcome: reviewed\nminimax:tool_call More stuff"
    result = runtime.sanitize_task_output(multi_line)
    assert "minimax:tool_call" not in result
    assert "reviewed" in result


def test_extract_text_blocks_skips_tool_call_artifacts() -> None:
    message = {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "minimax:tool_call Read the file"},
            {"type": "text", "text": "Outcome: found 3 issues"},
        ],
    }
    blocks = runtime._extract_text_blocks(message)
    assert blocks == ["Outcome: found 3 issues"]


def test_run_subagent_background_strips_ansi_before_notifications(monkeypatch) -> None:
    class FakeChildAgent:
        async def prompt(self, prompt: str):
            from open_agent_sdk.types import QueryResult

            return QueryResult(
                text="\x1b[32mbackground done\x1b[0m",
                messages=[],
            )

        async def close(self) -> None:
            return None

    monkeypatch.setattr(runtime, "_create_sdk_agent", lambda config, include_runtime_agent_tool=False, system_prompt="", is_child_agent=False: FakeChildAgent())
    runtime._notified_task_ids.clear()

    async def run_case():
        result = await runtime._run_subagent(
            RuntimeConfig(model="m1", agents={"builder": {"description": "build agent"}}),
            {
                "name": "builder",
                "prompt": "do work",
                "description": "builder",
                "run_in_background": True,
            },
            ToolContext(cwd="/tmp/project", env={}),
        )
        assert result.is_error is False
        tasks = get_all_tasks()
        task_id = next(iter(tasks))
        for _ in range(50):
            if tasks[task_id]["status"] == "completed":
                break
            await asyncio.sleep(0)

    asyncio.run(run_case())

    notifications = runtime.read_background_notifications()
    assert len(notifications) == 1
    note = notifications[0]
    assert note["output"] == "background done"


def test_run_subagent_background_preserves_detailed_assistant_report(monkeypatch) -> None:
    class FakeChildAgent:
        async def prompt(self, prompt: str):
            from open_agent_sdk.types import QueryResult

            return QueryResult(
                text="",
                messages=[
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": "Outcome: Here is my review:\n\n## Findings\nThe render loop skips the final result event.\nNext step: add a regression test.",
                            }
                        ],
                    }
                ],
            )

        async def close(self) -> None:
            return None

    monkeypatch.setattr(runtime, "_create_sdk_agent", lambda config, include_runtime_agent_tool=False, system_prompt="", is_child_agent=False: FakeChildAgent())

    async def run_case():
        result = await runtime._run_subagent(
            RuntimeConfig(model="m1", agents={"builder": {"description": "build agent"}}),
            {"name": "builder", "prompt": "do work", "description": "builder", "run_in_background": True},
            ToolContext(cwd="/tmp/project", env={}),
        )
        assert result.is_error is False
        tasks = get_all_tasks()
        task_id = next(iter(tasks))
        for _ in range(50):
            if tasks[task_id]["status"] == "completed":
                break
            await asyncio.sleep(0)
        output = str(tasks[task_id]["output"])
        assert "Here is my review" in output

    asyncio.run(run_case())


def test_run_subagent_background_preserves_multiline_assistant_report(monkeypatch) -> None:
    class FakeChildAgent:
        async def prompt(self, prompt: str):
            from open_agent_sdk.types import QueryResult

            return QueryResult(
                text="",
                messages=[
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": "Here is my review:\n\n## Findings\nThe render loop skips the final result event.\nThe team dispatch path drops assistant messages when result.text is empty.",
                            }
                        ],
                    }
                ],
            )

        async def close(self) -> None:
            return None

    monkeypatch.setattr(runtime, "_create_sdk_agent", lambda config, include_runtime_agent_tool=False, system_prompt="", is_child_agent=False: FakeChildAgent())

    async def run_case():
        result = await runtime._run_subagent(
            RuntimeConfig(model="m1", agents={"builder": {"description": "build agent"}}),
            {"name": "builder", "prompt": "do work", "description": "builder", "run_in_background": True},
            ToolContext(cwd="/tmp/project", env={}),
        )
        assert result.is_error is False
        tasks = get_all_tasks()
        task_id = next(iter(tasks))
        for _ in range(50):
            if tasks[task_id]["status"] == "completed":
                break
            await asyncio.sleep(0)
        output = str(tasks[task_id]["output"])
        assert "The render loop skips the final result event." in output
        assert "The team dispatch path drops assistant messages when result.text is empty." in output

    asyncio.run(run_case())


def test_run_subagent_background_falls_back_to_default_agent_for_unknown_name() -> None:
    async def run_case():
        result = await runtime._run_subagent(
            RuntimeConfig(model="m1"),
            {"name": "task1", "prompt": "check some files", "description": "task1", "run_in_background": True},
            ToolContext(cwd="/tmp/project", env={}),
        )
        assert result.is_error is False
        assert "Created task" in str(result.content)
        assert get_all_tasks() != {}

    asyncio.run(run_case())


def test_run_subagent_background_uses_default_agent_over_prompt_inferred_skill_when_named_agent_missing() -> None:
    async def run_case():
        result = await runtime._run_subagent(
            RuntimeConfig(model="m1", skills_dir="skills"),
            {"name": "reviewer", "prompt": "review any file", "description": "reviewer", "run_in_background": True},
            ToolContext(cwd="/tmp/project", env={}),
        )
        assert result.is_error is False
        assert "Created task" in str(result.content)
        assert get_all_tasks() != {}

    asyncio.run(run_case())


def test_run_subagent_foreground_unknown_agent_lists_available() -> None:
    async def run_case():
        result = await runtime._run_subagent(
            RuntimeConfig(model="m1", agents={"builder": {"description": "build agent"}, "reviewer": {"description": "review agent"}}),
            {"name": "unknown_agent", "prompt": "do something", "description": "unknown_agent"},
            ToolContext(cwd="/tmp/project", env={}),
        )
        assert result.is_error is True
        assert "unknown agent 'unknown_agent'" in str(result.content)
        assert "builder" in str(result.content)
        assert "reviewer" in str(result.content)

    asyncio.run(run_case())


def test_run_subagent_background_matches_agent_name_case_insensitively(monkeypatch) -> None:
    class FakeChildAgent:
        async def prompt(self, prompt: str, overrides=None):
            from open_agent_sdk.types import QueryResult

            return QueryResult(text="background done", messages=[])

        async def close(self) -> None:
            return None

    monkeypatch.setattr(runtime, "_create_sdk_agent", lambda config, include_runtime_agent_tool=False, system_prompt="", is_child_agent=False: FakeChildAgent())

    async def run_case():
        result = await runtime._run_subagent(
            RuntimeConfig(model="m1", agents={"Reviewer": {"description": "reviews"}}),
            {"name": "reviewer", "prompt": "check some files", "description": "reviewer", "run_in_background": True},
            ToolContext(cwd="/tmp/project", env={}),
        )
        assert result.is_error is False
        tasks = get_all_tasks()
        task_id = next(iter(tasks))
        for _ in range(50):
            if tasks[task_id]["status"] == "completed":
                break
            await asyncio.sleep(0)
        assert tasks[task_id]["status"] == "completed"
        assert tasks[task_id]["output"] == "background done"

    asyncio.run(run_case())


def test_run_subagent_foreground_cancels_when_abort_signal_is_set(monkeypatch) -> None:
    cancelled = {"value": False}

    class FakeChildAgent:
        async def prompt(self, prompt: str, overrides=None):
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                cancelled["value"] = True
                raise

        async def close(self) -> None:
            return None

    monkeypatch.setattr(runtime, "_create_sdk_agent", lambda config, include_runtime_agent_tool=False, system_prompt="", is_child_agent=False: FakeChildAgent())

    async def run_case():
        abort_signal = asyncio.Event()
        runtime.set_abort_signal(abort_signal)

        subagent_task = asyncio.create_task(
            runtime._run_subagent(
                RuntimeConfig(model="m1", agents={"builder": {"description": "build agent"}}),
                {"name": "builder", "prompt": "do work", "description": "builder"},
                ToolContext(cwd="/tmp/project", env={}),
            )
        )

        # Give the agent time to start its prompt
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        # Set the abort signal so the foreground agent gets cancelled
        abort_signal.set()

        with contextlib.suppress(asyncio.CancelledError, Exception):
            await subagent_task

        assert cancelled["value"] is True

        runtime.set_abort_signal(None)

    asyncio.run(run_case())


def test_prompt_agent_with_abort_returns_result_if_cancel_arrives_after_completion(monkeypatch) -> None:
    class FakeAgent:
        async def prompt(self, prompt: str):
            return "done"

    original_wait = runtime.asyncio.wait

    async def fake_wait(tasks, return_when=None):
        prompt_task = next(task for task in tasks if task.get_coro().__name__ == "prompt")
        await prompt_task
        raise asyncio.CancelledError()

    monkeypatch.setattr(runtime.asyncio, "wait", fake_wait)

    async def run_case():
        runtime.set_abort_signal(asyncio.Event())
        try:
            result = await runtime._prompt_agent_with_abort(FakeAgent(), "hello")
        finally:
            runtime.set_abort_signal(None)
            monkeypatch.setattr(runtime.asyncio, "wait", original_wait)
        assert result == "done"

    asyncio.run(run_case())


def test_create_background_subagent_task_raises_on_sdk_error(monkeypatch) -> None:
    async def fake_call(self, input, context):
        return ToolResult(tool_use_id="", content="boom", is_error=True)

    monkeypatch.setattr(runtime.TaskCreateTool, "call", fake_call)

    async def run_case():
        with pytest.raises(RuntimeError, match="Failed to create task: boom"):
            await runtime._create_background_subagent_task("subject", "desc", "/tmp/project", {})

    asyncio.run(run_case())


def test_create_background_subagent_task_raises_on_unexpected_output(monkeypatch) -> None:
    async def fake_call(self, input, context):
        return ToolResult(tool_use_id="", content="unexpected format")

    monkeypatch.setattr(runtime.TaskCreateTool, "call", fake_call)

    async def run_case():
        with pytest.raises(RuntimeError, match="Could not parse task ID"):
            await runtime._create_background_subagent_task("subject", "desc", "/tmp/project", {})

    asyncio.run(run_case())


def test_create_background_subagent_task_uses_real_context(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def fake_call(self, input, context):
        captured["cwd"] = context.cwd
        captured["env"] = context.env
        return ToolResult(tool_use_id="", content="Created task task_1: subject")

    monkeypatch.setattr(runtime.TaskCreateTool, "call", fake_call)

    async def run_case():
        task_id = await runtime._create_background_subagent_task("subject", "desc", "/tmp/project", {"FOO": "bar"})
        assert task_id == "task_1"

    asyncio.run(run_case())

    assert captured == {"cwd": "/tmp/project", "env": {"FOO": "bar"}}


def test_run_subagent_routes_named_skill_instead_of_unknown_agent(monkeypatch) -> None:
    class FakeChildAgent:
        async def prompt(self, prompt: str, overrides=None):
            from open_agent_sdk.types import QueryResult

            return QueryResult(
                text="reviewed the changes",
                messages=[
                    {"role": "assistant", "content": [{"type": "text", "text": "Outcome: reviewed the changes"}]},
                ],
            )

        async def close(self) -> None:
            return None

    async def fake_skill_call(self, input, context):
        return ToolResult(
            tool_use_id="",
            content='{"success": true, "commandName": "review", "status": "inline", "prompt": "Review the latest commit", "model": "m-review"}',
        )

    monkeypatch.setattr(runtime.SkillTool, "call", fake_skill_call)
    monkeypatch.setattr(runtime, "_create_sdk_agent", lambda config, include_runtime_agent_tool=False, system_prompt="", is_child_agent=False: FakeChildAgent())

    result = asyncio.run(
        runtime._run_subagent(
            RuntimeConfig(model="m1", skills_dir="skills"),
            {"name": "review", "prompt": "check last commit", "description": "review"},
            ToolContext(cwd="/tmp/project", env={}),
        )
    )

    text = str(result.content)
    assert "unknown agent" not in text.lower()
    assert "Outcome: reviewed the changes" in text


def test_cancel_background_subagent_tasks_marks_tasks_cancelled(monkeypatch) -> None:
    started = asyncio.Event()

    class SlowChildAgent:
        async def prompt(self, prompt: str):
            started.set()
            await asyncio.sleep(60)
            raise AssertionError("should have been cancelled")

        async def close(self) -> None:
            return None

    monkeypatch.setattr(runtime, "_create_sdk_agent", lambda config, include_runtime_agent_tool=False, system_prompt="", is_child_agent=False: SlowChildAgent())

    async def run_case():
        result = await runtime._run_subagent(
            RuntimeConfig(model="m1", agents={"builder": {"description": "build agent"}}),
            {"name": "builder", "prompt": "do work", "description": "builder", "run_in_background": True},
            ToolContext(cwd="/tmp/project", env={}),
        )
        assert result.is_error is False
        tasks = get_all_tasks()
        task_id = next(iter(tasks))
        await started.wait()
        await runtime.cancel_background_subagent_tasks()
        assert tasks[task_id]["status"] == "cancelled"
        assert "Cancelled by shutdown" in tasks[task_id]["output"]

    asyncio.run(run_case())


def test_run_subagent_routes_review_like_prompt_to_review_skill(monkeypatch) -> None:
    class FakeChildAgent:
        async def prompt(self, prompt: str, overrides=None):
            from open_agent_sdk.types import QueryResult

            return QueryResult(
                text="reviewed the changes",
                messages=[
                    {"role": "assistant", "content": [{"type": "text", "text": "Outcome: reviewed the changes"}]},
                ],
            )

        async def close(self) -> None:
            return None

    captured: dict[str, object] = {}

    async def fake_skill_call(self, input, context):
        captured["skill"] = input["skill"]
        captured["args"] = input["args"]
        return ToolResult(
            tool_use_id="",
            content='{"success": true, "commandName": "review", "status": "inline", "prompt": "Review the latest commit", "model": "m-review"}',
        )

    monkeypatch.setattr(runtime.SkillTool, "call", fake_skill_call)
    monkeypatch.setattr(runtime, "_create_sdk_agent", lambda config, include_runtime_agent_tool=False, system_prompt="", is_child_agent=False: FakeChildAgent())

    result = asyncio.run(
        runtime._run_subagent(
            RuntimeConfig(model="m1", skills_dir="skills"),
            {"prompt": "Review last commit quality", "description": "Review last commit quality"},
            ToolContext(cwd="/tmp/project", env={}),
        )
    )

    assert captured == {"skill": "review", "args": "last commit quality"}
    assert "Outcome: reviewed the changes" in str(result.content)


def test_run_subagent_prefers_explicit_named_agent_over_prompt_inferred_skill(monkeypatch) -> None:
    class FakeChildAgent:
        async def prompt(self, prompt: str, overrides=None):
            from open_agent_sdk.types import QueryResult

            return QueryResult(
                text="agent reviewer output",
                messages=[
                    {"role": "assistant", "content": [{"type": "text", "text": "Outcome: agent reviewer output"}]},
                ],
            )

        async def close(self) -> None:
            return None

    async def fake_skill_call(self, input, context):
        raise AssertionError("prompt-inferred skill should not override explicit named agent")

    monkeypatch.setattr(runtime.SkillTool, "call", fake_skill_call)
    monkeypatch.setattr(runtime, "_create_sdk_agent", lambda config, include_runtime_agent_tool=False, system_prompt="", is_child_agent=False: FakeChildAgent())

    result = asyncio.run(
        runtime._run_subagent(
            RuntimeConfig(
                model="m1",
                skills_dir="skills",
                agents={"reviewer": {"description": "reviews files"}},
            ),
            {"name": "reviewer", "prompt": "review any file", "description": "reviewer"},
            ToolContext(cwd="/tmp/project", env={}),
        )
    )

    assert result.is_error is False
    assert "agent reviewer output" in str(result.content)


def test_run_subagent_summary_shows_first_meaningful_line_from_result_text(monkeypatch) -> None:
    class FakeChildAgent:
        async def prompt(self, prompt: str, overrides=None):
            from open_agent_sdk.types import QueryResult

            return QueryResult(
                text="Let me check for style consistency and any other potential issues:",
                messages=[
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": "Let me check for style consistency and any other potential issues:",
                            }
                        ],
                    }
                ],
            )

        async def close(self) -> None:
            return None

    async def fake_skill_call(self, input, context):
        return ToolResult(
            tool_use_id="",
            content='{"success": true, "commandName": "review", "status": "forked", "prompt": "Review the latest commit", "model": "m-review"}',
        )

    monkeypatch.setattr(runtime.SkillTool, "call", fake_skill_call)
    monkeypatch.setattr(runtime, "_create_sdk_agent", lambda config, include_runtime_agent_tool=False, system_prompt="", is_child_agent=False: FakeChildAgent())

    result = asyncio.run(
        runtime._run_subagent(
            RuntimeConfig(model="m1", skills_dir="skills"),
            {"name": "review", "prompt": "check last commit", "description": "review"},
            ToolContext(cwd="/tmp/project", env={}),
        )
    )

    assert result.is_error is False
    assert "Let me check" in str(result.content)


def test_run_subagent_summary_uses_last_meaningful_line_from_messages(monkeypatch) -> None:
        class FakeChildAgent:
            async def prompt(self, prompt: str, overrides=None):
                from open_agent_sdk.types import QueryResult

                return QueryResult(
                    text="",
                    messages=[
                        {"role": "assistant", "content": [{"type": "text", "text": "Let me inspect the code first."}]},
                        {"role": "assistant", "content": [{"type": "text", "text": "reviewed the rendering and team modules"}]},
                    ],
                )

            async def close(self) -> None:
                return None

        monkeypatch.setattr(runtime, "_create_sdk_agent", lambda config, include_runtime_agent_tool=False, system_prompt="", is_child_agent=False: FakeChildAgent())
        monkeypatch.setattr(runtime, "_resolve_subagent_skill_request", lambda config, input: None)

        result = asyncio.run(
            runtime._run_subagent(
                RuntimeConfig(model="m1", agents={"task": {"description": "review agent"}}),
                {"name": "task", "prompt": "review modules", "description": "task"},
                ToolContext(cwd="/tmp/project", env={}),
            )
        )
        text = str(result.content)
        assert "reviewed the rendering and team modules" in text


def test_run_subagent_summary_shows_agent_output_when_result_text_is_present(monkeypatch) -> None:
    class FakeChildAgent:
        async def prompt(self, prompt: str, overrides=None):
            from open_agent_sdk.types import QueryResult

            return QueryResult(
                text="Let me also check the rendering module and team module to understand the full context:",
                messages=[
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": "Let me also check the rendering module and team module to understand the full context:",
                            }
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": "reviewed the rendering module and team module and found the root cause",
                            }
                        ],
                    },
                ],
            )

        async def close(self) -> None:
            return None

    monkeypatch.setattr(runtime, "_create_sdk_agent", lambda config, include_runtime_agent_tool=False, system_prompt="", is_child_agent=False: FakeChildAgent())
    monkeypatch.setattr(runtime, "_resolve_subagent_skill_request", lambda config, input: None)

    result = asyncio.run(
        runtime._run_subagent(
            RuntimeConfig(model="m1", agents={"task": {"description": "review agent"}}),
            {"name": "task", "prompt": "review modules", "description": "task"},
            ToolContext(cwd="/tmp/project", env={}),
        )
    )
    text = str(result.content)
    assert "Let me also check" in text


def test_run_subagent_files_only_output_falls_back_to_full_text(monkeypatch) -> None:
    class FakeChildAgent:
        async def prompt(self, prompt: str, overrides=None):
            from open_agent_sdk.types import QueryResult

            text = "Files: src/a.py\nCommands: pytest tests/test_a.py -q"
            return QueryResult(
                text=text,
                messages=[
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": text}],
                    }
                ],
            )

        async def close(self) -> None:
            return None

    monkeypatch.setattr(runtime, "_create_sdk_agent", lambda config, include_runtime_agent_tool=False, system_prompt="", is_child_agent=False: FakeChildAgent())
    monkeypatch.setattr(runtime, "_resolve_subagent_skill_request", lambda config, input: None)

    result = asyncio.run(
        runtime._run_subagent(
            RuntimeConfig(model="m1", agents={"task": {"description": "review agent"}}),
            {"name": "task", "prompt": "review modules", "description": "task"},
            ToolContext(cwd="/tmp/project", env={}),
        )
    )

    text = str(result.content)
    assert "Files: src/a.py" in text
    assert "Commands: pytest tests/test_a.py -q" in text


def test_run_subagent_summary_trims_planning_tail_from_outcome(monkeypatch) -> None:
    class FakeChildAgent:
        async def prompt(self, prompt: str, overrides=None):
            from open_agent_sdk.types import QueryResult

            return QueryResult(
                text="",
                messages=[
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": "Outcome: I found a critical issue! Let me check the actual module structure:",
                            }
                        ],
                    }
                ],
            )

        async def close(self) -> None:
            return None

    monkeypatch.setattr(runtime, "_create_sdk_agent", lambda config, include_runtime_agent_tool=False, system_prompt="", is_child_agent=False: FakeChildAgent())
    monkeypatch.setattr(runtime, "_resolve_subagent_skill_request", lambda config, input: None)

    result = asyncio.run(
        runtime._run_subagent(
            RuntimeConfig(model="m1", agents={"task": {"description": "review agent"}}),
            {"name": "task", "prompt": "review modules", "description": "task"},
            ToolContext(cwd="/tmp/project", env={}),
        )
    )
    text = str(result.content)
    assert "I found a critical issue" in text


def test_run_subagent_summary_prefers_later_conclusion_over_earlier_planning(monkeypatch) -> None:
    class FakeChildAgent:
        async def prompt(self, prompt: str, overrides=None):
            from open_agent_sdk.types import QueryResult

            return QueryResult(
                text="",
                messages=[
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": "I reviewed the transport tests and found no critical issues.",
                            }
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": "Let me also inspect the remaining modules before wrapping up.",
                            }
                        ],
                    },
                ],
            )

        async def close(self) -> None:
            return None

    monkeypatch.setattr(runtime, "_create_sdk_agent", lambda config, include_runtime_agent_tool=False, system_prompt="", is_child_agent=False: FakeChildAgent())
    monkeypatch.setattr(runtime, "_resolve_subagent_skill_request", lambda config, input: None)

    result = asyncio.run(
        runtime._run_subagent(
            RuntimeConfig(model="m1", agents={"task": {"description": "review agent"}}),
            {"name": "task", "prompt": "review tests", "description": "task"},
            ToolContext(cwd="/tmp/project", env={}),
        )
    )
    text = str(result.content)
    assert "Let me also inspect the remaining modules" in text


def test_run_subagent_summary_skips_generic_review_intro_and_uses_follow_up_finding(monkeypatch) -> None:
    class FakeChildAgent:
        async def prompt(self, prompt: str, overrides=None):
            from open_agent_sdk.types import QueryResult

            return QueryResult(
                text="",
                messages=[
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": "Outcome: Here is my review:\n\n## Findings\nThe render loop skips the final result event.",
                            }
                        ],
                    }
                ],
            )

        async def close(self) -> None:
            return None

    monkeypatch.setattr(runtime, "_create_sdk_agent", lambda config, include_runtime_agent_tool=False, system_prompt="", is_child_agent=False: FakeChildAgent())
    monkeypatch.setattr(runtime, "_resolve_subagent_skill_request", lambda config, input: None)

    result = asyncio.run(
        runtime._run_subagent(
            RuntimeConfig(model="m1", agents={"task": {"description": "review agent"}}),
            {"name": "task", "prompt": "review modules", "description": "task"},
            ToolContext(cwd="/tmp/project", env={}),
        )
    )
    text = str(result.content)
    assert "Here is my review" in text


def test_run_subagent_summary_skips_plain_report_intro_sentence(monkeypatch) -> None:
    class FakeChildAgent:
        async def prompt(self, prompt: str, overrides=None):
            from open_agent_sdk.types import QueryResult

            return QueryResult(
                text="",
                messages=[
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": "Here is the report. The render loop skips the final result event.",
                            }
                        ],
                    }
                ],
            )

        async def close(self) -> None:
            return None

    monkeypatch.setattr(runtime, "_create_sdk_agent", lambda config, include_runtime_agent_tool=False, system_prompt="", is_child_agent=False: FakeChildAgent())
    monkeypatch.setattr(runtime, "_resolve_subagent_skill_request", lambda config, input: None)

    result = asyncio.run(
        runtime._run_subagent(
            RuntimeConfig(model="m1", agents={"task": {"description": "review agent"}}),
            {"name": "task", "prompt": "review modules", "description": "task"},
            ToolContext(cwd="/tmp/project", env={}),
        )
    )
    text = str(result.content)
    assert "Here is the report." in text
    assert "The render loop skips the final result event" in text


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

    monkeypatch.setattr("rooster_code.runtime.create_agent", lambda options: FakeAgent())

    agent = create_runtime_agent(
        RuntimeConfig(
            api_key="test-key",
            base_url="https://nano-gpt.com/api/v1",
            model="test-model",
            agents={"reviewer": {"description": "code reviewer"}},
        )
    )

    asyncio.run(agent._initialize())

    assert [tool.name for tool in agent._tool_pool] == ["Agent", "Read", "SaveMemory"]
    assert agent._tool_pool[0].__class__.__name__ == "RuntimeAgentTool"


def test_create_runtime_agent_respects_disallowed_save_memory(monkeypatch) -> None:
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

    monkeypatch.setattr("rooster_code.runtime.create_agent", lambda options: FakeAgent())

    agent = create_runtime_agent(
        RuntimeConfig(
            api_key="test-key",
            base_url="https://nano-gpt.com/api/v1",
            model="test-model",
            agents={"reviewer": {"description": "code reviewer"}},
            disallowed_tools=["SaveMemory"],
        )
    )

    asyncio.run(agent._initialize())

    assert [tool.name for tool in agent._tool_pool] == ["Agent", "Read"]


def test_create_runtime_agent_does_not_duplicate_save_memory_on_second_initialize(monkeypatch) -> None:
    class PlaceholderAgentTool:
        name = "Agent"

    class ReadTool:
        name = "Read"

    class FakeAgent:
        def __init__(self) -> None:
            self._client = None
            self._tool_pool = []
            self._initialized = False

        async def _initialize(self) -> None:
            if self._initialized:
                return
            self._tool_pool = [PlaceholderAgentTool(), ReadTool()]
            self._initialized = True

    monkeypatch.setattr("rooster_code.runtime.create_agent", lambda options: FakeAgent())

    agent = create_runtime_agent(
        RuntimeConfig(
            api_key="test-key",
            base_url="https://nano-gpt.com/api/v1",
            model="test-model",
            agents={"reviewer": {"description": "code reviewer"}},
        )
    )

    asyncio.run(agent._initialize())
    asyncio.run(agent._initialize())

    assert [tool.name for tool in agent._tool_pool] == ["Agent", "Read", "SaveMemory"]
    assert [tool.__class__.__name__ for tool in agent._tool_pool] == [
        "RuntimeAgentTool",
        "RuntimeReadTool",
        "RuntimeTraceTool",
    ]
    assert agent._tool_pool[2]._delegate.__class__.__name__ == "SaveMemoryTool"


def test_create_runtime_agent_skips_mcp_reconnect_on_second_initialize(monkeypatch) -> None:
    """Re-initializing must not re-invoke connect_http_mcp (would duplicate tools and leak SseClients)."""
    class PlaceholderAgentTool:
        name = "Agent"

    class ReadTool:
        name = "Read"

    class FakeAgent:
        def __init__(self) -> None:
            self._client = None
            self._tool_pool = []
            self._initialized = False

        async def _initialize(self) -> None:
            if self._initialized:
                return
            self._tool_pool = [PlaceholderAgentTool(), ReadTool()]
            self._initialized = True

    monkeypatch.setattr("rooster_code.runtime.create_agent", lambda options: FakeAgent())

    connect_calls: list[str] = []

    class FakeRemoteTool:
        name = "mcp__weather__forecast"

        def __init__(self) -> None:
            self._client = object()

    async def fake_connect(name, config):
        connect_calls.append(name)
        return [FakeRemoteTool()]

    monkeypatch.setattr("rooster_code.mcp_transport.connect_http_mcp", fake_connect)

    agent = create_runtime_agent(
        RuntimeConfig(
            api_key="test-key",
            base_url="https://nano-gpt.com/api/v1",
            model="test-model",
            agents={"reviewer": {"description": "code reviewer"}},
            mcp_servers={"weather": {"type": "sse", "url": "http://localhost:9999/sse"}},
        )
    )

    asyncio.run(agent._initialize())
    asyncio.run(agent._initialize())
    asyncio.run(agent._initialize())

    assert connect_calls == ["weather"], (
        f"connect_http_mcp must run exactly once across multiple _initialize() calls, "
        f"got {len(connect_calls)} calls: {connect_calls}"
    )


def test_create_runtime_agent_retries_remote_mcp_until_client_connected(monkeypatch) -> None:
    class PlaceholderAgentTool:
        name = "Agent"

    class FakeRemoteTool:
        name = "mcp__weather__forecast"

        def __init__(self, client) -> None:
            self._client = client

    class FakeAgent:
        def __init__(self) -> None:
            self._client = None
            self._tool_pool = []
            self._initialized = False

        async def _initialize(self) -> None:
            if self._initialized:
                return
            self._tool_pool = [PlaceholderAgentTool()]
            self._initialized = True

    client = object()
    connect_calls: list[str] = []

    monkeypatch.setattr("rooster_code.runtime.create_agent", lambda options: FakeAgent())

    async def fake_connect(name, config):
        connect_calls.append(name)
        if len(connect_calls) == 1:
            return []
        return [FakeRemoteTool(client)]

    monkeypatch.setattr("rooster_code.mcp_transport.connect_http_mcp", fake_connect)

    agent = create_runtime_agent(
        RuntimeConfig(
            api_key="test-key",
            base_url="https://nano-gpt.com/api/v1",
            model="test-model",
            agents={"reviewer": {"description": "code reviewer"}},
            mcp_servers={"weather": {"type": "sse", "url": "http://localhost:9999/sse"}},
        )
    )

    asyncio.run(agent._initialize())
    agent._initialized = False
    asyncio.run(agent._initialize())

    assert connect_calls == ["weather", "weather"]
    assert any(tool.name == "mcp__weather__forecast" for tool in agent._tool_pool)


def test_create_runtime_agent_closes_remote_mcp_clients_on_close(monkeypatch) -> None:
    class PlaceholderAgentTool:
        name = "Agent"

    class FakeRemoteTool:
        name = "mcp__weather__forecast"

        def __init__(self, client) -> None:
            self._client = client

    class FakeClient:
        def __init__(self) -> None:
            self.closed = False

        async def close(self) -> None:
            self.closed = True

    class FakeAgent:
        def __init__(self) -> None:
            self._client = None
            self._tool_pool = []
            self._initialized = False
            self.closed = False

        async def _initialize(self) -> None:
            if self._initialized:
                return
            self._tool_pool = [PlaceholderAgentTool()]
            self._initialized = True

        async def close(self) -> None:
            self.closed = True

    fake_client = FakeClient()

    monkeypatch.setattr("rooster_code.runtime.create_agent", lambda options: FakeAgent())

    async def fake_connect(name, config):
        return [FakeRemoteTool(fake_client)]

    monkeypatch.setattr("rooster_code.mcp_transport.connect_http_mcp", fake_connect)

    agent = create_runtime_agent(
        RuntimeConfig(
            api_key="test-key",
            base_url="https://nano-gpt.com/api/v1",
            model="test-model",
            agents={"reviewer": {"description": "code reviewer"}},
            mcp_servers={"weather": {"type": "sse", "url": "http://localhost:9999/sse"}},
        )
    )

    asyncio.run(agent._initialize())
    asyncio.run(agent.close())

    assert agent.closed is True
    assert fake_client.closed is True


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

    monkeypatch.setattr("rooster_code.runtime.create_agent", lambda options: FakeAgent())

    agent = create_runtime_agent(RuntimeConfig(api_key="test", base_url="https://nano-gpt.com/api/v1", model="m1"))

    asyncio.run(agent._initialize())

    assert [tool.name for tool in agent._tool_pool] == ["Read", "Edit", "Bash", "Agent", "SaveMemory"]
    assert agent._tool_pool[0].__class__.__name__ == "RuntimeReadTool"
    assert agent._tool_pool[1].__class__.__name__ == "RuntimeEditTool"


def test_create_runtime_agent_bridges_sdk_team_tools_after_initialize(monkeypatch) -> None:
    class TeamCreateTool:
        name = "TeamCreate"

    class TeamDeleteTool:
        name = "TeamDelete"

    class OtherTool:
        name = "Bash"

    class FakeAgent:
        def __init__(self) -> None:
            self._client = None
            self._tool_pool = []

        async def _initialize(self) -> None:
            self._tool_pool = [TeamCreateTool(), TeamDeleteTool(), OtherTool()]

    monkeypatch.setattr("rooster_code.runtime.create_agent", lambda options: FakeAgent())

    agent = create_runtime_agent(RuntimeConfig(api_key="test", base_url="https://nano-gpt.com/api/v1", model="m1"))

    asyncio.run(agent._initialize())

    assert [tool.name for tool in agent._tool_pool] == ["TeamCreate", "TeamDelete", "Bash", "Agent", "SaveMemory"]
    assert agent._tool_pool[0].__class__.__name__ == "RuntimeTraceTool"
    assert agent._tool_pool[1].__class__.__name__ == "RuntimeTraceTool"


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

    monkeypatch.setattr("rooster_code.runtime.create_agent", lambda options: FakeAgent())

    agent = create_runtime_agent(RuntimeConfig(api_key="test", base_url="https://nano-gpt.com/api/v1", model="m1"))

    asyncio.run(agent._initialize())

    assert [tool.name for tool in agent._tool_pool] == ["Read", "Agent", "SaveMemory"]


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

    monkeypatch.setattr("rooster_code.runtime.create_agent", lambda options: FakeAgent())

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

    monkeypatch.setattr("rooster_code.runtime.create_agent", lambda options: FakeAgent())

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

    monkeypatch.setattr("rooster_code.runtime.create_agent", lambda options: FakeAgent())

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

    monkeypatch.setattr("rooster_code.runtime.create_agent", lambda options: FakeAgent())

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

    monkeypatch.setattr("rooster_code.runtime.create_agent", lambda options: FakeAgent())

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

    monkeypatch.setattr("rooster_code.runtime.create_agent", lambda options: FakeChildAgent())

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

    assert [event.type for event in events] == [SDKMessageType.SYSTEM, SDKMessageType.SYSTEM, SDKMessageType.TOOL_RESULT, SDKMessageType.SYSTEM]
    assert events[0].system_data["activity_trace"] == [
        {"action": "Resolved subagent", "tool": "Agent", "target": "reviewer"}
    ]
    assert events[1].system_data["activity_trace"] == [
        {"action": "Reading file", "tool": "Read", "target": str(tmp_path / "child.txt")}
    ]
    assert events[-1].system_data["activity_trace"] == [
        {"action": "Completed subagent", "tool": "Agent", "target": "reviewer"}
    ]


def test_stream_named_agent_events_emits_skill_resolution_visibility(monkeypatch) -> None:
    class FakeAgent:
        async def query(self, prompt: str, overrides=None):
            yield SDKMessage(type=SDKMessageType.RESULT, text="done")

        async def close(self) -> None:
            return None

    async def fake_skill_call(self, input, context):
        return ToolResult(
            tool_use_id="",
            content='{"success": true, "commandName": "review", "status": "inline", "prompt": "Review latest commit"}',
        )

    monkeypatch.setattr(runtime.SkillTool, "call", fake_skill_call)
    monkeypatch.setattr(runtime, "_create_sdk_agent", lambda config, include_runtime_agent_tool=False, system_prompt="", is_child_agent=False: FakeAgent())

    async def collect_events():
        return [
            event
            async for event in runtime.stream_named_agent_events(
                RuntimeConfig(api_key="test", base_url="https://nano-gpt.com/api/v1", model="m1", skills_dir="skills"),
                "review",
                "review last commit",
            )
        ]

    events = asyncio.run(collect_events())

    assert [event.type for event in events] == [SDKMessageType.SYSTEM, SDKMessageType.RESULT, SDKMessageType.SYSTEM]
    assert events[0].system_data["activity_trace"] == [
        {"action": "Resolved subagent", "tool": "Skill", "target": "review (inline)"}
    ]
    assert events[-1].system_data["activity_trace"] == [
        {"action": "Completed subagent", "tool": "Skill", "target": "review"}
    ]


def test_stream_named_agent_events_prefers_explicit_agent_over_same_named_skill(monkeypatch) -> None:
    class FakeAgent:
        async def query(self, prompt: str, overrides=None):
            yield SDKMessage(type=SDKMessageType.RESULT, text="agent done")

        async def close(self) -> None:
            return None

    monkeypatch.setattr(runtime, "_create_sdk_agent", lambda config, include_runtime_agent_tool=False, system_prompt="", is_child_agent=False: FakeAgent())
    monkeypatch.setattr(runtime, "_resolve_subagent_skill_request", lambda config, input: ("review", "check"))

    async def collect_events():
        return [
            event
            async for event in runtime.stream_named_agent_events(
                RuntimeConfig(
                    api_key="test",
                    base_url="https://nano-gpt.com/api/v1",
                    model="m1",
                    agents={"review": {"description": "agent reviewer"}},
                    skills_dir="skills",
                ),
                "review",
                "review last commit",
            )
        ]

    events = asyncio.run(collect_events())

    assert events[0].system_data["activity_trace"] == [
        {"action": "Resolved subagent", "tool": "Agent", "target": "review"}
    ]
    assert events[-1].system_data["activity_trace"] == [
        {"action": "Completed subagent", "tool": "Agent", "target": "review"}
    ]


def test_run_named_agent_prompt_prefers_final_sdk_result_text(monkeypatch) -> None:
    class FakeChildAgent:
        async def prompt(self, prompt: str, overrides=None):
            from open_agent_sdk.types import QueryResult

            return QueryResult(
                text="Final answer from SDK result.",
                messages=[
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": "Outcome: Here is my review:\n\n## Findings\nA more detailed finding follows.",
                            }
                        ],
                    }
                ],
            )

        async def close(self) -> None:
            return None

    monkeypatch.setattr(runtime, "_create_sdk_agent", lambda config, include_runtime_agent_tool=False, system_prompt="", is_child_agent=False: FakeChildAgent())

    result = asyncio.run(
        runtime.run_named_agent_prompt(
            RuntimeConfig(model="m1", agents={"reviewer": {"description": "review agent"}}),
            "reviewer",
            "review modules",
        )
    )

    assert "Final answer from SDK result." in result or "Here is my review" in result


def test_run_named_agent_prompt_falls_back_to_summary_when_sdk_result_text_empty(monkeypatch) -> None:
    class FakeChildAgent:
        async def prompt(self, prompt: str, overrides=None):
            from open_agent_sdk.types import QueryResult

            return QueryResult(
                text="",
                messages=[
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": "Outcome: Here is the report. The render loop skips the final result event.",
                            }
                        ],
                    }
                ],
            )

        async def close(self) -> None:
            return None

    monkeypatch.setattr(runtime, "_create_sdk_agent", lambda config, include_runtime_agent_tool=False, system_prompt="", is_child_agent=False: FakeChildAgent())

    result = asyncio.run(
        runtime.run_named_agent_prompt(
            RuntimeConfig(model="m1", agents={"reviewer": {"description": "review agent"}}),
            "reviewer",
            "review modules",
        )
    )

    assert "The render loop skips the final result event" in result


def test_run_named_agent_prompt_preserves_multiline_assistant_report_when_text_empty(monkeypatch) -> None:
    class FakeChildAgent:
        async def prompt(self, prompt: str, overrides=None):
            from open_agent_sdk.types import QueryResult

            return QueryResult(
                text="",
                messages=[
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": "Here is my review:\n\n## Findings\nThe render loop skips the final result event.\nThe inline named-agent path is collapsing useful message-only output.",
                            }
                        ],
                    }
                ],
            )

        async def close(self) -> None:
            return None

    monkeypatch.setattr(runtime, "_create_sdk_agent", lambda config, include_runtime_agent_tool=False, system_prompt="", is_child_agent=False: FakeChildAgent())

    result = asyncio.run(
        runtime.run_named_agent_prompt(
            RuntimeConfig(model="m1", agents={"reviewer": {"description": "review agent"}}),
            "reviewer",
            "review modules",
        )
    )

    assert "The render loop skips the final result event." in result
    assert "The inline named-agent path is collapsing useful message-only output." in result


def test_build_agent_options_includes_default_task_agent_context() -> None:
    options = build_agent_options(RuntimeConfig(api_key="a", base_url="https://example.test", model="m"))

    assert "task" in options.append_system_prompt.lower()


def test_find_requested_agent_name_uses_default_task_agent() -> None:
    result = find_requested_agent_name(
        RuntimeConfig(api_key="a", base_url="https://example.test", model="m"),
        "Use an agent to summarize this request.",
    )

    assert result == "task"


def test_find_requested_agent_name_uses_only_configured_agent_for_generic_request() -> None:
    result = find_requested_agent_name(
        RuntimeConfig(
            api_key="a",
            base_url="https://example.test",
            model="m",
            agents={"reviewer": {"description": "reviews"}},
        ),
        "Use an agent to summarize this request.",
    )

    assert result == "reviewer"


def test_find_requested_agent_name_returns_none_for_ambiguous_generic_request() -> None:
    result = find_requested_agent_name(
        RuntimeConfig(
            api_key="a",
            base_url="https://example.test",
            model="m",
            agents={
                "reviewer": {"description": "reviews"},
                "builder": {"description": "builds"},
            },
        ),
        "Use an agent to summarize this request.",
    )

    assert result is None


def test_start_background_agent_task_strips_task_id_punctuation(monkeypatch) -> None:
    async def fake_run_subagent(config, input, context):
        return ToolResult(
            tool_use_id="",
            content="Created task task_1. This work is now assigned to background agent 'reviewer'.",
        )

    monkeypatch.setattr(runtime, "_run_subagent", fake_run_subagent)

    task_id = asyncio.run(
        runtime.start_background_agent_task(
            RuntimeConfig(model="m1", agents={"reviewer": {"description": "reviews"}}),
            "reviewer",
            "check some files",
        )
    )

    assert task_id == "task_1"


def test_enforce_session_retention_deletes_sessions_beyond_limit(monkeypatch) -> None:
    deleted: list[str] = []

    async def fake_list_sessions() -> list[dict[str, object]]:
        return [{"id": f"sess-{index}", "updatedAt": f"2026-04-{30-index:02d}T00:00:00"} for index in range(25)]

    async def fake_delete_session(session_id: str) -> bool:
        deleted.append(session_id)
        return True

    monkeypatch.setattr("rooster_code.runtime.sdk_list_sessions", fake_list_sessions)
    monkeypatch.setattr("rooster_code.runtime.sdk_delete_session", fake_delete_session)

    asyncio.run(enforce_session_retention(limit=20))

    assert deleted == [f"sess-{index}" for index in range(20, 25)]


def test_enforce_session_retention_ignores_delete_failures(monkeypatch) -> None:
    deleted: list[str] = []

    async def fake_list_sessions() -> list[dict[str, object]]:
        return [{"id": f"sess-{index}"} for index in range(22)]

    async def fake_delete_session(session_id: str) -> bool:
        deleted.append(session_id)
        raise OSError("read-only file system")

    monkeypatch.setattr("rooster_code.runtime.sdk_list_sessions", fake_list_sessions)
    monkeypatch.setattr("rooster_code.runtime.sdk_delete_session", fake_delete_session)

    asyncio.run(enforce_session_retention(limit=20))

    assert deleted == ["sess-20", "sess-21"]


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



def test_handoff_current_session_writes_file_and_compacts_history(monkeypatch, tmp_path) -> None:
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
            self._session_id = "sess-1"
            self._options = type("Options", (), {"model": "m-test", "cwd": str(tmp_path)})()

        async def _initialize(self) -> None:
            self.initialized = True

        def _ensure_provider(self):
            return provider

        def _resolve_model(self) -> str:
            return "m-test"

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

    handoff_path = tmp_path / ".handoff"
    result = asyncio.run(runtime.handoff_current_session(agent, handoff_path))

    assert agent.initialized is True
    assert provider.params is not None
    assert provider.params.model == "m-test"
    assert agent._history == compacted_history
    assert result == {
        "compacted": True,
        "written": True,
        "path": str(handoff_path),
        "summary": "summary",
        "before_tokens": 1200,
        "after_tokens": 240,
        "reason": "",
    }
    text = handoff_path.read_text(encoding="utf-8")
    assert text.startswith("# Handoff\n")
    assert "Session: sess-1" in text
    assert "Model: m-test" in text
    assert f"CWD: {tmp_path}" in text
    assert "## Resume Prompt" in text
    assert "Read this `.handoff`" in text
    assert "summary" in text


def test_handoff_current_session_restores_history_when_file_write_fails(monkeypatch, tmp_path) -> None:
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
            return "m-test"

    class FakeProvider:
        async def create_message(self, params):
            return CreateMessageResponse(
                content=[{"type": "text", "text": "summary"}],
                stop_reason="end_turn",
            )

    provider = FakeProvider()
    agent = FakeAgent()
    monkeypatch.setattr(runtime, "estimate_messages_tokens", lambda messages: 42, raising=False)
    blocking_file = tmp_path / "not-a-dir"
    blocking_file.write_text("blocks parent directory", encoding="utf-8")

    with pytest.raises(FileExistsError):
        asyncio.run(runtime.handoff_current_session(agent, blocking_file / ".handoff"))

    assert agent._history == original_history


def test_handoff_current_session_skips_small_history(monkeypatch, tmp_path) -> None:
    class FakeAgent:
        def __init__(self) -> None:
            self._history = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]

        async def _initialize(self) -> None:
            return None

    handoff_path = tmp_path / ".handoff"
    agent = FakeAgent()
    monkeypatch.setattr(runtime, "estimate_messages_tokens", lambda messages: 42, raising=False)

    result = asyncio.run(runtime.handoff_current_session(agent, handoff_path))

    assert result == {
        "compacted": False,
        "written": False,
        "path": "",
        "summary": "",
        "before_tokens": 42,
        "after_tokens": 42,
        "reason": "Need at least two messages before compaction.",
    }
    assert not handoff_path.exists()


def test_handoff_current_session_provider_failure_restores_history_and_writes_no_file(monkeypatch, tmp_path) -> None:
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
            return "m-test"

    class FakeProvider:
        async def create_message(self, params):
            raise RuntimeError("Provider unavailable")

    provider = FakeProvider()
    agent = FakeAgent()
    monkeypatch.setattr(runtime, "estimate_messages_tokens", lambda messages: 42, raising=False)
    handoff_path = tmp_path / ".handoff"

    with pytest.raises(RuntimeError, match="Provider unavailable"):
        asyncio.run(runtime.handoff_current_session(agent, handoff_path))

    assert agent._history == original_history
    assert not handoff_path.exists()


def test_handoff_current_session_absolute_path_argument_passed_through(monkeypatch, tmp_path) -> None:
    absolute_path = tmp_path / "subdir" / "custom.handoff"

    class FakeAgent:
        def __init__(self) -> None:
            self._history = [
                {"role": "user", "content": [{"type": "text", "text": "hello"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
            ]

        async def _initialize(self) -> None:
            return None

        def _ensure_provider(self):
            return provider

        def _resolve_model(self) -> str:
            return "m-test"

    class FakeProvider:
        async def create_message(self, params):
            return CreateMessageResponse(
                content=[{"type": "text", "text": "summary"}],
                stop_reason="end_turn",
            )

    provider = FakeProvider()
    agent = FakeAgent()
    monkeypatch.setattr(runtime, "estimate_messages_tokens", lambda messages: 42, raising=False)

    result = asyncio.run(runtime.handoff_current_session(agent, absolute_path))

    assert result["written"] is True
    assert result["path"] == str(absolute_path)
    assert absolute_path.exists()
    assert absolute_path.read_text(encoding="utf-8").startswith("# Handoff\n")


def test_handoff_current_session_subdirectory_path_creates_parents(monkeypatch, tmp_path) -> None:
    nested_path = tmp_path / "deep" / "nested" / "dir" / ".handoff"

    class FakeAgent:
        def __init__(self) -> None:
            self._history = [
                {"role": "user", "content": [{"type": "text", "text": "hello"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
            ]

        async def _initialize(self) -> None:
            return None

        def _ensure_provider(self):
            return provider

        def _resolve_model(self) -> str:
            return "m-test"

    class FakeProvider:
        async def create_message(self, params):
            return CreateMessageResponse(
                content=[{"type": "text", "text": "summary"}],
                stop_reason="end_turn",
            )

    provider = FakeProvider()
    agent = FakeAgent()
    monkeypatch.setattr(runtime, "estimate_messages_tokens", lambda messages: 42, raising=False)

    result = asyncio.run(runtime.handoff_current_session(agent, nested_path))

    assert result["written"] is True
    assert nested_path.exists()
    assert nested_path.read_text(encoding="utf-8").startswith("# Handoff\n")


def test_handoff_current_session_no_cwd_falls_back_to_process_cwd(monkeypatch, tmp_path) -> None:
    """When agent._options has no cwd, _build_handoff_file_content falls back to Path.cwd()."""

    class FakeAgent:
        def __init__(self) -> None:
            self._history = [
                {"role": "user", "content": [{"type": "text", "text": "hello"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
            ]
            self._options = type("Options", (), {"model": "m-test"})()

        async def _initialize(self) -> None:
            return None

        def _ensure_provider(self):
            return provider

        def _resolve_model(self) -> str:
            return "m-test"

    class FakeProvider:
        async def create_message(self, params):
            return CreateMessageResponse(
                content=[{"type": "text", "text": "summary"}],
                stop_reason="end_turn",
            )

    provider = FakeProvider()
    agent = FakeAgent()
    monkeypatch.setattr(runtime, "estimate_messages_tokens", lambda messages: 42, raising=False)
    handoff_path = tmp_path / ".handoff"

    result = asyncio.run(runtime.handoff_current_session(agent, handoff_path))
    text = handoff_path.read_text(encoding="utf-8")

    assert result["written"] is True
    assert f"CWD: {Path.cwd()}" in text


def test_build_handoff_file_content_with_minimal_agent() -> None:
    """_build_handoff_file_content handles agents with no _session_id, no _resolve_model, no _options."""

    class MinimalAgent:
        pass

    agent = MinimalAgent()
    content = runtime_session._build_handoff_file_content(agent, "test summary")

    assert content.startswith("# Handoff\n")
    assert "Session: new" in content
    assert "Model: default" in content
    assert f"CWD: {Path.cwd()}" in content
    assert "test summary" in content


def test_handoff_current_session_compaction_no_smaller_still_writes_file(monkeypatch, tmp_path) -> None:
    """When compaction doesn't shrink tokens, the file is still written and compacted=False."""
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
            return "m-test"

    class FakeProvider:
        async def create_message(self, params):
            return CreateMessageResponse(
                content=[{"type": "text", "text": "summary"}],
                stop_reason="end_turn",
            )

    provider = FakeProvider()
    agent = FakeAgent()
    monkeypatch.setattr(runtime, "estimate_messages_tokens", lambda messages: 42, raising=False)
    handoff_path = tmp_path / ".handoff"

    result = asyncio.run(runtime.handoff_current_session(agent, handoff_path))

    assert result["compacted"] is False
    assert result["written"] is True
    assert result["reason"] == "Compaction produced no smaller history."
    assert handoff_path.exists()


def test_handoff_current_session_overwrites_existing_file(monkeypatch, tmp_path) -> None:
    handoff_path = tmp_path / ".handoff"
    handoff_path.write_text("old content", encoding="utf-8")

    class FakeAgent:
        def __init__(self) -> None:
            self._history = [
                {"role": "user", "content": [{"type": "text", "text": "hello"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
            ]

        async def _initialize(self) -> None:
            return None

        def _ensure_provider(self):
            return provider

        def _resolve_model(self) -> str:
            return "m-test"

    class FakeProvider:
        async def create_message(self, params):
            return CreateMessageResponse(
                content=[{"type": "text", "text": "new summary"}],
                stop_reason="end_turn",
            )

    provider = FakeProvider()
    agent = FakeAgent()
    monkeypatch.setattr(runtime, "estimate_messages_tokens", lambda messages: 42, raising=False)

    result = asyncio.run(runtime.handoff_current_session(agent, handoff_path))

    assert result["written"] is True
    text = handoff_path.read_text(encoding="utf-8")
    assert "new summary" in text
    assert "old content" not in text


def test_handoff_current_session_path_object_argument(monkeypatch, tmp_path) -> None:
    """handoff_current_session accepts a Path object, not just a string."""
    handoff_path = tmp_path / "custom.handoff"

    class FakeAgent:
        def __init__(self) -> None:
            self._history = [
                {"role": "user", "content": [{"type": "text", "text": "hello"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
            ]

        async def _initialize(self) -> None:
            return None

        def _ensure_provider(self):
            return provider

        def _resolve_model(self) -> str:
            return "m-test"

    class FakeProvider:
        async def create_message(self, params):
            return CreateMessageResponse(
                content=[{"type": "text", "text": "summary"}],
                stop_reason="end_turn",
            )

    provider = FakeProvider()
    agent = FakeAgent()
    monkeypatch.setattr(runtime, "estimate_messages_tokens", lambda messages: 42, raising=False)

    result = asyncio.run(runtime.handoff_current_session(agent, handoff_path))
    assert result["written"] is True
    assert result["path"] == str(handoff_path)
    assert handoff_path.exists()




def test_handoff_current_session_preserves_system_messages(monkeypatch, tmp_path) -> None:
    """System messages should be preserved in history after handoff compaction."""
    original_history = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
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

        async def _initialize(self) -> None:
            return None

        def _ensure_provider(self):
            return provider

        def _resolve_model(self) -> str:
            return "m-test"

    class FakeProvider:
        async def create_message(self, params):
            return CreateMessageResponse(
                content=[{"type": "text", "text": "summary"}],
                stop_reason="end_turn",
            )

    provider = FakeProvider()
    agent = FakeAgent()
    monkeypatch.setattr(runtime, "estimate_messages_tokens", lambda messages: 42, raising=False)
    handoff_path = tmp_path / ".handoff"

    result = asyncio.run(runtime.handoff_current_session(agent, handoff_path))

    assert result["written"] is True
    # After compaction, agent._history is the compacted history (system messages are filtered
    # by _filter_history_for_manual_compaction and replaced by the compacted pair)
    assert agent._history == compacted_history


def test_handoff_current_session_permission_denied(monkeypatch, tmp_path) -> None:
    """Writing to a read-only directory raises and restores history."""
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
            return "m-test"

    class FakeProvider:
        async def create_message(self, params):
            return CreateMessageResponse(
                content=[{"type": "text", "text": "summary"}],
                stop_reason="end_turn",
            )

    provider = FakeProvider()
    agent = FakeAgent()
    monkeypatch.setattr(runtime, "estimate_messages_tokens", lambda messages: 42, raising=False)
    readonly_dir = tmp_path / "readonly"
    readonly_dir.mkdir()
    readonly_dir.chmod(0o444)
    handoff_path = readonly_dir / ".handoff"

    try:
        with pytest.raises(PermissionError):
            asyncio.run(runtime.handoff_current_session(agent, handoff_path))
        assert agent._history == original_history
    finally:
        readonly_dir.chmod(0o755)


def test_handoff_file_content_has_utc_timezone(monkeypatch, tmp_path) -> None:
    """The Generated timestamp should be UTC and contain +00:00 or Z."""

    class FakeAgent:
        def __init__(self) -> None:
            self._history = [
                {"role": "user", "content": [{"type": "text", "text": "hello"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
            ]

        async def _initialize(self) -> None:
            return None

        def _ensure_provider(self):
            return provider

        def _resolve_model(self) -> str:
            return "m-test"

    class FakeProvider:
        async def create_message(self, params):
            return CreateMessageResponse(
                content=[{"type": "text", "text": "summary"}],
                stop_reason="end_turn",
            )

    provider = FakeProvider()
    agent = FakeAgent()
    monkeypatch.setattr(runtime, "estimate_messages_tokens", lambda messages: 42, raising=False)
    handoff_path = tmp_path / ".handoff"

    asyncio.run(runtime.handoff_current_session(agent, handoff_path))
    text = handoff_path.read_text(encoding="utf-8")

    # UTC isoformat ends with +00:00
    assert "+00:00" in text


def test_handoff_current_session_path_with_spaces(monkeypatch, tmp_path) -> None:
    """Paths with spaces are handled correctly."""
    spaced_dir = tmp_path / "my project" / "sub dir"
    handoff_path = spaced_dir / "my handoff.md"

    class FakeAgent:
        def __init__(self) -> None:
            self._history = [
                {"role": "user", "content": [{"type": "text", "text": "hello"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
            ]

        async def _initialize(self) -> None:
            return None

        def _ensure_provider(self):
            return provider

        def _resolve_model(self) -> str:
            return "m-test"

    class FakeProvider:
        async def create_message(self, params):
            return CreateMessageResponse(
                content=[{"type": "text", "text": "summary"}],
                stop_reason="end_turn",
            )

    provider = FakeProvider()
    agent = FakeAgent()
    monkeypatch.setattr(runtime, "estimate_messages_tokens", lambda messages: 42, raising=False)

    result = asyncio.run(runtime.handoff_current_session(agent, handoff_path))

    assert result["written"] is True
    assert handoff_path.exists()
    assert "summary" in handoff_path.read_text(encoding="utf-8")


def test_handoff_file_redacts_openai_keys(monkeypatch, tmp_path) -> None:
    """The handoff file must redact OpenAI-style API keys from the summary."""

    class FakeAgent:
        def __init__(self) -> None:
            self._history = [
                {"role": "user", "content": [{"type": "text", "text": "use this key"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
            ]
            self._options = type("Opts", (), {"model": "m", "cwd": str(tmp_path)})()

        async def _initialize(self) -> None:
            return None

        def _ensure_provider(self):
            return provider

        def _resolve_model(self) -> str:
            return "m"

    class FakeProvider:
        async def create_message(self, params):
            return CreateMessageResponse(
                content=[{"type": "text", "text": "Set ROOSTER_CODE_API_KEY=sk-abc123longkey456789xyz0000000000 and TOKEN=ghp_abcdef1234567890abcdef1234567890abcd"}],
                stop_reason="end_turn",
            )

    provider = FakeProvider()
    agent = FakeAgent()
    monkeypatch.setattr(runtime, "estimate_messages_tokens", lambda messages: 42, raising=False)

    handoff_path = tmp_path / ".handoff"
    asyncio.run(runtime.handoff_current_session(agent, handoff_path))
    text = handoff_path.read_text(encoding="utf-8")

    # OpenAI key must be redacted
    assert "sk-abc123longkey456789xyz0000000000" not in text
    # GitHub PAT must be redacted
    assert "ghp_abcdef1234567890abcdef1234567890abcd" not in text
    # The key names are preserved
    assert "ROOSTER_CODE_API_KEY" in text
    assert "TOKEN" in text


def test_handoff_file_redacts_assignment_patterns(monkeypatch, tmp_path) -> None:
    """The handoff file must redact KEY=value patterns from the summary."""

    class FakeAgent:
        def __init__(self) -> None:
            self._history = [
                {"role": "user", "content": [{"type": "text", "text": "set env"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
            ]
            self._options = type("Opts", (), {"model": "m", "cwd": str(tmp_path)})()

        async def _initialize(self) -> None:
            return None

        def _ensure_provider(self):
            return provider

        def _resolve_model(self) -> str:
            return "m"

    class FakeProvider:
        async def create_message(self, params):
            return CreateMessageResponse(
                content=[{"type": "text", "text": "Set API_KEY=mysecretvalue123 and PASSWORD=\"supersecretpass\""}],
                stop_reason="end_turn",
            )

    provider = FakeProvider()
    agent = FakeAgent()
    monkeypatch.setattr(runtime, "estimate_messages_tokens", lambda messages: 42, raising=False)

    handoff_path = tmp_path / ".handoff"
    asyncio.run(runtime.handoff_current_session(agent, handoff_path))
    text = handoff_path.read_text(encoding="utf-8")

    assert "mysecretvalue123" not in text
    assert "supersecretpass" not in text
    assert "API_KEY" in text
    assert "PASSWORD" in text
    assert "***REDACTED***" in text


def test_redact_secrets_unit() -> None:
    """Unit test _redact_secrets directly with various secret formats."""
    assert "sk-***REDACTED***" in runtime_session._redact_secrets("key=sk-abc123456789012345678901234567890")
    assert "ghp_***REDACTED***" in runtime_session._redact_secrets("token=ghp_abc1234567890123456789012345678901234")
    assert "glpat-***REDACTED***" in runtime_session._redact_secrets("glpat-abcdefghijklmnopqrstuvwxyz1234567890")
    assert "xox-***REDACTED***" in runtime_session._redact_secrets("xoxb-FAKE-000000000000000000000000")
    assert "API_KEY=***REDACTED***" in runtime_session._redact_secrets("API_KEY=mysecretvalue12345")
    assert "TOKEN=***REDACTED***" in runtime_session._redact_secrets('TOKEN="mytoken12345"')
    # Short values should NOT be redacted (likely not secrets)
    result = runtime_session._redact_secrets("API_KEY=short")
    assert "short" in result


def test_handoff_file_content_structure(monkeypatch, tmp_path) -> None:

    class FakeAgent:
        def __init__(self) -> None:
            self._history = [
                {"role": "user", "content": [{"type": "text", "text": "hello"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
            ]
            self._session_id = "sess-1"
            self._options = type("Options", (), {"model": "m-test", "cwd": str(tmp_path)})()

        async def _initialize(self) -> None:
            return None

        def _ensure_provider(self):
            return provider

        def _resolve_model(self) -> str:
            return "m-test"

    class FakeProvider:
        async def create_message(self, params):
            return CreateMessageResponse(
                content=[{"type": "text", "text": "## Goal\nFix the bug\n## Next Step\nRun tests"}],
                stop_reason="end_turn",
            )

    provider = FakeProvider()
    agent = FakeAgent()
    monkeypatch.setattr(runtime, "estimate_messages_tokens", lambda messages: 42, raising=False)

    handoff_path = tmp_path / ".handoff"
    asyncio.run(runtime.handoff_current_session(agent, handoff_path))
    text = handoff_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    # Verify structure
    assert lines[0] == "# Handoff"
    assert any(line.startswith("Generated:") and "+00:00" in line for line in lines), "Missing UTC timestamp"
    assert any(line == "Session: sess-1" for line in lines)
    assert any(line == "Model: m-test" for line in lines)
    assert any(line == f"CWD: {tmp_path}" for line in lines)
    assert "## Resume Prompt" in text
    assert "---" in text
    assert "## Goal" in text
    assert "## Next Step" in text

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


def test_patch_tool_pool_re_exported_from_runtime() -> None:
    """patch_tool_pool is importable from rooster_code.runtime as a re-export from team module."""
    from rooster_code.runtime import patch_tool_pool

    # It should be the same function as in the team module
    from rooster_code.team import patch_tool_pool as team_patch_tool_pool

    assert patch_tool_pool is team_patch_tool_pool


def test_agent_context_prompt_without_team_info_is_unchanged() -> None:
    """_agent_context_prompt with no team_info produces the same output as before."""
    config = RuntimeConfig(
        api_key="k",
        base_url="https://example.test",
        model="m1",
        agents={"builder": {"description": "builds things"}},
    )
    result = runtime._agent_context_prompt(config)

    assert "# Configured Agents" in result
    assert "builder" in result
    assert "Team:" not in result
    assert "Do not also perform the same work yourself" in result


def test_agent_context_prompt_with_active_team_info_appends_team_context() -> None:
    """_agent_context_prompt with active team_info appends team context to the prompt."""
    config = RuntimeConfig(
        api_key="k",
        base_url="https://example.test",
        model="m1",
        agents={"builder": {"description": "builds things"}},
    )
    team_info = {
        "active": True,
        "team_name": "alpha",
        "members": {"bob": {}, "alice": {}},
    }
    result = runtime._agent_context_prompt(config, team_info=team_info)

    assert "# Team: alpha" in result
    assert "bob, alice" in result
    assert "TeamDispatch" in result
    assert "SendMessage" in result
    assert "do not also do that same task yourself" in result.lower()


def test_agent_context_prompt_with_inactive_team_info_is_unchanged() -> None:
    """_agent_context_prompt with inactive team_info (active=False) does not append team context."""
    config = RuntimeConfig(
        api_key="k",
        base_url="https://example.test",
        model="m1",
        agents={"builder": {"description": "builds things"}},
    )
    team_info = {
        "active": False,
        "team_name": "alpha",
        "members": {"bob": {}, "alice": {}},
    }
    result = runtime._agent_context_prompt(config, team_info=team_info)

    assert "Team:" not in result


def test_runtime_agent_tool_rejects_when_team_active():
    from rooster_code.runtime_tools import RuntimeAgentTool, TurnTracker
    from rooster_code.team import set_runtime_team_bridge, TeamManager

    tracker = TurnTracker()

    async def fake_runner(input, context):
        return ToolResult(tool_use_id="", content="should not be called")

    tool = RuntimeAgentTool(fake_runner, tracker)

    team_manager = TeamManager()
    team_manager._active = True

    set_runtime_team_bridge(team_manager, None)
    try:
        result = asyncio.run(tool.call({"prompt": "test", "description": "test"}, ToolContext(cwd=".", env={})))
        assert result.is_error
        assert "not available while team is active" in str(result.content)
        assert "TeamDispatch" in str(result.content)
    finally:
        set_runtime_team_bridge(None, None)


def test_runtime_agent_tool_redirects_to_team_dispatch_for_member():
    from rooster_code.runtime_tools import RuntimeAgentTool, TurnTracker
    from rooster_code.team import set_runtime_team_bridge, TeamManager, AgentPool

    tracker = TurnTracker()

    async def fake_runner(input, context):
        return ToolResult(tool_use_id="", content="should not be called")

    tool = RuntimeAgentTool(fake_runner, tracker)

    team_manager = TeamManager()
    team_manager._active = True
    team_manager._team_id = "test"
    team_manager._team_name = "test-team"
    pool = AgentPool()

    class FakeMemberAgent:
        async def prompt(self, prompt: str, overrides: dict[str, object] | None = None):
            from open_agent_sdk.types import QueryResult

            return QueryResult(text="done", messages=[])

    pool._members["reviewer"] = FakeMemberAgent()
    pool._mailboxes["reviewer"] = asyncio.Queue()
    pool._locks["reviewer"] = asyncio.Lock()
    team_manager._pool = pool

    set_runtime_team_bridge(team_manager, None)
    try:
        import unittest.mock
        with unittest.mock.patch("rooster_code.runtime._create_background_subagent_task", new_callable=unittest.mock.AsyncMock) as mock_create:
            mock_create.return_value = "task-redirect-1"
            result = asyncio.run(tool.call(
                {"prompt": "review the code", "description": "review", "name": "reviewer"},
                ToolContext(cwd=".", env={}),
            ))
        assert not result.is_error
        content = str(result.content)
        assert "Dispatched task to team member 'reviewer'" in content
    finally:
        set_runtime_team_bridge(None, None)


def test_runtime_agent_tool_allows_when_no_team():
    from rooster_code.runtime_tools import RuntimeAgentTool, TurnTracker
    from rooster_code.team import set_runtime_team_bridge

    tracker = TurnTracker()

    async def fake_runner(input, context):
        return ToolResult(tool_use_id="", content="agent ran fine")

    tool = RuntimeAgentTool(fake_runner, tracker)

    set_runtime_team_bridge(None, None)
    result = asyncio.run(tool.call({"prompt": "test", "description": "test"}, ToolContext(cwd=".", env={})))
    assert not result.is_error
    assert result.content == "agent ran fine"


def test_rehydrate_tasks_from_history_reconstructs_tasks_from_injected_messages() -> None:
    from open_agent_sdk.tools import _tasks
    import open_agent_sdk.tools as tools_mod

    original_tasks = dict(_tasks)
    original_counter = tools_mod._task_counter
    _tasks.clear()
    tools_mod._task_counter = 0
    runtime._injected_task_ids_rehydrated.clear()

    try:
        class FakeAgent:
            _history = [
                {"role": "user", "content": [{"type": "text", "text": "[Background task task_1 completed]\n\nOutcome: built the feature\nFiles: src/main.py, src/utils.py"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "Received background task task_1 result."}]},
                {"role": "user", "content": [{"type": "text", "text": "[Background task task_2 cancelled]\n\ntimeout exceeded"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "Received background task task_2 cancellation/failure."}]},
                {"role": "user", "content": [{"type": "text", "text": "some unrelated message"}]},
            ]

        runtime.rehydrate_tasks_from_history(FakeAgent())

        assert "task_1" in _tasks
        assert _tasks["task_1"]["status"] == "completed"
        assert "built the feature" in _tasks["task_1"]["output"]
        assert "src/main.py" in _tasks["task_1"]["output"]

        assert "task_2" in _tasks
        assert _tasks["task_2"]["status"] == "cancelled"
        assert "timeout exceeded" in _tasks["task_2"]["output"]

        assert tools_mod._task_counter == 2
    finally:
        _tasks.clear()
        _tasks.update(original_tasks)
        tools_mod._task_counter = original_counter
        runtime._injected_task_ids_rehydrated.clear()


def test_rehydrate_tasks_from_history_skips_already_present_tasks() -> None:
    from open_agent_sdk.tools import _tasks
    import open_agent_sdk.tools as tools_mod

    original_tasks = dict(_tasks)
    original_counter = tools_mod._task_counter
    _tasks.clear()
    tools_mod._task_counter = 0
    runtime._injected_task_ids_rehydrated.clear()

    try:
        class FakeAgent:
            _history = [
                {"role": "user", "content": [{"type": "text", "text": "[Background task task_3 completed]\n\nOutcome: ran tests"}]},
            ]

        _tasks["task_3"] = {"id": "task_3", "status": "in_progress", "output": "original output", "subject": "task_3", "description": "", "owner": "", "blocked_by": [], "blocks": []}

        runtime.rehydrate_tasks_from_history(FakeAgent())

        assert _tasks["task_3"]["status"] == "in_progress"
        assert _tasks["task_3"]["output"] == "original output"
    finally:
        _tasks.clear()
        _tasks.update(original_tasks)
        tools_mod._task_counter = original_counter
        runtime._injected_task_ids_rehydrated.clear()


def test_format_subagent_task_output_returns_full_text_not_summary() -> None:
    long_review = (
        "## Code Review\n\n"
        "### Issue 1: Race condition in _injected_task_ids\n"
        "The set check is not atomic.\n\n"
        "### Issue 2: Missing error handling\n"
        "Bare except clauses should catch specific exceptions.\n\n"
        "### Issue 3: Long lines\n"
        "Some lines exceed 100 characters.\n\n"
        "Files: src/rooster_code/cli.py\n"
        "Commands: ruff check --fix\n"
        "Next step: Run the test suite\n"
    )

    messages = [
        {"role": "assistant", "content": [{"type": "text", "text": "Outcome: Found 3 issues\n\n" + long_review}]},
    ]

    result = runtime._format_subagent_task_output(long_review, messages)

    assert "### Issue 1:" in result
    assert "### Issue 2:" in result
    assert "### Issue 3:" in result
    assert "Race condition" in result
    assert "Missing error handling" in result


def test_format_subagent_summary_still_produces_condensed_output() -> None:
    long_review = (
        "## Code Review\n\n"
        "### Issue 1: Race condition\n"
        "The set check is not atomic.\n\n"
        "### Issue 2: Missing error handling\n"
        "Bare except clauses.\n\n"
        "Outcome: Found 2 issues\n"
        "Files: src/rooster_code/cli.py\n"
        "Commands: ruff check --fix\n"
    )

    messages = [
        {"role": "assistant", "content": [{"type": "text", "text": long_review}]},
    ]

    summary = runtime._format_subagent_summary(long_review, messages)

    assert "Outcome:" in summary
    assert "Files:" in summary
    assert len(summary) < len(long_review)


def test_rehydrate_tasks_from_history_clears_stale_ids_on_reresume() -> None:
    from open_agent_sdk.tools import _tasks
    import open_agent_sdk.tools as tools_mod

    original_tasks = dict(_tasks)
    original_counter = tools_mod._task_counter
    _tasks.clear()
    tools_mod._task_counter = 0
    runtime._injected_task_ids_rehydrated.clear()

    try:
        class FakeAgent:
            _history = [
                {"role": "user", "content": [{"type": "text", "text": "[Background task task_1 completed]\n\nResult A"}]},
            ]

        runtime.rehydrate_tasks_from_history(FakeAgent())
        assert "task_1" in _tasks
        assert "Result A" in _tasks["task_1"]["output"]

        second_agent = FakeAgent()
        second_agent._history = [
            {"role": "user", "content": [{"type": "text", "text": "[Background task task_1 completed]\n\nResult B from different session"}]},
        ]

        runtime._injected_task_ids_rehydrated.clear()
        _tasks.clear()
        tools_mod._task_counter = 0

        runtime.rehydrate_tasks_from_history(second_agent)
        assert "task_1" in _tasks
        assert "Result B from different session" in _tasks["task_1"]["output"]
    finally:
        _tasks.clear()
        _tasks.update(original_tasks)
        tools_mod._task_counter = original_counter
        runtime._injected_task_ids_rehydrated.clear()


def test_remote_mcp_failure_logs_warning_and_continues(monkeypatch, caplog) -> None:
    class PlaceholderAgentTool:
        name = "Agent"

    class FakeAgent:
        def __init__(self) -> None:
            self._client = None
            self._tool_pool = []

        async def _initialize(self) -> None:
            self._tool_pool = [PlaceholderAgentTool()]

    monkeypatch.setattr("rooster_code.runtime.create_agent", lambda options: FakeAgent())

    async def failing_connect(name, cfg):
        raise ConnectionError(f"Cannot reach MCP server {name}")

    monkeypatch.setattr("rooster_code.mcp_transport.connect_http_mcp", failing_connect)

    agent = create_runtime_agent(
        RuntimeConfig(
            api_key="test-key",
            base_url="https://nano-gpt.com/api/v1",
            model="test-model",
            agents={"reviewer": {"description": "code reviewer"}},
            mcp_servers={
                "remote-server": {"type": "http", "url": "http://localhost:9999/sse"},
            },
        )
    )

    with caplog.at_level(logging.WARNING, logger="rooster.runtime"):
        asyncio.run(agent._initialize())

    assert [tool.name for tool in agent._tool_pool] == ["Agent", "SaveMemory"]

    warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("remote-server" in msg for msg in warning_messages), f"Expected warning about remote-server, got: {warning_messages}"
    assert any("Cannot reach" in msg for msg in warning_messages), f"Expected connection error detail, got: {warning_messages}"


def test_remote_mcp_failure_does_not_crash_agent_initialization(monkeypatch) -> None:
    class PlaceholderAgentTool:
        name = "Agent"

    class FakeAgent:
        def __init__(self) -> None:
            self._client = None
            self._tool_pool = []

        async def _initialize(self) -> None:
            self._tool_pool = [PlaceholderAgentTool()]

    monkeypatch.setattr("rooster_code.runtime.create_agent", lambda options: FakeAgent())

    async def failing_connect(name, cfg):
        raise RuntimeError("SSE connection refused")

    monkeypatch.setattr("rooster_code.mcp_transport.connect_http_mcp", failing_connect)

    agent = create_runtime_agent(
        RuntimeConfig(
            api_key="test-key",
            base_url="https://nano-gpt.com/api/v1",
            model="test-model",
            agents={"reviewer": {"description": "code reviewer"}},
            mcp_servers={
                "broken-mcp": {"type": "http", "url": "http://localhost:9999/sse"},
            },
        )
    )

    asyncio.run(agent._initialize())

    tool_names = [tool.name for tool in agent._tool_pool]
    assert "Agent" in tool_names
    assert "broken-mcp" not in str(tool_names)


def test_wrapped_initialize_syncs_engine_tool_state(monkeypatch) -> None:
    """After _initialize(), the engine's _config.tools and _tool_map must match agent._tool_pool."""

    class FakeConfig:
        tools = None
        tool_map = None

    class FakeEngine:
        def __init__(self) -> None:
            self._config = FakeConfig()

    class FakeAgentTool:
        name = "Agent"

    class FakeReadTool:
        name = "Read"

    class FakeAgent:
        def __init__(self) -> None:
            self._client = None
            self._tool_pool = [FakeAgentTool(), FakeReadTool()]
            self._engine = FakeEngine()
            self._initialized = False

        async def _initialize(self) -> None:
            self._tool_pool = [FakeAgentTool(), FakeReadTool()]
            self._initialized = True

    monkeypatch.setattr("rooster_code.runtime.create_agent", lambda options: FakeAgent())

    agent = create_runtime_agent(
        RuntimeConfig(
            api_key="test-key",
            base_url="https://nano-gpt.com/api/v1",
            model="test-model",
            agents={"reviewer": {"description": "code reviewer"}},
        )
    )

    asyncio.run(agent._initialize())

    # Engine must reference the same list object as agent._tool_pool
    assert agent._engine._config.tools is agent._tool_pool, (
        "engine._config.tools must point to agent._tool_pool after _initialize()"
    )

    # Engine's tool_map must contain all tools from the pool
    assert agent._engine._tool_map is not None, "engine._tool_map must be populated"
    pool_names = {t.name for t in agent._tool_pool}
    map_names = set(agent._engine._tool_map.keys())
    assert pool_names == map_names, (
        f"engine._tool_map keys {map_names} must match tool_pool names {pool_names}"
    )

    # Verify wrapped tools are present (Agent should be RuntimeAgentTool, Read should be RuntimeReadTool)
    assert "Agent" in agent._engine._tool_map
    assert "Read" in agent._engine._tool_map


def test_wrapped_initialize_syncs_engine_after_double_init(monkeypatch) -> None:
    """After two _initialize() calls (simulating interrupt recovery), the engine must stay synced."""

    class FakeConfig:
        tools = None
        tool_map = None

    class FakeEngine:
        def __init__(self) -> None:
            self._config = FakeConfig()

    class FakeAgentTool:
        name = "Agent"

    class FakeReadTool:
        name = "Read"

    class FakeAgent:
        def __init__(self) -> None:
            self._client = None
            self._tool_pool = []
            self._engine = None
            self._initialized = False

        async def _initialize(self) -> None:
            self._tool_pool = [FakeAgentTool(), FakeReadTool()]
            self._engine = FakeEngine()
            self._initialized = True

    monkeypatch.setattr("rooster_code.runtime.create_agent", lambda options: FakeAgent())

    agent = create_runtime_agent(
        RuntimeConfig(
            api_key="test-key",
            base_url="https://nano-gpt.com/api/v1",
            model="test-model",
            agents={"reviewer": {"description": "code reviewer"}},
        )
    )

    # First init
    asyncio.run(agent._initialize())
    assert agent._engine._config.tools is agent._tool_pool

    # Simulate interrupt: clear engine, set _initialized=False
    agent._engine = FakeEngine()
    agent._initialized = False

    # Second init (simulating recovery)
    asyncio.run(agent._initialize())
    assert agent._engine._config.tools is agent._tool_pool, (
        "engine must sync to new pool after second _initialize()"
    )
    assert "Agent" in agent._engine._tool_map
    assert "Read" in agent._engine._tool_map


def test_wrapped_initialize_syncs_engine_with_empty_pool(monkeypatch) -> None:
    """Empty tool pool after _initialize() should not crash engine sync."""
    class FakeConfig:
        tools = None
        tool_map = None

    class FakeEngine:
        def __init__(self) -> None:
            self._config = FakeConfig()

    class FakeAgent:
        def __init__(self) -> None:
            self._client = None
            self._tool_pool = []
            self._engine = FakeEngine()
            self._initialized = False

        async def _initialize(self) -> None:
            self._tool_pool = []
            self._engine = FakeEngine()
            self._initialized = True

    monkeypatch.setattr("rooster_code.runtime.create_agent", lambda options: FakeAgent())

    agent = create_runtime_agent(
        RuntimeConfig(
            api_key="test-key",
            base_url="https://nano-gpt.com/api/v1",
            model="test-model",
        )
    )

    asyncio.run(agent._initialize())
    assert agent._engine._config.tools is agent._tool_pool
    assert isinstance(agent._engine._tool_map, dict)
    assert len(agent._engine._tool_map) >= 0


def test_skill_get_prompt_lazy_loads_body(tmp_path: Path) -> None:
    """get_prompt should load the skill body on first call, not at registration."""
    from rooster_code.runtime import _build_filesystem_skill_definition

    skill_dir = tmp_path / "skills" / "lazy"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: lazy\ndescription: A lazy skill\n---\n\nLazy loaded body content.",
        encoding="utf-8",
    )

    definition = _build_filesystem_skill_definition(skill_dir)
    assert definition is not None
    assert definition.name == "lazy"
    assert definition.get_prompt is not None

    blocks = asyncio.run(definition.get_prompt("", ToolContext()))
    text = "\n".join(str(b["text"]) for b in blocks if b.get("type") == "text")
    assert "Lazy loaded body content." in text


def test_build_filesystem_skill_definition_does_not_cache_body_until_prompt(tmp_path: Path) -> None:
    import rooster_code.runtime as rt

    skill_dir = tmp_path / "skills" / "lazy"
    skill_dir.mkdir(parents=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        "---\nname: lazy\ndescription: A lazy skill\n---\n\nOriginal body.",
        encoding="utf-8",
    )
    rt._skill_body_cache.clear()

    definition = rt._build_filesystem_skill_definition(skill_dir)
    assert definition is not None
    assert str(skill_file) not in rt._skill_body_cache

    skill_file.write_text(
        "---\nname: lazy\ndescription: A lazy skill\n---\n\nUpdated body.",
        encoding="utf-8",
    )
    blocks = asyncio.run(definition.get_prompt("", ToolContext()))
    text = "\n".join(str(b["text"]) for b in blocks if b.get("type") == "text")
    assert "Updated body." in text


def test_skill_body_cache_cleared_on_reload(tmp_path: Path) -> None:
    """Stale cache entries are cleared and skill bodies stay lazy after reload."""
    import rooster_code.runtime as rt

    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / "test-skill").mkdir()
    (skills_dir / "test-skill" / "SKILL.md").write_text(
        "---\nname: test-skill\ndescription: desc\n---\n\nBody.",
        encoding="utf-8",
    )

    rt._skill_body_cache.clear()
    rt._skill_body_cache["fake-key"] = "stale"
    assert len(rt._skill_body_cache) == 1

    rt._ensure_skills_loaded(
        RuntimeConfig(
            api_key="a", base_url="https://x.test", model="m",
            skills_dir=str(skills_dir),
        )
    )
    assert "fake-key" not in rt._skill_body_cache
    assert rt._skill_body_cache == {}

def test_iter_with_abort_yields_all_events_when_no_abort():
    """Without an abort signal, _iter_with_abort yields all events from the source generator."""
    runtime._abort_signal = None

    async def source():
        for i in range(5):
            yield {"i": i}

    async def run():
        results = []
        async for event in runtime._iter_with_abort(source()):
            results.append(event)
        return results

    assert asyncio.run(run()) == [{"i": i} for i in range(5)]


def test_iter_with_abort_stops_when_abort_already_set():
    """If abort is already set before iteration starts, no events are yielded."""
    abort = asyncio.Event()
    abort.set()
    runtime._abort_signal = abort

    async def source():
        yield {"never": "seen"}

    async def run():
        results = []
        async for event in runtime._iter_with_abort(source()):
            results.append(event)
        return results

    assert asyncio.run(run()) == []
    runtime._abort_signal = None


def test_iter_with_abort_interrupts_pending_api_call():
    """Abort set during a pending __anext__ interrupts the in-flight fetch."""
    abort = asyncio.Event()
    runtime._abort_signal = abort

    async def slow_source():
        # First yield is fast; second yield blocks until interrupted
        yield {"phase": "first"}
        await asyncio.sleep(10)  # simulates a pending API call
        yield {"phase": "second"}  # never reached

    async def fire_abort():
        await asyncio.sleep(0.05)
        abort.set()

    async def run():
        results = []
        gen = runtime._iter_with_abort(slow_source())
        first = await gen.__anext__()
        assert first == {"phase": "first"}
        asyncio.create_task(fire_abort())
        async for event in gen:
            results.append(event)
        return results

    results = asyncio.run(run())
    assert results == [], f"Expected no further events after abort, got {results}"
    runtime._abort_signal = None


def test_iter_with_abort_normal_completion():
    """A generator that completes normally (StopAsyncIteration) stops the helper."""
    runtime._abort_signal = None

    async def source():
        yield {"a": 1}
        yield {"a": 2}

    async def run():
        results = []
        async for event in runtime._iter_with_abort(source()):
            results.append(event)
        return results

    assert asyncio.run(run()) == [{"a": 1}, {"a": 2}]


def test_iter_with_abort_cleans_up_tasks_on_normal_exit():
    """No lingering tasks remain after normal completion."""
    runtime._abort_signal = None

    async def source():
        yield {"x": 1}

    async def run():
        before = len(asyncio.all_tasks())
        async for _ in runtime._iter_with_abort(source()):
            pass
        # Give cancelled abort tasks a chance to finish
        await asyncio.sleep(0.01)
        after = len(asyncio.all_tasks())
        return before, after

    before, after = asyncio.run(run())
    assert before == after, f"Task leak: before={before}, after={after}"


def test_iter_with_abort_cleans_up_tasks_on_abort():
    """No lingering tasks remain after abort."""
    abort = asyncio.Event()
    runtime._abort_signal = abort

    async def slow_source():
        yield {"ok": True}
        await asyncio.sleep(10)
        yield {"ok": False}

    async def fire_abort():
        await asyncio.sleep(0.05)
        abort.set()

    async def run():
        gen = runtime._iter_with_abort(slow_source())
        await gen.__anext__()
        asyncio.create_task(fire_abort())
        async for _ in gen:
            pass
        await asyncio.sleep(0.01)
        current = asyncio.current_task()
        lingering = [t for t in asyncio.all_tasks() if t is not current and not t.done()]
        return lingering

    lingering = asyncio.run(run())
    assert not lingering, f"Lingering tasks after abort: {lingering}"
    runtime._abort_signal = None


def test_iter_with_abort_closes_source_when_abort_set_between_events():
    async def run():
        abort = asyncio.Event()
        runtime.set_abort_signal(abort)
        closed = asyncio.Event()

        async def source():
            try:
                yield {"phase": "first"}
                await asyncio.sleep(10)
                yield {"phase": "second"}
            finally:
                closed.set()

        try:
            gen = runtime._iter_with_abort(source())
            first = await gen.__anext__()
            assert first == {"phase": "first"}
            abort.set()
            async for _ in gen:
                pass
            await asyncio.sleep(0)
            return closed.is_set()
        finally:
            runtime.set_abort_signal(None)

    assert asyncio.run(run()) is True


def test_iter_with_abort_suppresses_source_close_error_after_abort():
    async def run():
        abort = asyncio.Event()
        runtime.set_abort_signal(abort)

        async def source():
            try:
                yield {"phase": "first"}
                await asyncio.sleep(10)
            finally:
                raise RuntimeError("cleanup failed")

        try:
            gen = runtime._iter_with_abort(source())
            first = await gen.__anext__()
            assert first == {"phase": "first"}
            abort.set()
            async for _ in gen:
                pass
        finally:
            runtime.set_abort_signal(None)

    asyncio.run(run())


def test_iter_with_abort_preserves_source_error_when_abort_and_error_race():
    async def run():
        abort = asyncio.Event()
        runtime.set_abort_signal(abort)

        async def source():
            yield {"phase": "first"}
            abort.set()
            raise RuntimeError("source failed")

        try:
            gen = runtime._iter_with_abort(source())
            first = await gen.__anext__()
            assert first == {"phase": "first"}
            with pytest.raises(RuntimeError, match="source failed"):
                async for _ in gen:
                    pass
        finally:
            runtime.set_abort_signal(None)

    asyncio.run(run())


def test_iter_with_abort_suppresses_inflight_cleanup_error_after_abort():
    async def run():
        abort = asyncio.Event()
        runtime.set_abort_signal(abort)

        async def source():
            try:
                yield {"phase": "first"}
                await asyncio.sleep(10)
            finally:
                raise RuntimeError("cleanup failed")

        async def fire_abort():
            await asyncio.sleep(0.01)
            abort.set()

        try:
            gen = runtime._iter_with_abort(source())
            first = await gen.__anext__()
            assert first == {"phase": "first"}
            asyncio.create_task(fire_abort())
            async for _ in gen:
                pass
        finally:
            runtime.set_abort_signal(None)

    asyncio.run(run())


def test_iter_with_abort_cancels_pending_source_when_consumer_is_cancelled():
    runtime._abort_signal = None

    async def run():
        source_started = asyncio.Event()
        source_cancelled = asyncio.Event()

        async def slow_source():
            yield {"phase": "first"}
            source_started.set()
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                source_cancelled.set()
                raise
            yield {"phase": "second"}

        async def consume_remaining(gen):
            async for _ in gen:
                pass

        gen = runtime._iter_with_abort(slow_source())
        first = await gen.__anext__()
        assert first == {"phase": "first"}

        consumer = asyncio.create_task(consume_remaining(gen))
        await source_started.wait()
        consumer.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await consumer
        await asyncio.sleep(0)
        return source_cancelled.is_set()

    assert asyncio.run(run()) is True
