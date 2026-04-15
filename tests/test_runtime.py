import asyncio
from pathlib import Path
from typing import Coroutine, cast

import rooster_code.runtime as runtime
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

    assert [event.type for event in events] == [SDKMessageType.SYSTEM, SDKMessageType.SYSTEM, SDKMessageType.RESULT, SDKMessageType.SYSTEM]
    assert created[0].model == "m-fork"
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
    assert "Files:" not in text
    assert "Commands:" not in text
    assert "Findings:" not in text
    assert "Open issues:" not in text
    assert "Next step:" not in text


def test_run_subagent_summary_does_not_emit_outcome_none(monkeypatch) -> None:
    class FakeChildAgent:
        async def prompt(self, prompt: str):
            from open_agent_sdk.types import QueryResult

            return QueryResult(text="", messages=[])

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
    assert "Outcome: None" not in text
    assert "Outcome: No useful output returned" in text


def test_run_subagent_default_task_does_not_force_read_only_tool_subset(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeChildAgent:
        async def prompt(self, prompt: str):
            from open_agent_sdk.types import QueryResult

            return QueryResult(text="done", messages=[])

        async def close(self) -> None:
            return None

    def fake_create_sdk_agent(config, include_runtime_agent_tool=False, system_prompt=""):
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

    monkeypatch.setattr(runtime, "_create_sdk_agent", lambda config, include_runtime_agent_tool=False, system_prompt="": FakeChildAgent())

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
        assert "Outcome: background done" in tasks[task_id]["output"]

    asyncio.run(run_case())


def test_run_subagent_background_marks_failure_and_output(monkeypatch) -> None:
    class FakeChildAgent:
        async def prompt(self, prompt: str):
            raise RuntimeError("background exploded")

        async def close(self) -> None:
            return None

    monkeypatch.setattr(runtime, "_create_sdk_agent", lambda config, include_runtime_agent_tool=False, system_prompt="": FakeChildAgent())

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

    monkeypatch.setattr(runtime, "_create_sdk_agent", lambda config, include_runtime_agent_tool=False, system_prompt="": FakeChildAgent())
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
    assert "Outcome: background done" in str(note["output"])


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

    monkeypatch.setattr(runtime, "_create_sdk_agent", lambda config, include_runtime_agent_tool=False, system_prompt="": FakeChildAgent())
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
    assert note["output"] == "Outcome: background done"


def test_run_subagent_background_does_not_create_task_for_unknown_agent() -> None:
    async def run_case():
        result = await runtime._run_subagent(
            RuntimeConfig(model="m1"),
            {"name": "task1", "prompt": "check some files", "description": "task1", "run_in_background": True},
            ToolContext(cwd="/tmp/project", env={}),
        )
        assert result.is_error is True
        assert "unknown agent 'task1'" in str(result.content)
        assert get_all_tasks() == {}

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

    monkeypatch.setattr(runtime, "_create_sdk_agent", lambda config, include_runtime_agent_tool=False, system_prompt="": FakeChildAgent())

    async def run_case():
        abort_signal = asyncio.Event()
        runtime.set_abort_signal(abort_signal)

        async def trigger_abort():
            await asyncio.sleep(0)
            abort_signal.set()

        trigger_task = asyncio.create_task(trigger_abort())
        try:
            with pytest.raises(asyncio.CancelledError):
                await runtime._run_subagent(
                    RuntimeConfig(model="m1", agents={"builder": {"description": "build agent"}}),
                    {"name": "builder", "prompt": "do work", "description": "builder"},
                    ToolContext(cwd="/tmp/project", env={}),
                )
        finally:
            runtime.set_abort_signal(None)
            await trigger_task

        assert cancelled["value"] is True

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
    monkeypatch.setattr(runtime, "_create_sdk_agent", lambda config, include_runtime_agent_tool=False, system_prompt="": FakeChildAgent())

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

    monkeypatch.setattr(runtime, "_create_sdk_agent", lambda config, include_runtime_agent_tool=False, system_prompt="": SlowChildAgent())

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
    monkeypatch.setattr(runtime, "_create_sdk_agent", lambda config, include_runtime_agent_tool=False, system_prompt="": FakeChildAgent())

    result = asyncio.run(
        runtime._run_subagent(
            RuntimeConfig(model="m1", skills_dir="skills"),
            {"prompt": "Review last commit quality", "description": "Review last commit quality"},
            ToolContext(cwd="/tmp/project", env={}),
        )
    )

    assert captured == {"skill": "review", "args": "last commit quality"}
    assert "Outcome: reviewed the changes" in str(result.content)


def test_run_subagent_does_not_treat_planning_text_as_outcome(monkeypatch) -> None:
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
    monkeypatch.setattr(runtime, "_create_sdk_agent", lambda config, include_runtime_agent_tool=False, system_prompt="": FakeChildAgent())

    result = asyncio.run(
        runtime._run_subagent(
            RuntimeConfig(model="m1", skills_dir="skills"),
            {"name": "review", "prompt": "check last commit", "description": "review"},
            ToolContext(cwd="/tmp/project", env={}),
        )
    )

    assert result.is_error is False
    assert "Let me check" not in str(result.content)
    assert "Outcome: No useful output returned" in str(result.content)


def test_run_subagent_summary_falls_back_to_last_non_planning_assistant_text(monkeypatch) -> None:
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

    monkeypatch.setattr(runtime, "_create_sdk_agent", lambda config, include_runtime_agent_tool=False, system_prompt="": FakeChildAgent())

    result = asyncio.run(
        runtime._run_subagent(
            RuntimeConfig(model="m1", agents={"task": {"description": "review agent"}}),
            {"name": "task", "prompt": "review modules", "description": "task"},
            ToolContext(cwd="/tmp/project", env={}),
        )
    )

    text = str(result.content)
    assert "Let me inspect the code first." not in text
    assert "Outcome: reviewed the rendering and team modules" in text


def test_run_subagent_summary_uses_last_non_planning_text_when_result_text_is_planning(monkeypatch) -> None:
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

    monkeypatch.setattr(runtime, "_create_sdk_agent", lambda config, include_runtime_agent_tool=False, system_prompt="": FakeChildAgent())

    result = asyncio.run(
        runtime._run_subagent(
            RuntimeConfig(model="m1", agents={"task": {"description": "review agent"}}),
            {"name": "task", "prompt": "review modules", "description": "task"},
            ToolContext(cwd="/tmp/project", env={}),
        )
    )

    text = str(result.content)
    assert "Let me also check the rendering module" not in text
    assert "Outcome: reviewed the rendering module and team module and found the root cause" in text


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

    monkeypatch.setattr(runtime, "_create_sdk_agent", lambda config, include_runtime_agent_tool=False, system_prompt="": FakeChildAgent())

    result = asyncio.run(
        runtime._run_subagent(
            RuntimeConfig(model="m1", agents={"task": {"description": "review agent"}}),
            {"name": "task", "prompt": "review modules", "description": "task"},
            ToolContext(cwd="/tmp/project", env={}),
        )
    )

    text = str(result.content)
    assert "Let me check the actual module structure" not in text
    assert "Outcome: I found a critical issue!" in text


def test_run_subagent_summary_prefers_earlier_useful_text_over_later_planning(monkeypatch) -> None:
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

    monkeypatch.setattr(runtime, "_create_sdk_agent", lambda config, include_runtime_agent_tool=False, system_prompt="": FakeChildAgent())

    result = asyncio.run(
        runtime._run_subagent(
            RuntimeConfig(model="m1", agents={"task": {"description": "review agent"}}),
            {"name": "task", "prompt": "review tests", "description": "task"},
            ToolContext(cwd="/tmp/project", env={}),
        )
    )

    text = str(result.content)
    assert "Let me also inspect the remaining modules" not in text
    assert "Outcome: I reviewed the transport tests and found no critical issues." in text


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

    monkeypatch.setattr("rooster_code.runtime.create_agent", lambda options: FakeAgent())

    agent = create_runtime_agent(RuntimeConfig(api_key="test", base_url="https://nano-gpt.com/api/v1", model="m1"))

    asyncio.run(agent._initialize())

    assert [tool.name for tool in agent._tool_pool] == ["Read", "Edit", "Bash", "Agent"]
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

    assert [tool.name for tool in agent._tool_pool] == ["TeamCreate", "TeamDelete", "Bash", "Agent"]
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
    monkeypatch.setattr(runtime, "_create_sdk_agent", lambda config, include_runtime_agent_tool=False, system_prompt="": FakeAgent())

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

    monkeypatch.setattr("rooster_code.runtime.sdk_list_sessions", fake_list_sessions)
    monkeypatch.setattr("rooster_code.runtime.sdk_delete_session", fake_delete_session)

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
    from unittest.mock import AsyncMock

    tracker = TurnTracker()

    async def fake_runner(input, context):
        return ToolResult(tool_use_id="", content="should not be called")

    tool = RuntimeAgentTool(fake_runner, tracker)

    team_manager = TeamManager()
    team_manager._active = True
    team_manager._team_id = "test"
    team_manager._team_name = "test-team"
    pool = AgentPool()
    pool._members["reviewer"] = AsyncMock()
    pool._mailboxes["reviewer"] = asyncio.Queue()
    pool._locks["reviewer"] = asyncio.Lock()
    team_manager._pool = pool

    set_runtime_team_bridge(team_manager, None)
    try:
        import unittest.mock
        with unittest.mock.patch("rooster_code.runtime._create_background_subagent_task", new_callable=AsyncMock) as mock_create:
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
