import asyncio
import contextlib
from typing import Callable, Coroutine, cast
from prompt_toolkit import PromptSession

import rooster_code.cli as cli
from rooster_code.config import RuntimeConfig
from open_agent_sdk import SDKMessage, SDKMessageType
from rich.console import Console

from rooster_code.chat import parse_chat_command


def _fake_prompt_iter(prompts_iter):
    async def mock_prompt_async(*args, **kwargs):
        return next(prompts_iter)
    return mock_prompt_async

def _fake_prompt_keyboard_interrupt():
    async def mock_prompt_async(*args, **kwargs):
        raise KeyboardInterrupt()
    return mock_prompt_async


class SilentConsole:
    def print(self, *args, **kwargs) -> None:
        return None


def test_parse_model_command() -> None:
    command = parse_chat_command("/model claude-opus")

    assert command.name == "model"
    assert command.args == ["claude-opus"]


def test_run_chat_exits_cleanly(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeAgent:
        async def close(self) -> None:
            captured["closed"] = True

    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: FakeAgent())
    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())
    monkeypatch.setattr(PromptSession, "prompt_async", _fake_prompt_iter(iter(["/exit"])))

    exit_code = cli.asyncio.run(cli.run_chat(RuntimeConfig(model="m2")))

    assert exit_code == 0
    assert captured["closed"] is True


def test_run_chat_streams_user_prompt(monkeypatch) -> None:
    captured: dict[str, object] = {}
    prompts = iter(["hello", "/exit"])

    class FakeAgent:
        async def query(self, prompt: str):
            captured["prompt"] = prompt
            yield SDKMessage(type=SDKMessageType.ASSISTANT, text="hi")
            yield SDKMessage(type=SDKMessageType.RESULT, text="done")

        async def close(self) -> None:
            captured["closed"] = True

    async def fake_render_event_stream(console, events, omit_duplicate_result: bool = False, show_activity_trace: bool = False, **_kwargs) -> None:
        messages = []
        async for event in events:
            messages.append(event.type.value)
        captured["messages"] = messages
        captured["omit_duplicate_result"] = omit_duplicate_result
        captured["show_activity_trace"] = show_activity_trace

    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: FakeAgent())
    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())
    monkeypatch.setattr(cli, "render_event_stream", fake_render_event_stream)
    monkeypatch.setattr(PromptSession, "prompt_async", _fake_prompt_iter(prompts))

    exit_code = cli.asyncio.run(cli.run_chat(RuntimeConfig(model="m2")))

    assert exit_code == 0
    assert captured["prompt"] == "hello"
    assert captured["messages"] == ["assistant", "result"]
    assert captured["omit_duplicate_result"] is True
    assert captured["show_activity_trace"] is True
    assert captured["closed"] is True


def test_run_chat_requests_duplicate_result_omission(monkeypatch) -> None:
    captured: dict[str, object] = {}
    prompts = iter(["hello", "/exit"])

    class FakeAgent:
        async def query(self, prompt: str):
            yield SDKMessage(type=SDKMessageType.ASSISTANT, text="same")
            yield SDKMessage(type=SDKMessageType.RESULT, text="same")

        async def close(self) -> None:
            captured["closed"] = True

    async def fake_render_event_stream(console, events, omit_duplicate_result: bool = False, show_activity_trace: bool = False, **_kwargs) -> None:
        captured["omit_duplicate_result"] = omit_duplicate_result
        captured["show_activity_trace"] = show_activity_trace
        async for _event in events:
            pass

    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: FakeAgent())
    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())
    monkeypatch.setattr(cli, "render_event_stream", fake_render_event_stream)
    monkeypatch.setattr(PromptSession, "prompt_async", _fake_prompt_iter(prompts))

    exit_code = cli.asyncio.run(cli.run_chat(RuntimeConfig(model="m2")))

    assert exit_code == 0
    assert captured["omit_duplicate_result"] is True
    assert captured["show_activity_trace"] is True
    assert captured["closed"] is True


def test_run_chat_routes_explicit_agent_request(monkeypatch) -> None:
    captured: dict[str, object] = {}
    panels: list[tuple[str, str]] = []
    prompts = iter(["Use the reviewer agent to answer.", "/exit"])

    class FakeAgent:
        async def close(self) -> None:
            captured["closed"] = True

    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: FakeAgent())
    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())
    monkeypatch.setattr(PromptSession, "prompt_async", _fake_prompt_iter(prompts))
    monkeypatch.setattr(cli, "find_requested_agent_name", lambda config, prompt: "reviewer")

    async def fake_stream_named_agent_events(config, agent_name: str, prompt: str):
        captured["agent_name"] = agent_name
        captured["prompt"] = prompt
        yield SDKMessage(type=SDKMessageType.SYSTEM, text="starting")
        yield SDKMessage(type=SDKMessageType.RESULT, text="AGENT_PATH=used")

    async def fake_render_event_stream(console, events, omit_duplicate_result: bool = False, show_activity_trace: bool = False, **_kwargs) -> None:
        captured["omit_duplicate_result"] = omit_duplicate_result
        captured["show_activity_trace"] = show_activity_trace
        captured["messages"] = [event.type.value async for event in events]

    monkeypatch.setattr(cli, "stream_named_agent_events", fake_stream_named_agent_events)
    monkeypatch.setattr(cli, "render_event_stream", fake_render_event_stream)
    monkeypatch.setattr(cli, "render_agent_panel", lambda console, title, text, style: panels.append((title, text)))

    exit_code = cli.asyncio.run(
        cli.run_chat(RuntimeConfig(model="m2", agents={"reviewer": {"description": "reviewer"}}))
    )

    assert exit_code == 0
    assert captured["agent_name"] == "reviewer"
    assert captured["prompt"] == "Use the reviewer agent to answer."
    assert panels == [("Agent Started", "reviewer")]
    assert captured["omit_duplicate_result"] is True
    assert captured["show_activity_trace"] is True
    assert captured["messages"] == ["system", "result"]
    assert captured["closed"] is True


def test_run_chat_clears_agent_history(monkeypatch) -> None:
    captured: dict[str, object] = {}
    prompts = iter(["/clear", "/exit"])

    class FakeAgent:
        def clear(self) -> None:
            captured["cleared"] = True

        async def close(self) -> None:
            captured["closed"] = True

    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: FakeAgent())
    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())
    monkeypatch.setattr(PromptSession, "prompt_async", _fake_prompt_iter(prompts))

    exit_code = cli.asyncio.run(cli.run_chat(RuntimeConfig(model="m2")))

    assert exit_code == 0
    assert captured["cleared"] is True
    assert captured["closed"] is True


def test_run_chat_compacts_agent_history(monkeypatch) -> None:
    captured: dict[str, object] = {}
    prompts = iter(["/compact", "/exit"])
    notices: list[tuple[str, str, str]] = []

    class FakeAgent:
        async def close(self) -> None:
            captured["closed"] = True

    async def fake_compact_current_session(agent) -> dict[str, object]:
        captured["agent"] = agent
        return {
            "compacted": True,
            "summary": "summary text",
            "before_tokens": 1200,
            "after_tokens": 240,
            "reason": "",
        }

    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: FakeAgent())
    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())
    monkeypatch.setattr(cli, "compact_current_session", fake_compact_current_session, raising=False)
    monkeypatch.setattr(cli, "render_notice", lambda console, title, message, style="yellow": notices.append((title, message, style)))
    monkeypatch.setattr(PromptSession, "prompt_async", _fake_prompt_iter(prompts))

    exit_code = cli.asyncio.run(cli.run_chat(RuntimeConfig(model="m2")))

    assert exit_code == 0
    assert isinstance(captured["agent"], FakeAgent)
    assert captured["closed"] is True
    assert notices[0][0] == "Compacted"
    assert notices[0][2] == "green"
    assert "1200 → 240" in notices[0][1]
    assert "summary text" in notices[0][1]


def test_run_chat_shows_compact_error_when_compaction_fails(monkeypatch) -> None:
    captured: dict[str, object] = {}
    prompts = iter(["/compact", "/exit"])
    notices: list[tuple[str, str, str]] = []

    class FakeAgent:
        async def close(self) -> None:
            captured["closed"] = True

    async def fake_compact_current_session(agent) -> dict[str, object]:
        raise RuntimeError("Compaction failed.")

    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: FakeAgent())
    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())
    monkeypatch.setattr(cli, "compact_current_session", fake_compact_current_session, raising=False)
    monkeypatch.setattr(cli, "render_notice", lambda console, title, message, style="yellow": notices.append((title, message, style)))
    monkeypatch.setattr(PromptSession, "prompt_async", _fake_prompt_iter(prompts))

    exit_code = cli.asyncio.run(cli.run_chat(RuntimeConfig(model="m2")))

    assert exit_code == 0
    assert captured["closed"] is True
    assert notices[0] == ("Compact Error", "Compaction failed.", "red")


def test_run_chat_updates_model(monkeypatch) -> None:
    captured: dict[str, object] = {}
    prompts = iter(["/model new-model", "/exit"])

    class FakeAgent:
        async def set_model(self, model: str) -> None:
            captured["model"] = model

        async def close(self) -> None:
            captured["closed"] = True

    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: FakeAgent())
    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())
    monkeypatch.setattr(PromptSession, "prompt_async", _fake_prompt_iter(prompts))

    exit_code = cli.asyncio.run(cli.run_chat(RuntimeConfig(model="m2")))

    assert exit_code == 0
    assert captured["model"] == "new-model"
    assert captured["closed"] is True


def test_run_chat_updates_permission_mode(monkeypatch) -> None:
    captured: dict[str, object] = {}
    prompts = iter(["/permission acceptEdits", "/exit"])

    class FakeAgent:
        async def set_permission_mode(self, mode: str) -> None:
            captured["permission_mode"] = mode

        async def close(self) -> None:
            captured["closed"] = True

    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: FakeAgent())
    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())
    monkeypatch.setattr(PromptSession, "prompt_async", _fake_prompt_iter(prompts))

    config = RuntimeConfig(model="m2")
    exit_code = cli.asyncio.run(cli.run_chat(config))

    assert exit_code == 0
    assert captured["permission_mode"] == "acceptEdits"
    assert config.permission_mode == "acceptEdits"
    assert captured["closed"] is True


def test_run_chat_shows_tool_list(monkeypatch) -> None:
    captured: dict[str, object] = {}
    prompts = iter(["/tools", "/exit"])

    class FakeAgent:
        async def close(self) -> None:
            captured["closed"] = True

    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: FakeAgent())
    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())
    monkeypatch.setattr(cli, "list_tool_names", lambda: ["Read", "Write"])
    monkeypatch.setattr(cli, "render_tool_table", lambda console, tools: captured.setdefault("tools", tools))
    monkeypatch.setattr(PromptSession, "prompt_async", _fake_prompt_iter(prompts))

    exit_code = cli.asyncio.run(cli.run_chat(RuntimeConfig(model="m2")))

    assert exit_code == 0
    assert captured["tools"] == ["Read", "Write"]
    assert captured["closed"] is True


def test_run_chat_shows_task_list(monkeypatch) -> None:
    captured: dict[str, object] = {}
    prompts = iter(["/tasks", "/exit"])

    class FakeAgent:
        async def close(self) -> None:
            captured["closed"] = True

    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: FakeAgent())
    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())
    monkeypatch.setattr(cli, "get_state_snapshot", lambda name, agent_name=None: {"task_1": {"status": "completed"}} if name == "tasks" else {})
    monkeypatch.setattr(cli, "render_state", lambda console, title, data: captured.setdefault("state", (title, data)))
    monkeypatch.setattr(PromptSession, "prompt_async", _fake_prompt_iter(prompts))

    exit_code = cli.asyncio.run(cli.run_chat(RuntimeConfig(model="m2")))

    assert exit_code == 0
    assert captured["state"] == ("Tasks", {"task_1": {"status": "completed"}})
    assert captured["closed"] is True


def test_run_chat_shows_task_output(monkeypatch) -> None:
    captured: dict[str, object] = {}
    prompts = iter(["/task-output task_1", "/exit"])

    class FakeAgent:
        async def close(self) -> None:
            captured["closed"] = True

    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: FakeAgent())
    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())
    async def fake_get_task_output(task_id: str) -> str:
        return "done" if task_id == "task_1" else ""

    monkeypatch.setattr(cli, "get_task_output", fake_get_task_output)
    monkeypatch.setattr(cli, "render_state", lambda console, title, data: captured.setdefault("state", (title, data)))
    monkeypatch.setattr(PromptSession, "prompt_async", _fake_prompt_iter(prompts))

    exit_code = cli.asyncio.run(cli.run_chat(RuntimeConfig(model="m2")))

    assert exit_code == 0
    assert captured["state"] == ("Task Output", {"task_id": "task_1", "output": "done"})
    assert captured["closed"] is True


def test_run_chat_stops_task(monkeypatch) -> None:
    captured: dict[str, object] = {}
    prompts = iter(["/task-stop task_1", "/exit"])

    class FakeAgent:
        async def close(self) -> None:
            captured["closed"] = True

    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: FakeAgent())
    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())
    async def fake_stop_task(task_id: str) -> bool:
        captured.setdefault("stopped", task_id)
        return True

    monkeypatch.setattr(cli, "stop_task", fake_stop_task)
    monkeypatch.setattr(cli, "render_state", lambda console, title, data: captured.setdefault("state", (title, data)))
    monkeypatch.setattr(PromptSession, "prompt_async", _fake_prompt_iter(prompts))

    exit_code = cli.asyncio.run(cli.run_chat(RuntimeConfig(model="m2")))

    assert exit_code == 0
    assert captured["stopped"] == "task_1"
    assert captured["state"] == ("Task Stopped", {"task_id": "task_1", "stopped": True})
    assert captured["closed"] is True


def test_run_chat_shows_background_completion_notifications(monkeypatch) -> None:
    prompts = iter(["/exit"])
    rendered: list[str] = []

    class FakeAgent:
        def __init__(self) -> None:
            self._history: list[dict[str, object]] = []

        def get_session_id(self) -> str:
            return "session-1"

        async def close(self) -> None:
            return None

    agent = FakeAgent()
    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: agent)
    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())
    monkeypatch.setattr(
        cli,
        "read_background_notifications",
        lambda: [
            {
                "type": "background_task_completed",
                "task_id": "task_1",
                "status": "completed",
                "subject": "builder",
                "output": "Outcome: done",
            }
        ],
    )
    monkeypatch.setattr(cli, "print_formatted_text", lambda text: rendered.append("".join(fragment[1] for fragment in text)))
    monkeypatch.setattr(PromptSession, "prompt_async", _fake_prompt_iter(prompts))

    exit_code = cli.asyncio.run(cli.run_chat(RuntimeConfig(model="m2")))

    assert exit_code == 0
    assert "Background Task" in rendered[0]
    assert "╭─" in rendered[0]
    assert "builder (task_1) completed" in rendered[0]
    assert "Summary: done" in rendered[0]
    assert agent._history[-2] == {
        "role": "user",
        "content": [{"type": "text", "text": "[Background task task_1 completed]\n\nOutcome: done"}],
    }
    assert agent._history[-1] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "Received background task task_1 result. Ready to continue from that result without redoing the same delegated work."}],
    }


def test_render_task_notification_invalidates_prompt_session(monkeypatch) -> None:
    captured: dict[str, bool] = {"invalidated": False}
    rendered: list[str] = []

    class FakePromptApp:
        def invalidate(self) -> None:
            captured["invalidated"] = True

    class FakePromptSession:
        app = FakePromptApp()

    class FakeAgent:
        _history: list[dict[str, object]] = []

    monkeypatch.setattr(cli, "print_formatted_text", lambda text: rendered.append("".join(fragment[1] for fragment in text)))

    cli._render_task_notification(
        SilentConsole(),
        FakeAgent(),
        {
            "type": "background_task_completed",
            "task_id": "task_1",
            "status": "completed",
            "subject": "builder",
            "output": "Outcome: done",
        },
        cast(PromptSession[str], FakePromptSession()),
    )

    assert "Background Task" in rendered[0]
    assert "╭─" in rendered[0]
    assert "builder (task_1) completed" in rendered[0]
    assert "Summary: done" in rendered[0]
    assert captured["invalidated"] is True


def test_render_task_notification_uses_rich_notice_without_prompt_session(monkeypatch) -> None:
    notices: list[tuple[str, str, str]] = []

    class FakeAgent:
        _history: list[dict[str, object]] = []

    monkeypatch.setattr(cli, "render_notice", lambda console, title, message, style="yellow": notices.append((title, message, style)))

    cli._render_task_notification(
        SilentConsole(),
        FakeAgent(),
        {
            "type": "background_task_completed",
            "task_id": "task_1",
            "status": "completed",
            "subject": "builder",
            "output": "Outcome: done",
        },
        None,
    )

    assert notices[0] == (
        "Background Task",
        "builder (task_1) completed\nSummary: done\nFull output: /task-output task_1",
        "green",
    )


def test_render_task_notification_mentions_task_output_when_preview_truncated(monkeypatch) -> None:
    notices: list[tuple[str, str, str]] = []

    class FakeAgent:
        _history: list[dict[str, object]] = []

    monkeypatch.setattr(cli, "render_notice", lambda console, title, message, style="yellow": notices.append((title, message, style)))

    cli._render_task_notification(
        SilentConsole(),
        FakeAgent(),
        {
            "type": "background_task_completed",
            "task_id": "task_1",
            "status": "completed",
            "subject": "builder",
            "output": "line 1\nline 2\nline 3\nline 4\nline 5\nline 6\nline 7\nline 8\nline 9",
        },
        None,
    )

    assert "Full output: /task-output task_1" in notices[0][1]


def test_compact_task_output_picks_last_meaningful_line_from_review_report() -> None:
    output = (
        "Outcome: All tests pass. Here's my review:\n\n"
        "---\n\n"
        "## Code Review\n"
        "Lots of detailed follow-up analysis here"
    )

    summary = cli._compact_task_output(output)

    assert summary.startswith("Lots of detailed follow-up analysis here")


def test_compact_task_output_shows_last_meaningful_line() -> None:
    output = (
        "Outcome: Now I have a complete view of all the modified files.\n"
        "\n"
        "## Correctness Review\n"
        "Found no critical issues in the modified files."
    )

    summary = cli._compact_task_output(output)

    assert summary == "Found no critical issues in the modified files."


def test_compact_task_output_handles_long_single_line_output() -> None:
    output = (
        "Outcome: # Performance Review Report After reviewing all five files, I've identified several performance issues. "
        "Here's the detailed report: --- ## Critical Issues ### 1. runtime.py Line 479-486: Inefficient History Trimming"
    )

    summary = cli._compact_task_output(output)

    assert "# Performance Review Report" in summary
    assert "performance issues" in summary


def test_compact_task_output_shows_agent_output_with_now_prefix() -> None:
    output = "Outcome: Now I'll provide a detailed security analysis of the modified files."

    summary = cli._compact_task_output(output)

    assert "detailed security analysis" in summary


def test_compact_task_output_prefers_file_table_over_trailing_directory_line() -> None:
    output = (
        "**Top-level files:**\n\n"
        "| File | Size |\n"
        "|------|------|\n"
        "| `.env` | 205 bytes |\n"
        "| `pyproject.toml` | 566 bytes |\n\n"
        "**Top-level directories:** `src`, `tests`, `.venv`"
    )

    summary = cli._compact_task_output(output)

    assert ".env" in summary
    assert "pyproject.toml" in summary
    assert "Top-level directories" not in summary


def test_run_chat_shows_background_completion_before_prompt(monkeypatch) -> None:
    rendered: list[str] = []

    class FakeAgent:
        def __init__(self) -> None:
            self._history: list[dict[str, object]] = []

        def get_session_id(self) -> str:
            return "session-1"

        async def close(self) -> None:
            return None

    prompts = iter(["/exit"])

    notification_calls = iter([
        [{"type": "background_task_completed", "task_id": "task_1", "status": "completed", "subject": "builder", "output": "Outcome: done"}],
        [],
    ])

    agent = FakeAgent()
    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: agent)
    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())
    monkeypatch.setattr(cli, "read_background_notifications", lambda: next(notification_calls, []))
    monkeypatch.setattr(cli, "print_formatted_text", lambda text: rendered.append("".join(fragment[1] for fragment in text)))
    monkeypatch.setattr(PromptSession, "prompt_async", _fake_prompt_iter(prompts))

    exit_code = cli.asyncio.run(cli.run_chat(RuntimeConfig(model="m2")))

    assert exit_code == 0
    assert "Background Task" in rendered[0]
    assert "╭─" in rendered[0]
    assert "builder (task_1) completed" in rendered[0]
    assert "Summary: done" in rendered[0]
    assert agent._history[-2] == {
        "role": "user",
        "content": [{"type": "text", "text": "[Background task task_1 completed]\n\nOutcome: done"}],
    }
    assert agent._history[-1]["role"] == "assistant"


def test_run_chat_shows_background_completion_while_prompt_waits(monkeypatch) -> None:
    rendered: list[str] = []
    release_prompt = asyncio.Event()
    poll_count = 0

    class FakeAgent:
        def __init__(self) -> None:
            self._history: list[dict[str, object]] = []

        def get_session_id(self) -> str:
            return "session-1"

        async def close(self) -> None:
            return None

    async def fake_prompt_async(self, prompt_text: str, *args, **kwargs):
        await release_prompt.wait()
        return "/exit"

    def fake_read_background_notifications():
        nonlocal poll_count
        poll_count += 1
        if poll_count == 2:
            release_prompt.set()
            return [
                {
                    "type": "background_task_completed",
                    "task_id": "task_1",
                    "status": "completed",
                    "subject": "builder",
                    "output": "Outcome: done",
                }
            ]
        return []

    agent = FakeAgent()
    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: agent)
    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())
    monkeypatch.setattr(cli, "read_background_notifications", fake_read_background_notifications)
    monkeypatch.setattr(cli, "print_formatted_text", lambda text: rendered.append("".join(fragment[1] for fragment in text)))
    monkeypatch.setattr(PromptSession, "prompt_async", fake_prompt_async)

    exit_code = cli.asyncio.run(cli.run_chat(RuntimeConfig(model="m2")))

    assert exit_code == 0
    assert "Background Task" in rendered[0]
    assert "╭─" in rendered[0]
    assert "builder (task_1) completed" in rendered[0]
    assert "Summary: done" in rendered[0]
    assert poll_count >= 2
    assert agent._history[-2] == {
        "role": "user",
        "content": [{"type": "text", "text": "[Background task task_1 completed]\n\nOutcome: done"}],
    }
    assert agent._history[-1]["role"] == "assistant"


def test_run_chat_starts_background_agent_task(monkeypatch) -> None:
    captured: dict[str, object] = {}
    prompts = iter(["/agent-bg reviewer check last commit", "/exit"])

    class FakeAgent:
        async def close(self) -> None:
            captured["closed"] = True

    async def fake_start_background_agent_task(config, agent_name: str, prompt: str) -> str:
        captured["agent_name"] = agent_name
        captured["prompt"] = prompt
        return "task_1"

    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: FakeAgent())
    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())
    monkeypatch.setattr(cli, "start_background_agent_task", fake_start_background_agent_task)
    monkeypatch.setattr(cli, "render_state", lambda console, title, data: captured.setdefault("state", (title, data)))
    monkeypatch.setattr(PromptSession, "prompt_async", _fake_prompt_iter(prompts))

    exit_code = cli.asyncio.run(cli.run_chat(RuntimeConfig(model="m2")))

    assert exit_code == 0
    assert captured["agent_name"] == "reviewer"
    assert captured["prompt"] == "check last commit"
    assert captured["state"] == ("Background Task", {"agent": "reviewer", "task_id": "task_1"})
    assert captured["closed"] is True


def test_run_chat_starts_background_agent_task_with_bg_alias(monkeypatch) -> None:
    captured: dict[str, object] = {}
    prompts = iter(["/bg reviewer check last commit", "/exit"])

    class FakeAgent:
        async def close(self) -> None:
            captured["closed"] = True

    async def fake_start_background_agent_task(config, agent_name: str, prompt: str) -> str:
        captured["agent_name"] = agent_name
        captured["prompt"] = prompt
        return "task_1"

    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: FakeAgent())
    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())
    monkeypatch.setattr(cli, "start_background_agent_task", fake_start_background_agent_task)
    monkeypatch.setattr(cli, "render_state", lambda console, title, data: captured.setdefault("state", (title, data)))
    monkeypatch.setattr(PromptSession, "prompt_async", _fake_prompt_iter(prompts))

    exit_code = cli.asyncio.run(cli.run_chat(RuntimeConfig(model="m2")))

    assert exit_code == 0
    assert captured["agent_name"] == "reviewer"
    assert captured["prompt"] == "check last commit"
    assert captured["state"] == ("Background Task", {"agent": "reviewer", "task_id": "task_1"})
    assert captured["closed"] is True


def test_run_chat_waits_for_task_when_requested(monkeypatch) -> None:
    captured: dict[str, object] = {}
    prompts = iter(["/wait task_1", "/exit"])

    class FakeAgent:
        async def close(self) -> None:
            captured["closed"] = True

    async def fake_wait_for_task(task_id: str) -> dict[str, object]:
        captured["task_id"] = task_id
        return {"status": "completed", "output": "done"}

    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: FakeAgent())
    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())
    monkeypatch.setattr(cli, "wait_for_task", fake_wait_for_task)
    monkeypatch.setattr(cli, "render_state", lambda console, title, data: captured.setdefault("state", (title, data)))
    monkeypatch.setattr(PromptSession, "prompt_async", _fake_prompt_iter(prompts))

    exit_code = cli.asyncio.run(cli.run_chat(RuntimeConfig(model="m2")))

    assert exit_code == 0
    assert captured["task_id"] == "task_1"
    assert captured["state"] == ("Task Wait", {"task_id": "task_1", "status": "completed", "output": "done"})
    assert captured["closed"] is True


def test_run_chat_wait_injects_completed_task_output_into_context(monkeypatch) -> None:
    prompts = iter(["/wait task_1", "/exit"])

    class FakeAgent:
        def __init__(self) -> None:
            self._history: list[dict[str, object]] = []

        async def close(self) -> None:
            return None

    agent = FakeAgent()

    async def fake_wait_for_task(task_id: str) -> dict[str, object]:
        return {"status": "completed", "output": "Outcome: review complete"}

    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: agent)
    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())
    monkeypatch.setattr(cli, "wait_for_task", fake_wait_for_task)
    monkeypatch.setattr(cli, "render_state", lambda console, title, data: None)
    monkeypatch.setattr(PromptSession, "prompt_async", _fake_prompt_iter(prompts))

    exit_code = cli.asyncio.run(cli.run_chat(RuntimeConfig(model="m2")))

    assert exit_code == 0
    assert agent._history[-2] == {
        "role": "user",
        "content": [{"type": "text", "text": "[Background task task_1 completed]\n\nOutcome: review complete"}],
    }
    assert agent._history[-1] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "Received background task task_1 result. Ready to continue from that result without redoing the same delegated work."}],
    }


def test_append_task_result_to_context_marks_cancelled_tasks_as_cancelled() -> None:
    class FakeAgent:
        def __init__(self) -> None:
            self._history: list[dict[str, object]] = []

    cli._injected_task_ids.clear()
    agent = FakeAgent()

    cli.append_task_result_to_context(
        agent,
        "task_2",
        {"status": "cancelled", "output": "Error: failed to finish"},
    )

    assert agent._history[-2] == {
        "role": "user",
        "content": [{"type": "text", "text": "[Background task task_2 cancelled]\n\nError: failed to finish"}],
    }
    assert agent._history[-1] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "Received background task task_2 cancellation/failure. Do not assume the delegated work completed; decide whether to retry, take over, or change approach based on the failure output."}],
    }


def test_run_chat_deduplicates_task_injection(monkeypatch) -> None:
    cli._injected_task_ids.clear()
    prompts = iter(["/wait task_1", "/wait task_1", "/exit"])

    class FakeAgent:
        def __init__(self) -> None:
            self._history: list[dict[str, object]] = []

        async def close(self) -> None:
            return None

    agent = FakeAgent()

    async def fake_wait_for_task(task_id: str) -> dict[str, object]:
        return {"status": "completed", "output": "Outcome: done"}

    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: agent)
    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())
    monkeypatch.setattr(cli, "wait_for_task", fake_wait_for_task)
    monkeypatch.setattr(cli, "render_state", lambda console, title, data: None)
    monkeypatch.setattr(PromptSession, "prompt_async", _fake_prompt_iter(prompts))

    exit_code = cli.asyncio.run(cli.run_chat(RuntimeConfig(model="m2")))

    assert exit_code == 0
    injected_count = sum(1 for m in agent._history if m.get("role") == "user" and "task_1 completed" in str(m.get("content", "")))
    assert injected_count == 1
    captured: dict[str, object] = {}
    prompts = iter(["/agent-bg any check last commit", "/exit"])
    notices: list[tuple[str, str, str]] = []

    class FakeAgent:
        async def close(self) -> None:
            captured["closed"] = True

    async def fake_start_background_agent_task(config, agent_name: str, prompt: str) -> str:
        raise RuntimeError("Error: unknown agent 'any'")

    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: FakeAgent())
    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())
    monkeypatch.setattr(cli, "start_background_agent_task", fake_start_background_agent_task)
    monkeypatch.setattr(cli, "render_notice", lambda console, title, message, style="yellow": notices.append((title, message, style)))
    monkeypatch.setattr(PromptSession, "prompt_async", _fake_prompt_iter(prompts))

    exit_code = cli.asyncio.run(cli.run_chat(RuntimeConfig(model="m2")))

    assert exit_code == 0
    assert notices[0] == ("Error", "Error: unknown agent 'any'", "red")
    assert captured["closed"] is True


def test_run_chat_shows_skill_list(monkeypatch) -> None:
    captured: dict[str, object] = {}
    prompts = iter(["/skills", "/exit"])

    class FakeAgent:
        async def close(self) -> None:
            captured["closed"] = True

    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: FakeAgent())
    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())
    monkeypatch.setattr(cli, "list_skill_names", lambda: ["commit", "explain"])
    monkeypatch.setattr(cli, "render_state", lambda console, title, state: captured.setdefault("state", (title, state)))
    monkeypatch.setattr(PromptSession, "prompt_async", _fake_prompt_iter(prompts))

    exit_code = cli.asyncio.run(cli.run_chat(RuntimeConfig(model="m2")))

    assert exit_code == 0
    assert captured["state"] == ("Skills", {"skills": ["commit", "explain"]})
    assert captured["closed"] is True


def test_run_chat_routes_skill_command(monkeypatch) -> None:
    captured: dict[str, object] = {}
    prompts = iter(["/plan add auth support", "/exit"])
    panels: list[tuple[str, str]] = []

    class FakeAgent:
        async def close(self) -> None:
            captured["closed"] = True

    async def fake_stream_skill_events(config, agent, skill_name: str, args: str):
        captured["agent"] = agent
        captured["skill_name"] = skill_name
        captured["args"] = args
        yield SDKMessage(type=SDKMessageType.SYSTEM, text="skill-start")
        yield SDKMessage(type=SDKMessageType.RESULT, text="skill-result")

    async def fake_render_event_stream(console, events, omit_duplicate_result: bool = False, show_activity_trace: bool = False, **_kwargs) -> None:
        captured["omit_duplicate_result"] = omit_duplicate_result
        captured["show_activity_trace"] = show_activity_trace
        captured["messages"] = [event.type.value async for event in events]

    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: FakeAgent())
    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())
    monkeypatch.setattr(cli, "list_skill_names", lambda: ["plan", "commit"])
    monkeypatch.setattr(cli, "stream_skill_events", fake_stream_skill_events, raising=False)
    monkeypatch.setattr(cli, "render_event_stream", fake_render_event_stream)
    monkeypatch.setattr(cli, "render_agent_panel", lambda console, title, text, style: panels.append((title, text)))
    monkeypatch.setattr(PromptSession, "prompt_async", _fake_prompt_iter(prompts))

    exit_code = cli.asyncio.run(cli.run_chat(RuntimeConfig(model="m2")))

    assert exit_code == 0
    assert isinstance(captured["agent"], FakeAgent)
    assert captured["skill_name"] == "plan"
    assert captured["args"] == "add auth support"
    assert panels == [("Skill Started", "plan")]
    assert captured["omit_duplicate_result"] is True
    assert captured["show_activity_trace"] is True
    assert captured["messages"] == ["system", "result"]
    assert captured["closed"] is True


def test_run_chat_shows_unknown_skill_error(monkeypatch) -> None:
    captured: dict[str, object] = {}
    prompts = iter(["/unknown do thing", "/exit"])
    notices: list[tuple[str, str, str]] = []

    class FakeAgent:
        async def close(self) -> None:
            captured["closed"] = True

    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: FakeAgent())
    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())
    monkeypatch.setattr(cli, "list_skill_names", lambda: ["plan", "commit"])
    monkeypatch.setattr(cli, "render_notice", lambda console, title, message, style="yellow": notices.append((title, message, style)))
    monkeypatch.setattr(PromptSession, "prompt_async", _fake_prompt_iter(prompts))

    exit_code = cli.asyncio.run(cli.run_chat(RuntimeConfig(model="m2")))

    assert exit_code == 0
    assert notices[0] == ("Unknown command", "/unknown do thing", "red")
    assert captured["closed"] is True


def test_run_chat_shows_sessions(monkeypatch) -> None:
    captured: dict[str, object] = {}
    prompts = iter(["/sessions", "/exit"])

    class FakeAgent:
        async def close(self) -> None:
            captured["closed"] = True

    async def fake_list_sessions() -> list[dict[str, object]]:
        return [{"id": "sess-1"}]

    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: FakeAgent())
    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())
    monkeypatch.setattr(cli, "list_sessions", fake_list_sessions)
    monkeypatch.setattr(cli, "render_session_table", lambda console, sessions: captured.setdefault("count", len(sessions)))
    monkeypatch.setattr(PromptSession, "prompt_async", _fake_prompt_iter(prompts))

    exit_code = cli.asyncio.run(cli.run_chat(RuntimeConfig(model="m2")))

    assert exit_code == 0
    assert captured["count"] == 1
    assert captured["closed"] is True


def test_run_chat_resumes_a_different_session(monkeypatch) -> None:
    created: list[str | None] = []
    closed: list[str] = []
    bridge_agents: list[object | None] = []
    prompts = iter(["/resume sess-9", "/exit"])

    class FakeAgent:
        def __init__(self, name: str) -> None:
            self.name = name

        async def close(self) -> None:
            closed.append(self.name)

    def fake_create_runtime_agent(config: RuntimeConfig) -> FakeAgent:
        created.append(config.resume)
        return FakeAgent(config.resume or "initial")

    monkeypatch.setattr(cli, "create_runtime_agent", fake_create_runtime_agent)
    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())
    monkeypatch.setattr(PromptSession, "prompt_async", _fake_prompt_iter(prompts))
    monkeypatch.setattr(cli, "set_runtime_team_bridge", lambda manager, agent: bridge_agents.append(agent))
    async def fake_enforce_session_retention(limit=20):
        return None
    monkeypatch.setattr(cli, "enforce_session_retention", fake_enforce_session_retention)

    config = RuntimeConfig(model="m2", resume="start")
    exit_code = cli.asyncio.run(cli.run_chat(config))

    assert exit_code == 0
    assert created == ["start", "sess-9"]
    assert closed == ["start", "sess-9"]
    assert config.resume == "sess-9"
    assert len(bridge_agents) >= 1
    assert getattr(bridge_agents[0], "name", None) == "start"


def test_run_chat_resume_reapplies_team_state_to_new_agent(monkeypatch) -> None:
    prompts = iter(["/resume sess-9", "/exit"])
    ensure_calls: list[str] = []

    class FakeAgent:
        def __init__(self, name: str) -> None:
            self.name = name

        async def close(self) -> None:
            return None

    class FakeTeamManager:
        def __init__(self) -> None:
            self._active = True

        def is_active(self) -> bool:
            return self._active

        async def ensure_orchestrator_team_state(self, agent) -> None:
            ensure_calls.append(agent.name)

        async def clear(self) -> None:
            return None

        async def close_team(self, agent) -> None:
            self._active = False

    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: FakeAgent(config.resume or "initial"))
    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())
    monkeypatch.setattr(cli, "TeamManager", FakeTeamManager)
    monkeypatch.setattr(PromptSession, "prompt_async", _fake_prompt_iter(prompts))
    async def fake_enforce_session_retention(limit=20):
        return None
    monkeypatch.setattr(cli, "enforce_session_retention", fake_enforce_session_retention)

    exit_code = cli.asyncio.run(cli.run_chat(RuntimeConfig(model="m2", resume="start")))

    assert exit_code == 0
    assert "sess-9" in ensure_calls


def test_run_chat_help_renders_available_commands(monkeypatch) -> None:
    prompts = iter(["/help", "/exit"])
    console = Console(record=True, width=100)

    class FakeAgent:
        async def close(self) -> None:
            return None

    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: FakeAgent())
    monkeypatch.setattr(cli, "build_console", lambda: console)
    monkeypatch.setattr(PromptSession, "prompt_async", _fake_prompt_iter(prompts))

    exit_code = cli.asyncio.run(cli.run_chat(RuntimeConfig(model="m2")))

    assert exit_code == 0
    output = console.export_text()
    assert "Help" in output
    assert "/compact" in output
    assert "/skills" in output
    assert "/tasks" in output
    assert "/bg" in output
    assert "/agent-bg" in output
    assert "/wait" in output
    assert "/task-output" in output
    assert "/task-stop" in output
    assert "/exit" in output
    assert "/status" in output


def test_run_chat_status_renders_current_runtime_context(monkeypatch) -> None:
    captured: dict[str, object] = {}
    prompts = iter(["/status", "/exit"])

    class FakeAgent:
        async def close(self) -> None:
            captured["closed"] = True

    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: FakeAgent())
    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())
    monkeypatch.setattr(
        cli,
        "render_state",
        lambda console, title, data: captured.update({"title": title, "data": data}),
    )
    monkeypatch.setattr(PromptSession, "prompt_async", _fake_prompt_iter(prompts))

    config = RuntimeConfig(model="m2", resume="sess-9", permission_mode="acceptEdits")
    exit_code = cli.asyncio.run(cli.run_chat(config))

    assert exit_code == 0
    assert captured["title"] == "Chat Status"
    assert captured["data"] == {
        "model": "m2",
        "permission_mode": "acceptEdits",
        "session": "sess-9",
    }
    assert captured["closed"] is True


def test_run_chat_installs_and_clears_question_handler(monkeypatch) -> None:
    captured: list[str] = []
    prompts = iter(["/exit"])

    class FakeAgent:
        async def close(self) -> None:
            return None

    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: FakeAgent())
    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())
    monkeypatch.setattr(PromptSession, "prompt_async", _fake_prompt_iter(prompts))
    monkeypatch.setattr(cli, "set_question_handler", lambda handler: captured.append("set" if callable(handler) else "bad"))
    monkeypatch.setattr(cli, "clear_question_handler", lambda: captured.append("clear"))

    exit_code = cli.asyncio.run(cli.run_chat(RuntimeConfig(model="m2")))

    assert exit_code == 0
    assert captured == ["set", "clear"]


def test_run_chat_question_handler_uses_prompt_session(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def fake_prompt_async(self, prompt_text: str, *args, **kwargs):
        captured["prompt_text"] = prompt_text
        return "answer"

    monkeypatch.setattr(PromptSession, "prompt_async", fake_prompt_async)
    handler = cast(
        Callable[[str], Coroutine[object, object, str]],
        cli._create_question_handler(PromptSession(), asyncio.Event()),
    )

    answer = cli.asyncio.run(handler("Need input?"))
    assert answer == "answer"
    assert captured["prompt_text"] == "Need input? "


def test_run_chat_question_handler_cancels_on_abort_signal(monkeypatch) -> None:
    async def fake_prompt_async(self, prompt_text: str, *args, **kwargs):
        await asyncio.Future()

    monkeypatch.setattr(PromptSession, "prompt_async", fake_prompt_async)
    abort_signal = asyncio.Event()
    handler = cli._create_question_handler(PromptSession(), abort_signal)

    async def _run_handler() -> None:
        task = asyncio.create_task(handler("Need input?"))
        await asyncio.sleep(0)
        abort_signal.set()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    cli.asyncio.run(_run_handler())


def test_run_chat_unknown_command_shows_feedback(monkeypatch) -> None:
    prompts = iter(["/wat", "/exit"])
    console = Console(record=True, width=100)

    class FakeAgent:
        async def close(self) -> None:
            return None

    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: FakeAgent())
    monkeypatch.setattr(cli, "build_console", lambda: console)
    monkeypatch.setattr(PromptSession, "prompt_async", _fake_prompt_iter(prompts))

    exit_code = cli.asyncio.run(cli.run_chat(RuntimeConfig(model="m2")))

    assert exit_code == 0
    assert "Unknown command" in console.export_text()


def test_run_chat_returns_interrupt_exit_code(monkeypatch) -> None:
    """EOFError at the prompt exits chat with code 130."""
    captured: dict[str, object] = {}

    class FakeAgent:
        async def close(self) -> None:
            captured["closed"] = True

    async def fake_prompt_eof(*args, **kwargs):
        raise EOFError()

    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: FakeAgent())
    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())
    monkeypatch.setattr(PromptSession, "prompt_async", fake_prompt_eof)

    exit_code = cli.asyncio.run(cli.run_chat(RuntimeConfig(model="m2")))

    assert exit_code == 130
    assert captured["closed"] is True


def test_run_chat_interrupt_does_not_hang_on_slow_close(monkeypatch) -> None:
    """When interrupted (EOFError) with a slow agent.close(), the timeout prevents hanging."""
    class FakeAgent:
        async def close(self) -> None:
            await asyncio.Future()

    async def fake_prompt_eof(*args, **kwargs):
        raise EOFError()

    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: FakeAgent())
    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())
    monkeypatch.setattr(PromptSession, "prompt_async", fake_prompt_eof)

    exit_code = cli.asyncio.run(cli.asyncio.wait_for(cli.run_chat(RuntimeConfig(model="m2")), timeout=5))

    assert exit_code == 130


def test_run_chat_query_cancelled_by_interrupt_recovers(monkeypatch) -> None:
    prompts = iter(["hello", "/exit"])
    interrupted_once = {"count": 0}

    class FakeAgent:
        def __init__(self) -> None:
            self._history: list[dict[str, object]] = []

        async def query(self, prompt: str):
            if prompt == "hello":
                interrupted_once["count"] += 1
                raise asyncio.CancelledError()
            yield SDKMessage(type=SDKMessageType.RESULT, text="done")

        async def close(self) -> None:
            return None

    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: FakeAgent())
    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())
    monkeypatch.setattr(PromptSession, "prompt_async", _fake_prompt_iter(prompts))

    exit_code = cli.asyncio.run(cli.run_chat(RuntimeConfig(model="m2")))

    assert exit_code == 0
    assert interrupted_once["count"] == 1


def test_run_chat_sigint_cancels_active_query_task(monkeypatch) -> None:
    """Regression: _active_query_task must be visible to the SIGINT handler.

    A missing nonlocal caused _run_query_with_interrupt to shadow the outer
    _active_query_task, so the signal handler always saw None and never
    cancelled the in-flight query.
    """
    import signal

    prompts = iter(["hello", "/exit"])
    query_cancelled = {"count": 0}

    class FakeAgent:
        async def query(self, prompt: str):
            if prompt == "hello":
                try:
                    await asyncio.sleep(60)
                except asyncio.CancelledError:
                    query_cancelled["count"] += 1
                    raise
            yield SDKMessage(type=SDKMessageType.RESULT, text="done")

        async def close(self) -> None:
            return None

    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: FakeAgent())
    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())
    monkeypatch.setattr(PromptSession, "prompt_async", _fake_prompt_iter(prompts))

    async def _run_and_sigint():
        task = cli.asyncio.ensure_future(cli.run_chat(RuntimeConfig(model="m2")))
        await asyncio.sleep(0.1)
        import os
        os.kill(os.getpid(), signal.SIGINT)
        await asyncio.sleep(0.1)
        try:
            return await cli.asyncio.wait_for(task, timeout=2.0)
        except cli.asyncio.TimeoutError:
            task.cancel()
            return -1

    exit_code = cli.asyncio.run(_run_and_sigint())
    assert exit_code == 0
    assert query_cancelled["count"] == 1


def test_run_chat_interrupt_does_not_cancel_background_tasks(monkeypatch) -> None:
    prompts = iter(["/bg worker do stuff", "hello", "/exit"])
    bg_task_started = {"count": 0}

    class FakeAgent:
        async def query(self, prompt: str):
            if prompt == "hello":
                raise asyncio.CancelledError()
            yield SDKMessage(type=SDKMessageType.RESULT, text="done")

        async def close(self) -> None:
            return None

    async def fake_start_bg(config, agent_name: str, prompt: str) -> str:
        bg_task_started["count"] += 1
        return "task_bg1"

    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: FakeAgent())
    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())
    monkeypatch.setattr(cli, "start_background_agent_task", fake_start_bg)
    monkeypatch.setattr(cli, "render_state", lambda console, title, data: None)
    monkeypatch.setattr(PromptSession, "prompt_async", _fake_prompt_iter(prompts))

    exit_code = cli.asyncio.run(cli.run_chat(RuntimeConfig(model="m2")))

    assert exit_code == 0
    assert bg_task_started["count"] == 1


def test_render_event_stream_breaks_on_abort_signal() -> None:
    import io
    from rich.console import Console
    from rooster_code.rendering import render_event_stream
    from open_agent_sdk import SDKMessage, SDKMessageType

    abort_signal = cli.asyncio.Event()

    async def events():
        for i in range(100):
            yield SDKMessage(type=SDKMessageType.ASSISTANT, text=f"msg {i}")

    c = Console(file=io.StringIO(), width=80)

    async def _run():
        task = cli.asyncio.create_task(
            render_event_stream(c, events(), abort_signal=abort_signal)
        )
        await cli.asyncio.sleep(0.05)
        abort_signal.set()
        await task

    cli.asyncio.run(_run())
    assert abort_signal.is_set()


def test_run_chat_agents_list_command(monkeypatch) -> None:
    captured: dict[str, object] = {}
    prompts = iter(["/agents", "/exit"])

    class FakeAgent:
        async def close(self) -> None:
            captured["closed"] = True

    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: FakeAgent())
    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())
    monkeypatch.setattr(
        cli,
        "render_agents_list",
        lambda console, agents: captured.update({"agents": agents}),
    )
    monkeypatch.setattr(PromptSession, "prompt_async", _fake_prompt_iter(prompts))

    config = RuntimeConfig(model="m2", agents={"reviewer": {"description": "code reviewer"}})
    exit_code = cli.asyncio.run(cli.run_chat(config))

    assert exit_code == 0
    assert captured["agents"] == {"reviewer": {"description": "code reviewer"}}
    assert captured["closed"] is True


def test_run_chat_agents_add_command(monkeypatch) -> None:
    notices: list[tuple[str, str, str]] = []
    prompts = iter(["/agents add builder build things", "/exit"])

    class FakeAgent:
        async def close(self) -> None:
            pass

    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: FakeAgent())
    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())
    monkeypatch.setattr(
        cli,
        "render_notice",
        lambda console, title, message, style="yellow": notices.append((title, message, style)),
    )
    monkeypatch.setattr(PromptSession, "prompt_async", _fake_prompt_iter(prompts))

    config = RuntimeConfig(model="m2", agents={})
    exit_code = cli.asyncio.run(cli.run_chat(config))

    assert exit_code == 0
    assert ("Agent Added", "Agent 'builder' added.", "green") in notices
    assert "builder" in config.agents
    assert config.agents["builder"]["description"] == "build things"


def test_run_chat_agents_remove_command(monkeypatch) -> None:
    notices: list[tuple[str, str, str]] = []
    prompts = iter(["/agents remove reviewer", "/exit"])

    class FakeAgent:
        async def close(self) -> None:
            pass

    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: FakeAgent())
    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())
    monkeypatch.setattr(
        cli,
        "render_notice",
        lambda console, title, message, style="yellow": notices.append((title, message, style)),
    )
    monkeypatch.setattr(PromptSession, "prompt_async", _fake_prompt_iter(prompts))

    config = RuntimeConfig(model="m2", agents={"reviewer": {"description": "code reviewer"}})
    exit_code = cli.asyncio.run(cli.run_chat(config))

    assert exit_code == 0
    assert ("Agent Removed", "Agent 'reviewer' removed.", "green") in notices
    assert "reviewer" not in config.agents


def test_run_chat_agents_show_command(monkeypatch) -> None:
    captured: dict[str, object] = {}
    prompts = iter(["/agents show reviewer", "/exit"])

    class FakeAgent:
        async def close(self) -> None:
            pass

    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: FakeAgent())
    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())
    monkeypatch.setattr(
        cli,
        "render_state",
        lambda console, title, data: captured.update({"title": title, "data": data}),
    )
    monkeypatch.setattr(PromptSession, "prompt_async", _fake_prompt_iter(prompts))

    config = RuntimeConfig(model="m2", agents={"reviewer": {"description": "code reviewer", "prompt": "You are a reviewer."}})
    exit_code = cli.asyncio.run(cli.run_chat(config))

    assert exit_code == 0
    assert captured["title"] == "Agent: reviewer"


def test_run_chat_agents_remove_in_team_fails(monkeypatch) -> None:
    notices: list[tuple[str, str, str]] = []
    prompts = iter(["/agents remove reviewer", "/exit"])

    class FakeAgent:
        async def close(self) -> None:
            pass

    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: FakeAgent())
    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())
    monkeypatch.setattr(
        cli,
        "render_notice",
        lambda console, title, message, style="yellow": notices.append((title, message, style)),
    )
    monkeypatch.setattr(PromptSession, "prompt_async", _fake_prompt_iter(prompts))

    config = RuntimeConfig(model="m2", agents={"reviewer": {"description": "code reviewer"}})
    exit_code = cli.asyncio.run(cli.run_chat(config))

    assert exit_code == 0
    assert "reviewer" not in config.agents


def test_run_chat_clear_clears_team_histories(monkeypatch) -> None:
    cleared: dict[str, object] = {}
    prompts = iter(["/clear", "/exit"])

    class FakeAgent:
        def clear(self) -> None:
            cleared["agent_cleared"] = True

        async def close(self) -> None:
            cleared["closed"] = True

    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: FakeAgent())
    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())
    monkeypatch.setattr(
        cli,
        "render_notice",
        lambda console, title, message, style="yellow": None,
    )
    monkeypatch.setattr(PromptSession, "prompt_async", _fake_prompt_iter(prompts))

    exit_code = cli.asyncio.run(cli.run_chat(RuntimeConfig(model="m2")))

    assert exit_code == 0
    assert cleared.get("agent_cleared") is True


def test_run_chat_reapplies_team_state_after_interrupt(monkeypatch) -> None:
    prompts = iter(["hello", "/exit"])
    captured: dict[str, int] = {"ensure_calls": 0}

    class FakeAgent:
        def __init__(self) -> None:
            self._initialized = False
            self._tool_pool = []

        async def query(self, prompt: str):
            if prompt == "hello":
                raise asyncio.CancelledError()
            if False:
                yield None

        async def close(self) -> None:
            return None

    class FakeTeamManager:
        def is_active(self) -> bool:
            return True

        async def ensure_orchestrator_team_state(self, agent) -> None:
            captured["ensure_calls"] += 1

        async def clear(self) -> None:
            return None

        async def close_team(self, agent) -> None:
            return None

    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: FakeAgent())
    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())
    monkeypatch.setattr(cli, "TeamManager", FakeTeamManager)
    monkeypatch.setattr(PromptSession, "prompt_async", _fake_prompt_iter(prompts))

    exit_code = cli.asyncio.run(cli.run_chat(RuntimeConfig(model="m2")))

    assert exit_code == 0
    assert captured["ensure_calls"] == 2


def test_compact_task_output_skips_closing_xml_tags() -> None:
    output = (
        "<results>\n"
        "Found 3 issues in the module\n"
        "1. Race condition on global state\n"
        "2. Missing error handling in search backend\n"
        "3. Silent exception suppression\n"
        "</results>"
    )

    summary = cli._compact_task_output(output)

    assert "3. Silent exception suppression" in summary
    assert "</results>" not in summary
    assert "<results>" not in summary
