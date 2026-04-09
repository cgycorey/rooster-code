import asyncio

import cock_code.cli as cli
from cock_code.config import RuntimeConfig
from open_agent_sdk import SDKMessage, SDKMessageType
from rich.console import Console

from cock_code.chat import parse_chat_command


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
    monkeypatch.setattr(cli.Prompt, "ask", lambda *args, **kwargs: "/exit")

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

    async def fake_render_event_stream(console, events, omit_duplicate_result: bool = False, show_activity_trace: bool = False) -> None:
        messages = []
        async for event in events:
            messages.append(event.type.value)
        captured["messages"] = messages
        captured["omit_duplicate_result"] = omit_duplicate_result
        captured["show_activity_trace"] = show_activity_trace

    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: FakeAgent())
    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())
    monkeypatch.setattr(cli, "render_event_stream", fake_render_event_stream)
    monkeypatch.setattr(cli.Prompt, "ask", lambda *args, **kwargs: next(prompts))

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

    async def fake_render_event_stream(console, events, omit_duplicate_result: bool = False, show_activity_trace: bool = False) -> None:
        captured["omit_duplicate_result"] = omit_duplicate_result
        captured["show_activity_trace"] = show_activity_trace
        async for _event in events:
            pass

    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: FakeAgent())
    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())
    monkeypatch.setattr(cli, "render_event_stream", fake_render_event_stream)
    monkeypatch.setattr(cli.Prompt, "ask", lambda *args, **kwargs: next(prompts))

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
    monkeypatch.setattr(cli.Prompt, "ask", lambda *args, **kwargs: next(prompts))
    monkeypatch.setattr(cli, "find_requested_agent_name", lambda config, prompt: "reviewer")

    async def fake_stream_named_agent_events(config, agent_name: str, prompt: str):
        captured["agent_name"] = agent_name
        captured["prompt"] = prompt
        yield SDKMessage(type=SDKMessageType.SYSTEM, text="starting")
        yield SDKMessage(type=SDKMessageType.RESULT, text="AGENT_PATH=used")

    async def fake_render_event_stream(console, events, omit_duplicate_result: bool = False, show_activity_trace: bool = False) -> None:
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
    monkeypatch.setattr(cli.Prompt, "ask", lambda *args, **kwargs: next(prompts))

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
    monkeypatch.setattr(cli.Prompt, "ask", lambda *args, **kwargs: next(prompts))

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
    monkeypatch.setattr(cli.Prompt, "ask", lambda *args, **kwargs: next(prompts))

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
    monkeypatch.setattr(cli.Prompt, "ask", lambda *args, **kwargs: next(prompts))

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
    monkeypatch.setattr(cli.Prompt, "ask", lambda *args, **kwargs: next(prompts))

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
    monkeypatch.setattr(cli.Prompt, "ask", lambda *args, **kwargs: next(prompts))

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
    monkeypatch.setattr(cli.Prompt, "ask", lambda *args, **kwargs: next(prompts))

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
    monkeypatch.setattr(cli.Prompt, "ask", lambda *args, **kwargs: next(prompts))

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
    monkeypatch.setattr(cli.Prompt, "ask", lambda *args, **kwargs: next(prompts))

    exit_code = cli.asyncio.run(cli.run_chat(RuntimeConfig(model="m2")))

    assert exit_code == 0
    assert captured["stopped"] == "task_1"
    assert captured["state"] == ("Task Stopped", {"task_id": "task_1", "stopped": True})
    assert captured["closed"] is True


def test_run_chat_shows_background_completion_notifications(monkeypatch) -> None:
    prompts = iter(["/exit"])
    notices: list[tuple[str, str, str]] = []

    class FakeAgent:
        def get_session_id(self) -> str:
            return "session-1"

        async def close(self) -> None:
            return None

    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: FakeAgent())
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
    monkeypatch.setattr(cli, "render_notice", lambda console, title, message, style="yellow": notices.append((title, message, style)))
    monkeypatch.setattr(cli.Prompt, "ask", lambda *args, **kwargs: next(prompts))

    exit_code = cli.asyncio.run(cli.run_chat(RuntimeConfig(model="m2")))

    assert exit_code == 0
    assert notices[0] == ("Background Task", "builder (task_1) completed\nOutcome: done", "green")


def test_run_chat_shows_background_completion_while_waiting_for_input(monkeypatch) -> None:
    import threading

    notices: list[tuple[str, str, str]] = []
    release_prompt = threading.Event()
    observed: dict[str, object] = {"notified_before_return": False}

    class FakeAgent:
        def get_session_id(self) -> str:
            return "session-1"

        async def close(self) -> None:
            return None

    def fake_prompt(_label: str) -> str:
        observed["notified_before_return"] = release_prompt.wait(0.2)
        return "/exit"

    notification_calls = iter([
        [{"type": "background_task_completed", "task_id": "task_1", "status": "completed", "subject": "builder", "output": "Outcome: done"}],
        [],
    ])

    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: FakeAgent())
    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())
    def fake_read_notifications():
        notes = next(notification_calls, [])
        if notes:
            release_prompt.set()
        return notes

    monkeypatch.setattr(cli, "read_background_notifications", fake_read_notifications)
    monkeypatch.setattr(cli, "render_notice", lambda console, title, message, style="yellow": notices.append((title, message, style)))
    monkeypatch.setattr(cli.Prompt, "ask", fake_prompt)

    exit_code = cli.asyncio.run(cli.run_chat(RuntimeConfig(model="m2")))

    assert exit_code == 0
    assert observed["notified_before_return"] is True
    assert notices[0] == ("Background Task", "builder (task_1) completed\nOutcome: done", "green")


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
    monkeypatch.setattr(cli.Prompt, "ask", lambda *args, **kwargs: next(prompts))

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
    monkeypatch.setattr(cli.Prompt, "ask", lambda *args, **kwargs: next(prompts))

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
    monkeypatch.setattr(cli.Prompt, "ask", lambda *args, **kwargs: next(prompts))

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
    monkeypatch.setattr(cli.Prompt, "ask", lambda *args, **kwargs: next(prompts))

    exit_code = cli.asyncio.run(cli.run_chat(RuntimeConfig(model="m2")))

    assert exit_code == 0
    assert cli._pending_task_results[-1] == ("task_1", {"status": "completed", "output": "Outcome: review complete"})


def test_run_chat_handles_background_agent_task_error(monkeypatch) -> None:
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
    monkeypatch.setattr(cli.Prompt, "ask", lambda *args, **kwargs: next(prompts))

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
    monkeypatch.setattr(cli.Prompt, "ask", lambda *args, **kwargs: next(prompts))

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

    async def fake_render_event_stream(console, events, omit_duplicate_result: bool = False, show_activity_trace: bool = False) -> None:
        captured["omit_duplicate_result"] = omit_duplicate_result
        captured["show_activity_trace"] = show_activity_trace
        captured["messages"] = [event.type.value async for event in events]

    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: FakeAgent())
    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())
    monkeypatch.setattr(cli, "list_skill_names", lambda: ["plan", "commit"])
    monkeypatch.setattr(cli, "stream_skill_events", fake_stream_skill_events, raising=False)
    monkeypatch.setattr(cli, "render_event_stream", fake_render_event_stream)
    monkeypatch.setattr(cli, "render_agent_panel", lambda console, title, text, style: panels.append((title, text)))
    monkeypatch.setattr(cli.Prompt, "ask", lambda *args, **kwargs: next(prompts))

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
    monkeypatch.setattr(cli.Prompt, "ask", lambda *args, **kwargs: next(prompts))

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
    monkeypatch.setattr(cli.Prompt, "ask", lambda *args, **kwargs: next(prompts))

    exit_code = cli.asyncio.run(cli.run_chat(RuntimeConfig(model="m2")))

    assert exit_code == 0
    assert captured["count"] == 1
    assert captured["closed"] is True


def test_run_chat_resumes_a_different_session(monkeypatch) -> None:
    created: list[str | None] = []
    closed: list[str] = []
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
    monkeypatch.setattr(cli.Prompt, "ask", lambda *args, **kwargs: next(prompts))

    config = RuntimeConfig(model="m2", resume="start")
    exit_code = cli.asyncio.run(cli.run_chat(config))

    assert exit_code == 0
    assert created == ["start", "sess-9"]
    assert closed == ["start", "sess-9"]
    assert config.resume == "sess-9"


def test_run_chat_help_renders_available_commands(monkeypatch) -> None:
    prompts = iter(["/help", "/exit"])
    console = Console(record=True, width=100)

    class FakeAgent:
        async def close(self) -> None:
            return None

    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: FakeAgent())
    monkeypatch.setattr(cli, "build_console", lambda: console)
    monkeypatch.setattr(cli.Prompt, "ask", lambda *args, **kwargs: next(prompts))

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
    monkeypatch.setattr(cli.Prompt, "ask", lambda *args, **kwargs: next(prompts))

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
    monkeypatch.setattr(cli.Prompt, "ask", lambda *args, **kwargs: next(prompts))
    monkeypatch.setattr(cli, "set_question_handler", lambda handler: captured.append("set" if callable(handler) else "bad"))
    monkeypatch.setattr(cli, "clear_question_handler", lambda: captured.append("clear"))

    exit_code = cli.asyncio.run(cli.run_chat(RuntimeConfig(model="m2")))

    assert exit_code == 0
    assert captured == ["set", "clear"]


def test_run_chat_unknown_command_shows_feedback(monkeypatch) -> None:
    prompts = iter(["/wat", "/exit"])
    console = Console(record=True, width=100)

    class FakeAgent:
        async def close(self) -> None:
            return None

    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: FakeAgent())
    monkeypatch.setattr(cli, "build_console", lambda: console)
    monkeypatch.setattr(cli.Prompt, "ask", lambda *args, **kwargs: next(prompts))

    exit_code = cli.asyncio.run(cli.run_chat(RuntimeConfig(model="m2")))

    assert exit_code == 0
    assert "Unknown command" in console.export_text()


def test_run_chat_returns_interrupt_exit_code(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeAgent:
        async def close(self) -> None:
            captured["closed"] = True

    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: FakeAgent())
    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())
    monkeypatch.setattr(cli.Prompt, "ask", lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()))

    exit_code = cli.asyncio.run(cli.run_chat(RuntimeConfig(model="m2")))

    assert exit_code == 130
    assert captured["closed"] is True


def test_run_chat_interrupt_does_not_hang_on_slow_close(monkeypatch) -> None:
    class FakeAgent:
        async def close(self) -> None:
            await asyncio.Future()

    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: FakeAgent())
    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())
    monkeypatch.setattr(cli.Prompt, "ask", lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()))

    exit_code = cli.asyncio.run(cli.asyncio.wait_for(cli.run_chat(RuntimeConfig(model="m2")), timeout=0.2))

    assert exit_code == 130
