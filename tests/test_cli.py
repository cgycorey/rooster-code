import asyncio
import contextlib
import json
from rooster_code.cli import build_parser


import importlib
import signal
import sys
from typing import Callable, Coroutine, cast

import rooster_code.cli as cli
from rooster_code.config import RuntimeConfig
from open_agent_sdk import SDKMessage, SDKMessageType
from prompt_toolkit import PromptSession


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


def test_help_includes_top_level_commands() -> None:
    parser = build_parser()
    help_text = parser.format_help()
    assert "usage: rooster-code" in help_text
    assert "ask" in help_text
    assert "chat" in help_text
    assert "sessions" in help_text
    assert "tools" in help_text
    assert "state" in help_text


def test_ask_subcommand_help_mentions_rooster_code_env_override() -> None:
    parser = build_parser()
    subparsers = next(action for action in parser._actions if getattr(action, "choices", None))
    ask_help = subparsers.choices["ask"].format_help()

    assert "Override ROOSTER_CODE_MODEL for this run" in ask_help


def test_ask_command_accepts_shared_runtime_flags() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "ask",
            "hello",
            "--model",
            "m1",
            "--resume",
            "sess-1",
            "--allowed-tool",
            "Read",
            "--disallowed-tool",
            "Bash",
            "--search-url",
            "http://127.0.0.1:8080/search",
            "--skills-dir",
            "/tmp/skills",
        ]
    )

    assert args.command == "ask"
    assert args.prompt == "hello"
    assert args.model == "m1"
    assert args.resume == "sess-1"
    assert args.allowed_tools == ["Read"]
    assert args.disallowed_tools == ["Bash"]
    assert args.search_url == "http://127.0.0.1:8080/search"
    assert args.skills_dir == "/tmp/skills"


def test_main_dispatches_ask(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def fake_run_ask(prompt: str, config) -> int:
        captured["prompt"] = prompt
        captured["model"] = config.model
        return 0

    monkeypatch.setattr(cli, "run_ask", fake_run_ask)

    exit_code = cli.main(["ask", "hello", "--model", "m1"])

    assert exit_code == 0
    assert captured == {"prompt": "hello", "model": "m1"}


def test_ask_via_daemon_forwards_runtime_file_overrides(tmp_path, monkeypatch) -> None:
    import rooster_code.cli_daemon as cli_daemon

    captured: dict[str, object] = {}
    agents_file = tmp_path / "agents.json"
    hooks_file = tmp_path / "hooks.json"
    schema_file = tmp_path / "schema.json"
    mcp_file = tmp_path / "mcp.json"
    extra_file = tmp_path / "extra.json"
    skills_dir = tmp_path / "skills"
    agents_file.write_text(json.dumps({"reviewer": {"description": "reviews"}}), encoding="utf-8")
    hooks_file.write_text(json.dumps({"PreToolUse": []}), encoding="utf-8")
    schema_file.write_text(json.dumps({"type": "object"}), encoding="utf-8")
    mcp_file.write_text(json.dumps({"fs": {"type": "stdio", "command": "echo"}}), encoding="utf-8")
    extra_file.write_text(json.dumps({"temperature": 0}), encoding="utf-8")
    skills_dir.mkdir()

    async def fake_daemon_query(prompt: str, *, session_id: str = "", cwd: str = ".", overrides=None):
        captured["prompt"] = prompt
        captured["session_id"] = session_id
        captured["cwd"] = cwd
        captured["overrides"] = overrides
        return {"type": "done", "text": "ok"}

    parser = build_parser()
    args = parser.parse_args([
        "ask",
        "hello",
        "--daemon",
        "--search-url",
        "https://search.example.test",
        "--agents-file",
        str(agents_file),
        "--hooks-file",
        str(hooks_file),
        "--json-schema-file",
        str(schema_file),
        "--mcp-file",
        str(mcp_file),
        "--extra-args-file",
        str(extra_file),
        "--skills-dir",
        str(skills_dir),
    ])
    monkeypatch.setattr(cli_daemon, "daemon_query", fake_daemon_query)

    assert cli_daemon._ask_via_daemon(args.prompt, args) == 0

    overrides = captured["overrides"]
    assert overrides["search_url"] == "https://search.example.test"
    assert overrides["agents"] == {"reviewer": {"description": "reviews"}}
    assert overrides["hooks"] == {"PreToolUse": []}
    assert overrides["json_schema"] == {"type": "object"}
    assert overrides["mcp_servers"] == {"fs": {"type": "stdio", "command": "echo"}}
    assert overrides["extra_args"] == {"temperature": 0}
    assert overrides["skills_dir"] == str(skills_dir)


def test_main_agents_list_loads_configured_agents(tmp_path, monkeypatch) -> None:
    import rooster_code.config as config_mod

    captured: dict[str, object] = {}
    agents_file = tmp_path / "agents.json"
    agents_file.write_text(json.dumps({"reviewer": {"description": "reviews"}}), encoding="utf-8")

    monkeypatch.setattr(config_mod, "DEFAULT_AGENTS_PATH", agents_file)
    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())
    monkeypatch.setattr(
        cli,
        "render_agents_list",
        lambda console, agents: captured.update({"agents": agents}),
    )

    assert cli.main(["agents", "list"]) == 0
    assert captured["agents"] == {"reviewer": {"description": "reviews"}}


def test_run_ask_streams_events_and_closes_agent(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeAgent:
        async def query(self, prompt: str):
            captured["prompt"] = prompt
            yield SDKMessage(type=SDKMessageType.ASSISTANT, text="hello")
            yield SDKMessage(type=SDKMessageType.RESULT, text="done")

        async def close(self) -> None:
            captured["closed"] = True

    def fake_create_runtime_agent(config: RuntimeConfig) -> FakeAgent:
        captured["config"] = config
        return FakeAgent()

    async def fake_render_event_stream(console, events, omit_duplicate_result: bool = False, show_activity_trace: bool = False, **_kwargs) -> None:
        messages = []
        async for event in events:
            messages.append(event.type.value)
        captured["messages"] = messages
        captured["omit_duplicate_result"] = omit_duplicate_result
        captured["show_activity_trace"] = show_activity_trace

    monkeypatch.setattr(cli, "create_runtime_agent", fake_create_runtime_agent)
    monkeypatch.setattr(cli, "render_event_stream", fake_render_event_stream)
    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())

    exit_code = cli.asyncio.run(cli.run_ask("hello", RuntimeConfig(model="m1")))

    assert exit_code == 0
    assert captured["prompt"] == "hello"
    assert captured["messages"] == ["assistant", "result"]
    assert captured["omit_duplicate_result"] is True
    assert captured["show_activity_trace"] is False
    assert captured["closed"] is True


def test_main_dispatches_chat(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def fake_run_chat(config) -> int:
        captured["model"] = config.model
        return 0

    monkeypatch.setattr(cli, "run_chat", fake_run_chat)

    exit_code = cli.main(["chat", "--model", "m2"])

    assert exit_code == 0
    assert captured == {"model": "m2"}


def test_main_chat_handoff_command_uses_cli_cwd(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}
    prompts = iter(["/handoff", "/exit"])
    notices: list[tuple[str, str, str]] = []

    class FakeAgent:
        async def close(self) -> None:
            captured["closed"] = True

    async def fake_handoff_current_session(agent, path: str | None = None) -> dict[str, object]:
        captured["agent"] = agent
        captured["path"] = path
        return {
            "compacted": True,
            "written": True,
            "path": path or "",
            "summary": "summary text",
            "before_tokens": 1200,
            "after_tokens": 240,
            "reason": "",
        }

    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: FakeAgent())
    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())
    monkeypatch.setattr(cli, "handoff_current_session", fake_handoff_current_session, raising=False)
    monkeypatch.setattr(cli, "render_notice", lambda console, title, message, style="yellow": notices.append((title, message, style)))
    monkeypatch.setattr(PromptSession, "prompt_async", _fake_prompt_iter(prompts))

    exit_code = cli.main(["chat", "--model", "m2", "--cwd", str(tmp_path), "--no-persist-session"])

    assert exit_code == 0
    assert isinstance(captured["agent"], FakeAgent)
    assert captured["path"] == str(tmp_path / ".handoff")
    assert captured["closed"] is True
    assert notices[0][0] == "Handoff"
    assert notices[0][2] == "green"
    assert f"Saved {tmp_path / '.handoff'}" in notices[0][1]
    assert "1200 → 240" in notices[0][1]


def test_main_chat_handoff_with_relative_path_via_cli_cwd(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}
    prompts = iter(["/handoff custom.handoff", "/exit"])

    class FakeAgent:
        async def close(self) -> None:
            captured["closed"] = True

    async def fake_handoff_current_session(agent, path: str | None = None) -> dict[str, object]:
        captured["path"] = path
        return {
            "compacted": True,
            "written": True,
            "path": path or "",
            "summary": "summary text",
            "before_tokens": 12,
            "after_tokens": 4,
            "reason": "",
        }

    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: FakeAgent())
    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())
    monkeypatch.setattr(cli, "handoff_current_session", fake_handoff_current_session, raising=False)
    monkeypatch.setattr(cli, "render_notice", lambda *args, **kwargs: None)
    monkeypatch.setattr(PromptSession, "prompt_async", _fake_prompt_iter(prompts))

    exit_code = cli.main(["chat", "--model", "m2", "--cwd", str(tmp_path), "--no-persist-session"])

    assert exit_code == 0
    assert captured["path"] == str(tmp_path / "custom.handoff")


def test_main_chat_handoff_with_absolute_path_via_cli(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}
    abs_path = str(tmp_path / "absolute.handoff")
    prompts = iter([f"/handoff {abs_path}", "/exit"])

    class FakeAgent:
        async def close(self) -> None:
            captured["closed"] = True

    async def fake_handoff_current_session(agent, path: str | None = None) -> dict[str, object]:
        captured["path"] = path
        return {
            "compacted": True,
            "written": True,
            "path": path or "",
            "summary": "summary text",
            "before_tokens": 12,
            "after_tokens": 4,
            "reason": "",
        }

    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: FakeAgent())
    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())
    monkeypatch.setattr(cli, "handoff_current_session", fake_handoff_current_session, raising=False)
    monkeypatch.setattr(cli, "render_notice", lambda *args, **kwargs: None)
    monkeypatch.setattr(PromptSession, "prompt_async", _fake_prompt_iter(prompts))

    exit_code = cli.main(["chat", "--model", "m2", "--cwd", str(tmp_path), "--no-persist-session"])

    assert exit_code == 0
    assert captured["path"] == abs_path


def test_main_chat_handoff_error_shows_red_notice(monkeypatch, tmp_path) -> None:
    prompts = iter(["/handoff", "/exit"])
    notices: list[tuple[str, str, str]] = []

    class FakeAgent:
        async def close(self) -> None:
            return None

    async def fake_handoff_current_session(agent, path: str | None = None) -> dict[str, object]:
        raise PermissionError("Cannot write to /readonly/.handoff")

    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: FakeAgent())
    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())
    monkeypatch.setattr(cli, "handoff_current_session", fake_handoff_current_session, raising=False)
    monkeypatch.setattr(cli, "render_notice", lambda console, title, message, style="yellow": notices.append((title, message, style)))
    monkeypatch.setattr(PromptSession, "prompt_async", _fake_prompt_iter(prompts))

    exit_code = cli.main(["chat", "--model", "m2", "--cwd", str(tmp_path), "--no-persist-session"])

    assert exit_code == 0
    assert notices[0][0] == "Handoff Error"
    assert notices[0][2] == "red"
    assert "Cannot write to /readonly/.handoff" in notices[0][1]


def test_main_chat_handoff_no_cwd_flag_uses_process_cwd(monkeypatch) -> None:
    captured: dict[str, object] = {}
    prompts = iter(["/handoff", "/exit"])

    class FakeAgent:
        async def close(self) -> None:
            captured["closed"] = True

    async def fake_handoff_current_session(agent, path: str | None = None) -> dict[str, object]:
        captured["path"] = path
        return {
            "compacted": True,
            "written": True,
            "path": path or "",
            "summary": "summary text",
            "before_tokens": 12,
            "after_tokens": 4,
            "reason": "",
        }

    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: FakeAgent())
    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())
    monkeypatch.setattr(cli, "handoff_current_session", fake_handoff_current_session, raising=False)
    monkeypatch.setattr(cli, "render_notice", lambda *args, **kwargs: None)
    monkeypatch.setattr(PromptSession, "prompt_async", _fake_prompt_iter(prompts))

    exit_code = cli.main(["chat", "--model", "m2", "--no-persist-session"])

    assert exit_code == 0
    # Without --cwd, config.cwd is None, so CLI uses Path(".") / ".handoff" → ".handoff"
    assert captured["path"] == ".handoff"

def test_main_without_command_prints_rooster_code_help(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class ParserStub:
        @staticmethod
        def parse_args(argv):
            return type("Args", (), {"command": None})()

        @staticmethod
        def print_help():
            captured["printed"] = "usage: rooster-code"

    monkeypatch.setattr(cli, "build_parser", lambda: ParserStub())

    exit_code = cli.main([])

    assert exit_code == 0
    assert captured["printed"] == "usage: rooster-code"


def test_run_ask_routes_explicit_agent_request(monkeypatch) -> None:
    captured: dict[str, object] = {}
    panels: list[tuple[str, str]] = []

    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())
    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: (_ for _ in ()).throw(AssertionError("should not create top-level agent")))
    monkeypatch.setattr(cli, "find_requested_agent_name", lambda config, prompt: "reviewer")

    async def fake_run_named_agent_prompt(config, agent_name: str, prompt: str) -> str:
        captured["agent_name"] = agent_name
        captured["prompt"] = prompt
        return "AGENT_PATH=used"

    monkeypatch.setattr(cli, "run_named_agent_prompt", fake_run_named_agent_prompt)
    monkeypatch.setattr(cli, "render_agent_panel", lambda console, title, text, style: panels.append((title, text)))

    exit_code = cli.asyncio.run(
        cli.run_ask(
            "Use the reviewer agent to answer.",
            RuntimeConfig(model="m1", agents={"reviewer": {"description": "reviewer"}}),
        )
    )

    assert exit_code == 0
    assert captured["agent_name"] == "reviewer"
    assert captured["prompt"] == "Use the reviewer agent to answer."
    assert panels == [
        ("Agent Started", "reviewer"),
        ("Agent Result", "AGENT_PATH=used"),
    ]


def test_run_ask_installs_and_clears_question_handler(monkeypatch) -> None:
    captured: list[str] = []

    class FakeAgent:
        async def query(self, prompt: str):
            if False:
                yield None

        async def close(self) -> None:
            return None

    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())
    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: FakeAgent())
    monkeypatch.setattr(cli, "render_event_stream", lambda console, events, omit_duplicate_result=False, **_kwargs: (_ for _ in ()).throw(StopAsyncIteration()))
    monkeypatch.setattr(cli, "set_question_handler", lambda handler: captured.append("set" if callable(handler) else "bad"))
    monkeypatch.setattr(cli, "clear_question_handler", lambda: captured.append("clear"))

    try:
        cli.asyncio.run(cli.run_ask("hello", RuntimeConfig(model="m1")))
    except RuntimeError:
        pass

    assert captured == ["set", "clear"]


def test_run_ask_question_handler_uses_prompt_session(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeAgent:
        async def query(self, prompt: str):
            if False:
                yield None

        async def close(self) -> None:
            return None

    async def fake_prompt_async(self, prompt_text: str, *args, **kwargs):
        captured["prompt_text"] = prompt_text
        return "answer"

    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())
    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: FakeAgent())
    monkeypatch.setattr(cli, "render_event_stream", lambda console, events, omit_duplicate_result=False, **_kwargs: (_ for _ in ()).throw(StopAsyncIteration()))
    monkeypatch.setattr(PromptSession, "prompt_async", fake_prompt_async)
    monkeypatch.setattr(cli, "clear_question_handler", lambda: None)
    monkeypatch.setattr(cli, "set_question_handler", lambda handler: captured.setdefault("handler", handler))

    try:
        cli.asyncio.run(cli.run_ask("hello", RuntimeConfig(model="m1")))
    except RuntimeError:
        pass

    handler = cast(Callable[[str], Coroutine[object, object, str]], captured["handler"])
    answer = cli.asyncio.run(handler("Need input?"))
    assert answer == "answer"
    assert captured["prompt_text"] == "Need input? "


def test_run_ask_uses_runtime_abort_signal_and_sigint(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeAgent:
        async def query(self, prompt: str):
            if False:
                yield None

        async def close(self) -> None:
            captured["closed"] = True

    async def fake_render_event_stream(console, events, omit_duplicate_result=False, show_activity_trace=False, abort_signal=None):
        captured["abort_signal"] = abort_signal
        await asyncio.sleep(0)
        handler = cast(Callable[[object, object], None], captured["sigint_handler"])
        handler(signal.SIGINT, None)
        await asyncio.Future()

    signal_calls: list[tuple[object, object]] = []

    def fake_signal(sig, handler):
        signal_calls.append((sig, handler))
        if callable(handler):
            captured["sigint_handler"] = handler
        return "previous-handler"

    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())
    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: FakeAgent())
    monkeypatch.setattr(cli, "render_event_stream", fake_render_event_stream)
    monkeypatch.setattr(cli, "set_question_handler", lambda handler: None)
    monkeypatch.setattr(cli, "clear_question_handler", lambda: None)
    monkeypatch.setattr(cli, "cancel_background_subagent_tasks", lambda: asyncio.sleep(0))

    import rooster_code.runtime as runtime
    abort_values: list[object] = []
    monkeypatch.setattr(runtime, "set_abort_signal", lambda value: abort_values.append(value))
    monkeypatch.setattr(cli.signal, "signal", fake_signal)

    exit_code = cli.asyncio.run(cli.run_ask("hello", RuntimeConfig(model="m1")))

    assert exit_code == 130
    assert isinstance(captured["abort_signal"], asyncio.Event)
    assert abort_values[0] is captured["abort_signal"]
    assert abort_values[-1] is None
    assert signal_calls[0][0] == signal.SIGINT
    assert signal_calls[-1] == (signal.SIGINT, "previous-handler")
    assert captured["closed"] is True


def test_reset_session_closes_runtime_remote_mcp_clients(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeAgent:
        _client = None
        _provider = "provider"
        _engine = "engine"
        _initialized = True

        def clear(self) -> None:
            captured["cleared"] = True

        async def close_remote_mcp_clients(self) -> None:
            captured["remote_mcp_closed"] = True

        async def _initialize(self) -> None:
            captured["initialized"] = True

    class FakeTeamManager:
        def is_active(self) -> bool:
            return False

    monkeypatch.setattr(cli, "get_active_goal", lambda: None)
    monkeypatch.setattr(cli, "cancel_background_subagent_tasks", lambda: asyncio.sleep(0))
    monkeypatch.setattr(cli, "set_runtime_team_bridge", lambda team_manager, agent: None)
    monkeypatch.setattr(cli, "render_notice", lambda *args, **kwargs: None)
    import rooster_code.runtime as runtime
    monkeypatch.setattr(runtime, "rehydrate_tasks_from_history", lambda agent: captured.setdefault("rehydrated", True))

    cli.asyncio.run(cli._reset_session(SilentConsole(), FakeAgent(), FakeTeamManager()))

    assert captured["cleared"] is True
    assert captured["remote_mcp_closed"] is True
    assert captured["initialized"] is True
    assert captured["rehydrated"] is True


def test_run_chat_interrupt_closes_runtime_remote_mcp_clients(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeAgent:
        _client = None
        _provider = "provider"
        _engine = "engine"
        _initialized = True

        def clear(self) -> None:
            return None

        async def close_remote_mcp_clients(self) -> None:
            captured["remote_mcp_closed"] = True

        async def _initialize(self) -> None:
            return None

        async def close(self) -> None:
            captured["closed"] = True

        async def query(self, prompt: str):
            yield SDKMessage(type=SDKMessageType.ASSISTANT, text="ok")

    prompts = iter(["hello", "/exit"])
    monkeypatch.setattr(PromptSession, "prompt_async", _fake_prompt_iter(prompts))
    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: FakeAgent())
    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())
    monkeypatch.setattr(cli, "set_runtime_team_bridge", lambda team_manager, agent: None)
    import rooster_code.runtime as runtime
    monkeypatch.setattr(runtime, "rehydrate_tasks_from_history", lambda agent: None)

    async def fake_render_event_stream(console, events, **kwargs):
        raise asyncio.CancelledError()

    monkeypatch.setattr(cli, "render_event_stream", fake_render_event_stream)

    exit_code = cli.asyncio.run(cli.run_chat(RuntimeConfig(model="m1")))

    assert exit_code == 0
    assert captured["remote_mcp_closed"] is True
    assert captured["closed"] is True


def test_create_question_handler_cancels_when_abort_signal_is_set(monkeypatch) -> None:
    async def fake_prompt_async(self, prompt_text: str, *args, **kwargs):
        await asyncio.Future()

    monkeypatch.setattr(PromptSession, "prompt_async", fake_prompt_async)
    abort_signal = asyncio.Event()
    handler = cli._create_question_handler(PromptSession(), abort_signal)

    async def _run() -> None:
        task = asyncio.create_task(handler("Need input?"))
        await asyncio.sleep(0)
        abort_signal.set()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    cli.asyncio.run(_run())


def test_run_ask_installs_default_search_backend(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeAgent:
        async def query(self, prompt: str):
            if False:
                yield None

        async def close(self) -> None:
            return None

    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())
    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: FakeAgent())
    monkeypatch.setattr(cli, "render_event_stream", lambda console, events, omit_duplicate_result=False, **_kwargs: (_ for _ in ()).throw(StopAsyncIteration()))
    monkeypatch.setattr(cli, "set_question_handler", lambda handler: None)
    monkeypatch.setattr(cli, "clear_question_handler", lambda: None)
    monkeypatch.setattr(cli, "install_search_backend", lambda config: captured.setdefault("search_url", config.search_url))

    try:
        cli.asyncio.run(cli.run_ask("hello", RuntimeConfig(model="m1", search_url="https://searx.example/search")))
    except RuntimeError:
        pass

    assert captured == {"search_url": "https://searx.example/search"}


def test_install_search_backend_uses_local_default_url(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_set_search_fn(fn):
        captured["fn"] = fn

    monkeypatch.setattr(cli, "set_search_fn", fake_set_search_fn)

    cli.install_search_backend(RuntimeConfig(model="m1"))

    assert callable(captured["fn"])


def test_install_search_backend_uses_post_json_mapping(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"results": [{"title": "A", "url": "https://a.test", "content": "snippet a"}]}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json):
            captured["url"] = url
            captured["json"] = json
            return FakeResponse()

    def fake_set_search_fn(fn):
        captured["fn"] = fn

    monkeypatch.setattr(cli, "set_search_fn", fake_set_search_fn)
    monkeypatch.setattr(cli.httpx, "AsyncClient", FakeClient)

    cli.install_search_backend(RuntimeConfig(model="m1", search_url="http://127.0.0.1:8080/search"))

    search_fn = cast(Callable[[str, int], Coroutine[object, object, list[dict[str, str]]]], captured["fn"])
    results = cli.asyncio.run(search_fn("open agent sdk", 5))

    assert captured["url"] == "http://127.0.0.1:8080/search"
    assert captured["json"] == {"q": "open agent sdk", "format": "json", "pageno": 1, "safesearch": 1}
    assert results == [{"title": "A", "url": "https://a.test", "snippet": "snippet a"}]


def test_sessions_help_mentions_mutation_commands() -> None:
    parser = build_parser()
    help_text = parser.format_help()

    assert "sessions" in help_text
    assert "tools" in help_text
    assert "state" in help_text


def test_run_chat_renders_tool_table_for_tools_command(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeAgent:
        def clear(self) -> None:
            return None

        async def set_model(self, model: str) -> None:
            return None

        async def set_permission_mode(self, mode: str) -> None:
            return None

        async def query(self, prompt: str):
            if False:
                yield None

        async def close(self) -> None:
            captured["closed"] = True

    prompts = iter(["/tools", "/exit"])

    monkeypatch.setattr(PromptSession, "prompt_async", _fake_prompt_iter(prompts))
    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: FakeAgent())
    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())
    monkeypatch.setattr(cli, "list_tool_names", lambda: ["Read", "Write"])
    monkeypatch.setattr(cli, "render_tool_table", lambda console, tools: captured.setdefault("tools", tools))

    exit_code = cli.asyncio.run(cli.run_chat(RuntimeConfig(model="m1")))

    assert exit_code == 0
    assert captured["tools"] == ["Read", "Write"]
    assert captured["closed"] is True


def test_run_chat_uses_rooster_code_prompt_label(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeAgent:
        def clear(self) -> None:
            return None

        async def set_model(self, model: str) -> None:
            return None

        async def set_permission_mode(self, mode: str) -> None:
            return None

        async def query(self, prompt: str):
            if False:
                yield None

        async def close(self) -> None:
            captured["closed"] = True

    async def fake_prompt_async(self, prompt_text: str, *args, **kwargs):
        captured["prompt_text"] = prompt_text
        return "/exit"

    monkeypatch.setattr(PromptSession, "prompt_async", fake_prompt_async)
    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: FakeAgent())
    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())

    exit_code = cli.asyncio.run(cli.run_chat(RuntimeConfig(model="m1", persist_session=False)))

    assert exit_code == 0
    assert captured["prompt_text"] == "rooster-code> "
    assert captured["closed"] is True


def test_main_returns_130_on_keyboard_interrupt(monkeypatch) -> None:
    def fake_asyncio_run(coroutine):
        coroutine.close()
        raise KeyboardInterrupt()

    monkeypatch.setattr(cli, "build_parser", lambda: type("ParserStub", (), {"parse_args": staticmethod(lambda argv: type("Args", (), {"command": "chat"})())})())
    monkeypatch.setattr(cli, "config_from_namespace", lambda args, env: RuntimeConfig(model="m2"))
    monkeypatch.setattr(cli.asyncio, "run", fake_asyncio_run)

    exit_code = cli.main(["chat"])

    assert exit_code == 130


def test_importing_cli_does_not_eagerly_import_sdk(monkeypatch) -> None:
    import builtins
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in {"rooster_code.runtime", "open_agent_sdk"}:
            raise RuntimeError(f"unexpected eager import: {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    sys.modules.pop("rooster_code.cli", None)
    sys.modules.pop("rooster_code.rendering", None)

    imported = importlib.import_module("rooster_code.cli")

    assert imported is not None


def test_main_chat_installs_sigint_handler(monkeypatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(cli, "build_parser", lambda: type("ParserStub", (), {"parse_args": staticmethod(lambda argv: type("Args", (), {"command": "chat"})())})())
    monkeypatch.setattr(cli, "config_from_namespace", lambda args, env: RuntimeConfig(model="m2"))
    monkeypatch.setattr(
        cli,
        "run_async_with_sigint_exit",
        lambda coroutine: (captured.setdefault("called", True), coroutine.close(), 0)[2],
    )

    exit_code = cli.main(["chat"])

    assert exit_code == 0
    assert captured == {"called": True}


def test_run_async_with_sigint_exit_installs_and_restores_handler(monkeypatch) -> None:
    captured: list[tuple[object, object]] = []

    async def fake_coroutine() -> int:
        return 0

    monkeypatch.setattr("rooster_code.cli.signal.getsignal", lambda sig: "previous-handler")
    monkeypatch.setattr("rooster_code.cli.signal.signal", lambda sig, handler: captured.append((sig, handler)))
    monkeypatch.setattr(cli.asyncio, "run", lambda coroutine: (coroutine.close(), 0)[1])

    exit_code = importlib.import_module("rooster_code.cli").run_async_with_sigint_exit(fake_coroutine())

    assert exit_code == 0
    assert captured[0][0] == signal.SIGINT
    assert callable(captured[0][1])
    assert captured[1] == (signal.SIGINT, "previous-handler")


# --- Bug fix tests: cancellation issues ---


def test_run_ask_named_agent_closes_agent_on_cancel(monkeypatch) -> None:
    """Bug 2: named-agent path in run_ask must cancel background tasks on CancelledError.

    The non-named-agent finally block calls cancel_background_subagent_tasks(),
    but the named-agent finally block does not. Background tasks spawned by the
    named agent would be orphaned.
    """
    bg_cancelled = False

    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())
    monkeypatch.setattr(cli, "find_requested_agent_name", lambda config, prompt: "reviewer")

    async def fake_run_named_agent_prompt(config, agent_name, prompt):
        raise asyncio.CancelledError()

    monkeypatch.setattr(cli, "run_named_agent_prompt", fake_run_named_agent_prompt)
    monkeypatch.setattr(cli, "set_question_handler", lambda handler: None)
    monkeypatch.setattr(cli, "clear_question_handler", lambda: None)

    async def fake_cancel_bg():
        nonlocal bg_cancelled
        bg_cancelled = True

    monkeypatch.setattr(cli, "cancel_background_subagent_tasks", fake_cancel_bg)

    exit_code = cli.asyncio.run(
        cli.run_ask("Use the reviewer agent.", RuntimeConfig(model="m1", agents={"reviewer": {"description": "reviewer"}}))
    )

    assert exit_code == 130
    assert bg_cancelled is True


def test_run_chat_double_ctrl_c_stays_in_chat(monkeypatch) -> None:
    """Bug 1: SIGINT while at the prompt (no active query) should not exit chat.

    The SIGINT handler sets interrupted=True. When there's no active query
    task, nobody resets interrupted=False, so the main loop breaks and chat
    exits with code 130. After the fix, a stray SIGINT at the prompt should
    be treated as "cancel current input" not "exit chat", and the user stays.
    """
    captured: dict[str, object] = {}
    prompt_count = 0
    sigint_handler = None

    class FakeAgent:
        _initialized = True
        _client = None
        _provider = None
        _engine = None

        def clear(self) -> None:
            return None

        async def set_model(self, model: str) -> None:
            return None

        async def set_permission_mode(self, mode: str) -> None:
            return None

        async def query(self, prompt: str):
            yield SDKMessage(type=SDKMessageType.ASSISTANT, text="ok")

        async def close(self) -> None:
            captured["closed"] = True

    # Capture the SIGINT handler so we can fire it manually
    def fake_signal(sig, handler):
        nonlocal sigint_handler
        if callable(handler):
            sigint_handler = handler
        return lambda *a: None

    monkeypatch.setattr(cli.signal, "signal", fake_signal)

    async def fake_prompt_async(self, prompt_text: str, *args, **kwargs):
        nonlocal prompt_count
        prompt_count += 1
        if prompt_count == 1:
            return "hello"
        if prompt_count == 2:
            # SIGINT fires at idle prompt: handler sets interrupted=True
            # In the real app, prompt_toolkit sees the signal and may
            # raise KeyboardInterrupt, which prompt_once catches as None.
            # We simulate by firing the handler and returning None.
            assert sigint_handler is not None
            sigint_handler(signal.SIGINT, None)
            return None  # prompt_once returns None → sets interrupted=True again
        if prompt_count == 3:
            return "/exit"
        raise EOFError()

    monkeypatch.setattr(PromptSession, "prompt_async", fake_prompt_async)
    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: FakeAgent())
    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())

    async def fake_render_event_stream(console, events, **kwargs):
        async for e in events:
            pass

    monkeypatch.setattr(cli, "render_event_stream", fake_render_event_stream)

    exit_code = cli.asyncio.run(cli.run_chat(RuntimeConfig(model="m1")))

    # BUG: exit_code is 130 because interrupted=True → loop breaks → "exiting chat"
    # After fix, returning None from prompt at idle should not exit chat,
    # it should just redisplay the prompt (chat continues to /exit on round 3)
    assert exit_code == 0
    assert prompt_count == 3  # user gets a 3rd prompt after accidental signal


def test_run_ask_named_agent_non_cancel_path_closes_agent(monkeypatch) -> None:
    """Bug 2 companion: even the non-named-agent path should close agent."""
    captured: dict[str, object] = {}

    class FakeAgent:
        async def query(self, prompt: str):
            yield SDKMessage(type=SDKMessageType.ASSISTANT, text="ok")

        async def close(self) -> None:
            captured["agent_closed"] = True

    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())
    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: FakeAgent())
    monkeypatch.setattr(cli, "find_requested_agent_name", lambda config, prompt: None)

    async def fake_render(*args, **kwargs):
        pass

    monkeypatch.setattr(cli, "render_event_stream", fake_render)
    monkeypatch.setattr(cli, "set_question_handler", lambda handler: None)
    monkeypatch.setattr(cli, "clear_question_handler", lambda: None)
    monkeypatch.setattr(cli, "cancel_background_subagent_tasks", lambda: asyncio.sleep(0))

    exit_code = cli.asyncio.run(cli.run_ask("hello", RuntimeConfig(model="m1")))

    assert exit_code == 0
    assert captured.get("agent_closed") is True
