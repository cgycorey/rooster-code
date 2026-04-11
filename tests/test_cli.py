import asyncio
import contextlib
from cock_code.cli import build_parser


import importlib
import signal
import sys
from typing import Callable, Coroutine, cast

import cock_code.cli as cli
from cock_code.config import RuntimeConfig
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
    assert "ask" in help_text
    assert "chat" in help_text
    assert "sessions" in help_text
    assert "tools" in help_text
    assert "state" in help_text


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

    import cock_code.runtime as runtime
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
        if name in {"cock_code.runtime", "open_agent_sdk"}:
            raise RuntimeError(f"unexpected eager import: {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    sys.modules.pop("cock_code.cli", None)
    sys.modules.pop("cock_code.rendering", None)

    imported = importlib.import_module("cock_code.cli")

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

    monkeypatch.setattr("cock_code.cli.signal.getsignal", lambda sig: "previous-handler")
    monkeypatch.setattr("cock_code.cli.signal.signal", lambda sig, handler: captured.append((sig, handler)))
    monkeypatch.setattr(cli.asyncio, "run", lambda coroutine: (coroutine.close(), 0)[1])

    exit_code = importlib.import_module("cock_code.cli").run_async_with_sigint_exit(fake_coroutine())

    assert exit_code == 0
    assert captured[0][0] == signal.SIGINT
    assert callable(captured[0][1])
    assert captured[1] == (signal.SIGINT, "previous-handler")
