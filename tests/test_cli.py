from cock_code.cli import build_parser


import builtins
import importlib
import signal
import sys

import cock_code.cli as cli
from cock_code.config import RuntimeConfig
from open_agent_sdk import SDKMessage, SDKMessageType


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
        ]
    )

    assert args.command == "ask"
    assert args.prompt == "hello"
    assert args.model == "m1"
    assert args.resume == "sess-1"
    assert args.allowed_tools == ["Read"]
    assert args.disallowed_tools == ["Bash"]


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

    async def fake_render_event_stream(console, events, omit_duplicate_result: bool = False) -> None:
        messages = []
        async for event in events:
            messages.append(event.type.value)
        captured["messages"] = messages
        captured["omit_duplicate_result"] = omit_duplicate_result

    monkeypatch.setattr(cli, "create_runtime_agent", fake_create_runtime_agent)
    monkeypatch.setattr(cli, "render_event_stream", fake_render_event_stream)
    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())

    exit_code = cli.asyncio.run(cli.run_ask("hello", RuntimeConfig(model="m1")))

    assert exit_code == 0
    assert captured["prompt"] == "hello"
    assert captured["messages"] == ["assistant", "result"]
    assert captured["omit_duplicate_result"] is True
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

    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())
    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: (_ for _ in ()).throw(AssertionError("should not create top-level agent")))
    monkeypatch.setattr(cli, "find_requested_agent_name", lambda config, prompt: "reviewer")

    async def fake_run_named_agent_prompt(config, agent_name: str, prompt: str) -> str:
        captured["agent_name"] = agent_name
        captured["prompt"] = prompt
        return "AGENT_PATH=used"

    monkeypatch.setattr(cli, "run_named_agent_prompt", fake_run_named_agent_prompt)
    monkeypatch.setattr(cli, "render_text_panel", lambda console, title, text, style: captured.update({"title": title, "text": text}))

    exit_code = cli.asyncio.run(
        cli.run_ask(
            "Use the reviewer agent to answer.",
            RuntimeConfig(model="m1", agents={"reviewer": {"description": "reviewer"}}),
        )
    )

    assert exit_code == 0
    assert captured == {
        "agent_name": "reviewer",
        "prompt": "Use the reviewer agent to answer.",
        "title": "Assistant",
        "text": "AGENT_PATH=used",
    }


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

    monkeypatch.setattr(cli, "Prompt", type("PromptStub", (), {"ask": staticmethod(lambda _label: next(prompts))}))
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
