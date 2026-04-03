from __future__ import annotations

import argparse
import asyncio
import os
import signal

from rich.prompt import Prompt

from cock_code.chat import parse_chat_command
from cock_code.config import config_from_namespace
from cock_code.rendering import (
    build_console,
    render_banner,
    render_event_stream,
    render_help,
    render_notice,
    render_session_info,
    render_session_table,
    render_state,
    render_text_panel,
    render_tool_table,
    render_transcript,
)


def set_question_handler(handler):
    from open_agent_sdk import set_question_handler as sdk_set_question_handler

    return sdk_set_question_handler(handler)


def clear_question_handler():
    from open_agent_sdk import clear_question_handler as sdk_clear_question_handler

    return sdk_clear_question_handler()


def create_runtime_agent(config):
    from cock_code.runtime import create_runtime_agent as runtime_create_agent

    return runtime_create_agent(config)


def find_requested_agent_name(config, prompt: str):
    from cock_code.runtime import find_requested_agent_name as runtime_find_requested_agent_name

    return runtime_find_requested_agent_name(config, prompt)


async def run_named_agent_prompt(config, agent_name: str, prompt: str) -> str:
    from cock_code.runtime import run_named_agent_prompt as runtime_run_named_agent_prompt

    return await runtime_run_named_agent_prompt(config, agent_name, prompt)


async def list_sessions():
    from cock_code.runtime import list_sessions as runtime_list_sessions

    return await runtime_list_sessions()


async def get_session_messages(session_id: str):
    from cock_code.runtime import get_session_messages as runtime_get_session_messages

    return await runtime_get_session_messages(session_id)


async def get_session_info(session_id: str):
    from cock_code.runtime import get_session_info as runtime_get_session_info

    return await runtime_get_session_info(session_id)


async def delete_session(session_id: str):
    from cock_code.runtime import delete_session as runtime_delete_session

    return await runtime_delete_session(session_id)


async def fork_session(session_id: str, new_id: str | None):
    from cock_code.runtime import fork_session as runtime_fork_session

    return await runtime_fork_session(session_id, new_id)


async def rename_session(session_id: str, title: str):
    from cock_code.runtime import rename_session as runtime_rename_session

    return await runtime_rename_session(session_id, title)


async def tag_session(session_id: str, tags: list[str]):
    from cock_code.runtime import tag_session as runtime_tag_session

    return await runtime_tag_session(session_id, tags)


def get_state_snapshot(name: str, agent_name: str | None = None):
    from cock_code.runtime import get_state_snapshot as runtime_get_state_snapshot

    return runtime_get_state_snapshot(name, agent_name)


def list_tool_names() -> list[str]:
    from cock_code.runtime import list_tool_names as runtime_list_tool_names

    return runtime_list_tool_names()


def add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cwd", help="Working directory for the agent")
    parser.add_argument("--model", help="Override COCK_CODE_MODEL for this run")
    parser.add_argument("--permission-mode", help="SDK permission mode")
    parser.add_argument("--max-turns", type=int, help="Maximum SDK turns")
    parser.add_argument("--max-budget-usd", type=float, help="Maximum spend for this run")
    parser.add_argument("--max-tokens", type=int, help="Maximum output tokens")
    parser.add_argument("--thinking-budget", type=int, help="Extended thinking token budget")
    parser.add_argument("--allowed-tool", dest="allowed_tools", action="append", help="Allow a tool by name")
    parser.add_argument("--disallowed-tool", dest="disallowed_tools", action="append", help="Block a tool by name")
    parser.add_argument("--session-id", help="Explicit session id")
    parser.add_argument("--resume", help="Resume a specific session id")
    parser.add_argument("--continue-session", action="store_true", help="Continue the last session")
    parser.add_argument("--fork-session", help="Fork a source session before running")
    parser.add_argument("--persist-session", dest="persist_session", action="store_true", help="Persist session on close")
    parser.add_argument("--no-persist-session", dest="persist_session", action="store_false", help="Do not persist session on close")
    parser.add_argument("--sandbox", action="store_true", help="Enable SDK sandbox mode")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--include-partials", action="store_true", help="Include partial SDK stream events")
    parser.add_argument("--env", action="append", help="Pass KEY=VALUE into the tool environment")
    parser.add_argument("--custom-header", dest="custom_headers", action="append", help="Pass KEY=VALUE as an API header")
    parser.add_argument("--extra-args-file", help="JSON file for SDK extra_args")
    parser.add_argument("--json-schema-file", help="JSON file for structured output schema")
    parser.add_argument("--agents-file", help="JSON file for subagent definitions")
    parser.add_argument("--hooks-file", help="JSON file for hook configuration")
    parser.add_argument("--mcp-file", help="JSON file for MCP server configuration")
    parser.set_defaults(persist_session=True)


def run_async_with_sigint_exit(coroutine) -> int:
    previous_handler = signal.getsignal(signal.SIGINT)

    def handle_sigint(signum, frame):
        raise SystemExit(130)

    signal.signal(signal.SIGINT, handle_sigint)
    try:
        return asyncio.run(coroutine)
    except KeyboardInterrupt:
        return 130
    except SystemExit as exc:
        if exc.code == 130:
            return 130
        raise
    finally:
        signal.signal(signal.SIGINT, previous_handler)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cock-code", description="Rich CLI wrapper for the Open Agent SDK")
    subparsers = parser.add_subparsers(dest="command")

    ask_parser = subparsers.add_parser("ask", help="Run a one-shot prompt")
    ask_parser.add_argument("prompt")
    add_runtime_arguments(ask_parser)

    chat_parser = subparsers.add_parser("chat", help="Start an interactive chat session")
    add_runtime_arguments(chat_parser)

    sessions_parser = subparsers.add_parser("sessions", help="Inspect and manage persisted sessions")
    sessions_subparsers = sessions_parser.add_subparsers(dest="sessions_command")
    sessions_subparsers.add_parser("list", help="List saved sessions")
    info_parser = sessions_subparsers.add_parser("info", help="Show session metadata")
    info_parser.add_argument("session_id")
    delete_parser = sessions_subparsers.add_parser("delete", help="Delete a saved session")
    delete_parser.add_argument("session_id")
    fork_parser = sessions_subparsers.add_parser("fork", help="Fork a session into a new id")
    fork_parser.add_argument("session_id")
    fork_parser.add_argument("--new-id")
    rename_parser = sessions_subparsers.add_parser("rename", help="Rename a session")
    rename_parser.add_argument("session_id")
    rename_parser.add_argument("title")
    tag_parser = sessions_subparsers.add_parser("tag", help="Attach tags to a session")
    tag_parser.add_argument("session_id")
    tag_parser.add_argument("tags", nargs="+")
    show_parser = sessions_subparsers.add_parser("show", help="Show session transcript")
    show_parser.add_argument("session_id")

    tools_parser = subparsers.add_parser("tools", help="Inspect tool availability")
    tools_subparsers = tools_parser.add_subparsers(dest="tools_command")
    tools_subparsers.add_parser("list", help="List SDK base tools")

    state_parser = subparsers.add_parser("state", help="Inspect exported SDK runtime state")
    state_subparsers = state_parser.add_subparsers(dest="state_command")
    state_subparsers.add_parser("tasks", help="Show task store contents")
    state_subparsers.add_parser("teams", help="Show team store contents")
    mailboxes_parser = state_subparsers.add_parser("mailboxes", help="Show mailbox contents")
    mailboxes_parser.add_argument("--agent")
    state_subparsers.add_parser("config", help="Show config store contents")
    state_subparsers.add_parser("cron", help="Show cron store contents")
    state_subparsers.add_parser("plan", help="Show plan-mode state")
    state_subparsers.add_parser("todos", help="Show todo store contents")

    return parser


async def run_ask(prompt: str, config) -> int:
    console = build_console()
    render_banner(console, "ask", config)
    requested_agent = find_requested_agent_name(config, prompt)
    if requested_agent:
        try:
            text = await run_named_agent_prompt(config, requested_agent, prompt)
            render_text_panel(console, "Assistant", text, "cyan")
            return 0
        except Exception as exc:
            render_notice(console, "Error", str(exc), "red")
            return 1

    agent = create_runtime_agent(config)

    try:
        await render_event_stream(console, agent.query(prompt), omit_duplicate_result=True)
    except Exception as exc:
        render_notice(console, "Error", str(exc), "red")
        return 1
    finally:
        await agent.close()

    return 0


async def run_chat(config) -> int:
    console = build_console()
    render_banner(console, "chat", config)
    agent = create_runtime_agent(config)
    interrupted = False

    async def question_handler(question: str) -> str:
        return Prompt.ask(question)

    set_question_handler(question_handler)

    try:
        while True:
            try:
                user_input = Prompt.ask("cock-code")
            except (KeyboardInterrupt, EOFError):
                interrupted = True
                render_notice(console, "Interrupted", "Exiting chat.", "yellow")
                break
            command = parse_chat_command(user_input)
            if command.name == "exit":
                break
            if command.name == "clear":
                agent.clear()
                render_notice(console, "Cleared", "Agent history cleared.", "green")
                continue
            if command.name == "help":
                render_help(console)
                continue
            if command.name == "model" and command.args:
                config.model = command.args[0]
                await agent.set_model(command.args[0])
                render_notice(console, "Model", f"Switched to {command.args[0]}", "green")
                continue
            if command.name == "permission" and command.args:
                config.permission_mode = command.args[0]
                await agent.set_permission_mode(command.args[0])
                render_notice(console, "Permission", f"Permission mode set to {command.args[0]}", "green")
                continue
            if command.name == "tools":
                render_tool_table(console, list_tool_names())
                continue
            if command.name == "sessions":
                render_session_table(console, await list_sessions())
                continue
            if command.name == "status":
                render_state(
                    console,
                    "Chat Status",
                    {
                        "model": config.model or "default",
                        "permission_mode": config.permission_mode,
                        "session": config.resume or config.session_id or "new",
                    },
                )
                continue
            if command.name == "resume" and command.args:
                await agent.close()
                config.resume = command.args[0]
                agent = create_runtime_agent(config)
                render_notice(console, "Session", f"Resumed {command.args[0]}", "green")
                continue
            if user_input.startswith("/"):
                render_notice(console, "Unknown command", user_input, "red")
                continue

            requested_agent = find_requested_agent_name(config, user_input)
            if requested_agent:
                try:
                    text = await run_named_agent_prompt(config, requested_agent, user_input)
                    render_text_panel(console, "Assistant", text, "cyan")
                except Exception as exc:
                    render_notice(console, "Error", str(exc), "red")
                continue
            
            try:
                await render_event_stream(console, agent.query(user_input), omit_duplicate_result=True)
            except Exception as exc:
                render_notice(console, "Error", str(exc), "red")
                continue
    finally:
        clear_question_handler()
        if interrupted:
            try:
                await asyncio.wait_for(agent.close(), timeout=0.1)
            except TimeoutError:
                pass
        else:
            await agent.close()

    return 130 if interrupted else 0


def main(argv: list[str] | None = None) -> int:
    try:
        parser = build_parser()
        args = parser.parse_args(argv)

        if args.command == "ask":
            config = config_from_namespace(args, os.environ)
            return run_async_with_sigint_exit(run_ask(args.prompt, config))

        if args.command == "chat":
            config = config_from_namespace(args, os.environ)
            return run_async_with_sigint_exit(run_chat(config))

        if args.command == "sessions" and args.sessions_command == "list":
            console = build_console()
            sessions = asyncio.run(list_sessions())
            render_session_table(console, sessions)
            return 0

        if args.command == "sessions" and args.sessions_command == "show":
            console = build_console()
            messages = asyncio.run(get_session_messages(args.session_id))
            render_transcript(console, messages)
            return 0

        if args.command == "sessions" and args.sessions_command == "info":
            console = build_console()
            info = asyncio.run(get_session_info(args.session_id))
            render_session_info(console, info)
            return 0

        if args.command == "sessions" and args.sessions_command == "delete":
            console = build_console()
            deleted = asyncio.run(delete_session(args.session_id))
            render_state(console, "Session Deleted", {"deleted": deleted, "session_id": args.session_id})
            return 0

        if args.command == "sessions" and args.sessions_command == "fork":
            console = build_console()
            new_session_id = asyncio.run(fork_session(args.session_id, args.new_id))
            render_state(console, "Session Forked", {"forked_from": args.session_id, "new_session_id": new_session_id})
            return 0

        if args.command == "sessions" and args.sessions_command == "rename":
            console = build_console()
            asyncio.run(rename_session(args.session_id, args.title))
            render_state(console, "Session Renamed", {"renamed": True, "session_id": args.session_id, "title": args.title})
            return 0

        if args.command == "sessions" and args.sessions_command == "tag":
            console = build_console()
            asyncio.run(tag_session(args.session_id, args.tags))
            render_state(console, "Session Tagged", {"tagged": True, "session_id": args.session_id, "tags": args.tags})
            return 0

        if args.command == "tools" and args.tools_command == "list":
            console = build_console()
            render_tool_table(console, list_tool_names())
            return 0

        if args.command == "state" and args.state_command in {"tasks", "teams", "mailboxes", "config", "cron", "plan", "todos"}:
            console = build_console()
            render_state(console, args.state_command.title(), get_state_snapshot(args.state_command, getattr(args, "agent", None)))
            return 0

        parser.print_help()
        return 0
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
