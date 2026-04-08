from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import signal
from urllib.parse import urlencode

import httpx

from rich.prompt import Prompt

from cock_code.chat import parse_chat_command
from cock_code.config import config_from_namespace
from cock_code.rendering import (
    build_console,
    render_banner,
    render_event_stream,
    render_help,
    render_notice,
    render_agent_panel,
    render_session_info,
    render_session_table,
    render_state,
    render_text_panel,
    render_tool_table,
    render_transcript,
)


def set_question_handler(handler):
    from open_agent_sdk.tools.ask_user import set_question_handler as sdk_set_question_handler

    return sdk_set_question_handler(handler)


def clear_question_handler():
    from open_agent_sdk.tools.ask_user import set_question_handler as sdk_set_question_handler

    return sdk_set_question_handler(None)


def set_search_fn(fn):
    from open_agent_sdk.tools.web_search import set_search_fn as sdk_set_search_fn

    return sdk_set_search_fn(fn)


def install_search_backend(config) -> None:
    search_url = config.search_url or "http://127.0.0.1:8080/search"

    async def searxng_search(query: str, max_results: int) -> list[dict[str, str]]:
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.post(
                search_url,
                json={
                    "q": query,
                    "format": "json",
                    "pageno": 1,
                    "safesearch": 1,
                },
            )
            response.raise_for_status()
        payload = response.json()
        results = payload.get("results", [])[:max_results]
        return [
            {
                "title": str(item.get("title", "")),
                "url": str(item.get("url", "")),
                "snippet": str(item.get("content", "")),
            }
            for item in results
        ]

    set_search_fn(searxng_search)


def create_runtime_agent(config):
    from cock_code.runtime import create_runtime_agent as runtime_create_agent

    return runtime_create_agent(config)


def find_requested_agent_name(config, prompt: str):
    from cock_code.runtime import find_requested_agent_name as runtime_find_requested_agent_name

    return runtime_find_requested_agent_name(config, prompt)


async def run_named_agent_prompt(config, agent_name: str, prompt: str) -> str:
    from cock_code.runtime import run_named_agent_prompt as runtime_run_named_agent_prompt

    return await runtime_run_named_agent_prompt(config, agent_name, prompt)


async def stream_named_agent_events(config, agent_name: str, prompt: str):
    from cock_code.runtime import stream_named_agent_events as runtime_stream_named_agent_events

    async for event in runtime_stream_named_agent_events(config, agent_name, prompt):
        yield event


async def stream_skill_events(config, agent, skill_name: str, args: str):
    from cock_code.runtime import stream_skill_events as runtime_stream_skill_events

    async for event in runtime_stream_skill_events(config, agent, skill_name, args):
        yield event


async def compact_current_session(agent):
    from cock_code.runtime import compact_current_session as runtime_compact_current_session

    return await runtime_compact_current_session(agent)


def list_skill_names() -> list[str]:
    from cock_code.runtime import list_skill_names as runtime_list_skill_names

    return runtime_list_skill_names()


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


async def enforce_session_retention(limit: int = 20):
    from cock_code.runtime import enforce_session_retention as runtime_enforce_session_retention

    return await runtime_enforce_session_retention(limit)


async def rename_session(session_id: str, title: str):
    from cock_code.runtime import rename_session as runtime_rename_session

    return await runtime_rename_session(session_id, title)


async def tag_session(session_id: str, tags: list[str]):
    from cock_code.runtime import tag_session as runtime_tag_session

    return await runtime_tag_session(session_id, tags)


def get_state_snapshot(name: str, agent_name: str | None = None):
    from cock_code.runtime import get_state_snapshot as runtime_get_state_snapshot

    return runtime_get_state_snapshot(name, agent_name)


async def get_task_output(task_id: str) -> str:
    from cock_code.runtime import get_task_output as runtime_get_task_output

    return await runtime_get_task_output(task_id)


async def stop_task(task_id: str) -> bool:
    from cock_code.runtime import stop_task as runtime_stop_task

    return await runtime_stop_task(task_id)


async def start_background_agent_task(config, agent_name: str, prompt: str) -> str:
    from cock_code.runtime import start_background_agent_task as runtime_start_background_agent_task

    return await runtime_start_background_agent_task(config, agent_name, prompt)


async def wait_for_task(task_id: str) -> dict[str, object]:
    from cock_code.runtime import wait_for_task as runtime_wait_for_task

    return await runtime_wait_for_task(task_id)


def _render_task_notification(console, agent, note: dict[str, object]) -> None:
    status = str(note.get("status", "completed"))
    style = "green" if status == "completed" else "yellow"
    output = str(note.get("output", ""))
    subject = str(note.get("subject", "task"))
    task_id = str(note.get("task_id", ""))
    display_output = output[:500] + ("..." if len(output) > 500 else "")
    lines = [f"{subject} ({task_id}) {status}"]
    if display_output:
        lines.append(display_output)
    render_notice(console, "Background Task", "\n".join(lines), style)
    append_task_result_to_context(agent, task_id, {"status": status, "output": output})


def append_task_result_to_context(agent, task_id: str, task_result: dict[str, object]) -> None:
    output = str(task_result.get("output", "")).strip()
    status = str(task_result.get("status", "")).strip()
    if status not in {"completed", "cancelled"} or not output:
        return
    history = getattr(agent, "_history", None)
    if not isinstance(history, list):
        return
    history.append(
        {
            "role": "assistant",
            "content": [{"type": "text", "text": f"Background task {task_id} result:\n\n{output}"}],
        }
    )


def read_background_notifications() -> list[dict[str, object]]:
    from cock_code.runtime import read_background_notifications as runtime_read_background_notifications

    return runtime_read_background_notifications()


async def cancel_background_subagent_tasks() -> None:
    from cock_code.runtime import cancel_background_subagent_tasks as runtime_cancel_background_subagent_tasks

    await runtime_cancel_background_subagent_tasks()


def list_tool_names() -> list[str]:
    from cock_code.runtime import list_tool_names as runtime_list_tool_names

    return runtime_list_tool_names()


def add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cwd", help="Working directory for the agent")
    parser.add_argument("--model", help="Override COCK_CODE_MODEL for this run")
    parser.add_argument("--permission-mode", help="SDK permission mode")
    parser.add_argument("--search-url", help="SearXNG search endpoint URL")
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
    parser.add_argument("--skills-dir", help="Directory containing local SKILL.md bundles")
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
    install_search_backend(config)

    async def question_handler(question: str) -> str:
        return Prompt.ask(question)

    set_question_handler(question_handler)
    requested_agent = find_requested_agent_name(config, prompt)
    if requested_agent:
        try:
            render_agent_panel(console, "Agent Started", requested_agent, "blue")
            text = await run_named_agent_prompt(config, requested_agent, prompt)
            render_agent_panel(console, "Agent Result", text, "blue")
            return 0
        except Exception as exc:
            render_notice(console, "Error", str(exc), "red")
            return 1
        finally:
            clear_question_handler()

    agent = create_runtime_agent(config)

    try:
        await render_event_stream(console, agent.query(prompt), omit_duplicate_result=True, show_activity_trace=False)
    except Exception as exc:
        render_notice(console, "Error", str(exc), "red")
        return 1
    finally:
        clear_question_handler()
        await cancel_background_subagent_tasks()
        await agent.close()

    if config.persist_session:
        await enforce_session_retention()

    return 0


async def run_chat(config) -> int:
    console = build_console()
    render_banner(console, "chat", config)
    install_search_backend(config)
    agent = create_runtime_agent(config)
    interrupted = False

    def prompt_once() -> tuple[str, str | BaseException]:
        try:
            return ("ok", Prompt.ask("cock-code"))
        except (KeyboardInterrupt, EOFError) as exc:
            return ("error", exc)

    async def question_handler(question: str) -> str:
        return Prompt.ask(question)

    set_question_handler(question_handler)

    try:
        while True:
            prompt_task: asyncio.Task[object] | None = None
            try:
                prompt_task = asyncio.create_task(asyncio.to_thread(prompt_once))
                while not prompt_task.done():
                    for note in read_background_notifications():
                        if note.get("type") == "background_task_completed":
                            _render_task_notification(console, agent, note)
                    await asyncio.sleep(0.1)
                prompt_status, prompt_value = await prompt_task
                if prompt_status == "error":
                    if not isinstance(prompt_value, BaseException):
                        raise RuntimeError(f"Unexpected prompt error type: {type(prompt_value)}")
                    raise prompt_value
                user_input = str(prompt_value)
            except (KeyboardInterrupt, EOFError):
                interrupted = True
                render_notice(console, "Interrupted", "Exiting chat.", "yellow")
                break
            finally:
                if prompt_task is not None and prompt_task.done():
                    with contextlib.suppress(BaseException):
                        prompt_task.result()
            for note in read_background_notifications():
                if note.get("type") == "background_task_completed":
                    _render_task_notification(console, agent, note)
            command = parse_chat_command(user_input)
            if command.name == "exit":
                break
            if command.name == "clear":
                agent.clear()
                render_notice(console, "Cleared", "Agent history cleared.", "green")
                continue
            if command.name == "compact":
                try:
                    result = await compact_current_session(agent)
                except Exception as exc:
                    render_notice(console, "Compact Error", str(exc), "red")
                    continue

                title = "Compacted" if result["compacted"] else "Compaction skipped"
                style = "green" if result["compacted"] else "yellow"
                details = str(result["summary"] or result["reason"] or "No summary returned.")
                render_notice(
                    console,
                    title,
                    f"Tokens: {result['before_tokens']} → {result['after_tokens']}\n\n{details}",
                    style,
                )
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
            if command.name == "skills":
                render_state(console, "Skills", {"skills": list_skill_names()})
                continue
            if command.name == "tasks":
                render_state(console, "Tasks", get_state_snapshot("tasks"))
                continue
            if command.name in {"agent-bg", "bg"} and len(command.args) >= 2:
                try:
                    task_id = await start_background_agent_task(config, command.args[0], " ".join(command.args[1:]))
                    render_state(console, "Background Task", {"agent": command.args[0], "task_id": task_id})
                except Exception as exc:
                    render_notice(console, "Error", str(exc), "red")
                continue
            if command.name == "task-output" and command.args:
                render_state(console, "Task Output", {"task_id": command.args[0], "output": await get_task_output(command.args[0])})
                continue
            if command.name == "task-stop" and command.args:
                render_state(console, "Task Stopped", {"task_id": command.args[0], "stopped": await stop_task(command.args[0])})
                continue
            if command.name == "wait" and command.args:
                task_result = await wait_for_task(command.args[0])
                append_task_result_to_context(agent, command.args[0], task_result)
                from cock_code.runtime import _notified_task_ids
                _notified_task_ids.add(command.args[0])
                render_state(console, "Task Wait", {"task_id": command.args[0], **task_result})
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
            available_skills = set(list_skill_names())
            if command.name in available_skills:
                try:
                    render_agent_panel(console, "Skill Started", command.name, "blue")
                    await render_event_stream(
                        console,
                        stream_skill_events(config, agent, command.name, " ".join(command.args)),
                        omit_duplicate_result=True,
                        show_activity_trace=True,
                    )
                except Exception as exc:
                    render_notice(console, "Error", str(exc), "red")
                continue
            if user_input.startswith("/"):
                render_notice(console, "Unknown command", user_input, "red")
                continue

            requested_agent = find_requested_agent_name(config, user_input)
            if requested_agent:
                try:
                    render_agent_panel(console, "Agent Started", requested_agent, "blue")
                    await render_event_stream(
                        console,
                        stream_named_agent_events(config, requested_agent, user_input),
                        omit_duplicate_result=True,
                        show_activity_trace=True,
                    )
                except Exception as exc:
                    render_notice(console, "Error", str(exc), "red")
                continue
            
            try:
                await render_event_stream(console, agent.query(user_input), omit_duplicate_result=True, show_activity_trace=True)
            except Exception as exc:
                render_notice(console, "Error", str(exc), "red")
                continue
    finally:
        clear_question_handler()
        await cancel_background_subagent_tasks()
        if interrupted:
            try:
                await asyncio.wait_for(agent.close(), timeout=0.05)
            except TimeoutError:
                pass
        else:
            await agent.close()

    if config.persist_session:
        await enforce_session_retention()

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
            asyncio.run(enforce_session_retention())
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
