from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import re
import signal
import sys
import threading
from urllib.parse import urlencode

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.shortcuts import print_formatted_text
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.history import History
from prompt_toolkit.patch_stdout import patch_stdout

import httpx

from cock_code.chat import parse_chat_command
from cock_code.config import config_from_namespace
from cock_code.team import TeamManager, set_runtime_team_bridge
from cock_code.rendering import (
    build_console,
    render_agents_list,
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


_console_lock = threading.RLock()
_prompt_label = "cock-code> "
_injected_task_ids: set[str] = set()


def _create_question_handler(
    session: PromptSession[str],
    abort_signal: asyncio.Event | None = None,
):
    async def question_handler(question: str) -> str:
        if abort_signal is not None and abort_signal.is_set():
            raise asyncio.CancelledError()
        prompt_task = asyncio.create_task(session.prompt_async(f"{question} "))
        abort_task: asyncio.Task[bool] | None = None
        if abort_signal is not None:
            abort_task = asyncio.create_task(abort_signal.wait())
        try:
            if abort_task is None:
                return await prompt_task
            done, _ = await asyncio.wait(
                {prompt_task, abort_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if abort_task in done:
                prompt_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await prompt_task
                raise asyncio.CancelledError()
            return prompt_task.result()
        except (KeyboardInterrupt, EOFError) as exc:
            raise asyncio.CancelledError() from exc
        finally:
            if abort_task is not None:
                abort_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await abort_task

    return question_handler


def _render_task_notification(console, agent, note: dict[str, object], prompt_session: PromptSession[str] | None = None) -> None:
    from cock_code.runtime import sanitize_task_output

    status = str(note.get("status", "completed"))
    style = "green" if status == "completed" else "yellow"
    output = sanitize_task_output(str(note.get("output", "")))
    subject = str(note.get("subject", "task"))
    task_id = str(note.get("task_id", ""))
    lines = [f"{subject} ({task_id}) {status}"]
    if output:
        lines.append(f"Summary: {_compact_task_output(output)}")
        lines.append(f"Full output: /task-output {task_id}")
    app = getattr(prompt_session, "app", None) if prompt_session is not None else None
    notification_text = "\n".join(lines)
    with _console_lock:
        if app is not None:
            print_formatted_text(_prompt_box("Background Task", lines, style))
        else:
            render_notice(console, "Background Task", notification_text, style)
        append_task_result_to_context(agent, task_id, {"status": status, "output": output})
    if app is not None:
        app.invalidate()


def _prompt_box(title: str, lines: list[str], style: str) -> FormattedText:
    border_style = {
        "green": "ansigreen",
        "yellow": "ansiyellow",
        "red": "ansired",
        "blue": "ansiblue",
    }.get(style, "")
    content_lines = [line.rstrip() for line in lines] or [""]
    width = max(len(title), *(len(line) for line in content_lines))
    width = min(max(width, 20), 100)

    def pad(text: str) -> str:
        return text[:width].ljust(width)

    top = f"╭─ {title} {'─' * max(width - len(title) - 1, 0)}╮"
    bottom = f"╰{'─' * (width + 3)}╯"

    fragments: list[tuple[str, str]] = []
    if border_style:
        fragments.append((border_style, top))
    else:
        fragments.append(("", top))
    fragments.append(("", "\n"))

    for line in content_lines:
        if border_style:
            fragments.append((border_style, "│ "))
            fragments.append(("", pad(line)))
            fragments.append((border_style, " │"))
        else:
            fragments.append(("", f"│ {pad(line)} │"))
        fragments.append(("", "\n"))

    if border_style:
        fragments.append((border_style, bottom))
    else:
        fragments.append(("", bottom))
    fragments.append(("", "\n"))
    return FormattedText(fragments)


def _compact_task_output(output: str, max_chars: int = 240) -> str:
    from cock_code.runtime import _clean_summary_candidate, sanitize_task_output

    output = sanitize_task_output(output)
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return "No useful output returned"

    def _split_chunks(text: str) -> list[str]:
        chunks = re.split(r"\s+#{2,}\s+|\s+---\s+|\s+\*\*\*\s+", text)
        return [chunk.strip() for chunk in chunks if chunk.strip()]

    def _is_heading_like(text: str) -> bool:
        normalized = text.strip().lower().strip(":- ")
        if not normalized:
            return True
        patterns = (
            r"[a-z ]*(review|report|summary)",
            r"(critical|moderate|minor) issues",
            r"changes overview",
            r"executive summary",
            r"questions",
            r"recommendations?",
        )
        return any(re.fullmatch(pattern, normalized, re.IGNORECASE) for pattern in patterns)

    def _extract_review_sentence(text: str) -> str:
        sentence = text.strip()
        sentence = re.sub(r"^#+\s*", "", sentence)
        sentence = re.sub(r"^(?:[A-Za-z]+(?:\s+[A-Za-z]+){0,4}\s+(?:review|report|summary))(?:\s+|[:.-]\s*)", "", sentence, flags=re.IGNORECASE)
        sentence = re.sub(r"^(after reviewing\b[^,.:;]*[,.:;]?\s*)", "", sentence, flags=re.IGNORECASE)
        sentence = re.sub(r"^(now\b[^.]*[.:]?\s*)", "", sentence, flags=re.IGNORECASE)
        sentence = re.sub(r"^(here(?:'s| is)\s+(?:my|the)\s+review:?\s*)", "", sentence, flags=re.IGNORECASE)
        sentence = sentence.strip()
        if not sentence or sentence.lower() in {"now", "summary"}:
            return ""
        return sentence

    candidates: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped in {"---", "***"}:
            continue
        if re.fullmatch(r"#+", stripped):
            continue
        stripped = re.sub(r"^[>*#\-\s]+", "", stripped)
        for chunk in _split_chunks(stripped):
            chunk = re.sub(r"^[>*#\-\s]+", "", chunk).strip()
            if not chunk:
                continue
            if chunk.lower().startswith("outcome:"):
                normalized = _clean_summary_candidate(chunk.partition(":")[2].strip())
            else:
                normalized = _clean_summary_candidate(chunk)
            normalized = _extract_review_sentence(normalized)
            if normalized and not normalized.lower().startswith(("files:", "commands:", "open issues:", "next step:", "findings:")):
                if _is_heading_like(normalized):
                    continue
                candidates.append(normalized)

    for candidate in candidates:
        if candidate:
            return candidate if len(candidate) <= max_chars else f"{candidate[:max_chars].rstrip()}..."
    return "No useful output returned"


def append_task_result_to_context(agent, task_id: str, task_result: dict[str, object]) -> None:
    from cock_code.runtime import sanitize_task_output

    output = sanitize_task_output(str(task_result.get("output", "")).strip())
    status = str(task_result.get("status", "")).strip()
    if status not in {"completed", "cancelled"} or not output:
        return
    with _console_lock:
        if task_id in _injected_task_ids:
            return
        _injected_task_ids.add(task_id)
        history = getattr(agent, "_history", None)
        if not isinstance(history, list):
            return
        history.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": f"[Background task {task_id} completed]\n\n{output}"}],
            }
        )
        history.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": f"Received background task {task_id} result. Ready to continue from that result without redoing the same delegated work."}],
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
    from cock_code.runtime import set_abort_signal

    console = build_console()
    render_banner(console, "ask", config)
    _injected_task_ids.clear()
    install_search_backend(config)
    abort_signal = asyncio.Event()
    question_session: PromptSession[str] = PromptSession()
    set_question_handler(_create_question_handler(question_session, abort_signal))
    interrupted = False
    active_task: asyncio.Task[object] | None = None

    def _cancel_ask_on_sigint(signum: int, frame: object) -> None:
        nonlocal interrupted
        interrupted = True
        abort_signal.set()
        if active_task is not None and not active_task.done():
            active_task.cancel()

    previous = signal.signal(signal.SIGINT, _cancel_ask_on_sigint)
    set_abort_signal(abort_signal)
    requested_agent = find_requested_agent_name(config, prompt)
    if requested_agent:
        try:
            render_agent_panel(console, "Agent Started", requested_agent, "blue")
            active_task = asyncio.create_task(run_named_agent_prompt(config, requested_agent, prompt))
            text = await active_task
            render_agent_panel(console, "Agent Result", text, "blue")
            return 0
        except asyncio.CancelledError:
            render_notice(console, "Interrupted", "Query cancelled.", "yellow")
            return 130
        except Exception as exc:
            render_notice(console, "Error", str(exc), "red")
            return 1
        finally:
            active_task = None
            abort_signal.clear()
            set_abort_signal(None)
            signal.signal(signal.SIGINT, previous)
            clear_question_handler()
            await cancel_background_subagent_tasks()

    agent = create_runtime_agent(config)

    try:
        active_task = asyncio.create_task(
            render_event_stream(console, agent.query(prompt), omit_duplicate_result=True, show_activity_trace=False, abort_signal=abort_signal)
        )
        await active_task
    except asyncio.CancelledError:
        render_notice(console, "Interrupted", "Query cancelled.", "yellow")
        return 130
    except Exception as exc:
        render_notice(console, "Error", str(exc), "red")
        return 1
    finally:
        active_task = None
        abort_signal.clear()
        set_abort_signal(None)
        signal.signal(signal.SIGINT, previous)
        clear_question_handler()
        await cancel_background_subagent_tasks()
        await agent.close()

    if config.persist_session:
        await enforce_session_retention()

    return 0


def _trim_history(history: History, max_size: int) -> None:
    if not isinstance(history, InMemoryHistory):
        return
    strings = history.get_strings()
    overflow = len(strings) - max_size
    for _ in range(max(0, overflow)):
        if history._loaded_strings:
            history._loaded_strings.pop(0)


def _handle_agents_command(console, command, config, team_manager):
    """Handle /agents slash commands."""
    if not command.args:
        agents = config.agents or {}
        render_agents_list(console, agents)
        return

    subcmd = command.args[0]

    if subcmd == "list":
        agents = config.agents or {}
        render_agents_list(console, agents)
        return

    if subcmd == "add" and len(command.args) >= 3:
        name = command.args[1]
        description = " ".join(command.args[2:])
        if config.agents is None:
            config.agents = {}
        if name in config.agents:
            render_notice(console, "Error", f"Agent '{name}' already exists. Use /agents remove {name} first.", "red")
            return
        config.agents[name] = {"description": description, "prompt": description}
        render_notice(console, "Agent Added", f"Agent '{name}' added.", "green")
        return

    if subcmd == "remove" and len(command.args) >= 2:
        name = command.args[1]
        if config.agents and name in config.agents:
            if team_manager.is_active() and name in (team_manager.info().get("members", {}) or {}):
                render_notice(console, "Error", f"Agent '{name}' is in an active team. Use /team stop first.", "red")
                return
            del config.agents[name]
            render_notice(console, "Agent Removed", f"Agent '{name}' removed.", "green")
        else:
            render_notice(console, "Error", f"Agent '{name}' not found.", "red")
        return

    if subcmd == "show" and len(command.args) >= 2:
        name = command.args[1]
        agents = config.agents or {}
        if name in agents:
            render_state(console, f"Agent: {name}", agents[name])
        else:
            render_notice(console, "Error", f"Agent '{name}' not found.", "red")
        return

    render_notice(console, "Error", f"Unknown /agents subcommand: {subcmd}", "red")


async def _handle_team_command(console, command, config, team_manager, agent, abort_signal):
    """Handle /team slash commands. Returns a new agent if resume requires restart."""
    subcmd = command.args[0] if command.args else ""

    if subcmd == "create" and len(command.args) >= 3:
        team_name = command.args[1]
        members = command.args[2:]
        try:
            await team_manager.create_team(team_name, members, config, agent, abort_signal)
            render_notice(console, "Team Created", f"Team '{team_name}' with members: {', '.join(members)}", "green")
        except RuntimeError as exc:
            render_notice(console, "Error", str(exc), "red")
        except Exception as exc:
            render_notice(console, "Error", f"Failed to create team: {exc}", "red")
        return None

    if subcmd == "info":
        info = team_manager.info()
        render_state(console, "Team", info)
        return None

    if subcmd == "stop":
        try:
            await team_manager.close_team(agent)
            render_notice(console, "Team Stopped", "Team disbanded.", "green")
        except Exception as exc:
            render_notice(console, "Error", f"Failed to stop team: {exc}", "red")
        return None

    render_notice(console, "Error", f"Unknown /team subcommand: {subcmd}", "red")
    return None


async def run_chat(config) -> int:
    console = build_console()
    render_banner(console, "chat", config)
    install_search_backend(config)
    _injected_task_ids.clear()
    agent = create_runtime_agent(config)
    interrupted = False
    abort_signal = asyncio.Event()
    team_manager = TeamManager()
    set_runtime_team_bridge(team_manager, agent)
    prompt_session: PromptSession[str] = PromptSession(history=InMemoryHistory())
    question_session: PromptSession[str] = PromptSession()
    _MAX_HISTORY = 50

    def _cancel_query_on_sigint(signum: int, frame: object) -> None:
        nonlocal interrupted
        interrupted = True
        abort_signal.set()
        if _active_query_task is not None and not _active_query_task.done():
            _active_query_task.cancel()

    _active_query_task: asyncio.Task[None] | None = None

    async def _run_query_with_interrupt(events, **kwargs) -> None:
        nonlocal interrupted, _active_query_task
        from cock_code.runtime import set_abort_signal
        abort_signal.clear()
        set_abort_signal(abort_signal)
        previous = signal.signal(signal.SIGINT, _cancel_query_on_sigint)
        _active_query_task = asyncio.create_task(
            render_event_stream(console, events, abort_signal=abort_signal, **kwargs)
        )
        try:
            await _active_query_task
        except asyncio.CancelledError:
            if hasattr(agent, "_client") and agent._client:
                with contextlib.suppress(Exception):
                    await agent._client.close()
                agent._client = None
            if hasattr(agent, "_provider"):
                agent._provider = None
            if hasattr(agent, "_engine"):
                agent._engine = None
            agent._initialized = False
            if team_manager.is_active():
                await team_manager.ensure_orchestrator_team_state(agent)
            render_notice(console, "Interrupted", "Query cancelled.", "yellow")
            interrupted = False
        except Exception as exc:
            render_notice(console, "Error", str(exc), "red")
        finally:
            _active_query_task = None
            abort_signal.clear()
            set_abort_signal(None)
            signal.signal(signal.SIGINT, previous)

    def _poll_and_render_notifications() -> bool:
        rendered = False
        for note in read_background_notifications():
            if note.get("type") == "background_task_completed":
                _render_task_notification(console, agent, note, prompt_session)
                rendered = True
        return rendered

    async def _prompt_input(session: PromptSession[str]) -> str | None:
        try:
            with patch_stdout():
                return await session.prompt_async(_prompt_label)
        except KeyboardInterrupt:
            return None
        except EOFError:
            raise

    async def prompt_once(session: PromptSession[str]) -> str | None:
        prompt_task = asyncio.create_task(_prompt_input(session))
        try:
            while True:
                try:
                    user_input = await asyncio.wait_for(asyncio.shield(prompt_task), timeout=0.1)
                    if user_input is None:
                        return None
                    _trim_history(session.history, _MAX_HISTORY)
                    return user_input
                except TimeoutError:
                    _poll_and_render_notifications()
                    continue
        finally:
            if not prompt_task.done():
                prompt_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await prompt_task

    set_question_handler(_create_question_handler(question_session, abort_signal))

    try:
        while True:
            if interrupted:
                if _active_query_task is not None and not _active_query_task.done():
                    break
                interrupted = False
            _poll_and_render_notifications()
            try:
                user_input = await prompt_once(prompt_session)
                if user_input is None:
                    continue
            except KeyboardInterrupt:
                continue
            except EOFError:
                interrupted = True
                break
            _poll_and_render_notifications()
            command = parse_chat_command(user_input)
            if command.name == "exit":
                break
            if command.name == "clear":
                agent.clear()
                if team_manager.is_active():
                    await team_manager.clear()
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
                from cock_code.runtime import _notified_task_ids, _notified_task_ids_lock
                with _notified_task_ids_lock:
                    _notified_task_ids.add(command.args[0])
                append_task_result_to_context(agent, command.args[0], task_result)
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
            if command.name == "agents":
                _handle_agents_command(console, command, config, team_manager)
                continue
            if command.name == "team" and command.args:
                await _handle_team_command(console, command, config, team_manager, agent, abort_signal)
                continue
            available_skills = set(list_skill_names())
            if command.name in available_skills:
                render_agent_panel(console, "Skill Started", command.name, "blue")
                await _run_query_with_interrupt(
                    stream_skill_events(config, agent, command.name, " ".join(command.args)),
                    omit_duplicate_result=True,
                    show_activity_trace=True,
                )
                _poll_and_render_notifications()
                continue
            if user_input.startswith("/"):
                render_notice(console, "Unknown command", user_input, "red")
                continue

            requested_agent = find_requested_agent_name(config, user_input)
            if requested_agent:
                render_agent_panel(console, "Agent Started", requested_agent, "blue")
                await _run_query_with_interrupt(
                    stream_named_agent_events(config, requested_agent, user_input),
                    omit_duplicate_result=True,
                    show_activity_trace=True,
                )
                _poll_and_render_notifications()
                continue

            if team_manager.is_active():
                await team_manager.ensure_orchestrator_team_state(agent)

            await _run_query_with_interrupt(
                agent.query(user_input),
                omit_duplicate_result=True,
                show_activity_trace=True,
            )
            _poll_and_render_notifications()
            continue
    finally:
        set_runtime_team_bridge(None, None)
        if team_manager.is_active():
            with contextlib.suppress(Exception):
                await team_manager.close_team(agent)
        clear_question_handler()
        await cancel_background_subagent_tasks()
        if interrupted:
            try:
                await asyncio.wait_for(agent.close(), timeout=1.0)
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
