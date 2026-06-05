from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import re
import signal
import sys
import threading

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.shortcuts import print_formatted_text
from pathlib import Path

from prompt_toolkit.history import FileHistory
from prompt_toolkit.history import History
from prompt_toolkit.patch_stdout import patch_stdout

import httpx

from rooster_code.chat import parse_chat_command
from rooster_code.config import config_from_namespace, save_agents_file
from rooster_code.goal import set_goal, clear_goal, list_goals, get_active_goal, get_goal_check_prompt
from rooster_code.file_context import (
    resolve_at_references,
    _build_context_block,
    AtFileCompleter,
    AtFileError,
)
from rooster_code.team import TeamManager, set_runtime_team_bridge
from rooster_code.rendering import (
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
    render_tool_table,
    render_transcript,
)
from rooster_code.cli_daemon import _ask_via_daemon, _daemon_is_reachable, _handle_daemon_command

log = logging.getLogger("rooster.cli")


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
    from rooster_code.runtime import create_runtime_agent as runtime_create_agent

    return runtime_create_agent(config)


def find_requested_agent_name(config, prompt: str):
    from rooster_code.runtime import find_requested_agent_name as runtime_find_requested_agent_name

    return runtime_find_requested_agent_name(config, prompt)


async def run_named_agent_prompt(config, agent_name: str, prompt: str) -> str:
    from rooster_code.runtime import run_named_agent_prompt as runtime_run_named_agent_prompt

    return await runtime_run_named_agent_prompt(config, agent_name, prompt)


async def stream_named_agent_events(config, agent_name: str, prompt: str):
    from rooster_code.runtime import stream_named_agent_events as runtime_stream_named_agent_events

    async for event in runtime_stream_named_agent_events(config, agent_name, prompt):
        yield event


async def stream_skill_events(config, agent, skill_name: str, args: str):
    from rooster_code.runtime import stream_skill_events as runtime_stream_skill_events

    async for event in runtime_stream_skill_events(config, agent, skill_name, args):
        yield event


async def compact_current_session(agent):
    from rooster_code.runtime import compact_current_session as runtime_compact_current_session

    return await runtime_compact_current_session(agent)


def list_skill_names() -> list[str]:
    from rooster_code.runtime import list_skill_names as runtime_list_skill_names

    return runtime_list_skill_names()


async def list_sessions():
    from rooster_code.runtime import list_sessions as runtime_list_sessions

    return await runtime_list_sessions()


async def get_session_messages(session_id: str):
    from rooster_code.runtime import get_session_messages as runtime_get_session_messages

    return await runtime_get_session_messages(session_id)


async def get_session_info(session_id: str):
    from rooster_code.runtime import get_session_info as runtime_get_session_info

    return await runtime_get_session_info(session_id)


async def delete_session(session_id: str):
    from rooster_code.runtime import delete_session as runtime_delete_session

    return await runtime_delete_session(session_id)


async def fork_session(session_id: str, new_id: str | None):
    from rooster_code.runtime import fork_session as runtime_fork_session

    return await runtime_fork_session(session_id, new_id)


async def enforce_session_retention(limit: int = 20):
    from rooster_code.runtime import enforce_session_retention as runtime_enforce_session_retention

    return await runtime_enforce_session_retention(limit)


async def rename_session(session_id: str, title: str):
    from rooster_code.runtime import rename_session as runtime_rename_session

    return await runtime_rename_session(session_id, title)


async def tag_session(session_id: str, tags: list[str]):
    from rooster_code.runtime import tag_session as runtime_tag_session

    return await runtime_tag_session(session_id, tags)


def get_state_snapshot(name: str, agent_name: str | None = None):
    from rooster_code.runtime import get_state_snapshot as runtime_get_state_snapshot

    return runtime_get_state_snapshot(name, agent_name)


async def get_task_output(task_id: str) -> str:
    from rooster_code.runtime import get_task_output as runtime_get_task_output

    return await runtime_get_task_output(task_id)


async def stop_task(task_id: str) -> bool:
    from rooster_code.runtime import stop_task as runtime_stop_task

    return await runtime_stop_task(task_id)


async def start_background_agent_task(config, agent_name: str, prompt: str) -> str:
    from rooster_code.runtime import start_background_agent_task as runtime_start_background_agent_task

    return await runtime_start_background_agent_task(config, agent_name, prompt)


async def wait_for_task(task_id: str) -> dict[str, object]:
    from rooster_code.runtime import wait_for_task as runtime_wait_for_task

    return await runtime_wait_for_task(task_id)


_console_lock = threading.RLock()
_prompt_label = "rooster-code> "
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
    from rooster_code.runtime import sanitize_task_output

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
        if len(text) > width:
            return (text[: width - 1] + "…").ljust(width)
        return text.ljust(width)

    top = f"╭─ {title} {'─' * max(width - len(title) - 1, 0)}╮"
    bottom = f"╰{'─' * (width + 2)}╯"

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
    from rooster_code.runtime import sanitize_task_output

    output = sanitize_task_output(output)
    lines = [line.strip() for line in output.splitlines()]

    for index, line in enumerate(lines):
        if not line.lower().startswith("**top-level files:**"):
            continue
        table_rows: list[str] = []
        for row in lines[index + 1 :]:
            if not row:
                if table_rows:
                    break
                continue
            if row.startswith("|") and row.endswith("|"):
                if row == "|------|------|":
                    continue
                if row.lower() == "| file | size |":
                    continue
                table_rows.append(row)
                continue
            if table_rows:
                break
        if table_rows:
            summary = " ".join(table_rows)
            return summary if len(summary) <= max_chars else f"{summary[:max_chars].rstrip()}..."

    for line in reversed(output.splitlines()):
        line = line.strip()
        if not line:
            continue
        if line in {"---", "***"}:
            continue
        if re.fullmatch(r"#+", line):
            continue
        if re.fullmatch(r"</?\w[\w-]*>", line):
            continue
        if line == "```" or line.startswith("```"):
            continue
        if line.startswith("|") and line.endswith("|"):
            continue
        if line.lower().startswith("outcome:"):
            line = line.partition(":")[2].strip()
        if line and not re.match(r"(?i)(note|n\.?b\.?|tip|hint)\b", line):
            return line if len(line) <= max_chars else f"{line[:max_chars].rstrip()}..."
    return "No useful output returned"


def append_task_result_to_context(agent, task_id: str, task_result: dict[str, object]) -> None:
    from rooster_code.runtime import sanitize_task_output

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
                "content": [{"type": "text", "text": f"[Background task {task_id} {status}]\n\n{output}"}],
            }
        )
        assistant_text = (
            f"Received background task {task_id} result. Ready to continue from that result without redoing the same delegated work."
            if status == "completed"
            else f"Received background task {task_id} cancellation/failure. Do not assume the delegated work completed; decide whether to retry, take over, or change approach based on the failure output."
        )
        history.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_text}],
            }
        )


def _inject_goal_context(agent, goal_text: str) -> None:
    history = getattr(agent, "_history", None)
    if not isinstance(history, list):
        return
    history.append({
        "role": "user",
        "content": [{"type": "text", "text": f"[System: active goal — {goal_text}]"}],
    })
    history.append({
        "role": "assistant",
        "content": [{"type": "text", "text": f"Acknowledged. Working toward: {goal_text}"}],
    })
    _update_agent_goal_prompt(agent, goal_text)


def _clear_goal_context(agent) -> None:
    history = getattr(agent, "_history", None)
    if not isinstance(history, list):
        return
    history.append({
        "role": "user",
        "content": [{"type": "text", "text": "[System: goal cleared]"}],
    })
    history.append({
        "role": "assistant",
        "content": [{"type": "text", "text": "Understood. No active goal."}],
    })
    _update_agent_goal_prompt(agent, None)


def _check_goal_met(agent) -> bool:
    history = getattr(agent, "_history", None)
    if not isinstance(history, list):
        return False
    for msg in reversed(history):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", [])
        for block in content if isinstance(content, list) else []:
            if isinstance(block, dict) and block.get("type") == "text":
                if "GOAL_MET" in str(block.get("text", "")):
                    return True
    return False


def _update_agent_goal_prompt(agent, goal_text: str | None) -> None:
    """Update append_system_prompt to reflect the current goal."""
    opts = getattr(agent, "_options", None)
    if opts is None:
        return
    try:
        from rooster_code.goal import get_active_goal as _get_active
    except ImportError:
        return
    active = _get_active()
    # Rebuild the goal section that _agent_context_prompt would produce
    goal_section = ""
    if active:
        goal_section = (
            f"\n\n# Current Goal\n"
            f"You are working toward the following goal: {active.text}\n"
            f"Use /goal check to assess progress. Do not autonomously loop; wait for the user to check."
        )
    # Strip any previous goal section and append the current one
    current = getattr(opts, "append_system_prompt", "") or ""
    # Remove any existing "# Current Goal" section
    import re
    current = re.sub(r"\n*# Current Goal\n.*?(?=\n# |\Z)", "", current, flags=re.DOTALL)
    opts.append_system_prompt = (current + goal_section).strip()




def _pending_dispatch_notice() -> str:
    """Return a notice about in-flight tasks (team dispatches and /bg), or empty string if none."""
    from open_agent_sdk import get_all_tasks
    tasks = get_all_tasks()
    in_flight = [
        (tid, t.get("subject", tid))
        for tid, t in tasks.items()
        if str(t.get("status", "")).lower() == "in_progress"
    ]
    if not in_flight:
        return ""
    lines = ["[System notice: the following dispatched tasks are still in progress and their results are NOT yet available. Do not answer questions that depend on these results until they complete:]"]
    for tid, subject in in_flight:
        lines.append(f"  - {tid}: {subject}")
    return "\n".join(lines) + "\n\n"


def read_background_notifications() -> list[dict[str, object]]:
    from rooster_code.runtime import read_background_notifications as runtime_read_background_notifications

    return runtime_read_background_notifications()


async def cancel_background_subagent_tasks() -> None:
    from rooster_code.runtime import cancel_background_subagent_tasks as runtime_cancel_background_subagent_tasks

    await runtime_cancel_background_subagent_tasks()


def list_tool_names() -> list[str]:
    from rooster_code.runtime import list_tool_names as runtime_list_tool_names

    return runtime_list_tool_names()


from rooster_code.cli_parser import build_parser


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


async def run_ask(prompt: str, config) -> int:
    from rooster_code.runtime import set_abort_signal

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
            log.exception("ask agent query failed [agent=%s session=%s]: %s", requested_agent, config.resume or config.session_id or "new", exc)
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
    if hasattr(agent, "_initialize") and callable(agent._initialize):
        await agent._initialize()
    from rooster_code.runtime import rehydrate_tasks_from_history
    rehydrate_tasks_from_history(agent)

    try:
        active_task = asyncio.create_task(
            render_event_stream(console, agent.query(prompt), omit_duplicate_result=True, show_activity_trace=False, abort_signal=abort_signal)
        )
        await active_task
    except asyncio.CancelledError:
        render_notice(console, "Interrupted", "Query cancelled.", "yellow")
        return 130
    except Exception as exc:
        log.exception("ask query failed [session=%s model=%s]: %s", config.resume or config.session_id or "new", config.model or "default", exc)
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
    strings = history.get_strings()
    overflow = len(strings) - max_size
    if overflow <= 0:
        return
    for _ in range(overflow):
        if history._loaded_strings:
            history._loaded_strings.pop(0)
    if isinstance(history, FileHistory):
        with open(history.filename, "w") as fh:
            fh.write("\n".join(history._loaded_strings) + "\n")


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

    if subcmd == "add" and team_manager.is_active():
        render_notice(console, "Error", "Cannot add agents while a team is active. Use /team stop first.", "red")
        return

    if subcmd == "add" and len(command.args) >= 3:
        name = command.args[1]
        description = " ".join(command.args[2:])
        if config.agents is None:
            config.agents = {}
        if name in config.agents:
            render_notice(console, "Error", f"Agent '{name}' already exists. Use /agents remove {name} first.", "red")
            return
        try:
            new_agents = {**config.agents, name: {"description": description, "prompt": description}}
            save_agents_file(new_agents)
            config.agents = new_agents
        except Exception as exc:
            render_notice(console, "Error", f"Could not save agent: {exc}", "red")
            return
        render_notice(console, "Agent Added", f"Agent '{name}' added.", "green")
        return

    if subcmd == "remove" and len(command.args) >= 2:
        name = command.args[1]
        if config.agents and name in config.agents:
            if team_manager.is_active() and name in (team_manager.info().get("members", {}) or {}):
                render_notice(console, "Error", f"Agent '{name}' is in an active team. Use /team stop first.", "red")
                return
            try:
                new_agents = {k: v for k, v in config.agents.items() if k != name}
                save_agents_file(new_agents)
                config.agents = new_agents
            except Exception as exc:
                render_notice(console, "Error", f"Could not save agent: {exc}", "red")
                return
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
    _rehydrated = False
    _goal_loop_active = False
    _goal_loop_turns = 0
    _MAX_GOAL_TURNS = 20
    agent = create_runtime_agent(config)
    interrupted = False
    abort_signal = asyncio.Event()
    team_manager = TeamManager()
    set_runtime_team_bridge(team_manager, agent)
    history_path = Path.home() / ".rooster-code" / "history"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_session: PromptSession[str] = PromptSession(
        history=FileHistory(str(history_path)),
        completer=AtFileCompleter(cwd=config.cwd or "."),
    )
    question_session: PromptSession[str] = PromptSession()
    _MAX_HISTORY = 50

    def _cancel_query_on_sigint(signum: int, frame: object) -> None:
        nonlocal interrupted
        from rooster_code.runtime import _cancel_bg_tasks_sync
        interrupted = True
        abort_signal.set()
        if _active_query_task is not None and not _active_query_task.done():
            _active_query_task.cancel()
        _cancel_bg_tasks_sync()

    _active_query_task: asyncio.Task[None] | None = None

    signal.signal(signal.SIGINT, _cancel_query_on_sigint)

    async def _run_query_with_interrupt(events, **kwargs) -> None:
        nonlocal interrupted, _active_query_task
        from rooster_code.runtime import set_abort_signal
        aborted = False
        abort_signal.clear()
        set_abort_signal(abort_signal)
        previous = signal.signal(signal.SIGINT, _cancel_query_on_sigint)
        _active_query_task = asyncio.create_task(
            render_event_stream(console, events, abort_signal=abort_signal, **kwargs)
        )
        try:
            await _active_query_task
        except asyncio.CancelledError:
            aborted = True
            interrupted = False
        except Exception as exc:
            log.exception("chat query failed [session=%s model=%s]: %s", config.resume or config.session_id or "new", config.model or "default", exc)
            render_notice(console, "Error", str(exc), "red")
        finally:
            if aborted or abort_signal.is_set():
                render_notice(console, "Interrupted", "Query cancelled.", "yellow")
                interrupted = False
                await cancel_background_subagent_tasks()
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

    if hasattr(agent, "_initialize") and callable(agent._initialize):
        await agent._initialize()
    from rooster_code.runtime import rehydrate_tasks_from_history
    rehydrate_tasks_from_history(agent)
    _rehydrated = True

    try:
        while True:
            if interrupted:
                if _active_query_task is not None and not _active_query_task.done():
                    break
                render_notice(console, "Cancelled", "Background tasks cancelled.", "yellow")
                interrupted = False
            _poll_and_render_notifications()
            if _goal_loop_active:
                if interrupted or _goal_loop_turns >= _MAX_GOAL_TURNS:
                    if _goal_loop_turns >= _MAX_GOAL_TURNS:
                        render_notice(console, "Goal Loop", f"Stopped after {_MAX_GOAL_TURNS} turns.", "yellow")
                    _goal_loop_active = False
                    _goal_loop_turns = 0
                    continue
                if _goal_loop_turns > 0 and _check_goal_met(agent):
                    render_notice(console, "Goal Loop", "Goal met! Loop stopped.", "green")
                    _goal_loop_active = False
                    _goal_loop_turns = 0
                    continue
                _goal_loop_turns += 1
                from rooster_code.goal import get_active_goal as _loop_goal
                active = _loop_goal()
                if active is None:
                    render_notice(console, "Goal Loop", "No active goal. Loop stopped.", "yellow")
                    _goal_loop_active = False
                    _goal_loop_turns = 0
                    continue
                goal_text = active.text
                user_input = (
                    f"Continue working toward the goal: {goal_text}. "
                    f"If the goal is completely met, start your response with GOAL_MET."
                )
                render_notice(console, "Goal Loop", f"Turn {_goal_loop_turns}/{_MAX_GOAL_TURNS}", "blue")
            else:
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
                from rooster_code.runtime import _notified_task_ids, _notified_task_ids_lock
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
                with contextlib.suppress(Exception):
                    await agent.close()
                config.resume = command.args[0]
                agent = create_runtime_agent(config)
                if hasattr(agent, "_initialize") and callable(agent._initialize):
                    await agent._initialize()
                from rooster_code.runtime import rehydrate_tasks_from_history, _injected_task_ids_rehydrated
                _injected_task_ids_rehydrated.clear()
                rehydrate_tasks_from_history(agent)
                set_runtime_team_bridge(team_manager, agent)
                if team_manager.is_active():
                    await team_manager.ensure_orchestrator_team_state(agent)
                render_notice(console, "Session", f"Resumed {command.args[0]}", "green")
                continue
            if command.name == "agents":
                _handle_agents_command(console, command, config, team_manager)
                continue
            if command.name == "team" and command.args:
                await _handle_team_command(console, command, config, team_manager, agent, abort_signal)
                continue
            if command.name == "goal" and command.args:
                subcmd = command.args[0]
                if subcmd == "set":
                    if len(command.args) < 2:
                        render_notice(console, "Error", "Usage: /goal set <text>", "red")
                        continue
                    text = " ".join(command.args[1:])
                    g = set_goal(text)
                    render_notice(console, "Goal Set", f"[bold]Goal:[/bold] {g.text}\n[dim]ID: {g.id}[/dim]", "green")
                    _inject_goal_context(agent, g.text)
                elif subcmd == "show":
                    active = get_active_goal()
                    if active:
                        render_state(console, "Active Goal", {"id": active.id, "text": active.text, "status": active.status})
                    else:
                        render_notice(console, "Goal", "No active goal.", "yellow")
                elif subcmd == "clear":
                    cleared = clear_goal()
                    if cleared:
                        render_notice(console, "Goal Cleared", f"[bold]{cleared.text}[/bold]\n[dim]Marked as completed.[/dim]", "green")
                        _clear_goal_context(agent)
                    else:
                        render_notice(console, "Goal", "No active goal to clear.", "yellow")
                elif subcmd == "list":
                    goals = list_goals()
                    if goals:
                        render_state(console, "Goals", {g.id: f"[{g.status.upper()}] {g.text}" for g in goals})
                    else:
                        render_notice(console, "Goals", "No goals found.", "yellow")
                elif subcmd == "check":
                    prompt = get_goal_check_prompt()
                    if prompt is None:
                        render_notice(console, "Goal", "No active goal to check.", "yellow")
                    else:
                        render_notice(console, "Goal Check", "Assessing goal progress...", "blue")
                        await _run_query_with_interrupt(
                            agent.query(prompt),
                            omit_duplicate_result=True,
                            show_activity_trace=True,
                        )
                elif subcmd == "work":
                    active = get_active_goal()
                    if active is None:
                        render_notice(console, "Goal", "No active goal. Set one with /goal set first.", "yellow")
                    else:
                        _goal_loop_active = True
                        _goal_loop_turns = 0
                        render_notice(console, "Goal Loop", f"Working on: {active.text}\nCtrl+C to stop, max {_MAX_GOAL_TURNS} turns.", "blue")
                elif subcmd == "stop":
                    _goal_loop_active = False
                    _goal_loop_turns = 0
                    render_notice(console, "Goal Loop", "Stopped.", "yellow")
                else:
                    render_notice(console, "Error", f"Unknown /goal subcommand: {subcmd}", "red")
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

            # Resolve @file references in user input
            try:
                cleaned_input, files = resolve_at_references(
                    user_input, config.cwd or "."
                )
            except (AtFileError, OSError) as e:
                render_notice(console, "@ Error", str(e), "red")
                continue

            # Build effective input with context block and pending notices
            pending_notice = _pending_dispatch_notice()
            parts: list[str] = []
            if pending_notice:
                parts.append(pending_notice)
            if files:
                parts.append(_build_context_block(files))
                parts.append(f"[User message:]\n{cleaned_input}")
            else:
                parts.append(cleaned_input)
            effective_input = "".join(parts)

            await _run_query_with_interrupt(
                agent.query(effective_input),
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
            if getattr(args, "daemon", False):
                return _ask_via_daemon(args.prompt, args)
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
            if info is None:
                console.print(f"[red]Session '{args.session_id}' not found.[/red]")
                return 1
            render_session_info(console, info)
            return 0

        if args.command == "sessions" and args.sessions_command == "delete":
            console = build_console()
            if _daemon_is_reachable():
                from rooster_code.daemon import daemon_session_delete
                r = asyncio.run(daemon_session_delete(args.session_id))
                deleted = r.get("type") == "session_deleted"
            else:
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

        if args.command == "skills":
            console = build_console()
            from open_agent_sdk.skills.bundled import init_bundled_skills
            from open_agent_sdk.skills.registry import register_skill
            from rooster_code.runtime import _build_filesystem_skill_definition, _resolve_skills_dir
            from rooster_code.config import resolve_runtime_env
            init_bundled_skills()
            config = resolve_runtime_env(os.environ, cwd=".")
            skills_dir = _resolve_skills_dir(config)
            if skills_dir and skills_dir.exists():
                for skill_dir in sorted(skills_dir.iterdir()):
                    if skill_dir.is_dir():
                        skill_def = _build_filesystem_skill_definition(skill_dir)
                        if skill_def:
                            register_skill(skill_def)
            skills = list_skill_names()
            if getattr(args, "json", False):
                import json
                console.print(json.dumps(skills))
            else:
                render_state(console, "Skills", {"skills": skills})
            return 0

        if args.command == "agents" and args.agents_command == "list":
            console = build_console()
            from rooster_code.config import resolve_runtime_env
            config = resolve_runtime_env(os.environ, cwd=".")
            render_agents_list(console, config.agents)
            return 0

        if args.command == "sessions" and args.sessions_command == "append":
            console = build_console()
            from open_agent_sdk.session import append_to_session, load_session
            existing = asyncio.run(load_session(args.session_id))
            if existing is None:
                console.print(f"[red]Error:[/red] session [bold]{args.session_id}[/bold] does not exist")
                return 1
            asyncio.run(append_to_session(args.session_id, {"role": args.role, "content": args.message}))
            render_state(console, "Session Message Appended", {"session_id": args.session_id, "appended": True})
            return 0

        if args.command == "cron" and args.cron_command == "list":
            console = build_console()
            from rooster_code.runtime_session import _read_cron_jobs
            render_state(console, "Cron Jobs", list(_read_cron_jobs().values()))
            return 0

        if args.command == "cron" and args.cron_command == "show":
            console = build_console()
            from rooster_code.runtime_session import _read_cron_jobs
            jobs = _read_cron_jobs()
            if args.job_id not in jobs:
                console.print(f"[red]Error:[/red] cron job [bold]{args.job_id}[/bold] not found")
                return 1
            render_state(console, f"Cron Job {args.job_id}", jobs[args.job_id])
            return 0

        if args.command == "cron" and args.cron_command == "delete":
            console = build_console()
            from rooster_code.daemon import daemon_cron_delete
            try:
                result = asyncio.run(daemon_cron_delete(args.job_id))
            except (ConnectionRefusedError, FileNotFoundError, OSError):
                from rooster_code.daemon import _SOCKET_PATH
                console.print(f"[red]Error:[/red] daemon not reachable at {_SOCKET_PATH} (start with `rooster-daemon`)")
                return 1
            except Exception as exc:
                console.print(f"[red]Error:[/red] {exc}")
                return 1
            if result.get("type") == "error":
                console.print(f"[red]Error:[/red] {result['message']}")
                return 1
            console.print(f"[green]Deleted[/green] cron job [bold]{args.job_id}[/bold]")
            return 0

        if args.command == "state" and args.state_command in {"tasks", "teams", "mailboxes", "config", "cron", "plan", "todos"}:
            console = build_console()
            render_state(console, args.state_command.title(), get_state_snapshot(args.state_command, getattr(args, "agent", None)))
            return 0

        if args.command == "daemon":
            return _handle_daemon_command(args)

        parser.print_help()
        return 0
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
