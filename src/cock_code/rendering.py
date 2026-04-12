from __future__ import annotations

import asyncio
import re

from collections.abc import Mapping
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, cast

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.pretty import Pretty
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from cock_code.config import RuntimeConfig

if TYPE_CHECKING:
    from open_agent_sdk import SDKMessage


def summarize_tool_result(text: str, max_chars: int = 160) -> str:
    if len(text) <= max_chars:
        return text

    return f"{text[:max_chars]}..."


def compact_tool_result(text: str, max_chars: int = 220) -> str:
    from cock_code.runtime import sanitize_task_output

    text = sanitize_task_output(text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "No useful output returned"

    def normalize(line: str) -> str:
        line = re.sub(r"^Outcome:\s*", "", line, flags=re.IGNORECASE).strip()
        line = re.sub(r"^#+\s*", "", line)
        line = re.sub(r"^[>*\-\s]+", "", line)
        line = " ".join(line.split())
        line = re.sub(r"^(here(?:'s| is)\s+(?:my|the)\s+review:?\s*)", "", line, flags=re.IGNORECASE)
        line = re.sub(r"^(after reviewing\b[^,.:;]*[,.:;]?\s*)", "", line, flags=re.IGNORECASE)
        line = re.sub(r"^(now\b[^.]*[.:]?\s*)", "", line, flags=re.IGNORECASE)
        match = re.search(r"\b(let me\b|i'll\b|i will\b)", line, re.IGNORECASE)
        if match:
            line = line[: match.start()].rstrip(" :;-.,")
        return line.strip()

    for line in lines:
        candidate = normalize(line)
        if not candidate:
            continue
        lowered = candidate.lower().strip(":- ")
        if lowered in {
            "review",
            "report",
            "summary",
            "code review",
            "review report",
            "code review summary",
            "performance review report",
            "security review report",
            "correctness review",
            "critical issues",
            "changes summary",
            "executive summary",
            "questions",
            "suggestions",
        }:
            continue
        return candidate if len(candidate) <= max_chars else f"{candidate[:max_chars].rstrip()}..."

    return "No useful output returned"


def build_console() -> Console:
    return Console(highlight=False)


def build_rooster_head() -> Text:
    rows = [
        [("              ", ""), ("▄", "bold red"), ("▄", "bold red"), ("▄", "bold red")],
        [("           ", ""), ("▄████", "bold red"), ("▄", "bold red")],
        [("         ", ""), ("▄████████", "bold red"), ("▄", "bold red")],
        [("       ", ""), ("▄████████████", "bold red"), ("▄", "bold red")],
        [("      ", ""), ("████████", "red"), ("  ", ""), ("◉", "bold white"), ("  ", ""), ("████", "red"), (" ", ""), ("▶▶", "bold yellow")],
        [("      ", ""), ("████████████████", "red")],
        [("       ", ""), ("▀████████████", "red"), ("▄", "bold red")],
        [("         ", ""), ("▀██████", "red"), ("▄██", "bold red")],
        [("            ", ""), ("▀██", "red"), ("██", "bold red")],
    ]

    text = Text()
    for row in rows:
        for segment, style in row:
            text.append(segment, style=style or None)
        text.append("\n")
    return text


def build_cock_code_wordmark() -> Text:
    wordmark = Text()
    lines = [
        "  ██████  ██████   ██████ ██   ██     ██████  ██████  ██████  ███████ ",
        " ██      ██    ██ ██      ██  ██     ██      ██    ██ ██   ██ ██      ",
        " ██      ██    ██ ██      █████      ██      ██    ██ ██   ██ █████   ",
        " ██      ██    ██ ██      ██  ██     ██      ██    ██ ██   ██ ██      ",
        "  ██████  ██████   ██████ ██   ██     ██████  ██████  ██████  ███████ ",
    ]
    for line in lines:
        wordmark.append(line, style="bold white on red")
        wordmark.append("\n")
    return wordmark


def render_banner(console: Console, mode: str, config: RuntimeConfig) -> None:
    console.print(build_rooster_head())
    console.print(build_cock_code_wordmark())
    console.print(Rule(f"[bold cyan]COCK-CODE {mode.upper()}[/]", align="left", style="cyan"))

    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="bold blue")
    table.add_column()
    table.add_row("Model", config.model or "default")
    table.add_row("CWD", config.cwd or ".")
    table.add_row("Session", config.resume or config.session_id or "new")
    console.print(table)


def extract_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, Mapping):
                mapping = cast(Mapping[str, object], item)
                if mapping.get("type") == "text":
                    parts.append(str(mapping.get("text", "")))
        return "\n".join(part for part in parts if part)
    if isinstance(value, Mapping):
        mapping = cast(Mapping[str, object], value)
        if mapping.get("type") == "text":
            return str(mapping.get("text", ""))
    return str(value)


def render_text_panel(console: Console, title: str, text: str, style: str) -> None:
    body = Markdown(text) if text.strip() else "[dim]No content[/dim]"
    console.print(Panel(body, title=title, border_style=style, expand=True))


def render_help(console: Console) -> None:
    table = Table(title="Help", show_header=True, box=None, pad_edge=False)
    table.add_column("Command", style="bold cyan")
    table.add_column("What it does")
    table.add_row("/help", "Show available chat commands")
    table.add_row("/clear", "Clear current agent history")
    table.add_row("/compact", "Summarize the current chat history")
    table.add_row("/model <name>", "Switch models")
    table.add_row("/permission <mode>", "Update permission mode")
    table.add_row("/tools", "Show available tools")
    table.add_row("/skills", "Show available skills")
    table.add_row("/tasks", "Show background task store")
    table.add_row("/bg <name> <prompt>", "Start a background subagent task")
    table.add_row("/agent-bg <name> <prompt>", "Start a background subagent task")
    table.add_row("/task-output <id>", "Show a task output")
    table.add_row("/task-stop <id>", "Stop a background task")
    table.add_row("/wait <id>", "Wait for a background task to finish")
    table.add_row("/sessions", "Show saved sessions")
    table.add_row("/status", "Show current chat runtime state")
    table.add_row("/agents", "List configured agents")
    table.add_row("/agents add <name> <desc>", "Add an agent definition")
    table.add_row("/agents remove <name>", "Remove an agent definition")
    table.add_row("/agents show <name>", "Show agent definition details")
    table.add_row("/team create <name> <members>", "Create a team with named agents")
    table.add_row("/team info", "Show team members and status")
    table.add_row("/team stop", "Disband team, close all member agents")
    table.add_row("/resume <id>", "Resume another session")
    table.add_row("/exit", "Exit chat")
    console.print(Panel(table, title="Help", border_style="blue", expand=True))


def render_notice(console: Console, title: str, message: str, style: str = "yellow") -> None:
    console.print(Panel(message, title=title, border_style=style, expand=True))


def summarize_thinking(text: str, max_chars: int = 280) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}..."


def extract_thinking(message: object) -> str:
    if not hasattr(message, "content"):
        return ""
    content = getattr(message, "content")
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if isinstance(item, Mapping) and item.get("type") == "thinking":
            parts.append(str(item.get("thinking", "")))
    return "\n".join(part for part in parts if part)


def extract_unified_diff(text: str) -> str:
    marker = "\n--- "
    if text.startswith("--- "):
        return text
    if marker in text:
        return text[text.index(marker) + 1 :]
    return ""


def render_diff_panel(console: Console, title: str, diff_text: str) -> None:
    console.print(Panel(Syntax(diff_text, "diff", word_wrap=True), title=title, border_style="green", expand=True))


def render_agent_panel(console: Console, title: str, text: str, style: str = "blue") -> None:
    render_text_panel(console, title, text, style)


def render_activity_trace(console: Console, entries: list[Mapping[str, object]]) -> None:
    for entry in entries:
        action = str(entry.get("action", "Working"))
        tool = str(entry.get("tool", ""))
        target = str(entry.get("target", ""))
        parts = [action]
        if tool:
            parts.append(tool)
        if target:
            parts.append(target)
        console.print(Text(f"Activity: {' · '.join(parts)}", style="dim"))


async def render_event_stream(
    console: Console,
    events: AsyncIterator[SDKMessage],
    omit_duplicate_result: bool = False,
    show_activity_trace: bool = False,
    abort_signal: asyncio.Event | None = None,
) -> None:
    last_assistant_text = ""
    async for event in events:
        if abort_signal is not None and abort_signal.is_set():
            break
        if show_activity_trace:
            activity_trace = event.system_data.get("activity_trace", [])
            if isinstance(activity_trace, list):
                trace_entries = [entry for entry in activity_trace if isinstance(entry, Mapping)]
                if trace_entries:
                    render_activity_trace(console, trace_entries)
                elif event.type.value == "tool_result" and event.tool_name:
                    render_activity_trace(console, [{"action": "Running tool", "tool": event.tool_name}])
        thinking_text = summarize_thinking(extract_thinking(event.message))
        if thinking_text:
            render_text_panel(console, "Thinking", thinking_text, "magenta")

        if event.type.value == "tool_result":
            if event.is_error and event.result_content:
                render_notice(console, event.tool_name or "Tool Error", event.result_content, "red")
                continue
            if event.tool_name == "Agent" and event.result_content:
                render_agent_panel(console, "Agent Result", compact_tool_result(event.result_content), "blue")
                continue
            if event.tool_name in {"TeamCreate", "TeamDelete", "TeamDispatch", "SendMessage"} and event.result_content:
                render_notice(console, event.tool_name, compact_tool_result(event.result_content), "blue")
                continue
            if event.tool_name == "Edit":
                diff_text = extract_unified_diff(event.result_content)
                if diff_text:
                    render_diff_panel(console, "Edit Diff", diff_text)
                continue

        text = extract_text(event.text).strip()
        label = event.type.value.replace("_", " ").title()
        if omit_duplicate_result and label == "Result" and text and text == last_assistant_text:
            continue
        if text:
            style = "cyan" if label == "Assistant" else "green"
            render_text_panel(console, label, text, style)
            if label == "Assistant":
                last_assistant_text = text


def session_row_count(sessions: list[dict[str, object]]) -> int:
    return len(sessions)


def render_session_table(console: Console, sessions: list[dict[str, object]]) -> None:
    if not sessions:
        console.print(Panel("[dim]No sessions found[/dim]", title="Sessions", border_style="blue", expand=True))
        return

    table = Table(title="Sessions")
    table.add_column("ID")
    table.add_column("Title")
    table.add_column("Messages", justify="right")

    for session in sessions:
        table.add_row(
            str(session.get("id", "")),
            str(session.get("title", session.get("summary", ""))),
            str(session.get("message_count", session.get("messageCount", ""))),
        )

    console.print(table)


def render_session_info(console: Console, info: Mapping[str, object]) -> None:
    session_id = str(info.get("id", "session"))
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="bold blue")
    table.add_column()

    rows = [
        ("Updated", str(info.get("updatedAt", ""))),
        ("Messages", str(info.get("messageCount", info.get("message_count", "")))),
        ("CWD", str(info.get("cwd", ""))),
        ("Model", str(info.get("model", ""))),
    ]

    for label, value in rows:
        table.add_row(label, value)

    console.print(Panel(table, title=f"Session {session_id}", border_style="blue", expand=True))


def render_transcript(console: Console, messages: list[dict[str, object]]) -> None:
    for message in messages:
        role = str(message.get("role", "message")).upper()
        text = extract_text(message.get("content", []))
        render_text_panel(console, role, text, "magenta" if role == "USER" else "cyan")


def render_tool_table(console: Console, tools: list[str]) -> None:
    if not tools:
        console.print(Panel("[dim]No tools found[/dim]", title="Tools", border_style="yellow", expand=True))
        return

    table = Table(title="Tools")
    table.add_column("Tool", style="bold cyan")

    for tool in tools:
        table.add_row(tool)

    console.print(table)


def render_agents_list(console: Console, agents: dict[str, Any]) -> None:
    if not agents:
        console.print(Panel("[dim]No agents configured[/dim]", title="Agents", border_style="yellow", expand=True))
        return

    table = Table(title="Agents")
    table.add_column("Name", style="bold cyan")
    table.add_column("Description")

    for name, definition in agents.items():
        if isinstance(definition, dict):
            desc = str(definition.get("description") or definition.get("prompt") or definition.get("system_prompt") or "")
        else:
            desc = str(definition)
        table.add_row(name, desc)

    console.print(table)


def render_team_info(console: Console, team_info: object) -> None:
    render_state(console, "Team", team_info)


def render_state(console: Console, title: str, data: object) -> None:
    if isinstance(data, list) and not data:
        console.print(Panel(f"[dim]No {title.lower()} found[/dim]", title=title, border_style="yellow", expand=True))
        return

    if isinstance(data, Mapping) and not data:
        console.print(Panel(f"[dim]No {title.lower()} found[/dim]", title=title, border_style="yellow", expand=True))
        return

    if isinstance(data, Mapping):
        table = Table(show_header=False, box=None, pad_edge=False)
        table.add_column("Key", style="bold yellow")
        table.add_column("Value")
        for key, value in data.items():
            table.add_row(str(key), extract_text(value) if isinstance(value, (str, list, Mapping)) else str(value))
        console.print(Panel(table, title=title, border_style="yellow", expand=True))
        return

    if isinstance(data, list) and data and all(isinstance(item, Mapping) for item in data):
        keys: list[str] = []
        for item in data:
            mapping = cast(Mapping[str, object], item)
            for key in mapping:
                if key not in keys:
                    keys.append(key)

        if not keys:
            console.print(Panel(f"[dim]No {title.lower()} found[/dim]", title=title, border_style="yellow", expand=True))
            return

        table = Table(title=f"{title} ({len(data)})")
        for key in keys:
            table.add_column(str(key))

        for item in data:
            mapping = cast(Mapping[str, object], item)
            table.add_row(*[extract_text(mapping.get(key, "")) for key in keys])

        console.print(table)
        return

    console.print(Panel(Pretty(data), title=title, border_style="yellow", expand=True))
