from __future__ import annotations

from collections.abc import Mapping
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, cast

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.pretty import Pretty
from rich.rule import Rule
from rich.table import Table

from cock_code.config import RuntimeConfig

if TYPE_CHECKING:
    from open_agent_sdk import SDKMessage


def summarize_tool_result(text: str, max_chars: int = 160) -> str:
    if len(text) <= max_chars:
        return text

    return f"{text[:max_chars]}..."


def build_console() -> Console:
    return Console(highlight=False)


def render_banner(console: Console, mode: str, config: RuntimeConfig) -> None:
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
    table.add_row("/model <name>", "Switch models")
    table.add_row("/permission <mode>", "Update permission mode")
    table.add_row("/tools", "Show available tools")
    table.add_row("/sessions", "Show saved sessions")
    table.add_row("/status", "Show current chat runtime state")
    table.add_row("/resume <id>", "Resume another session")
    table.add_row("/exit", "Exit chat")
    console.print(Panel(table, title="Help", border_style="blue", expand=True))


def render_notice(console: Console, title: str, message: str, style: str = "yellow") -> None:
    console.print(Panel(message, title=title, border_style=style, expand=True))


async def render_event_stream(
    console: Console,
    events: AsyncIterator[SDKMessage],
    omit_duplicate_result: bool = False,
) -> None:
    last_assistant_text = ""
    async for event in events:
        text = extract_text(event.text)
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
