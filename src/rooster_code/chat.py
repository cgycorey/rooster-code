from __future__ import annotations

import shlex
from dataclasses import dataclass


@dataclass(slots=True)
class ChatCommand:
    name: str
    args: list[str]


def parse_chat_command(text: str) -> ChatCommand:
    try:
        parts = shlex.split(text.strip())
    except ValueError:
        # Malformed quoting — fall back to whitespace split
        parts = text.strip().split()

    if not parts:
        return ChatCommand(name="", args=[])

    if not parts[0].startswith("/"):
        return ChatCommand(name="", args=[])

    name = parts[0].removeprefix("/")
    return ChatCommand(name=name, args=parts[1:])
