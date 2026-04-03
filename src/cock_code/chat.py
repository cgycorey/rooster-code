from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ChatCommand:
    name: str
    args: list[str]


def parse_chat_command(text: str) -> ChatCommand:
    parts = text.strip().split()
    if not parts:
        return ChatCommand(name="", args=[])

    name = parts[0].removeprefix("/")
    return ChatCommand(name=name, args=parts[1:])
