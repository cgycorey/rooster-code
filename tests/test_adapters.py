import asyncio
import tempfile
from pathlib import Path
from typing import Any

from rooster_code.adapters.telegram import TelegramAdapter, _split_long_message


def test_split_short_message() -> None:
    assert _split_long_message("hello") == ["hello"]


def test_split_long_message() -> None:
    long = "A" * 5000
    chunks = _split_long_message(long)
    assert len(chunks) > 1
    assert all(len(c) <= 4000 for c in chunks)
    assert "".join(chunks) == long


def test_split_at_newline() -> None:
    part1 = "short\n"
    part2 = "A" * 5000
    chunks = _split_long_message(part1 + part2)
    assert len(chunks) > 1
    assert all(len(c) <= 4000 for c in chunks)
    assert "".join(chunks).replace("\n", "") == (part1 + part2).replace("\n", "")


def test_split_at_space() -> None:
    part1 = "A" * 3500 + " word"
    part2 = "B" * 500
    chunks = _split_long_message(part1 + part2)
    assert len(chunks) > 1
    assert all(len(c) <= 4000 for c in chunks)
    combined = "".join(chunks)
    assert "A" * 3500 in combined
    assert "word" in combined
    assert "B" * 500 in combined


def test_split_no_break_fallback() -> None:
    long = "X" * 5000
    chunks = _split_long_message(long)
    assert all(len(c) <= 4000 for c in chunks)
    assert "".join(chunks) == long


def test_telegram_adapter_health_before_start() -> None:
    async def _run() -> None:
        async def handler(session_id: str, user_id: str, msg: str) -> str:
            return "ok"

        adapter = TelegramAdapter(token="test-token", query_handler=handler)
        assert not await adapter.health()

    asyncio.run(_run())


def test_telegram_adapter_stop_before_start() -> None:
    async def _run() -> None:
        async def handler(session_id: str, user_id: str, msg: str) -> str:
            return "ok"

        adapter = TelegramAdapter(token="test-token", query_handler=handler)
        await adapter.stop()

    asyncio.run(_run())
