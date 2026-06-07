import asyncio

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


def test_telegram_exception_handler_does_not_leak_internal_details() -> None:
    async def _run() -> None:
        internal_error = RuntimeError("database connection failed at db.internal:5432")
        internal_error.__cause__ = FileNotFoundError("/etc/secrets/db.pem")

        async def handler(session_id: str, user_id: str, msg: str) -> str:
            raise internal_error

        TelegramAdapter(token="test-token", query_handler=handler, handler_timeout=5.0)

        try:
            await handler("tg-123", "456", "hello")
        except asyncio.TimeoutError:
            response = "I'm still working on that \u2014 please wait and try again."
        except Exception:
            response = "Something went wrong. Please try again."

        assert response == "Something went wrong. Please try again."
        assert "database" not in response
        assert "secrets" not in response
        assert "5432" not in response

    asyncio.run(_run())


def test_telegram_timeout_produces_distinct_message() -> None:
    async def _run() -> None:
        async def handler(session_id: str, user_id: str, msg: str) -> str:
            raise asyncio.TimeoutError()

        TelegramAdapter(token="test-token", query_handler=handler)

        try:
            await handler("tg-123", "456", "hello")
        except asyncio.TimeoutError:
            response = "I'm still working on that \u2014 please wait and try again."

        assert "still working" in response
        assert "Something went wrong" not in response

    asyncio.run(_run())
