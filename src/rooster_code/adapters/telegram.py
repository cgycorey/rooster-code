from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import Any

from rooster_code.adapters import ChannelAdapter, MessageHandler

log = logging.getLogger("rooster.telegram")


class TelegramAdapter(ChannelAdapter):
    """aiogram long-polling Telegram bot. DM = always respond. Group = @mention only."""

    def __init__(
        self,
        token: str,
        query_handler: MessageHandler,
        *,
        allowed_users: list[int] | None = None,
        handler_timeout: float = 300.0,
    ) -> None:
        self._token = token
        self._handler = query_handler
        self._allowed_users = set(allowed_users or [])
        self._handler_timeout = handler_timeout
        self._last_response: dict[str, float] = {}
        self._cooldown_seconds: float = 2.0
        self._bot: Any = None
        self._dp: Any = None
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        from aiogram import Bot, Dispatcher
        from aiogram.client.default import DefaultBotProperties
        from aiogram.enums import ParseMode
        from aiogram.types import Message

        self._bot = Bot(token=self._token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
        self._dp = Dispatcher()
        handler = self._handler
        allowed_users = self._allowed_users
        cooldown = self._cooldown_seconds
        last_response = self._last_response
        bot_username = (await self._bot.get_me()).username or "rooster"
        bot = self._bot

        @self._dp.message()
        async def on_message(message: Message) -> None:
            if not message.text:
                return

            user_id = message.from_user.id if message.from_user else 0
            if allowed_users and user_id not in allowed_users:
                return

            chat_id = str(message.chat.id)
            is_private = message.chat.type == "private"
            bot_mentioned = f"@{bot_username}" in message.text.lower()

            if not is_private and not bot_mentioned:
                return

            now = time.monotonic()
            since = now - last_response.get(chat_id, 0)
            if since < cooldown:
                return
            last_response[chat_id] = now
            if len(last_response) > 1000:
                cutoff = now - 3600
                stale = [k for k, v in last_response.items() if v < cutoff]
                for k in stale:
                    del last_response[k]

            cleaned = message.text
            for variant in (f"@{bot_username}", f"@{bot_username.lower()}"):
                cleaned = cleaned.replace(variant, "")
            cleaned = cleaned.strip() or "hello"

            session_id = f"tg-{chat_id}"

            with contextlib.suppress(Exception):
                await bot.send_chat_action(chat_id=message.chat.id, action="typing")

            try:
                result = await asyncio.wait_for(
                    handler(session_id, str(user_id), cleaned),
                    timeout=self._handler_timeout,
                )
                response = result["text"] if isinstance(result, dict) else str(result)
            except asyncio.TimeoutError:
                response = "I'm still working on that — please wait and try again."
            except Exception:
                log.exception("telegram handler failed for user %s", user_id)
                response = "Something went wrong. Please try again."

            try:
                for chunk in _split_long_message(response):
                    await message.answer(chunk)
            except Exception:
                log.exception("telegram: failed to send response to user %s", user_id)

        self._task = asyncio.create_task(self._dp.start_polling(self._bot))
        print(f"[telegram] @{bot_username} listening", flush=True)

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        if self._bot:
            with contextlib.suppress(Exception):
                await self._bot.session.close()
            self._bot = None
        self._dp = None

    async def health(self) -> bool:
        return self._bot is not None and self._task is not None and not self._task.done()


def _split_long_message(text: str, limit: int = 4000) -> list[str]:
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    while len(text) > limit:
        split_at = text.rfind("\n", 0, limit)
        if split_at == -1 or split_at < limit // 2:
            split_at = text.rfind(" ", 0, limit)
        if split_at == -1 or split_at < limit // 2:
            split_at = limit
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip()
    if text:
        chunks.append(text)
    return chunks
