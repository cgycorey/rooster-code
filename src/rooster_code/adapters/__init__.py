from __future__ import annotations

from collections.abc import Awaitable, Callable

MessageHandler = Callable[[str, str, str], Awaitable[str]]


class ChannelAdapter:

    async def start(self) -> None:
        raise NotImplementedError

    async def stop(self) -> None:
        raise NotImplementedError

    async def health(self) -> bool:
        raise NotImplementedError
