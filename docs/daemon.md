# Rooster Code Daemon & Telegram Gateway

## Overview

`rooster-daemon` turns Rooster Code into a long-running agent gateway accessible via Unix socket and messaging channels. It stays alive between requests, manages SDK agent sessions with automatic conversation persistence, and accepts work through multiple adapters simultaneously.

It is a **pure additive layer** — it imports from existing `rooster_code` modules (`runtime.py`, `config.py`) but never modifies them. `rooster-code ask` and `rooster-code chat` work identically.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      Agent Daemon                            │
│                                                              │
│  ┌─────────────────┐    ┌──────────────────────────────┐    │
│  │  Unix Socket     │    │     Channel Adapters         │    │
│  │  /tmp/rooster-   │    │                              │    │
│  │  code.sock       │    │  TelegramAdapter             │    │
│  │                  │    │  (WhatsAppAdapter — planned) │    │
│  │  actions:        │    │  (DiscordAdapter — planned)  │    │
│  │   - query        │    │                              │    │
│  │   - health       │    │                              │    │
│  │   - sessions     │    └──────────┬───────────────────┘    │
│  └────────┬─────────┘               │                        │
│           │                         │                        │
│           └─────────┬───────────────┘                        │
│                     ▼                                        │
│           ┌─────────────────┐                                │
│           │  StateStore     │   SQLite (~/.rooster-code/     │
│           │  daemon.db      │   daemon.db)                   │
│           └────────┬────────┘                                │
│                    │                                         │
│                    ▼                                         │
│           ┌─────────────────┐                                │
│           │  SDK Agent      │   create_runtime_agent()       │
│           │  (per-session)  │   → agent.prompt()             │
│           │                 │   → agent.close() persists     │
│           └─────────────────┘   to ~/.open-agent-sdk/       │
└──────────────────────────────────────────────────────────────┘
```

## Quick Start

### Install

```bash
# Base daemon (zero extra dependencies)
uv run rooster-daemon --help

# With Telegram support
uv pip install aiogram
```

### Run

```bash
# Daemon only (Unix socket)
rooster-daemon

# Daemon + Telegram bot
rooster-daemon --telegram "123456:ABC-DEF..."

# Telegram with user whitelist
rooster-daemon --telegram "123456:ABC-DEF..." --telegram-allowed "111,222"

# Custom paths
rooster-daemon --socket /tmp/my.sock --db /tmp/state.db
```

### Talk to it via Unix socket

```bash
# Health check
echo '{"action":"health"}' | nc -U /tmp/rooster-code.sock

# Run a query
echo '{"action":"query","prompt":"what is 2+2","session_id":"math"}' | nc -U /tmp/rooster-code.sock

# Resume same session (remembers context)
echo '{"action":"query","prompt":"what did I just ask?","session_id":"math"}' | nc -U /tmp/rooster-code.sock

# List tracked sessions
echo '{"action":"sessions"}' | nc -U /tmp/rooster-code.sock
```

Or from Python:

```python
import asyncio
from rooster_code.daemon import daemon_query, daemon_health

result = asyncio.run(daemon_query("hello", session_id="s1"))
# {"type": "done", "session_id": "s1", "text": "..."}
```

### Talk to it via Telegram

1. Create a bot with [@BotFather](https://t.me/BotFather) — get the token
2. Start the daemon with `--telegram` flag
3. DM the bot — response arrives within seconds
4. In groups: mention the bot with `@botname`

---

## Files

### `src/rooster_code/daemon.py` (290 lines)

Three components:

#### StateStore

SQLite store for daemon session metadata. Does **not** replace the SDK session store (`~/.open-agent-sdk/sessions/`) — conversation history is entirely SDK-managed.

```
sessions table:
  session_id     TEXT PRIMARY KEY
  cwd            TEXT
  created_at     REAL (epoch)
  last_active_at REAL
```

| Method | Purpose |
|--------|---------|
| `upsert_session(id, cwd)` | Create or update timestamp |
| `touch_session(id)` | Bump `last_active_at` |
| `remove_session(id)` | Delete |
| `list_sessions()` | All sessions with metadata |
| `get_session(id)` | Single session lookup |

#### AgentDaemon

Persistent asyncio process:

- Listens on Unix socket (`/tmp/rooster-code.sock` by default)
- JSON-Lines protocol: one JSON object per line, one line per response
- Creates SDK agents via `create_runtime_agent()`, runs via `agent.prompt()`
- Manages channel adapters alongside the socket
- Graceful shutdown on SIGTERM/SIGINT — stops adapters, closes server, saves state
- Single `_make_query_handler()` factory — both Unix socket and adapters use identical logic

```python
daemon = AgentDaemon()
daemon.add_telegram("bot-token", allowed_users=[123456789])
await daemon.start()
# ... runs until SIGTERM ...
await daemon.shutdown()
```

#### Client helpers

```python
await daemon_query("prompt", session_id="s1", cwd=".")
await daemon_health()
await daemon_list_sessions()
```

### `src/rooster_code/adapters/__init__.py` (15 lines)

Channel adapter interface. Not an abstract base class — uses `raise NotImplementedError` for zero-dependency compatibility:

```python
class ChannelAdapter:
    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def health(self) -> bool: ...
```

### `src/rooster_code/adapters/telegram.py` (125 lines)

Telegram bot using [aiogram](https://github.com/aiogram/aiogram) (Python async framework, MIT, 5,600+ stars).

Flow:
1. aiogram long polling receives message
2. Normalized to `(session_id, user_id, text)` → `query_handler`
3. Handler creates SDK agent, runs query, returns text
4. `agent.close()` persists conversation via SDK
5. Response sent back to Telegram (auto-split for >4000 chars)

Behavior:
- **DMs**: All messages processed
- **Groups**: Only `@botname` mentions trigger response
- **Cooldown**: 2-second per-chat throttle prevents message loops
- **User allowlist**: Optional whitelist by Telegram user ID
- **Session keys**: `tg-{chat_id}` — separate conversation per chat, auto-resume

### `tests/test_daemon.py` (16 tests)

StateStore unit:
- Creates DB with schema, upsert/get/touch/remove/list, timestamp updates, default path

Daemon integration:
- Health, error handling (empty request, unknown action, invalid JSON)
- Empty sessions list, missing prompt rejection
- Real query + session tracking, custom session IDs
- `add_telegram` registers adapter

### `tests/test_adapters.py` (7 tests)

- `_split_long_message()` edge cases: short, long, newline breaks, space breaks, no-break fallback
- Adapter health before start, stop before start

---

## Protocol

JSON Lines over Unix socket. One JSON object per line.

### `health`

```json
→ {"action": "health"}
← {"type": "health", "status": "ok"}
```

### `sessions`

```json
→ {"action": "sessions"}
← {
    "type": "sessions",
    "data": [
      {"session_id": "math", "cwd": ".", "created_at": 1777758887.4, "last_active_at": 1777758890.8}
    ]
  }
```

### `query`

```json
→ {"action": "query", "prompt": "what is 2+2", "session_id": "math", "cwd": "."}
← {"type": "done", "session_id": "math", "text": "2+2 = 4"}
```

Error:
```json
← {"type": "error", "session_id": "math", "message": "API key not configured"}
```

### Session semantics

- **New**: Set `session_id` to any string. Daemon creates a new SDK session.
- **Resume**: Reuse same `session_id`. SDK loads full history from `~/.open-agent-sdk/sessions/`.
- **Persistence**: `agent.close()` writes transcript. StateStore tracks only metadata.

---

## Adding a New Channel Adapter

```python
from rooster_code.adapters import ChannelAdapter

class DiscordAdapter(ChannelAdapter):
    def __init__(self, token: str, query_handler):
        self._token = token
        self._handler = query_handler

    async def start(self) -> None:
        # Connect, register handlers
        # On message: response = await self._handler(session_id, user_id, text)

    async def stop(self) -> None:
        # Disconnect gracefully

    async def health(self) -> bool:
        return True  # if connected
```

Wire into daemon:

```python
# In AgentDaemon:
def add_discord(self, token: str) -> None:
    handler = self._make_query_handler()
    self._adapters.append(DiscordAdapter(token=token, query_handler=handler))
```

Use `_make_query_handler()` for session management — it handles create, resume, persist, and errors identically for every channel.

---

## Design Principles

1. **Additive.** Nothing in the existing codebase is modified. The daemon imports and calls existing functions.
2. **SDK owns conversations.** Transcripts live in `~/.open-agent-sdk/sessions/`. StateStore is metadata only.
3. **Thin adapters.** Each channel translates between platform API and `(session_id, user_id, text)`. Agent logic is identical.
4. **JSON Lines.** Line-delimited, debuggable with `nc`. One line in, one line out.
5. **Zero deps for base mode.** Core uses only stdlib. aiogram is optional.

---

## Test Suite

```bash
uv run pytest tests/test_daemon.py -v      # 16 tests
uv run pytest tests/test_adapters.py -v    # 7 tests
uv run pytest tests/ -q                    # 342 tests, zero regressions
```
