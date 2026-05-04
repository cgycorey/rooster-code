# Plan: Evolve Rooster Code Into a Long-Running Agent

## Goal

Transform Rooster Code from a single-session, stdin-driven CLI tool into a long-running agent daemon that:
- Stays alive across multiple concurrent sessions and users
- Accepts work from multiple input sources (HTTP, WebSocket, message queues, CLI via socket)
- Maintains durable state across process restarts
- Manages resource limits, session lifecycle, and graceful degradation independently

---

## 1. Architecture Overview: Current vs Target

### 1.1 Current Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Single Process                        │
│                                                          │
│  main() → asyncio.run(run_chat(config))                  │
│                                                          │
│  ┌─────────────────────────────────────────────────┐    │
│  │         Chat Loop (cli.py:786 while True)        │    │
│  │                                                  │    │
│  │  stdin ──→ prompt_toolkit ──→ parse_command()   │    │
│  │                                    │              │    │
│  │                              agent.query()        │    │
│  │                                    │              │    │
│  │                        wrapped_query() generator  │    │
│  │                                    │              │    │
│  │              ┌─────────────────────┤              │    │
│  │              │                     │              │    │
│  │        TurnTracker          render_event_stream()│    │
│  │        (activity)           (Rich console)       │    │
│  │                                                  │    │
│  │  Background: _poll_and_render_notifications()    │    │
│  │  Background: _background_subagent_tasks set       │    │
│  │  Background: TeamManager + AgentPool (in-memory) │    │
│  └─────────────────────────────────────────────────┘    │
│                                                          │
│  State: ALL in-memory (agent._history, team state,      │
│         task queues, TurnTracker, config)                │
│  Persistence: SDK session store + ~/.rooster-code/history│
│  Shutdown: finally block → close team, cancel tasks      │
└─────────────────────────────────────────────────────────┘
```

### 1.2 Target Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Agent Daemon Process                       │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                     Input Gateway Layer                      │ │
│  │                                                              │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐  │ │
│  │  │ HTTP API │  │ WebSocket│  │   gRPC   │  │  CLI Socket │  │ │
│  │  │ (FastAPI)│  │  Server  │  │  Server  │  │ (Unix/TCP) │  │ │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────┬─────┘  │ │
│  │       │             │             │               │         │ │
│  │       └─────────────┴─────────────┴───────────────┘         │ │
│  │                         │                                     │ │
│  │                  ┌──────┴──────┐                              │ │
│  │                  │  Auth / ACL │                              │ │
│  │                  └──────┬──────┘                              │ │
│  │                         │                                     │ │
│  │                  ┌──────┴──────┐                              │ │
│  │                  │  Rate Limit │                              │ │
│  │                  └──────┬──────┘                              │ │
│  └─────────────────────────┼────────────────────────────────────┘ │
│                            │                                       │
│  ┌─────────────────────────┼────────────────────────────────────┐ │
│  │                   Work Queue Layer                            │ │
│  │                            │                                   │ │
│  │  ┌─────────────────────────┴──────────────────────────────┐  │ │
│  │  │              Priority Work Queue (Redis / SQLite)       │  │ │
│  │  │                                                        │  │ │
│  │  │  WorkItem { id, session_id, user_id, prompt,           │  │ │
│  │  │            priority, created_at, timeout, ... }        │  │ │
│  │  └─────────────────────────┬──────────────────────────────┘  │ │
│  └─────────────────────────────┼─────────────────────────────────┘ │
│                                │                                    │
│  ┌─────────────────────────────┼─────────────────────────────────┐ │
│  │                     Session Manager                            │ │
│  │                            │                                    │ │
│  │  ┌────────────────────────┐┌──────────────────────────────┐   │ │
│  │  │   Session Pool (N)     ││    Session Lifecycle          │   │ │
│  │  │                        ││                                │   │ │
│  │  │  Session 1 [busy    ]  ││  create → idle → working →    │   │ │
│  │  │  Session 2 [idle    ]  ││  idle → timeout → archived    │   │ │
│  │  │  Session 3 [working ]  ││                                │   │ │
│  │  │  Session 4 [draining]  ││  Max concurrent: configurable  │   │ │
│  │  │  ...                   ││  Idle timeout: configurable    │   │ │
│  │  └────────────────────────┘│  Max lifetime: configurable    │   │ │
│  └─────────────────────────────┴───────────────────────────────┘   │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │                     Runtime Engine (per-session)               │ │
│  │                                                                │ │
│  │  Same as current: _create_sdk_agent → wrapped_query → events  │ │
│  │  Same as current: TurnTracker, tool wrappers, skill loading   │ │
│  │  Same as current: TeamManager + AgentPool integration         │ │
│  │                                                                │ │
│  │  NEW: Output captured to structured log, not Rich console     │ │
│  │  NEW: Session state checkpoints to durable store              │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │                     State & Persistence Layer                  │ │
│  │                                                                │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐ │ │
│  │  │ Session  │  │  Team    │  │  Task    │  │  Config /    │ │ │
│  │  │ Store    │  │  Store   │  │  Store   │  │  Registry    │ │ │
│  │  │ (SDK)    │  │ (SQLite) │  │ (SQLite) │  │  (files)     │ │ │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────────┘ │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │                     Observability Layer                        │ │
│  │                                                                │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐ │ │
│  │  │ Structured│  │ Metrics  │  │  Health  │  │  Admin UI    │ │ │
│  │  │ Logging  │  │(Prometheus)│ │  Checks  │  │  (optional)  │ │ │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────────┘ │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │                     Process Supervisor                         │ │
│  │                                                                │ │
│  │  Signal handling (SIGTERM → graceful drain, SIGINT → faster)  │ │
│  │  Crash recovery (restart on failure, max restarts)            │ │
│  │  Hot reload (config changes, skill changes)                   │ │
│  │  Resource monitoring (memory, open FDs, event loop health)    │ │
│  └───────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

---

## 2. Implementation Phases

### Phase 0: Extract Runtime Engine (Foundation)

**Goal**: Decouple the agent runtime from the CLI rendering layer so it can be driven by any input source.

**Current coupling points**:
- `cli.py run_chat()` — mixes event loop, rendering, prompt input, command parsing
- `_run_query_with_interrupt()` — directly calls `render_event_stream()` (Rich-specific)
- `_poll_and_render_notifications()` — directly calls Rich rendering functions
- `append_task_result_to_context()` — mutates `agent._history` directly

**What to extract**:
```python
# New: src/rooster_code/engine.py

class AgentEngine:
    """Pure agent runtime — no rendering, no I/O assumptions."""
    
    def __init__(self, config: RuntimeConfig):
        self.agent = create_runtime_agent(config)
        self.events: asyncio.Queue[SDKMessage] = asyncio.Queue()
        self.abort_signal = asyncio.Event()
    
    async def run_query(self, prompt: str) -> AsyncIterator[SDKMessage]:
        """Run a query, yield SDK events. Caller decides what to do with them."""
        ...
    
    async def run_command(self, command: str, args: list[str]) -> AsyncIterator[SDKMessage]:
        """Handle slash commands, yield events. Caller decides rendering."""
        ...
    
    async def compact(self) -> dict:
        """Compact session, return result dict."""
        ...
    
    async def close(self):
        """Graceful shutdown of agent + background tasks."""
        ...
```

**Key principle**: `AgentEngine` is a pure async event generator. The CLI becomes ONE consumer; an HTTP handler becomes another.

**Files to create**:
- `src/rooster_code/engine.py` — `AgentEngine` class (extracted from cli.py + runtime.py)

**Files to modify**:
- `src/rooster_code/cli.py` — refactor `run_chat()` to use `AgentEngine`, keep only Rich rendering
- `src/rooster_code/runtime.py` — move query orchestration logic to `engine.py`, keep tool wrappers + SDK helpers

**Estimated effort**: 2-3 days
**Risk**: Medium — significant refactor but purely internal, test suite catches regressions

---

### Phase 1: Durable State Store

**Goal**: Make session state, team state, and task state survive process restarts.

**What's currently in-memory only**:

| Structure | Location | Lost on crash? |
|-----------|----------|----------------|
| `agent._history` | SDK session object | No (SDK persists) |
| `TeamManager._pool` | `team.py:287` | **Yes** |
| `AgentPool._members` | `team.py:63` | **Yes** |
| `AgentPool._mailboxes` | `team.py:64` | **Yes** |
| `AgentPool._busy` / `_unhealthy` | `team.py:66-67` | **Yes** |
| `_background_subagent_tasks` | `runtime.py:56` | **Yes** |
| `_notified_task_ids` | `runtime.py:57` | **Yes** |
| `TurnTracker` | `runtime_tools.py:19` | **Yes** |
| `_loaded_local_skill_names` | `runtime.py:54` | **Yes** |
| `_injected_task_ids` | cli.py local var | **Yes** |
| Config (`RuntimeConfig`) | Created at startup | **Yes** |

**Design: SQLite-based state store**

```python
# New: src/rooster_code/state.py

# Schema (single-file SQLite, WAL mode):
#
# CREATE TABLE sessions (
#     id TEXT PRIMARY KEY,
#     user_id TEXT NOT NULL,
#     status TEXT NOT NULL DEFAULT 'idle',  -- idle, working, draining, archived
#     config_json TEXT NOT NULL,
#     created_at TEXT NOT NULL,
#     last_active_at TEXT NOT NULL,
#     expires_at TEXT,
#     metadata_json TEXT DEFAULT '{}'
# );
#
# CREATE TABLE teams (
#     id TEXT PRIMARY KEY,
#     session_id TEXT NOT NULL REFERENCES sessions(id),
#     name TEXT NOT NULL,
#     member_defs_json TEXT NOT NULL,  -- {name: {definition}}
#     member_status_json TEXT NOT NULL, -- {name: "idle"|"busy"|"unhealthy"}
#     created_at TEXT NOT NULL
# );
#
# CREATE TABLE tasks (
#     id TEXT PRIMARY KEY,
#     session_id TEXT NOT NULL REFERENCES sessions(id),
#     team_id TEXT REFERENCES teams(id),
#     member_name TEXT,
#     status TEXT NOT NULL DEFAULT 'pending',  -- pending, running, completed, failed, cancelled
#     task_type TEXT NOT NULL,  -- direct, team_dispatch, background_agent
#     prompt TEXT NOT NULL,
#     result_text TEXT,
#     created_at TEXT NOT NULL,
#     completed_at TEXT,
#     notified BOOLEAN NOT NULL DEFAULT 0
# );
#
# CREATE TABLE mailbox_messages (
#     id INTEGER PRIMARY KEY AUTOINCREMENT,
#     team_id TEXT NOT NULL REFERENCES teams(id),
#     member_name TEXT NOT NULL,
#     sender TEXT NOT NULL,
#     content TEXT NOT NULL,
#     msg_type TEXT NOT NULL DEFAULT 'text',
#     created_at TEXT NOT NULL,
#     delivered BOOLEAN NOT NULL DEFAULT 0
# );

class StateStore:
    """Durable state backend for the agent daemon."""
    
    def __init__(self, db_path: str = "~/.rooster-code/state.db"):
        self.db_path = Path(db_path).expanduser()
        self._ensure_schema()
    
    # Session CRUD
    async def create_session(self, session_id, user_id, config) -> None: ...
    async def update_session_status(self, session_id, status) -> None: ...
    async def list_active_sessions(self) -> list[dict]: ...
    async def archive_session(self, session_id) -> None: ...
    async def cleanup_expired_sessions(self) -> int: ...
    
    # Team CRUD
    async def save_team(self, team) -> None: ...
    async def load_team(self, team_id) -> dict | None: ...
    async def delete_team(self, team_id) -> None: ...
    async def load_teams_for_session(self, session_id) -> list[dict]: ...
    
    # Task CRUD
    async def create_task(self, task) -> None: ...
    async def update_task_status(self, task_id, status, result=None) -> None: ...
    async def mark_notified(self, task_id) -> None: ...
    async def get_pending_notifications(self, session_id) -> list[dict]: ...
    
    # Mailbox
    async def enqueue_message(self, team_id, member, sender, content) -> None: ...
    async def drain_mailbox(self, team_id, member) -> list[dict]: ...
    async def restore_mailbox(self, team_id, member, messages) -> None: ...
    
    # Lifecycle
    async def vacuum(self) -> None: ...
    async def close(self) -> None: ...

# Optional: async wrapper via aiosqlite or thread-pool executor for sqlite3
```

**Recovery on startup**:
```
daemon starts
  → StateStore.list_active_sessions()
  → for each session: re-create AgentEngine with stored config
  → for each session with active team: re-create TeamManager + AgentPool
  → re-hydrate pending tasks, mailbox messages, notifications
  → mark crashed sessions as 'draining' → re-queue any in-flight work
```

**Key decisions**:
- SQLite over PostgreSQL: zero-dependency, single-file, WAL mode handles concurrent reads
- `aiosqlite` for async access OR thread-pool executor for sync sqlite3 (simpler, no new dep)
- State store is the **source of truth** — AgentEngine is a cache
- Checkpoint on every significant state change (task completion, team member status change)

**Files to create**:
- `src/rooster_code/state.py` — `StateStore` class with SQLite schema

**Files to modify**:
- `src/rooster_code/team.py` — `TeamManager` and `AgentPool` checkpoint to `StateStore`
- `src/rooster_code/runtime.py` — `_background_subagent_tasks` checkpoint to `StateStore`
- `src/rooster_code/cli.py` — use `StateStore` for durable notifications

**Estimated effort**: 3-4 days
**Risk**: Medium — schema design must be forward-compatible, migration strategy needed
**Dependencies**: Phase 0 (engine extraction) — `AgentEngine` becomes the thing we checkpoint

---

### Phase 2: Process Model & Lifecycle

**Goal**: Transform from `asyncio.run(coroutine)` to a managed daemon process.

**Three options (pick one)**:

| Option | Pros | Cons |
|--------|------|------|
| **A: Single-process, multi-session async** | Simple, same codebase, no IPC | One crash kills all sessions, GIL-bound |
| **B: Process-per-session (fork model)** | Strong isolation, crash containment | Complex IPC, high memory, harder to manage |
| **C: Supervisor + worker pool** | Balance of isolation and efficiency | More infrastructure, but proven pattern |

**Recommendation: Option C — Supervisor + worker pool**

```
┌──────────────────────────────────────────────────────────┐
│                    Supervisor Process                     │
│                                                          │
│  Responsibilities:                                       │
│  - Accept all incoming connections (HTTP, WS, gRPC)      │
│  - Manage worker pool (start, health-check, restart)     │
│  - Route work to appropriate worker                      │
│  - Handle graceful shutdown                              │
│  - Expose metrics + health endpoint                      │
│  - Single point of config loading                        │
└──────────┬───────────────────────────────────────────────┘
           │
           │ IPC: Unix socket or asyncio subprocess protocol
           │
     ┌─────┴─────┬─────────┬─────────┐
     │           │         │         │
  Worker 1   Worker 2  Worker 3  Worker N
  (Session)  (Session) (Session) (Session)

Each worker:
  - Own event loop (asyncio.run())
  - Owns 1-N AgentEngine instances (session pool)
  - Reports health to supervisor via heartbeat
  - Checkpoints state to shared SQLite (WAL-safe concurrent writes)
```

**Implementation approach** (simplest version):

```python
# src/rooster_code/daemon.py

class AgentDaemon:
    """Long-running agent process supervisor."""
    
    def __init__(self, config_path: str):
        self.state = StateStore()
        self.sessions: dict[str, SessionHandle] = {}
        self.input_adapters: list[InputAdapter] = []
        self.work_queue: asyncio.Queue[WorkItem] = asyncio.Queue()
    
    async def start(self):
        """Bootstrap: load state, start input adapters, enter main loop."""
        await self.state.initialize()
        await self._recover_sessions()
        await self._start_input_adapters()
        await self._main_loop()
    
    async def _main_loop(self):
        """Main event loop: pull work, dispatch to sessions."""
        while True:
            work = await self.work_queue.get()
            session = await self._get_or_create_session(work.session_id, work.user_id, work.config)
            asyncio.create_task(self._process_work(session, work))
    
    async def _process_work(self, session: SessionHandle, work: WorkItem):
        """Run query on session, stream results back to caller."""
        ...
    
    async def shutdown(self, signal=None):
        """Graceful shutdown: drain queue, close sessions, save state."""
        ...

class InputAdapter(ABC):
    """Pluggable input source."""
    
    @abstractmethod
    async def start(self, daemon: AgentDaemon) -> None: ...
    
    @abstractmethod
    async def stop(self) -> None: ...

class HTTPInputAdapter(InputAdapter):
    """FastAPI server adapter."""
    ...

class CLIInputAdapter(InputAdapter):
    """Unix socket adapter for rooster-code CLI client."""
    ...

class WebSocketInputAdapter(InputAdapter):
    """WebSocket adapter for real-time sessions."""
    ...
```

**Signal handling**:

| Signal | Behavior |
|--------|----------|
| `SIGTERM` | Graceful drain — stop accepting new work, finish in-flight work, save state, exit |
| `SIGINT` | Fast drain — like SIGTERM but with timeout, cancel long-running work after N seconds |
| `SIGHUP` | Reload config, reload skills, notify workers |
| `SIGUSR1` | Dump state snapshot to log (debugging) |

**Worker health**:
- Heartbeat every 5s from worker to supervisor
- If worker misses 3 heartbeats → mark session for recovery, spin up replacement worker
- Worker reports: memory usage, open session count, event loop latency

**Files to create**:
- `src/rooster_code/daemon.py` — `AgentDaemon`, `InputAdapter` ABC, `SessionHandle`
- `src/rooster_code/adapters/http.py` — `HTTPInputAdapter`
- `src/rooster_code/adapters/cli_socket.py` — `CLIInputAdapter` (so `rooster-code ask` talks to daemon)
- `src/rooster_code/adapters/websocket.py` — `WebSocketInputAdapter` (optional)

**Files to modify**:
- `src/rooster_code/cli.py` — add `rooster-code daemon` subcommand; `ask`/`chat` can target daemon via socket
- `pyproject.toml` — add optional deps: `fastapi`, `uvicorn`, `websockets`

**Estimated effort**: 4-5 days
**Risk**: High — introduces process model, IPC, lifecycle management, multiple new concepts
**Dependencies**: Phase 0 (engine), Phase 1 (state store)

---

### Phase 3: Concurrent Session Management

**Goal**: Handle multiple simultaneous sessions with resource isolation and fairness.

**Current state**: One session, one query at a time. `prompt_once()` blocks until input or timeout.

**Required capabilities**:

1. **Session Pool**: N concurrent sessions, configurable max
2. **Fair scheduling**: Round-robin or priority-based work distribution
3. **Resource limits per session**: Max turns, max tokens, max wall-clock time
4. **Backpressure**: Reject new work when pool is full (503 / queue full)
5. **Idle timeout**: Archive sessions inactive for > T seconds
6. **Max lifetime**: Force-close sessions older than T hours (prevents unbounded growth)

**Design**:

```python
class SessionPool:
    """Manages concurrent AgentEngine instances with resource limits."""
    
    def __init__(self, max_sessions: int = 10, idle_timeout: int = 3600, max_lifetime: int = 86400):
        self._sessions: dict[str, SessionHandle] = {}
        self._max_sessions = max_sessions
        self._idle_timeout = idle_timeout
        self._max_lifetime = max_lifetime
        self._gc_task: asyncio.Task | None = None
    
    async def acquire(self, session_id: str, user_id: str, config: RuntimeConfig) -> SessionHandle:
        """Get or create a session. Block if pool is full. Raise if denied."""
        ...
    
    async def release(self, session_id: str):
        """Mark session as idle. Don't destroy — let GC handle it."""
        ...
    
    async def _gc_loop(self):
        """Background task: archive idle sessions, enforce max lifetime."""
        while True:
            await asyncio.sleep(30)
            now = time.time()
            for sid, handle in list(self._sessions.items()):
                if handle.is_idle and (now - handle.last_active) > self._idle_timeout:
                    await self._archive_session(sid)
                elif (now - handle.created_at) > self._max_lifetime:
                    await self._archive_session(sid)
    
    def stats(self) -> dict:
        """Return pool stats for metrics."""
        return {
            "total": len(self._sessions),
            "busy": sum(1 for h in self._sessions.values() if h.is_busy),
            "idle": sum(1 for h in self._sessions.values() if h.is_idle),
            "draining": sum(1 for h in self._sessions.values() if h.is_draining),
        }
```

**Per-session resource enforcement**:

```python
@dataclass
class SessionLimits:
    max_turns: int = 100
    max_tokens: int = 16000
    max_wall_time_seconds: int = 300  # 5 min per query
    max_concurrent_subagents: int = 5
    max_budget_usd: float | None = None
```

**Files to create**:
- `src/rooster_code/pool.py` — `SessionPool`, `SessionHandle`, `SessionLimits`

**Files to modify**:
- `src/rooster_code/daemon.py` — integrate `SessionPool`
- `src/rooster_code/engine.py` — respect `SessionLimits`, report resource usage

**Estimated effort**: 2-3 days
**Risk**: Medium — concurrency bugs are subtle, needs thorough testing under load
**Dependencies**: Phase 0 (engine), Phase 2 (daemon process model)

---

### Phase 4: Observability & Operations

**Goal**: Make the daemon production-operable — logs, metrics, health checks, admin surface.

#### 4.1 Structured Logging

Replace `render_notice()` / `render_state()` calls with structured log events:

```python
# Replace render_notice(console, "Error", str(exc), "red")
# With:
logger.error("agent_error", session_id=sid, error=str(exc), traceback=tb)

# Replace render_state(console, "Tasks", get_state_snapshot("tasks"))
# With:
logger.info("state_snapshot", component="tasks", data=get_all_tasks())
```

Use `structlog` (already popular in async Python ecosystem):

```python
import structlog
logger = structlog.get_logger()

# Output: JSON lines to stdout/stderr, human-readable in dev
# {"event": "query_started", "session_id": "abc", "user_id": "x", "timestamp": "..."}
# {"event": "tool_called", "session_id": "abc", "tool": "Read", "target": "src/foo.py"}
# {"event": "query_completed", "session_id": "abc", "turns": 5, "tokens": 1200, "duration_ms": 3400}
```

#### 4.2 Metrics (Prometheus-compatible)

```python
# Key metrics:
# rooster_sessions_total{gauge} — active session count
# rooster_queries_total{counter} — total queries processed
# rooster_query_duration_seconds{histogram} — query latency
# rooster_turns_per_query{histogram} — turns per query
# rooster_tool_calls_total{counter, tool=Read|Edit|Bash|...} — tool usage
# rooster_background_tasks_total{gauge} — pending background tasks
# rooster_team_members_total{gauge, status=idle|busy|unhealthy} — team member status
# rooster_queue_depth{gauge} — work queue depth
# rooster_errors_total{counter, type=...} — error rate by type
```

Implementation: `prometheus_client` library or manual `/metrics` endpoint.

#### 4.3 Health Checks

```python
# GET /health → 200 { "status": "ok", "uptime_seconds": 1234 }
# GET /health/ready → 200 if ready to accept work, 503 if draining
# GET /health/live → 200 if process is alive (always returns 200 while process lives)

class HealthChecker:
    async def check(self) -> dict:
        checks = {
            "event_loop": await self._check_event_loop(),
            "state_store": await self._check_db(),
            "sdk_connection": await self._check_sdk(),
            "worker_heartbeats": await self._check_workers(),
        }
        healthy = all(v["healthy"] for v in checks.values())
        return {"status": "ok" if healthy else "degraded", "checks": checks}
```

#### 4.4 Admin CLI

```bash
# Talk to daemon for operational commands
rooster-code daemon status        # Show daemon health, session count, queue depth
rooster-code daemon sessions      # List active sessions
rooster-code daemon session <id>  # Inspect a session
rooster-code daemon archive <id>  # Force-archive a session
rooster-code daemon config reload # Reload config without restart
rooster-code daemon shutdown      # Graceful shutdown
```

These talk to the daemon via the HTTP API (or Unix socket for local-only operations).

**Files to create**:
- `src/rooster_code/observability.py` — structured logging setup, metrics registry, health checker
- `src/rooster_code/admin.py` — admin CLI handlers

**Files to modify**:
- `src/rooster_code/daemon.py` — integrate observability
- `src/rooster_code/cli.py` — add `daemon` subcommands
- `src/rooster_code/engine.py` — emit structured log events
- `src/rooster_code/rendering.py` — keep for CLI mode, daemon mode skips it

**Estimated effort**: 2-3 days
**Risk**: Low — additive changes, doesn't break existing behavior
**Dependencies**: Phase 2 (daemon)

---

### Phase 5: Advanced Features

**Conditional** — do these only if needed after phases 0-4 are stable.

| Feature | Effort | Why |
|---------|--------|-----|
| **Plugin/extension system** | 3-4 days | Custom input adapters, custom tools, custom notification channels |
| **Webhook triggers** | 1-2 days | Trigger agent work from GitHub, Slack, Jira events |
| **Scheduled/cron jobs** | 1-2 days | "Review this repo every morning", "Check for CVEs daily" |
| **Multi-user ACL** | 2-3 days | API keys, session ownership, tool allow/deny per user |
| **Streaming response protocol** | 2-3 days | SSE or WebSocket streaming so callers get real-time events |
| **Model failover** | 1-2 days | Automatic fallback if primary model returns errors |
| **Cost tracking & budgeting** | 1-2 days | Per-user, per-session cost tracking with hard limits |
| **Audit log** | 1 day | Full record of all queries, tool calls, results for compliance |

---

## 3. Migration Strategy

### Backward Compatibility (non-negotiable)

The CLI `rooster-code ask` and `rooster-code chat` must continue to work exactly as they do today. The daemon mode is a NEW mode, not a replacement.

```bash
# These work exactly like today — no daemon involved
rooster-code ask "What does this code do?"
rooster-code chat

# NEW: Start the daemon
rooster-code daemon start

# NEW: CLI can talk to daemon (optional client mode)
rooster-code ask --daemon "What does this code do?"
rooster-code chat --daemon
```

When `--daemon` is passed, the CLI becomes a thin client that connects to the daemon's Unix socket/HTTP endpoint, sends the prompt, and renders the streaming response.

### Dependency Philosophy

- **Phase 0-1**: Zero new dependencies. Pure refactoring + stdlib sqlite3.
- **Phase 2-3**: `fastapi` + `uvicorn` for HTTP adapter (optional, only if HTTP input needed). Default is Unix socket + CLI client, which uses stdlib `asyncio` streams.
- **Phase 4**: `structlog` (lightweight), `prometheus_client` (optional).

Everything behind extras in pyproject.toml:
```toml
[project.optional-dependencies]
daemon = ["fastapi", "uvicorn", "structlog"]
metrics = ["prometheus-client"]
all = ["rooster-code[daemon,metrics]"]
```

### Test Strategy

- **Unit tests**: Each new module (`engine.py`, `state.py`, `pool.py`) gets its own test file
- **Integration tests**: `test_daemon.py` — start daemon, send work via socket, verify response
- **Load tests**: `test_concurrent.py` — N concurrent sessions, verify no deadlocks, fair scheduling
- **Crash recovery tests**: Kill worker mid-query, verify state recovery on restart
- **Existing test suite**: Must continue to pass (319/319 currently)

---

## 4. File Manifest

### New Files

| File | Phase | Purpose |
|------|-------|---------|
| `src/rooster_code/engine.py` | 0 | `AgentEngine` — extracted runtime, no rendering |
| `src/rooster_code/state.py` | 1 | `StateStore` — SQLite-backed durable state |
| `src/rooster_code/daemon.py` | 2 | `AgentDaemon` — process supervisor, main loop |
| `src/rooster_code/pool.py` | 3 | `SessionPool` — concurrent session management |
| `src/rooster_code/observability.py` | 4 | Structured logging, metrics, health checks |
| `src/rooster_code/admin.py` | 4 | Admin CLI for daemon operations |
| `src/rooster_code/adapters/__init__.py` | 2 | Adapter package |
| `src/rooster_code/adapters/http.py` | 2 | FastAPI input adapter |
| `src/rooster_code/adapters/cli_socket.py` | 2 | Unix socket adapter for CLI client |
| `src/rooster_code/adapters/websocket.py` | 2 | WebSocket adapter (optional) |
| `src/rooster_code/adapters/base.py` | 2 | `InputAdapter` ABC |

### Modified Files

| File | Phase | Changes |
|------|-------|---------|
| `src/rooster_code/cli.py` | 0-4 | Extract engine, add daemon subcommands, client mode |
| `src/rooster_code/runtime.py` | 0-1 | Move query logic to engine, checkpoint tasks to state store |
| `src/rooster_code/team.py` | 1 | Checkpoint team state to StateStore |
| `src/rooster_code/rendering.py` | 0 | No change (CLI mode unchanged) |
| `pyproject.toml` | 2 | Add optional deps |
| `README.md` | 4 | Document daemon mode |

---

## 5. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Engine extraction breaks CLI | Medium | High | Extensive test coverage, do Phase 0 first in isolation |
| SQLite concurrent write contention | Low | Medium | WAL mode handles this; single-writer model |
| Process model over-engineering | Medium | Medium | Start with single-process option A, graduate to C if needed |
| Performance regression in CLI mode | Low | High | Benchmark before/after, isolate daemon code path |
| SDK API changes break integrations | Low | High | Pin SDK version, test against exact version |

---

## 6. Success Criteria

1. **CLI unchanged**: `rooster-code ask` and `rooster-code chat` produce identical output to current
2. **Daemon runs**: `rooster-code daemon start` stays alive, accepts work, survives SIGTERM
3. **State survives restart**: Kill daemon, restart, sessions resume from checkpoint
4. **Concurrent sessions**: 5+ simultaneous queries to different sessions complete without interference
5. **Graceful shutdown**: SIGTERM → finish in-flight work → save state → exit 0
6. **Test suite green**: All 319+ existing tests pass; new tests cover daemon paths
7. **Zero new dependencies for CLI mode**: Only daemon mode adds optional deps

---

## 7. Timeline Estimate

| Phase | Optimistic | Realistic | Pessimistic |
|-------|-----------|-----------|-------------|
| 0: Extract Engine | 1 day | 2-3 days | 4 days |
| 1: State Store | 2 days | 3-4 days | 5 days |
| 2: Process Model | 3 days | 4-5 days | 7 days |
| 3: Session Pool | 1 day | 2-3 days | 4 days |
| 4: Observability | 1 day | 2-3 days | 4 days |
| **Total** | **8 days** | **13-18 days** | **24 days** |

---

## Appendix A: Long-Running Agent Design Patterns (Research)

Patterns to incorporate from existing long-running agent systems:

1. **Event-driven architecture**: Rather than polling, use event emitters. The daemon subscribes to events (new work, task complete, timeout), reacts. Avoids busy-waiting.

2. **Idempotent work items**: Every work item has a unique ID. If the same work arrives twice (network retry, crash recovery), the daemon deduplicates by ID.

3. **Checkpoint-replay**: Periodically save session state. On crash, replay from last checkpoint. Similar to how databases use WAL — the state store IS the WAL.

4. **Circuit breaker**: If the LLM provider returns errors > threshold, stop sending work temporarily, retry with backoff. Prevents cascading failures.

5. **Backpressure propagation**: If the work queue is full, the HTTP adapter returns 503. Callers retry with exponential backoff. Prevents OOM from unbounded queuing.

6. **Heartbeat + lease**: Workers claim a session lease from the state store with a TTL. If they don't renew the lease within TTL, another worker can claim it. Prevents zombie sessions.

7. **Graceful degradation**: When overloaded, shed non-critical work (background tasks, notifications) before critical work (user queries). Priority levels in work queue.

## Appendix B: Open Questions

1. **Should teams be durable across daemon restarts?** Pro: seamless recovery. Con: stale team state. Decision: yes, with TTL — teams older than N hours without activity are garbage-collected.

2. **One daemon process or one daemon per user?** One daemon, multi-user. Simpler to operate, but requires auth/ACL. Per-user daemon is simpler for single-user desktop use.

3. **Should the daemon use the same SDK session IDs?** Yes. Session IDs are the stable identifier. The daemon adds a `daemon_session_id` for internal tracking that survives SDK session rotation.

4. **How does compaction work in daemon mode?** Auto-compaction when session exceeds token threshold. Configurable policy: aggressive (compact often), conservative (only when nearing context limit), manual (only via admin command).

## Appendix C: Research — How Other Systems Do Long-Running Agents

### C.1 OpenClaw (nicbarker/openclaw) — The Reference Architecture

OpenClaw is the most mature long-running personal AI agent. Its architecture is the gold standard for what we're building.

**Architecture Layers:**

| Layer | Responsibility | OpenClaw Pattern |
|-------|---------------|-----------------|
| **Gateway** | Connection mgmt, routing, auth | Single-process WebSocket server on `localhost:18789`, runs as daemon |
| **Execution** | Task ordering, concurrency | Per-session serial "Lane Queue" (one task at a time per session) |
| **Integration** | Platform normalization | Channel adapters (WhatsApp, Telegram, Slack, Discord, CLI, Web) |
| **Intelligence** | Agent behavior, knowledge | Embedded coding agent SDK (Pi) as brain + Skills + Memory + Heartbeat |

**Key Design Decisions:**

1. **Gateway-first, not brain-first.** The gateway was the original innovation. The brain is an embedded coding agent SDK. This separation means the agent doesn't know it's talking through WhatsApp — it just sees a task queue. Channel adapters normalize everything.

2. **Lane Queues (per-session serial execution).** Every session gets its own queue. Tasks execute one at a time within a session. Parallelism is opt-in (separate lanes for cron, subagents). Four queue modes:
   - `steer`: Inject into current run, skip pending tools
   - `followup`: Queue after current run completes
   - `collect`: Coalesce all waiting messages into single followup (default)
   - `interrupt`: Abort current run, process newest message

3. **Heartbeat loop.** A background process fires every 30 minutes. The agent reads a `HEARTBEAT.md` checklist, processes items, and can send proactive messages. The key insight: the agent schedules its own future. No external cron system needed.

4. **Embedded coding agent, not built from scratch.** OpenClaw embeds Pi (a Claude Code-like SDK) as the core runtime. It gets bash, file access, streaming, tool orchestration for free. The hard problem isn't the agent loop — it's the gateway, orchestration, and integration layers.

5. **Typed WebSocket protocol.** Every frame validated against JSON schema. Four event types: `agent`, `chat`, `presence`, `health`. All channels speak the same protocol.

6. **Process isolation (RFC underway).** Moving toward distributed runtime: separate control plane (gateway) from agent runtime (per-agent process). Enables per-agent lifecycle commands (start/stop/restart) without gateway restart. MVP is `local-split` mode (same host, separate processes, Unix sockets).

**What Rooster Code can learn:**
- Our gateway should emit typed events (not Rich-rendered text) — consumers decide rendering
- Per-session serial queues are essential — our current `prompt_once()` already serializes, but we need queue modes
- Heartbeat is the killer feature — Rooster Code should have a self-scheduling loop
- The SDK (brain) is already solved — our SDK integration is the equivalent of Pi

### C.2 Coder AI Agent — The Server-Side Agent

Coder runs agents as a control plane + workspace daemon architecture:

- **Control plane**: Runs the agent loop, streams to LLM, interprets tool calls, dispatches to workspaces
- **Workspace daemon**: Always dials OUT to control plane (never reverse) — control plane reaches back in via established tunnel
- **Lazy connections**: Workspace connection only established on first tool call, cached for session duration
- **Auto-compaction**: When token usage exceeds threshold, agent generates compressed summary, inserts as new message. Earlier messages excluded from context window but still visible to users.
- **Durable state**: All chat data stored in control plane database, not workspace. Survives workspace stops, rebuilds, deletions.

**What Rooster Code can learn:**
- Dial-out model (workspace → control plane) is safer than exposing ports — our daemon should be a local socket, not a network port
- Lazy initialization of subagents saves resources
- Separate chat state from workspace state — we already do this (SDK sessions vs filesystem)

### C.3 Claude Code — The Background Mode Evolution

Claude Code (Anthropic) faces the same challenge: moving from session-based to long-running. Key feature requests:

- **Background mode** for agents: `background: true` in agent frontmatter, agent always runs in background
- **Daemon mode / scheduling**: "keep this agent alive, processing a queue" — native watch + wake primitives
- **Event-driven wake**: Webhook triggers session resume — Slack reply arrives → Claude wakes up with context
- **Persistent processes**: Background task cleanup shouldn't kill long-running processes started via Bash tool

The common pattern: developers universally want agents that outlive sessions and respond to async triggers.

### C.4 agentctl — Universal Agent Control Plane

agentctl provides a standard CLI/daemon for managing multiple coding agent runtimes (Claude Code, Pi, OpenCode, OpenClaw, Codex):

- **Daemon with directory locks**: Prevents duplicate launches on same working directory
- **Fuse timers**: Directory-scoped TTL for automatic resource cleanup
- **Adapter model**: Each agent runtime gets an adapter — support added without changing CLI/daemon
- **State derived from native sources**: Reads actual agent state, never maintains shadow copies
- **Unix socket**: CLI ↔ daemon communication via `~/.agentctl/agentctl.sock`
- **Prometheus metrics**: Built-in observability

### C.5 Hosting Patterns for AI Agents (James Carr)

Seven proven patterns. Three most relevant to us:

| Pattern | Description | When to Use |
|---------|-------------|-------------|
| **Persistent Daemon** | Agent runs continuously, maintains state in memory, listens on socket or polls queue | Chatbots, interactive assistants, fast response with maintained context |
| **Durable Execution** | Each step is a retryable activity, orchestrator checkpoints after each step, resumes from last checkpoint on crash | Multi-step workflows, failure mid-way is expensive, long-running operations |
| **Self-Scheduling** | Agent schedules its own future work via cron-like internal loop | Variable-rate monitoring/research |

**Key insight**: The hosting pattern almost always evolves. Start with cron → add event subscription → daemon consuming from queue. Core logic (fetch, process, write) stays the same — just swapping the trigger.

### C.6 persistent-agent-framework (T33R0) — Supabase-Backed

A production-tested architecture for persistent Claude Code agents:

- **Supabase backend**: SQL schema for shared state across sessions
- **Marker processing engine**: Invisible side-effects (memory, state, learning) embedded in LLM responses via brace markers
- **Sweep daemon**: Processing sweeps every 10s for conversation tasks, 60s for embeddings
- **Circuit breaker**: 3 consecutive failures → disable sweep + alert
- **Multi-platform**: Same agent identity across CLI, Telegram, Discord, web
- **Agent hierarchy**: Subordinate agents with inter-agent communication and security boundaries

### C.7 Key Takeaways for Rooster Code

1. **Gateway-first design.** Start with how the agent shows up — not how it thinks. The gateway should emit typed events, not rendered text. Any consumer can then render however it wants.

2. **Per-session serial queues are non-negotiable.** Default to serial execution within a session. Opt into parallelism when provably safe. This prevents race condition bugs that are nearly impossible to debug later.

3. **Heartbeat is the minimum viable long-running feature.** A 30-minute pulse where the agent checks a checklist and decides what to do next. This alone makes Rooster Code feel "alive."

4. **Embedded SDK, don't build one.** We already embed the Open Agent SDK. This is correct. The hard work is the gateway, orchestration, state durability, and channel adapters.

5. **Durable execution is the endgame.** Long-term, every step should checkpoint to a database. Crash → resume from last checkpoint. But start with periodic checkpoints (every N turns) before full durable execution.

6. **Local-only by default, remote optional.** Bind to localhost/Unix socket. Remote access via SSH tunnel or Tailscale, not direct exposure. OpenClaw and Coder both do this.

7. **Adapters are the scaling model.** Each channel/input source gets an adapter. Core engine doesn't know about them. Add a new channel by writing a new adapter, never touching engine code.

### C.8 Additional Systems & Patterns (Second-Round Research)

#### Cloudflare Agents — Hibernation Model

Cloudflare's agent model is fundamentally different: agents are not long-running processes at all. They are Durable Objects that wake, work, and sleep:

- **Wake on event**: HTTP request, WebSocket message, scheduled alarm, inbound email triggers activation
- **Zero compute when idle**: Agent hibernates completely — no CPU, no memory
- **SQLite state**: `setState()` + `this.sql` persist across activations
- **Keep-alive**: `keepAlive()` prevents eviction during active work, resets inactivity timer
- **Crash recovery via fibers**: `runFiber()` persists a SQLite row for work duration; `stash()` saves intermediate state; `onFiberRecovered()` handles restart
- **Chat recovery**: `chatRecovery` wraps each chat turn in a fiber for automatic stream recovery

**Key lesson**: Not every long-running agent needs to be a persistent process. The hibernation model (wake → work → sleep) is more resource-efficient. But it requires a platform that handles wake triggers.

#### Lucyd — Python Daemon with Memory Evolution

A Python daemon for persona-rich agents (nicolasforstinger/lucyd):

- **Agentic tool-use loop**: Swappable strategy (multi-turn `ToolUseStrategy` or `SingleShotStrategy`), auto-fallback for models without tool support
- **Persistent sessions**: JSONL audit trail + atomic state snapshots, survives restarts
- **Long-term memory**: SQLite FTS5 + vector similarity search via OpenAI embeddings
- **Memory evolution**: Daily rewriting of workspace understanding files, anchored against static identity
- **Cost tracking**: Per-model, per-session, per-day token cost recording in SQLite
- **Channel adapters**: Telegram, CLI, HTTP API, headless
- **~1,100 tests**: Component, contract, integration, conversation replay, streaming

**Key lesson**: Memory evolution (daily consolidation) is a critical long-running agent feature. Without it, agent knowledge goes stale. Rooster Code should consider periodic memory rewrites.

#### Yodoca — Nano-Kernel + Extension Architecture

A TypeScript agent runtime (VitalyOborin/yodoca) with a minimal core:

- **Nano-kernel**: Tiny core with Loader → MessageRouter → Orchestrator → EventBus
- **Everything is an extension**: Channels, tools, services, schedulers, even other agents
- **Durable event bus**: Single SQLite WAL table as queue + audit log + workflow backbone
- **At-least-once delivery**: Handlers must be idempotent
- **Supervisor**: Process watcher for safe restarts
- **Graph-based memory**: Nodes, edges, entities with FTS5 + vector hybrid search
- **Task engine**: Durable multi-step background tasks with checkpointing, retries, human review

**Key lesson**: The nano-kernel pattern (tiny core, extensions for everything) is how you keep the daemon maintainable as it grows. OpenClaw follows the same pattern with channel adapters.

#### Pydantic AI + DBOS — Durable Execution as a Library

DBOS wraps any Pydantic AI agent for durable execution with ~10 lines of code:

```python
from pydantic_ai.durable_exec.dbos import DBOSAgent
dbos_agent = DBOSAgent(agent)
result = await dbos_agent.run("What is the capital of Mexico?")
```

- **Automatic checkpointing**: Wraps `Agent.run()` as DBOS workflow, model calls and MCP as steps
- **Crash recovery**: Resumes from last checkpointed step, replays completed calls from database
- **No external infrastructure**: SQLite or Postgres — runs in-process
- **Child workflows**: Agents calling other agents are automatically durable
- **Parallel execution**: `DBOS.start_workflow_async` for fan-out, fan-in

**Key lesson**: Durable execution doesn't need complex infrastructure. A library + SQLite is enough. This is the pattern we should aim for in Phase 1 (State Store).

#### OpenAI Agents SDK — Session & RunState Primitives

The Agents SDK provides built-in session management:

- **Session backends**: SQLite, AsyncSQLite, Redis, SQLAlchemy, MongoDB, Dapr, OpenAI Conversations, Encrypted
- **Automatic history**: Runner retrieves session history before each run, stores new items after
- **Compaction**: `OpenAIResponsesCompactionSession` wrapper auto-compacts long conversations
- **RunState**: Serializable snapshot of agent run — context, usage, interruptions, model responses
- **Human-in-the-loop**: `approve()`, `reject()` on tool calls, resume from RunState

**Key lesson**: Multiple session backends (SQLite for local, Redis for distributed) with a common interface is the right pattern. Rooster Code should design `StateStore` with backend abstraction from the start.

#### OmniDaemon — Process-Per-Agent Supervisor

Production-grade agent process isolation (omnirexflora-labs/OmniDaemon):

- **Agent Supervisor**: Each agent runs in its own isolated process
- **Fault isolation**: Agent A crashes → Agent B keeps running
- **Auto-recovery**: Crashed agents restart automatically
- **Health monitoring**: Heartbeat checks, timeout handling
- **Event-driven**: Redis/Kafka/RabbitMQ event bus for agent communication
- **Framework agnostic**: Works with any agent framework

**Key lesson**: True process isolation (separate Python processes) is valuable but complex. Start with in-process isolation (separate asyncio tasks + locks) and graduate to subprocess isolation when needed.

#### AgentPool — Unified Agent Orchestration Hub

YAML-configured multi-agent orchestration (phil65/agentpool):

- **Heterogeneous agents**: Native (PydanticAI), Claude Code, Codex, ACP agents, AG-UI agents — all in one YAML
- **Protocol bridging**: Expose all agents through ACP, OpenCode, MCP, AG-UI, OpenAI API
- **Team composition**: `agent1 & agent2` (parallel) or `agent1 | agent2 | agent3` (sequential pipeline)
- **Server modes**: `serve-acp` for IDE integration, `serve-opencode` for OpenCode TUI, `serve-mcp` for tool exposure

**Key lesson**: Rooster Code's team system could expose agents through standard protocols (ACP, MCP). This makes Rooster Code agents usable by other tools, not just the Rooster Code CLI.

#### Claude Code — Real-World Pain Points

The Claude Code issue tracker reveals exactly what developers struggle with:

1. **Agent continuity (#16375)**: 3 agents lost ~1,800 lines of work when session ran out of context → handoff protocol + state persistence needed
2. **Session persistence (#48799)**: 118 autonomous sessions with handoff protocols, context guardians, sleep cycle protocols, autonomous work queues — all built externally because Claude Code lacks native persistence
3. **Daemon mode (#28229)**: "Keep this agent alive, processing a queue" — native watch + wake primitives needed
4. **Multi-terminal (#51315)**: Single agent owning multiple terminals for daemon+client workflows
5. **Background mode (#22034)**: `background: true` in agent frontmatter so agents always run async

**Key lesson**: The #1 missing feature across ALL coding agents is session persistence and agent continuity. Every user eventually builds external orchestration. Rooster Code can leapfrog by building this in natively.

#### AgentIRC Codex — Daemon Spawning Agent Subprocess

Architecture for an IRC-native agent (agentirc.dev):

- **Daemon**: Python asyncio process that stays alive
- **Agent subprocess**: Spawns `codex app-server`, communicates via JSON-RPC over stdio
- **Supervisor**: Separate Sonnet 4.6 process that periodically evaluates agent activity, issues whispers/corrections
- **Thread persistence**: Codex thread persists between activations — each turn picks up from same thread ID
- **Isolated state**: `XDG_DATA_HOME` and `XDG_STATE_HOME` overridden per session, HOME preserved for auth

**Key lesson**: The daemon-owns-transport, agent-owns-reasoning split is clean. Our daemon doesn't need to do LLM reasoning — it just needs to spawn and manage agent sessions.
