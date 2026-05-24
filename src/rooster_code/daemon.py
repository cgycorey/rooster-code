from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import signal
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

from rooster_code.config import resolve_runtime_env
from rooster_code.runtime import create_runtime_agent

log = logging.getLogger("rooster.daemon")

_SCHEMA_SESSIONS = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    cwd TEXT NOT NULL DEFAULT '.',
    created_at REAL NOT NULL,
    last_active_at REAL NOT NULL
);
"""

_SCHEMA_CRON = """
CREATE TABLE IF NOT EXISTS cron_jobs (
    job_id TEXT PRIMARY KEY,
    schedule TEXT NOT NULL DEFAULT '',
    command TEXT NOT NULL DEFAULT '',
    name TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    last_run_at REAL,
    enabled INTEGER NOT NULL DEFAULT 1
);
"""

_SCHEMA_TEAMS = """
CREATE TABLE IF NOT EXISTS team_snapshots (
    team_id TEXT PRIMARY KEY,
    team_name TEXT NOT NULL DEFAULT '',
    members TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL
);
"""


class StateStore:

    def __init__(self, db_path: str | None = None) -> None:
        if db_path is None:
            db_path = str(Path.home() / ".rooster-code" / "daemon.db")
        self.db_path = Path(db_path)
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(_SCHEMA_SESSIONS)
            self._conn.execute(_SCHEMA_CRON)
            self._conn.commit()
        except (sqlite3.OperationalError, sqlite3.DatabaseError, OSError) as e:
            log.error("Cannot open state database at %s: %s", self.db_path, e)
            raise SystemExit(f"Cannot open state database at {self.db_path}: {e}") from e

    def upsert_session(self, session_id: str, cwd: str = ".") -> None:
        now = time.time()
        self._conn.execute(
            "INSERT INTO sessions (session_id, cwd, created_at, last_active_at) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(session_id) DO UPDATE SET last_active_at=excluded.last_active_at",
            (session_id, cwd, now, now),
        )
        self._conn.commit()

    def touch_session(self, session_id: str) -> None:
        self._conn.execute(
            "UPDATE sessions SET last_active_at = ? WHERE session_id = ?",
            (time.time(), session_id),
        )
        self._conn.commit()

    def remove_session(self, session_id: str) -> None:
        self._conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        self._conn.commit()

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()
        return row[0] if row else 0

    def list_sessions(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT session_id, cwd, created_at, last_active_at FROM sessions"
        ).fetchall()
        return [
            {
                "session_id": row[0],
                "cwd": row[1],
                "created_at": row[2],
                "last_active_at": row[3],
            }
            for row in rows
        ]

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT session_id, cwd, created_at, last_active_at FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "session_id": row[0],
            "cwd": row[1],
            "created_at": row[2],
            "last_active_at": row[3],
        }

    def close(self) -> None:
        self._conn.close()

    def cleanup_inactive(self, max_age_seconds: float = 86400) -> int:
        cutoff = time.time() - max_age_seconds
        cursor = self._conn.execute("DELETE FROM sessions WHERE last_active_at < ?", (cutoff,))
        self._conn.commit()
        return cursor.rowcount


class TeamSnapshotStore:

    def __init__(self, db_path: str | None = None) -> None:
        if db_path is None:
            db_path = str(Path.home() / ".rooster-code" / "daemon.db")
        self.db_path = Path(db_path)
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(_SCHEMA_TEAMS)
            self._conn.commit()
        except (sqlite3.OperationalError, sqlite3.DatabaseError, OSError) as e:
            log.error("Cannot open team snapshot database at %s: %s", self.db_path, e)
            raise SystemExit(f"Cannot open team snapshot database at {self.db_path}: {e}") from e

    def upsert_team(self, team_id: str, team_name: str, members: str) -> None:
        now = time.time()
        self._conn.execute(
            "INSERT OR REPLACE INTO team_snapshots (team_id, team_name, members, created_at) "
            "VALUES (?, ?, ?, ?)",
            (team_id, team_name, members, now),
        )
        self._conn.commit()

    def get_team(self, team_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT team_id, team_name, members, created_at FROM team_snapshots WHERE team_id = ?",
            (team_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "team_id": row[0],
            "team_name": row[1],
            "members": row[2],
            "created_at": row[3],
        }

    def list_teams(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT team_id, team_name, members, created_at FROM team_snapshots"
        ).fetchall()
        return [
            {"team_id": r[0], "team_name": r[1], "members": r[2], "created_at": r[3]}
            for r in rows
        ]

    def remove_team(self, team_id: str) -> None:
        self._conn.execute("DELETE FROM team_snapshots WHERE team_id = ?", (team_id,))
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


def save_team_snapshot(
    team_id: str, team_name: str, members: str, db_path: str | None = None
) -> None:
    """Persist a team snapshot to the daemon SQLite database."""
    store = TeamSnapshotStore(db_path=db_path)
    try:
        store.upsert_team(team_id, team_name, members)
    finally:
        store.close()


def drop_team_snapshot(team_id: str, db_path: str | None = None) -> None:
    """Remove a team snapshot from the daemon SQLite database."""
    store = TeamSnapshotStore(db_path=db_path)
    try:
        store.remove_team(team_id)
    finally:
        store.close()


def read_team_snapshots(db_path: str | None = None) -> list[dict[str, Any]]:
    """Read all team snapshots from the daemon SQLite database."""
    store = TeamSnapshotStore(db_path=db_path)
    try:
        return store.list_teams()
    finally:
        store.close()


_SOCKET_PATH = Path("/tmp/rooster-code.sock")


class PersistentCronStore(dict[str, dict[str, Any]]):
    """SQLite-backed dict for cron jobs that the SDK tools use transparently.

    The SDK's CronCreate/CronDelete/CronList tools do:
        from open_agent_sdk.tools import _cron_jobs
    at runtime inside call(). Rebinding the module attribute to an instance
    of this class gives persistence without replacing any SDK code.
    """

    def __init__(self, db_path: str) -> None:
        super().__init__()
        try:
            self._conn = sqlite3.connect(db_path)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(_SCHEMA_CRON)
            self._conn.commit()
            self._load_all()
        except (sqlite3.OperationalError, sqlite3.DatabaseError, OSError) as e:
            log.error("Cannot open cron database at %s: %s", db_path, e)
            raise SystemExit(f"Cannot open cron database at {db_path}: {e}") from e

    def _load_all(self) -> None:
        rows = self._conn.execute(
            "SELECT job_id, schedule, command, name, created_at, last_run_at, enabled FROM cron_jobs"
        ).fetchall()
        for row in rows:
            super().__setitem__(row[0], {
                "id": row[0],
                "schedule": row[1],
                "command": row[2],
                "name": row[3],
                "created_at": row[4],
                "last_run_at": row[5],
                "enabled": bool(row[6]),
            })

    def __setitem__(self, key: str, value: dict[str, Any]) -> None:
        super().__setitem__(key, value)
        self._conn.execute(
            "INSERT OR REPLACE INTO cron_jobs (job_id, schedule, command, name, created_at, last_run_at, enabled) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                key,
                value.get("schedule", ""),
                value.get("command", ""),
                value.get("name", f"job_{key}"),
                value.get("created_at", time.time()),
                value.get("last_run_at"),
                1 if value.get("enabled", True) else 0,
            ),
        )
        self._conn.commit()

    def __delitem__(self, key: str) -> None:
        super().__delitem__(key)
        self._conn.execute("DELETE FROM cron_jobs WHERE job_id = ?", (key,))
        self._conn.commit()

    def clear(self) -> None:
        super().clear()
        self._conn.execute("DELETE FROM cron_jobs")
        self._conn.commit()

    def mark_run(self, job_id: str) -> None:
        now = time.time()
        if job_id in self:
            self[job_id]["last_run_at"] = now
        self._conn.execute("UPDATE cron_jobs SET last_run_at = ? WHERE job_id = ?", (now, job_id))
        self._conn.commit()

    def update_last_run(self, job_id: str, last_run_at: float) -> None:
        if job_id in self:
            self[job_id]["last_run_at"] = last_run_at
            self._conn.execute("UPDATE cron_jobs SET last_run_at = ? WHERE job_id = ?", (last_run_at, job_id))
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()


def _get_agent_session_id(agent: Any) -> str:
    opts = getattr(agent, "_options", None)
    if opts is not None:
        sid = getattr(opts, "session_id", "") or getattr(opts, "resume", "")
        if sid:
            return sid
    return getattr(agent, "session_id", "") or getattr(agent, "_session_id", "")


class AgentDaemon:

    def __init__(
        self,
        socket_path: str | None = None,
        db_path: str | None = None,
        max_sessions: int = 0,
        heartbeat_interval: int = 0,
        file_configs: dict[str, Any] | None = None,
    ) -> None:
        self.socket_path = Path(socket_path) if socket_path else _SOCKET_PATH
        self.state = StateStore(db_path=db_path)
        self.cron = PersistentCronStore(db_path=db_path or str(Path.home() / ".rooster-code" / "daemon.db"))
        self._server: asyncio.Server | None = None
        self._shutdown_event = asyncio.Event()
        self._adapters: list[Any] = []
        self._max_sessions = max_sessions
        self._heartbeat_interval = heartbeat_interval
        self._file_configs = file_configs or {}
        self._start_time: float = 0.0
        self._queries: int = 0
        self._total_latency_ms: float = 0.0
        self._total_tokens: int = 0
        self._total_cost: float = 0.0
        self._state_lock = asyncio.Lock()
        self._running_cron: set[str] = set()
        self._query_handler = self._build_handler()

    def add_telegram(self, token: str, *, allowed_users: list[int] | None = None) -> None:
        from rooster_code.adapters.telegram import TelegramAdapter

        self._adapters.append(
            TelegramAdapter(token=token, query_handler=self._query_handler, allowed_users=allowed_users)
        )

    async def start(self) -> None:
        self._start_time = time.monotonic()
        self._setup_signal_handlers()
        if self.socket_path.exists():
            self.socket_path.unlink()
        self._server = await asyncio.start_unix_server(self._handle_client, path=str(self.socket_path))
        log.info("listening on %s", self.socket_path)
        import open_agent_sdk.tools as sdk_tools
        sdk_tools._cron_jobs = self.cron
        log.info("cron store patched (%d jobs loaded)", len(self.cron))
        for adapter in self._adapters:
            await adapter.start()
        cleanup = asyncio.create_task(self._cleanup_loop())
        cron_runner = asyncio.create_task(self._cron_runner_loop())
        heartbeat = asyncio.create_task(self._heartbeat_loop()) if self._heartbeat_interval else None
        await self._shutdown_event.wait()
        cleanup.cancel()
        cron_runner.cancel()
        if heartbeat:
            heartbeat.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await cleanup
        with contextlib.suppress(asyncio.CancelledError):
            await cron_runner
        if heartbeat:
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat

    async def shutdown(self) -> None:
        log.info("shutting down...")
        for adapter in self._adapters:
            with contextlib.suppress(Exception):
                await adapter.stop()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
        self.state.close()
        self.cron.close()
        if self.socket_path.exists():
            self.socket_path.unlink()
        log.info("shutdown complete")

    def _build_handler(self):
        self_ref = self

        async def handler(session_id: str, user_id: str, prompt: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
            state = self_ref.state
            overrides = overrides or {}
            cwd = str(overrides.pop("cwd", ".") or ".")

            if self_ref._max_sessions and not state.get_session(session_id):
                if state.count() >= self_ref._max_sessions:
                    return {"text": "Error: maximum sessions reached. Try again later or specify an existing session_id.", "tokens": 0, "cost": 0.0, "turns": 0}

            config = resolve_runtime_env(os.environ, cwd=cwd)
            config.persist_session = True
            if session_id and state.get_session(session_id):
                config.resume = session_id
            elif session_id:
                config.session_id = session_id
            for file_key, file_val in self_ref._file_configs.items():
                if file_val is not None and not getattr(config, file_key, None):
                    setattr(config, file_key, file_val)
            for key in _CONFIG_KEYS:
                val = overrides.get(key)
                if val is not None:
                    setattr(config, key, val)
            env_override: dict[str, str] | None = overrides.get("env")
            if env_override is not None:
                config.env = {**config.env, **env_override}
            headers_override: dict[str, str] | None = overrides.get("custom_headers")
            if headers_override is not None:
                config.custom_headers = headers_override
            agent = create_runtime_agent(config)
            try:
                if hasattr(agent, "_initialize") and callable(agent._initialize):
                    await agent._initialize()

                sid = _get_agent_session_id(agent) or session_id or "default"
                async with self_ref._state_lock:
                    state.upsert_session(sid, cwd=cwd)

                t0 = time.monotonic()
                try:
                    last_error: Exception | None = None
                    result = None
                    for attempt in range(_MAX_RETRIES):
                        try:
                            result = await asyncio.wait_for(agent.prompt(prompt), timeout=_QUERY_TIMEOUT)
                            break
                        except asyncio.TimeoutError:
                            return {"text": "Error: query timed out after 300s", "tokens": 0, "cost": 0.0, "turns": 0}
                        except Exception as exc:
                            last_error = exc
                            if not _is_retriable(exc) or attempt == _MAX_RETRIES - 1:
                                return {"text": f"Error: {exc}", "tokens": 0, "cost": 0.0, "turns": 0}
                            delay = _RETRY_BASE_DELAY * (2 ** attempt)
                            log.warning("retrying query after %ss (attempt %d/%d): %s", delay, attempt + 2, _MAX_RETRIES, exc)
                            await asyncio.sleep(delay)
                    if result is None:
                        return {"text": f"Error: {last_error or 'no result'}", "tokens": 0, "cost": 0.0, "turns": 0}
                    usage = getattr(result, "usage", None)
                    tokens = 0
                    if usage:
                        tokens = (getattr(usage, "input_tokens", 0) or 0) + (getattr(usage, "output_tokens", 0) or 0)
                    cost = getattr(result, "cost", 0.0) or 0.0
                    turns = getattr(result, "num_turns", 0) or 0
                    self_ref._total_tokens += tokens
                    self_ref._total_cost += cost
                    return {"text": result.text or "", "tokens": tokens, "cost": cost, "turns": turns}
                finally:
                    elapsed = (time.monotonic() - t0) * 1000
                    self_ref._queries += 1
                    self_ref._total_latency_ms += elapsed
                    async with self_ref._state_lock:
                        state.touch_session(sid)
            finally:
                with contextlib.suppress(Exception):
                    await agent.close()

        return handler

    def _setup_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._on_signal)

    async def _cleanup_loop(self) -> None:
        while True:
            await asyncio.sleep(3600)
            try:
                async with self._state_lock:
                    removed = self.state.cleanup_inactive()
                if removed:
                    log.info("cleaned %d inactive session(s)", removed)
            except Exception:
                log.warning("session cleanup failed", exc_info=True)

    async def _cron_runner_loop(self) -> None:
        while True:
            await asyncio.sleep(30)
            if not self.cron:
                continue
            try:
                jobs = list(self.cron.values())
                now = time.time()
                for job in jobs:
                    if not job.get("enabled", True):
                        continue
                    schedule = job.get("schedule", "")
                    if not schedule:
                        continue
                    job_id = job["id"]
                    if job_id in self._running_cron:
                        continue
                    last_run = job.get("last_run_at")
                    if last_run is None:
                        self.cron.mark_run(job_id)
                        continue
                    if not _cron_is_due(schedule, last_run, now, job_id=job_id):
                        continue
                    job_name = job.get("name", job_id)
                    self._running_cron.add(job_id)
                    log.info("cron: running job %s (%s)", job_name, job_id)
                    try:
                        result = await self._query_handler(
                            f"cron-{job_id}", "cron", job.get("command", "")
                        )
                        log.info("cron: job %s completed: tokens=%d cost=$%.4f",
                                 job_name, result["tokens"], result["cost"])
                    except Exception as exc:
                        log.error("cron: job %s failed: %s", job_name, exc)
                    self.cron.mark_run(job_id)
                    self._running_cron.discard(job_id)
            except Exception:
                log.warning("cron runner iteration failed", exc_info=True)

    async def _heartbeat_loop(self) -> None:
        heartbeat_prompt = (
            "Heartbeat: report daemon state concisely. "
            "Sessions, queries, tokens, cost. One sentence summary. No tools needed."
        )
        while True:
            await asyncio.sleep(self._heartbeat_interval)
            try:
                result = await self._query_handler("heartbeat", "heartbeat", heartbeat_prompt)
                log.info(
                    "heartbeat: sessions=%d queries=%d tokens=%d cost=$%.4f result=%s",
                    self.state.count(),
                    self._queries,
                    self._total_tokens,
                    round(self._total_cost, 4),
                    result["text"][:200] if result["text"] else "",
                )
            except Exception as exc:
                log.error("heartbeat failed: %s", exc)

    def _on_signal(self) -> None:
        self._shutdown_event.set()

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            raw = await reader.readline()
            if not raw:
                return
            request: dict[str, Any] = json.loads(raw.decode("utf-8"))
            action = request.get("action", "")
            if action == "query":
                await self._handle_query(request, writer)
            elif action == "sessions":
                await self._handle_list_sessions(writer)
            elif action == "session_info":
                await self._handle_session_info(request, writer)
            elif action == "session_delete":
                await self._handle_session_delete(request, writer)
            elif action == "session_rename":
                await self._handle_session_rename(request, writer)
            elif action == "session_tag":
                await self._handle_session_tag(request, writer)
            elif action == "health":
                await self._handle_health(writer)
            elif action == "shutdown":
                self._shutdown_event.set()
                await self._reply(writer, {"type": "shutdown", "status": "ok"})
            elif action == "cron_list":
                await self._handle_cron_list(writer)
            elif action == "cron_delete":
                await self._handle_cron_delete(request, writer)
            else:
                await self._reply(writer, {"type": "error", "message": f"unknown action: {action}"})
        except json.JSONDecodeError:
            await self._reply(writer, {"type": "error", "message": "invalid JSON"})
        except Exception as exc:
            await self._reply(writer, {"type": "error", "message": str(exc)})
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    async def _handle_query(self, request: dict[str, Any], writer: asyncio.StreamWriter) -> None:
        session_id = str(request.get("session_id") or "")
        prompt = str(request.get("prompt") or "")
        if not prompt:
            await self._reply(writer, {"type": "error", "message": "prompt is required"})
            return
        overrides = {}
        for k in _CONFIG_ALL_KEYS:
            v = request.get(k)
            if v is not None:
                overrides[k] = v
        cwd = str(request.get("cwd", ".") or ".")
        overrides["cwd"] = cwd
        result = await self._query_handler(session_id, "unix-socket", prompt, overrides)
        await self._reply(writer, {
            "type": "done",
            "session_id": session_id or "default",
            "text": result["text"],
            "tokens": result["tokens"],
            "cost": result["cost"],
            "turns": result["turns"],
        })

    async def _handle_list_sessions(self, writer: asyncio.StreamWriter) -> None:
        sessions = self.state.list_sessions()
        await self._reply(writer, {"type": "sessions", "data": sessions})

    async def _handle_session_delete(self, request: dict[str, Any], writer: asyncio.StreamWriter) -> None:
        session_id = str(request.get("session_id") or "")
        if not session_id:
            await self._reply(writer, {"type": "error", "message": "session_id required"})
            return
        try:
            from rooster_code.runtime import delete_session
            await delete_session(session_id)
        except Exception as exc:
            await self._reply(writer, {"type": "error", "message": str(exc)})
            return
        async with self._state_lock:
            self.state.remove_session(session_id)
        await self._reply(writer, {"type": "session_deleted", "session_id": session_id})

    async def _handle_session_info(self, request: dict[str, Any], writer: asyncio.StreamWriter) -> None:
        session_id = str(request.get("session_id") or "")
        if not session_id:
            await self._reply(writer, {"type": "error", "message": "session_id required"})
            return
        local = self.state.get_session(session_id)
        try:
            from rooster_code.runtime import get_session_info as sdk_get_session_info
            sdk_info = await sdk_get_session_info(session_id)
        except Exception:
            log.warning("failed to fetch SDK session info for %s", session_id, exc_info=True)
            sdk_info = None
        await self._reply(writer, {
            "type": "session_info",
            "session_id": session_id,
            "local": local,
            "sdk": sdk_info,
        })

    async def _handle_session_rename(self, request: dict[str, Any], writer: asyncio.StreamWriter) -> None:
        session_id = str(request.get("session_id") or "")
        title = str(request.get("title") or "")
        if not session_id or not title:
            await self._reply(writer, {"type": "error", "message": "session_id and title required"})
            return
        try:
            from rooster_code.runtime import rename_session
            await rename_session(session_id, title)
        except Exception as exc:
            await self._reply(writer, {"type": "error", "message": str(exc)})
            return
        await self._reply(writer, {"type": "session_renamed", "session_id": session_id, "title": title})

    async def _handle_session_tag(self, request: dict[str, Any], writer: asyncio.StreamWriter) -> None:
        session_id = str(request.get("session_id") or "")
        tags = request.get("tags", [])
        if not session_id:
            await self._reply(writer, {"type": "error", "message": "session_id required"})
            return
        if not isinstance(tags, list):
            tags = []
        try:
            from rooster_code.runtime import tag_session
            await tag_session(session_id, tags)
        except Exception as exc:
            await self._reply(writer, {"type": "error", "message": str(exc)})
            return
        await self._reply(writer, {"type": "session_tagged", "session_id": session_id, "tags": tags})

    async def _handle_health(self, writer: asyncio.StreamWriter) -> None:
        adapter_status: dict[str, str] = {}
        for adapter in self._adapters:
            name = type(adapter).__name__
            try:
                adapter_status[name] = "ok" if await adapter.health() else "degraded"
            except Exception:
                adapter_status[name] = "error"
        avg_latency = round(self._total_latency_ms / self._queries, 1) if self._queries else 0
        await self._reply(writer, {
            "type": "health",
            "status": "ok",
            "uptime_seconds": round(time.monotonic() - self._start_time, 1),
            "queries": self._queries,
            "avg_latency_ms": avg_latency,
            "total_tokens": self._total_tokens,
            "total_cost": round(self._total_cost, 4),
            "sessions": self.state.count(),
            "max_sessions": self._max_sessions or "unlimited",
            "adapters": adapter_status,
        })

    async def _handle_cron_list(self, writer: asyncio.StreamWriter) -> None:
        await self._reply(writer, {
            "type": "cron_list",
            "jobs": [dict(job) for job in self.cron.values()],
        })

    async def _handle_cron_delete(self, request: dict[str, Any], writer: asyncio.StreamWriter) -> None:
        job_id = str(request.get("job_id") or "")
        if not job_id:
            await self._reply(writer, {"type": "error", "message": "job_id required"})
            return
        if job_id not in self.cron:
            await self._reply(writer, {"type": "error", "message": f"cron job '{job_id}' not found"})
            return
        del self.cron[job_id]
        await self._reply(writer, {"type": "cron_deleted", "job_id": job_id})

    @staticmethod
    async def _reply(writer: asyncio.StreamWriter, data: dict[str, Any]) -> None:
        writer.write((json.dumps(data, default=str) + "\n").encode("utf-8"))
        await writer.drain()


_QUERY_TIMEOUT = 300
_CONNECT_TIMEOUT = 5
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0

# Config keys that can be applied directly via setattr (no merging)
_CONFIG_KEYS = ("model", "max_turns", "max_tokens", "permission_mode", "allowed_tools", "disallowed_tools", "thinking_budget", "max_budget_usd", "debug", "sandbox", "include_partials")
# Keys that require merge semantics (dict-on-dict) rather than simple replacement
_CONFIG_MERGE_KEYS = ("env", "custom_headers")
# All keys accepted as query overrides (setattr + merge keys)
_CONFIG_ALL_KEYS = _CONFIG_KEYS + _CONFIG_MERGE_KEYS


def _cron_is_due(schedule: str, last_run: float, now: float, *, job_id: str = "") -> bool:
    if not schedule:
        return False
    import datetime
    delta = now - last_run
    now_dt = datetime.datetime.fromtimestamp(now, tz=datetime.timezone.utc)
    parts = schedule.strip().split()
    if len(parts) != 5:
        log.warning("cron: job %s has unparseable schedule '%s', defaulting to 24h", job_id, schedule)
        return delta >= 86400
    minute, hour, dom, mon, dow = parts
    try:
        if minute.startswith("*/"):
            interval = int(minute[2:])
            return delta >= interval * 60
        if hour.startswith("*/"):
            interval = int(hour[2:])
            return delta >= interval * 3600
        if minute != "*" and minute.isdigit():
            if now_dt.minute == int(minute) and delta >= 60:
                return True
    except (ValueError, TypeError):
        log.warning("cron: job %s schedule '%s' not recognized, defaulting to 24h", job_id, schedule)
        return delta >= 86400
    log.warning("cron: job %s schedule '%s' not recognized, defaulting to 24h", job_id, schedule)
    return delta >= 86400


def _is_retriable(exc: Exception) -> bool:
    msg = str(exc).lower()
    non_retriable = (
        "insufficient balance",
        "payment required",
        "invalid api key",
        "unauthorized",
        "not found",
        "model not found",
    )
    return not any(phrase in msg for phrase in non_retriable)


async def _send_to_daemon(request: dict[str, Any]) -> dict[str, Any]:
    reader, writer = await asyncio.wait_for(
        asyncio.open_unix_connection(str(_SOCKET_PATH)),
        timeout=_CONNECT_TIMEOUT,
    )
    try:
        writer.write((json.dumps(request) + "\n").encode("utf-8"))
        await writer.drain()
        raw = await reader.readline()
        if not raw:
            return {"type": "error", "message": "no response from daemon"}
        return json.loads(raw.decode("utf-8"))
    finally:
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()


async def daemon_query(prompt: str, *, session_id: str = "", cwd: str = ".", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    request: dict[str, Any] = {"action": "query", "prompt": prompt, "session_id": session_id, "cwd": cwd}
    if overrides:
        for k, v in overrides.items():
            if v is not None:
                request[k] = v
    return await _send_to_daemon(request)


async def daemon_health() -> dict[str, Any]:
    return await _send_to_daemon({"action": "health"})


async def daemon_list_sessions() -> dict[str, Any]:
    return await _send_to_daemon({"action": "sessions"})


async def daemon_shutdown() -> dict[str, Any]:
    return await _send_to_daemon({"action": "shutdown"})


async def daemon_session_info(session_id: str) -> dict[str, Any]:
    return await _send_to_daemon({"action": "session_info", "session_id": session_id})


async def daemon_session_rename(session_id: str, title: str) -> dict[str, Any]:
    return await _send_to_daemon({"action": "session_rename", "session_id": session_id, "title": title})


async def daemon_session_tag(session_id: str, tags: list[str]) -> dict[str, Any]:
    return await _send_to_daemon({"action": "session_tag", "session_id": session_id, "tags": tags})


async def daemon_session_delete(session_id: str) -> dict[str, Any]:
    return await _send_to_daemon({"action": "session_delete", "session_id": session_id})


async def daemon_cron_list() -> dict[str, Any]:
    return await _send_to_daemon({"action": "cron_list"})


async def daemon_cron_delete(job_id: str) -> dict[str, Any]:
    return await _send_to_daemon({"action": "cron_delete", "job_id": job_id})


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S")

    parser = argparse.ArgumentParser(prog="rooster-daemon", description="Long-running agent daemon for Rooster Code")
    parser.add_argument("--socket", help="Unix socket path (default: /tmp/rooster-code.sock)", default=None)
    parser.add_argument("--db", help="SQLite state database path (default: ~/.rooster-code/daemon.db)", default=None)
    parser.add_argument("--telegram", help="Telegram bot token (optional)", default=None)
    parser.add_argument("--telegram-allowed", help="Comma-separated Telegram user IDs to allow", default=None)
    parser.add_argument("--max-sessions", type=int, default=0, help="Maximum concurrent sessions (0 = unlimited)")
    parser.add_argument("--heartbeat", type=int, default=0, metavar="SECONDS", help="Self-check interval in seconds (0 = disabled, e.g. 1800 for 30 min)")
    parser.add_argument("--mcp-file", help="JSON file for MCP server configuration", default=None)
    parser.add_argument("--hooks-file", help="JSON file for hook configuration", default=None)
    parser.add_argument("--agents-file", help="JSON file for agent definitions", default=None)
    parser.add_argument("--json-schema-file", help="JSON file for structured output schema", default=None)
    args = parser.parse_args()

    from rooster_code.config import load_json_file
    file_configs = {}
    for flag, attr in [("mcp_file", "mcp_servers"), ("hooks_file", "hooks"), ("agents_file", "agents"), ("json_schema_file", "json_schema")]:
        val = getattr(args, flag, None)
        if val:
            loaded = load_json_file(val)
            if loaded is not None:
                file_configs[attr] = loaded

    try:
        daemon = AgentDaemon(socket_path=args.socket, db_path=args.db, max_sessions=args.max_sessions, heartbeat_interval=args.heartbeat, file_configs=file_configs)
    except SystemExit:
        return 1

    if args.telegram:
        allowed = None
        if args.telegram_allowed:
            allowed = [int(uid.strip()) for uid in args.telegram_allowed.split(",") if uid.strip().isdigit()]
        daemon.add_telegram(args.telegram, allowed_users=allowed)

    async def _run() -> None:
        await daemon.start()
        await daemon.shutdown()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass
    finally:
        if daemon.socket_path.exists():
            daemon.socket_path.unlink()
    return 0


if __name__ == "__main__":
    sys.exit(main())
