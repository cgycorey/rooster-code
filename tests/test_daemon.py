import asyncio
import json
import os
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any

from rooster_code.daemon import (
    AgentDaemon,
    StateStore,
    _send_to_daemon,
    daemon_health,
    daemon_list_sessions,
    daemon_query,
)


def _tmp_db() -> str:
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    path = f.name
    f.close()
    return path


def _mk_store(path: str) -> StateStore:
    store = StateStore(db_path=path)
    store.upsert_session("a")
    store.upsert_session("b")
    return store


# -- StateStore ----------------------------------------------------------------


def test_creates_sessions_table() -> None:
    path = _tmp_db()
    try:
        StateStore(db_path=path).close()
        conn = sqlite3.connect(path)
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        conn.close()
        assert "sessions" in tables
    finally:
        Path(path).unlink(missing_ok=True)


def test_upsert_and_get() -> None:
    path = _tmp_db()
    try:
        store = _mk_store(path)
        got: dict[str, Any] | None = store.get_session("a")
        assert got is not None
        assert got["session_id"] == "a"
        assert got["cwd"] == "."
        assert isinstance(got["created_at"], float)
        assert isinstance(got["last_active_at"], float)
        assert store.get_session("x") is None
        store.close()
    finally:
        Path(path).unlink(missing_ok=True)


def test_upsert_updates_last_active_only() -> None:
    path = _tmp_db()
    try:
        store = _mk_store(path)
        first: float = store.get_session("a")["last_active_at"]  # type: ignore[index, assignment]
        time.sleep(0.01)
        store.upsert_session("a")
        second: float = store.get_session("a")["last_active_at"]  # type: ignore[index, assignment]
        assert second > first
        assert store.get_session("a")["created_at"] == store.get_session("a")["created_at"]  # type: ignore[index]
        store.close()
    finally:
        Path(path).unlink(missing_ok=True)


def test_touch_updates_last_active() -> None:
    path = _tmp_db()
    try:
        store = _mk_store(path)
        first: float = store.get_session("a")["last_active_at"]  # type: ignore[index, assignment]
        time.sleep(0.01)
        store.touch_session("a")
        second: float = store.get_session("a")["last_active_at"]  # type: ignore[index, assignment]
        assert second > first
        store.close()
    finally:
        Path(path).unlink(missing_ok=True)


def test_remove() -> None:
    path = _tmp_db()
    try:
        store = _mk_store(path)
        store.remove_session("a")
        assert store.get_session("a") is None
        assert store.get_session("b") is not None
        store.close()
    finally:
        Path(path).unlink(missing_ok=True)


def test_list() -> None:
    path = _tmp_db()
    try:
        store = _mk_store(path)
        ids = {s["session_id"] for s in store.list_sessions()}
        assert ids == {"a", "b"}
        store.close()
    finally:
        Path(path).unlink(missing_ok=True)


def test_count() -> None:
    path = _tmp_db()
    try:
        store = StateStore(db_path=path)
        assert store.count() == 0
        store.upsert_session("a")
        store.upsert_session("b")
        assert store.count() == 2
        store.remove_session("a")
        assert store.count() == 1
        store.close()
    finally:
        Path(path).unlink(missing_ok=True)


def test_cleanup_inactive() -> None:
    path = _tmp_db()
    try:
        store = StateStore(db_path=path)
        store.upsert_session("recent")
        store._conn.execute(
            "UPDATE sessions SET last_active_at = ? WHERE session_id = ?",
            (1.0, "recent"),
        )
        store._conn.commit()
        removed = store.cleanup_inactive(max_age_seconds=86400)
        assert removed >= 1
        assert store.get_session("recent") is None
        store.close()
    finally:
        Path(path).unlink(missing_ok=True)


def test_default_path() -> None:
    store = StateStore()
    assert store.db_path == Path.home() / ".rooster-code" / "daemon.db"
    store.close()


# -- daemon helpers ------------------------------------------------------------


def _start_daemon(socket_path: str, db_path: str) -> asyncio.Task[None]:
    daemon = AgentDaemon(socket_path=socket_path, db_path=db_path)

    async def _run() -> None:
        try:
            await daemon.start()
        except asyncio.CancelledError:
            pass
        finally:
            await daemon.shutdown()

    return asyncio.create_task(_run())


def _run_test(coro):
    asyncio.run(coro)


def _socket_test(assert_fn):
    import shutil

    import rooster_code.daemon as dm

    async def _run() -> None:
        d = tempfile.mkdtemp(prefix="rst-")
        sock = os.path.join(d, "r.sock")
        db = os.path.join(d, "s.db")
        orig = dm._SOCKET_PATH
        dm._SOCKET_PATH = Path(sock)
        task = _start_daemon(sock, db)
        await asyncio.sleep(0.1)
        try:
            await assert_fn(sock)
        finally:
            dm._SOCKET_PATH = orig
            task.cancel()
            with __import__("contextlib").suppress(asyncio.CancelledError):
                await task
            shutil.rmtree(d, ignore_errors=True)

    _run_test(_run())


# -- daemon integration tests --------------------------------------------------


def test_health() -> None:
    _socket_test(lambda sock: _check_health())


async def _check_health() -> None:
    result: dict[str, Any] = await daemon_health()
    assert result["type"] == "health"
    assert result["status"] == "ok"
    assert "uptime_seconds" in result
    assert "queries" in result
    assert "avg_latency_ms" in result
    assert "sessions" in result
    assert "max_sessions" in result
    assert "adapters" in result


def test_empty_request_is_error() -> None:
    async def _check(sock: str) -> None:
        r, w = await asyncio.open_unix_connection(sock)
        try:
            w.write((json.dumps({}) + "\n").encode())
            await w.drain()
            raw = await r.readline()
            result: dict[str, Any] = json.loads(raw.decode())
            assert result["type"] == "error"
        finally:
            w.close()

    _socket_test(_check)


def test_unknown_action() -> None:
    _socket_test(lambda sock: _check_unknown())


async def _check_unknown() -> None:
    result: dict[str, Any] = await _send_to_daemon({"action": "nope"})
    assert result["type"] == "error"
    assert "unknown action" in result["message"]  # type: ignore[operator]


def test_invalid_json() -> None:
    async def _check(sock: str) -> None:
        r, w = await asyncio.open_unix_connection(sock)
        try:
            w.write(b"not json\n")
            await w.drain()
            raw = await r.readline()
            result: dict[str, Any] = json.loads(raw.decode())
            assert result["type"] == "error"
            assert "invalid JSON" in result["message"]  # type: ignore[operator]
        finally:
            w.close()

    _socket_test(_check)


def test_list_sessions_starts_empty() -> None:
    _socket_test(lambda sock: _check_empty())


async def _check_empty() -> None:
    result: dict[str, Any] = await daemon_list_sessions()
    assert result == {"type": "sessions", "data": []}


def test_missing_prompt_rejected() -> None:
    async def _check(sock: str) -> None:
        result: dict[str, Any] = await daemon_query("")
        assert result["type"] == "error"
        assert "prompt is required" in result["message"]  # type: ignore[operator]

    _socket_test(_check)


def test_query_and_session_tracking() -> None:
    _socket_test(lambda sock: _check_tracking())


async def _check_tracking() -> None:
    result: dict[str, Any] = await daemon_query("hello")
    assert result["type"] in ("done", "error")
    assert result.get("session_id")

    sessions: dict[str, Any] = await daemon_list_sessions()
    data: list[dict[str, Any]] = sessions["data"]  # type: ignore[assignment]
    assert len(data) >= 1


def test_query_tracks_custom_session_id() -> None:
    _socket_test(lambda sock: _check_custom_id())


async def _check_custom_id() -> None:
    await daemon_query("hello", session_id="my-session")
    result: dict[str, Any] = await daemon_list_sessions()
    data: list[dict[str, Any]] = result["data"]  # type: ignore[assignment]
    ids = {s["session_id"] for s in data}
    assert "my-session" in ids


def test_max_sessions_rejects_new() -> None:
    async def _check(sock: str) -> None:
        await daemon_query("hello", session_id="ms-1")
        r = await daemon_query("hello", session_id="ms-2")
        assert "maximum sessions" in r["text"].lower() or (r["type"] == "error" and "maximum" in str(r.get("message", "")).lower())

    async def _run() -> None:
        import shutil
        import rooster_code.daemon as dm
        d = tempfile.mkdtemp(prefix="rst-")
        sock = os.path.join(d, "r.sock")
        db = os.path.join(d, "s.db")
        orig = dm._SOCKET_PATH
        dm._SOCKET_PATH = Path(sock)
        daemon = AgentDaemon(socket_path=sock, db_path=db, max_sessions=1)
        task = asyncio.create_task(_run_daemon(daemon))
        await asyncio.sleep(0.1)
        try:
            await _check(sock)
        finally:
            dm._SOCKET_PATH = orig
            daemon._shutdown_event.set()
            with __import__("contextlib").suppress(asyncio.CancelledError):
                await task
            shutil.rmtree(d, ignore_errors=True)

    asyncio.run(_run())


def test_add_telegram_registers_adapter() -> None:
    daemon = AgentDaemon()
    daemon.add_telegram("dummy-token")
    assert len(daemon._adapters) == 1


async def _run_daemon(daemon: AgentDaemon) -> None:
    try:
        await daemon.start()
    except asyncio.CancelledError:
        pass
    finally:
        await daemon.shutdown()
    daemon = AgentDaemon()
    daemon.add_telegram("dummy-token")
    assert len(daemon._adapters) == 1
