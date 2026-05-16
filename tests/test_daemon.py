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
    daemon_session_delete,
    daemon_session_info,
    daemon_session_rename,
    daemon_session_tag,
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


# -- connection failure --------------------------------------------------------


def test_connection_timeout_when_daemon_down() -> None:
    async def _run() -> None:
        import rooster_code.daemon as dm

        orig = dm._SOCKET_PATH
        dm._SOCKET_PATH = Path("/tmp/nonexistent-rooster-test.sock")
        try:
            with __import__("pytest").raises((FileNotFoundError, ConnectionRefusedError, asyncio.TimeoutError)):
                await daemon_health()
        finally:
            dm._SOCKET_PATH = orig

    asyncio.run(_run())


# -- session rename / tag / delete via daemon ----------------------------------


def test_session_rename() -> None:
    async def _check(_sock: str) -> None:
        await daemon_query("hello", session_id="ren-test")
        r = await daemon_session_rename("ren-test", "Renamed Session")
        assert r["type"] == "session_renamed"
        assert r["title"] == "Renamed Session"

    _socket_test(_check)


def test_session_tag() -> None:
    async def _check(_sock: str) -> None:
        await daemon_query("hello", session_id="tag-test")
        r = await daemon_session_tag("tag-test", ["alpha", "beta"])
        assert r["type"] == "session_tagged"
        assert set(r["tags"]) == {"alpha", "beta"}

    _socket_test(_check)


def test_session_delete() -> None:
    async def _check(_sock: str) -> None:
        await daemon_query("hello", session_id="del-test")
        r = await daemon_session_delete("del-test")
        assert r["type"] == "session_deleted"

        s = await daemon_list_sessions()
        ids = {entry["session_id"] for entry in s["data"]}
        assert "del-test" not in ids

    _socket_test(_check)


def test_session_rename_missing_params() -> None:
    _socket_test(lambda _: _check_rename_missing())


async def _check_rename_missing() -> None:
    r = await _send_to_daemon({"action": "session_rename", "session_id": ""})
    assert r["type"] == "error"
    r2 = await _send_to_daemon({"action": "session_rename", "session_id": "x", "title": ""})
    assert r2["type"] == "error"


def test_session_info_merges_sdk_data() -> None:
    async def _check(_sock: str) -> None:
        await daemon_query("hello", session_id="info-test")
        r = await daemon_session_info("info-test")
        assert r["type"] == "session_info"
        assert r["local"] is not None
        assert r["sdk"] is not None

    _socket_test(_check)


# -- concurrent session handling -----------------------------------------------


def test_concurrent_same_session() -> None:
    async def _run() -> None:
        import shutil
        import rooster_code.daemon as dm

        d = tempfile.mkdtemp(prefix="rst-")
        sock = os.path.join(d, "r.sock")
        db = os.path.join(d, "s.db")
        orig = dm._SOCKET_PATH
        dm._SOCKET_PATH = Path(sock)
        daemon = AgentDaemon(socket_path=sock, db_path=db)
        task = asyncio.create_task(_run_daemon(daemon))
        await asyncio.sleep(0.1)
        try:
            results = await asyncio.gather(
                daemon_query("hello", session_id="concurrent"),
                daemon_query("hello", session_id="concurrent"),
            )
            assert all(r["type"] == "done" for r in results)
        finally:
            dm._SOCKET_PATH = orig
            daemon._shutdown_event.set()
            with __import__("contextlib").suppress(asyncio.CancelledError):
                await task
            shutil.rmtree(d, ignore_errors=True)

    asyncio.run(_run())


# -- query timeout -------------------------------------------------------------


def test_query_timeout_configured() -> None:
    import rooster_code.daemon as dm

    assert dm._QUERY_TIMEOUT == 300
    assert dm._CONNECT_TIMEOUT == 5
    assert dm._MAX_RETRIES == 3
    assert dm._RETRY_BASE_DELAY == 1.0
    assert dm._is_retriable(Exception("rate limit exceeded")) is True
    assert dm._is_retriable(Exception("insufficient balance")) is False
    assert dm._is_retriable(Exception("invalid api key")) is False


# -- heartbeat -----------------------------------------------------------------


def test_heartbeat_disabled_by_default() -> None:
    daemon = AgentDaemon()
    assert daemon._heartbeat_interval == 0


def test_heartbeat_enabled_with_interval() -> None:
    daemon = AgentDaemon(heartbeat_interval=60)
    assert daemon._heartbeat_interval == 60


# -- config key constants ------------------------------------------------------


def test_config_keys_exclude_merge_keys() -> None:
    import rooster_code.daemon as dm

    assert "env" not in dm._CONFIG_KEYS
    assert "custom_headers" not in dm._CONFIG_KEYS
    assert "model" in dm._CONFIG_KEYS
    assert "max_turns" in dm._CONFIG_KEYS
    assert "sandbox" in dm._CONFIG_KEYS


def test_config_merge_keys_only_env_and_headers() -> None:
    import rooster_code.daemon as dm

    assert dm._CONFIG_MERGE_KEYS == ("env", "custom_headers")


def test_config_all_keys_is_union() -> None:
    import rooster_code.daemon as dm

    assert dm._CONFIG_ALL_KEYS == dm._CONFIG_KEYS + dm._CONFIG_MERGE_KEYS
    assert "env" in dm._CONFIG_ALL_KEYS
    assert "custom_headers" in dm._CONFIG_ALL_KEYS
    assert "model" in dm._CONFIG_ALL_KEYS


def test_handle_query_extracts_all_keys() -> None:
    """_handle_query uses _CONFIG_ALL_KEYS so env/custom_headers are extracted."""
    import rooster_code.daemon as dm

    all_keys = set(dm._CONFIG_ALL_KEYS)
    simple_keys = set(dm._CONFIG_KEYS)
    merge_keys = set(dm._CONFIG_MERGE_KEYS)
    assert all_keys == simple_keys | merge_keys
    assert merge_keys - simple_keys == {"env", "custom_headers"}


# -- PersistentCronStore -------------------------------------------------------


def _tmp_cron_db() -> str:
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    path = f.name
    f.close()
    return path


def test_cron_store_setitem_persists() -> None:
    from rooster_code.daemon import PersistentCronStore
    path = _tmp_cron_db()
    try:
        store = PersistentCronStore(path)
        store["abc"] = {"id": "abc", "schedule": "*/10 * * * *", "command": "echo hi", "name": "test"}
        assert "abc" in store
        assert store["abc"]["schedule"] == "*/10 * * * *"
        store.close()
        store2 = PersistentCronStore(path)
        assert "abc" in store2
        assert store2["abc"]["name"] == "test"
        store2.close()
    finally:
        Path(path).unlink(missing_ok=True)


def test_cron_store_delitem_removes() -> None:
    from rooster_code.daemon import PersistentCronStore
    path = _tmp_cron_db()
    try:
        store = PersistentCronStore(path)
        store["abc"] = {"id": "abc", "schedule": "* * * * *", "command": "test", "name": "t"}
        del store["abc"]
        assert "abc" not in store
        store.close()
        store2 = PersistentCronStore(path)
        assert "abc" not in store2
        store2.close()
    finally:
        Path(path).unlink(missing_ok=True)


def test_cron_store_clear_empties() -> None:
    from rooster_code.daemon import PersistentCronStore
    path = _tmp_cron_db()
    try:
        store = PersistentCronStore(path)
        store["a"] = {"id": "a", "schedule": "* * * * *", "command": "x", "name": "a"}
        store["b"] = {"id": "b", "schedule": "* * * * *", "command": "y", "name": "b"}
        assert len(store) == 2
        store.clear()
        assert len(store) == 0
        store.close()
        store2 = PersistentCronStore(path)
        assert len(store2) == 0
        store2.close()
    finally:
        Path(path).unlink(missing_ok=True)


def test_cron_store_values_and_iteration() -> None:
    from rooster_code.daemon import PersistentCronStore
    path = _tmp_cron_db()
    try:
        store = PersistentCronStore(path)
        store["x1"] = {"id": "x1", "schedule": "*/5 * * * *", "command": "cmd1", "name": "job1"}
        store["x2"] = {"id": "x2", "schedule": "0 * * * *", "command": "cmd2", "name": "job2"}
        vals = list(store.values())
        assert len(vals) == 2
        names = {v["name"] for v in vals}
        assert names == {"job1", "job2"}
        keys = set(store.keys())
        assert keys == {"x1", "x2"}
        store.close()
    finally:
        Path(path).unlink(missing_ok=True)


def test_cron_store_mark_run() -> None:
    from rooster_code.daemon import PersistentCronStore
    path = _tmp_cron_db()
    try:
        store = PersistentCronStore(path)
        store["r1"] = {"id": "r1", "schedule": "* * * * *", "command": "run", "name": "runner"}
        assert store["r1"].get("last_run_at") is None
        store.mark_run("r1")
        assert store["r1"]["last_run_at"] is not None
        store.close()
        store2 = PersistentCronStore(path)
        assert store2["r1"]["last_run_at"] is not None
        store2.close()
    finally:
        Path(path).unlink(missing_ok=True)


def test_cron_is_due_every_n_minutes() -> None:
    import rooster_code.daemon as dm
    now = time.time()
    assert dm._cron_is_due("*/10 * * * *", now - 601, now) is True
    assert dm._cron_is_due("*/10 * * * *", now - 300, now) is False


def test_cron_is_due_every_n_hours() -> None:
    import rooster_code.daemon as dm
    now = time.time()
    assert dm._cron_is_due("0 */2 * * *", now - 7201, now) is True
    assert dm._cron_is_due("0 */2 * * *", now - 3600, now) is False


def test_cron_is_due_empty_schedule() -> None:
    import rooster_code.daemon as dm
    now = time.time()
    assert dm._cron_is_due("", now - 10000, now) is False
