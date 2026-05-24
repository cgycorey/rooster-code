import asyncio
from pathlib import Path

import rooster_code.cli as cli
import pytest
from open_agent_sdk.tools import clear_mailboxes, clear_teams, write_to_mailbox

from rooster_code.daemon import PersistentCronStore
from rooster_code.runtime import get_state_snapshot
from rooster_code.runtime_session import _read_cron_jobs
from rooster_code.team import AgentPool, TeamManager, set_runtime_team_bridge


def test_get_state_snapshot_supports_todos() -> None:
    snapshot = get_state_snapshot("todos")

    assert isinstance(snapshot, list)


def test_get_state_snapshot_merges_runtime_team_state() -> None:
    manager = TeamManager()
    manager._active = True
    manager._team_id = "runtime123"
    manager._team_name = "dev-team"
    manager._member_definitions = {
        "reviewer": {"description": "reviews"},
        "builder": {"description": "builds"},
    }
    pool = AgentPool()
    pool._mailboxes["reviewer"] = asyncio.Queue()
    pool._mailboxes["builder"] = asyncio.Queue()
    pool.send_message("reviewer", {"from": "builder", "content": "check this first"})
    manager._pool = pool

    clear_teams()
    clear_mailboxes()
    write_to_mailbox("sdk-agent", {"type": "text", "from": "agent", "content": "sdk only"})
    set_runtime_team_bridge(manager, object())
    try:
        teams = get_state_snapshot("teams")
        mailboxes = get_state_snapshot("mailboxes")
        reviewer_mailbox = get_state_snapshot("mailboxes", "reviewer")
        builder_mailbox = get_state_snapshot("mailboxes", "builder")
    finally:
        set_runtime_team_bridge(None, None)
        clear_teams()
        clear_mailboxes()

    assert teams["runtime123"] == {
        "id": "runtime123",
        "name": "dev-team",
        "description": "",
        "members": ["reviewer", "builder"],
        "member_statuses": {"reviewer": "idle", "builder": "idle"},
        "runtime_managed": True,
    }
    assert mailboxes["sdk-agent"] == [{"type": "text", "from": "agent", "content": "sdk only"}]
    assert reviewer_mailbox == [{"type": "text", "from": "builder", "content": "check this first"}]
    assert builder_mailbox == []


def test_read_cron_jobs_reads_daemon_sqlite_store(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = tmp_path / ".rooster-code" / "daemon.db"
    db_path.parent.mkdir()
    store = PersistentCronStore(str(db_path))
    try:
        store["cron-1"] = {
            "id": "cron-1",
            "schedule": "*/5 * * * *",
            "command": "say hi",
            "name": "hello",
        }

        jobs = _read_cron_jobs()

        assert jobs["cron-1"]["job_id"] == "cron-1"
        assert jobs["cron-1"]["schedule"] == "*/5 * * * *"
        assert jobs["cron-1"]["command"] == "say hi"
        assert jobs["cron-1"]["name"] == "hello"
    finally:
        store.close()


def test_get_state_snapshot_cron_uses_daemon_sqlite_store(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = tmp_path / ".rooster-code" / "daemon.db"
    db_path.parent.mkdir()
    store = PersistentCronStore(str(db_path))
    try:
        store["cron-2"] = {
            "id": "cron-2",
            "schedule": "*/10 * * * *",
            "command": "check status",
            "name": "status",
        }

        snapshot = get_state_snapshot("cron")

        assert snapshot["cron-2"]["job_id"] == "cron-2"
        assert snapshot["cron-2"]["name"] == "status"
    finally:
        store.close()


def test_main_dispatches_state_todos(monkeypatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        cli,
        "get_state_snapshot",
        lambda name, agent_name=None: [{"content": "todo"}] if name == "todos" else [],
    )
    monkeypatch.setattr(cli, "build_console", lambda: object())
    monkeypatch.setattr(
        cli,
        "render_state",
        lambda console, title, data: captured.update({"title": title, "data": data}),
    )

    exit_code = cli.main(["state", "todos"])

    assert exit_code == 0
    assert captured == {"title": "Todos", "data": [{"content": "todo"}]}


def test_main_dispatches_tools_list(monkeypatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(cli, "list_tool_names", lambda: ["Read", "Write"])
    monkeypatch.setattr(cli, "build_console", lambda: object())
    monkeypatch.setattr(cli, "render_tool_table", lambda console, tools: captured.setdefault("tools", tools))

    exit_code = cli.main(["tools", "list"])

    assert exit_code == 0
    assert captured == {"tools": ["Read", "Write"]}


def test_main_dispatches_cron_list(monkeypatch) -> None:
    import rooster_code.runtime_session as runtime_session

    captured: dict[str, object] = {}
    monkeypatch.setattr(runtime_session, "_read_cron_jobs", lambda: {"cron-1": {"job_id": "cron-1", "name": "daily"}})
    monkeypatch.setattr(cli, "build_console", lambda: object())
    monkeypatch.setattr(
        cli,
        "render_state",
        lambda console, title, data: captured.update({"title": title, "data": data}),
    )

    exit_code = cli.main(["cron", "list"])

    assert exit_code == 0
    assert captured == {"title": "Cron Jobs", "data": [{"job_id": "cron-1", "name": "daily"}]}


def test_main_dispatches_cron_show(monkeypatch) -> None:
    import rooster_code.runtime_session as runtime_session

    captured: dict[str, object] = {}
    monkeypatch.setattr(runtime_session, "_read_cron_jobs", lambda: {"cron-1": {"job_id": "cron-1", "name": "daily"}})
    monkeypatch.setattr(cli, "build_console", lambda: object())
    monkeypatch.setattr(
        cli,
        "render_state",
        lambda console, title, data: captured.update({"title": title, "data": data}),
    )

    exit_code = cli.main(["cron", "show", "cron-1"])

    assert exit_code == 0
    assert captured == {"title": "Cron Job cron-1", "data": {"job_id": "cron-1", "name": "daily"}}


def test_main_cron_show_missing_returns_1(monkeypatch) -> None:
    import rooster_code.runtime_session as runtime_session

    messages: list[str] = []
    monkeypatch.setattr(runtime_session, "_read_cron_jobs", lambda: {})
    monkeypatch.setattr(cli, "build_console", lambda: type("Console", (), {"print": lambda self, msg: messages.append(str(msg))})())

    exit_code = cli.main(["cron", "show", "missing"])

    assert exit_code == 1
    assert "missing" in messages[0]


def test_main_dispatches_cron_delete(monkeypatch) -> None:
    import rooster_code.daemon as daemon

    messages: list[str] = []

    async def fake_delete(job_id: str) -> dict[str, object]:
        return {"type": "cron_deleted", "job_id": job_id}

    monkeypatch.setattr(daemon, "daemon_cron_delete", fake_delete)
    monkeypatch.setattr(cli, "build_console", lambda: type("Console", (), {"print": lambda self, msg: messages.append(str(msg))})())

    exit_code = cli.main(["cron", "delete", "cron-1"])

    assert exit_code == 0
    assert "cron-1" in messages[0]


def test_main_cron_delete_daemon_error_returns_1(monkeypatch) -> None:
    import rooster_code.daemon as daemon

    messages: list[str] = []

    async def fake_delete(job_id: str) -> dict[str, object]:
        return {"type": "error", "message": f"cron job '{job_id}' not found"}

    monkeypatch.setattr(daemon, "daemon_cron_delete", fake_delete)
    monkeypatch.setattr(cli, "build_console", lambda: type("Console", (), {"print": lambda self, msg: messages.append(str(msg))})())

    exit_code = cli.main(["cron", "delete", "missing"])

    assert exit_code == 1
    assert "not found" in messages[0]


def test_main_dispatches_mailboxes_with_agent_filter(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_get_state_snapshot(name: str, agent_name: str | None = None):
        captured["state_name"] = name
        captured["agent_name"] = agent_name
        return {"name": name, "agent_name": agent_name}

    monkeypatch.setattr(cli, "get_state_snapshot", fake_get_state_snapshot)
    monkeypatch.setattr(cli, "build_console", lambda: object())
    monkeypatch.setattr(
        cli,
        "render_state",
        lambda console, title, data: captured.update({"title": title, "data": data}),
    )

    exit_code = cli.main(["state", "mailboxes", "--agent", "reviewer"])

    assert exit_code == 0
    assert captured == {
        "state_name": "mailboxes",
        "agent_name": "reviewer",
        "title": "Mailboxes",
        "data": {"name": "mailboxes", "agent_name": "reviewer"},
    }


@pytest.mark.parametrize(
    ("argv", "state_name"),
    [
        (["state", "tasks"], "tasks"),
        (["state", "teams"], "teams"),
        (["state", "mailboxes"], "mailboxes"),
        (["state", "config"], "config"),
        (["state", "cron"], "cron"),
        (["state", "plan"], "plan"),
    ],
)
def test_main_dispatches_supported_state_commands(monkeypatch, argv: list[str], state_name: str) -> None:
    captured: dict[str, object] = {}

    def fake_get_state_snapshot(name: str, agent_name: str | None = None):
        captured["state_name"] = name
        return {"name": name}

    monkeypatch.setattr(cli, "get_state_snapshot", fake_get_state_snapshot)
    monkeypatch.setattr(cli, "build_console", lambda: object())
    monkeypatch.setattr(
        cli,
        "render_state",
        lambda console, title, data: captured.update({"title": title, "data": data}),
    )

    exit_code = cli.main(argv)

    assert exit_code == 0
    assert captured == {"state_name": state_name, "title": state_name.title(), "data": {"name": state_name}}
