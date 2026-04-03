import cock_code.cli as cli

from cock_code.rendering import session_row_count


def test_session_row_count_matches_sessions() -> None:
    assert session_row_count([{"id": "a"}, {"id": "b"}]) == 2


def test_main_dispatches_sessions_list(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def fake_list_sessions() -> list[dict[str, object]]:
        return [{"id": "sess-1"}]

    def fake_render_session_table(console, sessions: list[dict[str, object]]) -> None:
        captured["count"] = len(sessions)

    monkeypatch.setattr(cli, "list_sessions", fake_list_sessions)
    monkeypatch.setattr(cli, "build_console", lambda: object())
    monkeypatch.setattr(cli, "render_session_table", fake_render_session_table)

    exit_code = cli.main(["sessions", "list"])

    assert exit_code == 0
    assert captured == {"count": 1}


def test_main_dispatches_sessions_show(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def fake_get_session_messages(session_id: str) -> list[dict[str, object]]:
        captured["session_id"] = session_id
        return [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]

    def fake_render_transcript(console, messages: list[dict[str, object]]) -> None:
        captured["message_count"] = len(messages)

    monkeypatch.setattr(cli, "get_session_messages", fake_get_session_messages)
    monkeypatch.setattr(cli, "build_console", lambda: object())
    monkeypatch.setattr(cli, "render_transcript", fake_render_transcript)

    exit_code = cli.main(["sessions", "show", "sess-1"])

    assert exit_code == 0
    assert captured == {"session_id": "sess-1", "message_count": 1}


def test_main_dispatches_sessions_info(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def fake_get_session_info(session_id: str) -> dict[str, object]:
        captured["session_id"] = session_id
        return {"id": session_id, "messageCount": 3}

    def fake_render_session_info(console, data: dict[str, object]) -> None:
        captured["message_count"] = data["messageCount"]

    monkeypatch.setattr(cli, "get_session_info", fake_get_session_info)
    monkeypatch.setattr(cli, "build_console", lambda: object())
    monkeypatch.setattr(cli, "render_session_info", fake_render_session_info)

    exit_code = cli.main(["sessions", "info", "sess-2"])

    assert exit_code == 0
    assert captured == {"session_id": "sess-2", "message_count": 3}


def test_main_dispatches_sessions_delete(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def fake_delete_session(session_id: str) -> bool:
        captured["session_id"] = session_id
        return True

    def fake_render_state(console, title: str, data: object) -> None:
        captured["render_title"] = title
        captured["result"] = data

    monkeypatch.setattr(cli, "delete_session", fake_delete_session)
    monkeypatch.setattr(cli, "build_console", lambda: object())
    monkeypatch.setattr(cli, "render_state", fake_render_state)

    exit_code = cli.main(["sessions", "delete", "sess-3"])

    assert exit_code == 0
    assert captured == {
        "session_id": "sess-3",
        "render_title": "Session Deleted",
        "result": {"deleted": True, "session_id": "sess-3"},
    }


def test_main_dispatches_sessions_fork(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def fake_fork_session(session_id: str, new_id: str | None) -> str:
        captured["session_id"] = session_id
        captured["new_id"] = new_id
        return "forked-1"

    def fake_render_state(console, title: str, data: object) -> None:
        captured["render_title"] = title
        captured["result"] = data

    monkeypatch.setattr(cli, "fork_session", fake_fork_session)
    monkeypatch.setattr(cli, "build_console", lambda: object())
    monkeypatch.setattr(cli, "render_state", fake_render_state)

    exit_code = cli.main(["sessions", "fork", "sess-4", "--new-id", "forked-1"])

    assert exit_code == 0
    assert captured == {
        "session_id": "sess-4",
        "new_id": "forked-1",
        "render_title": "Session Forked",
        "result": {"forked_from": "sess-4", "new_session_id": "forked-1"},
    }


def test_main_dispatches_sessions_rename(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def fake_rename_session(session_id: str, title: str) -> None:
        captured["session_id"] = session_id
        captured["title"] = title

    def fake_render_state(console, title: str, data: object) -> None:
        captured["render_title"] = title
        captured["result"] = data

    monkeypatch.setattr(cli, "rename_session", fake_rename_session)
    monkeypatch.setattr(cli, "build_console", lambda: object())
    monkeypatch.setattr(cli, "render_state", fake_render_state)

    exit_code = cli.main(["sessions", "rename", "sess-5", "My session"])

    assert exit_code == 0
    assert captured == {
        "session_id": "sess-5",
        "title": "My session",
        "render_title": "Session Renamed",
        "result": {"renamed": True, "session_id": "sess-5", "title": "My session"},
    }


def test_main_dispatches_sessions_tag(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def fake_tag_session(session_id: str, tags: list[str]) -> None:
        captured["session_id"] = session_id
        captured["tags"] = tags

    def fake_render_state(console, title: str, data: object) -> None:
        captured["title"] = title
        captured["result"] = data

    monkeypatch.setattr(cli, "tag_session", fake_tag_session)
    monkeypatch.setattr(cli, "build_console", lambda: object())
    monkeypatch.setattr(cli, "render_state", fake_render_state)

    exit_code = cli.main(["sessions", "tag", "sess-6", "alpha", "beta"])

    assert exit_code == 0
    assert captured == {
        "session_id": "sess-6",
        "tags": ["alpha", "beta"],
        "title": "Session Tagged",
        "result": {"tagged": True, "session_id": "sess-6", "tags": ["alpha", "beta"]},
    }
