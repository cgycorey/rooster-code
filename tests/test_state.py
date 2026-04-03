import cock_code.cli as cli
import pytest

from cock_code.runtime import get_state_snapshot


def test_get_state_snapshot_supports_todos() -> None:
    snapshot = get_state_snapshot("todos")

    assert isinstance(snapshot, list)


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
