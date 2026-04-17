import asyncio

from open_agent_sdk import ConversationMessage, MessageRole, SDKMessage, SDKMessageType
from rich.console import Console

from rooster_code.config import RuntimeConfig
from rooster_code.rendering import (
    compact_tool_result,
    render_agents_list,
    render_banner,
    render_event_stream,
    render_help,
    render_session_info,
    render_session_table,
    render_state,
    render_team_info,
    render_tool_table,
    render_transcript,
    summarize_tool_result,
)


def test_render_event_stream_labels_assistant_output() -> None:
    console = Console(record=True, width=100)

    async def events():
        yield SDKMessage(type=SDKMessageType.ASSISTANT, text="hello")

    asyncio.run(render_event_stream(console, events(), show_activity_trace=True))

    output = console.export_text()

    assert "Assistant" in output
    assert "hello" in output


def test_render_event_stream_omits_duplicate_result_text_when_requested() -> None:
    console = Console(record=True, width=100)

    async def events():
        yield SDKMessage(type=SDKMessageType.ASSISTANT, text="same")
        yield SDKMessage(type=SDKMessageType.RESULT, text="same")

    asyncio.run(render_event_stream(console, events(), omit_duplicate_result=True))

    output = console.export_text()

    assert "Assistant" in output
    assert "same" in output
    assert "Result" not in output


def test_render_event_stream_skips_empty_assistant_panel() -> None:
    console = Console(record=True, width=100)

    async def events():
        yield SDKMessage(type=SDKMessageType.ASSISTANT, text="")
        yield SDKMessage(type=SDKMessageType.RESULT, text="done")

    asyncio.run(render_event_stream(console, events(), show_activity_trace=True))

    output = console.export_text()

    assert "No content" not in output
    assert "Assistant" not in output
    assert "done" in output


def test_render_event_stream_shows_short_thinking_panel() -> None:
    console = Console(record=True, width=100)

    async def events():
        yield SDKMessage(
            type=SDKMessageType.ASSISTANT,
            text="answer",
            message=ConversationMessage(
                role=MessageRole.ASSISTANT,
                content=[
                    {"type": "thinking", "thinking": "thinking line one\nthinking line two\nthinking line three"},
                    {"type": "text", "text": "answer"},
                ],
            ),
        )

    asyncio.run(render_event_stream(console, events(), show_activity_trace=True))

    output = console.export_text()

    assert "Thinking" in output
    assert "thinking line one" in output
    assert "answer" in output


def test_render_event_stream_shows_edit_diff_panel() -> None:
    console = Console(record=True, width=100)

    async def events():
        yield SDKMessage(
            type=SDKMessageType.TOOL_RESULT,
            tool_name="Edit",
            result_content="Successfully edited /tmp/a.py\n--- /tmp/a.py\n+++ /tmp/a.py\n@@ -1 +1 @@\n-old\n+new",
        )

    asyncio.run(render_event_stream(console, events()))

    output = console.export_text()

    assert "Edit Diff" in output
    assert "--- /tmp/a.py" in output
    assert "+new" in output


def test_render_event_stream_shows_tool_error_notice() -> None:
    console = Console(record=True, width=100)

    async def events():
        yield SDKMessage(
            type=SDKMessageType.TOOL_RESULT,
            tool_name="Edit",
            result_content="Edit blocked: read /tmp/a.py first in this turn, then retry.",
            is_error=True,
        )

    asyncio.run(render_event_stream(console, events()))

    output = console.export_text()

    assert "Edit" in output
    assert "read /tmp/a.py first" in output


def test_render_event_stream_shows_agent_result_panel() -> None:
    console = Console(record=True, width=100)

    async def events():
        yield SDKMessage(
            type=SDKMessageType.TOOL_RESULT,
            tool_name="Agent",
            result_content="AGENT_PATH=used",
        )

    asyncio.run(render_event_stream(console, events()))

    output = console.export_text()

    assert "Agent Result" in output
    assert "AGENT_PATH=used" in output


def test_render_event_stream_shows_full_agent_result() -> None:
    console = Console(record=True, width=100)

    async def events():
        yield SDKMessage(
            type=SDKMessageType.TOOL_RESULT,
            tool_name="Agent",
            result_content="Outcome: All tests pass. Let me analyze the code changes. Found 2 bugs.",
        )

    asyncio.run(render_event_stream(console, events()))

    output = console.export_text()

    assert "Agent Result" in output
    assert "All tests pass" in output
    assert "Found 2 bugs" in output


def test_render_event_stream_compacts_team_tool_result() -> None:
    console = Console(record=True, width=100)

    async def events():
        yield SDKMessage(
            type=SDKMessageType.TOOL_RESULT,
            tool_name="TeamDispatch",
            result_content="Outcome: Here is the full team dispatch report. Let me now continue with more analysis. Dispatch completed.",
        )

    asyncio.run(render_event_stream(console, events()))

    output = console.export_text()

    assert "TeamDispatch" in output
    assert "full team dispatch report" in output
    assert "Dispatch" in output
    assert "completed" in output


def test_compact_tool_result_strips_ansi_sequences() -> None:
    result = compact_tool_result("\x1b[32mOutcome: done\x1b[0m")

    assert result == "done"


def test_render_event_stream_shows_activity_trace_for_tool_result() -> None:
    console = Console(record=True, width=100)

    async def events():
        yield SDKMessage(
            type=SDKMessageType.TOOL_RESULT,
            tool_name="Read",
            result_content="hello",
            system_data={
                "activity_trace": [
                    {"action": "Reading file", "tool": "Read", "target": "/tmp/a.py"}
                ]
            },
        )

    asyncio.run(render_event_stream(console, events(), show_activity_trace=True))

    output = console.export_text()

    assert "Activity: Reading file · Read · /tmp/a.py" in output
    assert "╭" not in output


def test_render_banner_shows_mode_and_runtime_context() -> None:
    console = Console(record=True, width=100)
    config = RuntimeConfig(model="claude-sonnet-4-5", cwd="/tmp/project", resume="sess-1")

    render_banner(console, "chat", config)

    output = console.export_text()

    assert "ROOSTER CODE CHAT" in output
    assert "claude-sonnet-4-5" in output
    assert "/tmp/project" in output
    assert "sess-1" in output


def test_render_banner_shows_rooster_code_band_and_ascii_rooster() -> None:
    console = Console(record=True, width=100)

    render_banner(console, "ask", RuntimeConfig(model="m"))

    output = console.export_text()

    assert "ROOSTER CODE ASK" in output
    assert "◉" in output
    assert "▶▶" in output or "▶" in output
    assert "▄▄" in output
    assert "██████   ██████   ██████" in output


def test_summarize_tool_result_truncates_output() -> None:
    result = summarize_tool_result("x" * 200, max_chars=32)

    assert len(result) <= 35
    assert result.endswith("...")


def test_render_session_table_shows_metadata_columns() -> None:
    console = Console(record=True, width=100)

    render_session_table(
        console,
        [{"id": "sess-1", "title": "Review", "message_count": 3}],
    )

    output = console.export_text()

    assert "Title" in output
    assert "Messages" in output
    assert "Review" in output
    assert "3" in output


def test_render_session_table_handles_empty_state() -> None:
    console = Console(record=True, width=100)

    render_session_table(console, [])

    output = console.export_text()

    assert "No sessions found" in output


def test_render_transcript_formats_roles_and_text() -> None:
    console = Console(record=True, width=100)

    render_transcript(
        console,
        [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
        ],
    )

    output = console.export_text()

    assert "USER" in output
    assert "ASSISTANT" in output
    assert "hi" in output
    assert "hello" in output


def test_render_tool_table_shows_named_tool_rows() -> None:
    console = Console(record=True, width=100)

    render_tool_table(console, ["Read", "Write"])

    output = console.export_text()

    assert "Tools" in output
    assert "Tool" in output
    assert "Read" in output
    assert "Write" in output


def test_render_session_info_formats_friendly_labels() -> None:
    console = Console(record=True, width=100)

    render_session_info(
        console,
        {
            "id": "sess-123",
            "updatedAt": "2026-04-03T10:33:54",
            "messageCount": 3,
            "cwd": "/tmp/project",
            "model": "claude-sonnet-4-5",
        },
    )

    output = console.export_text()

    assert "Session sess-123" in output
    assert "Updated" in output
    assert "Messages" in output
    assert "CWD" in output
    assert "Model" in output
    assert "updatedAt" not in output
    assert "messageCount" not in output


def test_render_state_formats_mapping_as_key_value_table() -> None:
    console = Console(record=True, width=100)

    render_state(console, "Config", {"model": "claude", "cwd": "/tmp/project"})

    output = console.export_text()

    assert "Config" in output
    assert "model" in output
    assert "claude" in output
    assert "cwd" in output
    assert "/tmp/project" in output


def test_render_state_formats_list_of_mappings_as_table() -> None:
    console = Console(record=True, width=100)

    render_state(
        console,
        "Todos",
        [
            {"content": "Ship CLI", "status": "done"},
            {"content": "Polish TUI", "status": "pending"},
        ],
    )

    output = console.export_text()

    assert "Todos" in output
    assert "content" in output
    assert "status" in output
    assert "Ship CLI" in output
    assert "Polish TUI" in output


def test_render_state_formats_empty_list_as_named_empty_state() -> None:
    console = Console(record=True, width=100)

    render_state(console, "Tasks", [])

    output = console.export_text()

    assert "Tasks" in output
    assert "No tasks found" in output


def test_render_state_formats_empty_mapping_as_named_empty_state() -> None:
    console = Console(record=True, width=100)

    render_state(console, "Tasks", {})

    output = console.export_text()

    assert "Tasks" in output
    assert "No tasks found" in output


def test_render_state_list_table_shows_row_count_in_title() -> None:
    console = Console(record=True, width=100)

    render_state(console, "Todos", [{"content": "Ship CLI"}, {"content": "Polish TUI"}])

    output = console.export_text()

    assert "Todos (2)" in output


def test_render_state_formats_list_of_empty_mappings_as_named_empty_state() -> None:
    console = Console(record=True, width=100)

    render_state(console, "Tasks", [{}])

    output = console.export_text()

    assert "Tasks" in output
    assert "No tasks found" in output


def test_render_agents_list_shows_agent_names_and_descriptions() -> None:
    console = Console(record=True, width=100)

    render_agents_list(console, {"reviewer": "Reviews code", "writer": "Writes code"})

    output = console.export_text()

    assert "reviewer" in output
    assert "Reviews code" in output
    assert "writer" in output
    assert "Writes code" in output


def test_render_agents_list_uses_system_prompt_when_description_missing() -> None:
    console = Console(record=True, width=100)

    render_agents_list(console, {"reviewer": {"system_prompt": "Reviews code carefully"}})

    output = console.export_text()

    assert "reviewer" in output
    assert "Reviews code carefully" in output


def test_render_agents_list_handles_empty_dict() -> None:
    console = Console(record=True, width=100)

    render_agents_list(console, {})

    output = console.export_text()

    assert "No agents configured" in output


def test_render_team_info_formats_team_status() -> None:
    console = Console(record=True, width=100)

    render_team_info(console, {"status": "active", "name": "alpha", "members": [{"name": "reviewer", "status": "running"}]})

    output = console.export_text()

    assert "Team" in output
    assert "active" in output


def test_render_help_includes_agents_and_team_commands() -> None:
    console = Console(record=True, width=100)

    render_help(console)

    output = console.export_text()

    assert "/agents" in output
    assert "List configured agents" in output
    assert "/agents add" in output
    assert "Add an agent definition" in output
    assert "/agents remove" in output
    assert "Remove an agent definition" in output
    assert "/agents show" in output
    assert "Show agent definition details" in output
    assert "/team create" in output
    assert "Create a team with named agents" in output
    assert "/team info" in output
    assert "Show team members and status" in output
    assert "/team stop" in output
    assert "Disband team" in output
