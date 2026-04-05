import asyncio

from open_agent_sdk import ConversationMessage, MessageRole, SDKMessage, SDKMessageType
from rich.console import Console

from cock_code.config import RuntimeConfig
from cock_code.rendering import (
    render_banner,
    render_event_stream,
    render_session_info,
    render_session_table,
    render_state,
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

    assert "COCK-CODE CHAT" in output
    assert "claude-sonnet-4-5" in output
    assert "/tmp/project" in output
    assert "sess-1" in output


def test_render_banner_shows_cock_code_band_and_ascii_rooster() -> None:
    console = Console(record=True, width=100)

    render_banner(console, "ask", RuntimeConfig(model="m"))

    output = console.export_text()

    assert "COCK" in output
    assert "CODE" in output
    assert "◉" in output
    assert "▶▶" in output or "▶" in output
    assert "▄▄" in output


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
