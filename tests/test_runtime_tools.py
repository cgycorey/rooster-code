import asyncio
from pathlib import Path

from open_agent_sdk import ToolContext, ToolResult

from cock_code.runtime_tools import RuntimeEditTool, RuntimeReadTool, TurnTracker


class FakeReadTool:
    name = "Read"

    async def call(self, input, context):
        return ToolResult(tool_use_id="", content="1\tline")


class FakeEditTool:
    name = "Edit"

    async def call(self, input, context):
        path = Path(input["file_path"])
        path.write_text("new\n", encoding="utf-8")
        return ToolResult(tool_use_id="", content=f"Successfully edited {path}")


def test_runtime_edit_tool_blocks_unread_file(tmp_path) -> None:
    async def run_test() -> None:
        tracker = TurnTracker()
        tool = RuntimeEditTool(FakeEditTool(), tracker)
        target = tmp_path / "sample.txt"
        target.write_text("old\n", encoding="utf-8")

        result = await tool.call(
            {"file_path": str(target), "old_string": "old", "new_string": "new"},
            ToolContext(cwd=str(tmp_path), env={}),
        )

        assert result.is_error is True
        assert f"read {target} first in this turn" in str(result.content)

    asyncio.run(run_test())


def test_runtime_read_then_edit_returns_diff(tmp_path) -> None:
    async def run_test() -> None:
        tracker = TurnTracker()
        read_tool = RuntimeReadTool(FakeReadTool(), tracker)
        edit_tool = RuntimeEditTool(FakeEditTool(), tracker)
        target = tmp_path / "sample.txt"
        target.write_text("old\n", encoding="utf-8")

        await read_tool.call({"file_path": str(target)}, ToolContext(cwd=str(tmp_path), env={}))
        result = await edit_tool.call(
            {"file_path": str(target), "old_string": "old", "new_string": "new"},
            ToolContext(cwd=str(tmp_path), env={}),
        )

        text = str(result.content)
        assert result.is_error is False
        assert "Successfully edited" in text
        assert f"--- {target}" in text
        assert "+new" in text

    asyncio.run(run_test())
