import asyncio
import json
from pathlib import Path

from open_agent_sdk import BaseTool, ToolContext, ToolResult

from rooster_code.runtime_tools import RuntimeEditTool, RuntimeReadTool, RuntimeSkillTool, TurnTracker


class FakeReadTool(BaseTool):
    name = "Read"

    async def call(self, input, context):
        return ToolResult(tool_use_id="", content="1\tline")


class FakeEditTool(BaseTool):
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


def test_runtime_read_tool_records_activity_trace(tmp_path) -> None:
    async def run_test() -> None:
        tracker = TurnTracker()
        read_tool = RuntimeReadTool(FakeReadTool(), tracker)
        target = tmp_path / "sample.txt"
        target.write_text("old\n", encoding="utf-8")

        await read_tool.call({"file_path": str(target)}, ToolContext(cwd=str(tmp_path), env={}))

        assert tracker.consume_activity_trace() == [
            {"action": "Reading file", "tool": "Read", "target": str(target)}
        ]

    asyncio.run(run_test())


def test_runtime_read_tool_emits_activity_before_read_finishes(tmp_path) -> None:
    class SlowReadTool(BaseTool):
        name = "Read"

        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.finish = asyncio.Event()

        async def call(self, input, context):
            self.started.set()
            await self.finish.wait()
            return ToolResult(tool_use_id="", content="1\tline")

    async def run_test() -> None:
        tracker = TurnTracker()
        delegate = SlowReadTool()
        read_tool = RuntimeReadTool(delegate, tracker)
        target = tmp_path / "sample.txt"
        target.write_text("old\n", encoding="utf-8")

        task = asyncio.create_task(read_tool.call({"file_path": str(target)}, ToolContext(cwd=str(tmp_path), env={})))
        await delegate.started.wait()

        activity = await asyncio.wait_for(tracker.next_activity(), timeout=0.1)
        assert activity == {"action": "Reading file", "tool": "Read", "target": str(target)}

        delegate.finish.set()
        await task

    asyncio.run(run_test())


class FakeSkillDelegate(BaseTool):
    _name = "Skill"

    def __init__(self, response: ToolResult):
        self._response = response

    @property
    def name(self):
        return self._name

    async def call(self, input, context):
        return self._response


class FakeConfig:
    def __init__(self):
        self.model = "test-model"
        self.allowed_tools = None
        self.cwd = "/tmp"
        self.env = {}
        self.persist_session = False
        self.api_key = ""
        self.base_url = ""
        self.api_type = ""
        self.disallowed_tools = None
        self.session_id = ""
        self.resume = ""
        self.continue_session = None
        self.fork_session = ""
        self.max_turns = None
        self.max_budget_usd = None
        self.max_tokens = None
        self.thinking_budget = None
        self.debug = False
        self.sandbox = None
        self.include_partials = False
        self.hooks = None
        self.mcp_servers = None
        self.extra_args = None
        self.custom_headers = None
        self.json_schema = None
        self.agents = None
        self.skills_dir = None
        self.permission_mode = "default"


def test_runtime_skill_tool_returns_prompt_for_inline_skills() -> None:
    async def run_test():
        tracker = TurnTracker()
        payload = json.dumps({
            "success": True,
            "commandName": "plan",
            "status": "inline",
            "prompt": "Create a detailed plan for the user's request.",
        })
        delegate = FakeSkillDelegate(ToolResult(tool_use_id="", content=payload))
        tool = RuntimeSkillTool(delegate, FakeConfig(), tracker)

        result = await tool.call({"skill": "plan", "args": "add auth"}, ToolContext(cwd="/tmp", env={}))

        assert result.is_error is False
        assert "Skill \"plan\" activated" in str(result.content)
        assert "Create a detailed plan" in str(result.content)
        assert tracker.consume_activity_trace() == [
            {"action": "Running skill", "tool": "Skill", "target": "plan"}
        ]

    asyncio.run(run_test())


def test_runtime_skill_tool_passes_through_errors() -> None:
    async def run_test():
        tracker = TurnTracker()
        delegate = FakeSkillDelegate(ToolResult(tool_use_id="", content='Error: unknown skill "missing"', is_error=True))
        tool = RuntimeSkillTool(delegate, FakeConfig(), tracker)

        result = await tool.call({"skill": "missing"}, ToolContext(cwd="/tmp", env={}))

        assert result.is_error is True
        assert "unknown skill" in str(result.content)

    asyncio.run(run_test())


def test_runtime_skill_tool_handles_no_prompt_gracefully() -> None:
    async def run_test():
        tracker = TurnTracker()
        payload = json.dumps({"success": True, "commandName": "plan", "status": "inline", "prompt": ""})
        delegate = FakeSkillDelegate(ToolResult(tool_use_id="", content=payload))
        tool = RuntimeSkillTool(delegate, FakeConfig(), tracker)

        result = await tool.call({"skill": "plan"}, ToolContext(cwd="/tmp", env={}))

        assert result.is_error is False
        assert "no instructions" in str(result.content).lower() or "Skill" in str(result.content)

    asyncio.run(run_test())
