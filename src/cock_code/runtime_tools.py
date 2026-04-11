from __future__ import annotations

from dataclasses import dataclass, field
from difflib import unified_diff
import os
import asyncio
from typing import Any, Awaitable, Callable
from asyncio import QueueEmpty

from open_agent_sdk.types import BaseTool, ToolContext, ToolInputSchema, ToolResult


AgentRunner = Callable[[dict[str, Any], ToolContext], Awaitable[ToolResult]]


@dataclass
class TurnTracker:
    read_paths: set[str] = field(default_factory=set)
    activity_trace: list[dict[str, str]] = field(default_factory=list)
    activity_queue: asyncio.Queue[dict[str, str]] = field(default_factory=asyncio.Queue)

    def reset(self) -> None:
        self.read_paths.clear()
        self.activity_trace.clear()
        while not self.activity_queue.empty():
            self.activity_queue.get_nowait()

    def mark_read(self, file_path: str) -> None:
        self.read_paths.add(file_path)

    def record_activity(self, action: str, tool: str, target: str) -> None:
        entry = {"action": action, "tool": tool, "target": target}
        self.activity_trace.append(entry)
        self.activity_queue.put_nowait(entry)

    def _discard_recorded_activity(self, entry: dict[str, str]) -> None:
        for index, recorded in enumerate(self.activity_trace):
            if recorded == entry:
                del self.activity_trace[index]
                break

    def consume_activity_trace(self) -> list[dict[str, str]]:
        activities = list(self.activity_trace)
        self.activity_trace.clear()
        return activities

    async def next_activity(self) -> dict[str, str]:
        activity = await self.activity_queue.get()
        self._discard_recorded_activity(activity)
        return activity

    def consume_pending_activities(self) -> list[dict[str, str]]:
        pending: list[dict[str, str]] = []
        while True:
            try:
                activity = self.activity_queue.get_nowait()
            except QueueEmpty:
                break
            self._discard_recorded_activity(activity)
            pending.append(activity)
        return pending

    def has_read(self, file_path: str) -> bool:
        return file_path in self.read_paths


def resolve_tool_path(file_path: str, context: ToolContext) -> str:
    if os.path.isabs(file_path):
        return file_path
    return os.path.join(context.cwd, file_path)


class RuntimeAgentTool(BaseTool):
    _name = "Agent"
    _description = (
        "Launch a new agent to handle complex, multi-step tasks autonomously. "
        "Subagents can run in the background or foreground. Once you delegate work to another agent, do not also perform that same work yourself unless the delegated task fails, is cancelled, or you are explicitly asked to compare or verify it."
    )
    _input_schema = ToolInputSchema(
        properties={
            "prompt": {"type": "string", "description": "The task for the agent to perform"},
            "description": {"type": "string", "description": "A short (3-5 word) description of the task"},
            "subagent_type": {
                "type": "string",
                "description": "Type of specialized agent (e.g., 'Explore', 'Plan')",
            },
            "model": {"type": "string", "description": "Optional model override for this agent"},
            "name": {"type": "string", "description": "Name for the spawned agent"},
            "run_in_background": {
                "type": "boolean",
                "description": "Set to true to run this agent in the background",
            },
        },
        required=["prompt", "description"],
    )

    def __init__(self, runner: AgentRunner, tracker: TurnTracker):
        self._runner = runner
        self._tracker = tracker

    def is_read_only(self, input: dict[str, Any] | None = None) -> bool:
        return True

    def is_concurrency_safe(self, input: dict[str, Any] | None = None) -> bool:
        return True

    async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        target = str(input.get("name") or input.get("subagent_type") or input.get("description") or "agent")
        self._tracker.record_activity("Using agent", self.name, target)
        return await self._runner(input, context)


class RuntimeReadTool(BaseTool):
    def __init__(self, delegate: BaseTool, tracker: TurnTracker):
        self._delegate = delegate
        self._tracker = tracker
        self._name = delegate.name
        self._description = getattr(delegate, "description", "")
        self._input_schema = getattr(delegate, "input_schema", ToolInputSchema())

    def is_read_only(self, input: dict[str, Any] | None = None) -> bool:
        return self._delegate.is_read_only(input)

    def is_concurrency_safe(self, input: dict[str, Any] | None = None) -> bool:
        return self._delegate.is_concurrency_safe(input)

    async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        path = resolve_tool_path(str(input.get("file_path", "")), context)
        self._tracker.record_activity("Reading file", self.name, path)
        result = await self._delegate.call(input, context)
        if not result.is_error:
            self._tracker.mark_read(path)
        return result


class RuntimeTraceTool(BaseTool):
    def __init__(self, delegate: BaseTool, tracker: TurnTracker):
        self._delegate = delegate
        self._tracker = tracker
        self._name = delegate.name
        self._description = getattr(delegate, "description", "")
        self._input_schema = getattr(delegate, "input_schema", ToolInputSchema())

    def is_read_only(self, input: dict[str, Any] | None = None) -> bool:
        return self._delegate.is_read_only(input)

    def is_concurrency_safe(self, input: dict[str, Any] | None = None) -> bool:
        return self._delegate.is_concurrency_safe(input)

    async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        target = str(
            input.get("command")
            or input.get("file_path")
            or input.get("pattern")
            or input.get("name")
            or input.get("description")
            or self.name
        )
        self._tracker.record_activity("Running tool", self.name, target)
        return await self._delegate.call(input, context)


class RuntimeEditTool(BaseTool):
    def __init__(self, delegate: BaseTool, tracker: TurnTracker):
        self._delegate = delegate
        self._tracker = tracker
        self._name = delegate.name
        self._description = getattr(delegate, "description", "")
        self._input_schema = getattr(delegate, "input_schema", ToolInputSchema())

    def is_read_only(self, input: dict[str, Any] | None = None) -> bool:
        return self._delegate.is_read_only(input)

    def is_concurrency_safe(self, input: dict[str, Any] | None = None) -> bool:
        return self._delegate.is_concurrency_safe(input)

    async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        target_path = resolve_tool_path(str(input.get("file_path", "")), context)
        if not self._tracker.has_read(target_path):
            return ToolResult(
                tool_use_id="",
                content=f"Edit blocked: read {target_path} first in this turn, then retry.",
                is_error=True,
            )

        self._tracker.record_activity("Editing file", self.name, target_path)

        with open(target_path, "r", encoding="utf-8", errors="replace") as handle:
            before = handle.read()

        result = await self._delegate.call(input, context)
        if result.is_error:
            return result

        with open(target_path, "r", encoding="utf-8", errors="replace") as handle:
            after = handle.read()

        diff = "\n".join(
            unified_diff(
                before.splitlines(),
                after.splitlines(),
                fromfile=target_path,
                tofile=target_path,
                lineterm="",
            )
        )
        return ToolResult(
            tool_use_id="",
            content=f"Successfully edited {target_path}\n{diff}" if diff else f"Successfully edited {target_path}",
        )
