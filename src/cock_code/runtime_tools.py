from __future__ import annotations

from dataclasses import dataclass, field
from difflib import unified_diff
import os
from typing import Any, Awaitable, Callable

from open_agent_sdk.types import BaseTool, ToolContext, ToolInputSchema, ToolResult


AgentRunner = Callable[[dict[str, Any], ToolContext], Awaitable[ToolResult]]


@dataclass
class TurnTracker:
    read_paths: set[str] = field(default_factory=set)

    def reset(self) -> None:
        self.read_paths.clear()

    def mark_read(self, file_path: str) -> None:
        self.read_paths.add(file_path)

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
        "Subagents can run in the background or foreground."
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

    def __init__(self, runner: AgentRunner):
        self._runner = runner

    def is_read_only(self, input: dict[str, Any] | None = None) -> bool:
        return True

    def is_concurrency_safe(self, input: dict[str, Any] | None = None) -> bool:
        return True

    async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
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
        result = await self._delegate.call(input, context)
        if not result.is_error:
            self._tracker.mark_read(resolve_tool_path(str(input.get("file_path", "")), context))
        return result


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
