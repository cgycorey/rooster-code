"""SaveMemory tool — allows the agent to persist memories across sessions."""

from __future__ import annotations

from typing import Any

from open_agent_sdk.types import BaseTool, ToolContext, ToolInputSchema, ToolResult

from rooster_code.memory import save_memory


class SaveMemoryTool(BaseTool):
    """Tool for saving persistent memories across sessions."""

    def __init__(self) -> None:
        self._name = "SaveMemory"
        self._description = (
            "Save a persistent memory that will be loaded in future sessions. "
            "Use this when you learn something important about the user, their preferences, "
            "the project, or decisions that should persist. "
            "Memories are stored as markdown files and loaded at the start of every session."
        )
        self._input_schema = ToolInputSchema(
            properties={
                "name": {"type": "string", "description": "Short name for this memory"},
                "content": {"type": "string", "description": "The memory content in markdown format"},
                "description": {"type": "string", "description": "One-line description for relevance filtering"},
                "global_scope": {"type": "boolean", "description": "Save globally (true) or per-project (false, default)"},
            },
            required=["name", "content"],
        )

    def is_read_only(self, input: dict[str, Any] | None = None) -> bool:
        return False

    def is_concurrency_safe(self, input: dict[str, Any] | None = None) -> bool:
        return True

    async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        name = str(input.get("name", "")).strip()
        content = str(input.get("content", "")).strip()
        description = str(input.get("description", "")).strip()
        raw_global_scope = input.get("global_scope", False)
        if isinstance(raw_global_scope, bool):
            global_scope = raw_global_scope
        elif isinstance(raw_global_scope, str):
            global_scope = raw_global_scope.strip().lower() in {"1", "true", "yes", "on"}
        else:
            global_scope = bool(raw_global_scope)

        if not name or not content:
            return ToolResult(tool_use_id="", content="Error: name and content are required", is_error=True)

        try:
            project_cwd = context.cwd or None
            file_path = save_memory(name, content, description, global_scope=global_scope, project_cwd=project_cwd)
            scope = "global" if global_scope else "project"
            return ToolResult(
                tool_use_id="",
                content=f"Memory '{name}' saved ({scope} scope) to {file_path}",
            )
        except OSError as exc:
            return ToolResult(tool_use_id="", content=f"Error saving memory: {exc}", is_error=True)
