"""Multi-agent team orchestration for cock-code."""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from typing import Any

from open_agent_sdk.types import BaseTool, ToolContext, ToolInputSchema, ToolResult


MAX_TEAM_MEMBERS = 5

_runtime_team_manager: "TeamManager | None" = None
_runtime_orchestrator: Any = None


def set_runtime_team_bridge(team_manager: "TeamManager | None", orchestrator: Any | None) -> None:
    global _runtime_team_manager, _runtime_orchestrator
    _runtime_team_manager = team_manager
    _runtime_orchestrator = orchestrator


def get_runtime_team_bridge() -> tuple["TeamManager | None", Any | None]:
    return _runtime_team_manager, _runtime_orchestrator


class AgentPool:
    """Manages a pool of persistent Agent instances keyed by member name."""

    def __init__(self) -> None:
        self._members: dict[str, Any] = {}
        self._mailboxes: dict[str, asyncio.Queue[dict[str, str]]] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._unhealthy: set[str] = set()
        self._busy: set[str] = set()

    @property
    def member_names(self) -> list[str]:
        return list(self._members.keys())

    def has_member(self, name: str) -> bool:
        return name in self._members

    def is_busy(self, name: str) -> bool:
        return name in self._busy

    async def create_member(
        self,
        name: str,
        definition: dict[str, Any],
        config: Any,
        abort_signal: asyncio.Event | None = None,
    ) -> None:
        from cock_code.runtime import _build_subagent_config, _create_sdk_agent
        from open_agent_sdk import ToolContext

        child_config = _build_subagent_config(
            config, definition, {}, ToolContext(cwd=config.cwd or ".", env=config.env)
        )
        system_prompt = str(
            definition.get("prompt") or definition.get("system_prompt") or definition.get("description") or ""
        )
        agent = _create_sdk_agent(child_config, include_runtime_agent_tool=False, system_prompt=system_prompt)
        await agent._initialize()
        if abort_signal is not None:
            agent._options.abort_signal = abort_signal
        self._members[name] = agent
        self._mailboxes[name] = asyncio.Queue()
        self._locks[name] = asyncio.Lock()

    async def dispatch(self, member: str, task: str) -> str:
        if member not in self._members:
            return f"Error: unknown team member '{member}'"
        if member in self._unhealthy:
            return f"Error: team member '{member}' is unavailable due to a previous error"
        async with self._locks[member]:
            agent = self._members[member]
            task_with_messages = self._inject_mailbox(member, task)
            try:
                result = await agent.prompt(task_with_messages)
                return result.text or ""
            except Exception as exc:
                self._unhealthy.add(member)
                return f"Error: member '{member}' failed: {exc}"

    async def dispatch_async(
        self,
        member: str,
        task: str,
        task_id: str,
        cwd: str,
        env: dict[str, str],
    ) -> str:
        """Fire-and-forget dispatch. Creates SDK task, spawns asyncio.Task, returns task_id immediately."""
        from cock_code.runtime import (
            _track_background_task,
            _update_background_subagent_task,
            sanitize_task_output,
        )

        if member not in self._members:
            return f"Error: unknown team member '{member}'"
        if member in self._unhealthy:
            return f"Error: team member '{member}' is unavailable due to a previous error"
        if member in self._busy:
            return f"Error: team member '{member}' is busy with a previous dispatch. Wait for it to complete before dispatching again."

        self._busy.add(member)

        async def _run_member() -> None:
            try:
                result = await self.dispatch(member, task)
                await _update_background_subagent_task(
                    task_id,
                    status="completed",
                    output=sanitize_task_output(result),
                    cwd=cwd,
                    env=env,
                )
            except asyncio.CancelledError:
                await _update_background_subagent_task(
                    task_id,
                    status="cancelled",
                    output="Cancelled",
                    cwd=cwd,
                    env=env,
                )
                raise
            except Exception as exc:
                self._unhealthy.add(member)
                await _update_background_subagent_task(
                    task_id,
                    status="cancelled",
                    output=f"Error: {exc}",
                    cwd=cwd,
                    env=env,
                )
            finally:
                self._busy.discard(member)

        task_handle = asyncio.create_task(_run_member())
        _track_background_task(task_handle)
        return task_id

    def send_message(self, to: str, message: dict[str, str]) -> None:
        if to in self._mailboxes:
            self._mailboxes[to].put_nowait(message)

    def _inject_mailbox(self, member: str, task: str) -> str:
        messages: list[str] = []
        while not self._mailboxes[member].empty():
            try:
                msg = self._mailboxes[member].get_nowait()
                sender = msg.get("from", "unknown")
                content = msg.get("content", "")
                messages.append(f"[Message from {sender}]: {content}")
            except asyncio.QueueEmpty:
                break
        if not messages:
            return task
        header = "\n".join(messages)
        return f"{header}\n\n{task}"

    async def close_all(self) -> None:
        for agent in self._members.values():
            with contextlib.suppress(Exception):
                await agent.close()
        self._members.clear()
        self._mailboxes.clear()
        self._locks.clear()
        self._unhealthy.clear()
        self._busy.clear()

    async def clear_histories(self) -> None:
        for agent in self._members.values():
            if hasattr(agent, "clear"):
                agent.clear()


class TeamManager:
    """Manages team lifecycle: creation, dispatch, messaging, and teardown."""

    def __init__(self) -> None:
        self._pool: AgentPool | None = None
        self._team_id: str = ""
        self._team_name: str = ""
        self._member_definitions: dict[str, dict[str, Any]] = {}
        self._original_append_prompt: str = ""
        self._original_tool_pool: list[Any] | None = None
        self._active = False

    def is_active(self) -> bool:
        return self._active

    def _team_prompt(self) -> str:
        original = self._original_append_prompt
        lines = [
            f"# Team: {self._team_name}",
            f"You are the orchestrator for team '{self._team_name}'. Members: {', '.join(self._member_definitions.keys())}.",
            "Use TeamDispatch to assign tasks to members. Use SendMessage to communicate with members.",
            "Do NOT use the Agent tool for team members — use TeamDispatch instead. The Agent tool is not available while a team is active.",
            "When you assign work with TeamDispatch, the member processes it in the background. The result will appear when complete.",
            "Do not also perform the same work yourself unless the member fails, stops, or you are explicitly taking over.",
        ]
        if original:
            sanitized = "\n".join(
                line for line in original.splitlines()
                if "Use the Agent tool" not in line and "Agent tool or a background task" not in line
            ).strip()
            if sanitized:
                lines.insert(0, sanitized)
                lines.insert(1, "")
        return "\n".join(lines)

    def active_team_id(self) -> str:
        return self._team_id

    async def ensure_orchestrator_team_state(self, orchestrator: Any) -> None:
        if not self._active:
            return
        if hasattr(orchestrator, "_initialized") and not orchestrator._initialized and hasattr(orchestrator, "_initialize"):
            await orchestrator._initialize()
        patch_tool_pool(
            orchestrator,
            add_tools=[TeamDispatchTool(self), TeamSendMessageTool(self, sender_name="orchestrator"), TeamStatusTool(self)],
            remove_names=["TeamCreate", "TeamDelete", "Agent", "SendMessage", "TeamDispatch", "Read"],
        )
        if hasattr(orchestrator, "_options"):
            orchestrator._options.append_system_prompt = self._team_prompt()

    async def create_team(
        self,
        name: str,
        members: list[str],
        config: Any,
        orchestrator: Any,
        abort_signal: asyncio.Event | None = None,
    ) -> None:
        if self._active:
            raise RuntimeError("Team already active. Use /team stop first.")

        from cock_code.config import RuntimeConfig
        if not isinstance(config, RuntimeConfig):
            raise TypeError("config must be a RuntimeConfig")

        agents_def = config.agents
        if not agents_def:
            raise RuntimeError("No agent definitions found. Use /agents add or --agents-file first.")

        for member_name in members:
            if member_name not in agents_def:
                raise RuntimeError(f"Agent '{member_name}' is not defined. Use /agents add or --agents-file first.")

        if len(members) > MAX_TEAM_MEMBERS:
            raise RuntimeError(f"Team cannot have more than {MAX_TEAM_MEMBERS} members.")

        seen: set[str] = set()
        for member_name in members:
            if member_name in seen:
                raise RuntimeError(f"Duplicate member name '{member_name}' in team.")
            seen.add(member_name)

        pool = AgentPool()
        for member_name in members:
            definition = agents_def[member_name]
            await pool.create_member(member_name, definition, config, abort_signal)

        self._original_append_prompt = getattr(orchestrator._options, "append_system_prompt", "") or ""

        if hasattr(orchestrator, "_tool_pool"):
            self._original_tool_pool = list(orchestrator._tool_pool)

        dispatch_tool = TeamDispatchTool(self)
        send_tool = TeamSendMessageTool(self, sender_name="orchestrator")
        status_tool = TeamStatusTool(self)
        patch_tool_pool(
            orchestrator,
            add_tools=[dispatch_tool, send_tool, status_tool],
            remove_names=["TeamCreate", "TeamDelete", "Agent", "SendMessage", "Read"],
        )

        for member_name in members:
            member_agent = pool._members[member_name]
            member_send_tool = TeamSendMessageTool(self, sender_name=member_name)
            patch_tool_pool(
                member_agent,
                add_tools=[member_send_tool],
                remove_names=["TeamCreate", "TeamDelete", "Agent", "SendMessage"],
            )
            member_prompt = (
                f"You are '{member_name}', a member of team '{name}'. "
                f"Other members: {', '.join(m for m in members if m != member_name)}. "
                "Use SendMessage to communicate with other team members. "
                "If another team member already owns a task, do not duplicate that same work unless they fail, stop, or explicitly hand it off to you."
            )
            existing_prompt = getattr(member_agent._options, "append_system_prompt", "") or ""
            member_agent._options.append_system_prompt = f"{existing_prompt}\n\n{member_prompt}" if existing_prompt else member_prompt

        self._pool = pool
        self._team_id = str(uuid.uuid4())[:8]
        self._team_name = name
        self._member_definitions = {m: dict(agents_def[m]) for m in members}
        self._active = True
        await self.ensure_orchestrator_team_state(orchestrator)

    async def dispatch(self, member: str, task: str) -> str:
        if not self._active or self._pool is None:
            return "No team is active. Use /team create to start a team."
        return await self._pool.dispatch(member, task)

    async def dispatch_async(self, member: str, task: str, task_id: str, cwd: str, env: dict[str, str]) -> str:
        """Non-blocking dispatch. Returns task_id immediately; member processes in background."""
        if not self._active or self._pool is None:
            return "No team is active. Use /team create to start a team."
        result = await self._pool.dispatch_async(member, task, task_id, cwd, env)
        return result

    async def send_message(self, to: str, content: str, sender: str = "orchestrator") -> None:
        if not self._active or self._pool is None:
            raise RuntimeError("No team is active. Use /team create to start a team.")
        if not self._pool.has_member(to):
            raise RuntimeError(f"Unknown team member '{to}'")
        self._pool.send_message(to, {"from": sender, "content": content})

    async def clear(self) -> None:
        if self._pool is not None:
            await self._pool.clear_histories()

    async def close_team(self, orchestrator: Any) -> None:
        if not self._active:
            return

        if self._pool is not None:
            await self._pool.close_all()

        if self._original_tool_pool is not None:
            orchestrator._tool_pool = list(self._original_tool_pool)
        else:
            patch_tool_pool(
                orchestrator,
                add_tools=[],
                remove_names=["TeamDispatch", "SendMessage", "TeamStatus"],
            )

        orchestrator._options.append_system_prompt = self._original_append_prompt

        self._pool = None
        self._team_id = ""
        self._team_name = ""
        self._member_definitions = {}
        self._original_tool_pool = None
        self._active = False

    def info(self) -> dict[str, Any]:
        if not self._active:
            return {"active": False}
        members: dict[str, str] = {}
        if self._pool is not None:
            for name in self._pool.member_names:
                if name in self._pool._unhealthy:
                    members[name] = "unhealthy"
                elif name in self._pool._busy:
                    members[name] = "busy"
                else:
                    members[name] = "idle"
        return {
            "active": True,
            "team_id": self._team_id,
            "team_name": self._team_name,
            "members": members,
            "note": "Members are idle until you dispatch tasks via TeamDispatch. Dispatched members run in the background.",
        }


def patch_tool_pool(
    agent: Any,
    add_tools: list[BaseTool] | None = None,
    remove_names: list[str] | None = None,
) -> None:
    """Dynamically add/remove tools from an agent's tool pool after initialization."""
    if not hasattr(agent, "_tool_pool") or agent._tool_pool is None:
        return
    if remove_names:
        agent._tool_pool = [t for t in agent._tool_pool if t.name not in remove_names]
    if add_tools:
        agent._tool_pool.extend(add_tools)


class TeamDispatchTool(BaseTool):
    _name = "TeamDispatch"
    _description = "Dispatch a task to a team member for background processing. Returns immediately. The member processes the task asynchronously and the result appears when complete."
    _input_schema = ToolInputSchema(
        properties={
            "member": {"type": "string", "description": "Name of the team member to dispatch to"},
            "task": {"type": "string", "description": "Task description for the member"},
        },
        required=["member", "task"],
    )

    def __init__(self, team_manager: TeamManager):
        self._team_manager = team_manager

    def is_read_only(self, input: dict[str, Any] | None = None) -> bool:
        return False

    def is_concurrency_safe(self, input: dict[str, Any] | None = None) -> bool:
        return False

    async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        from cock_code.runtime import _create_background_subagent_task

        member = input.get("member", "")
        task = input.get("task", "")
        if not member:
            return ToolResult(tool_use_id="", content="Error: 'member' is required", is_error=True)
        if not task:
            return ToolResult(tool_use_id="", content="Error: 'task' is required", is_error=True)
        try:
            task_id = await _create_background_subagent_task(
                f"team-{member}",
                task,
                context.cwd,
                context.env,
            )
        except Exception as exc:
            return ToolResult(tool_use_id="", content=f"Error: could not create background task: {exc}", is_error=True)

        result = await self._team_manager.dispatch_async(member, task, task_id, context.cwd, context.env)

        if result.startswith("Error:"):
            from cock_code.runtime import _update_background_subagent_task
            await _update_background_subagent_task(
                task_id,
                status="cancelled",
                output=result,
                cwd=context.cwd,
                env=context.env,
            )
            return ToolResult(tool_use_id="", content=result, is_error=True)

        return ToolResult(
            tool_use_id="",
            content=(
                f"Dispatched task to team member '{member}' (task_id: {task_id}). "
                "They are processing it in the background. The result will appear when complete. "
                "Do not also perform the same task yourself unless they fail, stop, or you explicitly take over."
            ),
        )


class TeamStatusTool(BaseTool):
    _name = "TeamStatus"
    _description = "Check the status of team members — whether they are idle, busy, or unhealthy."
    _input_schema = ToolInputSchema(
        properties={},
        required=[],
    )

    def __init__(self, team_manager: TeamManager):
        self._team_manager = team_manager

    def is_read_only(self, input: dict[str, Any] | None = None) -> bool:
        return True

    def is_concurrency_safe(self, input: dict[str, Any] | None = None) -> bool:
        return True

    async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        info = self._team_manager.info()
        if not info.get("active"):
            return ToolResult(tool_use_id="", content="No team is active.")
        members = info.get("members", {})
        lines = [f"Team '{info.get('team_name', '')}' (id: {info.get('team_id', '')}):"]
        for name, status in members.items():
            lines.append(f"  {name}: {status}")
        return ToolResult(tool_use_id="", content="\n".join(lines))


class TeamSendMessageTool(BaseTool):
    _name = "SendMessage"
    _description = "Send a message to a team member. Messages are delivered before the member's next task."
    _input_schema = ToolInputSchema(
        properties={
            "to": {"type": "string", "description": "Team member name to send to"},
            "content": {"type": "string", "description": "Message content"},
        },
        required=["to", "content"],
    )

    def __init__(self, team_manager: TeamManager, sender_name: str = "orchestrator"):
        self._team_manager = team_manager
        self._sender_name = sender_name

    def is_read_only(self, input: dict[str, Any] | None = None) -> bool:
        return False

    def is_concurrency_safe(self, input: dict[str, Any] | None = None) -> bool:
        return True

    async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        to = input.get("to", "")
        content = input.get("content", "")
        if not to:
            return ToolResult(tool_use_id="", content="Error: 'to' is required", is_error=True)
        if not content:
            return ToolResult(tool_use_id="", content="Error: 'content' is required", is_error=True)
        try:
            await self._team_manager.send_message(to, content, sender=self._sender_name)
            return ToolResult(tool_use_id="", content=f"Message sent to {to}.")
        except Exception as exc:
            return ToolResult(tool_use_id="", content=f"Error: {exc}", is_error=True)


class SDKTeamCreateBridgeTool(BaseTool):
    _name = "TeamCreate"
    _description = "Create a team for multi-agent coordination."
    _input_schema = ToolInputSchema(
        properties={
            "name": {"type": "string", "description": "Team name"},
            "description": {"type": "string", "description": "Team purpose"},
            "members": {"type": "array", "description": "List of agent names"},
        },
        required=["name"],
    )

    def is_read_only(self, input: dict[str, Any] | None = None) -> bool:
        return False

    def is_concurrency_safe(self, input: dict[str, Any] | None = None) -> bool:
        return False

    async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        team_manager, orchestrator = get_runtime_team_bridge()
        if team_manager is None or orchestrator is None:
            return ToolResult(tool_use_id="", content="Error: team lifecycle unavailable", is_error=True)
        members = input.get("members", [])
        if not isinstance(members, list) or not all(isinstance(member, str) for member in members):
            return ToolResult(tool_use_id="", content="Error: members must be a list of agent names", is_error=True)
        if not members:
            return ToolResult(tool_use_id="", content="Error: members must not be empty", is_error=True)
        member_names: list[str] = [str(member) for member in members]
        config = getattr(orchestrator, "_cock_code_config", None)
        abort_signal = getattr(getattr(orchestrator, "_options", None), "abort_signal", None)
        if config is None:
            return ToolResult(tool_use_id="", content="Error: missing runtime config", is_error=True)
        team_name = str(input.get("name", "")).strip()
        team_description = str(input.get("description", "")).strip()
        if not isinstance(getattr(config, "agents", None), dict):
            config.agents = {}
        for member_name in member_names:
            if member_name in config.agents:
                continue
            description = team_description or f"{member_name} team member"
            config.agents[member_name] = {
                "description": description,
                "prompt": (
                    f"You are {member_name}, a persistent member of team '{team_name or 'team'}'. "
                    f"Team purpose: {team_description or 'Collaborate with the orchestrator on assigned tasks.'}"
                ),
            }
        try:
            await team_manager.create_team(team_name, member_names, config, orchestrator, abort_signal)
        except Exception as exc:
            return ToolResult(tool_use_id="", content=f"Error: {exc}", is_error=True)
        if abort_signal is not None:
            abort_signal.set()
        return ToolResult(
            tool_use_id="",
            content=(
                f"Created team {team_manager.active_team_id()}: {team_name}. "
                "Members are idle until you dispatch tasks via TeamDispatch."
            ),
        )


class SDKTeamDeleteBridgeTool(BaseTool):
    _name = "TeamDelete"
    _description = "Delete a team and cleanup resources."
    _input_schema = ToolInputSchema(
        properties={"team_id": {"type": "string", "description": "Team ID to delete"}},
        required=["team_id"],
    )

    def is_read_only(self, input: dict[str, Any] | None = None) -> bool:
        return False

    def is_concurrency_safe(self, input: dict[str, Any] | None = None) -> bool:
        return False

    async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        team_manager, orchestrator = get_runtime_team_bridge()
        if team_manager is None or orchestrator is None:
            return ToolResult(tool_use_id="", content="Error: team lifecycle unavailable", is_error=True)
        team_id = str(input.get("team_id", ""))
        if not team_manager.is_active() or team_manager.active_team_id() != team_id:
            return ToolResult(tool_use_id="", content=f"Error: team {team_id} not found", is_error=True)
        try:
            await team_manager.close_team(orchestrator)
        except Exception as exc:
            return ToolResult(tool_use_id="", content=f"Error: {exc}", is_error=True)
        return ToolResult(tool_use_id="", content=f"Deleted team {team_id}")