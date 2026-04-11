# Multi-Agent Teams Design

## Overview

Implement persistent multi-agent teams in cock-code with orchestrator-driven task dispatch, real-time inter-agent messaging, and interactive agent management. No changes to the Open Agent SDK — all implementation lives in cock-code.

## Design Decisions

- **Agent lifecycle:** Persistent (agents stay alive across turns within a team session)
- **Orchestration model:** Orchestrator-driven (the existing chat agent IS the orchestrator — no separate orchestrator agent is created)
- **Agent definition source:** Interactive `/agents` chat command (plus existing `--agents-file` CLI flag)
- **Approach:** AgentPool — team members are instantiated as persistent `Agent` objects in a pool

## Resolved Decisions (from review)

1. **Command syntax:** Simplified — no `--flags`. `/agents add` takes positional args only. Complex configs go in JSON file. This avoids shlex parser complexity.
2. **Mailbox injection:** Prepend to prompt text. Format: `[Message from sender]: content\n\n{original_task}`. Simplest approach that works with `agent.prompt()` without SDK changes.
3. **SDK team tools:** Filter out `TeamCreateTool` and `TeamDeleteTool` from the orchestrator's tool pool when a cock-code team is active. `state teams` still shows SDK team state; `/team info` shows cock-code team state.
4. **`/clear` semantics:** Clears orchestrator history AND all member histories. Stale member context causes confusion.
5. **Dynamic tool pool:** Add `patch_tool_pool(agent, add_tools, remove_names)` helper that mutates `agent._tool_pool` after initialization. SDK engine re-reads `_tool_pool` on each tool call.
6. **System prompt update:** Mutate `agent._options.append_system_prompt` on the live chat agent when team state changes. Store original for rollback on `/team stop`.
7. **`TeamDispatchTool` reference to pool:** Construct with explicit `team_manager` reference. Same for team-aware `SendMessageTool`.
8. **Concurrent dispatch:** Per-member `asyncio.Lock` in `AgentPool.dispatch()` to prevent interleaved history writes.
9. **Definition vs instance names:** Member names must exactly match unique definition keys in `config.agents`. No aliases or clones.
10. **Dispatch uses `agent.prompt()`, not `agent.query()`**: `prompt()` returns a collected result, which is what `TeamDispatch` needs. `query()` streams events for the outer rendering loop.

## Slash Commands

### `/agents` — Interactive Agent Management

```
/agents                              List configured agents
/agents add <name> <description>     Add an agent (uses name as default prompt)
/agents remove <name>                Remove an agent definition
/agents show <name>                  Show agent definition details
```

**Note:** `/agents add` takes only name and description. For custom prompts, tools, model, and max-turns, use `--agents-file` (JSON file). This keeps the CLI simple. The description doubles as the system prompt if no prompt is specified in the definition.

`/agents add` modifies `config.agents` at runtime. Modifications persist for the session only — nothing is written to disk. The orchestrator's system prompt is updated to include the new agent list via `_agent_context_prompt()`.

### `/team` — Team Orchestration

```
/team create <name> <member1> <member2> ...    Create a team with named agents
/team info                                       Show team members and status
/team stop                                       Disband team, close all member agents
```

**Edge cases:**
- `/team create` fails if a team is already active — must `/team stop` first (A2)
- `/agents remove <name>` fails if name is in an active team (A3)
- Duplicate member names are rejected with a descriptive error (A1)
- `MAX_TEAM_MEMBERS = 5` enforced (A7)
- Empty `config.agents` produces: "No agent definitions found. Use `/agents add` or `--agents-file` first" (A8)

## AgentPool — Persistent Agent Lifecycle

When `/team create` is called:

1. Each member name is resolved from `config.agents` (must be defined via `/agents add` or `--agents-file`)
2. A new `Agent` object is created per member via `_create_sdk_agent()` then `_initialize()`, with their definition's prompt, tools, and model
3. Agents are stored in a `dict[str, Agent]` pool, keyed by name
4. The `Agent` objects stay alive — conversation histories accumulate across turns
5. A `TeamDispatchTool` is registered in the orchestrator's tool pool via `patch_tool_pool()`
6. The SDK's `TeamCreateTool` and `TeamDeleteTool` are removed from the orchestrator's tool pool to avoid confusion
7. The orchestrator's system prompt is updated: "You are the orchestrator for team [name]. Members: [list]. Use TeamDispatch to assign tasks. Use SendMessage to communicate with members."

**The orchestrator IS the existing chat agent.** No separate orchestrator agent is created. `TeamDispatch` and team-aware `SendMessage` are injected into the existing agent's tool pool.

When a member is called via `TeamDispatch`:

1. The orchestrator calls `TeamDispatch(member="reviewer", task="review src/main.py")`
2. `TeamDispatch` finds the `Agent` in the pool (via `team_manager` reference)
3. Acquires per-member lock to prevent concurrent dispatch to the same member
4. Refreshes abort signal on the member agent's `_options`
5. Injects pending mailbox messages by prepending to the task prompt
6. Calls `agent.prompt(task)` — the member processes with full accumulated context
7. Wraps `agent.prompt()` in try/except — on API error returns `ToolResult(is_error=True, content=str(exc))`
8. The member's response is returned to the orchestrator as a `ToolResult` (raw text, no `_format_subagent_summary`)

When `/team stop` is called:

1. Cancel any in-flight dispatches via `abort_signal` set + `asyncio.Task.cancel()` for running member tasks
2. All member agents are `close()`d
3. The pool is cleared
4. `TeamDispatch` and team-aware `SendMessage` are removed from the orchestrator's tool pool
5. The orchestrator's system prompt is reverted to pre-team state
6. Original tool pool state is restored

## TeamDispatch — Orchestrator Routing Tool

A new tool constructed with an explicit `TeamManager` reference:

```python
class TeamDispatchTool(BaseTool):
    _name = "TeamDispatch"
    _description = "Dispatch a task to a team member. The member processes it with full context."
    _input_schema = ToolInputSchema(
        properties={
            "member": {"type": "string", "description": "Name of the team member"},
            "task": {"type": "string", "description": "Task description for the member"},
        },
        required=["member", "task"],
    )

    def __init__(self, team_manager: TeamManager):
        self._team_manager = team_manager

    async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        member = input.get("member", "")
        task = input.get("task", "")
        try:
            result = await self._team_manager.dispatch(member, task)
            return ToolResult(tool_use_id="", content=result)
        except Exception as exc:
            return ToolResult(tool_use_id="", content=f"Error: {exc}", is_error=True)
```

## SendMessage — Real-Time Inter-Agent Messaging

Current state: `SendMessageTool` writes to a `_mailboxes` dict but nobody reads from it.

New behavior:

1. Each member agent in the pool has an `asyncio.Queue[dict]` mailbox
2. When `SendMessage` is called (by orchestrator or any member), the message is put in the target's queue via `TeamManager.send_message()`
3. Before each `TeamDispatch` call, `_inject_mailbox()` drains the queue and prepends messages to the prompt: `[Message from sender]: content\n\n{original_task}`
4. This works because member agents are persistent — they're alive to receive messages

**Implementation detail:** The pool members get a **team-aware `SendMessageTool`** (a runtime wrapper) that routes through `TeamManager.send_message()` and writes to `asyncio.Queue` instead of the SDK's `_sdk_mailboxes`. The SDK's `SendMessageTool` is NOT given to pool members — it's replaced with this wrapper.

The orchestrator also gets the team-aware `SendMessageTool` to direct team members.

## TeamManager — Lifecycle Singleton

```python
class TeamManager:
    """Singleton managing team lifecycle, owned by run_chat()."""

    def __init__(self):
        self._pool: AgentPool | None = None
        self._team_name: str = ""
        self._original_append_prompt: str = ""  # for rollback

    async def create_team(self, name: str, members: list[str], config: RuntimeConfig) -> None:
        """Create team, instantiate agents, patch tools and prompts."""

    async def dispatch(self, member: str, task: str) -> str:
        """Dispatch task to pool member with locking and error handling."""

    async def send_message(self, to: str, content: str, sender: str = "") -> None:
        """Deliver message to member's mailbox queue."""

    async def close_team(self) -> None:
        """Disband team, close agents, revert tools and prompts."""

    def info(self) -> dict:
        """Return team status dict for /team info."""

    def is_active(self) -> bool:
        """Whether a team is currently active."""
```

`TeamManager` is instantiated in `run_chat()` and passed to relevant command handlers. Closed in the `finally` block alongside `agent.close()`.

## Abort Signal Integration

Each member agent in the pool must respect the `abort_signal`:

- Before each `dispatch()`, the member agent's `_options.abort_signal` is refreshed to the current signal (not the one from team creation time)
- `/team stop` sets `abort_signal` and cancels any running dispatch tasks
- SIGINT during a dispatch cancels the in-flight `agent.prompt()` call

## File Changes

### New files

- `src/cock_code/team.py` — `AgentPool`, `TeamManager`, `TeamDispatchTool`, `TeamSendMessageTool`, mailbox delivery, `patch_tool_pool()` helper

### Modified files

- `src/cock_code/cli.py` — `/agents` and `/team` slash commands, `TeamManager` lifecycle in `run_chat()`, cleanup in `finally` block
- `src/cock_code/runtime.py` — `_create_sdk_agent` supports pool member creation, `_agent_context_prompt()` updated for team awareness, `patch_tool_pool()` for dynamic tool injection
- `src/cock_code/rendering.py` — render team status (`/team info`), agent list (`/agents`), update `render_help()`
- `src/cock_code/chat.py` — no changes needed — `parse_chat_command()` returns `(name, args)` which already handles `/agents add name desc` and `/team create name m1 m2`

### Unchanged

- Open Agent SDK — no changes
- Existing `Agent` tool — still works for ad-hoc subagents
- Existing `/agent-bg`, `/model`, `/tools` commands — untouched
- Signal/interrupt architecture — untouched
- `abort_signal` propagation — untouched
- All existing tests must continue to pass

## AgentPool Implementation

```python
MAX_TEAM_MEMBERS = 5

class AgentPool:
    def __init__(self):
        self._members: dict[str, Agent] = {}
        self._mailboxes: dict[str, asyncio.Queue] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._unhealthy: set[str] = set()

    async def create_member(self, name: str, definition: dict, config: RuntimeConfig, abort_signal: asyncio.Event | None = None) -> None:
        """Instantiate a persistent agent for a team member."""
        child_config = _build_subagent_config(config, definition, {}, ToolContext(cwd=config.cwd or ".", env=config.env))
        agent = _create_sdk_agent(child_config, include_runtime_agent_tool=False, system_prompt=definition.get("prompt", ""))
        await agent._initialize()
        if abort_signal is not None:
            agent._options.abort_signal = abort_signal
        self._members[name] = agent
        self._mailboxes[name] = asyncio.Queue()
        self._locks[name] = asyncio.Lock()

    async def dispatch(self, member: str, task: str) -> str:
        """Send a task to a team member and return their response."""
        if member not in self._members:
            return f"Error: unknown team member '{member}'"
        if member in self._unhealthy:
            return f"Error: team member '{member}' is unavailable due to a previous error"
        async with self._locks[member]:
            agent = self._members[member]
            # Inject pending mailbox messages as context before the task
            task_with_messages = self._inject_mailbox(member, task)
            try:
                result = await agent.prompt(task_with_messages)
                return result.text or ""
            except Exception as exc:
                self._unhealthy.add(member)
                return f"Error: member '{member}' failed: {exc}"

    def send_message(self, to: str, message: dict) -> None:
        """Deliver a message to a team member's mailbox."""
        if to in self._mailboxes:
            self._mailboxes[to].put_nowait(message)

    def _inject_mailbox(self, member: str, task: str) -> str:
        """Drain pending mailbox messages and prepend to task prompt."""
        messages = []
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
        """Disband the team, closing all member agents."""
        for agent in self._members.values():
            with contextlib.suppress(Exception):
                await agent.close()
        self._members.clear()
        self._mailboxes.clear()
        self._locks.clear()
        self._unhealthy.clear()
```

## Error Handling

- If a member agent throws an exception during dispatch, catch it, mark member as unhealthy, return `ToolResult(is_error=True)` — the orchestrator can retry or reassign
- If a member is not found, return error immediately from `TeamDispatchTool`
- If the pool is empty (no team active), `TeamDispatch` returns "No team is active. Use /team create to start a team."
- If `/team create` is called while a team is active, return "Team already active. Use /team stop first."
- If `/agents remove <name>` is called for a name in the active team, return "Agent 'name' is in an active team. Use /team stop first."

## Tool Pool Patching

```python
def patch_tool_pool(agent: Agent, add_tools: list[BaseTool] | None = None, remove_names: list[str] | None = None) -> None:
    """Dynamically add/remove tools from an agent's tool pool after initialization."""
    if not hasattr(agent, '_tool_pool') or agent._tool_pool is None:
        return
    if remove_names:
        agent._tool_pool = [t for t in agent._tool_pool if t.name not in remove_names]
    if add_tools:
        agent._tool_pool.extend(add_tools)
    # Rebuild name-to-tool map
    agent._tool_map = {t.name: t for t in agent._tool_pool}
```

## Testing Strategy

### TDD Slices (RED → GREEN → REFACTOR)

1. **Parser slice:** Test `/agents add name desc` and `/team create team member1 member2` parsing → implement command routing → verify
2. **Agent definition slice:** Test add/list/show/remove + duplicates/empty → implement `config.agents` mutation → verify
3. **Team lifecycle slice:** Test create/info/stop + unknown members + duplicate members + active team exists → implement `TeamManager`/`AgentPool` → verify
4. **Dispatch slice:** Test success, unknown member, `agent.prompt()` exception, serialization → implement `TeamDispatchTool` + locks → verify
5. **Mailbox slice:** Test send/order/drain/unknown recipient/sender labeling → implement `TeamSendMessageTool` + prompt prepending → verify
6. **Abort/cleanup slice:** Test SIGINT during dispatch, `/team stop` with in-flight work, chat exit, `/resume` → implement shutdown helpers → verify

### Regression tests

- `/agent-bg`, `/model`, `/tools`, `/skills`, `/clear`, `/resume` all still work with an active team
- Agent tool still works for ad-hoc subagents while team is active
- All existing tests pass unchanged

## Non-Goals (V2)

- Parallel member dispatch (running multiple members simultaneously) — could be added later
- Team member tool sharing (members using each other's specialized tools)
- Team result aggregation (voting, consensus, merging)
- Persistent teams across sessions (teams live only within a chat session)