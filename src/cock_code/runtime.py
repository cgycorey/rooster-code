from __future__ import annotations

import asyncio
from dataclasses import replace
import contextlib
import json
from pathlib import Path
import threading
from typing import Any, cast
from open_agent_sdk import (
    AgentOptions,
    ToolContext,
    ToolResult,
    PermissionMode,
    ThinkingConfig,
    create_agent,
    delete_session as sdk_delete_session,
    estimate_messages_tokens,
    fork_session as sdk_fork_session,
    get_all_base_tools,
    get_all_cron_jobs,
    get_all_tasks,
    get_all_teams,
    get_config,
    get_current_plan,
    get_todos,
    is_plan_mode_active,
    get_session_info as sdk_get_session_info,
    get_session_messages as sdk_get_session_messages,
    get_skill,
    get_user_invocable_skills,
    init_bundled_skills,
    list_sessions as sdk_list_sessions,
    format_skills_for_prompt,
    rename_session as sdk_rename_session,
    register_skill,
    SkillDefinition,
    TaskCreateTool,
    TaskOutputTool,
    TaskStopTool,
    TaskUpdateTool,
    tag_session as sdk_tag_session,
    unregister_skill,
)
from open_agent_sdk.types import SDKMessage, SDKMessageType, SDKSystemSubtype
from open_agent_sdk.providers import CreateMessageParams
from open_agent_sdk.tools.skill_tool import SkillTool

from cock_code.config import RuntimeConfig
from cock_code.runtime_tools import RuntimeAgentTool, RuntimeEditTool, RuntimeReadTool, RuntimeTraceTool, TurnTracker


_loaded_local_skill_names: set[str] = set()
_background_subagent_tasks: set[asyncio.Task[None]] = set()
_notified_task_ids: set[str] = set()
_notified_task_ids_lock = threading.Lock()
_abort_signal: asyncio.Event | None = None


def set_abort_signal(event: asyncio.Event | None) -> None:
    global _abort_signal
    _abort_signal = event


def _parse_skill_metadata(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end == -1:
        return {}, text
    header = text[4:end]
    body = text[end + 5 :]
    metadata: dict[str, str] = {}
    for raw_line in header.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip()
    return metadata, body.strip()


def _parse_list_field(value: str) -> list[str]:
    raw = value.strip()
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]
    return [item.strip().strip('"').strip("'") for item in raw.split(",") if item.strip()]


def _build_filesystem_skill_definition(skill_dir: Path) -> SkillDefinition | None:
    skill_file = skill_dir / "SKILL.md"
    if not skill_file.exists():
        return None

    metadata, body = _parse_skill_metadata(skill_file.read_text(encoding="utf-8"))
    name = metadata.get("name") or skill_dir.name
    description = metadata.get("description") or body.splitlines()[0].strip() if body.strip() else skill_dir.name
    when_to_use = metadata.get("when_to_use", "")
    argument_hint = metadata.get("argument_hint", "")
    aliases = _parse_list_field(metadata["aliases"]) if "aliases" in metadata else []
    allowed_tools = _parse_list_field(metadata["allowed_tools"]) if "allowed_tools" in metadata else []
    model = metadata.get("model", "")
    context = metadata.get("context", "inline")
    agent = metadata.get("agent", "")
    user_invocable = metadata.get("user_invocable", "true").lower() != "false"

    async def get_prompt(args: str, ctx: ToolContext, *, content: str = body) -> list[dict[str, str]]:
        prompt_text = content.strip()
        if args.strip():
            prompt_text = f"{prompt_text}\n\nUser request: {args.strip()}"
        return [{"type": "text", "text": prompt_text}]

    return SkillDefinition(
        name=name,
        description=description,
        aliases=aliases,
        when_to_use=when_to_use,
        argument_hint=argument_hint,
        allowed_tools=allowed_tools,
        model=model,
        user_invocable=user_invocable,
        context="fork" if context == "fork" else "inline",
        agent=agent,
        get_prompt=get_prompt,
    )


def _resolve_skills_dir(config: RuntimeConfig) -> Path | None:
    if config.skills_dir:
        return Path(config.skills_dir)
    base = Path(config.cwd or ".")
    candidate = base / "skills"
    return candidate if candidate.exists() else None


def _ensure_skills_loaded(config: RuntimeConfig) -> None:
    global _loaded_local_skill_names
    init_bundled_skills()

    for name in list(_loaded_local_skill_names):
        unregister_skill(name)
    _loaded_local_skill_names.clear()

    skills_dir = _resolve_skills_dir(config)
    if not skills_dir or not skills_dir.exists():
        return

    for child in skills_dir.iterdir():
        if not child.is_dir():
            continue
        definition = _build_filesystem_skill_definition(child)
        if definition is None:
            continue
        register_skill(definition)
        _loaded_local_skill_names.add(definition.name)


def list_skill_names() -> list[str]:
    return sorted(skill.name for skill in get_user_invocable_skills())


async def get_task_output(task_id: str) -> str:
    result = await TaskOutputTool().call({"task_id": task_id}, ToolContext(cwd=".", env={}))
    return str(result.content)


async def stop_task(task_id: str) -> bool:
    result = await TaskStopTool().call({"task_id": task_id}, ToolContext(cwd=".", env={}))
    return not result.is_error


def read_background_notifications() -> list[dict[str, object]]:
    notifications: list[dict[str, object]] = []
    all_tasks = get_all_tasks()
    with _notified_task_ids_lock:
        for task_id, task in all_tasks.items():
            status = str(task.get("status", ""))
            if status in {"completed", "cancelled"} and task_id not in _notified_task_ids:
                _notified_task_ids.add(task_id)
                notifications.append({
                    "type": "background_task_completed",
                    "task_id": task_id,
                    "status": status,
                    "subject": str(task.get("subject", task_id)),
                    "output": str(task.get("output", "")),
                })
    return notifications


async def start_background_agent_task(config: RuntimeConfig, agent_name: str, prompt: str) -> str:
    result = await _run_subagent(
        config,
        {"name": agent_name, "prompt": prompt, "description": agent_name, "run_in_background": True},
        ToolContext(cwd=config.cwd or ".", env=config.env),
    )
    if result.is_error:
        raise RuntimeError(str(result.content))
    text = str(result.content)
    parts = text.split()
    if len(parts) >= 3 and parts[0] == "Created" and parts[1] == "task":
        return parts[2]
    raise RuntimeError(f"Could not parse background task ID from: {text}")


async def wait_for_task(task_id: str, poll_interval: float = 0.1, max_polls: int = 600) -> dict[str, object]:
    for _ in range(max_polls):
        task = get_all_tasks().get(task_id)
        if task is None:
            return {"status": "missing", "output": ""}
        status = str(task.get("status", ""))
        if status in {"completed", "cancelled"}:
            return {"status": status, "output": str(task.get("output", ""))}
        await asyncio.sleep(poll_interval)
    return {"status": str(get_all_tasks().get(task_id, {}).get("status", "in_progress")), "output": str(get_all_tasks().get(task_id, {}).get("output", ""))}


def _resolve_subagent_skill_request(config: RuntimeConfig, input: dict[str, Any]) -> tuple[str, str] | None:
    _ensure_skills_loaded(config)
    available = {name.lower(): name for name in list_skill_names()}
    if not available:
        return None

    requested = str(input.get("name") or input.get("subagent_type") or "").strip().lower()
    prompt = str(input.get("prompt") or "").strip()
    description = str(input.get("description") or "").strip()

    if requested and requested in available:
        return available[requested], prompt or description

    for source in (prompt, description):
        if not source:
            continue
        parts = source.split(maxsplit=1)
        head = parts[0].lower()
        if head in available:
            return available[head], parts[1] if len(parts) > 1 else ""

    return None


def _effective_agents(config: RuntimeConfig) -> dict[str, Any]:
    if config.agents:
        return config.agents
    return {
        "task": {
            "description": "General task agent",
            "prompt": "You are a careful general-purpose task agent. Use tools and skills deliberately, prefer minimal changes, and summarize your results clearly.",
            "max_turns": 3,
        }
    }


def _resolve_agent_definition(config: RuntimeConfig, input: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    agents = _effective_agents(config)

    requested = str(input.get("name") or input.get("subagent_type") or "")
    if requested:
        raw = agents.get(requested)
        return requested, raw if isinstance(raw, dict) else None

    if len(agents) == 1:
        key = next(iter(agents))
        raw = agents[key]
        return key, raw if isinstance(raw, dict) else None

    return "", None


def _build_subagent_config(
    config: RuntimeConfig,
    definition: dict[str, Any],
    input: dict[str, Any],
    context: ToolContext,
) -> RuntimeConfig:
    tools = definition.get("tools")
    disallowed_tools = definition.get("disallowed_tools")
    max_turns = definition.get("max_turns")
    return replace(
        config,
        model=str(input.get("model") or definition.get("model") or config.model or ""),
        cwd=context.cwd or config.cwd,
        allowed_tools=tools if isinstance(tools, list) else config.allowed_tools,
        disallowed_tools=disallowed_tools if isinstance(disallowed_tools, list) else config.disallowed_tools,
        max_turns=int(max_turns) if max_turns is not None else config.max_turns,
        persist_session=False,
    )


def _agent_context_prompt(config: RuntimeConfig) -> str:
    agents = _effective_agents(config)

    lines = ["# Configured Agents", "Use the Agent tool with the agent name when delegation is helpful."]
    for name, definition in agents.items():
        if isinstance(definition, dict):
            description = str(definition.get("description") or definition.get("prompt") or "")
        else:
            description = ""
        lines.append(f"- {name}: {description}".rstrip())

    skills_prompt = format_skills_for_prompt(config.max_tokens)
    if skills_prompt:
        lines.extend(["", "# Available Skills", skills_prompt])
    return "\n".join(lines)


def _extract_text_blocks(message: dict[str, Any]) -> list[str]:
    content = message.get("content", [])
    if isinstance(content, str):
        text = content.strip()
        return [text] if text else []
    if not isinstance(content, list):
        return []
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") != "text":
            continue
        text = str(block.get("text", "")).strip()
        if text:
            parts.append(text)
    return parts


def _format_subagent_summary(result_text: str, messages: list[dict[str, Any]]) -> str:
    outcomes: list[str] = []
    files: list[str] = []
    commands: list[str] = []
    open_issues: list[str] = []
    next_steps: list[str] = []
    findings: list[str] = []
    first_assistant_text: str = ""

    for message in messages:
        if str(message.get("role", "")) != "assistant":
            continue
        for text in _extract_text_blocks(message):
            if not first_assistant_text and text.strip():
                first_assistant_text = text.strip()
            for raw_line in text.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                lower = line.lower()
                if lower.startswith("outcome:"):
                    outcomes.append(line.partition(":")[2].strip())
                elif lower.startswith("files:"):
                    files.append(line.partition(":")[2].strip())
                elif lower.startswith("commands:"):
                    commands.append(line.partition(":")[2].strip())
                elif lower.startswith("findings:"):
                    findings.append(line.partition(":")[2].strip())
                elif lower.startswith("open issues:"):
                    open_issues.append(line.partition(":")[2].strip())
                elif lower.startswith("next step:"):
                    next_steps.append(line.partition(":")[2].strip())

    fallback = result_text or first_assistant_text or "No useful output returned"
    lines = [f"Outcome: {'; '.join(outcomes) if outcomes else fallback}" ]
    if files:
        lines.append(f"Files: {'; '.join(files)}")
    if commands:
        lines.append(f"Commands: {'; '.join(commands)}")
    if findings:
        lines.append(f"Findings: {'; '.join(findings)}")
    if open_issues:
        lines.append(f"Open issues: {'; '.join(open_issues)}")
    if next_steps:
        lines.append(f"Next step: {'; '.join(next_steps)}")
    return "\n".join(lines)


def _activity_status_event(action: str, tool: str, target: str) -> SDKMessage:
    return SDKMessage(
        type=SDKMessageType.SYSTEM,
        subtype=SDKSystemSubtype.STATUS,
        system_data={"activity_trace": [{"action": action, "tool": tool, "target": target}]},
    )


def _track_background_task(task: asyncio.Task[None]) -> None:
    _background_subagent_tasks.add(task)
    task.add_done_callback(_background_subagent_tasks.discard)


async def _create_background_subagent_task(subject: str, description: str, cwd: str, env: dict[str, str]) -> str:
    result = await TaskCreateTool().call(
        {"subject": subject, "description": description, "status": "in_progress"},
        ToolContext(cwd=cwd, env=env),
    )
    if result.is_error:
        raise RuntimeError(f"Failed to create task: {result.content}")
    text = str(result.content)
    parts = text.split()
    if len(parts) >= 3 and parts[0] == "Created" and parts[1] == "task":
        return parts[2].rstrip(":")
    raise RuntimeError(f"Could not parse task ID from: {text}")


async def _update_background_subagent_task(task_id: str, *, status: str | None = None, output: str | None = None, cwd: str = ".", env: dict[str, str] | None = None) -> None:
    payload: dict[str, Any] = {"task_id": task_id}
    if status is not None:
        payload["status"] = status
    if output is not None:
        payload["output"] = output
    await TaskUpdateTool().call(payload, ToolContext(cwd=cwd, env=env or {}))


async def cancel_background_subagent_tasks() -> None:
    tasks = list(_background_subagent_tasks)
    for task in tasks:
        task.cancel()
    for task in tasks:
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task



async def _run_subagent(config: RuntimeConfig, input: dict[str, Any], context: ToolContext) -> ToolResult:
    prompt = str(input.get("prompt", "")).strip()
    if not prompt:
        return ToolResult(tool_use_id="", content="Error: prompt is required", is_error=True)

    skill_request = _resolve_subagent_skill_request(config, input)
    agent_name, definition = _resolve_agent_definition(config, input)

    if skill_request is None and definition is None:
        if _effective_agents(config):
            return ToolResult(tool_use_id="", content=f"Error: unknown agent '{agent_name or 'unspecified'}'", is_error=True)
        return ToolResult(tool_use_id="", content="Error: no agents configured", is_error=True)

    if input.get("run_in_background"):
        task_id = await _create_background_subagent_task(
            str(input.get("name") or input.get("description") or "subagent"),
            prompt,
            context.cwd,
            context.env,
        )

        async def run_background() -> None:
            try:
                result = await _run_subagent(config, {**input, "run_in_background": False}, context)
                output = str(result.content)
                if result.is_error:
                    output = f"Error: {output}"
                await _update_background_subagent_task(
                    task_id,
                    status="completed",
                    output=output,
                    cwd=context.cwd,
                    env=context.env,
                )
            except asyncio.CancelledError:
                await _update_background_subagent_task(
                    task_id,
                    status="cancelled",
                    output="Error: Cancelled by shutdown",
                    cwd=context.cwd,
                    env=context.env,
                )
                raise
            except Exception as exc:
                await _update_background_subagent_task(
                    task_id,
                    status="cancelled",
                    output=f"Error: {exc}",
                    cwd=context.cwd,
                    env=context.env,
                )

        task = asyncio.create_task(run_background())
        _track_background_task(task)
        return ToolResult(tool_use_id="", content=f"Created task {task_id}")

    if skill_request:
        skill_name, args = skill_request
        working_config = replace(config, persist_session=False)
        working_agent = _create_sdk_agent(working_config, include_runtime_agent_tool=False)
        try:
            tool_context = ToolContext(cwd=context.cwd, env=context.env)
            result = await SkillTool().call({"skill": skill_name, "args": args}, tool_context)
            if result.is_error:
                return ToolResult(tool_use_id="", content=str(result.content), is_error=True)

            payload = json.loads(str(result.content))
            prompt_text = str(payload.get("prompt", "")).strip()
            if not prompt_text:
                return ToolResult(tool_use_id="", content=f'Error: Skill "{skill_name}" returned no prompt', is_error=True)

            overrides: dict[str, Any] = {}
            if payload.get("model"):
                overrides["model"] = str(payload["model"])
            if isinstance(payload.get("allowedTools"), list):
                overrides["allowed_tools"] = payload["allowedTools"]

            if payload.get("status") == "forked":
                child_config = replace(
                    working_config,
                    model=str(payload.get("model") or working_config.model or ""),
                    allowed_tools=payload.get("allowedTools") if isinstance(payload.get("allowedTools"), list) else working_config.allowed_tools,
                    persist_session=False,
                )
                child_agent = _create_sdk_agent(child_config, include_runtime_agent_tool=False)
                try:
                    query_result = await child_agent.prompt(prompt_text)
                finally:
                    await child_agent.close()
            else:
                query_result = await working_agent.prompt(prompt_text, overrides or None)

            text = query_result.text.strip() if query_result.text else ""
            summary = _format_subagent_summary(text, query_result.messages)
            return ToolResult(tool_use_id="", content=summary or f'Skill "{skill_name}" completed with no text output.')
        finally:
            await working_agent.close()

    if definition is None:
        return ToolResult(tool_use_id="", content="Error: no agent definition resolved", is_error=True)
    child_config = _build_subagent_config(config, definition, input, context)
    system_prompt = str(definition.get("prompt") or definition.get("system_prompt") or definition.get("description") or "")
    child_agent = _create_sdk_agent(child_config, include_runtime_agent_tool=False, system_prompt=system_prompt)
    try:
        result = await child_agent.prompt(prompt)
    finally:
        await child_agent.close()

    text = result.text.strip() if result.text else ""
    summary = _format_subagent_summary(text, result.messages)
    return ToolResult(tool_use_id="", content=summary or f"Agent {agent_name} completed with no text output.")


async def _stream_subagent(config: RuntimeConfig, input: dict[str, Any], context: ToolContext):
    prompt = str(input.get("prompt", "")).strip()
    if not prompt:
        yield SDKMessage(type=SDKMessageType.RESULT, text="Error: prompt is required", is_error=True)
        return

    if input.get("run_in_background"):
        result = await _run_subagent(config, input, context)
        yield SDKMessage(type=SDKMessageType.RESULT, text=str(result.content), is_error=result.is_error)
        return

    if skill_request := _resolve_subagent_skill_request(config, input):
        skill_name, args = skill_request
        working_agent = _create_sdk_agent(replace(config, persist_session=False), include_runtime_agent_tool=False)
        try:
            async for event in _stream_skill(config, working_agent, skill_name, args):
                if _abort_signal is not None and _abort_signal.is_set():
                    break
                yield event
        finally:
            await working_agent.close()
        return

    agent_name, definition = _resolve_agent_definition(config, input)
    if definition is None:
        message = f"Error: unknown agent '{agent_name or 'unspecified'}'" if _effective_agents(config) else "Error: no agents configured"
        yield SDKMessage(type=SDKMessageType.RESULT, text=message, is_error=True)
        return

    child_config = _build_subagent_config(config, definition, input, context)
    system_prompt = str(definition.get("prompt") or definition.get("system_prompt") or definition.get("description") or "")
    child_agent = _create_sdk_agent(child_config, include_runtime_agent_tool=False, system_prompt=system_prompt)
    try:
        yield _activity_status_event("Resolved subagent", "Agent", agent_name)
        async for event in child_agent.query(prompt):
            if _abort_signal is not None and _abort_signal.is_set():
                break
            yield event
    finally:
        await child_agent.close()
    yield _activity_status_event("Completed subagent", "Agent", agent_name)


def find_requested_agent_name(config: RuntimeConfig, prompt: str) -> str | None:
    lower_prompt = prompt.lower()
    for name in _effective_agents(config):
        lower_name = name.lower()
        if f"{lower_name} agent" in lower_prompt or f"use {lower_name}" in lower_prompt:
            return name
    if "use an agent" in lower_prompt or "use a subagent" in lower_prompt or "use an assistant agent" in lower_prompt:
        return "task"
    return None


async def run_named_agent_prompt(config: RuntimeConfig, agent_name: str, prompt: str) -> str:
    result = await _run_subagent(
        config,
        {"name": agent_name, "prompt": prompt, "description": agent_name},
        ToolContext(cwd=config.cwd or ".", env=config.env),
    )
    return str(result.content)


async def stream_named_agent_events(config: RuntimeConfig, agent_name: str, prompt: str):
    async for event in _stream_subagent(
        config,
        {"name": agent_name, "prompt": prompt, "description": agent_name},
        ToolContext(cwd=config.cwd or ".", env=config.env),
    ):
        yield event


async def _stream_skill(config: RuntimeConfig, agent, skill_name: str, args: str):
    _ensure_skills_loaded(config)
    context = ToolContext(cwd=config.cwd or ".", env=config.env)
    result = await SkillTool().call({"skill": skill_name, "args": args}, context)
    if result.is_error:
        yield SDKMessage(type=SDKMessageType.RESULT, text=str(result.content), is_error=True)
        return

    payload = json.loads(str(result.content))
    prompt_text = str(payload.get("prompt", "")).strip()
    if not prompt_text:
        yield SDKMessage(type=SDKMessageType.RESULT, text=f'Error: Skill "{skill_name}" returned no prompt', is_error=True)
        return

    command_name = str(payload.get("commandName", skill_name))
    status = str(payload.get("status") or "inline")
    yield _activity_status_event("Resolved subagent", "Skill", f"{command_name} ({status})")

    overrides: dict[str, Any] = {}
    if payload.get("model"):
        overrides["model"] = str(payload["model"])
    if isinstance(payload.get("allowedTools"), list):
        overrides["allowed_tools"] = payload["allowedTools"]

    if payload.get("status") == "forked":
        child_config = replace(
            config,
            model=str(payload.get("model") or config.model or ""),
            allowed_tools=payload.get("allowedTools") if isinstance(payload.get("allowedTools"), list) else config.allowed_tools,
            persist_session=False,
        )
        child_agent = _create_sdk_agent(child_config, include_runtime_agent_tool=True)
        try:
            async for event in child_agent.query(prompt_text):
                if _abort_signal is not None and _abort_signal.is_set():
                    break
                yield event
        finally:
            await child_agent.close()
        yield _activity_status_event("Completed subagent", "Skill", command_name)
        return

    query_overrides = overrides or None
    async for event in agent.query(prompt_text, query_overrides):
        if _abort_signal is not None and _abort_signal.is_set():
            break
        yield event
    yield _activity_status_event("Completed subagent", "Skill", command_name)


async def stream_skill_events(config: RuntimeConfig, agent, skill_name: str, args: str):
    async for event in _stream_skill(config, agent, skill_name, args):
        yield event


def _runtime_tools(config: RuntimeConfig, include_runtime_agent_tool: bool) -> list[Any]:
    return []


def build_agent_options(
    config: RuntimeConfig,
    *,
    include_runtime_agent_tool: bool = True,
    system_prompt: str = "",
) -> AgentOptions:
    _ensure_skills_loaded(config)
    agent_prompt = _agent_context_prompt(config)
    return AgentOptions(
        api_key=config.api_key or "",
        base_url=config.base_url or "",
        model=config.model or "",
        api_type=config.api_type or "",
        cwd=config.cwd or "",
        system_prompt=system_prompt,
        append_system_prompt=agent_prompt,
        tools=_runtime_tools(config, include_runtime_agent_tool),
        allowed_tools=config.allowed_tools,
        disallowed_tools=config.disallowed_tools,
        resume=config.resume or "",
        session_id=config.session_id or "",
        continue_session=config.continue_session,
        fork_session=config.fork_session or "",
        persist_session=config.persist_session,
        permission_mode=PermissionMode(config.permission_mode),
        max_turns=config.max_turns or 10,
        max_budget_usd=config.max_budget_usd,
        max_tokens=config.max_tokens or 16000,
        thinking=ThinkingConfig(budget_tokens=config.thinking_budget) if config.thinking_budget is not None else None,
        debug=config.debug,
        sandbox=config.sandbox,
        include_partial_messages=config.include_partials,
        env=config.env,
        custom_headers=config.custom_headers,
        agents=config.agents,
        hooks=config.hooks,
        json_schema=config.json_schema,
        mcp_servers=config.mcp_servers,
        extra_args=config.extra_args,
        abort_signal=_abort_signal,
    )


def _create_sdk_agent(
    config: RuntimeConfig,
    *,
    include_runtime_agent_tool: bool = True,
    system_prompt: str = "",
):
    agent = create_agent(
        build_agent_options(
            config,
            include_runtime_agent_tool=include_runtime_agent_tool,
            system_prompt=system_prompt,
        )
    )

    tracker = TurnTracker()

    if hasattr(agent, "query"):
        original_query = agent.query

        async def wrapped_query(prompt: str, overrides: dict[str, Any] | None = None):
            tracker.reset()
            event_queue: asyncio.Queue[SDKMessage | None] = asyncio.Queue()

            async def pump_query() -> None:
                try:
                    async for event in original_query(prompt, overrides):
                        await event_queue.put(event)
                finally:
                    await event_queue.put(None)

            query_task = asyncio.create_task(pump_query())
            activity_task = asyncio.create_task(tracker.next_activity())
            event_task = asyncio.create_task(event_queue.get())
            try:
                while True:
                    if _abort_signal is not None and _abort_signal.is_set():
                        break
                    done, _pending = await asyncio.wait(
                        {activity_task, event_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    if activity_task in done:
                        activity = activity_task.result()
                        yield SDKMessage(
                            type=SDKMessageType.SYSTEM,
                            subtype=SDKSystemSubtype.STATUS,
                            system_data={"activity_trace": [activity]},
                        )
                        activity_task = asyncio.create_task(tracker.next_activity())

                    if event_task in done:
                        event = event_task.result()
                        if event is None:
                            if query_task.done() and (exc := query_task.exception()) is not None:
                                raise exc
                            break
                        pending_activities = tracker.consume_pending_activities()
                        for activity in pending_activities:
                            yield SDKMessage(
                                type=SDKMessageType.SYSTEM,
                                subtype=SDKSystemSubtype.STATUS,
                                system_data={"activity_trace": [activity]},
                            )
                        sdk_event = event
                        if getattr(sdk_event, "type", None) and getattr(sdk_event.type, "value", "") == "tool_result":
                            activity_trace = tracker.consume_activity_trace()
                            if activity_trace:
                                sdk_event.system_data["activity_trace"] = activity_trace
                        yield sdk_event
                        event_task = asyncio.create_task(event_queue.get())
            finally:
                for task in (activity_task, event_task, query_task):
                    if task is query_task and task.done() and not task.cancelled():
                        continue
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await task

        setattr(agent, "query", wrapped_query)

    if include_runtime_agent_tool and _effective_agents(config) and hasattr(agent, "_initialize"):
        original_initialize = agent._initialize

        async def wrapped_initialize() -> None:
            await original_initialize()
            replaced = False
            new_pool = []
            for tool in getattr(agent, "_tool_pool", []):
                if getattr(tool, "name", "") == "Agent" and not replaced:
                    new_pool.append(RuntimeAgentTool(lambda input, context: _run_subagent(config, input, context), tracker))
                    replaced = True
                elif getattr(tool, "name", "") == "Read":
                    new_pool.append(RuntimeReadTool(tool, tracker))
                elif getattr(tool, "name", "") == "Edit":
                    new_pool.append(RuntimeEditTool(tool, tracker))
                elif getattr(tool, "name", "") != "Agent":
                    new_pool.append(RuntimeTraceTool(tool, tracker))
            if not replaced:
                new_pool.append(RuntimeAgentTool(lambda input, context: _run_subagent(config, input, context), tracker))
            agent._tool_pool = new_pool

        setattr(agent, "_initialize", wrapped_initialize)
    elif hasattr(agent, "_initialize"):
        original_initialize = agent._initialize

        async def wrapped_initialize_without_agent() -> None:
            await original_initialize()
            new_pool = []
            for tool in getattr(agent, "_tool_pool", []):
                if getattr(tool, "name", "") == "Read":
                    new_pool.append(RuntimeReadTool(tool, tracker))
                elif getattr(tool, "name", "") == "Edit":
                    new_pool.append(RuntimeEditTool(tool, tracker))
                elif getattr(tool, "name", "") != "Agent":
                    new_pool.append(RuntimeTraceTool(tool, tracker))
            agent._tool_pool = new_pool

        setattr(agent, "_initialize", wrapped_initialize_without_agent)
    return agent


def create_runtime_agent(config: RuntimeConfig):
    return _create_sdk_agent(config)


def _filter_history_for_manual_compaction(history: list[dict[str, object]]) -> list[dict[str, object]]:
    filtered: list[dict[str, object]] = []
    for message in history:
        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue

        content = message.get("content", [])
        if isinstance(content, str):
            text = content.strip()
            if text:
                filtered.append({"role": role, "content": [{"type": "text", "text": text}]})
            continue

        if not isinstance(content, list):
            continue

        text_blocks: list[dict[str, str]] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            mapping = cast(dict[str, Any], block)
            if mapping.get("type") != "text":
                continue
            text = str(mapping.get("text", "")).strip()
            if text:
                text_blocks.append({"type": "text", "text": text})

        if text_blocks:
            filtered.append({"role": role, "content": text_blocks})

    return filtered


def _build_manual_compaction_summary_prompt(messages: list[dict[str, object]]) -> str:
    conversation_text = ""
    for msg in messages:
        role = str(msg.get("role", "user"))
        content = msg.get("content", [])
        text_parts: list[str] = []
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                mapping = cast(dict[str, Any], block)
                if mapping.get("type") == "text":
                    text_parts.append(str(mapping.get("text", "")))
        elif isinstance(content, str):
            text_parts.append(content)

        text = "\n".join(part for part in text_parts if part).strip()
        if text:
            conversation_text += f"\n{role}: {text[:5000]}\n"

    return (
        "Summarize this session for immediate continuation. Be concise and preserve only information needed to keep working without re-discovery. "
        "Use the exact section headings below and prefer bullet points under each heading. If a section has nothing useful, write 'None'.\n\n"
        "## Goal\n"
        "- The current objective and success condition.\n\n"
        "## Current State\n"
        "- What is already done, in progress, and not started.\n\n"
        "## Key Decisions\n"
        "- Important implementation or product decisions and why they were made.\n\n"
        "## Code/Files\n"
        "- Files, modules, commands, or tests that matter for continuing the work.\n\n"
        "## Constraints / What to Avoid\n"
        "- Scope limits, invariants, failed approaches, or things that must not change.\n\n"
        "## Blockers / Open Questions\n"
        "- Only unresolved items that materially affect the next step.\n\n"
        "## Next Step\n"
        "- The single best next action to take.\n\n"
        "## Transcript\n"
        + conversation_text[:50000]
    )


async def _compact_with_provider(agent, messages: list[dict[str, object]]) -> tuple[str, list[dict[str, object]]]:
    ensure_provider = getattr(agent, "_ensure_provider", None)
    resolve_model = getattr(agent, "_resolve_model", None)
    if not callable(ensure_provider) or not callable(resolve_model):
        raise RuntimeError("Agent does not expose the SDK compaction prerequisites.")

    provider = ensure_provider()
    create_message = getattr(provider, "create_message", None)
    if not callable(create_message):
        raise RuntimeError("Agent provider does not support message creation.")

    response = await create_message(
        CreateMessageParams(
            model=resolve_model(),
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": _build_manual_compaction_summary_prompt(messages),
                }
            ],
        )
    )
    summary = "".join(
        str(block.get("text", ""))
        for block in response.content
        if isinstance(block, dict) and block.get("type") == "text"
    ).strip()
    if not summary:
        raise RuntimeError("Compaction produced an empty summary.")
    compacted_messages: list[dict[str, object]] = [
        {
            "role": "user",
            "content": [{"type": "text", "text": f"[Previous conversation summary]\n\n{summary}"}],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "I understand the context. Let me continue from where we left off."}],
        },
    ]
    return summary, compacted_messages


async def compact_current_session(agent) -> dict[str, object]:
    if hasattr(agent, "_initialize"):
        await agent._initialize()

    history = list(getattr(agent, "_history", []))
    compactable_history = _filter_history_for_manual_compaction(history)
    before_tokens = estimate_messages_tokens(compactable_history)

    if len(compactable_history) < 2:
        return {
            "compacted": False,
            "summary": "",
            "before_tokens": before_tokens,
            "after_tokens": before_tokens,
            "reason": "Need at least two messages before compaction.",
        }

    summary, compacted_history = await _compact_with_provider(agent, compactable_history)
    agent._history = compacted_history
    after_tokens = estimate_messages_tokens(compacted_history)

    compacted = compacted_history != compactable_history
    return {
        "compacted": compacted,
        "summary": summary,
        "before_tokens": before_tokens,
        "after_tokens": after_tokens,
        "reason": "" if compacted else "Compaction produced no smaller history.",
    }


async def list_sessions():
    return await sdk_list_sessions()


async def get_session_messages(session_id: str):
    return await sdk_get_session_messages(session_id)


async def get_session_info(session_id: str):
    return await sdk_get_session_info(session_id)


async def delete_session(session_id: str):
    return await sdk_delete_session(session_id)


async def enforce_session_retention(limit: int = 20) -> None:
    sessions = await sdk_list_sessions()
    if len(sessions) <= limit:
        return

    for session in sessions[limit:]:
        session_id = session.get("id")
        if isinstance(session_id, str) and session_id:
            await sdk_delete_session(session_id)


async def fork_session(session_id: str, new_id: str | None):
    return await sdk_fork_session(session_id, new_id)


async def rename_session(session_id: str, title: str):
    await sdk_rename_session(session_id, title)


async def tag_session(session_id: str, tags: list[str]):
    await sdk_tag_session(session_id, tags)


def get_state_snapshot(name: str, agent_name: str | None = None):
    if name == "todos":
        return get_todos()
    if name == "tasks":
        return get_all_tasks()
    if name == "teams":
        return get_all_teams()
    if name == "mailboxes":
        from open_agent_sdk.tools import _mailboxes as _sdk_mailboxes
        if agent_name is not None:
            return _sdk_mailboxes.get(agent_name, [])
        return dict(_sdk_mailboxes)
    if name == "config":
        return get_config()
    if name == "cron":
        return get_all_cron_jobs()
    if name == "plan":
        return {
            "active": is_plan_mode_active(),
            "plan": get_current_plan(),
        }

    raise ValueError(f"Unsupported state snapshot: {name}")


def list_tool_names() -> list[str]:
    return [tool.name for tool in get_all_base_tools()]
