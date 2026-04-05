from __future__ import annotations

from dataclasses import replace
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
    list_sessions as sdk_list_sessions,
    rename_session as sdk_rename_session,
    tag_session as sdk_tag_session,
)
from open_agent_sdk.providers import CreateMessageParams
from open_agent_sdk.tools import _mailboxes

from cock_code.config import RuntimeConfig
from cock_code.runtime_tools import RuntimeAgentTool, RuntimeEditTool, RuntimeReadTool, TurnTracker


def _effective_agents(config: RuntimeConfig) -> dict[str, Any]:
    if config.agents:
        return config.agents
    return {
        "task": {
            "description": "General read-only task agent",
            "prompt": "You are a careful read-only task agent. Use only safe inspection tools and answer briefly.",
            "tools": ["Read", "Grep", "Glob"],
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
    return "\n".join(lines)


async def _run_subagent(config: RuntimeConfig, input: dict[str, Any], context: ToolContext) -> ToolResult:
    prompt = str(input.get("prompt", "")).strip()
    if not prompt:
        return ToolResult(tool_use_id="", content="Error: prompt is required", is_error=True)

    if input.get("run_in_background"):
        return ToolResult(tool_use_id="", content="Error: background agents are not supported in cock-code yet", is_error=True)

    agent_name, definition = _resolve_agent_definition(config, input)
    if definition is None:
        if _effective_agents(config):
            return ToolResult(tool_use_id="", content=f"Error: unknown agent '{agent_name or 'unspecified'}'", is_error=True)
        return ToolResult(tool_use_id="", content="Error: no agents configured", is_error=True)

    child_config = _build_subagent_config(config, definition, input, context)
    system_prompt = str(definition.get("prompt") or definition.get("system_prompt") or definition.get("description") or "")
    child_agent = _create_sdk_agent(child_config, include_runtime_agent_tool=False, system_prompt=system_prompt)
    try:
        result = await child_agent.prompt(prompt)
    finally:
        await child_agent.close()

    text = result.text.strip() if result.text else ""
    return ToolResult(tool_use_id="", content=text or f"Agent {agent_name} completed with no text output.")


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


def _runtime_tools(config: RuntimeConfig, include_runtime_agent_tool: bool) -> list[Any]:
    return []


def build_agent_options(
    config: RuntimeConfig,
    *,
    include_runtime_agent_tool: bool = True,
    system_prompt: str = "",
) -> AgentOptions:
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
            async for event in original_query(prompt, overrides):
                yield event

        setattr(agent, "query", wrapped_query)

    if include_runtime_agent_tool and _effective_agents(config) and hasattr(agent, "_initialize"):
        original_initialize = agent._initialize

        async def wrapped_initialize() -> None:
            await original_initialize()
            replaced = False
            new_pool = []
            for tool in getattr(agent, "_tool_pool", []):
                if getattr(tool, "name", "") == "Agent" and not replaced:
                    new_pool.append(RuntimeAgentTool(lambda input, context: _run_subagent(config, input, context)))
                    replaced = True
                elif getattr(tool, "name", "") == "Read":
                    new_pool.append(RuntimeReadTool(tool, tracker))
                elif getattr(tool, "name", "") == "Edit":
                    new_pool.append(RuntimeEditTool(tool, tracker))
                elif getattr(tool, "name", "") != "Agent":
                    new_pool.append(tool)
            if not replaced:
                new_pool.append(RuntimeAgentTool(lambda input, context: _run_subagent(config, input, context)))
            agent._tool_pool = new_pool

        setattr(agent, "_initialize", wrapped_initialize)
    elif hasattr(agent, "_initialize"):
        original_initialize = agent._initialize

        async def wrapped_initialize_without_agent() -> None:
            await original_initialize()
            agent._tool_pool = [tool for tool in getattr(agent, "_tool_pool", []) if getattr(tool, "name", "") != "Agent"]

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
        "Summarize the following conversation concisely, "
        "preserving key decisions, code changes, and context needed to continue:\n\n"
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
        if agent_name is not None:
            return _mailboxes.get(agent_name, [])
        return _mailboxes
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
