from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from dataclasses import field
from typing import Any, Mapping


@dataclass(slots=True)
class RuntimeConfig:
    api_key: str | None = None
    base_url: str | None = None
    model: str | None = None
    api_type: str | None = None
    search_url: str | None = None
    cwd: str | None = None
    allowed_tools: list[str] | None = None
    disallowed_tools: list[str] | None = None
    resume: str | None = None
    session_id: str | None = None
    continue_session: bool = False
    fork_session: str | None = None
    persist_session: bool = True
    permission_mode: str = "bypassPermissions"
    max_turns: int | None = None
    max_budget_usd: float | None = None
    max_tokens: int | None = None
    thinking_budget: int | None = None
    debug: bool = False
    sandbox: bool = False
    include_partials: bool = False
    env: dict[str, str] = field(default_factory=dict)
    custom_headers: dict[str, str] = field(default_factory=dict)
    agents: dict[str, Any] = field(default_factory=dict)
    hooks: dict[str, Any] = field(default_factory=dict)
    json_schema: dict[str, Any] | None = None
    mcp_servers: dict[str, Any] = field(default_factory=dict)
    skills_dir: str | None = None
    extra_args: dict[str, Any] = field(default_factory=dict)


def load_dotenv_env(cwd: str | None = None) -> dict[str, str]:
    path = os.path.join(cwd, ".env") if cwd else ".env"
    if not os.path.exists(path):
        return {}

    values: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            parsed = value.strip()
            if len(parsed) >= 2 and parsed[0] == parsed[-1] and parsed[0] in {'"', "'"}:
                parsed = parsed[1:-1]
            values[key.strip()] = parsed

    return values


def resolve_runtime_env(env: Mapping[str, str], cwd: str | None = None) -> RuntimeConfig:
    merged_env = {**load_dotenv_env(cwd), **env}
    return RuntimeConfig(
        api_key=merged_env.get("COCK_CODE_API_KEY"),
        base_url=merged_env.get("COCK_CODE_BASE_URL"),
        model=merged_env.get("COCK_CODE_MODEL"),
        api_type=merged_env.get("COCK_CODE_API_TYPE"),
        search_url=merged_env.get("COCK_CODE_SEARCH_URL"),
    )


def parse_key_value_pairs(values: list[str] | None) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for value in values or []:
        key, raw = value.split("=", 1)
        pairs[key] = raw
    return pairs


def load_json_file(path: str | None) -> Any:
    if not path:
        return None

    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def config_from_namespace(args: argparse.Namespace, env: Mapping[str, str]) -> RuntimeConfig:
    resolved = resolve_runtime_env(env, cwd=getattr(args, "cwd", None))
    return RuntimeConfig(
        api_key=resolved.api_key,
        base_url=resolved.base_url,
        model=args.model or resolved.model,
        api_type=resolved.api_type,
        search_url=getattr(args, "search_url", None) or resolved.search_url,
        cwd=args.cwd,
        allowed_tools=getattr(args, "allowed_tools", None),
        disallowed_tools=getattr(args, "disallowed_tools", None),
        resume=args.resume,
        session_id=getattr(args, "session_id", None),
        continue_session=getattr(args, "continue_session", False),
        fork_session=getattr(args, "fork_session", None),
        persist_session=getattr(args, "persist_session", True),
        permission_mode=getattr(args, "permission_mode", None) or "bypassPermissions",
        max_turns=getattr(args, "max_turns", None),
        max_budget_usd=getattr(args, "max_budget_usd", None),
        max_tokens=getattr(args, "max_tokens", None),
        thinking_budget=getattr(args, "thinking_budget", None),
        debug=getattr(args, "debug", False),
        sandbox=getattr(args, "sandbox", False),
        include_partials=getattr(args, "include_partials", False),
        env=parse_key_value_pairs(getattr(args, "env", None)),
        custom_headers=parse_key_value_pairs(getattr(args, "custom_headers", None)),
        agents=load_json_file(getattr(args, "agents_file", None)) or {},
        hooks=load_json_file(getattr(args, "hooks_file", None)) or {},
        json_schema=load_json_file(getattr(args, "json_schema_file", None)),
        mcp_servers=load_json_file(getattr(args, "mcp_file", None)) or {},
        skills_dir=getattr(args, "skills_dir", None),
        extra_args=load_json_file(getattr(args, "extra_args_file", None)) or {},
    )
