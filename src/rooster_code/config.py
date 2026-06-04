from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any, Mapping

DEFAULT_AGENTS_PATH = Path.home() / ".rooster-code" / "agents.json"


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
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError:
        return {}

    try:
        values: dict[str, str] = {}
        with open(fd, "r", encoding="utf-8", closefd=False) as handle:
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
    finally:
        os.close(fd)


def resolve_runtime_env(env: Mapping[str, str], cwd: str | None = None) -> RuntimeConfig:
    merged_env = {**load_dotenv_env(cwd), **env}
    return RuntimeConfig(
        api_key=merged_env.get("ROOSTER_CODE_API_KEY"),
        base_url=merged_env.get("ROOSTER_CODE_BASE_URL"),
        model=merged_env.get("ROOSTER_CODE_MODEL"),
        api_type=merged_env.get("ROOSTER_CODE_API_TYPE"),
        search_url=merged_env.get("ROOSTER_CODE_SEARCH_URL"),
    )


def parse_key_value_pairs(values: list[str] | None) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for value in values or []:
        if "=" not in value:
            continue
        key, raw = value.split("=", 1)
        pairs[key] = raw
    return pairs


def load_json_file(path: str | None) -> Any:
    if not path:
        return None

    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError as e:
        print(f"rooster-code: cannot load {path}: {e}", file=sys.stderr)
        return {}
    except OSError:
        return {}
    try:
        with open(fd, "r", encoding="utf-8", closefd=False) as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"rooster-code: cannot load {path}: {e}", file=sys.stderr)
        return {}
    finally:
        os.close(fd)


def save_json_file(path: str, data: Any) -> None:
    """Save data as JSON to a file atomically, creating parent directories as needed."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(file_path.parent), suffix=".tmp")
    try:
        # Acquire exclusive advisory lock to prevent concurrent writes
        lock_file = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o644)
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2)
            os.replace(tmp_path, file_path)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
            os.close(lock_file)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


def save_agents_file(agents: dict[str, Any]) -> None:
    """Persist agent definitions to the default agents file."""
    save_json_file(str(DEFAULT_AGENTS_PATH), agents)


def _resolve_agents(agents_file: str | None) -> dict[str, Any]:
    """Resolve agent definitions, preferring explicit --agents-file over default."""
    if agents_file:
        return load_json_file(agents_file) or {}
    if DEFAULT_AGENTS_PATH.exists():
        return load_json_file(str(DEFAULT_AGENTS_PATH)) or {}
    return {}


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
        agents=_resolve_agents(getattr(args, "agents_file", None)),
        hooks=load_json_file(getattr(args, "hooks_file", None)) or {},
        json_schema=load_json_file(getattr(args, "json_schema_file", None)),
        mcp_servers=load_json_file(getattr(args, "mcp_file", None)) or {},
        skills_dir=getattr(args, "skills_dir", None),
        extra_args=load_json_file(getattr(args, "extra_args_file", None)) or {},
    )
