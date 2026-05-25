from __future__ import annotations

import argparse
import asyncio
import json
import sys

from rooster_code.daemon import (
    daemon_health, daemon_list_sessions, daemon_session_info, daemon_shutdown,
    daemon_query, _SOCKET_PATH,
)

_CONNECTION_ERRORS = (ConnectionRefusedError, FileNotFoundError, OSError)


def _daemon_is_reachable() -> bool:
    try:
        result = asyncio.run(daemon_health())
        return result.get("type") == "health"
    except _CONNECTION_ERRORS:
        return False
    except Exception:
        return False


def _print_error(msg: str) -> None:
    print(f"Error: {msg}", file=sys.stderr)


def _ask_via_daemon(prompt: str, args: argparse.Namespace) -> int:
    overrides: dict[str, object] = {}
    _FLAG_OVERRIDES: list[tuple[str, str]] = [
        ("model", "model"),
        ("max_turns", "max_turns"),
        ("max_tokens", "max_tokens"),
        ("permission_mode", "permission_mode"),
        ("thinking_budget", "thinking_budget"),
        ("max_budget_usd", "max_budget_usd"),
        ("sandbox", "sandbox"),
        ("debug", "debug"),
        ("include_partials", "include_partials"),
    ]
    for attr, key in _FLAG_OVERRIDES:
        val = getattr(args, attr, None)
        if val is not None:
            overrides[key] = val
    if getattr(args, "allowed_tools", None):
        overrides["allowed_tools"] = args.allowed_tools
    if getattr(args, "disallowed_tools", None):
        overrides["disallowed_tools"] = args.disallowed_tools
    if getattr(args, "env", None):
        env_dict: dict[str, str] = {}
        for pair in args.env:
            if "=" in pair:
                k, v = pair.split("=", 1)
                env_dict[k] = v
        if env_dict:
            overrides["env"] = env_dict
    if getattr(args, "custom_headers", None):
        headers_dict: dict[str, str] = {}
        for pair in args.custom_headers:
            if "=" in pair:
                k, v = pair.split("=", 1)
                headers_dict[k] = v
        if headers_dict:
            overrides["custom_headers"] = headers_dict

    async def _run() -> int:
        session_id = args.resume or args.session_id or ""
        result = await daemon_query(prompt, session_id=session_id, cwd=args.cwd or ".", overrides=overrides if overrides else None)
        if result["type"] == "done":
            print(result.get("text", ""))
            return 0
        print(f"Error: {result.get('message', 'unknown error')}", file=sys.stderr)
        return 1

    return asyncio.run(_run())


def _handle_daemon_command(args: argparse.Namespace) -> int:
    async def _status() -> int:
        try:
            r = await daemon_health()
        except _CONNECTION_ERRORS:
            _print_error(f"daemon not reachable at {_SOCKET_PATH}")
            return 1
        except Exception as exc:
            _print_error(f"daemon health check failed: {exc}")
            return 1
        print(f"status:       {r['status']}")
        print(f"uptime:       {r['uptime_seconds']}s")
        print(f"queries:      {r['queries']} (avg {r['avg_latency_ms']}ms)")
        print(f"tokens:       {r.get('total_tokens', 0)}")
        print(f"cost:         ${r.get('total_cost', 0):.4f}")
        print(f"sessions:     {r['sessions']} (max {r['max_sessions']})")
        print(f"adapters:     {json.dumps(r['adapters'])}")
        return 0

    async def _sessions() -> int:
        try:
            r = await daemon_list_sessions()
        except _CONNECTION_ERRORS:
            _print_error(f"daemon not reachable at {_SOCKET_PATH}")
            return 1
        except Exception as exc:
            _print_error(f"failed to list sessions: {exc}")
            return 1
        data = r.get("data", [])
        if not data:
            print("No sessions tracked.")
            return 0
        for s in data:
            print(f"  {s['session_id']}  (cwd={s['cwd']})")
        return 0

    async def _shutdown() -> int:
        try:
            await daemon_shutdown()
            print("daemon shutdown initiated")
            return 0
        except _CONNECTION_ERRORS:
            _print_error(f"daemon not reachable at {_SOCKET_PATH}")
            return 1
        except Exception as exc:
            _print_error(f"failed to shutdown daemon: {exc}")
            return 1

    async def _session() -> int:
        sid = getattr(args, "session_id", "")
        if not sid:
            print("Error: session_id required", file=sys.stderr)
            return 1
        try:
            r = await daemon_session_info(sid)
        except _CONNECTION_ERRORS:
            _print_error(f"daemon not reachable at {_SOCKET_PATH}")
            return 1
        except Exception as exc:
            _print_error(f"failed to get session info: {exc}")
            return 1
        local = r.get("local") or {}
        sdk = r.get("sdk") or {}
        print(f"session: {sid}")
        print(f"  tracked:  created={local.get('created_at')} last_active={local.get('last_active_at')}")
        if sdk:
            print(f"  sdk:      title={sdk.get('title', '-')} tags={sdk.get('tags', [])} messages={sdk.get('messageCount', '-')}")
        return 0

    cmd = args.daemon_command
    if cmd == "status":
        return asyncio.run(_status())
    if cmd == "sessions":
        return asyncio.run(_sessions())
    if cmd == "session":
        return asyncio.run(_session())
    if cmd == "shutdown":
        return asyncio.run(_shutdown())
    return 1
