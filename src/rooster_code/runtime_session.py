from __future__ import annotations

import contextlib
import logging
import re
from pathlib import Path
from typing import Any, cast

from open_agent_sdk import (
    get_all_base_tools,
    get_all_cron_jobs,
    get_all_tasks,
    get_all_teams,
    get_config,
    get_current_plan,
    get_todos,
    is_plan_mode_active,
)
from open_agent_sdk.providers import CreateMessageParams

_ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
_TOOL_CALL_ARTIFACT_RE = re.compile(r"^[a-z][\w-]*:tool_call\b.*$", re.MULTILINE | re.IGNORECASE)

_log = logging.getLogger("rooster.session")


def _read_cron_jobs() -> dict[str, dict[str, Any]]:
    """Read cron jobs from the daemon SQLite store, falling back to the SDK in-memory dict."""
    db_path = Path.home() / ".rooster-code" / "daemon.db"
    if db_path.exists():
        try:
            import sqlite3
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM cron_jobs").fetchall()
            conn.close()
            if rows:
                return {row["job_id"]: dict(row) for row in rows}
        except Exception:
            pass
    return get_all_cron_jobs()


def _sdk():
    """Late-import SDK session bindings from runtime.py so monkeypatches apply."""
    import rooster_code.runtime as _rt
    return _rt


# ---------------------------------------------------------------------------
# text helpers
# ---------------------------------------------------------------------------


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
        if text and not re.match(r"^[a-z][\w-]*:tool_call\b", text, re.IGNORECASE):
            parts.append(text)
    return parts


def sanitize_task_output(output: str) -> str:
    output = _ANSI_ESCAPE_RE.sub("", output)
    output = _TOOL_CALL_ARTIFACT_RE.sub("", output)
    return output


def _collect_assistant_text(messages: list[dict[str, Any]], *, last_only: bool = False) -> list[str]:
    parts: list[str] = []
    for message in messages:
        if str(message.get("role", "")) != "assistant":
            continue
        parts.extend(_extract_text_blocks(message))
    if last_only and parts:
        return [parts[-1]]
    return parts


def _extract_tool_result_blocks(message: dict[str, Any]) -> list[str]:
    content = message.get("content", [])
    if not isinstance(content, list):
        return []
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "tool_result":
            continue
        result_content = block.get("content", "")
        if isinstance(result_content, str):
            text = result_content.strip()
            if text:
                parts.append(text)
            continue
        if not isinstance(result_content, list):
            continue
        for nested in result_content:
            if not isinstance(nested, dict) or nested.get("type") != "text":
                continue
            text = str(nested.get("text", "")).strip()
            if text:
                parts.append(text)
    return parts


def _collect_tool_result_text(messages: list[dict[str, Any]]) -> list[str]:
    parts: list[str] = []
    for message in messages:
        if str(message.get("role", "")) != "user":
            continue
        parts.extend(_extract_tool_result_blocks(message))
    return parts


def _format_subagent_task_output(result_text: str, messages: list[dict[str, Any]]) -> str:
    raw_text = sanitize_task_output(result_text.strip())
    if raw_text:
        return raw_text
    assistant_text = "\n\n".join(part.strip() for part in _collect_assistant_text(messages, last_only=False) if part.strip())
    if assistant_text:
        return sanitize_task_output(assistant_text)
    tool_result_text = "\n\n".join(part.strip() for part in _collect_tool_result_text(messages) if part.strip())
    if tool_result_text:
        return sanitize_task_output(tool_result_text)
    return "Agent completed with no text output."


def _format_subagent_summary(result_text: str, messages: list[dict[str, Any]]) -> str:
    outcomes: list[str] = []
    files: list[str] = []
    commands: list[str] = []
    open_issues: list[str] = []
    next_steps: list[str] = []
    findings: list[str] = []
    has_outcome = False
    has_structured_fields = False

    for message in messages:
        if str(message.get("role", "")) != "assistant":
            continue
        for text in _extract_text_blocks(message):
            for raw_line in text.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                lower = line.lower()
                if lower.startswith("outcome:"):
                    value = line.partition(":")[2].strip()
                    if value:
                        outcomes.append(value)
                        has_outcome = True
                elif lower.startswith("files:"):
                    files.append(line.partition(":")[2].strip())
                    has_structured_fields = True
                elif lower.startswith("commands:"):
                    commands.append(line.partition(":")[2].strip())
                    has_structured_fields = True
                elif lower.startswith("findings:"):
                    value = line.partition(":")[2].strip()
                    if value:
                        findings.append(value)
                    has_structured_fields = True
                elif lower.startswith("open issues:"):
                    open_issues.append(line.partition(":")[2].strip())
                    has_structured_fields = True
                elif lower.startswith("next step:"):
                    next_steps.append(line.partition(":")[2].strip())
                    has_structured_fields = True

    # Fallback: last meaningful line from raw text, then from last assistant message
    fallback = "No useful output returned"
    fallback_from_result = "No useful output returned"
    for line in reversed(result_text.strip().splitlines()):
        line = line.strip()
        if line and line not in {"---", "***"} and not re.fullmatch(r"#+", line):
            fallback = line
            fallback_from_result = line
            break
    if fallback == "No useful output returned":
        for message in reversed(messages):
            if str(message.get("role", "")) != "assistant":
                continue
            for text in reversed(_extract_text_blocks(message)):
                for line in reversed(text.splitlines()):
                    line = line.strip()
                    if line and line not in {"---", "***"} and not re.fullmatch(r"#+", line):
                        fallback = line
                        break
                if fallback != "No useful output returned":
                    break
            if fallback != "No useful output returned":
                break

    if not has_outcome and has_structured_fields:
        normalized_fallback = fallback_from_result.lower()
        if fallback_from_result != "No useful output returned" and not normalized_fallback.startswith(
            ("files:", "commands:", "findings:", "open issues:", "next step:", "outcome:")
        ):
            outcomes.append(fallback_from_result)
            has_outcome = True

    if not has_outcome:
        return ""

    lines = [f"Outcome: {'; '.join(outcomes) if outcomes else fallback}"]
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


# ---------------------------------------------------------------------------
# compaction
# ---------------------------------------------------------------------------


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
    before_tokens = _sdk().estimate_messages_tokens(compactable_history)

    if len(compactable_history) < 2:
        return {
            "compacted": False,
            "summary": "",
            "before_tokens": before_tokens,
            "after_tokens": before_tokens,
            "reason": "Need at least two messages before compaction.",
        }

    pre_compaction_history = list(agent._history)
    try:
        summary, compacted_history = await _compact_with_provider(agent, compactable_history)
        after_tokens = _sdk().estimate_messages_tokens(compacted_history)
    except Exception:
        agent._history = pre_compaction_history
        raise

    agent._history = compacted_history
    compacted = after_tokens < before_tokens
    return {
        "compacted": compacted,
        "summary": summary,
        "before_tokens": before_tokens,
        "after_tokens": after_tokens,
        "reason": "" if compacted else "Compaction produced no smaller history.",
    }


# ---------------------------------------------------------------------------
# session management
# ---------------------------------------------------------------------------


async def list_sessions():
    return await _sdk().sdk_list_sessions()


async def get_session_messages(session_id: str):
    return await _sdk().sdk_get_session_messages(session_id)


async def get_session_info(session_id: str):
    return await _sdk().sdk_get_session_info(session_id)


async def delete_session(session_id: str):
    return await _sdk().sdk_delete_session(session_id)


async def enforce_session_retention(limit: int = 20) -> None:
    sessions = await _sdk().sdk_list_sessions()
    if len(sessions) <= limit:
        return

    for session in sessions[limit:]:
        session_id = session.get("id")
        if isinstance(session_id, str) and session_id:
            with contextlib.suppress(OSError, PermissionError):
                await _sdk().sdk_delete_session(session_id)


async def fork_session(session_id: str, new_id: str | None):
    return await _sdk().sdk_fork_session(session_id, new_id)


async def rename_session(session_id: str, title: str):
    await _sdk().sdk_rename_session(session_id, title)


async def tag_session(session_id: str, tags: list[str]):
    await _sdk().sdk_tag_session(session_id, tags)


def get_state_snapshot(name: str, agent_name: str | None = None):
    from rooster_code.team import get_runtime_team_bridge

    if name == "todos":
        return get_todos()
    if name == "tasks":
        return get_all_tasks()
    if name == "teams":
        team_manager, _ = get_runtime_team_bridge()
        snapshot: dict[str, Any] = dict(get_all_teams())
        if team_manager is not None:
            snapshot.update(team_manager.sdk_team_snapshot())
        from rooster_code.daemon import read_team_snapshots
        persisted = read_team_snapshots()
        if persisted:
            snapshot["_snapshots"] = persisted
        return snapshot
    if name == "mailboxes":
        from open_agent_sdk.tools import _mailboxes as _sdk_mailboxes
        team_manager, _ = get_runtime_team_bridge()
        snapshot = {name: [dict(message) for message in messages] for name, messages in _sdk_mailboxes.items()}
        if team_manager is not None:
            for member_name, messages in team_manager.sdk_mailboxes_snapshot().items():
                snapshot.setdefault(member_name, []).extend(messages)
        if agent_name is not None:
            return snapshot.get(agent_name, [])
        return snapshot
    if name == "config":
        return get_config()
    if name == "cron":
        return _read_cron_jobs()
    if name == "plan":
        return {
            "active": is_plan_mode_active(),
            "plan": get_current_plan(),
        }

    raise ValueError(f"Unsupported state snapshot: {name}")


def list_tool_names() -> list[str]:
    return [tool.name for tool in get_all_base_tools()]
