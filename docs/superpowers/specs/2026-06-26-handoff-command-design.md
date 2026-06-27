# Handoff Command Design

## Goal

Add a manual `/handoff` chat command to `rooster-code` that creates a compact continuation checkpoint for a new session while also compacting the current session in place.

## Background

Current `rooster-code` continuation mechanisms are:

- `--resume` and `/resume <id>`: load the full SDK transcript from `~/.open-agent-sdk/sessions/<session-id>/transcript.json`.
- `/compact`: summarize the current transcript and replace `agent._history` with a two-message compacted context.
- `sessions fork`: copy a persisted SDK session directory to a new session ID.

Research into modern agent CLIs showed the common split:

- Native resume keeps full local transcripts.
- Compact reduces active context.
- Export or handoff files provide a portable, human-readable checkpoint for a fresh session.

`rooster-code` already has the hard part: `_build_manual_compaction_summary_prompt()` asks the provider for a structured continuation summary with Goal, Current State, Key Decisions, Code/Files, Constraints, Blockers, Next Step, and Transcript sections. `/handoff` should reuse that path instead of inventing a second summarizer.

## User Decisions

- `/handoff` is manual only. No token-threshold automation.
- `/handoff` compacts in place. No session fork.
- `/handoff` writes a local handoff file and replaces `agent._history` with the compacted summary.
- The handoff file should be local and easy for another session to read. Default file path is `.handoff` in the active working directory.
- `.handoff` is a local checkpoint artifact and should be ignored by git.
- No SDK code changes.

## Proposed UX

Inside chat:

```text
/handoff
```

Expected result:

```text
Handoff
Saved .handoff
Tokens: 1200 → 240
```

A later session can continue with:

```text
Read .handoff and continue from the Next Step section. Verify the current repo state before acting.
```

`/compact` remains available for users who want only in-memory compaction and no file.

## File Format

The `.handoff` file is Markdown text without a `.md` extension so it behaves like a local agent checkpoint, not project documentation.

Template:

```markdown
# Handoff

Generated: 2026-06-26T12:34:56
Session: <session-id-or-new>
Model: <model-or-default>
CWD: <working-directory>

## Resume Prompt

Read this `.handoff`, inspect the current repository state, verify whether the described state is still accurate, then continue from the `## Next Step` section. Do not assume this handoff is fully up to date.

---

<provider-generated structured summary>
```

The provider-generated summary is exactly the summary returned by the existing manual compaction path. It should already contain:

- `## Goal`
- `## Current State`
- `## Key Decisions`
- `## Code/Files`
- `## Constraints / What to Avoid`
- `## Blockers / Open Questions`
- `## Next Step`
- `## Transcript`

## Architecture

### Runtime layer

Add a helper in `src/rooster_code/runtime_session.py`:

```python
async def handoff_current_session(agent, path: str | Path | None = None) -> dict[str, object]:
    """Compact the current session and write a local handoff file."""
```

Responsibilities:

1. Initialize the agent if needed.
2. Filter history with `_filter_history_for_manual_compaction()`.
3. Estimate `before_tokens`.
4. If fewer than two compactable messages exist, return this shape without writing a file:

```python
{
    "compacted": False,
    "written": False,
    "path": "",
    "summary": "",
    "before_tokens": before_tokens,
    "after_tokens": before_tokens,
    "reason": "Need at least two messages before compaction.",
}
```

5. Call `_compact_with_provider()` once.
6. Estimate `after_tokens`.
7. Write the handoff file to `Path(path)` or `Path(".handoff")` if called directly without a path.
8. Replace `agent._history` with `compacted_history` only after the file write succeeds.
9. Return a result dictionary containing:

```python
{
    "compacted": after_tokens < before_tokens,
    "written": True,
    "path": str(resolved_path),
    "summary": summary,
    "before_tokens": before_tokens,
    "after_tokens": after_tokens,
    "reason": "" if after_tokens < before_tokens else "Compaction produced no smaller history.",
}
```

If provider compaction or file writing fails, restore the original `agent._history` before raising. This preserves the existing `/compact` safety invariant.

### CLI layer

In `src/rooster_code/cli.py`, add:

```python
async def handoff_current_session(agent, path: str | None = None):
    from rooster_code.runtime import handoff_current_session as runtime_handoff_current_session
    return await runtime_handoff_current_session(agent, path)
```

Add command handling in the chat loop:

```python
    path = str(Path(config.cwd or ".") / command.args[0]) if command.args else str(Path(config.cwd or ".") / ".handoff")
    try:
        result = await handoff_current_session(agent, path)
    except Exception as exc:
        render_notice(console, "Handoff Error", str(exc), "red")
        continue
    title = "Handoff" if result["written"] else "Handoff skipped"
    style = "green" if result["written"] else "yellow"
    detail = f"Saved {result['path']}\nTokens: {result['before_tokens']} → {result['after_tokens']}" if result["written"] else str(result["reason"])
    render_notice(console, title, detail, style)
    continue
```

With no argument, the CLI resolves the default path as `Path(config.cwd or ".") / ".handoff"` and passes that concrete path to `handoff_current_session()`. This makes the output follow the agent's active working directory even when the process was launched elsewhere.

With an argument, relative paths are resolved against the agent working directory (`config.cwd`), matching the behavior of other path-taking commands like `/memory`. Absolute paths are passed through unchanged:

```text
/handoff .handoff
/handoff /tmp/rooster-handoff.md
/handoff docs/session-handoff.md
```

This ensures the handoff file lands in the project directory even when the process CWD differs from `config.cwd` (e.g. `rooster-code chat --cwd /some/project`).

### Help text

Add `/handoff [path]` to `render_help()` in `src/rooster_code/rendering.py`.

## Error Handling

- Provider failure: show `Handoff Error`, restore original `agent._history`, write no file.
- Empty summary: `_compact_with_provider()` already raises `Compaction produced an empty summary.` Keep that behavior.
- Not enough history: show `Handoff skipped` with `Need at least two messages before compaction.`, write no file.
- File write failure: show `Handoff Error`, restore original `agent._history`, surface the file error.
- Path parent directories: create parent directories with `mkdir(parents=True, exist_ok=True)`.

## Testing

Add focused tests without real provider calls.

Runtime tests in `tests/test_runtime.py`:

1. `test_handoff_current_session_writes_file_and_compacts_history`
   - Fake agent history with one user and one assistant text message.
   - Fake provider returns `summary`.
   - Call `runtime.handoff_current_session(agent, tmp_path / ".handoff")`.
   - Assert file exists and contains `# Handoff`, `## Resume Prompt`, and `summary`.
   - Assert `agent._history` equals compacted two-message history.
   - Assert result has `written=True`, `compacted=True`, and path.

2. `test_handoff_current_session_restores_history_when_file_write_fails`
   - Use a path whose parent is an existing file so write fails.
   - Assert the exception is raised.
   - Assert `agent._history` is unchanged.

3. `test_handoff_current_session_skips_small_history`
   - One compactable message.
   - Assert `written=False`, `compacted=False`, empty path, and reason.

CLI tests in `tests/test_chat.py`:

1. `test_run_chat_handoff_writes_default_file`
   - Fake `/handoff`, `/exit` prompts.
   - Monkeypatch `cli.handoff_current_session` to capture the concrete path and return success.
   - Assert path argument is `str(Path(config.cwd) / ".handoff")`.
   - Assert render notice contains `Saved` and token delta.

2. `test_run_chat_handoff_accepts_path_argument`
   - Prompt `/handoff custom.handoff`.
   - Assert path argument is `custom.handoff`.

3. `test_run_chat_handoff_shows_error_when_handoff_fails`
   - Fake handoff raises `RuntimeError("Handoff failed")`.
   - Assert notice is `("Handoff Error", "Handoff failed", "red")`.

Rendering test in `tests/test_rendering.py`:

- Assert help output contains `/handoff [path]`.

## Non-Goals

- No automatic threshold-based handoff.
- No session fork.
- No changes to SDK internals.
- No second LLM call dedicated only to file formatting.
- No JSON handoff format in the first implementation.
- No new config file setting for the path in the first implementation.

## Acceptance Criteria

- `/handoff` in chat writes `.handoff` by default and compacts current history.
- `/handoff <path>` writes to the requested path and compacts current history.
- Failed provider or file write leaves the original history intact.
- `/compact` behavior remains unchanged.
- Help lists `/handoff [path]`.
- `.handoff` is listed in `.gitignore` so local handoff files are not accidentally committed.
- Focused tests pass with `uv run pytest tests/test_runtime.py tests/test_chat.py tests/test_rendering.py`.
