# Steering + Undo: Design Spec

**Date:** 2026-06-20  
**Status:** Revised (v2)  
**Scope:** 5 new chat commands to prevent and recover from agent overengineering

---

## Problem

When using rooster-code chat, the agent frequently overengineers: refactors code you didn't mention, adds features you didn't request, creates new files unnecessarily. The only recovery is `/clear` or `/reset`, which loses all context and requires re-explaining from scratch.

## Root Cause

The agent lacks real-time constraints. Once a prompt is sent, the agent runs unchecked until it decides it's done. There's no mechanism to:
- Steer the agent mid-session with new instructions
- Constrain what the agent is allowed to do for a fixed number of turns
- Undo a bad turn without losing the whole conversation
- Restrict the agent to specific files
- Force minimal changes

---

## Design

### 1. `/narrow` — Toggle minimal-change mode

**What:** Injects a "minimal mode" system instruction: make the smallest possible change, no refactoring, no new files, no scope expansion.

**How:**
- Boolean flag `_narrow_mode: bool` (default False), stored as a local in `run_chat` scope
- When active, included in the managed sections rebuilt by `_rebuild_system_prompt`
- `/narrow` toggles the flag on/off
- Shows current state: "Minimal mode: ON" / "Minimal mode: OFF"
- On `/reset`, narrow mode is cleared (reset to OFF)
- On `/clear`, narrow mode persists (behavioral, not history)

**Minimal mode prompt:**
```
# Minimal Change Mode
Make the SMALLEST possible change to address the request.
- Do NOT refactor surrounding code
- Do NOT create new files
- Do NOT add features beyond what was asked
- Do NOT change code that isn't directly related to the request
- Prefer editing existing code over creating new code
```

**Usage:**
```
/narrow        # Toggle minimal mode
```

**Effort:** ~30min

---

### 2. `/system <text>` — Inject a persistent system instruction

**What:** Appends a system-level instruction to the agent's prompt. Persists until explicitly removed or session ends.

**How:**
- Store constraints in `_system_constraints: list[str]`, a local in `run_chat` scope
- On `/system <text>`, append text, call `_rebuild_system_prompt()`
- On `/system remove <N>`, remove by 1-based index (shown in listing), call `_rebuild_system_prompt()`
- On `/system clear`, remove all, call `_rebuild_system_prompt()`
- `/system` with no args shows current constraints with indices

**Usage:**
```
/system Do not create new files. Only modify files I explicitly mention.
/system Keep responses under 100 words
/system              # Show current constraints with indices
/system remove 2     # Remove constraint #2
/system clear         # Remove all constraints
```

**Effort:** ~1 hour

---

### 3. `/scope <files...>` — Restrict agent to specific files

**What:** Pins the agent to a set of files via a soft system-prompt constraint.

**How:**
- Store in `_scoped_files: set[Path] | None` (None = no restriction), local in `run_chat` scope
- On `/scope file1.py dir2/`, resolve paths to absolute paths relative to `config.cwd`
- Accepts files and directories — directories listed as `src/` in prompt, meaning "and any files under these directories"
- Non-existent paths produce a warning but are still added (agent may create them later)
- No glob support in v1 — explicit paths only
- `/scope clear` removes restriction, calls `_rebuild_system_prompt()`
- `/scope` with no args shows current scope
- On `/reset`, scope is cleared

**Interaction with existing code:**
- Soft constraint only — relies on system prompt, no tool-call interception
- If agent uses `@file` references outside scope, the reference resolver still works but the system prompt tells the agent not to
- Team dispatch creates child agents with their own `append_system_prompt` — they do NOT inherit scope
- `/scope` works best combined with `/narrow` or `/constrain`

**Usage:**
```
/scope src/bug.py src/utils.py    # Restrict to these files
/scope src/                        # Restrict to everything under src/
/scope                             # Show current scope
/scope clear                       # Remove restriction
```

**Effort:** ~1.5 hours

---

### 4. `/constrain <text> [--turns N]` — Add a temporary constraint

**What:** Like `/system`, but auto-expires after N turns (default 1). For one-off instructions like "don't create files for this one query."

**How:**
- Store in `_temporary_constraints: list[tuple[str, int]]` (text, turns remaining), local in `run_chat`
- Decrement counters only after normal user-initiated queries AND goal-loop turns
- `_run_query_with_interrupt` returns control to the main loop → after it returns, decrement
- `/goal check` queries do NOT decrement (they're read-only assessments)
- `/skill` queries do NOT decrement (they use separate agent instances)
- Named agent invocations via `/bg` do NOT decrement (separate agent, running in background)
- `/constrain clear` removes all temporary constraints
- `/constrain` with no args shows active constraints with remaining turns
- `/constrain remove <N>` removes a specific constraint by index
- Rebuild `append_system_prompt` after each decrement or removal
- Render notice when constraint expires: "Constraint expired: <text>"

**Edge cases:**
- `--turns 0` is invalid → error
- Multiple constraints with different expiry → each tracked independently
- After `/undo`, constraint counters are NOT un-decremented (documented behavior: user should re-add)
- After `/compact`, constraints are NOT decremented (compact is not an agent turn)

**Usage:**
```
/constrain Do not create new files                    # Expires after 1 turn
/constrain Only fix the bug, no refactoring --turns 3  # Expires after 3 turns
/constrain                                            # Show active constraints
/constrain remove 1                                    # Remove constraint #1
/constrain clear                                      # Remove all
```

**Effort:** ~2 hours

---

### 5. `/undo` — Drop last turn from history

**What:** Removes the last complete user turn from agent history, including all tool-use cycles. The agent forgets the entire exchange.

**How:**
- Walk `agent._history` backwards from the end
- Stop when we find a real user text message: a history entry with `role: "user"` whose `content` blocks are ALL `type: "text"` (no `type: "tool_result"`)
- Remove everything after (and including) that entry
- This removes: the user prompt, all assistant tool_use messages, all tool_result injections, and the final assistant response
- Do NOT remove injected context pairs from `_inject_goal_context` or `append_task_result_to_context` — remove them only if they belong to the undone turn
- After removal, call `_rebuild_system_prompt()` to ensure the prompt still matches the remaining history
- If the undone turn incremented a goal-loop counter, also decrement `_goal_loop_turns`
- Render notice: "Last turn undone. [N turns remaining in history]"

**Edge cases:**
- If history has < 2 entries with a real user text message → show "Nothing to undo"
- If `/undo` called twice → undoes the turn before the first undo
- Does NOT revert file changes on disk
- Does NOT un-decrement constraint counters (documented)
- After `/clear`, no history to undo → "Nothing to undo"
- Must handle empty history gracefully

**Usage:**
```
/undo          # Drop last turn
```

**Effort:** ~1.5 hours

---

## Shared Infrastructure

All five commands share a common mechanism for managing the agent's system prompt. The key decision: **do NOT use regex-based section stripping.** Instead, store a reference to the immutable base prompt at agent creation time and reconstruct the full prompt deterministically from source data on every change.

### `_rebuild_system_prompt`

```python
# At agent creation time, after _initialize():
_BASE_APPEND_PROMPT = agent._options.append_system_prompt or ""
# This captures the base prompt containing Tool Use Guidance, Configured Agents,
# Available Skills, and Saved Memories — before any goal/constraint sections.

def _rebuild_system_prompt() -> None:
    """Rebuild append_system_prompt from base + managed sections."""
    sections = []
    
    # 1. Goal section (existing, from build_goal_prompt_section)
    import re
    goal_section = build_goal_prompt_section() if get_active_goal() else ""
    if goal_section:
        sections.append(goal_section)
    
    # 2. Additional system constraints (persistent, from /system)
    if _system_constraints:
        lines = ["# Additional Constraints"]
        for i, c in enumerate(_system_constraints, 1):
            lines.append(f"  {i}. {c}")
        sections.append("\n".join(lines))
    
    # 3. Temporary constraints (auto-expiring, from /constrain)
    if _temporary_constraints:
        lines = ["# Temporary Constraints"]
        for text, turns in _temporary_constraints:
            expire = "expires after this turn" if turns == 1 else f"expires after {turns} turns"
            lines.append(f"  - {text} ({expire})")
        sections.append("\n".join(lines))
    
    # 4. File scope restriction (from /scope)
    if _scoped_files is not None:
        lines = ["# File Scope Restriction"]
        lines.append("You may ONLY read, edit, or create these files and directories:")
        for f in sorted(_scoped_files):
            label = str(f) + ("/" if f.is_dir() else "")
            lines.append(f"  - {label}")
        lines.append("Do NOT access, read, or modify any other files or directories.")
        sections.append("\n".join(lines))
    
    # 5. Minimal change mode (from /narrow)
    if _narrow_mode:
        sections.append(_MINIMAL_MODE_PROMPT)
    
    # Rebuild: base prompt + managed sections
    base = _BASE_APPEND_PROMPT
    existing_goal = build_goal_prompt_section()
    if existing_goal and existing_goal in base:
        base = base.replace(existing_goal + "\n", "").replace(existing_goal, "")
        base = base.rstrip()
    
    if sections:
        opts.append_system_prompt = (base + "\n\n" + "\n\n".join(sections)).strip()
    else:
        opts.append_system_prompt = base
```

This replaces the current `_update_agent_goal_prompt`. Each command sets its state variables and calls `_rebuild_system_prompt()`.

### State storage

All steering state is stored as **local variables in `run_chat` scope**, matching the existing pattern for `_goal_loop_active`, `_goal_loop_turns`, `_goal_stop`, `_injected_task_ids`:

```python
# New locals in run_chat()
_system_constraints: list[str] = []
_temporary_constraints: list[tuple[str, int]] = []
_scoped_files: set[Path] | None = None
_narrow_mode: bool = False

# At agent creation:
_BASE_APPEND_PROMPT = agent._options.append_system_prompt or ""
```

This avoids module-level state leaking across sessions or daemon requests.

### Call sites

| Action | Effect | Calls |
|---|---|---|
| `/narrow` | Toggle `_narrow_mode` | `_rebuild_system_prompt()` |
| `/system <text>` | Append to `_system_constraints` | `_rebuild_system_prompt()` |
| `/system remove <N>` | Remove by index | `_rebuild_system_prompt()` |
| `/system clear` | Clear `_system_constraints` | `_rebuild_system_prompt()` |
| `/scope <files>` | Set `_scoped_files` | `_rebuild_system_prompt()` |
| `/scope clear` | Set `_scoped_files = None` | `_rebuild_system_prompt()` |
| `/constrain <text>` | Append to `_temporary_constraints` | `_rebuild_system_prompt()` |
| `/constrain remove <N>` | Remove by index | `_rebuild_system_prompt()` |
| `/constrain clear` | Clear `_temporary_constraints` | `_rebuild_system_prompt()` |
| After each agent turn | Decrement `_temporary_constraints` | `_rebuild_system_prompt()` |
| `/undo` | Pop from history | N/A |
| `/reset` | Clear all state | `_rebuild_system_prompt()` |

---

## /clear vs /reset semantics

| | `/clear` | `/reset` |
|---|---|---|
| History | ✅ Cleared | ✅ Cleared |
| Steering state | ❌ Preserved | ✅ Cleared |
| Goal | ❌ Preserved | ✅ Cleared |
| Team | ❌ Preserved | ✅ Cleared |
| Tasks | ❌ Preserved | ✅ Cancelled |
| Agent client | ❌ Unchanged | ✅ Re-initialized |

`/clear` = reset history only, keep behavioral settings.  
`/reset` = full teardown to fresh state.

---

## Interaction with Existing Features

| Feature | Interaction |
|---|---|
| **`/goal work`** | Goal-loop turns decrement temporary constraints. `_rebuild_system_prompt` re-adds goal section alongside active constraints/scope/narrow. |
| **`/goal check`** | Does NOT decrement constraints. Check is read-only. |
| **`/clear`** | Preserves all steering state (system, constrain, scope, narrow). Calls `_rebuild_system_prompt()` after clearing to ensure prompt matches empty history. |
| **`/reset`** | Clears all steering state, calls `_rebuild_system_prompt()`. |
| **`/compact`** | Does NOT decrement constraints. Compact replaces history with summary but prompt state is unchanged. |
| **`/resume <id>`** | Creates a new agent from resumed session. Steering state is lost (local variables). Documented behavior — persistence is a future feature. |
| **`/team`** | Team dispatch creates child agents with their own `append_system_prompt`. They do NOT inherit scope/constrain/narrow from parent. |
| **`/bg` / `/agent-bg`** | Background agents are separate instances. Steering state does not apply. |

---

## Command Summary

| Command | What | Persists after /clear? | Persists after /reset? | Effort |
|---|---|---|---|---|
| `/narrow` | Toggle minimal-change mode | Yes | No | 30min |
| `/system <text>` | Persistent system instruction | Yes | No | 1h |
| `/scope <files...>` | Restrict to files | Yes | No | 1.5h |
| `/constrain <text> [--turns N]` | Temporary constraint | Yes | No | 2h |
| `/undo` | Drop last turn | N/A (no history) | N/A | 1.5h |

**Total estimated effort:** ~6.5 hours

---

## Implementation Order

1. **`_rebuild_system_prompt`** — Store `_BASE_APPEND_PROMPT` at agent creation. Replace `_update_agent_goal_prompt` with the unified builder. Test goal prompt still works.
2. **`/narrow`** — Simplest command, validates `_rebuild_system_prompt` end-to-end.
3. **`/system`** — Persistent constraints with add/remove/clear and listing.
4. **`/scope`** — File restriction with path resolution relative to `config.cwd`.
5. **`/constrain`** — Temporary constraints with turn tracking, decrement logic, expiry.
6. **`/undo`** — Most complex — proper history walk, tool-cycle detection, goal decrement.

---

## Testing Plan

Each command:
1. **Command parsing**: `/system`, `/system clear`, `/system remove 3`, `/constrain --turns 5`, `/scope ../foo.py`
2. **State management**: Toggle, add, remove, clear, list
3. **Prompt rebuilding**: Verify `append_system_prompt` contains correct managed sections after each operation
4. **Interaction tests**:
   - `/narrow` + `/scope` + `/constrain` all active simultaneously
   - `/clear` preserves state
   - `/reset` clears state
   - `/undo` after `/constrain` (constraint expires but undo doesn't restore it)
   - Goal loop + `/constrain` with `--turns` decrementing correctly
   - `/compact` does not decrement constraints
   - `/resume` loses steering state
5. **Edge cases**: Empty state, invalid indices, non-existent paths, zero turns

Key regression checks:
- `/goal` prompt still works
- `_rebuild_system_prompt` does not corrupt base prompt content
- All 611 existing tests pass + new tests

---

## Affected Files

| File | Changes |
|---|---|
| `cli.py` | Add 5 command dispatch handlers, `_rebuild_system_prompt`, 4 state variables, `_BASE_APPEND_PROMPT`, replace `_update_agent_goal_prompt`, `/undo` history walk, `/reset` clearing, decrement logic in main loop |
| `rendering.py` | Add 5 new help rows |
| `chat.py` | None — `parse_chat_command` handles generically; `--turns` parsed by handler |
| `runtime.py` | None — uses existing `append_system_prompt` API |
| `tests/` | New test file or additions to regression suite |

---

## Out of Scope

- **Hard file scoping** (tool-call interception) — soft prompt-based only
- **`/diff` or `/revert`** (file change revert) — separate feature
- **`/checkpoint` / `/rollback`** (session snapshots) — separate feature
- **`/trace`** (verbose tool logging) — visibility feature
- **Constraint persistence across `/resume`** — future feature
- **Changes to the SDK** — only use existing `append_system_prompt` and `_history` APIs
- **Thread safety for daemon multi-session** — current `run_chat` scoping is sufficient