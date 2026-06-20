# Steering + Undo: Design Spec

**Date:** 2026-06-20  
**Status:** Draft  
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

## Design

### 1. `/undo` — Drop last turn from history

**What:** Removes the last user-assistant exchange from agent history. The agent forgets the last turn entirely.

**How:**
- Pop the last 2 entries from `agent._history` (assistant message + user message)
- If the last entry is a system-injected message (background task result, goal context), keep popping until we find a real user-assistant pair
- Clear any related `_injected_task_ids` entries for tasks from that turn
- Render notice: "Last turn undone. [N turns remaining in history]"

**Edge cases:**
- If history has < 2 messages → show "Nothing to undo"
- If `/undo` called twice → undoes the turn before the first undo
- Does NOT revert file changes on disk — only removes the turn from conversation context
- After `/undo`, the agent won't know it made the previous change, so you may need to manually revert files

**Usage:**
```
/undo          # Drop last turn
```

**Effort:** ~15min

---

### 2. `/system <text>` — Inject a mid-session system message

**What:** Appends a system-level instruction to the agent's `append_system_prompt`. Persists until cleared or session ends.

**How:**
- Follow the pattern from `_update_agent_goal_prompt` (cli.py:455)
- Store constraints in a module-level list `_system_constraints: list[str]`
- On `/system <text>`, append the text to `_system_constraints`
- Rebuild `agent._options.append_system_prompt` with: goal section + constraint section
- Constraint section formatted as:
  ```
  # Additional Constraints
  - <constraint 1>
  - <constraint 2>
  ```
- `/system` with no args shows current constraints
- `/system clear` removes all constraints

**Interaction with existing code:**
- Must compose with `_update_agent_goal_prompt` — constraints go after the goal section
- On `/reset`, constraints are cleared (they live in `run_chat` scope like `_goal_loop_active`)
- On `/clear`, constraints persist (they're not history, they're behavioral)

**Usage:**
```
/system Do not create new files. Only modify files I explicitly mention.
/system Keep responses under 100 words
/system              # Show current constraints
/system clear         # Remove all constraints
```

**Effort:** ~30min

---

### 3. `/constrain <text> [--turns N]` — Add a temporary constraint

**What:** Like `/system`, but auto-expires after N turns (default 1). For one-off instructions like "don't create files for this one query."

**How:**
- Store in `_temporary_constraints: list[tuple[str, int]]` (constraint text, turns remaining)
- Injected into `append_system_prompt` like system constraints, but in a separate section:
  ```
  # Temporary Constraints (expire after N turns)
  - <constraint> [1 turn remaining]
  ```
- After each agent turn (in the main loop, after `_run_query_with_interrupt`), decrement turn counters and remove expired constraints
- Rebuild `append_system_prompt` on each decrement
- Render notice when constraint expires: "Constraint expired: <text>"

**Edge cases:**
- `--turns 0` is invalid → error
- Multiple constraints with different expiry → each tracked independently
- `/constrain` with no args shows active temporary constraints and their remaining turns
- `/constrain clear` removes all temporary constraints

**Usage:**
```
/constrain Do not create new files                    # Expires after 1 turn
/constrain Only fix the bug, no refactoring --turns 3  # Expires after 3 turns
/constrain                                            # Show active constraints
/constrain clear                                      # Remove all
```

**Effort:** ~45min

---

### 4. `/scope <files...>` — Restrict agent to specific files

**What:** Pins the agent to a set of files. Any tool call targeting a file outside the scope gets blocked with a warning injected into context.

**How:**
- Store in `_scoped_files: set[Path] | None` (None = no restriction)
- On `/scope file1.py file2.py`, resolve to absolute paths and store in `_scoped_files`
- Before each query, check if `_scoped_files` is set
- If set, inject into `append_system_prompt`:
  ```
  # File Scope Restriction
  You may ONLY read, edit, or create these files:
  - src/file1.py
  - src/file2.py
  Do NOT access, read, or modify any other files.
  ```
- This is a *soft* constraint — it relies on the system prompt. A determined agent can still access other files. For hard enforcement, we'd need a tool wrapper that intercepts and blocks, but that's significantly more complex and SDK-dependent.
- `/scope` with no args shows current scope
- `/scope clear` removes the restriction
- On `/reset`, scope is cleared

**Interaction with existing code:**
- Must compose with goal prompt and system/constrain sections
- If agent uses `@file` references outside scope, the reference resolver still works but the system prompt tells the agent not to
- `/scope` works best combined with `/constrain` — scope for file restriction, constrain for behavioral restriction

**Usage:**
```
/scope src/bug.py src/utils.py    # Restrict to these files
/scope                             # Show current scope
/scope clear                       # Remove restriction
```

**Effort:** ~30min

---

### 5. `/narrow` — Toggle minimal-change mode

**What:** Injects a "minimal mode" system instruction: make the smallest possible change, no refactoring, no new files, no scope expansion.

**How:**
- Boolean flag `_narrow_mode: bool` (default False)
- When active, injects into `append_system_prompt`:
  ```
  # Minimal Change Mode
  Make the SMALLEST possible change to address the request.
  - Do NOT refactor surrounding code
  - Do NOT create new files
  - Do NOT add features beyond what was asked
  - Do NOT change code that isn't directly related to the request
  - Prefer editing existing code over creating new code
  ```
- `/narrow` toggles the flag on/off
- Shows current state: "Minimal mode: ON" / "Minimal mode: OFF"
- On `/reset`, narrow mode is cleared (reset to OFF)

**Usage:**
```
/narrow        # Toggle minimal mode
```

**Effort:** ~20min

---

## Shared Infrastructure

All five commands share a common mechanism for injecting content into the agent's system prompt. Rather than each command independently manipulating `append_system_prompt`, we need a single prompt builder:

```python
def _rebuild_system_prompt() -> None:
    """Rebuild agent._options.append_system_prompt from all active prompt sections."""
    sections = []
    
    # 1. Goal section (existing)
    goal_section = build_goal_prompt_section() if get_active_goal() else ""
    if goal_section:
        sections.append(goal_section)
    
    # 2. System constraints (persistent)
    if _system_constraints:
        lines = ["# Additional Constraints"]
        for c in _system_constraints:
            lines.append(f"- {c}")
        sections.append("\n".join(lines))
    
    # 3. Temporary constraints (auto-expiring)
    if _temporary_constraints:
        lines = ["# Temporary Constraints"]
        for text, turns in _temporary_constraints:
            lines.append(f"- {text} [{turns} turn{'s' if turns != 1 else ''} remaining]")
        sections.append("\n".join(lines))
    
    # 4. File scope restriction
    if _scoped_files is not None:
        lines = ["# File Scope Restriction"]
        lines.append("You may ONLY read, edit, or create these files:")
        for f in sorted(_scoped_files):
            lines.append(f"- {f}")
        lines.append("Do NOT access, read, or modify any other files.")
        sections.append("\n".join(lines))
    
    # 5. Minimal change mode
    if _narrow_mode:
        sections.append(_MINIMAL_MODE_PROMPT)
    
    # Compose: strip existing managed sections, append new ones
    current = getattr(opts, "append_system_prompt", "") or ""
    # Remove all managed sections (they're re-added below)
    current = re.sub(r"\n*# (Current Goal|Additional Constraints|Temporary Constraints|File Scope Restriction|Minimal Change Mode)\n.*?(?=\n# |\Z)", "", current, flags=re.DOTALL).rstrip()
    
    if sections:
        opts.append_system_prompt = (current + "\n\n" + "\n\n".join(sections)).strip()
    else:
        opts.append_system_prompt = current
```

This replaces the current `_update_agent_goal_prompt` with a unified prompt builder. Each command just sets its state variables and calls `_rebuild_system_prompt()`.

---

## Command Summary

| Command | What | Persists after /clear? | Persists after /reset? | Effort |
|---|---|---|---|---|
| `/undo` | Drop last turn | N/A (history is cleared) | N/A | 15min |
| `/system <text>` | Persistent system instruction | Yes | No | 30min |
| `/constrain <text> [--turns N]` | Temporary constraint | Yes | No | 45min |
| `/scope <files...>` | Restrict to files | Yes | No | 30min |
| `/narrow` | Toggle minimal-change mode | Yes | No | 20min |

**Total estimated effort:** ~2.5 hours

---

## Implementation Order

1. **`_rebuild_system_prompt`** — Shared infrastructure (all commands depend on this)
2. **`/undo`** — Simplest, no prompt infrastructure needed
3. **`/narrow`** — Simplest prompt injection, validates `_rebuild_system_prompt`
4. **`/system`** — Persistent constraints, validates prompt rebuild with state
5. **`/scope`** — File restriction, adds path resolution
6. **`/constrain`** — Most complex (turn tracking, auto-expiry, decrement logic)

---

## Testing Plan

Each command needs:

1. **Unit test**: Command parsing, state management, prompt rebuilding
2. **Interactive test**: Run via CLI, verify behavior
3. **Edge case test**: Empty state, double invocation, interaction with /reset and /clear
4. **Composition test**: Multiple commands active simultaneously (e.g., `/narrow` + `/scope` + `/constrain`)

Key regression checks:
- `/goal` prompt still works after `_rebuild_system_prompt` replaces `_update_agent_goal_prompt`
- `/clear` preserves system/constrain/scope/narrow (they're behavioral, not history)
- `/reset` clears everything
- Goal loop turn counter still decrements temporary constraints correctly
- `_rebuild_system_prompt` doesn't corrupt existing `append_system_prompt` content

---

## Out of Scope

- **Hard file scoping** (tool-call interception) — soft prompt-based only for now
- **`/diff` or `/revert`** (show/undo file changes) — separate feature
- **`/checkpoint` / `/rollback`** (session snapshots) — separate feature
- **`/trace`** (verbose tool logging) — visibility feature, separate from steering
- Changes to the SDK — we only use the existing `append_system_prompt` and `_history` APIs
