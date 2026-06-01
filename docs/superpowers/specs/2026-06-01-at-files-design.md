# @ Files Feature — Design Spec

**Date:** 2026-06-01
**Status:** approved

## Overview

Add `@filename` shorthand to include file contents in chat context. Users type `@path/to/file.py` to reference files, and the contents are injected as labeled context blocks before the user's message reaches the LLM.

## Syntax

| Pattern | Behavior |
|---------|----------|
| `@src/foo.py` | Single file — reads and includes contents |
| `@src/rooster_code/*.py` | Glob expansion — includes all matching files |
| `@src/bar.py @src/baz.py` | Multiple references in one message |

**Paths are relative to `config.cwd`** (the project root). Absolute paths also supported.

**Out of scope:** line ranges (`@file:10-20`), recursive globs (`@**/*.py`), image files.

## Architecture

### New Module: `src/rooster_code/file_context.py`

Single exported function:

```python
def resolve_at_references(text: str, cwd: str) -> tuple[str, list[FileContext]]:
```

- **Input:** raw user text, working directory
- **Output:** cleaned text (with `@path` tokens removed) plus list of `FileContext(path, content)`
- **Raises:** `FileNotFoundError`, `GlobNoMatchError`, `FileTooLargeError`, `BinaryFileError`, `PermissionError`

**Implementation details:**

1. Scan `text` for `@<path>` patterns via regex — skip `@` inside backtick code blocks
2. For each match, resolve path relative to `cwd`
3. If path contains glob characters (`*`, `?`, `[`), expand via `pathlib.Path.glob`
4. Read each resolved file (deduplicated)
5. Detect binary files by attempting UTF-8 decode
6. Reject files over 100KB
7. Return cleaned text + file contexts

### CLI Integration: `src/rooster_code/cli.py` lines 905-917

**Current flow:**
```
user_input → pending_notice prepend → agent.query()
```

**New flow:**
```
user_input → resolve @references → error? show notice, skip
                                 → build context block → pending_notice prepend → agent.query()
```

Pseudo:

```python
# Resolve @file references
try:
    cleaned_input, files = resolve_at_references(user_input, config.cwd or ".")
except (FileNotFoundError, GlobNoMatchError, ...) as e:
    render_notice(console, "@ Error", str(e), "red")
    continue

# Build context block
context = ""
if files:
    context = "[Files referenced by the user:]\n\n"
    for f in files:
        context += f"--- {f.path} ---\n{f.content}\n\n"

# Compose effective input
effective_input = ""
if pending_notice:
    effective_input += pending_notice
if context:
    effective_input += context
    effective_input += f"[User message:]\n{cleaned_input}"
else:
    effective_input += cleaned_input

await _run_query_with_interrupt(agent.query(effective_input), ...)
```

When no `@` references exist, the flow is byte-for-byte identical to current behavior.

### Auto-Completion: Custom `AtFileCompleter`

A `prompt_toolkit.completion.Completer` that activates when the word under cursor starts with `@`:

```python
class AtFileCompleter(Completer):
    def __init__(self, cwd: str): ...
    def get_completions(self, document, complete_event):
        # strip @, delegate to PathCompleter for matching
        # re-add @ prefix to completions
```

**Wired in at `cli.py:660`:**
```python
prompt_session = PromptSession(
    history=FileHistory(str(history_path)),
    completer=AtFileCompleter(cwd=config.cwd or "."),
)
```

- Typing `@src/roo` + Tab → shows `@src/rooster_code/` completions
- Only activates when the current word starts with `@`
- Falls back to no completions for non-@ text

## Error Handling

All errors block the message (not sent to LLM) and show a red notice:

| Condition | Error message |
|-----------|---------------|
| File not found | `@ src/noexist.py: file not found` |
| Glob matches nothing | `@ src/*.rs: no files matched` |
| Binary file | `@ image.png: binary file not supported` |
| File too large (>100KB) | `@ huge.log: file too large (5.2MB, limit 100KB)` |
| Permission denied | `@ /root/secret: permission denied` |

## Edge Cases

- **`@` inside backtick code blocks:** ignored (not treated as a reference)
- **Duplicate references:** each file included once
- **Message is only `@file`:** cleaned input is empty string, context block includes `[User message:]\n(no additional text)`
- **Multiple @ references:** all resolved, listed in order of first appearance
- **DAEMON MODE:** NOT supported. The daemon calls `agent.prompt()` directly and does not go through CLI input processing. @file resolution is CLI-chat only.

## Testing

### Unit tests (`tests/test_file_context.py`)

- `resolve_at_references` with single file
- `resolve_at_references` with glob expansion
- `resolve_at_references` with multiple references
- `resolve_at_references` with duplicate dedup
- `resolve_at_references` with `@` inside backtick blocks — ignored
- `resolve_at_references` with empty @ text — no-op
- `FileNotFoundError` raised for missing file
- `GlobNoMatchError` raised for empty glob
- `BinaryFileError` raised for binary content
- `FileTooLargeError` raised for oversized files
- `AtFileCompleter` produces completions for `@`-prefixed text
- `AtFileCompleter` produces no completions for regular text

### Integration test

- End-to-end chat: `@file` reference → file content appears in context
- End-to-end chat: invalid `@file` → error notice, message not sent
- End-to-end chat: no `@` → behavior unchanged

## Files Changed

| File | Change |
|------|--------|
| `src/rooster_code/file_context.py` | **New** — `resolve_at_references()`, `AtFileCompleter`, exception classes |
| `src/rooster_code/cli.py` | Add @-resolution block at ~line 905, add completer at ~line 660 |
| `tests/test_file_context.py` | **New** — unit and integration tests |
