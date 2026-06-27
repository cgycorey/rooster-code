# Handoff Command Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a manual `/handoff` chat command that writes a local `.handoff` continuation file and compacts the current session history in place.

**Architecture:** Reuse the existing manual compaction pipeline in `runtime_session.py`: filter compactable history, ask the active provider for the structured continuation summary, and build the same two-message compacted history. Add one runtime wrapper that writes the summary to disk before swapping `agent._history`, then expose it through `runtime.py` and the chat command loop.

**Tech Stack:** Python 3.12, `open_agent_sdk`, Rich CLI rendering, pytest via `uv run pytest`.

---

## Pre-flight

- Do not modify SDK files under `.venv/` or `open_agent_sdk`.
- Do not commit unless the user explicitly asks.
- Use TDD: write each test, run it and watch it fail for the expected reason, then implement the minimal code.
- Keep `/compact` behavior unchanged.
- Treat `.handoff` as a local generated artifact; it must be ignored by git.

## File Structure

- Modify `src/rooster_code/runtime_session.py`
  - Add `_build_handoff_file_content()`.
  - Add `handoff_current_session()`.
  - Reuse `_filter_history_for_manual_compaction()` and `_compact_with_provider()`.
- Modify `src/rooster_code/runtime.py`
  - Re-export `handoff_current_session` from `runtime_session.py` beside `compact_current_session`.
- Modify `src/rooster_code/cli.py`
  - Add a wrapper `handoff_current_session(agent, path)`.
  - Add `/handoff [path]` handling in the chat loop.
- Modify `src/rooster_code/rendering.py`
  - Add `/handoff [path]` to help output.
- Modify `.gitignore`
  - Add `.handoff` so local handoff files are not accidentally committed.
- Modify `tests/test_runtime.py`
  - Runtime behavior tests for file writing, compaction, skip path, and history restoration.
- Modify `tests/test_chat.py`
  - CLI command tests for default path, explicit path, skipped handoff, and errors.
- Modify `tests/test_rendering.py`
  - Help text test.

---

### Task 1: Runtime handoff writes file and compacts history

**Files:**
- Modify: `tests/test_runtime.py`
- Modify: `src/rooster_code/runtime_session.py`
- Modify: `src/rooster_code/runtime.py`

- [ ] **Step 1: Add failing runtime success test**

Append this test near the existing compaction tests in `tests/test_runtime.py`, after `test_compact_current_session_rewrites_agent_history`:

```python
def test_handoff_current_session_writes_file_and_compacts_history(monkeypatch, tmp_path) -> None:
    original_history = [
        {"role": "user", "content": [{"type": "text", "text": "hello"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
    ]
    compacted_history = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "[Previous conversation summary]\n\nsummary"}],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "I understand the context. Let me continue from where we left off."}],
        },
    ]

    class FakeAgent:
        def __init__(self) -> None:
            self._history = list(original_history)
            self.initialized = False
            self._session_id = "sess-1"
            self._options = type("Options", (), {"model": "m-test", "cwd": str(tmp_path)})()

        async def _initialize(self) -> None:
            self.initialized = True

        def _ensure_provider(self):
            return provider

        def _resolve_model(self) -> str:
            return "m-test"

    class FakeProvider:
        def __init__(self) -> None:
            self.params = None

        async def create_message(self, params):
            self.params = params
            return CreateMessageResponse(
                content=[{"type": "text", "text": "summary"}],
                stop_reason="end_turn",
            )

    provider = FakeProvider()
    agent = FakeAgent()
    monkeypatch.setattr(
        runtime,
        "estimate_messages_tokens",
        lambda messages: 1200 if messages == original_history else 240,
        raising=False,
    )

    handoff_path = tmp_path / ".handoff"
    result = asyncio.run(runtime.handoff_current_session(agent, handoff_path))

    assert agent.initialized is True
    assert provider.params is not None
    assert provider.params.model == "m-test"
    assert agent._history == compacted_history
    assert result == {
        "compacted": True,
        "written": True,
        "path": str(handoff_path),
        "summary": "summary",
        "before_tokens": 1200,
        "after_tokens": 240,
        "reason": "",
    }
    text = handoff_path.read_text(encoding="utf-8")
    assert text.startswith("# Handoff\n")
    assert "Session: sess-1" in text
    assert "Model: m-test" in text
    assert f"CWD: {tmp_path}" in text
    assert "## Resume Prompt" in text
    assert "Read this `.handoff`" in text
    assert "summary" in text
```

- [ ] **Step 2: Run test and verify RED**

Run:

```bash
uv run pytest tests/test_runtime.py::test_handoff_current_session_writes_file_and_compacts_history -q
```

Expected failure:

```text
AttributeError: module 'rooster_code.runtime' has no attribute 'handoff_current_session'
```

- [ ] **Step 3: Implement runtime handoff helper**

In `src/rooster_code/runtime_session.py`, add `datetime` to imports:

```python
from datetime import datetime
```

Then add this helper after `_compact_with_provider()` and before `compact_current_session()`:

```python
def _build_handoff_file_content(agent: object, summary: str) -> str:
    session_id = "new"
    get_session_id = getattr(agent, "get_session_id", None)
    if callable(get_session_id):
        with contextlib.suppress(Exception):
            session_id = str(get_session_id() or "new")
    else:
        raw_session_id = getattr(agent, "_session_id", "")
        if raw_session_id:
            session_id = str(raw_session_id)

    model = "default"
    resolve_model = getattr(agent, "_resolve_model", None)
    if callable(resolve_model):
        with contextlib.suppress(Exception):
            model = str(resolve_model() or "default")
    else:
        options = getattr(agent, "_options", None)
        raw_model = getattr(options, "model", "") if options is not None else ""
        if raw_model:
            model = str(raw_model)

    options = getattr(agent, "_options", None)
    cwd = getattr(options, "cwd", "") if options is not None else ""
    cwd_text = str(cwd or Path.cwd())
    generated = datetime.now().isoformat(timespec="seconds")

    return (
        "# Handoff\n\n"
        f"Generated: {generated}\n"
        f"Session: {session_id}\n"
        f"Model: {model}\n"
        f"CWD: {cwd_text}\n\n"
        "## Resume Prompt\n\n"
        "Read this `.handoff`, inspect the current repository state, verify whether the described state is still accurate, "
        "then continue from the `## Next Step` section. Do not assume this handoff is fully up to date.\n\n"
        "---\n\n"
        f"{summary.strip()}\n"
    )
```

Then add `handoff_current_session()` after the helper:

```python
async def handoff_current_session(agent, path: str | Path | None = None) -> dict[str, object]:
    if hasattr(agent, "_initialize"):
        await agent._initialize()

    history = list(getattr(agent, "_history", []))
    compactable_history = _filter_history_for_manual_compaction(history)
    before_tokens = _sdk().estimate_messages_tokens(compactable_history)

    if len(compactable_history) < 2:
        return {
            "compacted": False,
            "written": False,
            "path": "",
            "summary": "",
            "before_tokens": before_tokens,
            "after_tokens": before_tokens,
            "reason": "Need at least two messages before compaction.",
        }

    pre_compaction_history = list(agent._history)
    handoff_path = Path(path) if path is not None else Path(".handoff")
    try:
        summary, compacted_history = await _compact_with_provider(agent, compactable_history)
        after_tokens = _sdk().estimate_messages_tokens(compacted_history)
        handoff_path.parent.mkdir(parents=True, exist_ok=True)
        handoff_path.write_text(_build_handoff_file_content(agent, summary), encoding="utf-8")
    except Exception:
        agent._history = pre_compaction_history
        raise

    agent._history = compacted_history
    compacted = after_tokens < before_tokens
    return {
        "compacted": compacted,
        "written": True,
        "path": str(handoff_path),
        "summary": summary,
        "before_tokens": before_tokens,
        "after_tokens": after_tokens,
        "reason": "" if compacted else "Compaction produced no smaller history.",
    }
```

In `src/rooster_code/runtime.py`, add `handoff_current_session` to the import list from `rooster_code.runtime_session`:

```python
    handoff_current_session,  # noqa: F401
```

- [ ] **Step 4: Run test and verify GREEN**

Run:

```bash
uv run pytest tests/test_runtime.py::test_handoff_current_session_writes_file_and_compacts_history -q
```

Expected:

```text
1 passed
```

---

### Task 2: Runtime failure and skip behavior

**Files:**
- Modify: `tests/test_runtime.py`
- Modify: `src/rooster_code/runtime_session.py` only if Task 1 code needs correction

- [ ] **Step 1: Add failing test for file-write restoration**

Append this test near the runtime handoff success test:

```python
def test_handoff_current_session_restores_history_when_file_write_fails(monkeypatch, tmp_path) -> None:
    original_history = [
        {"role": "user", "content": [{"type": "text", "text": "hello"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
    ]

    class FakeAgent:
        def __init__(self) -> None:
            self._history = list(original_history)

        async def _initialize(self) -> None:
            return None

        def _ensure_provider(self):
            return provider

        def _resolve_model(self) -> str:
            return "m-test"

    class FakeProvider:
        async def create_message(self, params):
            return CreateMessageResponse(
                content=[{"type": "text", "text": "summary"}],
                stop_reason="end_turn",
            )

    provider = FakeProvider()
    agent = FakeAgent()
    monkeypatch.setattr(runtime, "estimate_messages_tokens", lambda messages: 42, raising=False)
    blocking_file = tmp_path / "not-a-dir"
    blocking_file.write_text("blocks parent directory", encoding="utf-8")

    with pytest.raises(FileExistsError):
        asyncio.run(runtime.handoff_current_session(agent, blocking_file / ".handoff"))

    assert agent._history == original_history
```

- [ ] **Step 2: Run test and verify RED if Task 1 did not already cover it**

Run:

```bash
uv run pytest tests/test_runtime.py::test_handoff_current_session_restores_history_when_file_write_fails -q
```

Expected before correct implementation if Task 1 did not already include the restoration behavior:

```text
FAILED tests/test_runtime.py::test_handoff_current_session_restores_history_when_file_write_fails
```

If Task 1 implementation already passes this test, record that the broader implementation covered the failure path and continue.

- [ ] **Step 3: Ensure implementation restores before raising**

Confirm `handoff_current_session()` wraps provider + file writing in:

```python
    try:
        summary, compacted_history = await _compact_with_provider(agent, compactable_history)
        after_tokens = _sdk().estimate_messages_tokens(compacted_history)
        handoff_path.parent.mkdir(parents=True, exist_ok=True)
        handoff_path.write_text(_build_handoff_file_content(agent, summary), encoding="utf-8")
    except Exception:
        agent._history = pre_compaction_history
        raise
```

No code change is needed if this exact behavior is present.

- [ ] **Step 4: Add skip test**

Append:

```python
def test_handoff_current_session_skips_small_history(monkeypatch, tmp_path) -> None:
    class FakeAgent:
        def __init__(self) -> None:
            self._history = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]

        async def _initialize(self) -> None:
            return None

    handoff_path = tmp_path / ".handoff"
    agent = FakeAgent()
    monkeypatch.setattr(runtime, "estimate_messages_tokens", lambda messages: 42, raising=False)

    result = asyncio.run(runtime.handoff_current_session(agent, handoff_path))

    assert result == {
        "compacted": False,
        "written": False,
        "path": "",
        "summary": "",
        "before_tokens": 42,
        "after_tokens": 42,
        "reason": "Need at least two messages before compaction.",
    }
    assert not handoff_path.exists()
```

- [ ] **Step 5: Run runtime handoff tests**

Run:

```bash
uv run pytest tests/test_runtime.py::test_handoff_current_session_writes_file_and_compacts_history tests/test_runtime.py::test_handoff_current_session_restores_history_when_file_write_fails tests/test_runtime.py::test_handoff_current_session_skips_small_history -q
```

Expected:

```text
3 passed
```

---

### Task 3: Chat command exposes `/handoff [path]`

**Files:**
- Modify: `tests/test_chat.py`
- Modify: `src/rooster_code/cli.py`

- [ ] **Step 1: Add failing default-path CLI test**

Append near the `/compact` tests in `tests/test_chat.py`:

```python
def test_run_chat_handoff_writes_default_file(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}
    prompts = iter(["/handoff", "/exit"])
    notices: list[tuple[str, str, str]] = []

    class FakeAgent:
        async def close(self) -> None:
            captured["closed"] = True

    async def fake_handoff_current_session(agent, path: str | None = None) -> dict[str, object]:
        captured["agent"] = agent
        captured["path"] = path
        return {
            "compacted": True,
            "written": True,
            "path": path or "",
            "summary": "summary text",
            "before_tokens": 1200,
            "after_tokens": 240,
            "reason": "",
        }

    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: FakeAgent())
    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())
    monkeypatch.setattr(cli, "handoff_current_session", fake_handoff_current_session, raising=False)
    monkeypatch.setattr(cli, "render_notice", lambda console, title, message, style="yellow": notices.append((title, message, style)))
    monkeypatch.setattr(PromptSession, "prompt_async", _fake_prompt_iter(prompts))

    exit_code = cli.asyncio.run(cli.run_chat(RuntimeConfig(model="m2", cwd=str(tmp_path))))

    assert exit_code == 0
    assert isinstance(captured["agent"], FakeAgent)
    assert captured["path"] == str(tmp_path / ".handoff")
    assert captured["closed"] is True
    assert notices[0][0] == "Handoff"
    assert notices[0][2] == "green"
    assert f"Saved {tmp_path / '.handoff'}" in notices[0][1]
    assert "1200 → 240" in notices[0][1]
```

- [ ] **Step 2: Run default-path test and verify RED**

Run:

```bash
uv run pytest tests/test_chat.py::test_run_chat_handoff_writes_default_file -q
```

Expected failure:

```text
FAILED tests/test_chat.py::test_run_chat_handoff_writes_default_file
```

The failure should occur because `/handoff` is not handled yet, so the fake `handoff_current_session()` is never called and `captured["path"]` is missing.

- [ ] **Step 3: Implement CLI wrapper and command branch**

In `src/rooster_code/cli.py`, add this function after `compact_current_session()`:

```python
async def handoff_current_session(agent, path: str | None = None):
    from rooster_code.runtime import handoff_current_session as runtime_handoff_current_session

    return await runtime_handoff_current_session(agent, path)
```

In the `run_chat()` command loop, immediately after the `/compact` branch, add:

```python
            if command.name == "handoff":
                handoff_path = str(Path(config.cwd or ".") / command.args[0]) if command.args else str(Path(config.cwd or ".") / ".handoff")
                try:
                    result = await handoff_current_session(agent, handoff_path)
                except Exception as exc:
                    render_notice(console, "Handoff Error", str(exc), "red")
                    continue

                if result["written"]:
                    render_notice(
                        console,
                        "Handoff",
                        f"Saved {result['path']}\nTokens: {result['before_tokens']} → {result['after_tokens']}",
                        "green",
                    )
                else:
                    render_notice(console, "Handoff skipped", str(result["reason"] or "No handoff written."), "yellow")
                continue
```

- [ ] **Step 4: Run default-path test and verify GREEN**

Run:

```bash
uv run pytest tests/test_chat.py::test_run_chat_handoff_writes_default_file -q
```

Expected:

```text
1 passed
```

- [ ] **Step 5: Add explicit-path CLI test**

Append:

```python
def test_run_chat_handoff_resolves_relative_path_argument_against_cwd(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}
    prompts = iter(["/handoff custom.handoff", "/exit"])

    class FakeAgent:
        async def close(self) -> None:
            captured["closed"] = True

    async def fake_handoff_current_session(agent, path: str | None = None) -> dict[str, object]:
        captured["path"] = path
        return {
            "compacted": True,
            "written": True,
            "path": path or "",
            "summary": "summary text",
            "before_tokens": 12,
            "after_tokens": 4,
            "reason": "",
        }

    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: FakeAgent())
    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())
    monkeypatch.setattr(cli, "handoff_current_session", fake_handoff_current_session, raising=False)
    monkeypatch.setattr(cli, "render_notice", lambda *args, **kwargs: None)
    monkeypatch.setattr(PromptSession, "prompt_async", _fake_prompt_iter(prompts))

    exit_code = cli.asyncio.run(cli.run_chat(RuntimeConfig(model="m2", cwd=str(tmp_path))))

    assert exit_code == 0
    assert captured["path"] == str(tmp_path / "custom.handoff")
    assert captured["closed"] is True
```

- [ ] **Step 6: Add skipped and error CLI tests**

Append:

```python
def test_run_chat_handoff_shows_skipped_notice(monkeypatch, tmp_path) -> None:
    prompts = iter(["/handoff", "/exit"])
    notices: list[tuple[str, str, str]] = []

    class FakeAgent:
        async def close(self) -> None:
            return None

    async def fake_handoff_current_session(agent, path: str | None = None) -> dict[str, object]:
        return {
            "compacted": False,
            "written": False,
            "path": "",
            "summary": "",
            "before_tokens": 42,
            "after_tokens": 42,
            "reason": "Need at least two messages before compaction.",
        }

    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: FakeAgent())
    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())
    monkeypatch.setattr(cli, "handoff_current_session", fake_handoff_current_session, raising=False)
    monkeypatch.setattr(cli, "render_notice", lambda console, title, message, style="yellow": notices.append((title, message, style)))
    monkeypatch.setattr(PromptSession, "prompt_async", _fake_prompt_iter(prompts))

    exit_code = cli.asyncio.run(cli.run_chat(RuntimeConfig(model="m2", cwd=str(tmp_path))))

    assert exit_code == 0
    assert notices[0] == ("Handoff skipped", "Need at least two messages before compaction.", "yellow")


def test_run_chat_handoff_shows_error_when_handoff_fails(monkeypatch) -> None:
    captured: dict[str, object] = {}
    prompts = iter(["/handoff", "/exit"])
    notices: list[tuple[str, str, str]] = []

    class FakeAgent:
        async def close(self) -> None:
            captured["closed"] = True

    async def fake_handoff_current_session(agent, path: str | None = None) -> dict[str, object]:
        raise RuntimeError("Handoff failed")

    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: FakeAgent())
    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())
    monkeypatch.setattr(cli, "handoff_current_session", fake_handoff_current_session, raising=False)
    monkeypatch.setattr(cli, "render_notice", lambda console, title, message, style="yellow": notices.append((title, message, style)))
    monkeypatch.setattr(PromptSession, "prompt_async", _fake_prompt_iter(prompts))

    exit_code = cli.asyncio.run(cli.run_chat(RuntimeConfig(model="m2")))

    assert exit_code == 0
    assert captured["closed"] is True
    assert notices[0] == ("Handoff Error", "Handoff failed", "red")
```

- [ ] **Step 7: Run CLI handoff tests**

Run:

```bash
uv run pytest tests/test_chat.py::test_run_chat_handoff_writes_default_file tests/test_chat.py::test_run_chat_handoff_accepts_path_argument tests/test_chat.py::test_run_chat_handoff_shows_skipped_notice tests/test_chat.py::test_run_chat_handoff_shows_error_when_handoff_fails -q
```

Expected:

```text
4 passed
```

---

### Task 4: Help text advertises `/handoff [path]` and `.handoff` is ignored

**Files:**
- Modify: `tests/test_rendering.py`
- Modify: `src/rooster_code/rendering.py`
- Modify: `.gitignore`

- [ ] **Step 1: Add failing help test**

Append near other rendering tests in `tests/test_rendering.py`:

```python
def test_render_help_mentions_handoff_command() -> None:
    console = Console(record=True, width=100)

    render_help(console)

    output = console.export_text()
    assert "/handoff [path]" in output
```

- [ ] **Step 2: Run help test and verify RED**

Run:

```bash
uv run pytest tests/test_rendering.py::test_render_help_mentions_handoff_command -q
```

Expected failure:

```text
FAILED tests/test_rendering.py::test_render_help_mentions_handoff_command
AssertionError: assert '/handoff [path]' in output
```

- [ ] **Step 3: Add help row**

In `src/rooster_code/rendering.py`, add this row after `/compact`:

```python
    table.add_row("/handoff [path]", "Write a local handoff file and compact current history")
```

- [ ] **Step 4: Run help test and verify GREEN**

Run:

```bash
uv run pytest tests/test_rendering.py::test_render_help_mentions_handoff_command -q
```

Expected:

```text
1 passed
```
- [ ] **Step 5: Ignore generated `.handoff` files**

Add this line to `.gitignore`:

```gitignore
.handoff
```

- [ ] **Step 6: Verify `.handoff` is ignored**

Run:

```bash
git check-ignore -q .handoff
```

Expected: exit code 0 and no output.

---

### Task 5: Focused regression suite

**Files:**
- No new file changes unless tests expose a bug.

- [ ] **Step 1: Run focused handoff and compaction tests**

Run:

```bash
uv run pytest tests/test_runtime.py::test_compact_current_session_rewrites_agent_history tests/test_runtime.py::test_build_manual_compaction_summary_prompt_uses_structured_handoff tests/test_runtime.py::test_handoff_current_session_writes_file_and_compacts_history tests/test_runtime.py::test_handoff_current_session_restores_history_when_file_write_fails tests/test_runtime.py::test_handoff_current_session_skips_small_history tests/test_chat.py::test_run_chat_compacts_agent_history tests/test_chat.py::test_run_chat_shows_compact_error_when_compaction_fails tests/test_chat.py::test_run_chat_handoff_writes_default_file tests/test_chat.py::test_run_chat_handoff_accepts_path_argument tests/test_chat.py::test_run_chat_handoff_shows_skipped_notice tests/test_chat.py::test_run_chat_handoff_shows_error_when_handoff_fails tests/test_rendering.py::test_render_help_mentions_handoff_command -q
```

Expected:

```text
12 passed
```

- [ ] **Step 2: Run broader touched-file tests**

Run:

```bash
uv run pytest tests/test_runtime.py tests/test_chat.py tests/test_rendering.py -q
```

Expected:

```text
all selected tests pass
```

- [ ] **Step 3: Run lint**

Run:

```bash
uv run ruff check
```

Expected:

```text
All checks passed!
```

- [ ] **Step 4: Inspect working tree**

Run:

```bash
git status --short
```

Expected:

```text
 M .gitignore
 M src/rooster_code/cli.py
 M src/rooster_code/rendering.py
 M src/rooster_code/runtime.py
 M src/rooster_code/runtime_session.py
 M tests/test_chat.py
 M tests/test_rendering.py
 M tests/test_runtime.py
?? docs/superpowers/specs/2026-06-26-handoff-command-design.md
?? docs/superpowers/plans/2026-06-26-handoff-command.md
?? image.png
```

If `image.png` was already untracked before implementation, leave it untouched.
