# Manual Compaction Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expose a `/compact` chat command in `cock-code` that manually summarizes the current conversation without relying on private engine state.

**Architecture:** Implement manual compaction in `../open-agent-sdk-python` as a new async `Agent.compact_history()` method that rewrites `Agent._history`, because `Agent._history` is the only conversation state reused across turns and persisted on `close()`. Keep `cock-code` thin: `src/cock_code/cli.py` should call the new agent method directly, use the existing `render_notice()` UI to show the result, and add `/compact` to `render_help()`. Do **not** add a `runtime.py` helper, do **not** read or write `agent._engine` from `cock-code`, and do **not** expand the first implementation into PRE/POST compact hook wiring.

**Tech Stack:** Python 3.12, pytest, Rich, local sibling repo `../open-agent-sdk-python`, `open_agent_sdk.utils.compact`, `open_agent_sdk.utils.tokens`

## Non-goals

- Do not mutate `agent._engine._messages` or `agent._engine._compact_state` from `cock-code`.
- Do not add a new `render_compact_result()` surface unless the existing `render_notice()` output is clearly insufficient.
- Do not wire `PRE_COMPACT` / `POST_COMPACT` in this first pass; treat that as a follow-up feature after `/compact` works end-to-end.
- Do not change `src/cock_code/runtime.py` unless the SDK API shape forces it. With the plan below, it should remain unchanged.

## API shape to implement

Use a typed SDK return value so the CLI can render the result without inspecting raw dicts.

```python
@dataclass
class CompactionResult:
    compacted: bool
    summary: str = ""
    before_tokens: int = 0
    after_tokens: int = 0
    messages_before: int = 0
    messages_after: int = 0
    reason: str = ""
```

```python
async def compact_history(self) -> CompactionResult:
    await self._initialize()

    if len(self._history) < 2:
        return CompactionResult(
            compacted=False,
            before_tokens=estimate_messages_tokens(self._history),
            after_tokens=estimate_messages_tokens(self._history),
            messages_before=len(self._history),
            messages_after=len(self._history),
            reason="Need at least two messages before compaction.",
        )

    before_messages = list(self._history)
    before_tokens = estimate_messages_tokens(before_messages)
    result = await compact_conversation(
        self._ensure_client(),
        self._resolve_model(),
        before_messages,
        create_auto_compact_state(),
    )

    compacted_messages = result["compacted_messages"]
    self._history = list(compacted_messages)
    after_tokens = estimate_messages_tokens(self._history)

    return CompactionResult(
        compacted=compacted_messages != before_messages,
        summary=result.get("summary", ""),
        before_tokens=before_tokens,
        after_tokens=after_tokens,
        messages_before=len(before_messages),
        messages_after=len(self._history),
        reason="" if compacted_messages != before_messages else "Compaction produced no smaller history.",
    )
```

## Task 1: Add a public SDK compaction API

**Files:**
- Modify: `../open-agent-sdk-python/tests/test_agent.py:53-151`
- Modify: `../open-agent-sdk-python/src/open_agent_sdk/types.py:364-371`
- Modify: `../open-agent-sdk-python/src/open_agent_sdk/agent.py:30-43, 144-151, 254-299`
- Modify: `../open-agent-sdk-python/src/open_agent_sdk/__init__.py:178-234, 236-433`

**Step 1: Write the failing SDK tests**

Add two tests to `../open-agent-sdk-python/tests/test_agent.py`.

```python
@pytest.mark.asyncio
async def test_compact_history_rewrites_history_and_returns_summary(self):
    agent = Agent(AgentOptions(api_key="test-key", model="claude-sonnet-4-5"))
    agent._history = [
        {"role": "user", "content": [{"type": "text", "text": "hello"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
    ]

    with patch("open_agent_sdk.agent.compact_conversation", new_callable=AsyncMock) as mock_compact:
        mock_compact.return_value = {
            "compacted_messages": [
                {"role": "user", "content": [{"type": "text", "text": "[Previous conversation summary]\n\nsummary"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "I understand the context. Let me continue from where we left off."}]},
            ],
            "summary": "summary",
            "state": create_auto_compact_state(),
        }

        result = await agent.compact_history()

    assert result.compacted is True
    assert result.summary == "summary"
    assert result.before_tokens >= result.after_tokens
    assert agent.get_messages()[0]["role"] == "user"
    assert "summary" in agent.get_messages()[0]["content"][0]["text"]


@pytest.mark.asyncio
async def test_compact_history_skips_small_history(self):
    agent = Agent()
    agent._history = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]

    result = await agent.compact_history()

    assert result.compacted is False
    assert result.reason == "Need at least two messages before compaction."
```

**Step 2: Run the SDK tests to verify they fail**

From `../open-agent-sdk-python`, run:

```bash
pytest tests/test_agent.py -k compact_history -q
```

Expected: FAIL with an `AttributeError` or import error because `Agent.compact_history` and `CompactionResult` do not exist yet.

**Step 3: Add the minimal typed result and agent method**

In `../open-agent-sdk-python/src/open_agent_sdk/types.py`, add the result type near `QueryResult`.

```python
@dataclass
class CompactionResult:
    compacted: bool
    summary: str = ""
    before_tokens: int = 0
    after_tokens: int = 0
    messages_before: int = 0
    messages_after: int = 0
    reason: str = ""
```

In `../open-agent-sdk-python/src/open_agent_sdk/agent.py`, add the imports and method.

```python
from open_agent_sdk.utils.compact import compact_conversation, create_auto_compact_state
from open_agent_sdk.utils.tokens import estimate_messages_tokens
from open_agent_sdk.types import CompactionResult
```

```python
async def compact_history(self) -> CompactionResult:
    await self._initialize()

    before_messages = list(self._history)
    before_tokens = estimate_messages_tokens(before_messages)

    if len(before_messages) < 2:
        return CompactionResult(
            compacted=False,
            before_tokens=before_tokens,
            after_tokens=before_tokens,
            messages_before=len(before_messages),
            messages_after=len(before_messages),
            reason="Need at least two messages before compaction.",
        )

    result = await compact_conversation(
        self._ensure_client(),
        self._resolve_model(),
        before_messages,
        create_auto_compact_state(),
    )

    compacted_messages = list(result["compacted_messages"])
    self._history = compacted_messages
    after_tokens = estimate_messages_tokens(compacted_messages)

    return CompactionResult(
        compacted=compacted_messages != before_messages,
        summary=str(result.get("summary", "")),
        before_tokens=before_tokens,
        after_tokens=after_tokens,
        messages_before=len(before_messages),
        messages_after=len(compacted_messages),
        reason="" if compacted_messages != before_messages else "Compaction produced no smaller history.",
    )
```

Export `CompactionResult` from `../open-agent-sdk-python/src/open_agent_sdk/__init__.py` beside the other public types.

**Step 4: Re-run the SDK tests**

From `../open-agent-sdk-python`, run:

```bash
pytest tests/test_agent.py -k compact_history -q
```

Expected: PASS.

**Step 5: Commit the SDK change**

From `../open-agent-sdk-python`, run:

```bash
git add src/open_agent_sdk/agent.py src/open_agent_sdk/types.py src/open_agent_sdk/__init__.py tests/test_agent.py
git commit -m "feat: add manual agent compaction API"
```

## Task 2: Expose `/compact` in cock-code chat

**Files:**
- Modify: `tests/test_chat.py:139-158, 281-300`
- Modify: `src/cock_code/cli.py:15-28, 312-357`
- Modify: `src/cock_code/rendering.py:106-119`

**Step 1: Write the failing cock-code tests**

Add one new command test and extend the help assertion in `tests/test_chat.py`.

```python
def test_run_chat_compacts_agent_history(monkeypatch) -> None:
    captured: dict[str, object] = {}
    prompts = iter(["/compact", "/exit"])

    class FakeAgent:
        async def compact_history(self):
            from types import SimpleNamespace
            captured["called"] = True
            return SimpleNamespace(
                compacted=True,
                summary="summary text",
                before_tokens=1200,
                after_tokens=240,
                reason="",
            )

        async def close(self) -> None:
            captured["closed"] = True

    notices: list[tuple[str, str, str]] = []
    monkeypatch.setattr(cli, "create_runtime_agent", lambda config: FakeAgent())
    monkeypatch.setattr(cli, "build_console", lambda: SilentConsole())
    monkeypatch.setattr(cli.Prompt, "ask", lambda *args, **kwargs: next(prompts))
    monkeypatch.setattr(cli, "render_notice", lambda console, title, message, style="yellow": notices.append((title, message, style)))

    exit_code = cli.asyncio.run(cli.run_chat(RuntimeConfig(model="m2")))

    assert exit_code == 0
    assert captured["called"] is True
    assert captured["closed"] is True
    assert notices[0][0] == "Compacted"
    assert "1200 → 240" in notices[0][1]
    assert "summary text" in notices[0][1]
```

Update the existing help test to also assert:

```python
assert "/compact" in output
```

**Step 2: Run the cock-code tests to verify they fail**

From `/home/kali/cock-code`, run:

```bash
pytest tests/test_chat.py -k compact -q
```

Expected: FAIL because `run_chat()` does not recognize `/compact` yet.

**Step 3: Implement the chat command with the existing UI helpers**

In `src/cock_code/cli.py`, add a new branch near `/clear`.

```python
if command.name == "compact":
    try:
        result = await agent.compact_history()
    except Exception as exc:
        render_notice(console, "Compact Error", str(exc), "red")
        continue

    title = "Compacted" if result.compacted else "Compaction skipped"
    style = "green" if result.compacted else "yellow"
    details = result.summary.strip() or result.reason or "No summary returned."
    message = f"Tokens: {result.before_tokens} → {result.after_tokens}\n\n{details}"
    render_notice(console, title, message, style)
    continue
```

In `src/cock_code/rendering.py`, update `render_help()` with one more row.

```python
table.add_row("/compact", "Summarize the current chat history")
```

**Step 4: Re-run the cock-code tests**

From `/home/kali/cock-code`, run:

```bash
pytest tests/test_chat.py -q
```

Expected: PASS.

**Step 5: Commit the cock-code change**

From `/home/kali/cock-code`, run:

```bash
git add src/cock_code/cli.py src/cock_code/rendering.py tests/test_chat.py
git commit -m "feat: add manual compact chat command"
```

## Task 3: Verify the integration does not regress existing behavior

**Files:**
- No new files
- Re-run tests in both repos

**Step 1: Run the targeted SDK test cluster**

From `../open-agent-sdk-python`, run:

```bash
pytest tests/test_agent.py tests/test_compact.py -q
```

Expected: PASS.

**Step 2: Run the targeted cock-code test cluster**

From `/home/kali/cock-code`, run:

```bash
pytest tests/test_chat.py tests/test_rendering.py tests/test_runtime.py -q
```

Expected: PASS.

**Step 3: Run the full test suite in both repos**

From `../open-agent-sdk-python`, run:

```bash
pytest -q
```

From `/home/kali/cock-code`, run:

```bash
pytest -q
```

Expected: PASS in both repos.

**Step 4: Manual smoke check**

From `/home/kali/cock-code`, run a real chat session:

```bash
uv run cock-code chat --model claude-sonnet-4-5
```

Inside chat:

```text
hello
/compact
/exit
```

Expected:
- `/compact` shows a green `Compacted` notice or a yellow `Compaction skipped` notice.
- `/exit` closes cleanly.
- If session persistence is enabled, the compacted history is what gets saved on close.

## Deferred follow-up (do not include in the first implementation)

Only start this after the main feature is merged and stable:

1. Add hook plumbing for `HookEvent.PRE_COMPACT` / `HookEvent.POST_COMPACT` in `../open-agent-sdk-python/src/open_agent_sdk/engine.py` and the manual `Agent.compact_history()` path.
2. Add tests in `../open-agent-sdk-python/tests/test_engine.py` and `../open-agent-sdk-python/tests/test_hooks.py` proving those events fire.
3. Decide whether compaction state should persist across resume; do not mix that architectural cleanup into the initial `/compact` feature branch.
