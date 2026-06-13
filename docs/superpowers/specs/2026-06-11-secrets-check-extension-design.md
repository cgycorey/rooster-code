# Secrets Check Extension — Design

## Summary

An OMP extension that scans all tool inputs and outputs for secrets (API keys, tokens, private keys, connection strings) using battle-tested patterns. Supports two actions per tool: **block** (prevent the tool call from executing or its result from reaching the LLM) and **redact** (strip the secret, replace with a placeholder, log the event).

No config file required. Works out of the box with sensible defaults.

## Architecture

One extension file: `index.ts` (default export factory). Uses an npm library for secret patterns. Optional JSON config at `~/.omp/agent/secrets-check.json` for overrides.

```
~/.omp/agent/extensions/secrets-check/
  index.ts          — extension factory
  package.json      — omp.extensions manifest (optional, for plugin installs)
```

## Event hooks

The extension hooks two events:

### `tool_call` (pre-execution)

1. Deep-scan `event.input` for secrets.
2. If found:
   - **block mode**: return `{ block: true, reason: "Secret detected: <type>" }`. Tool never runs.
   - **redact mode**: mutate `event.input` in place, replacing secrets with `[REDACTED:<type>]`. Let tool execute.
3. Log detection via `pi.logger` and notify via `ctx.ui.notify`.

### `tool_result` (post-execution)

1. Deep-scan the result content for secrets the tool may have produced.
2. If found:
   - **block mode**: patch result to a safe message (`"Result blocked: contained secrets"`) + `isError: true`.
   - **redact mode**: patch result content, replacing secrets with `[REDACTED:<type>]`.
3. Notify user.

Redaction on `tool_result` is critical — a `read` of `.env` or a `search` hitting a config file can surface secrets the user didn't know were there.

## Default tool mode assignments

| Mode | Tools |
|---|---|
| **block** | `bash`, `write`, `edit`, `browser`, `web_search`, `task`, `eval`, `irc` |
| **redact** | `read`, `search`, `find`, `lsp`, `ast_grep`, `debug` |
| **passthrough** | `ask`, `todo`, `job`, `resolve` |

Rationale: tools that can exfiltrate (network, shell, file write) get blocked. Tools that read local state get redacted. Pure UI/coordination tools pass through.

## Pattern library

Use `detect-secrets` or similar npm package for pattern definitions. The extension imports the library, extracts its regex rules, and compiles them once at load time.

If no suitable library is available as a Bun-compatible npm package, fall back to an embedded set of patterns ported from GitHub's secret scanning documentation (~200 patterns covering AWS, GCP, Azure, GitHub, GitLab, OpenAI, Anthropic, Slack, Stripe, JWT, private keys, connection strings, etc.).

## Config file (optional)

`~/.omp/agent/secrets-check.json`:

```json
{
  "toolModes": {
    "bash": "redact",
    "browser": "passthrough"
  },
  "excludePatterns": ["^ghp_test", "sk-test-"],
  "additionalPatterns": [
    { "name": "Custom Internal Key", "regex": "corp-key-[a-f0-9]{32}" }
  ],
  "notifyOnly": false
}
```

- `toolModes`: override default mode per tool. Values: `"block"`, `"redact"`, `"passthrough"`.
- `excludePatterns`: regex patterns that, if matched, suppress detection (e.g., known test keys).
- `additionalPatterns`: custom secrets to detect.
- `notifyOnly`: if `true`, never block/redact — just notify. "Audit mode."

All fields optional. Missing config = zero-config defaults.

## Deep scan behavior

Secrets can hide anywhere in tool inputs — nested objects, arrays, strings deep in a parameter tree. The scanner walks every string value recursively before applying regex. Same for result content arrays.

Performance: compile all patterns once. Walk with early exit — stop on first match per value. For large tool results, scan is bounded by O(strings × patterns) but in practice <1ms for typical tool payloads.

## Notifications and logging

- On detection: `ctx.ui.notify("Secret detected: <type> in <tool>", "warn")`
- Logged at `warn` level via `pi.logger` with tool name, secret type, and action taken.
- No secret values ever logged or displayed — only the type label.

## Testing

- Unit tests for the scanner against known secret patterns and non-secrets.
- Integration: load extension in a test session, verify `tool_call` blocking and `tool_result` redaction.
- Test that config overrides work and that missing config falls back to defaults.

## Non-goals

- No custom TUI renderer. Notifications + logging are sufficient.
- No slash commands. Pure background policy — no user interaction needed.
- No persistence of scan history. Logs go to the existing omp log stream.
- No provider-level scanning. This is tool-scoped only.
