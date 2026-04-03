# Cock Code

<!-- A CLI coding agent powered by Rich and Open Agent SDK -->

Cock Code is a Rich-powered coding agent CLI built on the local Open Agent SDK at `../open-agent-sdk-python`.

## Environment

Cock Code reads these env vars and passes them into the SDK explicitly:

- `COCK_CODE_API_KEY`
- `COCK_CODE_BASE_URL`
- `COCK_CODE_MODEL`

## Install and run

```bash
uv run cock-code --help
uv run cock-code ask "Summarize this repository"
uv run cock-code chat
```

## Primary commands

```bash
uv run cock-code ask "Find the main entry point"
uv run cock-code chat --model claude-sonnet-4-5
```

Interactive chat supports slash commands such as `/clear`, `/model <name>`, `/status`, and `/exit`.

## Session management

```bash
uv run cock-code sessions list
uv run cock-code sessions show <session-id>
uv run cock-code sessions info <session-id>
uv run cock-code sessions fork <session-id> --new-id <new-session-id>
uv run cock-code sessions rename <session-id> "My session"
uv run cock-code sessions tag <session-id> alpha beta
uv run cock-code sessions delete <session-id>
```

## Tools and state inspection

```bash
uv run cock-code tools list
uv run cock-code state tasks
uv run cock-code state teams
uv run cock-code state mailboxes
uv run cock-code state mailboxes --agent reviewer
uv run cock-code state config
uv run cock-code state cron
uv run cock-code state plan
uv run cock-code state todos
```

## Shared runtime flags for ask/chat

```bash
uv run cock-code ask "Review src" \
  --model claude-sonnet-4-5 \
  --cwd /home/kali/cock-code \
  --resume existing-session \
  --allowed-tool Read \
  --disallowed-tool Bash
```

Supported shared flags:

- `--cwd`
- `--model`
- `--permission-mode`
- `--max-turns`
- `--max-budget-usd`
- `--max-tokens`
- `--thinking-budget`
- `--allowed-tool`
- `--disallowed-tool`
- `--session-id`
- `--resume`
- `--continue-session`
- `--fork-session`
- `--persist-session` / `--no-persist-session`
- `--sandbox`
- `--debug`
- `--include-partials`
- `--env KEY=VALUE`
- `--custom-header KEY=VALUE`
- `--extra-args-file`
- `--json-schema-file`
- `--agents-file`
- `--hooks-file`
- `--mcp-file`

JSON-backed options expect a file path whose contents can be loaded directly into the corresponding SDK option.
