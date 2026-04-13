# Cock Code

<!-- A CLI coding agent powered by Rich and Open Agent SDK -->

Cock Code is a Rich-powered coding agent CLI built on Open Agent SDK.

## Implemented agentic features

Cock Code currently includes:

- **ask** and **interactive chat** agent workflows for one-shot queries and persistent terminal sessions
- **session management** for listing, resuming, forking, renaming, tagging, and inspecting saved conversations
- **background agents and tasks** with notifications, task output inspection, waiting, and context injection on completion
- **multi-agent team orchestration** with team creation, non-blocking dispatch, team status, and inter-member messaging
- **named/custom agents** loaded from JSON agent definitions
- **tool-gated execution** with allowed/disallowed tool controls and permission-mode support
- **MCP server integration** through SDK-backed MCP configuration
- **hooks-file passthrough** into the SDK runtime configuration
- **state inspection commands** for tasks, teams, mailboxes, config, cron, plans, and todos
- **manual session compaction** for trimming conversation history while preserving working context

> [!IMPORTANT]
> `open-agent-sdk` is **not published yet**. Right now the supported source is
> **cgycorey's fork**: `https://github.com/cgycorey/open-agent-sdk-python`.
>
> This repository currently expects the SDK as a **local sibling checkout** at
> `../open-agent-sdk-python` via `tool.uv.sources` in `pyproject.toml`.

## Environment

Cock Code reads these env vars and passes them into the SDK explicitly:

- `COCK_CODE_API_KEY`
- `COCK_CODE_BASE_URL`
- `COCK_CODE_MODEL`

## Install and run

### One-command bootstrap (recommended)

After cloning `cock-code`, run:

```bash
./scripts/bootstrap.sh
```

What it does:

- clones **cgycorey's** `open-agent-sdk-python` into the required sibling path
  (`../open-agent-sdk-python`) if it is missing
- reuses the existing sibling SDK checkout if you already have one
- runs `uv sync` for this repo

This gives you the layout `pyproject.toml` already expects.

### Fresh setup from scratch

If you have not cloned `cock-code` yet, this is the simplest flow:

```bash
git clone https://github.com/cgycorey/cock-code
cd cock-code
./scripts/bootstrap.sh
```

### Supported setup (manual)

Clone `cock-code` and `open-agent-sdk-python` side by side so the existing
`uv` source override works without any edits:

```bash
git clone https://github.com/cgycorey/open-agent-sdk-python
git clone https://github.com/cgycorey/cock-code

cd cock-code
uv sync
```

Your directory layout should look like this:

```text
parent-dir/
├── cock-code/
└── open-agent-sdk-python/
```

### If you already cloned `cock-code` somewhere else

You still need the SDK checkout from cgycorey's repository:

```bash
git clone https://github.com/cgycorey/open-agent-sdk-python ../open-agent-sdk-python
uv sync
```

If you do not want the sibling layout, update `tool.uv.sources.open-agent-sdk`
in `pyproject.toml` to point at your local checkout before running `uv sync`.

### Run

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
