---
name: authoring-extensions
description: Use when creating a new omp extension, registering LLM-callable tools, adding slash commands, or subscribing to session lifecycle events. Also use when asked to extend omp with custom behavior, safety guards, or TUI renderers.
user_invocable: true
---

# Authoring omp Extensions

Extensions are the primary way to add capabilities to omp. One TypeScript file can register tools the LLM calls, slash commands users invoke, and event handlers across the session lifecycle.

## Minimum Viable Extension

```ts
import type { ExtensionAPI } from "@oh-my-pi/pi-coding-agent";

export default function (pi: ExtensionAPI) {
  pi.on("session_start", async (_event, ctx) => {
    ctx.ui.notify("Loaded!", "info");
  });
}
```

Drop into `~/.omp/agent/extensions/my-ext.ts` and restart.

## Discovery Paths

omp loads from (first seen wins, deduplicated by absolute path):

| Source | Path |
|---|---|
| Project | `<cwd>/.omp/extensions/` |
| User | `~/.omp/agent/extensions/` |
| CLI | `omp --extension ./my-ext.ts` or `-e` |
| Config | `extensions:` array in `~/.omp/agent/config.yml` |
| Plugins | Marketplace-installed packages with `omp.extensions` manifest |

For directories: resolves `package.json` manifest → `index.ts` → `index.js`.

## Core Patterns

### Registering a Tool

```ts
const z = pi.zod;  // zod/v4, canonical for schemas

pi.registerTool({
  name: "my_tool",              // snake_case, unique
  label: "My Tool",             // human-readable for TUI
  description: "What this tool does — shown to the LLM",
  parameters: z.object({
    text: z.string().describe("Input text"),
    limit: z.number().default(10).optional(),
  }),
  async execute(_toolCallId, params, signal, onUpdate, _ctx) {
    if (signal?.aborted) {
      return { content: [{ type: "text", text: "Cancelled" }] };
    }
    onUpdate?.({ content: [{ type: "text", text: "Working..." }] });
    // ... work ...
    return {
      content: [{ type: "text", text: "Result" }],
      details: { /* structured data for history */ },
    };
  },
});
```

Tool fields: `hidden` (omit from LLM context), `defaultInactive` (opt-in only), `deferrable` (async execution). Optional: `renderCall`/`renderResult` for custom TUI visualization, `onSession` for session lifecycle within the tool.

### Registering a Slash Command

```ts
pi.registerCommand("my-cmd", {
  description: "What /my-cmd does",
  handler: async (args, ctx) => {
    // args: everything after /my-cmd in the prompt
    // ctx: ExtensionCommandContext — session controls below
    ctx.ui.notify(`Args: ${args}`, "info");
  },
});
```

Command context session controls (ONLY from command handlers, not event handlers):

| Method | Effect |
|---|---|
| `ctx.waitForIdle()` | Block until agent finishes streaming |
| `ctx.newSession(opts?)` | Open fresh session |
| `ctx.switchSession(path)` | Switch to existing session |
| `ctx.branch(entryId)` | Fork from history entry |
| `ctx.navigateTree(id, opts?)` | Jump to session tree node |
| `ctx.compact(opts?)` | Trigger context compaction |
| `ctx.reload()` | Reload runtime (terminal for current handler) |

### Subscribing to Events

```ts
pi.on("tool_call", async (event, ctx) => {
  // event.toolName, event.input, event.toolCallId
  if (event.toolName !== "bash") return;
  const cmd = String((event.input as any).command ?? "");
  if (cmd.includes("rm -rf /")) {
    return { block: true, reason: "Blocked by safety policy" };
  }
});
```

Key events: `session_start`, `session_shutdown`, `input`, `turn_start`, `turn_end`, `tool_call` (pre-exec, can block), `tool_result` (post-exec, can patch), `before_agent_start`, `before_provider_request`, `after_provider_response`, `message_start`/`message_update`/`message_end`.

Pre-events that support cancellation: `session_before_switch`, `session_before_branch`, `session_before_compact`, `session_before_tree` — return `{ cancel: true }`.

### Sending Messages

```ts
// Interrupt current run (default)
pi.sendMessage({ content: "Stop — tests failed." }, { deliverAs: "steer" });

// Queue after current run
pi.sendMessage({ content: "Follow-up" }, { deliverAs: "followUp" });

// Inject on next user prompt
pi.sendMessage({ content: "Next turn context" }, { deliverAs: "nextTurn" });

// As user message (goes through prompt flow)
pi.sendUserMessage("User-facing message", { deliverAs: "steer" });
```

### Extension vs Hook

| Need | Use |
|---|---|
| Tools + commands + events in one module | Extension |
| Pure event interception | Extension (preferred) or Hook |
| Custom message/TUI renderers | Extension only |
| Provider registration | Extension only |
| Marketplace plugin | Extension with `package.json` manifest |

Extensions are a strict superset of hooks. Always start with an extension.

## Common Mistakes

### Fabricating your own interfaces

The #1 mistake: inventing types instead of importing the real API.

```ts
// ❌ WRONG — completely fabricated, will not work
interface ExtensionApi {
  registerTool(name: string, description: string, handler: (params: any) => string): void;
  registerCommand(name: string, description: string, handler: (params: any) => void): void;
  onSessionStart(handler: (ctx: any) => void): void;
}
export default function register(api: ExtensionApi) { ... }

// ✅ CORRECT — import the real type
import type { ExtensionAPI } from "@oh-my-pi/pi-coding-agent";
export default function (pi: ExtensionAPI) { ... }
```

**Never** write your own `ExtensionApi` interface, `ToolContext`, `SessionStartContext`, or similar types. The real API is `ExtensionAPI` from `@oh-my-pi/pi-coding-agent`. The factory receives it directly — no wrapper interfaces.

### Wrong registration signatures

```ts
// ❌ WRONG — positional args
pi.registerTool("name", "description", (params) => { ... });
pi.registerCommand("/name", "description", (args, ctx) => { ... });

// ✅ CORRECT — config objects
pi.registerTool({ name: "name", description: "...", parameters: z.object({}), execute() { ... } });
pi.registerCommand("name", { description: "...", handler(args, ctx) { ... } });
//                   ^ no slash prefix         ^ config object, not positional
```

### Wrong tool return type

```ts
// ❌ WRONG — returning a plain string
async execute() { return "result"; }

// ✅ CORRECT — content array with type/text wrapper
async execute() {
  return { content: [{ type: "text", text: "result" }], details: {} };
}
```

### Missing Zod schemas on tools

Without `parameters: z.object({...})`, the LLM cannot understand what inputs the tool expects. Every tool MUST declare its parameters with Zod.

### Calling runtime actions during load

```ts
// ❌ WRONG — throws ExtensionRuntimeNotInitializedError
export default function (pi: ExtensionAPI) {
  pi.sendMessage({ content: "Hello" });  // NO — not initialized yet
  pi.on("session_start", async () => {
    pi.sendMessage({ content: "Hello" });  // YES — inside event handler
  });
}
```

Register during load. Act from handlers, tools, or commands.

### Not checking signal.aborted in long-running tools

```ts
async execute(_id, params, signal, _onUpdate, _ctx) {
  for (const item of items) {
    if (signal?.aborted) return { content: [{ type: "text", text: "Cancelled" }] };
    await process(item);
  }
}
```

### Throwing in tool_call handlers

`tool_call` errors are fail-closed — the tool is blocked. Always catch internally and return `{ block: true, reason: "..." }` instead of throwing.

### name vs label confusion

- `name`: snake_case, machine identifier, used in tool calls
- `label`: Human-readable, shown in TUI. Defaults to name if omitted.
- `description`: Shown to the LLM — be precise about what the tool does and when to use it.

### Ignoring onUpdate for long operations

Call `onUpdate` with partial results so the user sees progress, not a blank tool block until completion.

## Quick Reference

```ts
// Minimal skeleton
import type { ExtensionAPI } from "@oh-my-pi/pi-coding-agent";
export default function (pi: ExtensionAPI) {
  const z = pi.zod;
  pi.setLabel("My Extension");

  // Event hooks
  pi.on("session_start", async (ev, ctx) => { /* ctx.ui.notify, ctx.cwd */ });
  pi.on("tool_call", async (ev, ctx) => { /* ev.toolName, can block */ });
  pi.on("turn_end", async (ev, ctx) => { /* ctx.getContextUsage() */ });

  // Slash command
  pi.registerCommand("name", {
    description: "...",
    handler: async (args, ctx) => { /* ctx.waitForIdle, ctx.newSession */ },
  });

  // LLM tool
  pi.registerTool({
    name: "my_tool", label: "My Tool", description: "...",
    parameters: z.object({ query: z.string() }),
    async execute(id, params, signal, onUpdate, ctx) {
      if (signal?.aborted) return { content: [{ type: "text", text: "Cancelled" }] };
      return { content: [{ type: "text", text: "Done" }], details: {} };
    },
  });
}
```

## Debugging

```bash
omp --log-level debug                    # See extension load diagnostics
```

```yaml
# ~/.omp/agent/config.yml — disable without removing file
disabledExtensions:
  - extension-module:my-ext   # name derived from filename stem
```

## Constraints

- Command names must not clash with built-ins (skipped with diagnostic log).
- Reserved shortcuts silently ignored: `ctrl+c/d/z/k/p/l/o/t/g/q`, `alt+m/enter`, `shift+tab`, `shift+ctrl+p`, `escape`, `enter`.
- Extensions are NOT sandboxed — same process, shared event bus.
- Call `reload()` only as the terminal action in a command handler — it destroys the current context.

## Red Flags — STOP and Re-read This Skill

- Writing your own `interface ExtensionApi` or `ToolContext`
- Using positional args for `registerTool`/`registerCommand`
- Returning a plain string from tool `execute`
- No `parameters` Zod schema on a tool
- Calling `pi.sendMessage()` at the top level of the factory
- Including a `/` prefix in the command name string

**All of these mean: you are fabricating APIs. Re-read the patterns above.**

| Excuse | Reality |
|---|---|
| "I'll just sketch the interface first" | The real interface exists. Import it. Fabricated interfaces are dead code. |
| "Positional args are cleaner" | The real API takes config objects. Positional args won't work. |
| "My tool is simple, no Zod needed" | Without Zod, the LLM cannot call your tool. |
| "I know TypeScript, I can infer the API" | The omp API has specific shapes. Inferring from general TS knowledge produces wrong code. |
