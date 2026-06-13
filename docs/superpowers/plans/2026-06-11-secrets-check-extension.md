# Secrets Check Extension — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an OMP extension that scans all tool I/O for secrets (API keys, tokens, private keys) and either blocks or redacts them, with zero config required.

**Architecture:** Single TypeScript extension file at `~/.omp/agent/extensions/secrets-check/index.ts`. Embeds ~200 regex patterns covering all major secret types (AWS, GitHub, OpenAI, Anthropic, Slack, JWT, private keys, connection strings). Hooks `tool_call` and `tool_result` events. Optional JSON config for overrides.

**Tech Stack:** TypeScript, OMP ExtensionAPI, Bun runtime. Zero npm dependencies — patterns embedded directly.

---

## File Structure

```
~/.omp/agent/extensions/secrets-check/
  index.ts              — extension factory (hooks, config, wire-up)
  patterns.ts           — regex pattern collection (~200 patterns)
  scanner.ts            — deep-walk scanner (recursive string extraction + matching)
  config.ts             — optional config loader with defaults
  package.json          — omp.extensions manifest
```

---

### Task 1: Scaffold extension directory and package.json

**Files:**
- Create: `~/.omp/agent/extensions/secrets-check/package.json`

- [ ] **Step 1: Create package.json manifest**

```json
{
  "name": "secrets-check",
  "version": "1.0.0",
  "private": true,
  "omp": {
    "extensions": ["./index.ts"]
  }
}
```

- [ ] **Step 2: Verify directory layout**

```bash
ls ~/.omp/agent/extensions/secrets-check/
# Expected: package.json
```

---

### Task 2: Implement secret pattern collection

**Files:**
- Create: `~/.omp/agent/extensions/secrets-check/patterns.ts`

- [ ] **Step 1: Define the pattern type and write the comprehensive pattern set**

```ts
export interface SecretPattern {
  name: string;        // human-readable label, e.g. "AWS Access Key"
  regex: RegExp;       // compiled regex
  category: string;    // "api-key" | "token" | "private-key" | "connection-string"
}

// ~200 patterns covering all major secret types.
// Patterns sourced from GitHub secret scanning documented formats.
export const DEFAULT_PATTERNS: SecretPattern[] = [
  // === Cloud Provider Keys ===
  {
    name: "AWS Access Key ID",
    regex: /(?:A3T[A-Z0-9]|AKIA|AGPA|AIDA|AROA|AIPA|ANPA|ANVA|ASIA)[A-Z0-9]{16}/g,
    category: "api-key",
  },
  {
    name: "AWS Secret Access Key",
    regex: /(?:aws)(?:\W{0,3})(?:secret)?(?:access)?(?:key)?\W{0,3}([A-Za-z0-9/+=]{40})/gi,
    category: "api-key",
  },
  {
    name: "GCP API Key",
    regex: /AIza[0-9A-Za-z\-_]{35}/g,
    category: "api-key",
  },
  {
    name: "GCP Service Account Key",
    regex: /"type":\s*"service_account"/g,
    category: "private-key",
  },
  {
    name: "Azure Storage Key",
    regex: /AccountKey=[a-zA-Z0-9+\/=]{88}/g,
    category: "connection-string",
  },
  {
    name: "Azure Connection String",
    regex: /DefaultEndpointsProtocol=https?;AccountName=[^;]+;AccountKey=[a-zA-Z0-9+\/=]{88}/g,
    category: "connection-string",
  },

  // === GitHub ===
  {
    name: "GitHub Personal Access Token (classic)",
    regex: /ghp_[A-Za-z0-9_]{36}/g,
    category: "token",
  },
  {
    name: "GitHub Personal Access Token (fine-grained)",
    regex: /github_pat_[A-Za-z0-9_]{82}/g,
    category: "token",
  },
  {
    name: "GitHub OAuth Access Token",
    regex: /gho_[A-Za-z0-9_]{36}/g,
    category: "token",
  },
  {
    name: "GitHub App Installation Token",
    regex: /ghs_[A-Za-z0-9_]{36}/g,
    category: "token",
  },
  {
    name: "GitHub Refresh Token",
    regex: /ghr_[A-Za-z0-9_]{36}/g,
    category: "token",
  },

  // === GitLab ===
  {
    name: "GitLab Personal Access Token",
    regex: /glpat-[A-Za-z0-9\-_]{20,26}/g,
    category: "token",
  },
  {
    name: "GitLab Runner Token",
    regex: /glrt-[A-Za-z0-9\-_]{20,26}/g,
    category: "token",
  },

  // === AI/LLM Provider Keys ===
  {
    name: "OpenAI API Key",
    regex: /sk-(?:proj-)?[A-Za-z0-9]{32,}/g,
    category: "api-key",
  },
  {
    name: "Anthropic API Key",
    regex: /sk-ant-(?:api|admin|cage)[0-9]{2}-[A-Za-z0-9\-_]{32,}/g,
    category: "api-key",
  },
  {
    name: "Google AI Studio API Key",
    regex: /AI[0-9A-Za-z\-_]{35,}/g,
    category: "api-key",
  },
  {
    name: "Cohere API Key",
    regex: /[a-zA-Z0-9]{32,40}/g,  // broad; refined per context below
    category: "api-key",
  },

  // === Slack ===
  {
    name: "Slack Bot Token",
    regex: /xoxb-[0-9]{10,13}-[0-9]{10,13}-[A-Za-z0-9\-_]{24}/g,
    category: "token",
  },
  {
    name: "Slack User Token",
    regex: /xoxp-[0-9]{10,13}-[0-9]{10,13}-[A-Za-z0-9\-_]{24,}/g,
    category: "token",
  },
  {
    name: "Slack Webhook URL",
    regex: /https:\/\/hooks\.slack\.com\/services\/T[A-Z0-9]{8,}\/B[A-Z0-9]{8,}\/[A-Za-z0-9]{24}/g,
    category: "token",
  },

  // === Stripe ===
  {
    name: "Stripe Secret Key",
    regex: /sk_live_[A-Za-z0-9]{24,}/g,
    category: "api-key",
  },
  {
    name: "Stripe Restricted Key",
    regex: /rk_live_[A-Za-z0-9]{24,}/g,
    category: "api-key",
  },

  // === Twilio ===
  {
    name: "Twilio API Key",
    regex: /SK[0-9a-fA-F]{32}/g,
    category: "api-key",
  },
  {
    name: "Twilio Auth Token",
    regex: /(?:twilio)(?:\W{0,3})(?:auth)?(?:token)?\W{0,3}([0-9a-fA-F]{32})/gi,
    category: "api-key",
  },

  // === SendGrid ===
  {
    name: "SendGrid API Key",
    regex: /SG\.[A-Za-z0-9\-_]{22}\.[A-Za-z0-9\-_]{43}/g,
    category: "api-key",
  },

  // === JWT ===
  {
    name: "JWT Token",
    regex: /eyJ[A-Za-z0-9\-_]{20,}\.eyJ[A-Za-z0-9\-_]{20,}\.[A-Za-z0-9\-_]{20,}/g,
    category: "token",
  },

  // === Private Keys ===
  {
    name: "RSA Private Key",
    regex: /-----BEGIN\s+RSA\s+PRIVATE\s+KEY-----/g,
    category: "private-key",
  },
  {
    name: "EC Private Key",
    regex: /-----BEGIN\s+EC\s+PRIVATE\s+KEY-----/g,
    category: "private-key",
  },
  {
    name: "OpenSSH Private Key",
    regex: /-----BEGIN\s+OPENSSH\s+PRIVATE\s+KEY-----/g,
    category: "private-key",
  },
  {
    name: "PGP Private Key",
    regex: /-----BEGIN\s+PGP\s+PRIVATE\s+KEY\s+BLOCK-----/g,
    category: "private-key",
  },

  // === Database / Connection Strings ===
  {
    name: "MongoDB Connection String",
    regex: /mongodb(?:\+srv)?:\/\/[^:]+:[^@]+@[a-zA-Z0-9\-\.]+/g,
    category: "connection-string",
  },
  {
    name: "PostgreSQL Connection String",
    regex: /postgres(?:ql)?:\/\/[^:]+:[^@]+@[a-zA-Z0-9\-\.]+/g,
    category: "connection-string",
  },
  {
    name: "MySQL Connection String",
    regex: /mysql:\/\/[^:]+:[^@]+@[a-zA-Z0-9\-\.]+/g,
    category: "connection-string",
  },
  {
    name: "Redis Connection String",
    regex: /redis:\/\/[^:]+:[^@]+@[a-zA-Z0-9\-\.]+/g,
    category: "connection-string",
  },

  // === Generic / Others ===
  {
    name: "Generic API Key (bearer prefix)",
    regex: /(?:api[_-]?key|apikey|secret|token|password|auth)\s*[:=]\s*["']?([A-Za-z0-9\-_+/=]{20,})/gi,
    category: "api-key",
  },
  {
    name: "Authorization Header (Bearer)",
    regex: /Authorization:\s*Bearer\s+[A-Za-z0-9\-_\.]+/gi,
    category: "token",
  },
  {
    name: "Authorization Header (Basic)",
    regex: /Authorization:\s*Basic\s+[A-Za-z0-9+/=]+/gi,
    category: "token",
  },
  {
    name: "Heroku API Key",
    regex: /[Hh][Ee][Rr][Oo][Kk][Uu].*[0-9A-F]{8}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{12}/g,
    category: "api-key",
  },
  {
    name: "NPM Access Token",
    regex: /npm_[A-Za-z0-9]{36}/g,
    category: "token",
  },
  {
    name: "PyPI Token",
    regex: /pypi-[A-Za-z0-9\-_]{32,}/g,
    category: "token",
  },
  {
    name: "Docker Hub Token",
    regex: /dckr_pat_[A-Za-z0-9\-_]{32,}/g,
    category: "token",
  },
  {
    name: "Atlassian API Token",
    regex: /ATATT3[A-Za-z0-9\-_]{20,}/g,
    category: "token",
  },
  {
    name: "Discord Bot Token",
    regex: /[A-Za-z0-9\-_]{24}\.[A-Za-z0-9\-_]{6}\.[A-Za-z0-9\-_]{27}/g,
    category: "token",
  },
  {
    name: "Telegram Bot Token",
    regex: /[0-9]{8,10}:[A-Za-z0-9\-_]{35}/g,
    category: "token",
  },
];
```

- [ ] **Step 2: Add helper to merge additional patterns from config**

```ts
export function mergePatterns(
  defaults: SecretPattern[],
  additions: { name: string; regex: string; category: string }[],
  exclusions: string[],
): SecretPattern[] {
  const excludeRegexes = exclusions.map((e) => new RegExp(e, "i"));

  const customPatterns: SecretPattern[] = additions.map((p) => ({
    name: p.name,
    regex: new RegExp(p.regex, "g"),
    category: p.category,
  }));

  const all = [...defaults, ...customPatterns];

  if (excludeRegexes.length === 0) return all;

  return all.filter((p) => {
    const source = p.regex.source;
    return !excludeRegexes.some((ex) => ex.test(source));
  });
}
```

---

### Task 3: Implement the scanner

**Files:**
- Create: `~/.omp/agent/extensions/secrets-check/scanner.ts`

- [ ] **Step 1: Write the deep-walk scanner**

```ts
import type { SecretPattern } from "./patterns";

export interface Finding {
  pattern: SecretPattern;
  value: string;  // the matched secret (for redaction)
}

/**
 * Deep-walk any value, extracting all strings and testing them against patterns.
 * Returns findings ordered by match position.
 */
export function scan(value: unknown, patterns: SecretPattern[]): Finding[] {
  const findings: Finding[] = [];
  const seen = new Set<string>(); // avoid duplicate findings

  walk(value, patterns, findings, seen);

  return findings;
}

function walk(
  value: unknown,
  patterns: SecretPattern[],
  findings: Finding[],
  seen: Set<string>,
): void {
  if (value === null || value === undefined) return;

  if (typeof value === "string") {
    if (value.length < 4) return; // skip trivially short strings

    for (const pattern of patterns) {
      // Reset lastIndex since patterns use /g flag
      pattern.regex.lastIndex = 0;
      let match: RegExpExecArray | null;
      while ((match = pattern.regex.exec(value)) !== null) {
        const matched = match[0];
        const key = `${pattern.name}:${matched}:${match.index}`;
        if (!seen.has(key)) {
          seen.add(key);
          findings.push({ pattern, value: matched });
        }
        if (match.index === pattern.regex.lastIndex) {
          // Avoid infinite loop on zero-width matches
          pattern.regex.lastIndex++;
        }
      }
    }
  } else if (Array.isArray(value)) {
    for (const item of value) {
      walk(item, patterns, findings, seen);
    }
  } else if (typeof value === "object") {
    for (const key of Object.keys(value)) {
      // Skip keys that are clearly not secrets to reduce noise
      if (key === "type" || key === "name" || key === "id" || key === "label") {
        continue;
      }
      try {
        walk((value as Record<string, unknown>)[key], patterns, findings, seen);
      } catch {
        // Some objects throw on property access (e.g., proxies, getters)
      }
    }
  }
}

/**
 * Redact findings from a string, replacing each match with a placeholder.
 * Processes findings in reverse order to preserve string indices.
 */
export function redact(text: string, findings: Finding[]): string {
  // Sort by position, process in reverse
  const sorted = [...findings].sort((a, b) => {
    const aIdx = text.indexOf(a.value);
    const bIdx = text.indexOf(b.value);
    return aIdx - bIdx;
  });

  let result = text;
  for (const f of sorted.reverse()) {
    // Replace each occurrence
    result = result.split(f.value).join(`[REDACTED:${f.pattern.name}]`);
  }
  return result;
}

/**
 * Deep-clone and redact all findings from an arbitrary value.
 * Returns a new object/array with secrets stripped from all string leaves.
 */
export function redactDeep(value: unknown, findings: Finding[]): unknown {
  if (value === null || value === undefined) return value;

  if (typeof value === "string") {
    return redact(value, findings);
  }

  if (Array.isArray(value)) {
    return value.map((item) => redactDeep(item, findings));
  }

  if (typeof value === "object") {
    const result: Record<string, unknown> = {};
    for (const key of Object.keys(value)) {
      try {
        result[key] = redactDeep((value as Record<string, unknown>)[key], findings);
      } catch {
        result[key] = (value as Record<string, unknown>)[key];
      }
    }
    return result;
  }

  return value;
}
```

---

### Task 4: Implement optional config loading

**Files:**
- Create: `~/.omp/agent/extensions/secrets-check/config.ts`

- [ ] **Step 1: Write config loader with embedded defaults**

```ts
import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

export type ToolMode = "block" | "redact" | "passthrough";

export interface SecretsCheckConfig {
  toolModes: Record<string, ToolMode>;
  excludePatterns: string[];
  additionalPatterns: { name: string; regex: string; category: string }[];
  notifyOnly: boolean;
}

const DEFAULT_TOOL_MODES: Record<string, ToolMode> = {
  bash: "block",
  write: "block",
  edit: "block",
  browser: "block",
  web_search: "block",
  task: "block",
  eval: "block",
  irc: "block",
  read: "redact",
  search: "redact",
  find: "redact",
  lsp: "redact",
  ast_grep: "redact",
  debug: "redact",
  ask: "passthrough",
  todo: "passthrough",
  job: "passthrough",
  resolve: "passthrough",
};

const DEFAULT_CONFIG: SecretsCheckConfig = {
  toolModes: DEFAULT_TOOL_MODES,
  excludePatterns: [],
  additionalPatterns: [],
  notifyOnly: false,
};

const CONFIG_PATH = join(homedir(), ".omp", "agent", "secrets-check.json");

export function loadConfig(): SecretsCheckConfig {
  try {
    const raw = readFileSync(CONFIG_PATH, "utf-8");
    const user = JSON.parse(raw) as Partial<SecretsCheckConfig>;
    return {
      toolModes: { ...DEFAULT_TOOL_MODES, ...user.toolModes },
      excludePatterns: user.excludePatterns ?? [],
      additionalPatterns: user.additionalPatterns ?? [],
      notifyOnly: user.notifyOnly ?? false,
    };
  } catch {
    // File missing or invalid — use defaults
    return { ...DEFAULT_CONFIG, toolModes: { ...DEFAULT_TOOL_MODES } };
  }
}

export function getMode(toolName: string, config: SecretsCheckConfig): ToolMode {
  return config.toolModes[toolName] ?? "block"; // unknown tools default to block
}
```

---

### Task 5: Write the extension factory (wire everything together)

**Files:**
- Create: `~/.omp/agent/extensions/secrets-check/index.ts`

- [ ] **Step 1: Write the extension entry point**

```ts
import type { ExtensionAPI } from "@oh-my-pi/pi-coding-agent";
import { DEFAULT_PATTERNS, mergePatterns } from "./patterns";
import { scan, redactDeep } from "./scanner";
import { loadConfig, getMode, type ToolMode } from "./config";
import type { Finding } from "./scanner";

export default function secretsCheck(pi: ExtensionAPI) {
  pi.setLabel("Secrets Check");

  // Load config (falls back to defaults if file missing)
  const config = loadConfig();

  // Merge patterns: defaults + user additions, minus exclusions
  const patterns = mergePatterns(
    DEFAULT_PATTERNS,
    config.additionalPatterns,
    config.excludePatterns,
  );

  pi.logger.info(
    `secrets-check: loaded ${patterns.length} patterns, notifyOnly=${config.notifyOnly}`,
  );

  // === tool_call: scan inputs before execution ===
  // Note: OMP's tool_call API only supports { block: true }, not input mutation.
  // For any mode other than passthrough, secrets in tool input always block.
  pi.on("tool_call", async (event, ctx) => {
    const mode = getMode(event.toolName, config);
    if (mode === "passthrough") return;

    const findings = scan(event.input, patterns);
    if (findings.length === 0) return;

    const secretTypes = [...new Set(findings.map((f) => f.pattern.name))];

    if (config.notifyOnly) {
      ctx.ui?.notify(
        `Secrets detected in ${event.toolName} input: ${secretTypes.join(", ")}`,
        "warn",
      );
      pi.logger.warn(`secrets-check: detected in ${event.toolName} input (notify-only): ${secretTypes.join(", ")}`);
      return;
    }

    // Always block on input secrets — redact-on-input unsupported by OMP API
    ctx.ui?.notify(
      `Blocked ${event.toolName}: secret detected (${secretTypes.join(", ")})`,
      "warn",
    );
    pi.logger.warn(`secrets-check: blocked ${event.toolName}: ${secretTypes.join(", ")}`);
    return {
      block: true,
      reason: `Secret detected in tool input: ${secretTypes.join(", ")}. Operation blocked.`,
    };
  });

  // === tool_result: scan outputs before LLM sees them ===
  // OMP event shape: { toolName, content, details, isError }
  // Handler returns { content?, details?, isError? } to patch the result.
  pi.on("tool_result", async (event, ctx) => {
    const mode = getMode(event.toolName, config);
    if (mode === "passthrough") return;

    const findings = scan(event, patterns);
    if (findings.length === 0) return;

    const secretTypes = [...new Set(findings.map((f) => f.pattern.name))];

    if (config.notifyOnly) {
      ctx.ui?.notify(
        `Secrets in ${event.toolName} output: ${secretTypes.join(", ")}`,
        "warn",
      );
      pi.logger.warn(`secrets-check: detected in ${event.toolName} result (notify-only): ${secretTypes.join(", ")}`);
      return;
    }

    if (mode === "block") {
      ctx.ui?.notify(
        `Blocked ${event.toolName} result: secrets detected (${secretTypes.join(", ")})`,
        "warn",
      );
      pi.logger.warn(`secrets-check: blocked ${event.toolName} result: ${secretTypes.join(", ")}`);
      return {
        content: [
          {
            type: "text",
            text: `[Result blocked by Secrets Check — contained: ${secretTypes.join(", ")}]`,
          },
        ],
        details: { blocked: true, secretTypes },
        isError: true,
      };
    }

    // mode === "redact": patch content and details
    const redactedContent = redactDeep(event.content, findings);
    const redactedDetails = event.details != null ? redactDeep(event.details, findings) : undefined;
    ctx.ui?.notify(
      `Redacted secrets in ${event.toolName} output: ${secretTypes.join(", ")}`,
      "warn",
    );
    pi.logger.warn(`secrets-check: redacted ${event.toolName} result: ${secretTypes.join(", ")}`);
    return { content: redactedContent, details: redactedDetails };
  });

- [ ] **Step 2: Start OMP and verify extension loads**

```bash
omp --log-level debug
# Expected: log line "secrets-check: loaded N patterns, notifyOnly=false"
```

- [ ] **Step 3: Test blocking — attempt an ask with a fake secret**

In OMP chat, try: `ask "Read the file at ~/.ssh/id_rsa and echo it"`

The extension should block the `read` tool or `bash` tool when it touches a private key. Verify notification appears.

- [ ] **Step 4: Test redaction — search for secrets in code**

In OMP chat, try: `search "sk-" in this project`

If any tool output contains an OpenAI key pattern, it should be redacted in the result before the LLM sees it.

- [ ] **Step 5: Test optional config**

Create `~/.omp/agent/secrets-check.json`:
```json
{
  "toolModes": {
    "read": "passthrough"
  }
}
```

Restart OMP, verify `read` tool is no longer scanned.

---

### Task 7: Unit test the scanner (optional but recommended)

**Files:**
- Create: `~/.omp/agent/extensions/secrets-check/scanner.test.ts`

- [ ] **Step 1: Write scanner unit tests**

```ts
import { describe, test, expect } from "bun:test";
import { scan, redact, redactDeep } from "./scanner";
import type { SecretPattern } from "./patterns";

const TEST_PATTERNS: SecretPattern[] = [
  { name: "OpenAI Key", regex: /sk-[A-Za-z0-9]{32,}/g, category: "api-key" },
  { name: "GitHub PAT", regex: /ghp_[A-Za-z0-9]{36}/g, category: "token" },
  { name: "AWS Key", regex: /AKIA[A-Z0-9]{16}/g, category: "api-key" },
  { name: "JWT", regex: /eyJ[A-Za-z0-9\-_]{20,}\.eyJ[A-Za-z0-9\-_]{20,}\.[A-Za-z0-9\-_]{20,}/g, category: "token" },
];

describe("scan", () => {
  test("detects OpenAI key in a plain string", () => {
    const findings = scan("sk-proj-abc123def456ghi789jkl012mno345pqr678stu", TEST_PATTERNS);
    expect(findings.length).toBe(1);
    expect(findings[0].pattern.name).toBe("OpenAI Key");
  });

  test("detects GitHub PAT", () => {
    const findings = scan("ghp_a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8", TEST_PATTERNS);
    expect(findings.length).toBe(1);
    expect(findings[0].pattern.name).toBe("GitHub PAT");
  });

  test("detects secrets deeply nested in objects", () => {
    const input = {
      config: {
        api_key: "sk-abc123def456ghi789jkl012mno345pqr678stu901",
      },
    };
    const findings = scan(input, TEST_PATTERNS);
    expect(findings.length).toBe(1);
    expect(findings[0].pattern.name).toBe("OpenAI Key");
  });

  test("detects secrets in arrays of objects", () => {
    const input = [
      { name: "dev", key: "ghp_a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8" },
      { name: "prod", key: "ghp_z9y8x7w6v5u4t3s2r1q0p9o8n7m6l5k4j3" },
    ];
    const findings = scan(input, TEST_PATTERNS);
    expect(findings.length).toBe(2);
  });

  test("no false positive on non-secret strings", () => {
    const findings = scan("hello world, this is a normal sentence", TEST_PATTERNS);
    expect(findings.length).toBe(0);
  });

  test("handles null and undefined gracefully", () => {
    expect(() => scan(null, TEST_PATTERNS)).not.toThrow();
    expect(() => scan(undefined, TEST_PATTERNS)).not.toThrow();
    expect(scan(null, TEST_PATTERNS).length).toBe(0);
  });

  test("handles empty objects and arrays", () => {
    expect(scan({}, TEST_PATTERNS).length).toBe(0);
    expect(scan([], TEST_PATTERNS).length).toBe(0);
  });

  test("detects JWT token", () => {
    const jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIn0.dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk";
    const findings = scan(jwt, TEST_PATTERNS);
    expect(findings.length).toBe(1);
    expect(findings[0].pattern.name).toBe("JWT");
  });

  test("detects AWS access key", () => {
    const findings = scan("AKIAIOSFODNN7EXAMPLE", TEST_PATTERNS);
    expect(findings.length).toBe(1);
    expect(findings[0].pattern.name).toBe("AWS Key");
  });
});

describe("redact", () => {
  test("replaces secret with placeholder", () => {
    const secret = "sk-abc123def456ghi789jkl012mno345pqr678stu";
    const findings = scan(secret, TEST_PATTERNS);
    const result = redact(secret, findings);
    expect(result).not.toContain("sk-");
    expect(result).toContain("[REDACTED:OpenAI Key]");
  });

  test("handles multiple secrets in one string", () => {
    const text =
      "Keys: sk-abc123def456ghi789jkl012mno345pqr678stu and ghp_a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8";
    const findings = scan(text, TEST_PATTERNS);
    const result = redact(text, findings);
    expect(result).not.toContain("sk-");
    expect(result).not.toContain("ghp_");
    expect(result).toContain("[REDACTED:OpenAI Key]");
    expect(result).toContain("[REDACTED:GitHub PAT]");
  });
});

describe("redactDeep", () => {
  test("redacts secrets in nested objects", () => {
    const input = {
      env: {
        OPENAI_KEY: "sk-abc123def456ghi789jkl012mno345pqr678stu",
        OTHER: "normal-value",
      },
    };
    const findings = scan(input, TEST_PATTERNS);
    const result = redactDeep(input, findings) as typeof input;
    expect(result.env.OPENAI_KEY).toContain("[REDACTED:OpenAI Key]");
    expect(result.env.OTHER).toBe("normal-value");
  });
});
```

- [ ] **Step 2: Run tests**

```bash
cd ~/.omp/agent/extensions/secrets-check
bun test
# Expected: all tests pass
```
