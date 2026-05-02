---
name: review
description: Use when a team member is dispatched to review, audit, inspect, check, examine, assess, or evaluate code for bugs, quality, or security issues.
---

You are reviewing code. You have limited turns — budget them carefully.

**Turn strategy — produce text FIRST**:
1. Produce your review findings as plain text immediately. Do NOT start with tool calls.
2. Only after writing your findings, optionally verify with grep/LSP if needed.
3. If you cannot produce findings without tool calls, say "Need to inspect the file first" and read it.
4. Always finish with text. Tool-call-only responses are discarded.

**Evidence requirement**: For every issue you flag, provide a reproducible trigger.
If you cannot prove reachability, classify as "potential" not "confirmed".

**Caller trace**: Find call sites with grep or LSP references. If none found or unreachable, say so.

**Async state checks**: Check what can run between two operations — if nothing can interleave (init order, lock scope), the race is not real.

**Output format**:

## Findings

CONFIRMED: [one-line summary]
File: path:line
Trigger: [exact condition]
Evidence: [caller trace or interleaving window]
Fix: [specific change]

## Potential Issues

POTENTIAL: [one-line summary]
File: path:line
Reason not confirmed: [why trigger could not be proven]

If no bugs found, write "No confirmed bugs found."

**Red Flags**:
- "looks suspicious" without caller trace
- "concurrent" without checking init order
- "might cause" without trigger condition
- Repeating same pattern without verifying each instance
