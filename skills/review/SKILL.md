---
name: review
description: Use when a team member is dispatched to review, audit, inspect, check, examine, assess, or evaluate code for bugs, quality, or security issues.
---

You are reviewing code. Follow these rules or your findings will be discarded:

**Evidence requirement**: For every issue you flag, you MUST provide a reproducible trigger.
If you cannot prove the code path is reachable, classify it as "potential" not "confirmed".

**Caller trace**: Before flagging any bug, find every call site with grep or LSP references.
List the callers. If none exist or the path is guarded, state that explicitly.

**Async state checks**: For concurrent code, check what can run between two operations —
if nothing can interleave (init order, lock scope), the race is not real.

**Output format for each finding**:
```
CONFIRMED / POTENTIAL: [one-line summary]
File: path:line
Trigger: [exact condition that causes the issue, or "could not prove reachable"]
Evidence: [caller trace, interleaving window, or reproduction steps]
Fix: [specific change]
```

**Red Flags — stop and verify before reporting**:
- "this looks suspicious" without a caller trace
- "could be called concurrently" without checking init order
- "might cause" without a specific trigger condition
- Reporting the same bug pattern in multiple locations without verifying each

Only report CONFIRMED issues as bugs. POTENTIAL issues go in a separate section at the end.
If you find zero confirmed bugs, say so explicitly rather than inventing issues.
