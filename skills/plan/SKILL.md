---
name: plan
description: Create a detailed implementation plan for a coding task.
when_to_use: When a request is multi-step, architectural, or likely to touch multiple files and needs a concrete implementation plan before coding.
aliases: [planning]
user_invocable: true
---

Create a detailed implementation plan for the user's request.

Requirements:
- Do not implement yet.
- Do not save the plan unless the user explicitly asks.
- Be generic across projects: infer the repo structure from the current project rather than assuming a specific framework.
- Prefer concrete, executable guidance over vague advice.

Return the plan with these exact sections:

Goal
- Summarize the objective and success condition.

Assumptions
- State assumptions or missing context that could affect the plan.

Files to inspect
- List the files, directories, or entrypoints that should be inspected first.

Files likely to change
- List the files or areas most likely to require edits.

Implementation plan
- Give a step-by-step sequence of changes.

Tests to add or run
- Specify the tests to add or execute.

Verification commands
- Provide concrete commands to verify the work.

Risks / open questions
- List uncertainties, tradeoffs, or blockers.

Do not write code or claim the work is complete.
