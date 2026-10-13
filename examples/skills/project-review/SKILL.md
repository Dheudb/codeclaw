---
name: project-review
description: Review a codebase, explain its architecture, and identify concrete implementation risks with file references. Use when the user asks for project analysis or a code review.
---

Read [the checklist](references/checklist.md) before starting.

1. Inspect the project entry points and dependency configuration.
2. Trace the relevant execution flow rather than judging filenames alone.
3. Check session isolation, resource cleanup, error propagation and permissions.
4. Report concrete findings with file paths, evidence and suggested fixes.
5. Clearly separate verified behavior from assumptions. State which tests ran.

Use read-only tools. Do not modify the project during a review.
