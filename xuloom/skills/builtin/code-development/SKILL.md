---
name: code-development
description: Implement, debug and refactor software with relevant repository conventions and executable checks. Use for programming tasks, not ordinary conversation, writing or general research.
---

Read relevant source and project instructions before changing behavior. Keep changes within the user's task, preserve existing work, and avoid speculative abstractions or unnecessary dependencies.

Find the project's documented build and test commands. Choose meaningful executable checks for the change, reproduce reported defects where possible, and report what ran, what passed, what failed and what remains unverified. Do not infer required validation from the number of edited files. Do not execute generated scripts merely because their names resemble entry points.

For visual web changes, discover `browser_tool` and use `action="render"` with a workspace HTML `file_path` or a running local development `url` to capture and inspect the real UI without opening a visible window. Screenshot images are returned to vision-capable models; a saved file alone is not evidence that the model inspected it. Respect user instructions about whether to run checks.

Use existing file/search tools and run commands through the normal permission gate. Independent review is useful for substantial changes; assign actual acceptance criteria and permitted checks rather than blindly requiring every build or linter.

Avoid command injection, unsafe handling of untrusted input and credentials in output. Make Git commits, pushes, dependency changes and external publication only within the user's authorized scope. Worktree copies isolate changes, not host access or network activity.
