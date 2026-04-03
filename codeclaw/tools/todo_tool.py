import uuid
from typing import List
from pydantic import BaseModel, Field, field_validator
from codeclaw.tools.base import BaseAgenticTool


class TodoItemInput(BaseModel):
    id: str = Field(default="", description="Stable identifier for the todo item.")

    @field_validator("id", mode="before")
    @classmethod
    def _fill_id(cls, v):
        if not v:
            return f"auto_{uuid.uuid4().hex[:8]}"
        return v
    content: str = Field(..., description="Human-readable task description in imperative form (e.g., 'Run tests').")
    status: str = Field(
        ...,
        description="One of: pending, in_progress, completed, cancelled.",
    )


class TodoWriteToolInput(BaseModel):
    merge: bool = Field(
        True,
        description="If true, update the provided todo items onto the existing list. If false, replace the list entirely.",
    )
    todos: List[TodoItemInput] = Field(
        ...,
        description="Structured todo items to create or update.",
    )


class TodoWriteTool(BaseAgenticTool):
    name = "todo_write_tool"
    description = "Update the todo list for the current session. Use proactively and often to track progress and pending tasks. Ensure at least one task is in_progress at all times."
    input_schema = TodoWriteToolInput
    risk_level = "medium"

    def prompt(self) -> str:
        return """Use this tool to create and manage a structured task list for your current coding session. This helps you track progress, organize complex tasks, and demonstrate thoroughness to the user.

## When to Use This Tool
Use proactively when:
1. Complex multi-step tasks (3+ distinct steps)
2. Non-trivial tasks requiring careful planning
3. User explicitly requests todo list
4. User provides multiple tasks (numbered/comma-separated)
5. After receiving new instructions — capture requirements as todos
6. When starting a task — mark as in_progress BEFORE beginning
7. After completing a task — mark completed and add follow-ups

## When NOT to Use
Skip when:
1. Single, straightforward task
2. Trivial task with no organizational benefit
3. Completable in < 3 trivial steps
4. Purely conversational/informational

## Task States
 - pending: Not yet started
 - in_progress: Currently working on (only ONE at a time)
 - completed: Finished successfully

## Task Management
 - Update status in real-time as you work
 - Mark tasks complete IMMEDIATELY after finishing (don't batch)
 - Exactly ONE task must be in_progress at any time
 - Complete current tasks before starting new ones
 - Remove tasks that are no longer relevant from the list entirely
 - ONLY mark a task as completed when FULLY accomplished
 - If you encounter errors or blockers, keep task as in_progress
 - Never mark a task completed if tests are failing, implementation is partial, or unresolved errors exist

## Examples

<example>
User: "Add dark mode toggle to the settings. Run the tests when done!"
→ Create todos: 1) Create dark mode toggle component, 2) Add dark mode state management, 3) Implement CSS styles for dark theme, 4) Update components to support theme switching, 5) Run tests and build
Reasoning: Multi-step feature with multiple UI/state/styling changes plus explicit test request.
</example>

<example>
User: "Rename getCwd to getCurrentWorkingDirectory across my project"
→ First search to understand scope, then create todos for each file group.
Reasoning: Multiple occurrences across files require systematic tracking.
</example>

<example>
User: "Fix the typo in the README"
→ Do NOT create a todo list. Just fix it directly.
Reasoning: Single straightforward task, no tracking needed.
</example>

<example>
User: "What does git status do?"
→ Do NOT create a todo list. Just answer.
Reasoning: Informational request, no coding task.
</example>

When in doubt, use this tool. Being proactive with task management demonstrates attentiveness and ensures you complete all requirements successfully."""

    def build_permission_summary(self, merge: bool = True, todos: List[dict] = None) -> str:
        todos = todos or []
        lines = [
            "Structured todo update requested.",
            f"merge: {merge}",
            f"item_count: {len(todos)}",
        ]
        for item in todos[:4]:
            lines.append(
                f"- [{item.get('status', '?')}] {item.get('id', '<missing>')}: {item.get('content', '')}"
            )
        return "\n".join(lines)

    async def execute(self, merge: bool = True, todos: List[dict] = None) -> str:
        todo_manager = self.context.get("todo_manager")
        if not todo_manager:
            return "Error: Structured Todo Manager is unavailable."

        normalized = []
        for item in todos or []:
            if isinstance(item, dict):
                normalized.append(item)
            else:
                normalized.append(item.model_dump())

        return todo_manager.write(normalized, merge=merge)
