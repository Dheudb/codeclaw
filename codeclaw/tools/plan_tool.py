from pydantic import BaseModel, Field
from codeclaw.tools.base import BaseAgenticTool


class PlanToolInput(BaseModel):
    action: str = Field(..., description="One of: read, write, append, clear.")
    content: str = Field(None, description="Plan content for write or append.")


class PlanTool(BaseAgenticTool):
    name = "plan_tool"
    description = "Reads or updates the session-scoped implementation plan. Use plan_tool with action='write' to persist your plan."
    input_schema = PlanToolInput
    risk_level = "medium"

    def is_read_only_call(self, action: str, content: str = None) -> bool:
        return action == "read"

    def build_permission_summary(self, action: str, content: str = None) -> str:
        preview = (content or "")[:160] + ("..." if content and len(content) > 160 else "")
        return (
            "Plan operation requested.\n"
            f"action: {action}\n"
            f"content_preview: {preview or '<empty>'}"
        )

    async def execute(self, action: str, content: str = None) -> str:
        plan_manager = self.context.get("plan_manager")
        if not plan_manager:
            return "Error: Plan Manager is unavailable."

        if action == "read":
            current = plan_manager.get_plan()
            if not current:
                return "Current plan is empty."
            return current
        if action == "write":
            if content is None:
                return "Error: content is required for write."
            return plan_manager.write_plan(content)
        if action == "append":
            if content is None:
                return "Error: content is required for append."
            return plan_manager.append_plan(content)
        if action == "clear":
            return plan_manager.clear_plan()
        return f"Error: Unsupported plan action '{action}'."
