from pydantic import BaseModel, Field
from codeclaw.tools.base import BaseAgenticTool


class EnterPlanModeInput(BaseModel):
    pass


class EnterPlanModeTool(BaseAgenticTool):
    name = "enter_plan_mode"
    description = (
        "Switch to plan mode. In plan mode, you can only use read-only tools "
        "and planning tools. Use this when you need to think through a complex "
        "task before making changes."
    )
    input_schema = EnterPlanModeInput
    risk_level = "low"
    is_read_only = True

    def prompt(self) -> str:
        return (
            "Use enter_plan_mode when you need to step back and plan before "
            "making changes. In plan mode, only read-only tools and planning "
            "tools (plan_tool, todo_write_tool) are available. This prevents "
            "accidental modifications while you're still thinking. "
            "Call exit_plan_mode when you're ready to execute your plan."
        )

    async def execute(self) -> str:
        plan_manager = self.context.get("plan_manager")
        if not plan_manager:
            return "Error: Plan Manager is unavailable."
        current_mode = plan_manager.get_mode()
        if current_mode == "plan":
            return "Already in plan mode."
        plan_manager.set_mode("plan")
        return (
            "Switched to plan mode. You can now only use read-only tools and "
            "planning tools (plan_tool, todo_write_tool). "
            "Use exit_plan_mode when you're ready to execute."
        )


class ExitPlanModeInput(BaseModel):
    pass


class ExitPlanModeTool(BaseAgenticTool):
    name = "exit_plan_mode"
    description = (
        "Exit plan mode and return to normal execution mode where all tools "
        "are available."
    )
    input_schema = ExitPlanModeInput
    risk_level = "low"
    is_read_only = True

    def prompt(self) -> str:
        return (
            "Use exit_plan_mode when you've finished planning and are ready "
            "to execute your plan. This restores full tool access."
        )

    async def execute(self) -> str:
        plan_manager = self.context.get("plan_manager")
        if not plan_manager:
            return "Error: Plan Manager is unavailable."
        current_mode = plan_manager.get_mode()
        if current_mode == "normal":
            return "Already in normal mode."
        plan_manager.set_mode("normal")
        return "Exited plan mode. All tools are now available."
