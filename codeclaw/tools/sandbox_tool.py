from pydantic import BaseModel, Field
from codeclaw.tools.base import BaseAgenticTool
from codeclaw.context.tool_results import build_tool_result

class SandboxToolInput(BaseModel):
    action: str = Field(..., description="'create', 'merge', 'abort', or 'status'")
    branch_name: str = Field(None, description="Only required for 'create' action. A short descriptive branch name like 'refactor_utils'")

class SandboxTool(BaseAgenticTool):
    name = "sandbox_tool"
    description = "Enter an isolated Git worktree sandbox for high-risk or widespread changes. Changes do not affect the main workspace. Use 'merge' to integrate changes or 'abort' to discard."
    input_schema = SandboxToolInput
    risk_level = "high"

    def build_permission_summary(self, action: str, branch_name: str = None) -> str:
        return (
            "Sandbox lifecycle action requested.\n"
            f"action: {action}\n"
            f"branch_name: {branch_name or '<none>'}"
        )
    
    async def execute(self, action: str, branch_name: str = None) -> dict:
        sandbox = self.context.get("sandbox_manager")
        if not sandbox:
            return build_tool_result(
                ok=False,
                content="Sandbox Engine is disabled.",
                metadata={"action": action, "branch_name": branch_name or ""},
                is_error=True,
            )
            
        if action == "create":
            if not branch_name:
                return build_tool_result(
                    ok=False,
                    content="branch_name is required to spawn a sandbox.",
                    metadata={"action": action},
                    is_error=True,
                )
            result = sandbox.create_sandbox(branch_name)
            return build_tool_result(
                ok=not str(result).startswith("Error:"),
                content=result,
                metadata={"action": action, "branch_name": branch_name, "active_sandbox": sandbox.export_state()},
                is_error=str(result).startswith("Error:"),
            )
        elif action == "merge":
            result = sandbox.merge_sandbox()
            return build_tool_result(
                ok=not str(result).startswith("Critical") and not str(result).startswith("Error:"),
                content=result,
                metadata={"action": action, "active_sandbox": sandbox.export_state()},
                is_error=str(result).startswith("Critical") or str(result).startswith("Error:"),
            )
        elif action == "abort":
            result = sandbox.abort_sandbox()
            return build_tool_result(
                ok=not str(result).startswith("Error:") and not str(result).startswith("Crashed"),
                content=result,
                metadata={"action": action, "active_sandbox": sandbox.export_state()},
                is_error=str(result).startswith("Error:") or str(result).startswith("Crashed"),
            )
        elif action == "status":
            status = sandbox.get_status()
            return build_tool_result(
                ok=True,
                content="Sandbox status retrieved." if status.get("active") else "No active sandbox.",
                metadata={"action": action, "sandbox_status": status},
                is_error=False,
            )
            
        return build_tool_result(
            ok=False,
            content=f"Unknown action '{action}'",
            metadata={"action": action, "branch_name": branch_name or ""},
            is_error=True,
        )
