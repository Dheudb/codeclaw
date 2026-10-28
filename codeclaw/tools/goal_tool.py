from pydantic import BaseModel, Field
from typing import Literal
from codeclaw.tools.base import BaseAgenticTool
from codeclaw.context.tool_results import build_tool_result


class GoalReportInput(BaseModel):
    status: Literal["continue", "complete", "blocked"]
    summary: str = Field(min_length=1)
    evidence: list[str] = Field(default_factory=list, description="Evidence for the saved acceptance criteria; a claim does not itself pass verification.")


class GoalReportTool(BaseAgenticTool):
    name = "goal_report_tool"
    description = "Report progress or a concrete blocker for an explicitly running goal. Host acceptance and budgets remain authoritative."
    input_schema = GoalReportInput
    is_read_only = True

    async def execute(self, status, summary, evidence=None):
        goal = self.context.get("goal_budget")
        if not goal or self.context.get("agent_depth", 0):
            return build_tool_result(ok=False, is_error=True, content="Only the owning active goal may report; this tool cannot create goals or change authorization.")
        goal["report"] = dict(status=status, summary=summary, evidence=list(evidence or []))
        return build_tool_result(ok=True, content="Report recorded. The host verifies acceptance, progress and budgets after this round.")
