import time
from typing import Literal, Optional
from pydantic import BaseModel, Field
from codeclaw.tools.base import BaseAgenticTool
from codeclaw.context.tool_results import build_tool_result


class ScheduleInput(BaseModel):
    action: Literal["add", "list", "pause", "resume", "cancel"]
    prompt: str = ""
    delay_seconds: float = Field(default=1800, ge=0, le=31536000)
    interval_seconds: Optional[float] = Field(default=None, ge=60, le=31536000)
    schedule_id: str = ""


class ScheduleTool(BaseAgenticTool):
    name = "schedule_tool"
    description = "Schedule an explicitly requested delayed or repeating message to this conversation. Requires the desktop app or codeclaw schedule serve to be running; busy sessions queue. It does not hold a model request open."
    input_schema = ScheduleInput
    is_read_only = False

    def is_read_only_call(self, action, **kwargs): return action == "list"

    async def execute(self, action, prompt="", delay_seconds=1800, interval_seconds=None, schedule_id=""):
        from codeclaw.features import schedules
        callback = self.context.get("schedule_add")
        owner = self.context.get("session_id")
        if self.context.get("agent_depth", 0): raise ValueError("Only the conversation owner may schedule messages")
        if action == "add":
            if not callback: raise ValueError("Scheduling is unavailable in this context")
            result = callback(prompt, due_at=time.time()+delay_seconds, interval=interval_seconds)
        else:
            rows = [row for row in schedules.list_items() if row["session_id"] == owner]
            if action == "list": result = rows
            else:
                if not any(row["id"] == schedule_id for row in rows): raise ValueError("Unknown schedule for this conversation")
                result = schedules.change(schedule_id, action)
        return build_tool_result(ok=True, content=__import__('json').dumps(result, ensure_ascii=False))
