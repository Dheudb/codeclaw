"""A disposable worker exits at exact execution boundaries; never targets a user app."""
import asyncio
import os
from pathlib import Path
import sys
from unittest.mock import patch
from pydantic import BaseModel
from xuloom.core.engine import QueryEngine
from xuloom.core.messages import create_user_message, create_assistant_message
from xuloom.memory.memory import MemoryCompactor
from xuloom.tools.base import BaseAgenticTool


class Input(BaseModel):
    value: str


class EffectTool(BaseAgenticTool):
    name = "effect_tool"
    description = "Append one durable side effect to a disposable fixture."
    input_schema = Input

    def __init__(self, root, phase):
        self.root, self.phase = Path(root), phase

    async def execute(self, value):
        with (self.root / "effects.txt").open("a", encoding="utf-8") as output:
            output.write(value + "\n")
            output.flush()
            os.fsync(output.fileno())
        if self.phase == "after_effect":
            os._exit(87)
        return "EFFECT_SAVED"


async def main(root, phase):
    with patch.object(MemoryCompactor, "_resolve_token_encoder", return_value=None):
        engine = QueryEngine(model="fixture", model_provider="openai", cwd=root)
    Path(root, "session-id.txt").write_text(engine.session_id, encoding="utf-8")
    tool = EffectTool(root, phase)
    engine.available_tools[tool.name] = tool
    engine.permission_manager.approval_policy.set_mode("full")
    block = {"type": "tool_use", "id": "crash-call", "name": tool.name, "input": {"value": "one-effect"}}
    engine.messages = [create_user_message("Execute exactly one effect"), create_assistant_message([block])]
    original = engine.persist_session_state
    def save():
        ok = original()
        if phase == "before_execution" and engine.in_progress_tools.get("crash-call", {}).get("phase") == "executing":
            os._exit(87)
        return ok
    engine.persist_session_state = save
    def activity(event):
        if phase == "after_result" and event.get("type") == "tool_result":
            os._exit(87)
    engine.tool_context["capability_event"] = activity
    await engine._execute_tool_wrapper(block, lambda _: None)
    raise AssertionError("Crash point was not reached")


if __name__ == "__main__":
    asyncio.run(main(*sys.argv[1:]))
