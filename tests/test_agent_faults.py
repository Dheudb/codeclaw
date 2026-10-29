import asyncio
from collections import Counter
import json
import unittest
from unittest.mock import patch
import test_agent_lifecycle as lifecycle
from xuloom.core.engine import QueryEngine
from xuloom.core.messages import create_user_message, create_assistant_message
from xuloom.core.run_result import RunResult


class AgentFaultTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = lifecycle.LifecycleTests.asyncSetUp
    asyncTearDown = lifecycle.LifecycleTests.asyncTearDown
    make_engine = lifecycle.LifecycleTests.make_engine

    async def test_parallel_mixed_outcomes_followup_and_owner_exit_release_capacity(self):
        self.engine.set_collaboration_mode("ultra", persist=False)
        entered = []
        gate, all_started = asyncio.Event(), asyncio.Event()
        events = []
        self.engine.tool_context["capability_event"] = events.append
        async def model(child, user_input, sys_print_callback):
            self.assertEqual(child.model_provider, self.engine.model_provider)
            self.assertEqual(child.api_base_url, self.engine.api_base_url)
            child.messages.append(create_user_message(user_input))
            child.messages.append(create_assistant_message("checkpoint"))
            self.assertTrue(child.persist_session_state())
            if user_input == "followup":
                return RunResult("continued", "completed", True)
            index = len(entered)
            entered.append(child.agent_id)
            if len(entered) == 6:
                all_started.set()
            await gate.wait()
            return RunResult("FAILED" if index == 2 else "DONE", "api_error" if index == 2 else "completed", True)
        with patch.object(QueryEngine, "run", model):
            batch = json.loads(await self.tool.execute(prompt="work", count=6, run_in_background=True, response_format="json"))
            ids = [row["agent_id"] for row in batch["agents"]]
            await asyncio.wait_for(all_started.wait(), 15)
            for index in (1, 4):
                await self.tool.execute(action="cancel", agent_id=entered[index])
            gate.set()
            await asyncio.gather(*self.engine.tool_context["subagent_tasks"].tasks.values(), return_exceptions=True)
            rows = json.loads(await self.tool.execute(action="list"))["agents"]
            self.assertEqual(Counter(row["status"] for row in rows), {"completed": 3, "crashed": 1, "cancelled": 2})
            notifications = Counter(event["agent"]["agent_id"] for event in events
                if event.get("type") == "subagent_status" and event["agent"].get("status") in {"completed", "crashed", "cancelled"})
            self.assertEqual(notifications, Counter({identity: 1 for identity in ids}))
            capacity = json.loads(await self.tool.execute(action="capacity"))
            self.assertEqual(capacity["active"], 0)
            self.assertIsNone(capacity["limit"])
            self.assertIsNone(capacity["available"])
            continued = json.loads(await self.tool.execute(action="followup", agent_id=entered[0], prompt="followup"))
            self.assertEqual(continued["agent_id"], entered[0])
            self.assertEqual(continued["status"], "completed")
            self.assertEqual(continued["turn"], 2)
            gate.clear()
            await self.tool.execute(prompt="hold again", count=6, run_in_background=True)
            await asyncio.sleep(0)
            self.assertTrue(await self.engine.request_abort())
            self.assertFalse(any(not task.done() for task in self.engine.tool_context["subagent_tasks"].tasks.values()))
            capacity = json.loads(await self.tool.execute(action="capacity"))
            self.assertEqual(capacity["active"], 0)
            self.assertIsNone(capacity["available"])

    async def test_closing_owner_refuses_reentrant_launch_without_coroutine_leak(self):
        manager = self.engine.tool_context["subagent_tasks"]
        closing = asyncio.Event()
        async def work():
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                closing.set()
                await asyncio.sleep(0)
                raise
        manager.launch("one", work())
        await asyncio.sleep(0)
        close = asyncio.create_task(manager.close())
        await closing.wait()
        with self.assertRaises(ValueError):
            manager.launch("late", work())
        await close
        self.assertEqual(set(manager.tasks), {"one"})
