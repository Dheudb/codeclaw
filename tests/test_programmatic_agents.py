"""Host coordination uses the real authorization/journal and native worker records."""
import asyncio
import unittest
from unittest.mock import patch

from xuloom.core.engine import QueryEngine
from xuloom.core.messages import create_assistant_message, create_user_message
from xuloom.core.run_result import RunResult
import test_agent_lifecycle as lifecycle


class ProgrammaticAgentTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = lifecycle.LifecycleTests.asyncSetUp
    asyncTearDown = lifecycle.LifecycleTests.asyncTearDown
    make_engine = lifecycle.LifecycleTests.make_engine

    async def test_idempotent_launch_receipt_survives_restart_and_argument_mismatch(self):
        calls = []
        async def run(child, user_input, sys_print_callback):
            calls.append(child.agent_id)
            child.messages.extend([create_user_message(user_input), create_assistant_message("REAL_CHILD_RESULT")])
            return RunResult("REAL_CHILD_RESULT", "completed", True)
        self.engine.permission_manager.approval_policy.set_mode("full")
        parameters = {"prompt": "Say hello", "description": "host test", "count": 2, "request_id": "host-greeting"}
        with patch.object(QueryEngine, "run", run):
            first = await self.engine.agents("spawn", **parameters)
            self.assertTrue(first["ok"], first)
            identities = [row["agent_id"] for row in first["data"]["agents"]]
            self.assertEqual(len(identities), 2)
            repeated = await self.engine.agents("spawn", **parameters)
            self.assertTrue(repeated["data"]["reused"])
            self.assertEqual([row["agent_id"] for row in repeated["data"]["agents"]], identities)
            mismatch = await self.engine.agents("spawn", **{**parameters, "prompt": "different assignment"})
            self.assertFalse(mismatch["ok"])
            self.assertEqual(len(calls), 2)
            sid = self.engine.session_id
            await self.engine.aclose()
            restored = self.make_engine()
            self.assertTrue(restored.load_session(sid))
            restored.permission_manager.approval_policy.set_mode("full")
            after_restart = await restored.agents("spawn", **parameters)
            self.assertTrue(after_restart["data"]["reused"])
            self.assertEqual([row["agent_id"] for row in after_restart["data"]["agents"]], identities)
            for _ in range(2):
                result = await restored.agents("result", agent_id=identities[0])
                self.assertEqual(result["data"]["receipt"]["status"], "completed")
            self.assertEqual(len(calls), 2, "Result reads and reconnects must not launch workers again")

    async def test_programmatic_launch_obeys_permissions_and_reports_persistence_failure(self):
        self.engine.permission_manager.always_deny_tools.add("agent_tool")
        result = await self.engine.agents("spawn", prompt="denied", request_id="denied")
        self.assertFalse(result["ok"])
        self.assertEqual(self.engine.subagent_registry, [])
        self.engine.permission_manager.always_deny_tools.clear()
        self.engine.permission_manager.approval_policy.set_mode("full")
        with patch.object(self.engine, "persist_session_state", return_value=False):
            result = await self.engine.agents("spawn", prompt="not durable", request_id="not-durable")
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "persistence_error")
        self.assertEqual(self.engine.subagent_registry, [])

    async def test_large_worker_result_remains_complete_in_the_host_response(self):
        answer = "Evidence " * 2000 + "TAIL_PROOF"
        async def run(child, user_input, sys_print_callback):
            return RunResult(answer, "completed", True)
        self.engine.permission_manager.approval_policy.set_mode("full")
        with patch.object(QueryEngine, "run", run):
            result = await self.engine.agents("spawn", prompt="long result", request_id="long-result")
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["data"]["agents"][0]["result"], answer)
        self.assertTrue(self.engine.content_replacement_manager.registry, "Inline display still uses the existing complete-result archive")

    async def test_native_wait_cancel_and_followup_keep_actual_worker_identity(self):
        gate = asyncio.Event()
        async def run(child, user_input, sys_print_callback):
            child.messages.append(create_user_message(user_input))
            await gate.wait()
            return RunResult("finished", "completed", True)
        self.engine.permission_manager.approval_policy.set_mode("full")
        with patch.object(QueryEngine, "run", run):
            receipt = await self.engine.agents("spawn", prompt="hold", run_in_background=True, request_id="hold")
            identity = receipt["data"]["agents"][0]["agent_id"]
            waiting = await self.engine.agents("wait", agent_ids=[identity], timeout_ms=0)
            self.assertTrue(waiting["data"]["agents"][0]["executing"])
            cancelled = await self.engine.agents("cancel", agent_id=identity)
            self.assertEqual(cancelled["data"]["status"], "cancelled")
            gate.set()
            resumed = await self.engine.agents("followup", agent_id=identity, prompt="continue")
            self.assertEqual(resumed["data"]["agent_id"], identity)
            self.assertEqual(resumed["data"]["status"], "completed")
            self.assertEqual(resumed["data"]["turn"], 2)
