import asyncio
import json
from pathlib import Path
import unittest
from unittest.mock import AsyncMock, patch

import test_tool_recovery as recovery
from codeclaw.features import goals
from codeclaw.core.run_result import PersistenceError


class GoalTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = recovery.RecoveryTests.asyncSetUp
    asyncTearDown = recovery.RecoveryTests.asyncTearDown
    fixture = recovery.RecoveryTests.fixture

    async def emit(self, event): self.events.append(event)

    def create(self, **kwargs):
        return goals.create(self.engine, "Complete the task", ["Evidence is saved"], **kwargs)

    def report(self, status="complete"):
        return "goal_report_tool", json.dumps(dict(status=status, summary="Result saved", evidence=["result.txt"]))

    async def test_explicit_checks_accept_and_checkpoint_matches_receipt(self):
        self.engine.permission_manager.approval_policy.set_mode("full")
        self.create(checks=[dict(kind="file_contains", path="result.txt", text="PROOF")])
        self.fixture([("file_write_tool", json.dumps(dict(file_path=str(self.root/'result.txt'), content="PROOF"))), self.report(), "DONE"])
        result = await goals.run(self.engine, self.emit)
        self.assertEqual(result["status"], "completed")
        saved = self.engine.session_manager.load_session_state(self.engine.session_id)["metadata"]["goal"]
        self.assertEqual(saved, result)
        self.assertEqual(result["output_reserved"], 0)
        self.assertEqual(result["output_used"], 15)
        self.assertTrue(all(r["max_tokens"] <= 50000 for r in self.requests))
        self.assertEqual([e["type"] for e in self.events].count("goal_final"), 1)

    async def test_model_claim_needs_host_acceptance_and_does_not_automatically_complete(self):
        self.create()
        self.fixture([self.report(), "Claimed complete"])
        result = await goals.run(self.engine, self.emit)
        self.assertEqual(result["status"], "awaiting_review")
        accepted = await goals.control(self.engine, "accept", "User checked the evidence")
        self.assertEqual(accepted["accepted_by"], "user_or_host")
        with self.assertRaises(ValueError): await goals.run(self.engine, self.emit)

    async def test_failed_acceptance_continues_and_stops_without_tool_progress(self):
        self.create(checks=[dict(kind="file_exists", path="missing.txt")])
        self.fixture([self.report(), "Claimed complete"] * 5)
        result = await goals.run(self.engine, self.emit)
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["rounds"], 3)
        self.assertFalse(result["check_results"][0]["passed"])

    async def test_round_budget_is_durable_and_recovery_does_not_restart(self):
        self.create(rounds=1)
        self.fixture(["Need more work"])
        result = await goals.run(self.engine, self.emit)
        self.assertEqual(result["status"], "budget_exhausted")
        with self.assertRaises(ValueError): await goals.run(self.engine, self.emit)
        interrupted = goals.recover({**result, "status": "active", "phase": "executing", "output_reserved": 100})
        self.assertEqual(interrupted["status"], "interrupted")
        self.assertEqual(interrupted["output_used"], result["output_used"] + 100)
        self.assertEqual(len(self.requests), 1)

    async def test_time_budget_aborts_a_waiting_model_without_waiting_for_its_response(self):
        self.create(seconds=.2)
        async def hang(**kwargs): await asyncio.Event().wait()
        self.engine.client.chat.completions.create = hang
        result = await asyncio.wait_for(goals.run(self.engine, self.emit), 8)
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertFalse(self.engine.has_active_turn())

    async def test_persistent_pause_mailbox_and_resume_use_same_session(self):
        self.create()
        started = asyncio.Event()
        async def hang(**kwargs): started.set(); await asyncio.Event().wait()
        self.engine.client.chat.completions.create = hang
        task = asyncio.create_task(goals.run(self.engine, self.emit))
        await started.wait()
        sid = self.engine.session_id
        goals.request_control(self.root, sid, self.engine.goal["id"], "pause")
        result = await asyncio.wait_for(task, 8)
        self.assertEqual(result["status"], "paused")
        self.assertEqual(result["output_reserved"], 0)
        self.fixture([self.report(), "AFTER_PAUSE"])
        result = await goals.run(self.engine, self.emit, validator=AsyncMock(return_value=True))
        self.assertEqual(result["status"], "completed")
        self.assertEqual(self.engine.session_id, sid)
        self.assertEqual(result["rounds"], 2)

    async def test_initial_save_failure_prevents_request_and_reverts_creation(self):
        with patch.object(self.engine, "persist_session_state", return_value=False):
            with self.assertRaises(PersistenceError): self.create()
        self.assertIsNone(self.engine.goal)
        self.assertEqual(self.requests, [])

    async def test_checks_cannot_escape_workspace_or_claim_empty_criteria(self):
        with self.assertRaises(ValueError): self.create(checks=[dict(kind="file_exists", path="../secret")])
        with self.assertRaises(ValueError): goals.create(self.engine, "x", [])

    async def test_goal_budget_is_shared_and_caps_simultaneous_model_reservations(self):
        self.create(output_tokens=20)
        self.engine.tool_context["goal_budget"] = self.engine.goal
        self.engine.session_token_usage["output_tokens"] = 900000
        self.assertEqual(self.engine._remaining_output_budget(self.engine._build_loop_state()), 20, "An explicit goal budget is scoped to the goal, not previous conversation usage")
        entered, release = asyncio.Event(), asyncio.Event()
        async def receive(**kwargs):
            self.assertEqual(kwargs["max_tokens"], 20)
            entered.set(); await release.wait()
            from types import SimpleNamespace as NS
            return NS(usage=NS(output_tokens=7)), [], False
        with patch.object(self.engine, "_receive_model_response", side_effect=receive):
            first = asyncio.create_task(self.engine._stream_model_response(max_tokens=200))
            await entered.wait()
            with self.assertRaisesRegex(RuntimeError, "budget"):
                await self.engine._stream_model_response(max_tokens=200)
            release.set(); await first
        self.assertEqual(self.engine.goal["output_used"], 7)
        self.assertEqual(goals.available_output(self.engine.goal), 13)

    async def test_compaction_and_memory_auxiliary_calls_share_the_goal_budget(self):
        from codeclaw.core.config import safe_llm_call
        from types import SimpleNamespace as NS
        self.create(output_tokens=100)
        token = goals.CURRENT_BUDGET.set(self.engine.goal)
        client = NS(chat=NS(completions=NS(create=AsyncMock(return_value=NS(usage=NS(completion_tokens=13), choices=[NS(message=NS(content="Summary"))])))))
        try:
            self.assertEqual(await safe_llm_call(client,"fixture","openai",[],500),"Summary")
            self.assertEqual(client.chat.completions.create.call_args.kwargs["max_tokens"],100)
            self.assertEqual(self.engine.goal["output_used"],13)
            self.engine.goal["output_used"] = 100
            with self.assertRaisesRegex(RuntimeError,"budget"): await safe_llm_call(client,"fixture","openai",[],500)
            self.assertEqual(client.chat.completions.create.await_count,1)
        finally: goals.CURRENT_BUDGET.reset(token)

    async def test_pause_wins_a_race_with_host_acceptance(self):
        self.create()
        self.fixture([self.report(), "DONE"])
        entered, finish = asyncio.Event(), asyncio.Event()
        async def validate(goal): entered.set(); await finish.wait(); return True
        task = asyncio.create_task(goals.run(self.engine,self.emit,validator=validate))
        await entered.wait()
        await goals.control(self.engine,"pause")
        finish.set()
        result = await task
        self.assertEqual(result["status"],"paused")

    async def test_satisfied_host_callback_cannot_skip_explicit_failed_file_checks(self):
        self.create(checks=[dict(kind="file_exists",path="missing.txt")],rounds=1)
        self.fixture([self.report(), "DONE"])
        result = await goals.run(self.engine,self.emit,validator=AsyncMock(return_value=True))
        self.assertEqual(result["status"],"budget_exhausted")

    async def test_status_of_a_live_owner_is_not_misreported_as_interrupted(self):
        from codeclaw.core.session import SessionManager
        self.create()
        self.engine.goal["status"], self.engine.goal["phase"] = "active", "executing"
        with SessionManager(cwd=self.root) as viewer:
            current = goals.snapshot(viewer,self.engine.session_id,self.engine.goal)
            self.assertEqual(current["status"],"active")
            self.assertTrue(current["owner_busy"])
            self.engine.session_manager.release_writer()
            recovered = goals.snapshot(viewer,self.engine.session_id,self.engine.goal)
            self.assertEqual(recovered["status"],"interrupted")
            self.assertFalse(recovered["owner_busy"])
