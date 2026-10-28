"""Mode policy tested at native delegation, checkpoint and model-budget boundaries."""
import asyncio
import copy
import json
import unittest
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch

import test_agent_lifecycle as lifecycle
import test_tool_recovery as recovery
from xuloom.agents.collaboration import policy, new_run, DELEGATION_BUDGET
from xuloom.core.engine import QueryEngine
from xuloom.core.run_result import RunResult
from xuloom.core.config import safe_llm_call
from xuloom.core.output_limits import active_budgets, reserve, settle
from xuloom.desktop.bridge import DesktopBridge
from xuloom.features.goals import CURRENT_BUDGET


class CollaborationTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = lifecycle.LifecycleTests.asyncSetUp
    asyncTearDown = lifecycle.LifecycleTests.asyncTearDown
    make_engine = lifecycle.LifecycleTests.make_engine

    async def test_solo_rejects_native_and_host_spawns_even_with_full_access(self):
        self.engine.permission_manager.approval_policy.set_mode("full")
        self.engine.set_collaboration_mode("solo")
        with self.assertRaisesRegex(ValueError, "切换"):
            await self.tool.execute(prompt="hello", count=1)
        result = await self.engine.agents("spawn", prompt="hello")
        self.assertFalse(result["ok"], result)
        self.assertEqual(self.engine.subagent_registry, [])
        capacity = json.loads(await self.tool.execute(action="capacity"))
        self.assertEqual((capacity["limit"], capacity["available"]), (0, 0))

    async def test_auto_and_ultra_enforce_parallel_capacity_before_any_child_returns(self):
        for mode, limit in (("auto", 3), ("ultra", 8)):
            self.engine.set_collaboration_mode(mode)
            entered, gate = [], asyncio.Event()
            async def run(child, user_input, sys_print_callback):
                self.assertEqual(child.collaboration_mode, "solo")
                self.assertNotIn("agent_tool", child.available_tools)
                self.assertNotIn("agent_tool", child.latent_tools)
                entered.append(child.agent_id)
                await gate.wait()
                return RunResult("hello", "completed", True)
            with patch.object(QueryEngine, "run", run):
                await self.tool.execute(prompt="hello", count=limit, run_in_background=True)
                await asyncio.sleep(0)
                self.assertEqual(len(entered), limit)
                with self.assertRaisesRegex(ValueError, "at once"):
                    await self.tool.execute(prompt="overflow")
                with self.assertRaisesRegex(ValueError, "停止"):
                    self.engine.set_collaboration_mode("solo")
                gate.set()
                await self.tool.execute(action="wait")

    async def test_attempt_quota_and_idempotency_cover_followup_and_survive_restart(self):
        self.engine.tool_context["skill_registry"].update(collaboration_profiles={"auto": {"max_delegations": 2}})
        self.engine.set_collaboration_mode("auto")
        async def run(child, user_input, sys_print_callback):
            return RunResult("done", "completed", True)
        with patch.object(QueryEngine, "run", run):
            first = json.loads(await self.tool.execute(prompt="one", request_id="same", response_format="json"))
            identity = first["agents"][0]["agent_id"]
            repeated = json.loads(await self.tool.execute(prompt="one", request_id="same", response_format="json"))
            self.assertTrue(repeated["reused"])
            await self.tool.execute(action="followup", agent_id=identity, prompt="two")
            with self.assertRaisesRegex(ValueError, "累计"):
                await self.tool.execute(prompt="three")
            self.engine.persist_session_state()
            self.engine.session_manager.release_writer()
            restored = self.make_engine()
            self.assertTrue(restored.load_session(self.engine.session_id))
            self.assertEqual(restored.tool_context["collaboration_run"]["submitted"], 2)
            with self.assertRaisesRegex(ValueError, "累计"):
                await restored.available_tools["agent_tool"].execute(action="followup", agent_id=identity, prompt="three")
            restored.set_collaboration_mode("solo")
            with self.assertRaisesRegex(ValueError, "切换"):
                await restored.available_tools["agent_tool"].execute(action="followup", agent_id=identity, prompt="three")

    async def test_output_reservations_cover_parallel_children_aux_calls_and_cancellation(self):
        self.engine.set_collaboration_mode("ultra", persist=False)
        run = self.engine.tool_context["collaboration_run"]
        entered, release = [], asyncio.Event()
        async def model(child, user_input, sys_print_callback):
            self.assertIs(DELEGATION_BUDGET.get(), run)
            async def receive(**kwargs):
                entered.append(kwargs["max_tokens"])
                await release.wait()
                return NS(usage=NS(output_tokens=5, input_tokens=1)), [], False
            with patch.object(child, "_receive_model_response", receive):
                await child._stream_model_response(max_tokens=100000)
            client = NS(chat=NS(completions=NS(create=AsyncMock(return_value=NS(usage=NS(completion_tokens=2), choices=[NS(message=NS(content="aux"))])))))
            await safe_llm_call(client, "fixture", "openai", [], 5)
            return RunResult("done", "completed", True)
        with patch.object(QueryEngine, "run", model):
            await self.tool.execute(prompt="work", count=8, run_in_background=True)
            await asyncio.sleep(0)
            self.assertEqual(entered, [64000] * 8)
            self.assertEqual(run["output_reserved"], 512000)
            release.set()
            await self.tool.execute(action="wait")
        self.assertEqual((run["output_reserved"], run["output_used"]), (0, 56))
        self.assertIsNone(DELEGATION_BUDGET.get())
        token = DELEGATION_BUDGET.set(run)
        try:
            async def hang(**kw): raise asyncio.CancelledError()
            with patch.object(self.engine, "_receive_model_response", hang):
                with self.assertRaises(asyncio.CancelledError):
                    await self.engine._stream_model_response(max_tokens=10)
        finally: DELEGATION_BUDGET.reset(token)
        self.assertEqual((run["output_reserved"], run["output_used"]), (0, 66))

    async def test_long_response_default_is_independent_of_concurrency_and_remaining_budget_is_enforced(self):
        for mode in ("auto", "ultra"):
            profile = policy(mode)
            self.assertEqual(profile["output_tokens"], 1000000)
            self.assertEqual(new_run(profile)["request_cap"], 64000)
        run = new_run(policy("auto", {"collaboration_profiles": {"auto": {"concurrency": 24}}}))
        self.assertEqual(run["request_cap"], 64000)
        token = DELEGATION_BUDGET.set(run)
        try:
            with patch.dict("os.environ", {"XULOOM_MAX_OUTPUT_TOKENS": "", "XULOOM_MODEL_MAX_OUTPUT_TOKENS": ""}):
                self.assertEqual(self.engine._build_loop_state()["max_tokens"], 64000)
            run["output_used"] = 990000
            amount = reserve(64000, active_budgets())
            self.assertEqual(amount, 10000)
            with self.assertRaises(RuntimeError): reserve(1, active_budgets())
            settle(active_budgets(), amount, 9000)
            self.assertEqual(run["output_used"], 999000)
        finally: DELEGATION_BUDGET.reset(token)
        small = policy("auto", {"collaboration_profiles": {"auto": {"request_output_tokens": 2048, "output_tokens": 5000}}})
        self.assertEqual(new_run(small)["request_cap"], 2048)

    async def test_mode_does_not_change_reasoning_permissions_or_private_skills(self):
        before = (self.engine.reasoning_effort, self.engine.permission_manager.approval_policy.mode)
        for mode in ("solo", "ultra", "auto"):
            self.engine.set_collaboration_mode(mode)
            self.assertEqual(before, (self.engine.reasoning_effort, self.engine.permission_manager.approval_policy.mode))
            self.assertIn("# Collaboration: " + mode, self.engine._build_tool_prompt_summary())
        self.engine.set_collaboration_mode("ultra")
        self.engine.session_manager.release_writer()
        restored = self.make_engine()
        self.assertTrue(restored.load_session(self.engine.session_id))
        self.assertEqual(restored.collaboration_mode, "ultra")
        self.assertEqual(restored.tool_context["collaboration"]["concurrency"], 8)

    async def test_gui_preferences_save_without_engine_and_respect_writer_lease(self):
        self.engine.persist_session_state()
        bridge = DesktopBridge(lambda _: None, cwd=self.root)
        bridge.resume_id = self.engine.session_id
        with self.assertRaises(Exception):
            await bridge.dispatch("collaboration.set", {"mode": "ultra"})
        self.assertEqual(bridge.collaboration_mode, "auto")
        self.engine.session_manager.release_writer()
        result = await bridge.dispatch("collaboration.set", {"mode": "ultra"})
        self.assertEqual(result["collaborationMode"], "ultra")
        self.assertIsNone(bridge.engine)
        reopened = DesktopBridge(lambda _: None, cwd=self.root)
        initial = await reopened.dispatch("initialize", {"sessionId": self.engine.session_id})
        self.assertEqual(initial["collaborationMode"], "ultra")
        with patch("xuloom.core.session.SessionManager.save_session_state", return_value=False):
            with self.assertRaises(OSError):
                await reopened.dispatch("collaboration.set", {"mode": "solo"})
        self.assertEqual(reopened.collaboration_mode, "ultra")

    async def test_goal_and_collaboration_limits_are_both_enforced(self):
        run = new_run(policy("auto"))
        goal = {"budget": {"output_tokens": 30}, "output_used": 0, "output_reserved": 0}
        t1, t2 = CURRENT_BUDGET.set(goal), DELEGATION_BUDGET.set(run)
        try:
            budgets = active_budgets(goal)
            self.assertEqual(len(budgets), 2)
            amount = reserve(100, budgets)
            self.assertEqual(amount, 30)
            with self.assertRaises(RuntimeError): reserve(1, budgets)
            settle(budgets, amount, 7)
            self.assertEqual((run["output_used"], goal["output_used"]), (7, 7))
        finally:
            CURRENT_BUDGET.reset(t1); DELEGATION_BUDGET.reset(t2)

    async def test_invalid_profiles_fail_closed_and_engine_save_failure_rolls_back(self):
        for mode, settings in (("invalid", {}), ("auto", {"agent_concurrency": True}), ("auto", {"collaboration_profiles": []}), ("ultra", {"collaboration_profiles": {"ultra": {"concurrency": 0}}})):
            with self.assertRaises(ValueError): policy(mode, settings)
        with patch.object(self.engine, "persist_session_state", return_value=False):
            with self.assertRaises(OSError): self.engine.set_collaboration_mode("ultra")
        self.assertEqual(self.engine.collaboration_mode, "auto")

    async def test_legacy_planning_and_verification_share_native_lifecycle_and_limits(self):
        from xuloom.features.verification import run_verification_agent
        self.engine.permission_manager.approval_policy.set_mode("full")
        self.engine.set_collaboration_mode("solo")
        solo = await self.engine.launch_ultraplan("Plan a study")
        self.assertFalse(solo["ok"])
        self.assertIn("切换", solo["error"])
        self.assertIn("切换", await run_verification_agent(self.engine, ["report.md"]))
        self.assertEqual(self.engine.subagent_registry, [])
        self.engine.set_collaboration_mode("auto")
        async def model(child, user_input, sys_print_callback):
            self.assertEqual(child.collaboration_mode, "solo")
            return RunResult("EVIDENCE_AND_PLAN", "completed", True)
        with patch.object(QueryEngine, "run", model):
            plan = await self.engine.launch_ultraplan("Plan a study")
            self.assertTrue(plan["ok"], plan)
            self.assertEqual(plan["plan"], "EVIDENCE_AND_PLAN")
            self.assertEqual(await run_verification_agent(self.engine, ["report.md"]), "EVIDENCE_AND_PLAN")
        self.assertEqual(self.engine.tool_context["collaboration_run"]["submitted"], 2)
        self.assertEqual([row["agent_type"] for row in self.engine.subagent_registry], ["plan", "verification"])
        self.assertTrue(all(row["status"] == "completed" for row in self.engine.subagent_registry))


class ConversationPolicyTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = recovery.RecoveryTests.asyncSetUp
    asyncTearDown = recovery.RecoveryTests.asyncTearDown
    fixture = recovery.RecoveryTests.fixture

    async def test_simple_question_no_workers_and_each_new_turn_has_its_own_limits(self):
        for mode in ("solo", "auto", "ultra"):
            self.engine.set_collaboration_mode(mode)
            self.requests = []
            self.fixture(["4"])
            prior = copy.deepcopy(self.engine.tool_context["collaboration_run"])
            result = await self.engine._run_loop("2+2?", sys_print_callback=lambda _: None)
            self.assertEqual(result.reason, "completed")
            self.assertEqual(self.engine.subagent_registry, [])
            self.assertNotEqual(prior["id"], self.engine.tool_context["collaboration_run"]["id"])
            self.assertEqual(self.engine.tool_context["collaboration_run"]["submitted"], 0)
            self.assertIn("# Collaboration: " + mode, json.dumps(self.requests[0]["messages"]))
