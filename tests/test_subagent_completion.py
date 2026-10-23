import asyncio
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock, patch

from codeclaw.core.engine import QueryEngine
from codeclaw.memory.memory import MemoryCompactor


class CompletionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.env = patch.dict(os.environ, {"CODECLAW_DATA_DIR": str(self.root / "data"),
            "USERPROFILE": str(self.root / "home"), "HOME": str(self.root / "home"),
            "OPENAI_API_KEY": "fixture", "CODECLAW_MODEL_PROVIDER": "openai"})
        self.env.start()
        self.tokens = patch.object(MemoryCompactor, "_resolve_token_encoder", return_value=None)
        self.tokens.start()
        self.engine = QueryEngine(model="fixture", model_provider="openai", cwd=str(self.root))
        self.workers = self.engine.tool_context["subagent_tasks"]
        self.engine.session_memory_manager.maybe_extract = AsyncMock()
        self.engine.memory_extractor.maybe_extract = AsyncMock()
        self.engine._builtin_syntax_check = AsyncMock(return_value=None)
        self.engine._maybe_run_verification = AsyncMock(return_value=None)
        self.engine._maybe_prepare_auto_commit_proposal = AsyncMock()
        self.requests = []
        async def stream(**request):
            self.requests.append(request)
            answer = "PROVISIONAL" if len(self.requests) == 1 else "ALL_RESULTS_SUMMARY"
            return NS(content=[NS(type="text", text=answer)], stop_reason="end_turn", usage=NS(input_tokens=10, output_tokens=5)), [], False
        self.engine._stream_model_response = stream

    async def asyncTearDown(self):
        await self.workers.close()
        await self.engine.tool_context["command_sessions"].close()
        await self.engine.client.close()
        await self.engine.compactor.client.close()
        self.tokens.stop()
        self.env.stop()
        self.temp.cleanup()

    async def job(self, identity, gate, status="completed"):
        await gate.wait()
        return {"agent_id": identity, "status": status, "result": identity + "_OUTPUT"}

    async def test_final_waits_for_every_worker_then_resamples_without_replaying(self):
        fast, slow, waiting = asyncio.Event(), asyncio.Event(), asyncio.Event()
        self.workers.launch("FAST", self.job("FAST", fast))
        self.workers.launch("SLOW", self.job("SLOW", slow, "crashed"))
        events = []
        async def emit(event):
            events.append(event)
            if event["type"] == "subagents_waiting" and event["count"]:
                waiting.set()
        task = asyncio.create_task(self.engine._run_loop("Summarize both agents", sys_print_callback=lambda _: None, event_callback=emit))
        await asyncio.wait_for(waiting.wait(), 10)
        fast.set()
        await self.workers.tasks["FAST"]
        await asyncio.sleep(0)
        self.assertFalse(task.done())
        self.assertEqual(len(self.requests), 1, "Waiting must not spend model turns or relaunch workers")
        slow.set()
        self.assertEqual(await asyncio.wait_for(task, 10), "ALL_RESULTS_SUMMARY")
        request = json.dumps(self.requests[-1]["api_messages"])
        for text in ["FAST_OUTPUT", "SLOW_OUTPUT", "crashed", "Delegation completion guard"]:
            self.assertIn(text, request)
        self.assertEqual(set(self.workers.tasks), {"FAST", "SLOW"})
        self.assertTrue(any(e["type"] == "subagents_waiting" and e["count"] == 0 for e in events))

    async def test_result_arriving_during_final_generation_is_not_lost(self):
        gate = asyncio.Event()
        self.workers.launch("RACE", self.job("RACE", gate))
        original = self.engine._stream_model_response
        async def stream(**request):
            if not self.requests:
                gate.set()
                await self.workers.tasks["RACE"]
            return await original(**request)
        self.engine._stream_model_response = stream
        result = await self.engine._run_loop("Summarize", sys_print_callback=lambda _: None)
        self.assertEqual(result, "ALL_RESULTS_SUMMARY")
        self.assertIn("RACE_OUTPUT", json.dumps(self.requests[1]["api_messages"]))
        self.assertEqual(len(self.requests), 2)

    async def test_previously_delivered_results_do_not_force_extra_model_calls(self):
        gate = asyncio.Event(); gate.set()
        self.workers.launch("DONE", self.job("DONE", gate))
        await self.workers.tasks["DONE"]
        result = await self.engine._run_loop("Summarize", sys_print_callback=lambda _: None)
        self.assertEqual(result, "PROVISIONAL")
        self.assertEqual(len(self.requests), 1)
        self.assertIn("DONE_OUTPUT", json.dumps(self.requests[0]["api_messages"]))

    async def test_wait_is_abortable_without_accepting_the_provisional_answer(self):
        gate = asyncio.Event()
        self.workers.launch("SLOW", self.job("SLOW", gate))
        async def emit(event):
            if event["type"] == "subagents_waiting" and event["count"]:
                self.engine.abort_event.set()
        result = await self.engine._run_loop("Summarize", sys_print_callback=lambda _: None, event_callback=emit)
        self.assertIn("Turn aborted", result)
        self.assertNotIn("ALL_RESULTS_SUMMARY", result)
        self.assertEqual(len(self.requests), 1)

    async def test_empty_worker_output_is_a_failure_with_a_retained_transcript(self):
        with patch.object(QueryEngine, "run", AsyncMock(return_value="   ")):
            receipt = json.loads(await self.engine.available_tools["agent_tool"].execute(prompt="fixture", response_format="json"))
        row = receipt["agents"][0]
        self.assertEqual(row["status"], "crashed")
        self.assertIn("without a textual result", row["result"])
        saved = json.loads(Path(self.engine.session_manager.get_subagent_transcript_path(row["agent_id"])).read_text(encoding="utf-8"))
        self.assertEqual(saved["status"], "crashed")
