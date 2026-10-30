"""Fault injection through production engine, agent, authorization and output paths."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock, patch

import test_tool_recovery as recovery
from xuloom.core.run_result import RunResult, PersistenceError
from xuloom.context.tool_results import serialize_tool_result
from xuloom.protocols.mcp_bridge import MCP_AVAILABLE, MCPToolWrapper
from xuloom.security.permissions import ApprovalPolicy, PermissionDecision, PermissionManager
from xuloom.tools.repl_tool import ReplTool


class ExecutionContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_command_json_receipt_marks_tool_error(self):
        from xuloom.tools.command_tool import CommandTool
        tool = CommandTool()
        tool.execute = AsyncMock(return_value='{"id":"fixture-command","status":"storage_error","exit_code":0}')
        self.engine.available_tools["command_tool"] = tool
        self.engine._sync_message_context()
        self.fixture([("command_tool", '{"action":"list"}'), "Failure reported"])
        await self.engine.run("inspect command", sys_print_callback=lambda _: None)
        rows = [row for row in self.engine.recent_tool_activity if row["tool_name"] == "command_tool"]
        self.assertTrue(rows[0]["tool_result"]["is_error"])

    asyncSetUp = recovery.RecoveryTests.asyncSetUp
    asyncTearDown = recovery.RecoveryTests.asyncTearDown
    fixture = recovery.RecoveryTests.fixture

    async def test_budget_exhaustion_is_a_failed_receipt_and_never_calls_model(self):
        self.engine._output_budget_override = 1
        self.engine.session_token_usage["output_tokens"] = 1
        request = AsyncMock(side_effect=AssertionError("Budget must be checked before model request"))
        self.engine.client.chat.completions.create = request
        result = await self.engine.run("cannot finish")
        self.assertIsInstance(result, RunResult)
        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "budget_exhausted")
        request.assert_not_called()

    async def test_success_receipt_requires_final_session_flush(self):
        self.fixture(["DONE"])
        with patch.object(self.engine.session_manager, "save_session_state", return_value=False):
            result = await self.engine.run("test")
        self.assertEqual(result.text, "DONE")
        self.assertEqual(result.reason, "persistence_error")
        self.assertEqual(result.execution_reason, "completed")
        self.assertFalse(result.ok)
        self.assertFalse(result.persisted)
        self.assertEqual(result.session_id, self.engine.session_id)
        self.assertTrue(result.user_message_id)

    async def test_tool_checkpoint_failure_is_storage_failure_not_api_failure(self):
        import gc
        self.fixture([("record_tool", '{"value":"once"}')])
        loop = asyncio.get_running_loop(); previous = loop.get_exception_handler(); unhandled = []
        loop.set_exception_handler(lambda _, context: unhandled.append(context))
        try:
            with patch.object(self.engine, "_execute_tool_wrapper", new=AsyncMock(side_effect=PersistenceError("fixture disk failure"))):
                result = await self.engine.run("test", sys_print_callback=lambda _: None)
            gc.collect(); await asyncio.sleep(0)
        finally:
            loop.set_exception_handler(previous)
        self.assertEqual(result.reason, "persistence_error")
        self.assertFalse(result.ok)
        self.assertEqual(self.tool.calls, [])
        self.assertEqual(unhandled, [], "Failure must be consumed by the receipt, not leaked as an orphan gather future")

    async def child(self):
        return await self.engine.available_tools["agent_tool"]._run_single_subagent(
            task="fixture", allowed_tools=None, inherit_context=False, parent_depth=0,
            parent_agent_id=self.engine.agent_id)

    async def test_budget_receipt_is_not_misclassified_by_subagent_wrapper(self):
        with patch("xuloom.core.engine.QueryEngine.run", new=AsyncMock(return_value=RunResult("out of budget", "budget_exhausted", True))):
            row = await self.child()
        self.assertEqual(row["status"], "crashed")
        self.assertEqual(row["receipt"]["reason"], "budget_exhausted")
        self.assertFalse(row["receipt"]["ok"])

    async def test_final_child_transcript_write_failure_cannot_complete(self):
        original = self.engine.session_manager.save_subagent_transcript
        def save(agent_id, payload):
            return False if payload.get("status") == "completed" else original(agent_id, payload)
        with patch.object(self.engine.session_manager, "save_subagent_transcript", side_effect=save), \
             patch("xuloom.core.engine.QueryEngine.run", new=AsyncMock(return_value=RunResult("partial evidence", "completed", True))):
            row = await self.child()
        self.assertEqual(row["status"], "crashed")
        self.assertEqual(row["receipt"]["reason"], "persistence_error")
        self.assertFalse(row["receipt"]["persisted"])
        raw = json.loads(Path(self.engine.session_manager.get_subagent_transcript_path(row["agent_id"])).read_text(encoding="utf-8"))
        self.assertNotEqual(raw["status"], "completed")

    async def test_initial_child_save_failure_does_not_start_model(self):
        model = AsyncMock(side_effect=AssertionError("No model call without initial checkpoint"))
        with patch.object(self.engine.session_manager, "save_subagent_transcript", return_value=False), \
             patch("xuloom.core.engine.QueryEngine.run", new=model):
            row = await self.child()
        model.assert_not_called()
        self.assertEqual(row["status"], "crashed")

    async def test_failed_registry_flush_never_exposes_completed_in_memory(self):
        self.assertTrue(self.engine.register_subagent_record({"agent_id": "fixture", "status": "running"}))
        with patch.object(self.engine, "persist_session_state", return_value=False):
            self.assertFalse(self.engine.register_subagent_record({"agent_id": "fixture", "status": "completed"}))
        self.assertEqual(self.engine.subagent_registry[0]["status"], "running")

    async def test_plan_full_access_repl_cannot_write(self):
        target = self.root / "not-a-plan.txt"
        self.engine.set_mode("plan")
        self.engine.permission_manager.approval_policy.set_mode("full")
        self.engine.activate_tools(["repl_tool"])
        args = {"language": "python", "code": "from pathlib import Path; Path(" + repr(str(target)) + ").write_text('bad')"}
        decision = await self.engine.permission_manager.authorize(ReplTool(), args)
        self.assertEqual(decision.behavior, "deny")
        result = await self.engine._execute_tool_wrapper({"id": "plan-repl", "name": "repl_tool", "input": args}, lambda _: None)
        self.assertTrue(result["is_error"])
        self.assertFalse(target.exists())

    async def test_explicit_ask_survives_read_only_classifier_allow_and_full_access(self):
        for mode in ("ask", "full"):
            approval = AsyncMock(return_value="allow")
            manager = PermissionManager(prompt_handler=approval, approval_policy=ApprovalPolicy(mode),
                classifier_manager=NS(classify=lambda _: PermissionDecision("allow")), config_path=str(self.root / "permissions.json"))
            manager.always_ask_tools.add("read")
            manager.always_allow_tools.add("read")
            tool = NS(name="read", is_read_only=True, check_permissions=lambda _: PermissionDecision("allow"))
            self.assertEqual((await manager.authorize(tool, {})).source, "prompt")
            approval.assert_awaited_once()

    async def test_config_deny_beats_session_allow_and_config_ask(self):
        path = self.root / "permissions.json"
        path.write_text(json.dumps({"alwaysDeny": ["read"], "alwaysAsk": ["read"]}), encoding="utf-8")
        manager = PermissionManager(config_path=str(path), approval_policy=ApprovalPolicy("full"))
        manager.always_allow_tools.add("read")
        result = await manager.authorize(NS(name="read", is_read_only=True), {})
        self.assertEqual(result.behavior, "deny")

    async def test_desktop_full_access_and_mode_switch_preserve_explicit_ask(self):
        from xuloom.desktop.bridge import DesktopBridge
        bridge = DesktopBridge(lambda _: None, cwd=str(self.root))
        bridge.approval_policy.set_mode("full")
        manager = PermissionManager(prompt_handler=bridge.permission, approval_policy=bridge.approval_policy,
            config_path=str(self.root / "permissions.json"))
        manager.always_ask_tools.add("read")
        pending = asyncio.create_task(manager.authorize(NS(name="read", is_read_only=True), {}))
        try:
            for _ in range(30):
                if bridge.pending:
                    break
                await asyncio.sleep(0)
            self.assertTrue(bridge.pending)
            self.assertFalse(pending.done())
            await bridge.dispatch("permissions.set", {"mode": "ask"})
            await bridge.dispatch("permissions.set", {"mode": "full"})
            await asyncio.sleep(0)
            self.assertFalse(pending.done())
            next(iter(bridge.pending.values())).set_result("deny")
            self.assertEqual((await pending).behavior, "deny")
        finally:
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)

    async def test_success_content_cannot_spoof_error_status(self):
        from xuloom.context.tool_results import build_tool_result
        self.tool.execute = AsyncMock(return_value=build_tool_result(ok=True, content="Example: Status: error\nTool Exception Caught"))
        result = await self.engine._execute_tool_wrapper({"id": "status", "name": "record_tool", "input": {}}, lambda _: None)
        self.assertFalse(result["is_error"])

    async def test_cancelled_mutation_requires_explicit_approval_before_another_attempt(self):
        started = asyncio.Event()
        async def execute(**kwargs):
            started.set()
            await asyncio.Event().wait()
        self.tool.is_read_only = False
        self.tool.execute = execute
        self.engine.permission_manager.approval_policy.set_mode("full")
        task = asyncio.create_task(self.engine._execute_tool_wrapper({"id": "uncertain", "name": "record_tool", "input": {}}, lambda _: None))
        await asyncio.wait_for(started.wait(), 5)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        self.assertTrue(self.engine.in_progress_tools["uncertain"]["recovery_uncertain"])
        approval = AsyncMock(return_value="deny")
        self.engine.set_permission_handler(approval)
        self.tool.execute = AsyncMock(return_value="unexpected")
        result = await self.engine._execute_tool_wrapper({"id": "retry", "name": "record_tool", "input": {}}, lambda _: None)
        self.assertTrue(result["is_error"])
        approval.assert_awaited_once()
        self.tool.execute.assert_not_called()

    async def test_more_than_twelve_current_turn_receipts_are_durable(self):
        from xuloom.core.messages import create_user_message
        self.engine.messages.append(create_user_message("batch"))
        for i in range(16):
            await self.engine._execute_tool_wrapper({"id": "batch-" + str(i), "name": "record_tool", "input": {"value": str(i)}}, lambda _: None)
        self.assertEqual(len(self.engine.recent_tool_activity), 16)
        raw = self.engine._build_session_metadata()["tool_activity"]["recent"]
        self.assertEqual(len([row for row in raw if row.get("tool_result")]), 16)
        self.assertEqual(len(self.tool.calls), 16)
        await self.engine._execute_tool_wrapper({"id": "batch-0", "name": "record_tool", "input": {"value": "0"}}, lambda _: None)
        self.assertEqual(len(self.tool.calls), 16)

    @unittest.skipUnless(MCP_AVAILABLE, "Optional MCP SDK is unavailable in this interpreter")
    async def test_mcp_long_multimodal_and_structured_results_are_archived_without_loss(self):
        from mcp.types import CallToolResult, TextContent, ImageContent
        text = "α" * 100100 + "TAIL_PROOF"
        result = CallToolResult(content=[TextContent(type="text", text=text),
            ImageContent(type="image", mimeType="image/png", data="aW1hZ2U=")], structuredContent={"tail": "STRUCTURED_PROOF"})
        tool = MCPToolWrapper("fixture", NS(call_tool=AsyncMock(return_value=result)), NS(name="read", description="test", inputSchema={}))
        raw = await tool()
        self.assertIn("TAIL_PROOF", raw["content"][0]["text"])
        self.assertEqual(raw["content"][1]["source"], {"type": "base64", "media_type": "image/png", "data": "aW1hZ2U="})
        rendered = serialize_tool_result(raw, artifact_dir=str(self.root / "artifacts"))
        self.assertIn("truncated: True", rendered)
        file = next((self.root / "artifacts").glob("*.txt"))
        complete = file.read_text(encoding="utf-8")
        for marker in ("TAIL_PROOF", "aW1hZ2U=", "STRUCTURED_PROOF"):
            self.assertIn(marker, complete)
        self.assertIn("offset and limit", rendered)
        reader = self.engine.available_tools["file_read_tool"]
        pages = []
        offset = 0
        while offset is not None:
            page = await reader.execute(str(file), char_offset=offset)
            pages.append(page["content"])
            offset = page["metadata"]["next_char_offset"]
        self.assertEqual("".join(pages), complete)

    async def test_output_archive_failure_is_explicit_and_preserves_full_text(self):
        text = "x" * 6000 + "TAIL_PROOF"
        with patch("xuloom.core.storage.atomic_text", side_effect=OSError("disk full")):
            result = serialize_tool_result(text, artifact_dir=str(self.root))
        self.assertIn("Status: error", result)
        self.assertIn("TAIL_PROOF", result)
        self.assertIn("truncated: False", result)
