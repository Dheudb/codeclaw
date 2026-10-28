import asyncio
import shutil
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from xuloom.core.run_result import PersistenceError
from xuloom.core.session import SessionManager
from xuloom.core.session_lock import SessionBusyError, SessionConflictError
from xuloom.desktop.bridge import DesktopBridge
from xuloom.protocols.mcp_bridge import MCP_AVAILABLE
import test_agent_lifecycle as lifecycle


class ShutdownTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = lifecycle.LifecycleTests.asyncSetUp
    make_engine = lifecycle.LifecycleTests.make_engine

    async def asyncTearDown(self):
        for manager in getattr(self, "other_managers", []): manager.release_writer()
        await lifecycle.LifecycleTests.asyncTearDown(self)

    def own(self):
        self.engine.require_writer()
        self.assertTrue(self.engine.persist_session_state())
        manager = SessionManager(cwd=self.root)
        self.other_managers = getattr(self, "other_managers", []) + [manager]
        return manager

    async def test_concurrent_close_is_single_and_cancelling_waiter_does_not_cancel_cleanup(self):
        other = self.own(); entered, release = asyncio.Event(), asyncio.Event()
        original = self.engine.browser_manager.close
        async def blocked():
            entered.set(); await release.wait(); await original()
        with patch.object(self.engine.browser_manager, "close", side_effect=blocked) as close:
            first = asyncio.create_task(self.engine.aclose())
            try:
                await asyncio.wait_for(entered.wait(), 3)
                first.cancel()
                with self.assertRaises(asyncio.CancelledError): await first
                with self.assertRaises(SessionBusyError): other.acquire_writer(self.engine.session_id)
                result = await self.engine.run("do not restart a closing engine", lambda _: None)
                self.assertEqual(result.reason, "session_conflict")
                self.assertFalse((await self.engine.agents("capacity"))["ok"])
                self.engine.permission_manager.approval_policy.set_mode("full")
                target = self.root / "must-not-write.txt"
                tool = await self.engine._execute_tool_wrapper({"id": "closed-write", "name": "file_write_tool", "input": {"file_path": str(target), "content": "forbidden"}}, lambda _: None)
                self.assertTrue(tool["is_error"]); self.assertFalse(target.exists())
                with self.assertRaises(SessionConflictError): await self.engine.start_protocols(lambda _: None)
            finally: release.set()
            await asyncio.gather(self.engine.aclose(), self.engine.aclose())
            await self.engine.aclose(); close.assert_awaited_once()
        self.assertFalse(self.engine.persist_session_state())
        self.assertFalse(self.engine.session_manager.owns_writer(self.engine.session_id))
        other.acquire_writer(self.engine.session_id)

    async def test_client_failure_attempts_other_resources_and_close_can_retry_without_reopening(self):
        other = self.own()
        server = SimpleNamespace(stop=AsyncMock())
        self.engine.lsp_manager = SimpleNamespace(servers={"fixture": server})
        self.engine.mcp_bridge = SimpleNamespace(cleanup=AsyncMock())
        with patch.object(self.engine.client, "close", side_effect=OSError("client close failure")), patch.object(self.engine.browser_manager, "close", wraps=self.engine.browser_manager.close) as browser:
            with self.assertRaisesRegex(OSError, "client close failure"): await self.engine.aclose()
            browser.assert_awaited_once(); server.stop.assert_awaited_once(); self.engine.mcp_bridge.cleanup.assert_awaited_once()
            with self.assertRaises(SessionBusyError): other.acquire_writer(self.engine.session_id)
            self.assertEqual((await self.engine.run("closed", lambda _: None)).reason, "session_conflict")
        await self.engine.aclose(); other.acquire_writer(self.engine.session_id)

    async def test_final_checkpoint_failure_is_reported_and_retried_with_original_intact(self):
        other = self.own()
        path = Path(self.engine.session_manager.session_dir) / (self.engine.session_id + ".json")
        original = path.read_bytes()
        with patch.object(self.engine, "persist_session_state", return_value=False), patch.object(self.engine.client, "close", wraps=self.engine.client.close) as client:
            with self.assertRaises(PersistenceError): await self.engine.aclose()
            client.assert_awaited_once(); self.assertEqual(path.read_bytes(), original)
            with self.assertRaises(SessionBusyError): other.acquire_writer(self.engine.session_id)
        await self.engine.aclose(); other.acquire_writer(self.engine.session_id)

    async def test_failed_command_kill_does_not_skip_other_commands_or_hide_live_process_on_retry(self):
        other = self.own(); commands = self.engine.tool_context["command_sessions"]
        identities = [await commands.start(argv=[sys.executable, "-c", "import time;time.sleep(30)"], cwd=self.root, owner="parent") for _ in range(2)]
        original = commands._kill
        async def fail_first(row):
            if row["id"] == identities[0]: raise PermissionError("injected kill failure")
            await original(row)
        with patch.object(commands, "_kill", side_effect=fail_first):
            with self.assertRaises(PermissionError): await self.engine.aclose()
            self.assertEqual(commands.records[identities[0]]["status"], "cancelled")
            self.assertIsNone(commands.records[identities[0]]["process"].returncode)
            self.assertIsNotNone(commands.records[identities[1]]["process"].returncode)
            self.assertTrue(commands.records[identities[1]]["monitor"].done())
            with self.assertRaises(SessionBusyError): other.acquire_writer(self.engine.session_id)
        await self.engine.aclose()
        self.assertTrue(all(commands.records[identity]["process"].returncode is not None for identity in identities))
        other.acquire_writer(self.engine.session_id)

    async def test_child_close_only_cancels_its_commands_and_keeps_parent_resources_and_lease(self):
        other = self.own(); commands = self.engine.tool_context["command_sessions"]
        parent = await commands.start(argv=[sys.executable, "-c", "import time;time.sleep(30)"], cwd=self.root, owner="parent")
        owned = await commands.start(argv=[sys.executable, "-c", "import time;time.sleep(30)"], cwd=self.root, owner="child")
        child = self.make_engine(); child.agent_depth = 1; child.agent_id = "child"
        self.tool._share_runtime_managers(child)
        with patch.object(self.engine.browser_manager, "close", wraps=self.engine.browser_manager.close) as browser:
            await child.aclose(); browser.assert_not_called()
        self.assertIsNotNone(commands.records[owned]["process"].returncode)
        self.assertIsNone(commands.records[parent]["process"].returncode)
        with self.assertRaises(SessionBusyError): other.acquire_writer(self.engine.session_id)
        await self.engine.aclose(); other.acquire_writer(self.engine.session_id)

    async def test_desktop_close_still_closes_engine_after_stop_failure(self):
        other = self.own()
        bridge = DesktopBridge(lambda _: None, self.root); bridge.engine = self.engine
        with patch.object(bridge, "dispatch", side_effect=OSError("stop failure")), patch.object(self.engine, "aclose", wraps=self.engine.aclose) as close:
            with self.assertRaisesRegex(OSError, "stop failure"): await bridge.close()
            close.assert_awaited_once(); self.assertTrue(bridge.protocol_shutdown.is_set())
        other.acquire_writer(self.engine.session_id)

    @unittest.skipUnless(MCP_AVAILABLE and shutil.which("node"), "optional MCP SDK and Node.js required")
    async def test_desktop_protocol_owner_finishes_real_stdio_connection_before_common_cleanup(self):
        other = self.own()
        bridge = DesktopBridge(lambda _: None, self.root); bridge.engine = self.engine
        fixture = Path(__file__).with_name("fixtures") / "mcp-server.cjs"
        bridge.mcp_store.save("fixture", {"command": shutil.which("node"), "args": [str(fixture)]})
        bridge.protocol_ready = asyncio.get_running_loop().create_future()
        bridge.protocol_task = asyncio.create_task(bridge.protocol_lifetime(self.engine))
        try:
            await asyncio.wait_for(asyncio.shield(bridge.protocol_ready), 15)
            self.assertTrue(self.engine.mcp_bridge.sessions)
            connections = [task for task, _ in self.engine.mcp_bridge.connections]
            await asyncio.wait_for(bridge.close(), 15)
            self.assertTrue(bridge.protocol_task.done()); self.assertTrue(all(task.done() for task in connections))
            self.assertEqual(self.engine.mcp_bridge.sessions, {})
            other.acquire_writer(self.engine.session_id)
        finally:
            await bridge.close()
