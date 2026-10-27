"""MCP discovery uses native catalogs, authorization and protocol ownership."""
import asyncio
import json
import shutil
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch

from codeclaw.desktop.bridge import DesktopBridge
from codeclaw.protocols.mcp_bridge import MCP_AVAILABLE, MCPToolWrapper
import test_tool_recovery as recovery


class ToolDiscoveryTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = recovery.RecoveryTests.asyncSetUp
    asyncTearDown = recovery.RecoveryTests.asyncTearDown
    fixture = recovery.RecoveryTests.fixture

    def remote(self):
        session = NS(call_tool=AsyncMock(return_value=NS(content=[NS(type="text", text="REMOTE_PROOF")], isError=False)))
        return MCPToolWrapper("fixture", session, NS(name="lookup", description="REMOTE_DETAILS_PROOF",
            inputSchema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}))

    async def test_cli_discovery_loads_schema_on_next_request_and_keeps_explicit_ask(self):
        remote = self.remote()
        with patch("codeclaw.core.engine.MCPBridge.load_and_connect", new=AsyncMock(return_value={remote.name: remote})):
            await self.engine.start_protocols(lambda _: None)
        self.assertNotIn(remote.name, self.engine.available_tools)
        self.assertIn(remote.name, self.engine.latent_tools)
        approval = AsyncMock(return_value="allow")
        self.engine.set_permission_handler(approval)
        self.engine.permission_manager.approval_policy.set_mode("full")
        self.engine.permission_manager.always_ask_tools.add(remote.name)
        self.fixture([
            ("tool_search_tool", json.dumps({"query": "fixture lookup", "activate": [remote.name]})),
            (remote.name, '{"query":"evidence"}'), "COMPLETE",
        ])
        result = await self.engine.run("Use the remote lookup when needed", sys_print_callback=lambda _: None)
        self.assertTrue(result.ok, result.to_dict())
        self.assertNotIn("REMOTE_DETAILS_PROOF", json.dumps(self.requests[0]))
        self.assertIn("REMOTE_DETAILS_PROOF", json.dumps(self.requests[1]["tools"]))
        self.assertIn("REMOTE_PROOF", json.dumps(self.requests[2]["messages"]))
        remote.session.call_tool.assert_awaited_once_with("lookup", arguments={"query": "evidence"})
        approval.assert_awaited_once()

    async def test_activation_does_not_bypass_deny_or_plan(self):
        remote = self.remote()
        self.engine.register_mcp_tools({remote.name: remote})
        call = {"id": "hidden", "name": remote.name, "input": {"query": "evidence"}}
        self.assertTrue((await self.engine._execute_tool_wrapper(call, lambda _: None))["is_error"])
        activated = await self.engine._execute_tool_wrapper({"id": "discover", "name": "tool_search_tool",
            "input": {"query": "lookup", "activate": [remote.name]}}, lambda _: None)
        self.assertFalse(activated["is_error"])
        self.engine.permission_manager.approval_policy.set_mode("full")
        self.engine.permission_manager.always_deny_tools.add(remote.name)
        call["id"] = "denied"
        self.assertTrue((await self.engine._execute_tool_wrapper(call, lambda _: None))["is_error"])
        self.engine.permission_manager.always_deny_tools.clear()
        self.engine.set_mode("plan")
        call["id"] = "plan-denied"
        self.assertTrue((await self.engine._execute_tool_wrapper(call, lambda _: None))["is_error"])
        remote.session.call_tool.assert_not_awaited()

    async def test_registration_checks_active_and_latent_conflicts_before_mutating(self):
        for conflict in ("file_read_tool", "repl_tool"):
            remote = self.remote()
            with self.assertRaisesRegex(ValueError, "conflicts"):
                self.engine.register_mcp_tools({remote.name: remote, conflict: remote})
            self.assertNotIn(remote.name, self.engine.latent_tools)
            self.assertNotIn(remote.name, self.engine.available_tools)

    @unittest.skipUnless(MCP_AVAILABLE and shutil.which("node"), "optional MCP SDK and Node.js required")
    async def test_desktop_reload_replaces_wrappers_preserves_activation_and_removes_disabled_tools(self):
        bridge = DesktopBridge(lambda _: None, self.root)
        bridge.engine = self.engine
        bridge.mcp_store.save("fixture", {"command": shutil.which("node"),
            "args": [str(Path(__file__).with_name("fixtures") / "mcp-server.cjs")]})
        bridge.protocol_ready = asyncio.get_running_loop().create_future()
        bridge.protocol_task = asyncio.create_task(bridge.protocol_lifetime(self.engine))
        try:
            await asyncio.wait_for(asyncio.shield(bridge.protocol_ready), 15)
            original = self.engine.latent_tools["fixture_search"]
            self.assertNotIn("fixture_search", self.engine.available_tools)
            self.engine.activate_tools(["fixture_search"])
            await asyncio.wait_for(bridge.reload_mcp(), 15)
            replacement = self.engine.available_tools["fixture_search"]
            self.assertIsNot(replacement, original)
            self.assertIsNot(replacement.session, original.session)
            self.engine.permission_manager.approval_policy.set_mode("full")
            result = await self.engine._execute_tool_wrapper({"id": "reloaded", "name": replacement.name,
                "input": {"query": "reload-proof"}}, lambda _: None)
            self.assertFalse(result["is_error"])
            self.assertIn("SEARCH_FIXTURE_PROOF: reload-proof", result["content"])
            bridge.mcp_store.toggle("fixture", False)
            await asyncio.wait_for(bridge.reload_mcp(), 15)
            self.assertNotIn("fixture_search", self.engine.available_tools)
            self.assertNotIn("fixture_search", self.engine.latent_tools)
        finally:
            await bridge.close()
