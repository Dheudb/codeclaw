import asyncio
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from codeclaw.desktop.bridge import DesktopBridge
from codeclaw.protocols.mcp_bridge import MCPBridge, MCP_AVAILABLE
from codeclaw.protocols.mcp_config import MCPConfigStore, validate_server
from codeclaw.protocols.mcp_market import MCPMarket, BUILTIN, materialize, registry_entry
from codeclaw.context.attachments import AttachmentCollector
from codeclaw.tools.web_tool import WebSearchTool


class MCPSettingsTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.store = MCPConfigStore(self.root / "mcp_servers.json")

    def test_credentials_redacted_preserved_and_explicitly_removed(self):
        self.store.save("search", {"url": "https://example.invalid/mcp?token=hidden-query", "headers": {"Authorization": "Bearer hidden-key"}})
        snapshot = self.store.snapshot()[0]
        self.assertNotIn("hidden-", json.dumps(snapshot))
        self.store.save("search", snapshot, existing=True)
        self.assertEqual(self.store.read()["mcpServers"]["search"]["headers"]["Authorization"], "Bearer hidden-key")
        self.assertIn("hidden-query", self.store.read()["mcpServers"]["search"]["url"])
        snapshot["headers"] = {}
        self.store.save("search", snapshot, existing=True)
        self.assertEqual(self.store.read()["mcpServers"]["search"]["headers"], {})

    def test_install_does_not_overwrite_and_keeps_unrelated_settings(self):
        self.store.path.write_text(json.dumps({"mcpServers": {}, "extra": "preserve"}))
        self.store.save("search", {"command": "node", "args": []})
        with self.assertRaises(ValueError):
            self.store.save("search", {"command": "other"})
        self.store.toggle("search", False)
        self.assertFalse(self.store.read()["mcpServers"]["search"]["enabled"])
        self.store.remove("search")
        self.assertEqual(self.store.read(), {"mcpServers": {}, "extra": "preserve"})

    def test_invalid_config_is_not_overwritten(self):
        self.store.path.write_text("broken")
        with self.assertRaises(ValueError):
            self.store.save("search", {"command": "node"})
        self.assertEqual(self.store.path.read_text(), "broken")

    def test_rejects_header_injection_invalid_names_and_transports(self):
        for config in ({"url": "file:///tmp"}, {"url": "https://user:pass@example.com"}, {"url": "https://example.com", "headers": {"X-Key": "one\r\ntwo"}}, {"command": "node", "args": "--bad"}, {"type": "sse", "url": "https://example.com"}):
            with self.assertRaises(ValueError):
                validate_server(config)
        with self.assertRaises(ValueError):
            self.store.save("../../escape", {"command": "node"})

    def test_catalog_requires_inputs_and_installs_disabled(self):
        brave = next(e for e in BUILTIN if e["id"] == "brave-search")
        with self.assertRaises(ValueError):
            materialize(brave, {})
        config = materialize(brave, {"BRAVE_API_KEY": "fixture-key"})
        self.assertFalse(config["enabled"])
        self.assertEqual(config["env"], {"BRAVE_API_KEY": "fixture-key"})
        exa = materialize(BUILTIN[0], {})
        self.assertEqual(exa["headers"], {})

    def test_registry_preserves_pinned_package_arguments_and_inputs(self):
        entry = registry_entry({"server": {"name": "org.example/search", "packages": [{"registryType": "npm", "identifier": "@example/search", "version": "1.2.3", "packageArguments": [{"type": "named", "name": "--mode", "value": "search"}], "environmentVariables": [{"name": "API_KEY", "isRequired": True, "isSecret": True}]}]}})
        self.assertEqual(entry["config"]["args"], ["-y", "@example/search@1.2.3", "--mode", "search"])
        self.assertEqual(materialize(entry, {"API_KEY": "test"})["env"], {"API_KEY": "test"})
        unsupported = registry_entry({"name": "example/unsupported", "packages": [{"registryType": "npm", "identifier": "example", "runtimeArguments": [{"value": "--custom"}]}]})
        self.assertIsNone(unsupported)

    def test_registry_remote_variable_is_expanded(self):
        entry = registry_entry({"name": "example/remote", "remotes": [{"type": "streamable-http", "url": "https://example.invalid/mcp", "headers": [{"name": "Authorization", "value": "Bearer {api_key}", "variables": {"api_key": {"isRequired": True}}}]}]})
        self.assertEqual(materialize(entry, {"api_key": "fixture-key"})["headers"], {"Authorization": "Bearer fixture-key"})
        implicit = registry_entry({"name": "example/implicit", "remotes": [{"type": "streamable-http", "url": "https://example.invalid/mcp", "headers": [{"name": "Authorization", "value": "Bearer {api_key}"}]}]})
        with self.assertRaises(ValueError):
            materialize(implicit, {})
        self.assertEqual(materialize(implicit, {"api_key": "fixture-key"})["headers"], {"Authorization": "Bearer fixture-key"})

    def test_mcp_secrets_excluded_from_automatic_attachments(self):
        self.store.save("search", {"command": "node", "env": {"API_KEY": "secret-proof"}})
        collector = AttachmentCollector()
        attachment = collector._build_file_change_delta(str(self.root), [str(self.store.path)])
        self.assertIsNone(attachment)


class MCPMarketTests(unittest.IsolatedAsyncioTestCase):
    async def test_builtin_search_remains_available_without_mcp_and_does_not_block(self):
        with patch("codeclaw.tools.web_tool.DDGS") as ddgs:
            ddgs.return_value.text.return_value = [{"title": "Fixture", "href": "https://example.invalid/source", "body": "Snippet"}]
            result = await WebSearchTool().execute("fixture", 50)
        self.assertIn("\nURL: https://example.invalid/source\n", result["content"])
        self.assertNotIn("\\n", result["content"])
        ddgs.return_value.text.assert_called_once_with("fixture", max_results=10)

    async def test_market_no_network_for_curated_and_pagination_for_registry(self):
        market = MCPMarket()
        self.assertEqual(len((await market.search())["entries"]), len(BUILTIN))
        requests = []
        def respond(request):
            requests.append(request)
            return httpx.Response(200, json={"servers": [{"server": {"name": "example/remote", "remotes": [{"type": "streamable-http", "url": "https://example.invalid/mcp"}]}}, {"server": {"name": "example/unsupported"}}], "metadata": {"nextCursor": "page-2"}})
        factory = httpx.AsyncClient
        with patch("codeclaw.protocols.mcp_market.httpx.AsyncClient", side_effect=lambda **kwargs: factory(transport=httpx.MockTransport(respond), **kwargs)):
            result = await market.search("search", "registry", "page-1")
        self.assertEqual(result["cursor"], "page-2")
        self.assertEqual(result["skipped"], 1)
        self.assertEqual(requests[0].url.params["search"], "search")
        self.assertEqual(requests[0].url.params["cursor"], "page-1")
        self.assertIn(result["entries"][0]["id"], market.entries)

    async def test_unavailable_registry_keeps_curated_available(self):
        market = MCPMarket()
        factory = httpx.AsyncClient
        def failure(request):
            raise httpx.ConnectError("fixture-token-must-not-leak")
        with patch("codeclaw.protocols.mcp_market.httpx.AsyncClient", side_effect=lambda **kwargs: factory(transport=httpx.MockTransport(failure), **kwargs)):
            with self.assertRaises(ValueError) as caught:
                await market.search(source="registry")
        self.assertNotIn("fixture-token", str(caught.exception))
        self.assertTrue((await market.search())["entries"])

    async def test_desktop_market_without_model_and_shared_config(self):
        with tempfile.TemporaryDirectory() as directory:
            bridge = DesktopBridge(lambda e: None, directory)
            await bridge.dispatch("mcp.install", {"id": "exa", "values": {}})
            snapshot = await bridge.dispatch("mcp.list", {})
            self.assertFalse(snapshot["servers"][0]["enabled"])
            self.assertIsNone(bridge.engine)
            shared = MCPConfigStore(Path(directory) / "mcp_servers.json")
            self.assertEqual(shared.read()["mcpServers"]["exa"]["url"], BUILTIN[0]["config"]["url"])
            await bridge.dispatch("mcp.remove", {"name": "exa"})
            self.assertEqual(shared.read()["mcpServers"], {})


@unittest.skipUnless(MCP_AVAILABLE and shutil.which("node"), "Requires Python 3.10+, mcp SDK and Node.js")
class MCPConnectionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.fixture = str(Path(__file__).parent / "fixtures" / "mcp-server.cjs")
        self.server = await asyncio.create_subprocess_exec(shutil.which("node"), self.fixture, "--http", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        port = json.loads(await asyncio.wait_for(self.server.stdout.readline(), 10))["port"]
        self.url = "http://127.0.0.1:" + str(port) + "/mcp"
        self.store = MCPConfigStore(self.root / "mcp_servers.json")

    async def asyncTearDown(self):
        self.server.terminate()
        await self.server.wait()

    async def test_real_stdio_and_http_tool_lifecycle(self):
        self.store.save("local", {"command": shutil.which("node"), "args": [self.fixture], "env": {"SEARCH_API_KEY": "explicit-fixture"}})
        self.store.save("remote", {"url": self.url, "headers": {"Authorization": "Bearer fixture-secret"}})
        self.store.save("disabled", {"command": "does-not-exist", "enabled": False})
        bridge = MCPBridge(str(self.store.path))
        try:
            with patch.dict(os.environ, {"CODECLAW_TEST_MODEL_SECRET": "must-not-inherit"}):
                tools = await bridge.load_and_connect()
            self.assertEqual(set(tools), {"local_search", "remote_search"})
            self.assertIn("SEARCH_FIXTURE_PROOF", await tools["local_search"](query="local"))
            self.assertEqual(await tools["local_search"](query="__env__"), "absent:explicit-fixture")
            self.assertEqual(await tools["remote_search"](query="__header__"), "Bearer fixture-secret")
            self.assertTrue((await tools["remote_search"](query="fail")).startswith("MCP tool returned an error"))
            self.assertEqual(bridge.statuses["disabled"]["status"], "disabled")
        finally:
            await bridge.cleanup()
        self.assertEqual(bridge.sessions, {})

    async def test_failed_service_does_not_prevent_next_service(self):
        self.store.save("missing", {"command": "codeclaw-fixture-command-does-not-exist"})
        self.store.save("working", {"url": self.url})
        bridge = MCPBridge(str(self.store.path))
        try:
            tools = await bridge.load_and_connect()
            self.assertEqual(bridge.statuses["missing"]["status"], "error")
            self.assertIn("working_search", tools)
        finally:
            await bridge.cleanup()

    async def test_desktop_test_needs_no_model_and_owner_reloads_in_place(self):
        self.store.save("remote", {"url": self.url, "enabled": False})
        desktop = DesktopBridge(lambda e: None, self.root)
        tested = await desktop.dispatch("mcp.test", {"name": "remote"})
        self.assertEqual(tested["status"], "connected")
        self.assertIsNone(desktop.engine)
        engine = SimpleNamespace(available_tools={}, tool_context={})
        desktop.engine = engine
        desktop.protocol_ready = asyncio.get_running_loop().create_future()
        desktop.protocol_task = asyncio.create_task(desktop.protocol_lifetime(engine))
        try:
            await desktop.protocol_ready
            self.assertNotIn("remote_search", engine.available_tools)
            self.store.toggle("remote", True)
            await desktop.reload_mcp()
            self.assertIn("remote_search", engine.available_tools)
            self.store.toggle("remote", False)
            await desktop.reload_mcp()
            self.assertNotIn("remote_search", engine.available_tools)
        finally:
            await desktop.mcp_requests.put(None)
            await desktop.protocol_task
