import asyncio
import os
import tempfile
import threading
import unittest
from unittest.mock import patch

from codeclaw.tools.web_tool import WebSearchTool, WebFetchInput
from codeclaw.tools.tool_search_tool import ToolSearchTool


class WebTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_results_and_provider_failures_are_distinct(self):
        with patch("codeclaw.tools.web_tool.DDGS") as client:
            client.return_value.text.return_value = [{"title": "Paper", "href": "https://example.org/paper", "body": "Chemical reasoning"}, {"href": "javascript:invalid"}]
            result = await WebSearchTool().execute("chemistry reasoning", 3)
            self.assertTrue(result["ok"])
            self.assertIn("https://example.org/paper", result["content"])
            self.assertEqual(result["metadata"]["result_count"], 1)
            client.return_value.text.return_value = []
            result = await WebSearchTool().execute("chemistry reasoning")
            self.assertTrue(result["is_error"])
            self.assertEqual(result["metadata"]["reason"], "no_results")
            client.return_value.text.side_effect = RuntimeError("provider blocked")
            result = await WebSearchTool().execute("chemistry reasoning")
            self.assertTrue(result["is_error"])
            self.assertIn("not a missing tool", result["content"])
        self.assertEqual(WebFetchInput(url="https://example.org").prompt, "")

    async def test_stop_does_not_wait_for_synchronous_search(self):
        entered, release = threading.Event(), threading.Event()
        def slow_search(*args, **kwargs):
            entered.set()
            release.wait(3)
            return []
        with patch("codeclaw.tools.web_tool.DDGS") as client:
            client.return_value.text.side_effect = slow_search
            task = asyncio.create_task(WebSearchTool().execute("query"))
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 1))
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 0.5)
            finally:
                release.set()

    async def test_natural_language_discovery_and_explicit_activation(self):
        from codeclaw.tools.web_tool import WebFetchTool
        search, fetch = WebSearchTool(), WebFetchTool()
        tool = ToolSearchTool()
        active = {search.name: search}
        latent = {fetch.name: fetch}
        def activate(names):
            for name in names:
                active[name] = latent.pop(name)
            return names
        tool.context = {"engine_available_tools": active, "latent_tools_registry": latent, "activate_tools": activate}
        for query, expected in [("web search internet access search the web for information", "web_search_tool"), ("fetch URL http request browse page", "web_fetch_tool"), ("帮我联网搜索网页", "web_search_tool")]:
            result = await tool.execute(query)
            self.assertIn(expected, result["content"])
            self.assertGreater(result["metadata"]["match_count"], 0)
        self.assertEqual((await tool.execute("unmatchablexyz123"))["metadata"]["match_count"], 0)
        result = await tool.execute("fetch URL", activate=["web_fetch_tool"])
        self.assertEqual(result["metadata"]["activated"], ["web_fetch_tool"])

    async def test_engine_exposes_search_without_discovery_and_reports_real_status(self):
        from codeclaw.core.engine import QueryEngine
        from codeclaw.memory.memory import MemoryCompactor
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"OPENAI_API_KEY": "fixture", "CODECLAW_MODEL_PROVIDER": "openai"}), patch.object(MemoryCompactor, "_resolve_token_encoder", return_value=None):
            engine = QueryEngine(model="fixture", model_provider="openai", cwd=temp)
            try:
                events = []
                engine.tool_context["capability_event"] = events.append
                self.assertIn("web_search_tool", engine.available_tools)
                self.assertIn("web_fetch_tool", engine.available_tools)
                self.assertNotIn("web_search_tool", engine.latent_tools)
                calls = [{"id": "search", "name": "web_search_tool", "input": {"query": "chemistry reasoning"}}]
                with patch("codeclaw.tools.web_tool.DDGS") as client:
                    client.return_value.text.return_value = [{"title": "Paper", "href": "https://example.org/paper", "body": "proof"}]
                    results = await engine._execute_tool_batches(calls, lambda text: None)
                    self.assertFalse(results[0]["is_error"])
                    self.assertIn("https://example.org/paper", str(results[0]))
                    client.return_value.text.return_value = []
                    results = await engine._execute_tool_batches(calls, lambda text: None)
                    self.assertTrue(results[0]["is_error"])
                    self.assertTrue([event for event in events if event["type"] == "tool_result"][-1]["is_error"])
            finally:
                await engine.client.close()
                await engine.compactor.client.close()
