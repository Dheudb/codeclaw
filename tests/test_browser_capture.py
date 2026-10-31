import asyncio
import base64
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, AsyncMock
from types import SimpleNamespace

from PIL import Image
from xuloom.features.browser import BrowserSessionManager
from xuloom.tools.browser_tool import BrowserTool, BrowserToolInput
from xuloom.desktop.bridge import DesktopBridge
from xuloom.security.execution import ExecutionSandbox, SandboxUnavailable


class BrowserCaptureTests(unittest.IsolatedAsyncioTestCase):
    async def test_partial_browser_startup_is_cleaned_and_retry_succeeds(self):
        manager = BrowserSessionManager()
        driver = SimpleNamespace(chromium=SimpleNamespace(launch=AsyncMock(side_effect=RuntimeError("launch failed"))), stop=AsyncMock())
        factory = SimpleNamespace(start=AsyncMock(return_value=driver))
        with patch("xuloom.features.browser.PLAYWRIGHT_AVAILABLE", True), patch("xuloom.features.browser.async_playwright", return_value=factory):
            with self.assertRaisesRegex(RuntimeError, "launch failed"):
                await manager.ensure_page()
            driver.stop.assert_awaited_once()
            self.assertIsNone(manager._playwright)
            page = SimpleNamespace(close=AsyncMock())
            context = SimpleNamespace(new_page=AsyncMock(return_value=page), close=AsyncMock())
            browser = SimpleNamespace(new_context=AsyncMock(return_value=context), close=AsyncMock())
            driver.chromium.launch = AsyncMock(return_value=browser)
            self.assertIs(await manager.ensure_page(), page)
            await manager.close()

    async def test_close_attempts_all_handles_even_when_page_close_fails(self):
        manager = BrowserSessionManager()
        manager._page = SimpleNamespace(close=AsyncMock(side_effect=RuntimeError("page close failed")))
        context = manager._context = SimpleNamespace(close=AsyncMock())
        browser = manager._browser = SimpleNamespace(close=AsyncMock())
        driver = manager._playwright = SimpleNamespace(stop=AsyncMock())
        with self.assertRaisesRegex(RuntimeError, "page close failed"):
            await manager.close()
        context.close.assert_awaited_once(); browser.close.assert_awaited_once(); driver.stop.assert_awaited_once()
        await manager.close()

    async def test_malformed_host_response_is_an_explicit_failure(self):
        manager = BrowserSessionManager()
        manager.capture_callback = AsyncMock(return_value=["invalid"])
        with self.assertRaisesRegex(RuntimeError, "no valid result"):
            await manager.render(url="http://localhost")

    def png(self):
        output = io.BytesIO()
        Image.new("RGB", (800, 600), "#aabbcc").save(output, format="PNG")
        return base64.b64encode(output.getvalue()).decode()

    async def test_desktop_capture_returns_native_image_and_preserves_screenshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "page.html"
            source.write_text("<h1>fixture</h1>")
            manager = BrowserSessionManager(str(root / "captures"))
            calls = []
            async def host(input):
                calls.append(input)
                return {"png_base64": self.png(), "url": str(source), "title": "fixture", "backend": "electron-hidden"}
            manager.capture_callback = host
            tool = BrowserTool()
            tool.context = {"browser_manager": manager, "session_cwd": str(root), "model_input_modalities": ["text", "image"]}
            with patch("xuloom.features.browser.PLAYWRIGHT_AVAILABLE", False):
                result = await tool(action="render", file_path="page.html", width=800, height=600, wait_ms=0)
            self.assertTrue(result["ok"])
            self.assertEqual(result["metadata"]["width"], 800)
            self.assertTrue(Path(result["metadata"]["path"]).is_file())
            self.assertTrue(Path(calls[0]["file_path"]).samefile(source))
            self.assertTrue(any(block.get("type") == "image" for block in result["content"]))
            # A text-only model gets a truthful receipt, never a fake visual inspection.
            tool.context["model_input_modalities"] = ["text"]
            text = await tool(action="render", file_path="page.html")
            self.assertFalse(text["metadata"]["image_attached"])
            self.assertIn("Visual inspection unavailable", text["content"])
            self.assertTrue(Path(text["metadata"]["path"]).is_file())

    async def test_path_and_protocol_validation_happens_before_host(self):
        with tempfile.TemporaryDirectory() as directory:
            tool = BrowserTool()
            tool.context = {"session_cwd": directory}
            for input in [dict(file_path="../outside.html"), dict(file_path="page.py"), dict(url="file:///secret.html"),
                dict(url="https://user:password@example.com"), dict(url="javascript:alert(1)"), {}, dict(url="https://example.com", file_path="page.html")]:
                with self.assertRaises(ValueError):
                    await tool(action="render", **input)
            with self.assertRaises(ValueError): BrowserToolInput(action="render", width=99999)
            with self.assertRaises(SandboxUnavailable): ExecutionSandbox(directory, "workspace").check_tool("browser_tool")

    async def test_host_failures_do_not_become_successful_images(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = BrowserSessionManager(directory)
            async def broken(input): return {"error": "navigation failed"}
            manager.capture_callback = broken
            tool = BrowserTool()
            tool.context = {"browser_manager": manager}
            result = await tool(action="render", url="http://localhost:1234")
            self.assertFalse(result["ok"])
            self.assertIn("navigation failed", result["content"])
            self.assertEqual(list(Path(directory).glob("*.png")), [])

    async def test_host_cancellation_and_late_replies_release_requests(self):
        # Exercise the real RPC control path without constructing a model engine.
        bridge = object.__new__(DesktopBridge)
        bridge.capture_pending = {}
        events = []
        bridge.output = events.append
        task = asyncio.create_task(bridge.capture({"url": "http://localhost"}))
        await asyncio.sleep(0)
        request = events[0]["requestId"]
        self.assertTrue(await bridge.dispatch("browser.capture_result", {"requestId": request, "result": {"title": "done"}}))
        self.assertEqual(await task, {"title": "done"})
        self.assertEqual(bridge.capture_pending, {})
        self.assertEqual(events[-1]["type"], "browser_capture_cancel")
        task = asyncio.create_task(bridge.capture({"url": "http://localhost"}))
        await asyncio.sleep(0)
        request = events[-1]["requestId"]
        task.cancel()
        with self.assertRaises(asyncio.CancelledError): await task
        self.assertEqual(bridge.capture_pending, {})
        self.assertFalse(await bridge.dispatch("browser.capture_result", {"requestId": request, "result": {}}))

    async def test_cli_render_uses_optional_playwright_with_explicit_missing_dependency(self):
        with tempfile.TemporaryDirectory() as directory, patch("xuloom.features.browser.PLAYWRIGHT_AVAILABLE", False):
            tool = BrowserTool()
            tool.context = {"browser_manager": BrowserSessionManager(directory)}
            result = await tool(action="render", url="http://localhost:1234")
            self.assertFalse(result["ok"])
            self.assertIn("Playwright is not installed", result["content"])


if __name__ == "__main__": unittest.main()
