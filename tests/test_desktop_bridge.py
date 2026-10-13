import asyncio
import tempfile
import unittest
from pathlib import Path

from codeclaw.desktop.bridge import DesktopBridge
from codeclaw.tools.plan_mode_tools import ExitPlanModeTool
from codeclaw.features.plans import PlanManager
from codeclaw.context.attachments import AttachmentCollector
from unittest.mock import patch


class BridgeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.events = []
        self.bridge = DesktopBridge(self.events.append, self.root)

    async def test_initialization_does_not_require_model_or_create_engine(self):
        initial = await self.bridge.dispatch("initialize", {})
        self.assertEqual(initial["project"], str(self.root.resolve()))
        self.assertIsNone(self.bridge.engine)
        self.assertEqual(initial["history"], [])
        self.assertIn("explore", [agent["name"] for agent in initial["capabilities"]["agents"]])

    async def test_parallel_interactions_route_by_id(self):
        first = asyncio.create_task(self.bridge.ask("permission", {"request": {"tool_name": "one"}}))
        second = asyncio.create_task(self.bridge.ask("permission", {"request": {"tool_name": "two"}}))
        await asyncio.sleep(0)
        first_id, second_id = [event["requestId"] for event in self.events]
        await self.bridge.dispatch("respond", {"requestId": second_id, "answer": "deny"})
        self.assertEqual(await second, "deny")
        self.assertFalse(first.done())
        await self.bridge.dispatch("respond", {"requestId": first_id, "answer": "allow"})
        self.assertEqual(await first, "allow")
        with self.assertRaises(ValueError):
            await self.bridge.dispatch("respond", {"requestId": first_id, "answer": "allow"})

    async def test_stop_releases_waiting_approval(self):
        approval = asyncio.create_task(self.bridge.ask("permission", {}))
        await asyncio.sleep(0)
        await self.bridge.dispatch("stop", {})
        await asyncio.gather(approval, return_exceptions=True)
        self.assertEqual(self.bridge.pending, {})

    async def test_invalid_resume_id_cannot_escape_session_directory(self):
        with self.assertRaises(ValueError):
            await self.bridge.dispatch("initialize", {"sessionId": "../../config"})

    async def test_session_logs_do_not_feed_back_into_model_context(self):
        internal = self.root / ".codeclaw" / "sessions" / "test.json"
        internal.parent.mkdir(parents=True)
        internal.write_text("internal history", encoding="utf-8")
        source = self.root / "app.py"
        source.write_text("print('hello')", encoding="utf-8")
        collector = AttachmentCollector()
        attachment = collector._build_file_change_delta(str(self.root), [str(internal), str(source)])
        self.assertNotIn(".codeclaw", attachment["content"])
        self.assertIn("app.py", attachment["content"])

    async def test_capability_import_toggle_and_disabled_read(self):
        directory = self.root / "external" / "audit"
        directory.mkdir(parents=True)
        (directory / "SKILL.md").write_text("---\nname: audit\ndescription: Audit the project\n---\nCheck tests", encoding="utf-8")
        snapshot = await self.bridge.dispatch("capabilities.import", {"kind": "skills", "path": str(directory)})
        skill = snapshot["skills"][0]
        await self.bridge.dispatch("capabilities.toggle", {"id": skill["id"], "enabled": False})
        with self.assertRaises(ValueError):
            await self.bridge.dispatch("skills.read", {"name": "audit"})
        restored = DesktopBridge(self.events.append, self.root)
        self.assertFalse(restored.skills.discover()[0]["enabled"])

    async def test_plan_cannot_transition_until_approved(self):
        manager = PlanManager("desktop-plan-test")
        tool = ExitPlanModeTool()
        manager.set_mode("plan")
        waiting = asyncio.Event()
        async def approve(content):
            self.assertEqual(manager.get_mode(), "plan")
            await waiting.wait()
            return "deny"
        tool.context = {"plan_manager": manager, "plan_approval_callback": approve}
        with patch.object(manager, "get_plan", return_value="A concrete plan"):
            task = asyncio.create_task(tool.execute())
            await asyncio.sleep(0)
            self.assertFalse(task.done())
            waiting.set()
            result = await task
            self.assertIn("not approved", result)
            self.assertEqual(manager.get_mode(), "plan")
            tool.context["plan_approval_callback"] = lambda content: "allow"
            await tool.execute()
            self.assertEqual(manager.get_mode(), "normal")


if __name__ == "__main__":
    unittest.main()
