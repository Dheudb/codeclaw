"""Controlled cross-domain acceptance. Model outputs are scripted, never claimed as live-model evidence."""
import asyncio
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import threading
import time
import unittest
from unittest.mock import patch

from xuloom.core.engine import QueryEngine
from xuloom.core.messages import create_assistant_message, create_user_message
from xuloom.core.run_result import RunResult
import test_tool_recovery as recovery

records = []


class DomainAcceptanceTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = recovery.RecoveryTests.asyncSetUp
    asyncTearDown = recovery.RecoveryTests.asyncTearDown
    fixture = recovery.RecoveryTests.fixture

    async def evaluate(self, domain, prompt, steps, check, artifacts=(), selected_skills=None):
        started, accepted, result = time.monotonic(), False, None
        self.fixture(steps)
        self.engine.permission_manager.approval_policy.set_mode("full")
        try:
            result = await self.engine.run(prompt, selected_skills=selected_skills, sys_print_callback=lambda _: None)
            self.assertTrue(result.ok, result.to_dict())
            check()
            accepted = True
        finally:
            activity = self.engine.recent_tool_activity
            mutations = Counter((row["tool_name"], row.get("input_digest")) for row in activity
                if row["tool_name"] in {"command_tool", "file_write_tool", "file_edit_tool", "agent_tool"} and row["status"] == "ok")
            records.append({"domain": domain, "test_kind": "deterministic_model_standin_with_real_tools",
                "domain_accepted": accepted, "framework_completed": bool(result and result.ok),
                "elapsed_seconds": round(time.monotonic() - started, 4), "model_request_count": len(self.requests),
                "mechanical_errors_observed": sum(row["status"] != "ok" for row in activity),
                "duplicate_mutations_observed": sum(max(0, count - 1) for count in mutations.values()),
                "missing_artifacts": sum(not (self.root / name).is_file() for name in artifacts),
                "worker_count": len(self.engine.subagent_registry), "api_retry_count": None, "token_usage": None, "cost": None,
                "unknown_metrics_reason": "Scripted model streams provide no provider retry, usage or price telemetry"})

    @staticmethod
    def call(tool, **arguments):
        return tool, json.dumps(arguments)

    def document(self, name, content):
        return self.call("file_write_tool", file_path=str(self.root / name), content=content)

    async def test_conversation(self):
        def check():
            self.assertEqual(len(self.requests), 1)
            self.assertEqual(self.engine.recent_tool_activity, [])
            self.assertEqual(self.engine.subagent_registry, [])
        await self.evaluate("conversation", "Explain the supplied sentence without tools", ["DIRECT_ANSWER"], check)

    async def test_writing(self):
        names = ["summary.md", "letter.md", "outline.md"]
        def check():
            for name in names:
                self.assertEqual((self.root / name).read_text(encoding="utf-8"), "A concise draft.\n")
            self.assertEqual(self.engine.subagent_registry, [])
            self.assertEqual(len(self.engine.recent_tool_activity), 3)
        await self.evaluate("writing", "Save three concise drafts; do not execute software checks",
            [self.document(name, "A concise draft.\n") for name in names] + ["DRAFTS_SAVED"], check, names)

    async def test_data_analysis(self):
        original = "value,unit\n1,kg\n2,kg\n,kg\n3,kg\n"
        (self.root / "input.csv").write_text(original, encoding="utf-8")
        code = "import csv,json; from pathlib import Path; rows=list(csv.DictReader(Path('input.csv').open())); values=[float(r['value']) for r in rows if r['value']]; Path('summary.json').write_text(json.dumps({'count':len(values),'mean':sum(values)/len(values),'unit':'kg'})); Path('effects.txt').open('a').write('calculated\\n')"
        def check():
            self.assertEqual(json.loads((self.root / "summary.json").read_text()), {"count": 3, "mean": 2, "unit": "kg"})
            self.assertEqual((self.root / "input.csv").read_text(), original)
            self.assertEqual((self.root / "effects.txt").read_text(), "calculated\n")
        await self.evaluate("data_analysis", "Calculate the mean in kg excluding missing values; retain the input",
            [self.call("command_tool", action="start", argv=[sys.executable, "-c", code], wait_ms=10000), "DATA_SAVED"], check, ["summary.json", "effects.txt"])

    async def test_programming(self):
        source = "def mean(values):\n    if not values: raise ValueError('empty')\n    return sum(values) / len(values)\n"
        tests = "import unittest\nfrom calculation import mean\nclass Contract(unittest.TestCase):\n    def test_values(self): self.assertEqual(mean([-2, 4]), 1)\n    def test_empty(self):\n        with self.assertRaises(ValueError): mean([])\n"
        def check():
            self.assertEqual((self.root / "calculation.py").read_text(), source)
            command = next(iter(self.engine.tool_context["command_sessions"].records.values()))
            self.assertEqual((command["status"], command["exit_code"]), ("completed", 0))
            self.assertEqual(self.engine.subagent_registry, [])
        await self.evaluate("programming", "Implement the mean function with an empty-input error and run the contract tests",
            [self.document("calculation.py", source), self.document("test_calculation.py", tests),
             self.call("command_tool", action="start", argv=[sys.executable, "-m", "unittest", "test_calculation", "-v"], wait_ms=10000), "CODE_VERIFIED"],
            check, ["calculation.py", "test_calculation.py"], ["code-development"])

    async def test_literature_research(self):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_GET(self):
                self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.end_headers()
                self.wfile.write(b"<html><body><h1>Fixture study</h1><p>Observed value: 42. This is a controlled test source, not a scientific claim.</p></body></html>")
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        url = "http://127.0.0.1:{}/study".format(server.server_port)
        def check():
            report = (self.root / "literature.md").read_text()
            self.assertIn(url, report)
            self.assertIn("42", report)
            self.assertTrue(any("Observed value: 42" in row["tool_result"]["content"] for row in self.engine.recent_tool_activity))
        try:
            with patch.dict(os.environ, {"NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"}), patch("xuloom.tools.web_tool.DDGS") as search:
                search.return_value.text.return_value = [{"title": "Fixture study", "href": url, "body": "Controlled evidence"}]
                await self.evaluate("literature_research", "Find and read the fixture study; retain its citation and distinguish the test from scientific evidence",
                    [self.call("web_search_tool", query="fixture study"), self.call("web_fetch_tool", url=url),
                     self.document("literature.md", "Observed value: 42. Controlled test source, not scientific evidence. [Source](" + url + ")\n"), "RESEARCH_SAVED"], check, ["literature.md"])
        finally:
            server.shutdown(); server.server_close(); thread.join(timeout=5)

    async def test_multiagent(self):
        original_run = QueryEngine.run
        entered, gate = [], asyncio.Event()
        async def run(engine, user_input, sys_print_callback=print, **kwargs):
            if not engine.agent_depth:
                return await original_run(engine, user_input, sys_print_callback=sys_print_callback, **kwargs)
            entered.append(engine.agent_id)
            if len(entered) == 3: gate.set()
            await gate.wait()
            engine.messages.extend([create_user_message(user_input), create_assistant_message("Hello from " + engine.agent_id)])
            return RunResult("Hello from " + engine.agent_id, "completed", True)
        def check():
            self.assertEqual(len(set(entered)), 3)
            self.assertTrue(all(row["status"] == "completed" for row in self.engine.subagent_registry))
            for identity in entered:
                self.assertTrue(Path(self.engine.session_manager.get_subagent_transcript_path(identity)).is_file())
        with patch.object(QueryEngine, "run", run):
            await self.evaluate("multiagent", "Start three parallel greeting workers and summarize the actual results",
                [self.call("agent_tool", prompt="Hello", count=3, request_id="acceptance-workers"), "THREE_ACTUAL_RESULTS"], check)
