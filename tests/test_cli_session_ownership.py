"""Actual CLI process clears a resumed conversation, then continues in a fresh one."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from xuloom.core.session import SessionManager


class CliSessionOwnershipTests(unittest.TestCase):
    def test_clear_then_continue_cannot_restore_the_deleted_conversation(self):
        requests = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self.send_response(200)
                if not body.get("stream"):
                    self.send_header("Content-Type", "application/json"); self.end_headers()
                    self.wfile.write(json.dumps({"choices": [{"message": {"role": "assistant", "content": "fixture memory"}}]}).encode())
                    return
                requests.append(body)
                self.send_header("Content-Type", "text/event-stream"); self.end_headers()
                events = [{"choices": [{"index": 0, "delta": {"role": "assistant", "content": "CLEARED_RESPONSE"}, "finish_reason": None}]},
                    {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}]
                self.wfile.write(("".join("data: " + json.dumps(event) + "\n\n" for event in events) + "data: [DONE]\n\n").encode())
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        try:
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                env = {**os.environ, "HOME": str(root / "home"), "USERPROFILE": str(root / "home"),
                    "XULOOM_DATA_DIR": str(root / "data"), "XULOOM_CONFIG_DIR": str(root / "config"),
                    "XULOOM_MODEL_PROVIDER": "openai", "OPENAI_API_KEY": "fixture", "OPENAI_MODEL": "fixture",
                    "OPENAI_BASE_URL": "http://127.0.0.1:{}/v1".format(server.server_port), "PYTHONIOENCODING": "utf-8",
                    "PYTHONPATH": str(Path(__file__).resolve().parents[1]), "XULOOM_SKILLS_PATHS": ""}
                with patch.dict(os.environ, env):
                    with SessionManager(cwd=root) as manager:
                        self.assertTrue(manager.save_session_state("deleted-session", [{"role": "user", "content": "OLD_PRIVATE_CONVERSATION"}]))
                process = subprocess.run([sys.executable, str(Path(__file__).with_name("fixtures") / "cli_owner.py"), "--resume", "deleted-session"],
                    input="/clear sessions\ny\nHello after clearing\n/quit\n", capture_output=True, text=True, encoding="utf-8", env=env, cwd=root, timeout=50)
                self.assertEqual(process.returncode, 0, process.stderr + process.stdout[-2000:])
                self.assertIn("CLEARED_RESPONSE", process.stdout)
                self.assertEqual(len(requests), 1)
                self.assertNotIn("OLD_PRIVATE_CONVERSATION", json.dumps(requests))
                with patch.dict(os.environ, env):
                    with SessionManager(cwd=root) as manager:
                        identities = manager.list_session_ids()
                        self.assertEqual(len(identities), 1)
                        self.assertNotIn("deleted-session", identities)
        finally:
            server.shutdown(); server.server_close(); thread.join(timeout=5)
