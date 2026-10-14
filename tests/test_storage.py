import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from codeclaw.core.session import SessionManager
from codeclaw.core.storage import session_directory, history_root


class StorageTests(unittest.TestCase):
    def test_cli_migration_preserves_metadata_and_artifact_references(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"CODECLAW_DATA_DIR": str(Path(temp) / "data")}):
            root = Path(temp) / "workspace"; root.mkdir()
            old = root / ".codeclaw"; (old / "sessions/subagents").mkdir(parents=True)
            proof = old / "tool-results/session/result.txt"; proof.parent.mkdir(parents=True); proof.write_text("large result", encoding="utf-8")
            (old / "sessions/session.json").write_text(json.dumps({"messages": [{"role": "user", "content": str(proof)}], "metadata": {"turn_traces": [{"id": "turn"}]}}), encoding="utf-8")
            (old / "sessions/subagents/child.json").write_text('{"session_id":"session"}', encoding="utf-8")
            (old / "session-index.json").write_text('{"version":1,"sessions":{"session":{"draft":"draft","archived":true}}}', encoding="utf-8")
            (old / "capabilities.json").write_text("{}", encoding="utf-8")
            manager = SessionManager(cwd=root); stored = manager.load_session_state("session")
            self.assertEqual(stored["metadata"]["turn_traces"], [{"id": "turn"}])
            self.assertIn(str(history_root(root) / "tool-results/session/result.txt"), stored["messages"][0]["content"])
            self.assertEqual(json.loads((history_root(root) / "index.json").read_text())["sessions"]["session"]["draft"], "draft")
            self.assertTrue((Path(manager.session_dir) / "subagents/child.json").exists())
            self.assertFalse((old / "sessions").exists()); self.assertTrue((old / "capabilities.json").exists())
            self.assertEqual(str(session_directory(root)), manager.session_dir)
            self.assertTrue(manager.save_session_state("session", [{"role": "assistant", "content": "continued"}]))
            self.assertFalse((old / "sessions").exists())

    def test_conflicting_history_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"CODECLAW_DATA_DIR": str(Path(temp) / "data")}):
            root = Path(temp) / "workspace"; root.mkdir(); manager = SessionManager(cwd=root)
            manager.save_session_state("same", [{"role": "user", "content": "central"}])
            legacy = root / ".codeclaw/sessions/same.json"; legacy.parent.mkdir(parents=True)
            legacy.write_text('{"messages":[{"role":"user","content":"legacy"}]}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "冲突"): SessionManager(cwd=root)
            self.assertTrue(legacy.exists()); self.assertEqual(manager.load_session_state("same")["messages"][0]["content"], "central")
