import asyncio
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from codeclaw.core.engine import QueryEngine
from codeclaw.memory.memory import MemoryCompactor


class CrashRecoveryTests(unittest.TestCase):
    def test_execution_checkpoints_survive_process_exit_without_replaying_effects(self):
        repository = Path(__file__).resolve().parents[1]
        for phase in ("before_execution", "after_effect", "after_result"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                env = {**os.environ, "CODECLAW_DATA_DIR": str(root / "data"), "HOME": str(root / "home"),
                       "USERPROFILE": str(root / "home"), "OPENAI_API_KEY": "fixture", "PYTHONPATH": str(repository)}
                process = subprocess.run([sys.executable, str(repository / "tests/fixtures/crash_tool.py"), str(root), phase],
                    cwd=repository, env=env, timeout=45, capture_output=True, text=True)
                self.assertEqual(process.returncode, 87, process.stderr)
                effects = root / "effects.txt"
                before = effects.read_text(encoding="utf-8") if effects.exists() else ""
                self.assertEqual(before.count("one-effect"), 0 if phase == "before_execution" else 1)
                async def recover():
                    with patch.dict(os.environ, env), patch.object(MemoryCompactor, "_resolve_token_encoder", return_value=None):
                        engine = QueryEngine(model="fixture", model_provider="openai", cwd=temp)
                        initial_clients = [engine.client, engine.compactor.client]
                        try:
                            self.assertTrue(engine.load_session((root / "session-id.txt").read_text(encoding="utf-8")))
                            result = await engine._execute_tool_wrapper({"id": "crash-call", "name": "effect_tool", "input": {"value": "one-effect"}}, lambda _: None)
                            self.assertEqual(result["is_error"], phase != "after_result")
                            if phase == "after_result":
                                self.assertEqual(result["content"], "EFFECT_SAVED")
                            else:
                                self.assertIn("uncertain", result["content"])
                            self.assertTrue(any(isinstance(b, dict) and b.get("type") == "tool_result" for m in engine.messages for b in m.get("content", [])))
                        finally:
                            for client in initial_clients + [engine.client, engine.compactor.client]:
                                if client:
                                    await client.close()
                asyncio.run(recover())
                after = effects.read_text(encoding="utf-8") if effects.exists() else ""
                self.assertEqual(after, before)
