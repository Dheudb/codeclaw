import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from codeclaw.context.context import ContextBuilder
from codeclaw.core.engine import QueryEngine
from codeclaw.desktop.prompt_edit import replace_latest_turn


class PromptEditTests(unittest.TestCase):
    def test_edit_targets_latest_human_turn_and_preserves_files_and_recovery_copy(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"CODECLAW_DATA_DIR": directory}):
            file = Path(directory) / "work.md"
            file.write_text("already written", encoding="utf-8")
            messages = [{"role": "user", "_uuid": "u1", "content": "earlier"},
                        {"role": "assistant", "content": "previous answer"},
                        {"role": "user", "_uuid": "u2", "content": "edit me"},
                        {"role": "assistant", "content": [{"type": "tool_use", "id": "write"}]},
                        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "write", "content": "written"}]}]
            engine = SimpleNamespace(messages=messages, turn_traces=[{"userMessageId": "u1"}, {"userMessageId": "u2"}],
                                     session_cwd=directory, session_id="s1", _sync_message_context=Mock(),
                                     persist_session_state=Mock(), session_manager=SimpleNamespace(last_save_ok=True))
            for invalid in ["u1", "", "../invalid"]:
                with self.assertRaises(ValueError): replace_latest_turn(engine, invalid)
            self.assertIs(engine.messages, messages)
            replace_latest_turn(engine, "u2")
            self.assertEqual(engine.messages, messages[:2])
            self.assertEqual(engine.turn_traces, [{"userMessageId": "u1"}])
            backup = next(Path(directory).rglob("revisions/*.json"))
            self.assertEqual(json.loads(backup.read_text(encoding="utf-8"))["messages"], messages)
            self.assertEqual(file.read_text(encoding="utf-8"), "already written")
            engine.session_manager.last_save_ok = False
            with self.assertRaises(OSError): replace_latest_turn(engine, "u1")
            self.assertEqual(engine.messages, messages[:2])

    def test_model_metadata_is_not_a_claim_of_verified_backend_identity(self):
        builder = ContextBuilder()
        with patch.object(builder, "_load_project_instructions", return_value=""), patch.object(builder, "get_git_state", return_value={}):
            prompt = builder.generate_system_prompt(model_name="gateway-model-alias")
        self.assertIn("Configured request model ID: gateway-model-alias", prompt)
        self.assertIn("does not verify the actual backend model", prompt)
        self.assertNotIn("Claude", prompt)
        self.assertNotIn("CLAUDE", prompt)

    def test_native_anthropic_cache_preserves_exact_model_and_role_addons(self):
        engine = QueryEngine.__new__(QueryEngine)
        engine.model_provider = "anthropic"
        engine.api_base_url = "https://api.anthropic.com"
        engine.context_builder = ContextBuilder()
        prompt = "static instructions\n\n# Environment\nConfigured request model ID: fallback-id\n\n# Assigned role\nROLE_PROOF"
        blocks = engine._build_cached_system_blocks(prompt)
        self.assertEqual("\n\n".join(block["text"] for block in blocks), prompt)
        self.assertIn("cache_control", blocks[0])
        engine.model_provider = "openai"
        self.assertEqual(engine._build_cached_system_blocks(prompt), prompt)
