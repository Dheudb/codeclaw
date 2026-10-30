import asyncio
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from xuloom.core.model_catalog import save_connection, select_model, public_config
from xuloom.core.model_inputs import configured_inputs, validate_media, filter_tool_media, UnsupportedInput
from xuloom.features.file_revisions import restore_revision, revision_root
from xuloom.features.incremental_write_queue import IncrementalWriteQueue
from xuloom.tools.file_write_tool import FileWriteTool
from xuloom.tools.file_edit_tool import FileEditTool
from xuloom.context.file_state_cache import FileStateCache


class InputRevisionTests(unittest.IsolatedAsyncioTestCase):
    def config(self):
        return save_connection({"model_connections": []}, {"connectionId": "one", "provider": "openai", "model": "opaque-text",
            "models": ["opaque-text", "opaque-vision", "opaque-unknown"], "baseUrl": "https://example.invalid/v1", "apiKey": "fixture",
            "modelInputs": {"opaque-text": ["text"], "opaque-vision": ["text", "image"]}})

    def test_input_contract_is_per_model_and_endpoint_never_guessed(self):
        cfg = select_model(self.config(), "one", "opaque-vision")
        with patch("xuloom.core.model_inputs.load_config", return_value=cfg):
            self.assertEqual(configured_inputs("openai", "opaque-text", "https://example.invalid/v1"), ["text"])
            self.assertIsNone(configured_inputs("openai", "opaque-unknown", "https://example.invalid/v1"))
            self.assertIsNone(configured_inputs("openai", "opaque-text", "https://other.invalid/v1"))
        self.assertNotIn("fixture", json.dumps(public_config(cfg)))
        self.assertEqual(public_config(cfg)["modelInputs"]["opaque-text"], ["text"])
        with self.assertRaises(ValueError):
            save_connection(cfg, {"connectionId": "one", "provider": "openai", "model": "opaque-text", "modelInputs": {"opaque-text": ["audio"]}})

    def test_current_history_nested_tool_and_fallback_media_contract(self):
        image = {"type": "image", "source": {"data": "fixture"}}
        messages = [{"role": "user", "content": [image]}]
        original = copy.deepcopy(messages)
        with self.assertRaises(UnsupportedInput): validate_media(messages, ["text"])
        validate_media(messages, None)
        validate_media(messages, ["text", "image"])
        with self.assertRaises(UnsupportedInput): validate_media(messages, ["text"])  # text-only fallback
        result = filter_tool_media([{ "type": "tool_result", "content": [{"type": "text", "text": "evidence"}, image]}], ["text"])
        self.assertIn("evidence", str(result))
        self.assertNotIn("'data'", str(result))
        self.assertIn("not sent or understood", str(result))
        self.assertEqual(messages, original)

    async def test_text_only_engine_rejects_before_network_and_without_losing_history(self):
        from xuloom.core.engine import QueryEngine
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"XULOOM_DATA_DIR": directory, "OPENAI_API_KEY": "fixture"}), patch("xuloom.core.model_inputs.load_config", return_value=self.config()):
            engine = QueryEngine(model_provider="openai", model="opaque-text", api_base_url="https://example.invalid/v1", cwd=directory)
            try:
                with patch.object(engine, "_receive_model_response", side_effect=AssertionError("network must not be used")):
                    events = [event async for event in engine.submit_message("look", input_blocks=[{"type": "image", "source": {"data": "fixture"}}])]
                self.assertEqual(events[-1]["result"]["reason"], "unsupported_input")
                self.assertFalse(engine.messages)
                events = [event async for event in engine.submit_message("look", input_blocks=[{"type": "document", "source": {"data": "fixture"}}])]
                self.assertEqual(events[-1]["result"]["reason"], "unsupported_input")
                self.assertIn("协议", events[-1]["text"])
            finally:
                await engine.aclose()

    async def test_file_tool_media_reaches_provider_as_pixels_not_python_repr(self):
        from xuloom.core.engine import QueryEngine
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"XULOOM_DATA_DIR": directory, "OPENAI_API_KEY": "fixture"}), patch("xuloom.core.model_inputs.load_config", return_value=self.config()):
            path = Path(directory) / "image.png"; Image.new("RGB", (2, 2), "blue").save(path)
            engine = QueryEngine(model_provider="openai", model="opaque-vision", api_base_url="https://example.invalid/v1", cwd=directory)
            try:
                result = await engine._execute_tool_wrapper({"id": "image-read", "name": "file_read_tool", "input": {"file_path": str(path)}}, lambda value: None)
                self.assertFalse(result["is_error"], result)
                self.assertIsInstance(result["content"], list)
                self.assertTrue(any(block.get("type") == "image" for block in result["content"]))
                rendered = engine._clean_roles_for_openai_api([{"role": "assistant", "content": [{"type": "tool_use", "id": "image-read", "name": "file_read_tool", "input": {"file_path": str(path)}}]}, {"role": "user", "content": [result]}], "system")
                self.assertTrue(any(block.get("type") == "image_url" for message in rendered if isinstance(message.get("content"), list) for block in message["content"]))
                engine.switch_model_runtime(model="opaque-text", persist=False)
                denied = await engine._execute_tool_wrapper({"id": "denied-read", "name": "file_read_tool", "input": {"file_path": str(path)}}, lambda value: None)
                self.assertTrue(denied["is_error"])
                self.assertNotIn("base64", str(denied["content"]))
            finally:
                await engine.aclose()

    async def test_mcp_visual_and_text_blocks_keep_native_types_and_complete_text(self):
        from types import SimpleNamespace
        from unittest.mock import AsyncMock
        from xuloom.protocols.mcp_bridge import MCPToolWrapper
        from xuloom.context.tool_results import native_tool_content
        text = "evidence " * 2000 + "END"
        session = SimpleNamespace(call_tool=AsyncMock(return_value=SimpleNamespace(content=[SimpleNamespace(type="text", text=text),
            SimpleNamespace(type="image", mimeType="image/png", data="fixture")], structuredContent={"proof": True}, isError=False)))
        tool = MCPToolWrapper("fixture", session, SimpleNamespace(name="visual", description="test", inputSchema={"type": "object"}))
        raw = await tool()
        self.assertEqual(raw["content"][1]["type"], "image")
        with tempfile.TemporaryDirectory() as directory:
            blocks = native_tool_content(raw, artifact_dir=directory)
            self.assertTrue(any(block.get("type") == "image" for block in blocks))
            self.assertTrue(any(path.read_text().endswith("END") for path in Path(directory).glob("*.txt")))
        denied = filter_tool_media(raw, ["text"])
        self.assertTrue(denied["is_error"])
        self.assertIn("evidence", str(denied))
        self.assertNotIn("'data'", str(denied))
        with tempfile.TemporaryDirectory() as directory, patch("xuloom.core.storage.atomic_text", side_effect=OSError("archive full")):
            errors = []
            blocks = native_tool_content(raw, artifact_dir=directory, errors=errors)
            self.assertEqual(errors, ["content_archive_failed"])
            self.assertIn("END", str(blocks))

    async def test_fallback_checks_its_own_inputs_before_opening_stream(self):
        from xuloom.core.engine import QueryEngine
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"XULOOM_DATA_DIR": directory, "OPENAI_API_KEY": "fixture"}), patch("xuloom.core.model_inputs.load_config", return_value=self.config()):
            engine = QueryEngine(model_provider="openai", model="opaque-vision", api_base_url="https://example.invalid/v1", cwd=directory)
            try:
                with self.assertRaises(UnsupportedInput):
                    await engine._receive_model_response(system_prompt="", api_messages=[{"role": "user", "content": [{"type": "image"}]}],
                        tools_schema=[], turn_count=1, model_name="opaque-text", max_tokens=100)
            finally: await engine.aclose()

    async def test_programmatic_undo_refuses_active_turn_and_isolation(self):
        from xuloom.core.engine import QueryEngine
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"XULOOM_DATA_DIR": directory, "OPENAI_API_KEY": "fixture"}):
            engine = QueryEngine(model_provider="openai", model="fixture", cwd=directory)
            try:
                engine.active_run_task = asyncio.current_task()
                with self.assertRaises(ValueError): await engine.undo_file("a" * 32)
                engine.active_run_task = None
                engine.execution_sandbox.mode = "workspace"
                with self.assertRaises(ValueError): await engine.undo_file("a" * 32)
                engine.execution_sandbox.mode = "host"
            finally: await engine.aclose()

    def context(self, directory):
        return {"session_cwd": directory, "session_id": "fixture-session", "file_state_cache": FileStateCache(),
            "incremental_write_queue": IncrementalWriteQueue(lock_directory=Path(directory) / "locks")}

    async def write(self, context, path, text):
        if path.exists(): context["file_state_cache"].record_read(str(path), kind="text")
        tool = FileWriteTool(); tool.context = context
        result = await tool.execute(file_path=str(path), content=text)
        self.assertTrue(result["ok"], result)
        return result["metadata"]["revision_id"]

    async def test_create_edit_exact_bytes_undo_survive_restart_and_retry(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"XULOOM_DATA_DIR": directory}):
            context = self.context(directory); path = Path(directory) / "file.txt"
            created = await self.write(context, path, "first")
            await restore_revision(self.context(directory), created)
            self.assertFalse(path.exists())
            path.write_bytes(b"original\r\nbytes\r\n")
            edited = await self.write(context, path, "replacement")
            self.assertTrue((revision_root(context) / edited / "before.bin").is_file())
            await restore_revision(self.context(directory), edited)
            self.assertEqual(path.read_bytes(), b"original\r\nbytes\r\n")
            again = await restore_revision(self.context(directory), edited)
            self.assertTrue(again["alreadyRestored"])

    async def test_undo_refuses_external_changes_and_tampered_backup(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"XULOOM_DATA_DIR": directory}):
            context = self.context(directory); path = Path(directory) / "file.txt"; path.write_text("original")
            revision = await self.write(context, path, "written")
            path.write_text("user's later edit")
            with self.assertRaises(ValueError): await restore_revision(context, revision)
            self.assertEqual(path.read_text(), "user's later edit")
            path.write_text("written")
            (revision_root(context) / revision / "before.bin").write_text("corrupt")
            with self.assertRaises(OSError): await restore_revision(context, revision)
            self.assertEqual(path.read_text(), "written")
            with self.assertRaises(ValueError): await restore_revision(context, "../escape")

    async def test_backup_failure_prevents_mutation_and_queue_is_released(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"XULOOM_DATA_DIR": directory}):
            context = self.context(directory); path = Path(directory) / "file.txt"; path.write_text("original")
            context["file_state_cache"].record_read(str(path), kind="text")
            tool = FileEditTool(); tool.context = context
            with patch("xuloom.features.file_revisions.atomic_copy", side_effect=OSError("no backup space")):
                result = await tool.execute(file_path=str(path), old_string="original", new_string="new")
            self.assertTrue(result["is_error"])
            self.assertEqual(path.read_text(), "original")
            revision = await self.write(context, path, "new")
            await restore_revision(context, revision)
            self.assertFalse(context["file_state_cache"].has_been_read(str(path)))

    def test_corrupt_config_is_never_silently_replaced(self):
        from xuloom.core.config import load_config, update_config
        with tempfile.TemporaryDirectory() as directory, patch("xuloom.core.config.CONFIG_FILE", Path(directory) / "config.json"):
            path = Path(directory) / "config.json"; path.write_text("broken{secret")
            with self.assertRaises(ValueError): load_config()
            with self.assertRaises(ValueError): update_config(lambda cfg: {"new": "data"})
            self.assertEqual(path.read_text(), "broken{secret")

    async def test_concurrent_config_processes_merge_unrelated_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            code = ("import sys,time;sys.path.insert(0,sys.argv[1]);from xuloom.core.config import update_config;"
                "update_config(lambda cfg:(time.sleep(.15) or dict(cfg,**{sys.argv[2]:True})))")
            env = {**os.environ, "XULOOM_CONFIG_DIR": directory}
            processes = [await asyncio.create_subprocess_exec(sys.executable, "-c", code, str(Path(__file__).resolve().parents[1]), key,
                env=env, stdout=subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE) for key in ("desktop", "cli")]
            for process in processes:
                _, errors = await process.communicate(); self.assertEqual(process.returncode, 0, errors)
            self.assertEqual(json.loads((Path(directory) / "config.json").read_text()), {"desktop": True, "cli": True})
