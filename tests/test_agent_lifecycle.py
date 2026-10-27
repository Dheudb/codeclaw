from codeclaw.core.run_result import RunResult
import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from codeclaw.agents.registry import AgentRegistry
from codeclaw.core.engine import QueryEngine
from codeclaw.core.messages import create_user_message, create_assistant_message
from codeclaw.memory.memory import MemoryCompactor
from codeclaw.protocols.mcp_config import validate_server


class LifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.env = patch.dict(os.environ, {"CODECLAW_DATA_DIR": str(self.root / "data"), "HOME": str(self.root / "home"),
            "USERPROFILE": str(self.root / "home"), "OPENAI_API_KEY": "fixture", "CODECLAW_MODEL_PROVIDER": "openai"})
        self.env.start()
        self.tokens = patch.object(MemoryCompactor, "_resolve_token_encoder", return_value=None)
        self.tokens.start()
        self.engines = []
        self.engine = self.make_engine()
        self.tool = self.engine.available_tools["agent_tool"]

    def make_engine(self):
        engine = QueryEngine(model="fixture-model", model_provider="openai", cwd=str(self.root))
        self.engines.append(engine)
        return engine

    async def asyncTearDown(self):
        for engine in self.engines:
            await engine.tool_context["subagent_tasks"].close()
            await engine.tool_context["command_sessions"].close()
            await engine.client.close()
            await engine.compactor.client.close()
            engine.session_manager.release_writer()
        self.tokens.stop()
        self.env.stop()
        self.temp.cleanup()

    async def test_same_id_followup_after_restart_preserves_history_and_frozen_role(self):
        calls = []
        async def run(child, user_input, sys_print_callback):
            calls.append((child.agent_id, json.dumps(child.messages), child.tool_context["agent_system_prompt"]))
            child.messages.append(create_user_message(user_input))
            child.messages.append(create_assistant_message([{ "type": "text", "text": "answer:" + user_input}]))
            child.persist_session_state()
            return RunResult("answer:" + user_input, "completed", True)
        with patch.object(QueryEngine, "run", run):
            receipt = json.loads(await self.tool.execute(prompt="REMEMBER_FIRST", response_format="json"))
            identity = receipt["agents"][0]["agent_id"]
            self.assertEqual(json.loads(await self.tool.execute(action="result", agent_id=identity))["status"], "completed")
            sent = json.loads(await self.tool.execute(action="send", agent_id=identity, prompt="queued note"))
            self.assertTrue(sent["requires_followup"])
            self.engine.persist_session_state()
            self.engine.session_manager.release_writer()
            restored = self.make_engine()
            self.assertTrue(restored.load_session(self.engine.session_id))
            result = json.loads(await restored.available_tools["agent_tool"].execute(action="followup", agent_id=identity, prompt="SECOND"))
            self.assertEqual(result["agent_id"], identity)
            self.assertEqual(result["turn"], 2)
            self.assertIn("REMEMBER_FIRST", calls[1][1])
            self.assertEqual(calls[0][2], calls[1][2])
            self.assertEqual(len(restored.subagent_registry), 1)
            restored.tool_context["api_base_url"] = "https://different-provider.invalid/v1"
            with self.assertRaisesRegex(ValueError, "original provider"):
                await restored.available_tools["agent_tool"].execute(action="followup", agent_id=identity, prompt="wrong provider")
            restored.tool_context["api_base_url"] = self.engine.api_base_url
            await restored.available_tools["agent_tool"].execute(action="close", agent_id=identity)
            with self.assertRaises(ValueError):
                await restored.available_tools["agent_tool"].execute(action="followup", agent_id=identity, prompt="late")
            with self.assertRaises(ValueError):
                await self.make_engine().available_tools["agent_tool"].execute(action="result", agent_id=identity)

    async def test_structured_inventory_bounded_wait_and_capacity_during_execution(self):
        gate = asyncio.Event()
        async def run(child, user_input, sys_print_callback):
            await gate.wait()
            return RunResult("done", "completed", True)
        with patch.object(QueryEngine, "run", run):
            receipt = json.loads(await self.tool.execute(prompt="hold", run_in_background=True, count=2, response_format="json"))
            ids = [r["agent_id"] for r in receipt["agents"]]
            await asyncio.sleep(0)
            capacity = json.loads(await self.tool.execute(action="capacity"))
            self.assertEqual((capacity["active"], capacity["available"]), (2, 8))
            rows = json.loads(await self.tool.execute(action="list"))["agents"]
            self.assertTrue(all(row["executing"] for row in rows))
            result = json.loads(await self.tool.execute(action="wait", agent_ids=ids, timeout_ms=5, response_format="json"))
            self.assertTrue(result["timed_out"])
            with self.assertRaises(ValueError):
                await self.tool.execute(action="followup", agent_id=ids[0], prompt="race")
            with self.assertRaises(ValueError):
                await self.tool.execute(action="close", agent_id=ids[0])
            gate.set()
            await self.tool.execute(action="wait")
            self.assertEqual(json.loads(await self.tool.execute(action="capacity"))["active"], 0)

    async def test_mailbox_delivery_uses_turn_boundary_and_durable_ack(self):
        async def run(child, user_input, sys_print_callback):
            child.messages.append(create_user_message(user_input))
            mailbox = child.tool_context["delegated_mailbox"]
            ready.set()
            await gate.wait()
            messages = mailbox()
            self.assertEqual([m["text"] for m in messages], ["NOTE"])
            child.messages.extend(create_user_message(m["text"]) for m in messages)
            self.assertTrue(child.persist_session_state())
            child.tool_context["ack_delegated_mailbox"]({m["id"] for m in messages})
            return RunResult("done", "completed", True)
        ready, gate = asyncio.Event(), asyncio.Event()
        with patch.object(QueryEngine, "run", run):
            receipt = json.loads(await self.tool.execute(prompt="work", response_format="json", run_in_background=True))
            identity = receipt["agents"][0]["agent_id"]
            await ready.wait()
            sent = json.loads(await self.tool.execute(action="send", agent_id=identity, prompt="NOTE"))
            self.assertFalse(sent["requires_followup"])
            self.assertEqual(json.loads(await self.tool.execute(action="result", agent_id=identity))["pending_message_count"], 1)
            gate.set()
            await self.tool.execute(action="wait")
            self.assertEqual(json.loads(await self.tool.execute(action="result", agent_id=identity))["pending_message_count"], 0)

    async def test_role_tool_and_mcp_restrictions_survive_latent_activation(self):
        directory = self.root / ".codeclaw/agents"
        directory.mkdir(parents=True)
        (directory / "limited.md").write_text("---\nname: limited\ndescription: limited worker\nweb_search: disabled\nmcp_servers: []\nread_only: true\n---\nOnly inspect.", encoding="utf-8")
        mcp = SimpleNamespace(server_name="papers", name="papers_search", context={}, is_read_only=False)
        self.engine.available_tools[mcp.name] = mcp
        latent_mcp = SimpleNamespace(server_name="papers", name="papers_metadata", context={}, is_read_only=True)
        self.engine.latent_tools[latent_mcp.name] = latent_mcp
        async def run(child, user_input, sys_print_callback):
            for denied in ["command_tool", "bash_tool", "file_write_tool", "file_edit_tool", "web_search_tool", "web_fetch_tool", "papers_search", "papers_metadata"]:
                self.assertNotIn(denied, child.available_tools)
                self.assertNotIn(denied, child.latent_tools)
            self.assertNotIn("browser_tool", child.latent_tools)
            self.assertEqual(child.activate_tools(["browser_tool"]), [])
            return RunResult("done", "completed", True)
        with patch.object(QueryEngine, "run", run):
            receipt = json.loads(await self.tool.execute(prompt="inspect", subagent_type="limited", response_format="json"))
            self.assertEqual(receipt["agents"][0]["status"], "completed")

    async def test_explicit_role_loads_its_specialized_tool_without_activating_it_for_parent(self):
        directory = self.root / ".codeclaw/agents"
        directory.mkdir(parents=True)
        (directory / "calculator.md").write_text("---\nname: calculator\ndescription: Calculate the assigned expression\ntools: [repl_tool]\n---\nCalculate only the assigned expression.", encoding="utf-8")
        self.assertIn("repl_tool", self.engine.latent_tools)
        async def run(child, user_input, sys_print_callback):
            self.assertEqual(set(child.available_tools), {"repl_tool"})
            self.assertNotIn("repl_tool", child.latent_tools)
            return RunResult("4", "completed", True)
        with patch.object(QueryEngine, "run", run):
            receipt = json.loads(await self.tool.execute(prompt="2+2", subagent_type="calculator", response_format="json"))
        self.assertEqual(receipt["agents"][0]["result"], "4")
        self.assertNotIn("repl_tool", self.engine.available_tools)

    async def test_capacity_configuration_and_mcp_timeout_validation(self):
        self.engine.tool_context["skill_registry"].update(agent_concurrency=24)
        capacity = json.loads(await self.tool.execute(action="capacity"))
        self.assertEqual(capacity["limit"], 24)
        self.engine.tool_context["skill_registry"].update(agent_concurrency=True)
        with self.assertRaises(ValueError):
            await self.tool.execute(action="capacity")
        server = validate_server({"command": "python", "tool_timeout_sec": 90, "env": {"ss-api-key": "fixture"}})
        self.assertEqual(server["tool_timeout_sec"], 90)
        for timeout in [True, 0, 601, "90"]:
            with self.assertRaises(ValueError):
                validate_server({"command": "python", "tool_timeout_sec": timeout})

    async def test_configured_capacity_applies_to_real_parallel_admissions(self):
        self.engine.tool_context["skill_registry"].update(agent_concurrency=24)
        entered, gate = [], asyncio.Event()
        async def run(child, user_input, sys_print_callback):
            entered.append(child.agent_id)
            await gate.wait()
            return RunResult("done", "completed", True)
        with patch.object(QueryEngine, "run", run):
            for _ in range(2):
                await self.tool.execute(prompt="hold", count=10, run_in_background=True)
            await asyncio.sleep(0)
            self.assertEqual(len(entered), 20)
            with self.assertRaises(ValueError):
                await self.tool.execute(prompt="overflow", count=5, run_in_background=True)
            self.assertEqual(len(self.engine.subagent_registry), 20)
            gate.set()
            await self.tool.execute(action="wait")

    async def test_idle_worker_retains_owned_command_until_explicit_close(self):
        commands = []
        async def run(child, user_input, sys_print_callback):
            manager = child.tool_context["command_sessions"]
            identity = await manager.start(argv=[sys.executable, "-c", "import time; time.sleep(30)"],
                cwd=str(self.root), owner=child.agent_id)
            commands.append(identity)
            return RunResult("command launched", "completed", True)
        with patch.object(QueryEngine, "run", run):
            result = json.loads(await self.tool.execute(prompt="start command", response_format="json"))
            identity = result["agents"][0]["agent_id"]
            snapshot = json.loads(await self.tool.execute(action="result", agent_id=identity))
            self.assertFalse(snapshot["executing"])
            self.assertEqual(snapshot["running_commands"], 1)
            await self.tool.execute(action="close", agent_id=identity)
            self.assertEqual(self.engine.tool_context["command_sessions"].records[commands[0]]["status"], "cancelled")


if __name__ == "__main__":
    unittest.main()
