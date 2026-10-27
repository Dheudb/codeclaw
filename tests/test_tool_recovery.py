"""Exercise recovery through the real engine loop with deterministic model streams."""
import json
import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from pydantic import BaseModel, field_validator
from codeclaw.core.engine import QueryEngine
from codeclaw.desktop.turns import TurnRecorder, conversation_history
from codeclaw.memory.memory import MemoryCompactor
from codeclaw.protocols.dsml import ToolProtocolError, parse_tool_arguments
from codeclaw.tools.base import BaseAgenticTool
from codeclaw.tools.todo_tool import TodoWriteTool


class RecordInput(BaseModel):
    value: str = "DANGEROUS_DEFAULT"

    @field_validator("value")
    @classmethod
    def rule(cls, value):
        if value == "RULE_FAILURE":
            raise ValueError("fixture rule rejected input")
        return value


class RecordTool(BaseAgenticTool):
    name = "record_tool"
    description = "Record a fixture operation."
    input_schema = RecordInput
    is_read_only = True

    def __init__(self):
        self.calls = []

    async def execute(self, value="DANGEROUS_DEFAULT"):
        self.calls.append(value)
        return "saved:" + value


def dsml(key="value", value="after"):
    return ('<｜DSML｜tool_calls><｜DSML｜invoke name="record_tool">'
            f'<｜DSML｜parameter name="{key}" string="true">{value}</｜DSML｜parameter>'
            '</｜DSML｜invoke></｜DSML｜tool_calls>')


class RecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.env = patch.dict(os.environ, {"CODECLAW_DATA_DIR": str(self.root / "data"),
            "USERPROFILE": str(self.root / "home"), "HOME": str(self.root / "home"),
            "OPENAI_API_KEY": "fixture", "CODECLAW_MODEL_PROVIDER": "openai"})
        self.env.start()
        self.tokens = patch.object(MemoryCompactor, "_resolve_token_encoder", return_value=None)
        self.tokens.start()
        self.engine = QueryEngine(model="fixture", model_provider="openai", cwd=str(self.root))
        self.tool = RecordTool()
        self.engine.available_tools["record_tool"] = self.tool
        self.engine._sync_message_context()
        self.engine.session_memory_manager.maybe_extract = AsyncMock()
        self.engine.memory_extractor.maybe_extract = AsyncMock()
        self.requests = []
        self.events = []
        self.recorder = TurnRecorder()

    async def asyncTearDown(self):
        await self.engine.tool_context["subagent_tasks"].close()
        await self.engine.tool_context["command_sessions"].close()
        await self.engine.client.close()
        await self.engine.compactor.client.close()
        self.engine.session_manager.release_writer()
        self.tokens.stop()
        self.env.stop()
        self.temp.cleanup()

    def fixture(self, completions):
        async def create(**request):
            self.requests.append(request)
            completion = completions[len(self.requests) - 1]
            async def stream():
                if isinstance(completion, tuple):
                    name, arguments = completion
                    delta = NS(content=None, tool_calls=[NS(index=0, id="call-" + str(len(self.requests)),
                        function=NS(name=name, arguments=arguments))])
                    finish = "tool_calls"
                else:
                    delta = NS(content=completion, tool_calls=[])
                    finish = "stop"
                yield NS(choices=[NS(delta=delta, finish_reason=finish)], usage=NS(prompt_tokens=10, completion_tokens=5))
            return stream()
        self.engine.client.chat.completions.create = create

    async def run_task(self):
        async def emit(event):
            self.events.append(event)
            self.recorder.accept(event)
        result = (await self.engine._run_loop("Use the selected fixture skill.", sys_print_callback=lambda _: None, event_callback=emit)).text
        return result

    async def test_repair_retains_prior_work_and_feedback_and_duration_trace(self):
        self.fixture([("record_tool", '{"value":"before"}'), "Continuing.\n" + dsml("unknown"), dsml(), "DONE"])
        result = await self.run_task()
        self.assertEqual(result, "DONE")
        self.assertEqual(self.tool.calls, ["before", "after"])
        self.assertEqual(len(self.requests), 4)
        repair_request = json.dumps(self.requests[2]["messages"])
        self.assertIn("Unknown parameter unknown", repair_request)
        self.assertIn("do not repeat", repair_request.lower())
        self.assertIn("saved:before", repair_request)
        event = next(e for e in self.events if e["type"] == "tool_protocol_error")
        self.assertTrue(event["retrying"])
        self.assertEqual(json.loads(Path(event["diagnostic"]).read_text(encoding="utf-8"))["raw_completion"], "Continuing.\n" + dsml("unknown"))
        self.assertTrue(any(step.get("protocolError") for step in self.recorder.record["steps"]))
        self.assertEqual(self.engine.session_token_usage["output_tokens"], 20)
        self.assertNotIn("tool_protocol", str(conversation_history(self.engine.messages)))

    async def test_malformed_batch_is_atomic_and_repair_budget_stops(self):
        bad = dsml().replace('</｜DSML｜tool_calls>', dsml("unknown").replace('<｜DSML｜tool_calls>', ''))
        self.fixture([bad, bad, bad])
        result = await self.run_task()
        self.assertIn("多次修正失败", result)
        self.assertNotIn("尚未执行任何操作", result)
        self.assertEqual(self.tool.calls, [])
        self.assertEqual(len(self.requests), 3)
        self.assertEqual([e["retrying"] for e in self.events if e["type"] == "tool_protocol_error"], [True, True, False])
        saved = self.engine.session_manager.load_session_state(self.engine.session_id)
        self.assertTrue(any(m.get("_origin") == "tool_protocol" for m in saved["messages"]))

    async def test_corrupt_native_arguments_never_execute_default_and_can_repair(self):
        self.fixture([("record_tool", '{"value":'), ("record_tool", '{"value":"corrected"}'), "DONE"])
        self.assertEqual(await self.run_task(), "DONE")
        self.assertEqual(self.tool.calls, ["corrected"])
        tool_result = next(m for m in self.requests[1]["messages"] if m["role"] == "tool")
        self.assertIn("executed: False", tool_result["content"])
        block = next(b for m in self.engine.messages if m["role"] == "assistant" for b in m["content"] if isinstance(b, dict) and b.get("type") == "tool_use")
        self.assertIn("_argument_error", block)
        # VCR/history replay must retain the rejection rather than executing {}.
        await self.engine._execute_tool_wrapper(block, lambda _: None)
        self.assertEqual(self.tool.calls, ["corrected"])

    async def test_native_schema_errors_have_a_bounded_repair_budget(self):
        bad = ("record_tool", '{"value":123}')
        self.fixture([bad, bad, bad])
        self.assertIn("多次修正失败", await self.run_task())
        self.assertEqual(len(self.requests), 3)
        self.assertEqual(self.tool.calls, [])
        self.assertIn("string_type", json.dumps(self.requests[1]["messages"]))
        self.assertEqual([e["retrying"] for e in self.events if e["type"] == "tool_protocol_error"], [True, True, False])

    async def test_abort_during_repair_does_not_issue_another_request(self):
        self.fixture([dsml("unknown")])
        async def emit(event):
            if event["type"] == "tool_protocol_error":
                self.engine.abort_event.set()
        result = (await self.engine._run_loop("fixture", sys_print_callback=lambda _: None, event_callback=emit)).text
        self.assertIn("Turn aborted", result)
        self.assertEqual(len(self.requests), 1)

    async def test_long_skill_instructions_are_not_replaced_by_a_preview(self):
        root = self.root / '.codeclaw/skills/fixture'
        root.mkdir(parents=True)
        root.joinpath('SKILL.md').write_text('---\nname: fixture\ndescription: Fixture.\n---\n' + 'instruction\n' * 1000 + 'END_OF_SKILL', encoding='utf-8')
        result = await self.engine._execute_tool_wrapper({"id": "skill", "name": "skill_tool", "input": {"action": "read", "skill_name": "fixture"}}, lambda _: None)
        self.assertIn('END_OF_SKILL', result['content'])
        self.assertNotIn('content_spilled', result['content'])

    def workflow_skill(self):
        self.engine.permission_manager.always_allow_tools.add('command_tool')
        root = self.root / '.codeclaw/skills/fixture-workflow'
        root.mkdir(parents=True)
        root.joinpath('SKILL.md').write_text('---\nname: fixture-workflow\ndescription: Fixture.\nmetadata:\n  codeclaw:\n    completion-protocol: workflow-json-v1\n---\nFollow the runtime.', encoding='utf-8')

    def workflow_command(self, finished):
        payload = {'run_handle': 'fixture:one', 'workflow_finished': finished, 'required_main_action': {'action': 'read-cards'}}
        return ('command_tool', json.dumps({'action': 'start', 'argv': [sys.executable, '-c', 'print(' + repr(json.dumps(payload)) + ')'], 'wait_ms': 1000, 'max_bytes': 4}))

    async def test_unfinished_selected_skill_cannot_end_with_a_promise(self):
        self.workflow_skill()
        self.fixture([self.workflow_command(False), 'I will read the cards next.', self.workflow_command(True), 'ACTUAL_DONE'])
        result = (await self.engine._run_loop('Run the workflow.', selected_skills=['fixture-workflow'], sys_print_callback=lambda _: None,
                                             event_callback=self._workflow_event)).text
        self.assertEqual(result, 'ACTUAL_DONE')
        self.assertEqual(len(self.requests), 4)
        self.assertIn('Skill workflow completion guard', json.dumps(self.requests[2]['messages']))
        self.assertEqual(len(self.engine.tool_context['command_sessions'].records), 2)
        self.assertEqual(len([e for e in self.events if e['type'] == 'workflow_incomplete']), 1)
        self.assertIn('技能流程尚未完成', str(self.recorder.record['steps']))

    async def _workflow_event(self, event):
        self.events.append(event)
        self.recorder.accept(event)

    async def test_workflow_guard_stops_bounded_and_next_chat_is_unaffected(self):
        self.workflow_skill()
        self.fixture([('skill_tool', '{"action":"read","skill_name":"fixture-workflow"}'), self.workflow_command(False), 'Next I will work.', 'Next I will work.', 'Next I will work.'])
        result = (await self.engine._run_loop('Run the workflow.', sys_print_callback=lambda _: None, event_callback=self._workflow_event)).text
        self.assertIn('System Halt', result)
        self.assertEqual([e['retrying'] for e in self.events if e['type'] == 'workflow_incomplete'], [True, True, False])
        self.assertEqual(len(self.engine.tool_context['command_sessions'].records), 1)
        self.requests = []
        self.fixture(['HELLO'])
        self.assertEqual(await self.run_task(), 'HELLO')
        self.assertEqual(len(self.requests), 1)

    async def test_workflow_guard_does_not_override_stop_or_unselected_skills(self):
        self.workflow_skill()
        self.fixture([self.workflow_command(False), 'STATUS_ONLY'])
        self.assertEqual(await self.run_task(), 'STATUS_ONLY')
        self.requests = []
        self.fixture([self.workflow_command(False), 'I will continue.'])
        async def stop(event):
            if event['type'] == 'workflow_incomplete':
                self.engine.abort_event.set()
        result = (await self.engine._run_loop('Run the workflow.', selected_skills=['fixture-workflow'], sys_print_callback=lambda _: None, event_callback=stop)).text
        self.assertIn('Turn aborted', result)
        self.assertEqual(len(self.requests), 2)

    async def test_validator_context_and_errors_after_execution_are_distinguished(self):
        result = await self.engine._execute_tool_wrapper({"id": "rule", "name": "record_tool", "input": {"value": "RULE_FAILURE"}}, lambda _: None)
        self.assertTrue(result["_input_error"])
        self.assertIn("fixture rule rejected input", result["content"])
        self.assertEqual(self.tool.calls, [])
        async def partially_executed(value):
            self.tool.calls.append(value)
            RecordInput(value=123)
        self.tool.execute = partially_executed
        result = await self.engine._execute_tool_wrapper({"id": "partial", "name": "record_tool", "input": {"value": "started"}}, lambda _: None)
        self.assertTrue(result["is_error"])
        self.assertNotIn("_input_error", result)
        self.assertIn("Execution may have begun", result["content"])
        self.assertEqual(self.tool.calls, ["started"])

    def test_json_validation_and_nested_tool_schema_are_complete(self):
        for value in ['', 'null', '[]', '{"value":NaN}', '{"value":"first","value":"second"}']:
            with self.subTest(value=value), self.assertRaises(ToolProtocolError):
                parse_tool_arguments(value)
        schema = TodoWriteTool.get_tool_definition()["input_schema"]
        self.assertIn("TodoItemInput", schema["$defs"])
        self.assertIn("content", schema["$defs"]["TodoItemInput"]["required"])
        self.assertEqual(schema["properties"]["todos"]["items"]["$ref"], "#/$defs/TodoItemInput")

    async def test_finish_marker_closes_without_socket_eof_and_keeps_late_usage(self):
        closed = asyncio.Event()
        async def create(**request):
            self.requests.append(request)
            async def stream():
                try:
                    yield NS(choices=[NS(delta=NS(content='EARLY_TEXT'), finish_reason=None)], usage=None)
                    yield NS(choices=[NS(delta=NS(content=None), finish_reason='stop')], usage=None)
                    await asyncio.sleep(.01)
                    yield NS(choices=[], usage=NS(prompt_tokens=12, completion_tokens=3))
                    await asyncio.Event().wait()
                finally:
                    closed.set()
            return stream()
        self.engine.client.chat.completions.create = create
        with patch('codeclaw.protocols.stream_watch.stream_limits', return_value=(1, .1, .05)):
            self.assertEqual(await self.run_task(), 'EARLY_TEXT')
        self.assertTrue(closed.is_set())
        self.assertEqual(self.engine.session_token_usage['input_tokens'], 12)
        metrics = self.recorder.record['modelStreams'][0]
        self.assertEqual(metrics['finishReason'], 'stop')
        self.assertIn('firstTextMs', metrics)

    async def test_keepalives_cannot_hide_stall_and_partial_survives_without_fallback(self):
        async def create(**request):
            self.requests.append(request)
            async def stream():
                yield NS(choices=[NS(delta=NS(content='PARTIAL_VISIBLE'), finish_reason=None)], usage=None)
                while True:
                    await asyncio.sleep(.01)
                    yield NS(choices=[], usage=None)
            return stream()
        self.engine.client.chat.completions.create = create
        self.engine.fallback_model = 'different-fallback'
        with patch('codeclaw.protocols.stream_watch.stream_limits', return_value=(1, .06, .02)):
            result = await self.run_task()
        self.assertIn('没有新的文本', result)
        self.assertEqual(len(self.requests), 1)
        self.recorder.accept({'type': 'turn_error'})
        self.assertEqual(self.recorder.record['answer'], 'PARTIAL_VISIBLE')
        self.assertEqual(self.engine.messages[-1]['_stop_reason'], 'stream_interrupted')
        self.engine.persist_session_state()
        restored = self.engine.session_manager.load_session_state(self.engine.session_id)
        self.assertEqual(restored['messages'][-1]['content'], 'PARTIAL_VISIBLE')
        self.assertEqual(self.recorder.record['modelStreams'][0]['status'], 'error')

    async def test_header_wait_is_bounded_and_stop_does_not_wait_for_headers(self):
        async def create(**request):
            self.requests.append(request)
            await asyncio.Event().wait()
        self.engine.client.chat.completions.create = create
        with patch('codeclaw.protocols.stream_watch.stream_limits', return_value=(.05, .1, .02)):
            result = await self.run_task()
        self.assertIn('首个有效响应', result)
        self.assertEqual(len(self.requests), 1)
        async def stop(event):
            if event['type'] == 'stream_status':
                self.engine.abort_event.set()
        result = (await self.engine._run_loop('stop', sys_print_callback=lambda _:None, event_callback=stop)).text
        self.assertIn('Turn aborted', result)

    async def test_reasoning_progress_is_not_idle_or_exposed_as_answer(self):
        async def create(**request):
            async def stream():
                for _ in range(4):
                    await asyncio.sleep(.06)
                    yield NS(choices=[NS(delta=NS(content=None, reasoning_content='PRIVATE_REASONING'), finish_reason=None)], usage=None)
                yield NS(choices=[NS(delta=NS(content='FINAL_TEXT'), finish_reason='stop')], usage=NS(prompt_tokens=10, completion_tokens=4))
            return stream()
        self.engine.client.chat.completions.create = create
        with patch('codeclaw.protocols.stream_watch.stream_limits', return_value=(1, .15, .02)):
            self.assertEqual(await self.run_task(), 'FINAL_TEXT')
        self.assertEqual(self.recorder.record['modelStreams'][0]['reasoningChunks'], 4)
        self.assertNotIn('PRIVATE_REASONING', str(self.recorder.record))

    async def test_stall_preserves_already_executed_streaming_tool_results(self):
        async def create(**request):
            self.requests.append(request)
            async def stream():
                yield NS(choices=[NS(delta=NS(content=None, tool_calls=[NS(index=0, id='first', function=NS(name='record_tool', arguments='{"value":"once"}'))]), finish_reason=None)], usage=None)
                yield NS(choices=[NS(delta=NS(content=None, tool_calls=[NS(index=1, id='second', function=NS(name='record_tool', arguments='{'))]), finish_reason=None)], usage=None)
                await asyncio.Event().wait()
            return stream()
        self.engine.client.chat.completions.create = create
        with patch('codeclaw.protocols.stream_watch.stream_limits', return_value=(1, .15, .02)):
            result = await self.run_task()
        self.assertIn('没有新的文本', result)
        self.assertEqual(self.tool.calls, ['once'])
        self.assertEqual(len(self.requests), 1)
        self.assertIn('saved:once', str(self.engine.messages[-1]))
        self.assertEqual(self.engine.messages[-1]['_origin'], 'stream_interrupted')
