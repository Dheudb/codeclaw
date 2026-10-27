"""Real SDK parsers against deterministic HTTP/SSE, without remote model claims."""
import json
import unittest
import httpx
from anthropic import AsyncAnthropic
from anthropic import _base_client as anthropic_transport
from openai import AsyncOpenAI
import test_tool_recovery as recovery


class ProviderContractTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = recovery.RecoveryTests.asyncSetUp
    asyncTearDown = recovery.RecoveryTests.asyncTearDown

    async def verify_provider(self, provider, truncate):
        transport_http = getattr(anthropic_transport, "httpx2", httpx) if provider == "anthropic" else httpx
        await self.engine.client.close()
        self.engine.model_provider = provider
        self.engine.api_base_url = "https://api.anthropic.com" if provider == "anthropic" else "https://fixture.invalid/v1"
        requests = []
        successful = 0
        def transport(request):
            nonlocal successful
            payload = json.loads(request.content)
            requests.append(payload)
            if len(requests) == 1:
                return transport_http.Response(429, json={"error": {"type": "rate_limit_error", "message": "fixture retry"}}, headers={"retry-after": "0"})
            successful += 1
            tools = successful == 1
            partial = successful == 2 and truncate
            if provider == "openai":
                chunks = []
                if tools:
                    for index, value in enumerate(("FIRST", "SECOND")):
                        chunks.append({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": index, "id": "call-" + str(index), "type": "function",
                            "function": {"name": "record_tool", "arguments": '{"value":'}}]}, "finish_reason": None}]})
                        chunks.append({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": index, "function": {"arguments": json.dumps(value) + "}"}}]}, "finish_reason": None}]})
                else:
                    chunks.append({"choices": [{"index": 0, "delta": {"content": "PARTIAL" if partial else "FINAL"}, "finish_reason": None}]})
                chunks.append({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls" if tools else "length" if partial else "stop"}]})
                chunks.append({"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "provider_extra": "accepted"}})
                data = "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks) + "data: [DONE]\n\n"
            else:
                events = [{"type": "message_start", "message": {"id": "msg-fixture", "type": "message", "role": "assistant", "model": "fixture",
                    "content": [], "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 10, "output_tokens": 0}}}]
                if tools:
                    for index, value in enumerate(("FIRST", "SECOND")):
                        events += [{"type": "content_block_start", "index": index, "content_block": {"type": "tool_use", "id": "call-" + str(index), "name": "record_tool", "input": {}}},
                            {"type": "content_block_delta", "index": index, "delta": {"type": "input_json_delta", "partial_json": '{"value":'}},
                            {"type": "content_block_delta", "index": index, "delta": {"type": "input_json_delta", "partial_json": json.dumps(value) + "}"}},
                            {"type": "content_block_stop", "index": index}]
                else:
                    events += [{"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "PARTIAL" if partial else "FINAL"}},
                        {"type": "content_block_stop", "index": 0}]
                events += [{"type": "message_delta", "delta": {"stop_reason": "tool_use" if tools else "max_tokens" if partial else "end_turn", "stop_sequence": None}, "usage": {"output_tokens": 5}},
                    {"type": "message_stop"}]
                data = "".join("event: " + event["type"] + "\ndata: " + json.dumps(event) + "\n\n" for event in events)
            return transport_http.Response(200, headers={"content-type": "text/event-stream"}, content=data.encode())
        http = transport_http.AsyncClient(transport=transport_http.MockTransport(transport))
        self.engine.client = (AsyncAnthropic(api_key="fixture", base_url=self.engine.api_base_url, http_client=http, max_retries=1)
            if provider == "anthropic" else AsyncOpenAI(api_key="fixture", base_url=self.engine.api_base_url, http_client=http, max_retries=1))
        self.engine._sync_message_context()
        result = await self.engine.run("Perform both fixture operations and report", sys_print_callback=lambda _: None)
        self.assertTrue(result.ok, result.to_dict())
        self.assertIn("FINAL", result.text)
        self.assertCountEqual(self.tool.calls, ["FIRST", "SECOND"])
        self.assertEqual(len(self.tool.calls), 2, "SDK retry and token continuation must not replay tool effects")
        self.assertEqual(len(requests), 4 if truncate else 3)
        self.assertTrue(all(request["model"] == "fixture" for request in requests))
        self.assertIn("saved:FIRST", json.dumps(requests[-1]))
        self.assertIn("saved:SECOND", json.dumps(requests[-1]))

    async def test_openai_parallel_streamed_arguments_and_http_retry(self):
        await self.verify_provider("openai", False)

    async def test_anthropic_parallel_streamed_arguments_and_http_retry(self):
        await self.verify_provider("anthropic", False)

    async def test_openai_output_limit_continuation_does_not_replay_tools(self):
        await self.verify_provider("openai", True)

    async def test_anthropic_output_limit_continuation_does_not_replay_tools(self):
        await self.verify_provider("anthropic", True)
