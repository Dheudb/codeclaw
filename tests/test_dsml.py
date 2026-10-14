import unittest
from codeclaw.protocols.dsml import ToolTextFilter, decode_tool_text

SCHEMA = [{"type": "function", "function": {"name": "agent_tool", "parameters": {"properties": {
    "prompt": {}, "run_in_background": {}, "values": {}}, "required": ["prompt"]}}}]


def envelope(parameters, name="agent_tool"):
    return f'<｜DSML｜tool_calls>\n<｜DSML｜invoke name="{name}">\n{parameters}\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>'


class DSMLTests(unittest.TestCase):
    def test_decode_strings_json_and_parallel_calls(self):
        parameters = '<｜DSML｜parameter name="prompt" string="true">你好\n<>& raw</｜DSML｜parameter>\n<｜DSML｜parameter name="run_in_background" string="false">true</｜DSML｜parameter>\n<｜DSML｜parameter name="values" string="false">[1,{"nested":false}]</｜DSML｜parameter>'
        body = envelope(parameters)
        import json
        visible, calls = decode_tool_text("开始。\n\n" + body, SCHEMA)
        self.assertEqual(visible, "开始。\n\n")
        args = json.loads(calls[0]["arguments"])
        self.assertEqual(args["prompt"], "你好\n<>& raw")
        self.assertIs(args["run_in_background"], True)
        self.assertEqual(args["values"], [1, {"nested": False}])
        ten = body.replace('<｜DSML｜invoke name="agent_tool">\n' + parameters + '\n</｜DSML｜invoke>', '\n'.join(['<｜DSML｜invoke name="agent_tool">\n' + parameters + '\n</｜DSML｜invoke>'] * 10))
        self.assertEqual(len(decode_tool_text(ten, SCHEMA)[1]), 10)

    def test_stream_holds_split_tags_without_breaking_plain_text(self):
        text = "现在开始。\n" + envelope('<｜DSML｜parameter name="prompt" string="true">你好</｜DSML｜parameter>')
        gate = ToolTextFilter()
        visible = ''.join(gate.feed(char) for char in text)
        clean, calls = decode_tool_text(text, SCHEMA)
        visible += gate.finish(clean)
        self.assertEqual(visible, "现在开始。\n")
        self.assertEqual(len(calls), 1)
        gate = ToolTextFilter()
        self.assertEqual(gate.feed("x <") + gate.finish("x <"), "x <")

    def test_documentation_is_text_and_not_a_tool_invocation(self):
        block = envelope('<｜DSML｜parameter name="prompt" string="true">example</｜DSML｜parameter>')
        for text in ["```xml\n" + block + "\n```", "Here is `" + block + "`", "> " + block, block + "\nThis was an example."]:
            self.assertEqual(decode_tool_text(text, SCHEMA), (text, []))

    def test_malformed_unavailable_or_duplicate_calls_fail_as_a_whole(self):
        valid = '<｜DSML｜parameter name="prompt" string="true">hello</｜DSML｜parameter>'
        cases = [envelope(valid, "unavailable"), envelope(valid)[:-8], envelope(valid + valid),
                 envelope(valid.replace('name="prompt"', 'name="unknown"')),
                 envelope(valid + '<｜DSML｜parameter name="values" string="false">NaN</｜DSML｜parameter>'),
                 envelope('')]
        for text in cases:
            with self.subTest(text=text[:60]), self.assertRaises(ValueError):
                decode_tool_text(text, SCHEMA)

    def test_old_history_does_not_present_unexecuted_markup_as_a_result(self):
        from codeclaw.desktop.bridge import DesktopBridge
        text = "准备启动。\n" + envelope('<｜DSML｜parameter name="prompt" string="true">hello</｜DSML｜parameter>')
        history = DesktopBridge.history([{"role": "assistant", "content": [{"type": "text", "text": text}]}])
        self.assertNotIn("DSML", history[0]["text"])
        self.assertIn("未执行", history[0]["text"])
