"""Decode leaked DeepSeek tool envelopes from assistant completion text."""
import json
import re
import uuid

MARKERS = ("<｜DSML｜", "<|DSML|")
STARTS = tuple(marker + "tool_calls" for marker in MARKERS)


def invalid_json_constant(value):
    raise ValueError("Invalid JSON constant")


def marker_position(text):
    positions = sorted(match.start() for marker in STARTS for match in re.finditer(re.escape(marker), text))
    for position in positions:
        prefix = text[:position]
        line = prefix.rsplit("\n", 1)[-1]
        if not line.strip() and prefix.count("`") % 2 == 0 and prefix.count("~~~") % 2 == 0:
            return position
    return None


class ToolTextFilter:
    def __init__(self):
        self.text = ""
        self.emitted = 0

    def feed(self, delta):
        self.text += delta
        end = marker_position(self.text)
        if end is None:
            end = len(self.text)
            for marker in STARTS:
                for size in range(1, len(marker)):
                    if self.text.endswith(marker[:size]):
                        end = min(end, len(self.text) - size)
        value = self.text[self.emitted:end]
        self.emitted = end
        return value

    def finish(self, visible):
        value = visible[self.emitted:]
        self.emitted = len(visible)
        return value


def decode_tool_text(text, tools_schema):
    position = marker_position(text)
    if position is None:
        return text, []
    prefix, envelope = text[:position], text[position:]
    token = "｜DSML｜" if envelope.startswith(MARKERS[0]) else "|DSML|"
    close = f"</{token}tool_calls>"
    if close in envelope and envelope.split(close, 1)[1].strip() not in {"", "<｜end▁of▁sentence｜>"}:
        return text, []
    pattern = rf"<{re.escape(token)}tool_calls>\s*(.*?)\s*{re.escape(close)}(?:\s*<｜end▁of▁sentence｜>)?\s*"
    outer = re.fullmatch(pattern, envelope, flags=re.S)
    if not outer:
        raise ValueError("模型返回了不完整的工具调用，尚未执行任何操作，请重试。")
    declared = {tool["function"]["name"]: tool["function"].get("parameters", {}) for tool in tools_schema}
    invoke = re.compile(rf'<{re.escape(token)}invoke\s+name="([^"<>]+)">\s*(.*?)\s*</{re.escape(token)}invoke>', re.S)
    parameter = re.compile(rf'<{re.escape(token)}parameter\s+name="([^"<>]+)"\s+string="(true|false)">(.*?)</{re.escape(token)}parameter>', re.S)
    body, cursor, calls = outer.group(1), 0, []
    for match in invoke.finditer(body):
        if body[cursor:match.start()].strip():
            raise ValueError("模型的工具调用格式不正确，尚未执行任何操作。")
        name, parameters = match.group(1), match.group(2)
        if name not in declared:
            raise ValueError("模型请求了当前不可用的工具，尚未执行任何操作。")
        args, offset = {}, 0
        for item in parameter.finditer(parameters):
            key, string, value = item.groups()
            if parameters[offset:item.start()].strip() or key in args or key not in declared[name].get("properties", {}):
                raise ValueError("模型的工具参数不正确，尚未执行任何操作。")
            try:
                args[key] = value if string == "true" else json.loads(value, parse_constant=invalid_json_constant)
            except (ValueError, TypeError):
                raise ValueError("模型的工具参数不是有效 JSON，尚未执行任何操作。") from None
            offset = item.end()
        if parameters[offset:].strip() or not set(declared[name].get("required", [])).issubset(args):
            raise ValueError("模型的工具调用缺少有效参数，尚未执行任何操作。")
        calls.append({"id": "call_dsml_" + uuid.uuid4().hex, "name": name, "arguments": json.dumps(args, ensure_ascii=False)})
        cursor = match.end()
    if not calls or body[cursor:].strip() or len(calls) > 100:
        raise ValueError("模型的工具调用格式不正确，尚未执行任何操作。")
    return prefix, calls
