"""Decode leaked DeepSeek tool envelopes from assistant completion text."""
import json
import re
import uuid

MARKERS = ("<｜DSML｜", "<|DSML|")
STARTS = tuple(marker + "tool_calls" for marker in MARKERS)


class ToolProtocolError(ValueError):
    """A rejected, unexecuted completion; safe to ask the model to repair."""

    def __init__(self, detail, *, code="invalid_tool_call", raw_text="", visible_text="", tool_name=""):
        super().__init__(detail)
        self.code = code
        self.raw_text = raw_text
        self.visible_text = visible_text
        self.tool_name = tool_name
        self.usage = None


def parse_tool_arguments(raw):
    """Never replace corrupt, duplicate-key or non-object arguments with {}."""
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key: " + key)
            result[key] = value
        return result
    try:
        result = json.loads(raw, parse_constant=invalid_json_constant, object_pairs_hook=unique_object)
        if not isinstance(result, dict):
            raise ValueError("arguments must be a JSON object")
        return result
    except (ValueError, TypeError) as exc:
        raise ToolProtocolError("Tool arguments must be one valid JSON object with unique keys: " + str(exc),
                                code="invalid_json_arguments", raw_text=raw) from None


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
    def reject(detail, code="invalid_dsml", tool_name=""):
        raise ToolProtocolError(detail, code=code, raw_text=text, visible_text=prefix, tool_name=tool_name)
    token = "｜DSML｜" if envelope.startswith(MARKERS[0]) else "|DSML|"
    close = f"</{token}tool_calls>"
    if close in envelope and envelope.split(close, 1)[1].strip() not in {"", "<｜end▁of▁sentence｜>"}:
        return text, []
    pattern = rf"<{re.escape(token)}tool_calls>\s*(.*?)\s*{re.escape(close)}(?:\s*<｜end▁of▁sentence｜>)?\s*"
    outer = re.fullmatch(pattern, envelope, flags=re.S)
    if not outer:
        reject("Incomplete DSML tool_calls envelope. Close every invoke/parameter and the tool_calls envelope.")
    declared = {tool["function"]["name"]: tool["function"].get("parameters", {}) for tool in tools_schema}
    invoke = re.compile(rf'<{re.escape(token)}invoke\s+name="([^"<>]+)">\s*(.*?)\s*</{re.escape(token)}invoke>', re.S)
    parameter = re.compile(rf'<{re.escape(token)}parameter\s+name="([^"<>]+)"\s+string="(true|false)">(.*?)</{re.escape(token)}parameter>', re.S)
    body, cursor, calls = outer.group(1), 0, []
    for match in invoke.finditer(body):
        if body[cursor:match.start()].strip():
            reject("Unexpected text between DSML invoke blocks.")
        name, parameters = match.group(1), match.group(2)
        if name not in declared:
            reject("Tool is unavailable: " + name + ". Use a tool declared in the current tools schema.", "unavailable_tool", name)
        args, offset = {}, 0
        items = list(parameter.finditer(parameters))
        properties = declared[name].get("properties", {})
        # Some compatible gateways wrap the full native argument object in one
        # DSML parameter. Only unwrap this unambiguous variant; never merge it
        # with sibling parameters or bypass the real tool's input validation.
        if len(items) == 1 and items[0].group(1) == "arguments" and "arguments" not in properties:
            item = items[0]
            if parameters[:item.start()].strip() or parameters[item.end():].strip() or item.group(2) != "false":
                reject("Invalid wrapped arguments parameter in " + name, tool_name=name)
            try:
                args = parse_tool_arguments(item.group(3))
            except ToolProtocolError as exc:
                reject(str(exc), tool_name=name)
            unknown = args.keys() - properties.keys()
            if unknown:
                reject("Unknown wrapped parameters in " + name + ": " + ", ".join(sorted(unknown)), tool_name=name)
            offset = len(parameters)
            items = []
        for item in items:
            key, string, value = item.groups()
            if parameters[offset:item.start()].strip():
                reject("Malformed parameter markup before " + key + " in " + name, tool_name=name)
            if key in args:
                reject("Duplicate parameter " + key + " in " + name, tool_name=name)
            if key not in declared[name].get("properties", {}):
                reject("Unknown parameter " + key + " in " + name + ". Allowed parameters: " +
                       ", ".join(declared[name].get("properties", {})), "unknown_parameter", name)
            try:
                args[key] = value if string == "true" else json.loads(value, parse_constant=invalid_json_constant)
            except (ValueError, TypeError):
                reject("Parameter " + key + " in " + name + " is not valid JSON. Use string=true only for strings.", tool_name=name)
            offset = item.end()
        if parameters[offset:].strip():
            reject("Malformed or unclosed parameter markup in " + name, tool_name=name)
        missing = set(declared[name].get("required", [])) - args.keys()
        if missing:
            reject("Missing required parameters in " + name + ": " + ", ".join(sorted(missing)), tool_name=name)
        calls.append({"id": "call_dsml_" + uuid.uuid4().hex, "name": name, "arguments": json.dumps(args, ensure_ascii=False)})
        cursor = match.end()
    if not calls or body[cursor:].strip() or len(calls) > 100:
        reject("Invalid DSML call list; supply between 1 and 100 complete invoke blocks.")
    return prefix, calls
