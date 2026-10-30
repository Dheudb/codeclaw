"""Explicit per-connection input contracts; unknown aliases remain unknown."""
import copy
from xuloom.core.config import load_config, normalize_api_base_url


class UnsupportedInput(ValueError):
    pass


def validate_transport(value, provider):
    if provider == "openai" and "document" in media_types(value):
        raise UnsupportedInput("当前 OpenAI Chat 协议未接入原生文档输入。请用文件读取工具提取文本；未将文档静默替换为占位说明。")


def configured_inputs(provider, model, base_url):
    from xuloom.core.model_catalog import active_connection, connections
    cfg = load_config()
    active = active_connection(cfg)
    matches = [item for item in connections(cfg) if item["provider"] == provider
        and normalize_api_base_url(provider, item.get("base_url", "")) == normalize_api_base_url(provider, base_url or "")
        and model in item.get("models", [])]
    selected = next((item for item in matches if item["id"] == active["id"]), None)
    # The same gateway/model may belong to different accounts. Never guess between them.
    selected = selected or (matches[0] if len(matches) == 1 else None)
    return (selected or {}).get("model_inputs", {}).get(model)


def media_types(value):
    if isinstance(value, list):
        return set().union(*(media_types(item) for item in value)) if value else set()
    if not isinstance(value, dict):
        return set()
    kind = value.get("type")
    found = {"image"} if kind in {"image", "image_url", "input_image"} else {"document"} if kind in {"document", "input_file"} else set()
    for key in ("content", "parts"):
        if isinstance(value.get(key), (list, dict)):
            found |= media_types(value[key])
    return found


def validate_media(value, inputs):
    if inputs is None:
        return
    denied = media_types(value) - set(inputs)
    if denied:
        raise UnsupportedInput("当前模型配置不支持 " + ", ".join(sorted(denied)) +
            " 输入。请选择支持该输入的模型，或在模型设置中核对能力；历史附件不会被静默删除，也不会自动 OCR。")


def filter_tool_media(value, inputs):
    """Keep textual evidence and explicitly explain omitted unsupported tool media."""
    if inputs is None or not (media_types(value) - set(inputs)):
        return value
    def visit(item):
        if isinstance(item, list):
            return [visit(part) for part in item]
        if isinstance(item, dict):
            if media_types({"type": item.get("type")}) - set(inputs):
                return {"type": "text", "text": "Error: this model is configured without " + item["type"] + " input; this media was not sent or understood. Select a compatible model to inspect it."}
            result = copy.deepcopy(item)
            for key in ("content", "parts"):
                if isinstance(result.get(key), (list, dict)):
                    result[key] = visit(result[key])
            return result
        return item
    result = visit(value)
    if isinstance(result, dict) and "ok" in result:
        result.update(ok=False, is_error=True)
        result["metadata"] = {**result.get("metadata", {}), "code": "unsupported_input", "media_omitted": True}
    elif isinstance(result, list):
        result.insert(0, {"type": "text", "text": "Error: unsupported media omitted; textual tool output follows."})
    return result
