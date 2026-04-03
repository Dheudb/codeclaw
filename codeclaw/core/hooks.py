import json
import os
import copy
from dataclasses import dataclass
from typing import Dict, List


class _SafeFormatDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


@dataclass
class StopHookDecision:
    behavior: str
    reason: str = ""


@dataclass
class BeforeToolUseDecision:
    behavior: str
    outputs: List[str]
    updated_input: Dict[str, object]
    reason: str = ""
    history: List[Dict[str, object]] = None


@dataclass
class PostSamplingHookResult:
    outputs: List[Dict[str, object]]


class HookManager:
    """
    Lightweight, config-driven lifecycle hooks.

    Hooks are loaded from `.codeclaw/hooks.json` and rendered as text snippets
    that get injected back into the agent context.
    """

    def __init__(self, config_path=".codeclaw/hooks.json"):
        self.config_path = config_path

    def execute(self, event: str, payload: Dict[str, object]) -> List[str]:
        try:
            config = self._load_config()
        except Exception:
            return []

        rendered_outputs = []
        for hook in config.get(event, []):
            try:
                if not self._hook_matches(hook, payload):
                    continue

                template = str(hook.get("content", "")).strip()
                if not template:
                    continue

                rendered = template.format_map(_SafeFormatDict(self._stringify_payload(payload))).strip()
                if rendered:
                    rendered_outputs.append(rendered)
            except Exception:
                continue

        return rendered_outputs

    def evaluate_stop_hooks(self, payload: Dict[str, object]) -> StopHookDecision:
        try:
            config = self._load_config()
        except Exception:
            return StopHookDecision("allow", "")

        blocking_reasons = []
        retry_reasons = []

        for hook in config.get("StopHook", []):
            try:
                if not self._hook_matches(hook, payload):
                    continue

                template = str(hook.get("content", "")).strip()
                rendered = ""
                if template:
                    rendered = template.format_map(_SafeFormatDict(self._stringify_payload(payload))).strip()

                behavior = str(
                    hook.get("behavior", hook.get("action", "allow"))
                ).strip().lower()

                if behavior == "block":
                    blocking_reasons.append(rendered or "Blocked by stop hook.")
                elif behavior == "retry":
                    retry_reasons.append(rendered or "Retry requested by stop hook.")
            except Exception:
                continue

        if blocking_reasons:
            return StopHookDecision("block", "\n".join(blocking_reasons))
        if retry_reasons:
            return StopHookDecision("retry", "\n".join(retry_reasons))
        return StopHookDecision("allow", "")

    def evaluate_before_tool_hooks(self, payload: Dict[str, object]) -> BeforeToolUseDecision:
        try:
            config = self._load_config()
        except Exception:
            return BeforeToolUseDecision(
                behavior="allow",
                outputs=[],
                updated_input=copy.deepcopy(payload.get("tool_input", {}) or {}),
                history=[],
            )

        current_input = copy.deepcopy(payload.get("tool_input", {}) or {})
        rendered_outputs = []
        history = []

        for index, hook in enumerate(config.get("BeforeToolUse", []), start=1):
            try:
                effective_payload = dict(payload)
                effective_payload["tool_input"] = current_input
                if not self._hook_matches(hook, effective_payload):
                    continue

                template = str(hook.get("content", "")).strip()
                rendered = ""
                if template:
                    rendered = self._render_template(template, effective_payload)
                    if rendered:
                        rendered_outputs.append(rendered)

                previous_input = copy.deepcopy(current_input)
                if isinstance(hook.get("updated_input"), dict):
                    current_input = self._render_structured_value(
                        hook.get("updated_input"),
                        effective_payload,
                    )
                elif isinstance(hook.get("input_patch"), dict):
                    rendered_patch = self._render_structured_value(
                        hook.get("input_patch"),
                        effective_payload,
                    )
                    if isinstance(rendered_patch, dict):
                        current_input = {
                            **current_input,
                            **rendered_patch,
                        }

                behavior = str(
                    hook.get("behavior", hook.get("action", "allow"))
                ).strip().lower()
                normalized_behavior = {
                    "approve": "allow",
                    "accept": "allow",
                    "allow": "allow",
                    "continue": "allow",
                    "deny": "reject",
                    "block": "reject",
                    "reject": "reject",
                }.get(behavior, "allow")

                input_changed = previous_input != current_input
                history.append({
                    "hook_index": index,
                    "behavior": normalized_behavior,
                    "tool_name": payload.get("tool_name", ""),
                    "tool_use_id": payload.get("tool_use_id", ""),
                    "rendered": rendered,
                    "input_changed": input_changed,
                    "updated_input": copy.deepcopy(current_input) if input_changed else None,
                })

                if normalized_behavior == "reject":
                    return BeforeToolUseDecision(
                        behavior="reject",
                        outputs=rendered_outputs,
                        updated_input=current_input,
                        reason=rendered or "Rejected by BeforeToolUse hook.",
                        history=history,
                    )
            except Exception:
                continue

        return BeforeToolUseDecision(
            behavior="allow",
            outputs=rendered_outputs,
            updated_input=current_input,
            history=history,
        )

    def evaluate_post_sampling_hooks(self, payload: Dict[str, object]) -> PostSamplingHookResult:
        try:
            config = self._load_config()
        except Exception:
            return PostSamplingHookResult(outputs=[])

        outputs = []
        for index, hook in enumerate(config.get("PostSamplingHook", []), start=1):
            try:
                if not self._hook_matches(hook, payload):
                    continue

                template = str(hook.get("content", "")).strip()
                rendered = ""
                if template:
                    rendered = self._render_template(template, payload)

                if not rendered:
                    continue

                outputs.append({
                    "hook_index": index,
                    "behavior": str(hook.get("behavior", "observe") or "observe").strip().lower(),
                    "severity": str(hook.get("severity", "info") or "info").strip().lower(),
                    "rendered": rendered,
                })
            except Exception:
                continue

        return PostSamplingHookResult(outputs=outputs)

    def _load_config(self) -> Dict[str, list]:
        if not os.path.exists(self.config_path):
            return {}
        with open(self.config_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            return {}
        return payload

    def _hook_matches(self, hook: Dict[str, object], payload: Dict[str, object]) -> bool:
        tools = hook.get("tools")
        if tools:
            tool_name = str(payload.get("tool_name", "")).strip()
            if tool_name not in set(str(item) for item in tools):
                return False

        modes = hook.get("modes")
        if modes:
            current_mode = str(payload.get("mode", "")).strip()
            if current_mode not in set(str(item) for item in modes):
                return False

        return True

    def _stringify_payload(self, payload: Dict[str, object]) -> Dict[str, str]:
        rendered = {}
        for key, value in payload.items():
            if isinstance(value, (dict, list)):
                rendered[key] = json.dumps(value, ensure_ascii=False)
            else:
                rendered[key] = str(value)
        return rendered

    def _render_template(self, template: str, payload: Dict[str, object]) -> str:
        return template.format_map(_SafeFormatDict(self._stringify_payload(payload))).strip()

    def _render_structured_value(self, value, payload: Dict[str, object]):
        if isinstance(value, dict):
            return {
                key: self._render_structured_value(inner, payload)
                for key, inner in value.items()
            }
        if isinstance(value, list):
            return [self._render_structured_value(item, payload) for item in value]
        if isinstance(value, str):
            return self._render_template(value, payload)
        return copy.deepcopy(value)
