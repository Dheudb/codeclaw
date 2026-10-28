import copy
import json
from typing import Optional


class StructuredOutputManager:
    def __init__(self):
        self.request = None

    def set_request(self, request: Optional[dict]):
        if request is None:
            self.request = None
            return None

        normalized = {
            "type": str(request.get("type", "json_object")).strip().lower(),
            "required_keys": list(request.get("required_keys", []) or []),
            "description": str(request.get("description", "") or "").strip(),
        }
        if normalized["type"] not in {"json_object", "json_array"}:
            raise ValueError("Unsupported structured output type.")
        self.request = normalized
        return copy.deepcopy(self.request)

    def clear(self):
        self.request = None

    def export_state(self) -> dict:
        return {
            "request": copy.deepcopy(self.request),
        }

    def load_state(self, payload):
        payload = payload or {}
        request = payload.get("request")
        if request:
            self.set_request(request)
        else:
            self.request = None

    def render_prompt_summary(self) -> str:
        if not self.request:
            return ""

        lines = [
            "--- STRUCTURED OUTPUT REQUIREMENT ---",
            f"Return the final answer as valid {self.request.get('type')}.",
        ]
        if self.request.get("description"):
            lines.append(self.request["description"])
        required_keys = self.request.get("required_keys", [])
        if required_keys:
            lines.append("Required keys: " + ", ".join(required_keys))
        lines.append("Do not wrap the JSON in markdown code fences.")
        return "\n".join(lines)

    def validate(self, text: str) -> tuple[bool, str]:
        if not self.request:
            return True, ""

        rendered = str(text or "").strip()
        if not rendered:
            return False, "Final answer is empty."

        try:
            parsed = json.loads(rendered)
        except Exception as exc:
            return False, f"Final answer is not valid JSON: {exc}"

        expected_type = self.request.get("type")
        if expected_type == "json_object":
            if not isinstance(parsed, dict):
                return False, "Final answer must be a JSON object."
            missing_keys = [
                key for key in self.request.get("required_keys", [])
                if key not in parsed
            ]
            if missing_keys:
                return False, "Missing required keys: " + ", ".join(missing_keys)
        elif expected_type == "json_array":
            if not isinstance(parsed, list):
                return False, "Final answer must be a JSON array."

        return True, ""
