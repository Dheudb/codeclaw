import copy
import hashlib
import json
import os
import time


class VCRReplayResponse:
    def __init__(self, payload):
        payload = payload or {}
        self.stop_reason = payload.get("stop_reason")
        self.content = [VCRReplayBlock(item) for item in payload.get("content", []) or []]
        self.usage = VCRReplayUsage(payload.get("usage", {}) or {})


class VCRReplayUsage:
    def __init__(self, payload):
        payload = payload or {}
        for key, value in payload.items():
            setattr(self, key, value)


class VCRReplayBlock:
    def __init__(self, payload):
        payload = payload or {}
        for key, value in payload.items():
            setattr(self, key, value)
        if not hasattr(self, "type"):
            self.type = "text"
        if self.type == "text" and not hasattr(self, "text"):
            self.text = ""
        if self.type == "tool_use":
            if not hasattr(self, "id"):
                self.id = ""
            if not hasattr(self, "name"):
                self.name = ""
            if not hasattr(self, "input"):
                self.input = {}


class VCRManager:
    def __init__(self, cassette_dir: str = ".codeclaw/vcr", mode: str = None):
        self.cassette_dir = cassette_dir
        self.mode = str(mode or os.environ.get("CODECLAW_VCR_MODE", "off") or "off").lower()
        self.history = []

    def export_state(self) -> dict:
        return {
            "cassette_dir": self.cassette_dir,
            "mode": self.mode,
            "history": copy.deepcopy(self.history),
        }

    def load_state(self, payload):
        payload = payload or {}
        cassette_dir = payload.get("cassette_dir")
        if cassette_dir:
            self.cassette_dir = cassette_dir
        self.mode = str(payload.get("mode", self.mode or "off") or "off").lower()
        self.history = list(payload.get("history", []) or [])[:40]

    def _hash_request(self, request_payload: dict) -> str:
        raw = json.dumps(request_payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    def _cassette_path(self, request_hash: str) -> str:
        return os.path.abspath(os.path.join(self.cassette_dir, f"{request_hash}.json"))

    def build_request_payload(self, *, model: str, system: str, messages: list, tools: list, max_tokens: int, thinking: dict = None, provider: str = "anthropic") -> dict:
        return {
            "provider": provider,
            "model": model,
            "system": system,
            "messages": copy.deepcopy(messages),
            "tools": copy.deepcopy(tools),
            "max_tokens": int(max_tokens),
            "thinking": copy.deepcopy(thinking),
        }

    def try_replay(self, request_payload: dict):
        if self.mode != "replay":
            return None
        request_hash = self._hash_request(request_payload)
        path = self._cassette_path(request_hash)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            return None
        self._record_history("replay_hit", request_hash, path)
        return VCRReplayResponse(payload.get("response", {}))

    def record(self, request_payload: dict, response_payload: dict):
        if self.mode not in {"record", "replay_record"}:
            return None
        request_hash = self._hash_request(request_payload)
        path = self._cassette_path(request_hash)
        try:
            os.makedirs(self.cassette_dir, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "request_hash": request_hash,
                        "recorded_at": time.time(),
                        "request": request_payload,
                        "response": response_payload,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            self._record_history("record", request_hash, path)
            return path
        except Exception:
            return None

    def _record_history(self, event: str, request_hash: str, path: str):
        self.history.insert(0, {
            "event": event,
            "request_hash": request_hash,
            "path": path,
            "timestamp": time.time(),
        })
        self.history = self.history[:40]
