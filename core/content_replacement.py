import copy
import os
import time


class ContentReplacementManager:
    def __init__(self, base_dir: str = ".codeclaw/tool-results", session_id: str = None):
        self.base_dir = base_dir
        self.session_id = session_id
        self.registry = {}
        self.cleanup_history = []

    def set_session(self, session_id: str):
        self.session_id = session_id

    def get_session_dir(self) -> str:
        if self.session_id:
            return os.path.abspath(os.path.join(self.base_dir, self.session_id))
        return os.path.abspath(self.base_dir)

    def register_spill(
        self,
        *,
        file_path: str,
        tool_name: str = "",
        tool_use_id: str = "",
        original_content_chars: int = 0,
        metadata: dict = None,
    ) -> dict:
        abs_path = os.path.abspath(file_path)
        entry = {
            "path": abs_path,
            "tool_name": tool_name,
            "tool_use_id": tool_use_id,
            "original_content_chars": int(original_content_chars or 0),
            "metadata": copy.deepcopy(metadata or {}),
            "timestamp": time.time(),
            "exists": os.path.exists(abs_path),
        }
        self.registry[abs_path] = entry
        return entry

    def cleanup_orphans(self) -> list[dict]:
        session_dir = self.get_session_dir()
        if not os.path.isdir(session_dir):
            return []

        known_paths = set(self.registry.keys())
        removed = []
        for file_name in os.listdir(session_dir):
            file_path = os.path.abspath(os.path.join(session_dir, file_name))
            if not os.path.isfile(file_path):
                continue
            if file_path in known_paths:
                continue
            try:
                size = os.path.getsize(file_path)
            except Exception:
                size = 0
            try:
                os.remove(file_path)
                removed.append({
                    "path": file_path,
                    "size": size,
                    "timestamp": time.time(),
                })
            except Exception:
                continue

        if removed:
            self.cleanup_history = (removed + self.cleanup_history)[:20]
        return removed

    def mark_existing_entries(self):
        for path, entry in list(self.registry.items()):
            entry["exists"] = os.path.exists(path)

    def export_state(self) -> dict:
        self.mark_existing_entries()
        return {
            "session_id": self.session_id,
            "registry": copy.deepcopy(self.registry),
            "cleanup_history": copy.deepcopy(self.cleanup_history),
        }

    def load_state(self, payload):
        payload = payload or {}
        self.session_id = payload.get("session_id", self.session_id)
        self.registry = dict(payload.get("registry", {}) or {})
        self.cleanup_history = list(payload.get("cleanup_history", []) or [])[:20]
        self.mark_existing_entries()

    def render_summary(self) -> str:
        self.mark_existing_entries()
        active_entries = [item for item in self.registry.values() if item.get("exists")]
        missing_entries = [item for item in self.registry.values() if not item.get("exists")]
        return (
            f"Content replacements: active={len(active_entries)}, "
            f"missing={len(missing_entries)}, "
            f"cleanup_events={len(self.cleanup_history)}"
        )
