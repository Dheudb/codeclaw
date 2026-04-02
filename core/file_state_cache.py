import hashlib
import os
from typing import Dict, Optional


class FileStateCache:
    def __init__(self):
        self.entries: Dict[str, dict] = {}

    def _normalize_path(self, path: str) -> str:
        return os.path.abspath(path)

    def _range_key(self, start_line=None, end_line=None) -> str:
        return f"{start_line or 1}:{end_line or '*'}"

    def _safe_stat(self, abs_path: str):
        try:
            return os.stat(abs_path)
        except Exception:
            return None

    def _compute_sha256(self, abs_path: str) -> str:
        digest = hashlib.sha256()
        with open(abs_path, "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()[:16]

    def snapshot(self, path: str) -> Optional[dict]:
        abs_path = self._normalize_path(path)
        if not os.path.exists(abs_path) or not os.path.isfile(abs_path):
            return None

        stat = self._safe_stat(abs_path)
        if stat is None:
            return None

        try:
            sha256 = self._compute_sha256(abs_path)
        except Exception:
            sha256 = ""

        return {
            "path": abs_path,
            "size": int(getattr(stat, "st_size", 0) or 0),
            "mtime": float(getattr(stat, "st_mtime", 0.0) or 0.0),
            "sha256": sha256,
        }

    def has_been_read(self, path: str) -> bool:
        abs_path = self._normalize_path(path)
        entry = self.entries.get(abs_path, {})
        return bool(entry.get("has_been_read"))

    def get_entry(self, path: str) -> Optional[dict]:
        return self.entries.get(self._normalize_path(path))

    def should_skip_redundant_read(self, path: str, *, start_line=None, end_line=None) -> bool:
        abs_path = self._normalize_path(path)
        entry = self.entries.get(abs_path)
        if not entry:
            return False

        current_snapshot = self.snapshot(abs_path)
        if not current_snapshot:
            return False

        if current_snapshot.get("sha256") != entry.get("sha256"):
            return False

        range_key = self._range_key(start_line, end_line)
        ranges = entry.get("ranges", {})
        return range_key in ranges

    def record_read(
        self,
        path: str,
        *,
        kind: str,
        start_line=None,
        end_line=None,
        chars: int = 0,
    ) -> Optional[dict]:
        abs_path = self._normalize_path(path)
        snapshot = self.snapshot(abs_path)
        if not snapshot:
            self.entries.pop(abs_path, None)
            return None

        range_key = self._range_key(start_line, end_line)
        entry = {
            **snapshot,
            "kind": kind,
            "has_been_read": True,
            "last_read_range": range_key,
            "last_read_chars": int(chars or 0),
            "ranges": dict(self.entries.get(abs_path, {}).get("ranges", {}) or {}),
        }
        entry["ranges"][range_key] = {
            "kind": kind,
            "chars": int(chars or 0),
        }
        self.entries[abs_path] = entry
        return entry

    def record_write(self, path: str) -> Optional[dict]:
        abs_path = self._normalize_path(path)
        snapshot = self.snapshot(abs_path)
        if not snapshot:
            self.entries.pop(abs_path, None)
            return None

        previous_ranges = dict(self.entries.get(abs_path, {}).get("ranges", {}) or {})
        entry = {
            **snapshot,
            "kind": self.entries.get(abs_path, {}).get("kind", "text"),
            "has_been_read": True,
            "last_read_range": self.entries.get(abs_path, {}).get("last_read_range", "1:*"),
            "last_read_chars": self.entries.get(abs_path, {}).get("last_read_chars", 0),
            "ranges": previous_ranges,
            "last_write": {
                "size": snapshot.get("size", 0),
                "mtime": snapshot.get("mtime", 0.0),
                "sha256": snapshot.get("sha256", ""),
            },
        }
        self.entries[abs_path] = entry
        return entry

    def export_state(self) -> dict:
        return {
            "entries": self.entries,
        }

    def load_state(self, payload):
        payload = payload or {}
        entries = payload.get("entries", {})
        if isinstance(entries, dict):
            self.entries = entries
        else:
            self.entries = {}

    def render_summary(self) -> str:
        if not self.entries:
            return "File state cache: empty"

        lines = [f"File state cache ({len(self.entries)})"]
        for path, entry in list(self.entries.items())[:8]:
            lines.append(
                f"- {path} size={entry.get('size', 0)} sha={entry.get('sha256', '')} range={entry.get('last_read_range', '1:*')}"
            )
        return "\n".join(lines)
