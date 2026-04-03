import hashlib
import os
from pathlib import Path


class MemoryFileManager:
    def __init__(self, cwd: str = None, home: str = None):
        self.cwd = os.path.abspath(cwd or os.getcwd())
        self.home = os.path.abspath(home or str(Path.home()))
        self.loaded_files = []

    def set_cwd(self, cwd: str):
        self.cwd = os.path.abspath(cwd or os.getcwd())

    def _iter_parent_directories(self):
        current = Path(self.cwd)
        directories = [current]
        directories.extend(current.parents)
        return directories

    def _candidate_paths(self):
        seen = set()
        candidates = []

        global_candidates = [
            ("global", Path(self.home) / ".claude" / "CLAUDE.md"),
            ("global", Path(self.home) / "CLAUDE.md"),
        ]
        for scope, path in global_candidates:
            normalized = str(path.resolve()) if path.exists() else str(path)
            if normalized in seen:
                continue
            seen.add(normalized)
            candidates.append((scope, path))

        project_candidates = []
        for directory in reversed(self._iter_parent_directories()):
            project_candidates.append(("project", directory / ".claude" / "CLAUDE.md"))
            project_candidates.append(("project", directory / "CLAUDE.md"))

        for scope, path in project_candidates:
            normalized = str(path.resolve()) if path.exists() else str(path)
            if normalized in seen:
                continue
            seen.add(normalized)
            candidates.append((scope, path))

        return candidates

    def refresh(self, *, max_chars_per_file: int = 6000, max_total_chars: int = 12000):
        loaded = []
        consumed_chars = 0

        for scope, path in self._candidate_paths():
            if not path.exists() or not path.is_file():
                continue

            try:
                raw_text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                try:
                    raw_text = path.read_text(encoding="utf-8-sig")
                except Exception:
                    continue
            except Exception:
                continue

            content = raw_text.strip()
            if not content:
                continue

            truncated = False
            if len(content) > max_chars_per_file:
                content = content[:max_chars_per_file].rstrip()
                truncated = True

            remaining = max_total_chars - consumed_chars
            if remaining <= 0:
                break
            if len(content) > remaining:
                content = content[:remaining].rstrip()
                truncated = True

            if not content:
                break

            loaded.append({
                "scope": scope,
                "path": str(path),
                "content": content,
                "chars": len(content),
                "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest()[:16],
                "truncated": truncated,
            })
            consumed_chars += len(content)

        self.loaded_files = loaded
        return loaded

    def render_prompt_summary(self) -> str:
        if not self.loaded_files:
            return ""

        lines = [
            "--- MEMORY FILES ---",
            "Apply these persistent instructions in order; later project-level files are more specific and override earlier global guidance when they conflict.",
        ]

        for item in self.loaded_files:
            scope = item.get("scope", "unknown").upper()
            path = item.get("path", "")
            lines.append(f"[{scope}] {path}")
            lines.append(item.get("content", ""))
            if item.get("truncated"):
                lines.append("[Truncated to fit prompt budget.]")
            lines.append("")

        return "\n".join(lines).strip()

    def export_state(self):
        return {
            "cwd": self.cwd,
            "loaded_files": [
                {
                    "scope": item.get("scope"),
                    "path": item.get("path"),
                    "chars": item.get("chars"),
                    "sha256": item.get("sha256"),
                    "truncated": item.get("truncated", False),
                }
                for item in self.loaded_files
            ],
        }

    def load_state(self, payload):
        payload = payload or {}
        cwd = payload.get("cwd")
        if cwd:
            self.cwd = os.path.abspath(cwd)
        self.loaded_files = list(payload.get("loaded_files", []) or [])
