"""
Attachment auto-injection system.

Generates contextual attachments that are injected as system-reminder
messages before each model turn. Mirrors Claude Code's utils/attachments.ts.

Attachment types:
  - file_change_delta: diffs of files modified since last turn
  - plan_mode_reminder: plan mode workflow instructions
  - todo_reminder: nudge to use todo_write_tool for complex tasks
  - agent_listing: available agent types for agent_tool
  - diagnostics: compiler/linter warnings from LSP
  - skill_recommendation: auto-surfaced relevant skills
"""

import os
import time
from typing import Any, Dict, List, Optional
try:
    import git as gitmodule
except ImportError:
    gitmodule = None


SYSTEM_REMINDER_OPEN = "<system-reminder>"
SYSTEM_REMINDER_CLOSE = "</system-reminder>"


def _wrap_reminder(content: str) -> str:
    return f"{SYSTEM_REMINDER_OPEN}\n{content}\n{SYSTEM_REMINDER_CLOSE}"


class AttachmentCollector:
    """
    Collects and formats contextual attachments to inject before each turn.
    Instantiated once per engine and refreshed at each turn boundary.
    """

    def __init__(self):
        self._last_file_snapshots: Dict[str, str] = {}
        self._last_turn_time: float = 0
        self._plan_mode_injected: bool = False
        self._todo_reminder_cooldown: int = 0
        self._recent_tool_names: List[str] = []
        self._skill_recommended_at_turn: int = -10
        self._skill_cache: Optional[List[dict]] = None

    def collect_attachments(
        self,
        *,
        cwd: str,
        plan_mode: str = "normal",
        plan_content: str = "",
        todo_summary: str = "",
        todo_count: int = 0,
        active_tools: List[str] = None,
        changed_files: Optional[List[str]] = None,
        lsp_diagnostics: str = "",
        turn_count: int = 0,
        recent_tool_names: Optional[List[str]] = None,
        user_query: str = "",
        skill_roots: Optional[List[str]] = None,
    ) -> List[dict]:
        """
        Build all attachment messages for the current turn.
        Returns a list of user-role messages with system-reminder content.
        """
        attachments = []

        diff_att = self._build_file_change_delta(cwd, changed_files)
        if diff_att:
            attachments.append(diff_att)

        plan_att = self._build_plan_mode_reminder(plan_mode, plan_content, turn_count)
        if plan_att:
            attachments.append(plan_att)

        todo_att = self._build_todo_reminder(
            todo_summary, todo_count, turn_count, recent_tool_names or []
        )
        if todo_att:
            attachments.append(todo_att)

        if lsp_diagnostics:
            attachments.append(self._build_diagnostics_attachment(lsp_diagnostics))

        skill_att = self._build_skill_recommendation(
            cwd, user_query, turn_count, skill_roots
        )
        if skill_att:
            attachments.append(skill_att)

        self._last_turn_time = time.time()
        return attachments

    def _build_file_change_delta(
        self, cwd: str, changed_files: Optional[List[str]]
    ) -> Optional[dict]:
        """Generate a diff summary for files that changed since the last turn."""
        if not changed_files:
            return None

        diffs = []
        for fpath in changed_files[:10]:
            abs_path = fpath if os.path.isabs(fpath) else os.path.join(cwd, fpath)
            if not os.path.isfile(abs_path):
                diffs.append(f"--- {fpath}\n(file deleted or moved)")
                if fpath in self._last_file_snapshots:
                    del self._last_file_snapshots[fpath]
                continue

            try:
                with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                    current = f.read(50000)
            except Exception:
                continue

            prev = self._last_file_snapshots.get(fpath)
            if prev is not None and prev == current:
                continue

            self._last_file_snapshots[fpath] = current

            if prev is None:
                diffs.append(f"--- {fpath}\n(file newly tracked, {len(current)} chars)")
            else:
                diff_text = self._simple_diff(prev, current, fpath)
                if diff_text:
                    diffs.append(diff_text)

        if not diffs:
            return None

        content = _wrap_reminder(
            "Files changed since your last turn:\n\n" + "\n\n".join(diffs[:5])
        )
        return {"role": "user", "content": content, "_is_meta": True, "_attachment_type": "file_change_delta"}

    def _simple_diff(self, old: str, new: str, path: str, context: int = 3) -> str:
        """Generate a minimal unified-style diff summary."""
        old_lines = old.splitlines(keepends=True)
        new_lines = new.splitlines(keepends=True)

        if old_lines == new_lines:
            return ""

        try:
            import difflib
            diff = list(difflib.unified_diff(
                old_lines, new_lines,
                fromfile=f"a/{path}", tofile=f"b/{path}",
                n=context,
            ))
        except Exception:
            return f"--- {path}\n(changed, diff unavailable)"

        if not diff:
            return ""

        diff_text = "".join(diff[:100])
        if len(diff) > 100:
            diff_text += f"\n... ({len(diff) - 100} more diff lines truncated)"
        return diff_text

    def _build_plan_mode_reminder(
        self, plan_mode: str, plan_content: str, turn_count: int
    ) -> Optional[dict]:
        """Inject plan mode instructions when entering plan mode."""
        if plan_mode != "plan":
            self._plan_mode_injected = False
            return None

        lines = [
            "You are currently in PLAN MODE. In this mode:",
            "1. You can ONLY use read-only tools (file_read_tool, grep_tool, glob_tool) to explore the codebase.",
            "2. You CANNOT modify files (file_edit_tool, file_write_tool, bash_tool are blocked).",
            "3. Use plan_tool with action='write' to create your implementation plan.",
            "4. Use ask_user_question_tool to clarify requirements.",
            "5. When your plan is ready, the user will approve it and you'll exit plan mode to implement.",
        ]
        if plan_content:
            lines.append(f"\nCurrent plan draft ({len(plan_content)} chars):\n{plan_content[:500]}")

        content = _wrap_reminder("\n".join(lines))
        return {"role": "user", "content": content, "_is_meta": True, "_attachment_type": "plan_mode_reminder"}

    def _build_todo_reminder(
        self,
        todo_summary: str,
        todo_count: int,
        turn_count: int,
        recent_tool_names: List[str],
    ) -> Optional[dict]:
        """Nudge the model to use todo_write_tool for complex multi-step tasks."""
        if self._todo_reminder_cooldown > 0:
            self._todo_reminder_cooldown -= 1
            return None

        if todo_count > 0:
            return None

        if turn_count < 3:
            return None

        has_used_todo = "todo_write_tool" in recent_tool_names
        if has_used_todo:
            return None

        multi_step_signals = sum(1 for t in recent_tool_names[-10:] if t in {
            "file_edit_tool", "file_write_tool", "bash_tool", "file_read_tool"
        })
        if multi_step_signals < 3:
            return None

        self._todo_reminder_cooldown = 5

        content = _wrap_reminder(
            "You are working on a multi-step task but have not created a todo list. "
            "Consider using todo_write_tool to track your progress — this helps the user "
            "understand what you're doing and ensures you don't miss steps."
        )
        return {"role": "user", "content": content, "_is_meta": True, "_attachment_type": "todo_reminder"}

    def _build_skill_recommendation(
        self,
        cwd: str,
        user_query: str,
        turn_count: int,
        skill_roots: Optional[List[str]] = None,
    ) -> Optional[dict]:
        """
        Surface relevant skills based on the user's query keywords.
        Only triggers once every 8 turns to avoid spam.
        """
        if not user_query or turn_count - self._skill_recommended_at_turn < 8:
            return None

        if self._skill_cache is None:
            self._skill_cache = self._discover_available_skills(cwd, skill_roots)

        if not self._skill_cache:
            return None

        query_lower = user_query.lower()
        query_words = set(query_lower.split())
        relevant = []
        for skill in self._skill_cache:
            name_lower = skill["name"].lower()
            keywords = set(skill.get("keywords", []))
            if name_lower in query_lower:
                relevant.append(skill)
            elif query_words & keywords:
                relevant.append(skill)

        if not relevant:
            return None

        self._skill_recommended_at_turn = turn_count

        lines = [
            "Relevant skills detected for the current task. "
            "Consider using `skill_tool` with action='read' to load them:",
        ]
        for s in relevant[:5]:
            lines.append(f"  - {s['name']}: {s['path']}")

        content = _wrap_reminder("\n".join(lines))
        return {"role": "user", "content": content, "_is_meta": True, "_attachment_type": "skill_recommendation"}

    def _discover_available_skills(
        self, cwd: str, extra_roots: Optional[List[str]] = None
    ) -> List[dict]:
        """Scan filesystem for SKILL.md files and extract keywords."""
        from pathlib import Path

        roots = []
        if extra_roots:
            for r in extra_roots:
                p = Path(r)
                if p.is_dir():
                    roots.append(p)

        cwd_path = Path(cwd)
        roots.extend([
            cwd_path / ".codeclaw" / "skills",
            cwd_path / ".claude" / "skills",
            Path.home() / ".codeclaw" / "skills",
            Path.home() / ".claude" / "skills",
        ])

        env_paths = os.environ.get("CODECLAW_SKILLS_PATHS", "")
        for item in env_paths.split(os.pathsep):
            item = item.strip()
            if item:
                roots.append(Path(item))

        discovered = []
        seen = set()
        for root in roots:
            if not root.exists() or not root.is_dir():
                continue
            for current_root, _dirs, filenames in os.walk(root):
                if "SKILL.md" not in filenames:
                    continue
                skill_path = os.path.join(current_root, "SKILL.md")
                abs_path = os.path.abspath(skill_path)
                if abs_path in seen:
                    continue
                seen.add(abs_path)
                skill_name = os.path.basename(current_root)
                keywords = self._extract_skill_keywords(abs_path, skill_name)
                discovered.append({
                    "name": skill_name,
                    "path": abs_path,
                    "keywords": keywords,
                })

        return discovered

    @staticmethod
    def _extract_skill_keywords(skill_path: str, skill_name: str) -> List[str]:
        """Extract searchable keywords from a SKILL.md file header."""
        keywords = set()
        for word in skill_name.lower().replace("-", " ").replace("_", " ").split():
            if len(word) >= 3:
                keywords.add(word)
        try:
            with open(skill_path, "r", encoding="utf-8", errors="replace") as f:
                header = f.read(500).lower()
            for word in header.split():
                cleaned = word.strip(".,;:!?()[]{}#*`\"'")
                if len(cleaned) >= 4 and cleaned.isalpha():
                    keywords.add(cleaned)
        except Exception:
            pass
        return list(keywords)[:20]

    def _build_diagnostics_attachment(self, diagnostics: str) -> dict:
        content = _wrap_reminder(
            f"Compiler/linter diagnostics detected:\n{diagnostics}"
        )
        return {"role": "user", "content": content, "_is_meta": True, "_attachment_type": "diagnostics"}

    def track_tool_use(self, tool_name: str):
        """Record tool usage for reminder heuristics."""
        self._recent_tool_names.append(tool_name)
        if len(self._recent_tool_names) > 50:
            self._recent_tool_names = self._recent_tool_names[-30:]

    def snapshot_file(self, path: str, content: str):
        """Record a file snapshot for change tracking."""
        self._last_file_snapshots[path] = content

    def get_changed_files_from_git(self, cwd: str) -> List[str]:
        """Get list of changed files from git status."""
        if gitmodule is None:
            return []
        try:
            repo = gitmodule.Repo(cwd, search_parent_directories=True)
            changed = []
            for item in repo.index.diff(None):
                changed.append(item.a_path)
            for item in repo.index.diff("HEAD"):
                changed.append(item.a_path)
            for upath in repo.untracked_files[:10]:
                changed.append(upath)
            return list(set(changed))[:15]
        except Exception:
            return []

    def export_state(self) -> dict:
        return {
            "last_turn_time": self._last_turn_time,
            "todo_reminder_cooldown": self._todo_reminder_cooldown,
            "tracked_file_count": len(self._last_file_snapshots),
            "recent_tool_count": len(self._recent_tool_names),
        }
