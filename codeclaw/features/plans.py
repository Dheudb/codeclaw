import os
import hashlib


def _make_plan_slug(session_id: str) -> str:
    """Derive a short, filesystem-safe slug from the session UUID."""
    digest = hashlib.sha256(session_id.encode()).hexdigest()[:8]
    return f"plan-{digest}"


def get_plans_directory() -> str:
    """Return the directory where plan files are stored."""
    return os.path.join(os.path.expanduser("~"), ".codeclaw", "plans")


def get_plan_file_path(session_id: str, agent_id: str = None) -> str:
    """
    Return the absolute path to the plan file for a session.
    Mirrors TS: .claude/plans/{slug}.md  →  ~/.codeclaw/plans/{slug}.md
    """
    slug = _make_plan_slug(session_id)
    plans_dir = get_plans_directory()
    if agent_id:
        return os.path.join(plans_dir, f"{slug}-agent-{agent_id}.md")
    return os.path.join(plans_dir, f"{slug}.md")


def is_session_plan_file(abs_path: str, session_id: str) -> bool:
    """Check whether *abs_path* is the plan file for this session."""
    try:
        expected = os.path.normpath(get_plan_file_path(session_id))
        return os.path.normpath(abs_path) == expected
    except Exception:
        return False


class PlanManager:
    """
    Session-scoped planning state backed by a real .md file on disk.

    The plan file lives at  ~/.codeclaw/plans/{slug}.md  and can be
    written/edited by the model via file_write_tool / file_edit_tool in
    plan mode (auto-allowed by the permission system).
    """

    def __init__(self, session_id: str = "default", agent_id: str = None):
        self._session_id = session_id
        self._agent_id = agent_id
        self.mode = "normal"
        self._pre_plan_mode: str = "normal"

    @property
    def plan_file_path(self) -> str:
        return get_plan_file_path(self._session_id, self._agent_id)

    @property
    def plan_exists(self) -> bool:
        return os.path.isfile(self.plan_file_path)

    # ---- mode management ----

    def set_mode(self, mode: str) -> str:
        normalized = str(mode).strip().lower()
        if normalized not in {"normal", "plan"}:
            return f"Error: Unsupported mode '{mode}'."
        if normalized == "plan" and self.mode != "plan":
            self._pre_plan_mode = self.mode
        self.mode = normalized
        return self.mode

    def get_mode(self) -> str:
        return self.mode

    @property
    def pre_plan_mode(self) -> str:
        return self._pre_plan_mode

    # ---- plan content (backed by disk) ----

    def get_plan(self) -> str:
        """Read plan content from the file on disk."""
        try:
            with open(self.plan_file_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return ""
        except Exception:
            return ""

    def write_plan(self, content: str) -> str:
        """Overwrite the plan file."""
        path = self.plan_file_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content or "")
        return "Plan content updated."

    def append_plan(self, content: str) -> str:
        existing = self.get_plan()
        chunk = content or ""
        new_content = (existing + "\n" + chunk) if existing and chunk else (existing + chunk)
        return self.write_plan(new_content)

    def clear_plan(self) -> str:
        path = self.plan_file_path
        if os.path.exists(path):
            os.remove(path)
        return "Plan content cleared."

    # ---- serialization ----

    def load(self, mode: str = "normal", content: str = ""):
        self.mode = mode if mode in {"normal", "plan"} else "normal"
        if content:
            self.write_plan(content)

    def export(self) -> dict:
        return {
            "mode": self.mode,
            "content": self.get_plan(),
        }

    def render_prompt_summary(self) -> str:
        lines = [
            "--- SESSION MODE ---",
            f"Current mode: {self.mode}",
        ]

        if self.mode == "plan":
            pfp = self.plan_file_path
            lines.extend([
                "",
                "## PLAN MODE IS ACTIVE — MANDATORY CONSTRAINTS",
                "",
                "The user indicated that they do NOT want you to execute yet. "
                "You MUST NOT make any edits (with the sole exception of the plan file listed below), "
                "run any non-readonly tools (no bash, no file writes outside the plan file, no commits), "
                "or otherwise mutate the system. This supersedes any other instructions.",
                "",
                f"Plan file (the ONLY file you may write/edit): {pfp}",
                "",
                "### Plan Workflow",
                "Phase 1 – Initial Understanding: Read code and ask clarifying questions. "
                "Use read-only tools (file_read_tool, grep_tool, glob_tool, agent_tool with Explore type).",
                "Phase 2 – Design: Synthesize findings into an implementation approach.",
                "Phase 3 – Review: Verify alignment with the user's request; use ask_user_question_tool for remaining questions.",
                f"Phase 4 – Final Plan: Write the plan to {pfp} using file_write_tool (new) or file_edit_tool (update). "
                "Include a Context section, recommended approach, file paths, reuse opportunities, and a verification section.",
                "Phase 5 – Call exit_plan_mode: When your plan is ready, call exit_plan_mode. "
                "Your turn MUST end with either ask_user_question_tool (for clarifications) or exit_plan_mode (for plan approval). "
                "Do NOT end your turn with plain text — always use one of these two tools.",
                "",
                "IMPORTANT: Do NOT answer the user's question directly. Do NOT write code files. "
                "Your sole job right now is to PLAN.",
            ])

        plan_content = self.get_plan()
        if plan_content:
            plan_preview = plan_content
            if len(plan_preview) > 4000:
                plan_preview = plan_preview[:4000] + "\n...[plan truncated]..."
            lines.append("--- CURRENT PLAN ---")
            lines.append(plan_preview)

        return "\n".join(lines)
