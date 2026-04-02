import os


class PlanManager:
    """
    Session-scoped planning state with lightweight persistence to `.codeclaw/plan.md`.
    """

    def __init__(self, plan_path=".codeclaw/plan.md"):
        self.plan_path = plan_path
        self.mode = "normal"
        self.plan_content = ""

    def set_mode(self, mode: str) -> str:
        normalized = str(mode).strip().lower()
        if normalized not in {"normal", "plan"}:
            return f"Error: Unsupported mode '{mode}'."
        self.mode = normalized
        return self.mode

    def get_mode(self) -> str:
        return self.mode

    def write_plan(self, content: str) -> str:
        self.plan_content = content or ""
        self._flush_to_disk()
        return "Plan content updated."

    def append_plan(self, content: str) -> str:
        chunk = content or ""
        if self.plan_content and chunk:
            self.plan_content += "\n" + chunk
        else:
            self.plan_content += chunk
        self._flush_to_disk()
        return "Plan content appended."

    def clear_plan(self) -> str:
        self.plan_content = ""
        self._flush_to_disk()
        return "Plan content cleared."

    def get_plan(self) -> str:
        return self.plan_content

    def load(self, mode: str = "normal", content: str = ""):
        self.mode = mode if mode in {"normal", "plan"} else "normal"
        self.plan_content = content or ""
        self._flush_to_disk()

    def export(self) -> dict:
        return {
            "mode": self.mode,
            "content": self.plan_content,
        }

    def render_prompt_summary(self) -> str:
        lines = [
            "--- SESSION MODE ---",
            f"Current mode: {self.mode}",
        ]

        if self.mode == "plan":
            lines.append("Plan mode is active. Avoid mutating the workspace. Focus on analysis, planning, todos, and writing/updating the plan.")

        if self.plan_content:
            plan_preview = self.plan_content
            if len(plan_preview) > 4000:
                plan_preview = plan_preview[:4000] + "\n...[plan truncated]..."
            lines.append("--- CURRENT PLAN ---")
            lines.append(plan_preview)

        return "\n".join(lines)

    def _flush_to_disk(self):
        os.makedirs(os.path.dirname(self.plan_path), exist_ok=True)
        with open(self.plan_path, "w", encoding="utf-8") as f:
            f.write(self.plan_content)
