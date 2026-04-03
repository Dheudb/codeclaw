import os
from pathlib import Path

from pydantic import BaseModel, Field

from codeclaw.core.tool_results import build_tool_result
from codeclaw.tools.base import BaseAgenticTool


class SkillToolInput(BaseModel):
    action: str = Field(
        "list",
        description="One of: list, search, read. `read` returns the full skill content.",
    )
    skill_name: str = Field(None, description="Skill name to read. Usually the folder name that contains SKILL.md.")
    query: str = Field(None, description="Optional search query for list/search.")
    max_chars: int = Field(8000, description="Maximum characters to return when reading a skill.")


class SkillTool(BaseAgenticTool):
    name = "skill_tool"
    description = "Discovers and reads local SKILL.md instruction bundles from standard skill directories. Use this when you want reusable capability guidance instead of re-deriving it from scratch."
    input_schema = SkillToolInput
    is_read_only = True
    risk_level = "low"

    def prompt(self) -> str:
        return (
            "Use `skill_tool` to discover or load reusable local skills from SKILL.md files. "
            "Prefer `search` or `list` before concluding a specialized workflow is unavailable."
        )

    def validate_input(
        self,
        action: str = "list",
        skill_name: str = None,
        query: str = None,
        max_chars: int = 8000,
    ):
        if action not in {"list", "search", "read"}:
            return "action must be one of: list, search, read."
        if action == "read" and not str(skill_name or "").strip():
            return "skill_name is required when action='read'."
        if int(max_chars or 0) <= 0:
            return "max_chars must be greater than 0."
        return None

    def _candidate_roots(self):
        cwd = Path(os.getcwd())
        roots = []
        env_paths = os.environ.get("CODECLAW_SKILLS_PATHS", "")
        for item in env_paths.split(os.pathsep):
            item = item.strip()
            if item:
                roots.append(Path(item))
        roots.extend([
            cwd / ".codeclaw" / "skills",
            cwd / ".claude" / "skills",
            Path.home() / ".codeclaw" / "skills",
            Path.home() / ".claude" / "skills",
        ])
        seen = set()
        normalized = []
        for root in roots:
            rendered = str(root)
            if rendered in seen:
                continue
            seen.add(rendered)
            normalized.append(root)
        return normalized

    def _discover_skills(self):
        discovered = []
        seen = set()
        for root in self._candidate_roots():
            if not root.exists() or not root.is_dir():
                continue
            for current_root, dirnames, filenames in os.walk(root):
                if "SKILL.md" not in filenames:
                    continue
                skill_path = Path(current_root) / "SKILL.md"
                abs_skill_path = str(skill_path.resolve())
                if abs_skill_path in seen:
                    continue
                seen.add(abs_skill_path)
                skill_dir = skill_path.parent
                discovered.append({
                    "name": skill_dir.name,
                    "path": abs_skill_path,
                    "root": str(root.resolve()),
                })
        discovered.sort(key=lambda item: (item.get("name", ""), item.get("path", "")))
        return discovered

    def _match_skill(self, skill_name: str, discovered):
        requested = str(skill_name or "").strip().lower()
        exact_matches = [
            item for item in discovered
            if item.get("name", "").lower() == requested
        ]
        if exact_matches:
            return exact_matches[0]
        for item in discovered:
            if requested in item.get("path", "").lower():
                return item
        return None

    async def execute(
        self,
        action: str = "list",
        skill_name: str = None,
        query: str = None,
        max_chars: int = 8000,
    ) -> dict:
        discovered = self._discover_skills()
        query_lower = str(query or "").strip().lower()

        if action in {"list", "search"}:
            filtered = discovered
            if action == "search" and query_lower:
                filtered = [
                    item for item in discovered
                    if query_lower in item.get("name", "").lower()
                    or query_lower in item.get("path", "").lower()
                ]
            lines = [f"Discovered skills: {len(filtered)}"]
            for item in filtered[:40]:
                lines.append(f"- {item['name']}: {item['path']}")
            if not filtered:
                lines.append("- No matching skills found.")
            return build_tool_result(
                ok=True,
                content="\n".join(lines),
                metadata={
                    "action": action,
                    "query": query,
                    "skill_count": len(filtered),
                    "search_roots": [str(root) for root in self._candidate_roots()],
                },
            )

        matched = self._match_skill(skill_name, discovered)
        if not matched:
            return build_tool_result(
                ok=False,
                content=f"Skill '{skill_name}' was not found.",
                metadata={
                    "action": action,
                    "skill_name": skill_name,
                    "available_skills": [item.get("name") for item in discovered[:20]],
                },
                is_error=True,
            )

        try:
            raw = Path(matched["path"]).read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return build_tool_result(
                ok=False,
                content=f"Skill '{skill_name}' could not be decoded as UTF-8.",
                metadata={"action": action, "skill_name": skill_name, "path": matched["path"]},
                is_error=True,
            )
        except Exception as e:
            return build_tool_result(
                ok=False,
                content=f"Error reading skill '{skill_name}': {str(e)}",
                metadata={"action": action, "skill_name": skill_name, "path": matched["path"]},
                is_error=True,
            )

        content = raw.strip()
        truncated = False
        if len(content) > int(max_chars):
            content = content[: int(max_chars)].rstrip()
            truncated = True

        return build_tool_result(
            ok=True,
            content=content,
            metadata={
                "action": action,
                "skill_name": matched["name"],
                "path": matched["path"],
                "truncated": truncated,
                "available_skill_count": len(discovered),
            },
        )
