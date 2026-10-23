from pydantic import BaseModel, Field
from codeclaw.context.tool_results import build_tool_result
from codeclaw.skills.registry import SkillRegistry
from codeclaw.tools.base import BaseAgenticTool


class SkillToolInput(BaseModel):
    action: str = Field("list", description="list, search, read, or resource")
    skill_name: str = Field(None, description="Exact name of an enabled skill")
    query: str = Field(None, description="Search names and descriptions")
    resource_path: str = Field(None, description="For resource: path relative to the skill directory")
    max_chars: int = Field(100000, ge=1, le=1000000)


class SkillTool(BaseAgenticTool):
    name = "skill_tool"
    description = "Discover and load enabled Agent Skills and their referenced text resources."
    input_schema = SkillToolInput
    is_read_only = True
    risk_level = "low"

    def registry(self):
        return self.context.get("skill_registry") or SkillRegistry(self.context.get("session_cwd"))

    def prompt(self):
        return ("Available skills (metadata only):\n" + self.registry().catalog() +
                "\nWhen the user names a skill or its description matches the task, use skill_tool action=read "
                "before performing that workflow. Load referenced resources on demand. Disabled skills are unavailable.")

    async def execute(self, action="list", skill_name=None, query=None, resource_path=None, max_chars=100000):
        try:
            registry = self.registry()
            if action in {"list", "search"}:
                items = registry.active()
                if action == "search":
                    items = [s for s in items if (query or "").lower() in (s["name"] + " " + s["description"]).lower()]
                text = "\n".join("{}: {}\n{}".format(s["name"], s["description"], s["path"]) for s in items) or "No enabled skills found."
            elif action == "read":
                text = registry.read(skill_name)
                completion = self.context.get("workflow_completion")
                if completion:
                    completion.activate(registry.resolve(skill_name))
                loaded = self.context.setdefault("loaded_skills", [])
                if skill_name not in loaded:
                    loaded.append(skill_name)
                callback = self.context.get("capability_event")
                if callback:
                    callback({"type": "skill_loaded", "name": skill_name})
            elif action == "resource":
                if not resource_path:
                    raise ValueError("resource_path is required")
                text = registry.read_resource(skill_name, resource_path)
            else:
                raise ValueError("Unknown skill action")
            return build_tool_result(ok=True, content=text[:max_chars], metadata={"action": action, "truncated": len(text) > max_chars})
        except (OSError, UnicodeError, ValueError) as exc:
            return build_tool_result(ok=False, content=str(exc), metadata={"action": action}, is_error=True)
