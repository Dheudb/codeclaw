"""Load custom subagent definitions without conflating them with skills."""
import yaml
from codeclaw.agents.builtin_agents import BUILTIN_AGENTS
from codeclaw.skills.registry import SkillRegistry, read_frontmatter

TOOL_ALIASES = {"Read": "file_read_tool", "Write": "file_write_tool", "Edit": "file_edit_tool", "Bash": "bash_tool", "Grep": "grep_tool", "Glob": "glob_tool", "Skill": "skill_tool"}


def tool_list(value, aliases=True):
    if value is None:
        return ["*"]
    if isinstance(value, str):
        value = [item.strip() for item in value.split(",") if item.strip()]
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ValueError("tools and skills must be lists or comma-separated strings")
    return [TOOL_ALIASES.get(v, v) if aliases else v for v in value]


class AgentRegistry:
    def __init__(self, cwd=None, home=None):
        self.skills = SkillRegistry(cwd, home)

    def discover(self):
        entries, names = [], set()
        for root, scope in self.skills.roots("agents"):
            if not root.is_dir():
                continue
            for path in sorted(root.glob("*.md")):
                entry = {"name": path.stem, "path": str(path.resolve()), "scope": scope, "valid": True, "error": "", "shadowed": False}
                try:
                    metadata, body = read_frontmatter(path)
                    name, description = metadata.get("name"), metadata.get("description")
                    if not isinstance(name, str) or not name.strip() or not isinstance(description, str) or not description.strip() or not body:
                        raise ValueError("name, description and a prompt body are required")
                    if metadata.get("model") is not None and not isinstance(metadata["model"], str):
                        raise ValueError("model must be a string")
                    if metadata.get("model") == "inherit":
                        metadata["model"] = None
                    entry.update(name=name, description=description, system_prompt=body, model=metadata.get("model"), allowed_tools=tool_list(metadata.get("tools")), skills=tool_list(metadata.get("skills", []), aliases=False), agent_type=name, agent_role="custom", disallowed_tools=set())
                    if name in BUILTIN_AGENTS:
                        raise ValueError("Custom agent name conflicts with a built-in agent")
                    if any("(" in tool for tool in entry["allowed_tools"]):
                        raise ValueError("Per-command tool patterns are unsupported; use exact tool names")
                    for skill_name in entry["skills"]:
                        self.skills.resolve(skill_name)
                    entry["shadowed"] = name in names
                    names.add(name)
                except (OSError, UnicodeError, ValueError, yaml.YAMLError) as exc:
                    entry.update(valid=False, error=str(exc))
                entries.append(entry)
        return entries

    def definitions(self):
        result = {name: dict(value) for name, value in BUILTIN_AGENTS.items()}
        result.update({e["name"]: e for e in self.discover() if e["valid"] and not e["shadowed"]})
        return result

    def resolve(self, name):
        definitions = self.definitions()
        if name not in definitions:
            raise ValueError("Unknown or invalid subagent: " + str(name))
        return definitions[name]
