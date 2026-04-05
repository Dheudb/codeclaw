from pydantic import BaseModel, Field

from codeclaw.context.tool_results import build_tool_result
from codeclaw.tools.base import BaseAgenticTool


class ToolSearchToolInput(BaseModel):
    query: str = Field(
        ...,
        description="Describe the capability you need, such as browser automation, web search, notebook editing, or language-server diagnostics.",
    )
    activate: list[str] = Field(
        default_factory=list,
        description="Optional list of matching latent tool names to activate for the current session.",
    )


class ToolSearchTool(BaseAgenticTool):
    name = "tool_search_tool"
    description = "Searches the tool catalog and optionally activates specialized tools for the current session."
    input_schema = ToolSearchToolInput
    is_read_only = True
    risk_level = "low"

    def prompt(self) -> str:
        return (
            "If you need a specialized capability that is not currently exposed as an active tool, "
            "use `tool_search_tool` to discover and activate matching tools first."
        )

    async def execute(self, query: str, activate: list[str] = None) -> dict:
        activate = activate or []
        query_lower = (query or "").strip().lower()
        active_tools = self.context.get("engine_available_tools", {}) or {}
        latent_tools = self.context.get("latent_tools_registry", {}) or {}
        activate_tools = self.context.get("activate_tools")

        catalog = {}
        for name, tool in active_tools.items():
            catalog[name] = {
                "name": name,
                "description": getattr(tool, "description", ""),
                "prompt": tool.prompt() if callable(getattr(tool, "prompt", None)) else "",
                "active": True,
            }
        for name, tool in latent_tools.items():
            catalog[name] = {
                "name": name,
                "description": getattr(tool, "description", ""),
                "prompt": tool.prompt() if callable(getattr(tool, "prompt", None)) else "",
                "active": False,
            }

        matches = []
        for item in catalog.values():
            haystack = " ".join([
                item.get("name", ""),
                item.get("description", ""),
                item.get("prompt", ""),
            ]).lower()
            if not query_lower or query_lower in haystack:
                matches.append(item)

        matches = sorted(matches, key=lambda item: (item.get("active") is False, item.get("name")))

        activated = []
        if activate and callable(activate_tools):
            activated = activate_tools(activate)

        lines = [f"Matched tools for query '{query}':"]
        if matches:
            for item in matches[:12]:
                state = "active" if item.get("active") else "latent"
                lines.append(f"- {item['name']} [{state}] {item['description']}")
        else:
            lines.append("- No matching tools found.")

        if activated:
            lines.append("")
            lines.append("Activated tools:")
            for name in activated:
                lines.append(f"- {name}")

        return build_tool_result(
            ok=True,
            content="\n".join(lines),
            metadata={
                "query": query,
                "match_count": len(matches),
                "activated": activated,
                "active_tool_count": len(active_tools),
                "latent_tool_count": len(latent_tools),
            },
        )
