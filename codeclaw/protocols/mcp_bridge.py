import os
import json
import asyncio
from contextlib import AsyncExitStack
from typing import Dict, Any, List

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

class MCPToolWrapper:
    """
    Wraps an official remote MCP Tool to look and feel exactly like a local CodeClaw BaseAgenticTool,
    transparently forwarding payload validation limits to Anthropic.
    """
    def __init__(self, server_name: str, session, mcp_tool):
        # We prefix the tool name to avoid collisions, e.g., "sqlite_query"
        self.name = f"{server_name}_{mcp_tool.name}"
        self.description = mcp_tool.description or f"MCP Server tool '{mcp_tool.name}' offloaded via {server_name}"
        self.mcp_tool_name = mcp_tool.name
        self.session = session
        self.raw_schema = mcp_tool.inputSchema
        self.context = {} # Shared context injection point
        self.is_read_only = False
        self.risk_level = "high"
        
    def get_tool_definition(self) -> Dict[str, Any]:
        """Bypass Pydantic logic and directly vend the raw remote schema."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.raw_schema
        }

    def is_read_only_call(self, **kwargs):
        return self.is_read_only

    def is_concurrency_safe_call(self, **kwargs):
        return self.is_read_only_call(**kwargs)

    def get_risk_level(self, **kwargs):
        return self.risk_level

    def build_permission_summary(self, **kwargs):
        preview_lines = []
        for key, value in list(kwargs.items())[:4]:
            rendered = str(value)
            if len(rendered) > 160:
                rendered = rendered[:157] + "..."
            preview_lines.append(f"{key}: {rendered}")

        summary = f"Remote MCP tool '{self.name}' requested."
        if preview_lines:
            summary += "\n" + "\n".join(preview_lines)
        return summary
        
    async def __call__(self, **kwargs):
        """Invoke the remote tool via the established STDIO ClientSession."""
        try:
            result = await self.session.call_tool(self.mcp_tool_name, arguments=kwargs)
            # Result has content block array (e.g. result.content = [TextContent(...)])
            text_blocks = []
            for c in result.content:
                if getattr(c, "type", "text") == "text":
                    text_blocks.append(c.text)
                else:
                    text_blocks.append(f"[{c.type} block hidden]")
            return "\\n".join(text_blocks)
        except Exception as e:
            return f"Error executing remote MCP tool '{self.name}': {str(e)}"

class MCPBridge:
    """
    Managers the asynchronous lifecycle of multiple STDIO-based Model Context Protocol sessions
    by reading local mcp_servers.json configuration files.
    """
    def __init__(self, config_file="mcp_servers.json"):
        self.config_file = config_file
        self.sessions = {}
        self.exit_stack = AsyncExitStack()
        
    async def load_and_connect(self) -> Dict[str, MCPToolWrapper]:
        if not MCP_AVAILABLE:
            return {}
            
        if not os.path.exists(self.config_file):
            return {}
            
        discovered_tools = {}
        try:
            with open(self.config_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            servers = data.get("mcpServers", {})
            for name, config in servers.items():
                cmd = config.get("command")
                args = config.get("args", [])
                env = os.environ.copy()
                if "env" in config:
                    env.update(config["env"])
                    
                server_params = StdioServerParameters(command=cmd, args=args, env=env)
                
                try:
                    # Safely bind lifecycle to our stack
                    stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
                    read, write = stdio_transport
                    session = await self.exit_stack.enter_async_context(ClientSession(read, write))
                    
                    await session.initialize()
                    self.sessions[name] = session
                    
                    # Interrogate its tool capabilities!
                    result = await session.list_tools()
                    for t in result.tools:
                        wrapper = MCPToolWrapper(name, session, t)
                        discovered_tools[wrapper.name] = wrapper
                        
                except Exception as e:
                    print(f"\\n[MCP Warning] Failed to connect to local MCP Server '{name}': {e}")
                    
        except Exception as e:
            print(f"\\n[MCP Warning] Failed to parse {self.config_file}: {e}")
            
        return discovered_tools
        
    async def cleanup(self):
        await self.exit_stack.aclose()
