import json
import os
from typing import Tuple, Optional
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from codeclaw.core.config import load_config

class IdleBot:
    def __init__(self):
        self.conversations = {} # type: dict[str, list]
        
        cfg = load_config()
        self.provider = cfg.get("model_provider", "openai")
        
        self.openai_base_url = os.environ.get("OPENAI_BASE_URL") or cfg.get("openai_base_url")
        self.openai_api_key = os.environ.get("OPENAI_API_KEY") or cfg.get("openai_api_key") or "empty"
        self.openai_model = os.environ.get("OPENAI_MODEL") or cfg.get("openai_model") or "gpt-4o-mini"
        
        self.anthropic_base_url = os.environ.get("ANTHROPIC_BASE_URL") or cfg.get("anthropic_base_url")
        self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY") or cfg.get("anthropic_api_key") or "empty"
        self.anthropic_model = os.environ.get("ANTHROPIC_MODEL") or cfg.get("anthropic_model") or "claude-3-5-haiku-20241022"

        if self.provider == "anthropic":
            self.client = AsyncAnthropic(api_key=self.anthropic_api_key, base_url=self.anthropic_base_url)
        else:
            self.client = AsyncOpenAI(api_key=self.openai_api_key, base_url=self.openai_base_url)
        
        self.system_prompt = (
            "You are CodeClaw Gateway, an intelligent assistant inside Feishu. "
            "You help the user with daily tasks. If the user wants to debug, write code, or explore a local project repository, "
            "you MUST use the `mount_codeclaw` tool to specify the absolute path of the directory they want to work on. "
            "If the user did not specify a full path (e.g. they only said 'claude-code'), respond directly asking them for the exact absolute path on their host machine (e.g. '请问绝对路径是什么？')."
        )
        
        self.tools_openai = [
            {
                "type": "function",
                "function": {
                    "name": "mount_codeclaw",
                    "description": "Start the full CodeClaw Agentic UI and mount a local directory for deep debugging/coding.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "project_path": {
                                "type": "string",
                                "description": "The absolute path of the project directory on the host (e.g. E:\\LLM\\claude-code-main)"
                            }
                        },
                        "required": ["project_path"]
                    }
                }
            }
        ]
        
        self.tools_anthropic = [
            {
                "name": "mount_codeclaw",
                "description": "Start the full CodeClaw Agentic UI and mount a local directory for deep debugging/coding.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "project_path": {
                            "type": "string",
                            "description": "The absolute path of the project directory on the host (e.g. E:\\LLM\\claude-code-main)"
                        }
                    },
                    "required": ["project_path"]
                }
            }
        ]

    async def get_response(self, chat_id: str, text: str) -> Tuple[str, Optional[str]]:
        """
        Returns (response_text, path_to_mount)
        If path_to_mount is not None, the Gateway will hijack the chat.
        """
        history = self.conversations.setdefault(chat_id, [])
        history.append({"role": "user", "content": text})
        
        # Keep only last 10 messages to save tokens in idle chat
        if len(history) > 10:
            history = history[-10:]
            
        try:
            if self.provider == "anthropic":
                response = await self.client.messages.create(
                    model=self.anthropic_model,
                    system=self.system_prompt,
                    max_tokens=1024,
                    messages=history,
                    tools=self.tools_anthropic,
                )
                for block in response.content:
                    if getattr(block, "type", "") == "tool_use":
                        if block.name == "mount_codeclaw":
                            path = block.input.get("project_path")
                            return "🚀 正在挂载 CodeClaw 引擎，准备接管会话...", path
                            
                for block in response.content:
                    if getattr(block, "type", "") == "text":
                        history.append({"role": "assistant", "content": block.text})
                        return block.text, None
                        
                return "无响应", None
            else:
                response = await self.client.chat.completions.create(
                    model=self.openai_model,
                messages=[
                        {"role": "system", "content": self.system_prompt}
                    ] + history,
                    tools=self.tools_openai,
                    tool_choice="auto"
                )
                
                message = response.choices[0].message
                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        if tool_call.function.name == "mount_codeclaw":
                            args = json.loads(tool_call.function.arguments)
                            path = args.get("project_path")
                            return "🚀 正在挂载 CodeClaw 引擎，准备接管会话...", path
                            
                reply_text = message.content or "无响应"
                history.append({"role": "assistant", "content": reply_text})
                return reply_text, None
            
        except Exception as e:
            return f"❌ 闲聊助手出现错误: {str(e)}", None
