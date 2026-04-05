import asyncio
import builtins
import os
import json
import re
import threading
from typing import Dict, Any


def strip_rich_markup(text: str) -> str:
    """Remove Rich console markup tags like [bold cyan]...[/bold cyan] for plain text output."""
    return re.sub(r'\[/?[^\]]*\]', '', str(text))


class FeishuInputBridge:
    """
    Bridges the engine's blocking input() calls with async Feishu messages.
    When the engine calls input(), this bridge:
      1. Sends the prompt to Feishu
      2. Blocks the engine thread until a Feishu reply arrives
      3. Returns the reply as if the user typed it on stdin
    """
    def __init__(self):
        self._event = threading.Event()
        self._response = ""
        self._prompt = ""
        self._waiting = False
        self._send_func = None  # will be set to a callable that sends text to Feishu
    
    @property
    def is_waiting(self):
        return self._waiting
    
    def install(self, send_func):
        """Install the bridge, replacing builtins.input."""
        self._send_func = send_func
        self._original_input = builtins.input
        builtins.input = self._input_override
    
    def uninstall(self):
        """Restore original input."""
        builtins.input = self._original_input
    
    def _input_override(self, prompt=""):
        """Replacement for builtins.input() that routes through Feishu."""
        self._prompt = prompt
        self._waiting = True
        self._event.clear()
        
        # Send the prompt to Feishu
        if self._send_func and prompt:
            self._send_func(strip_rich_markup(prompt))
        
        # Block until user replies in Feishu
        self._event.wait(timeout=300)  # 5 min timeout
        self._waiting = False
        return self._response
    
    def provide_input(self, text: str):
        """Called from the Feishu message handler to feed user's reply."""
        self._response = text
        self._event.set()


from loguru import logger
import lark_oapi as lark

from codeclaw.core.engine import QueryEngine
from bot.feishu.idle_bot import IdleBot
from bot.feishu.renderer import FeishuRenderer

class FeishuGateway:
    def __init__(self, app_id: str, app_secret: str, encrypt_key: str = "", verification_token: str = ""):
        self.app_id = app_id
        self.app_secret = app_secret
        self.encrypt_key = encrypt_key
        self.verification_token = verification_token
        
        self.client = lark.Client.builder().app_id(app_id).app_secret(app_secret).log_level(lark.LogLevel.WARNING).build()
        self.idle_bot = IdleBot()
        
        # State: chat_id -> QueryEngine
        self.active_sessions: Dict[str, QueryEngine] = {}
        # State: chat_id -> Message buffer context
        self.message_locks: Dict[str, asyncio.Lock] = {}
        self._running = False
        
        # To run codeclaw's engine.run() in the background without blocking WS thread
        self.loop = asyncio.get_event_loop()
        
        # Input bridge for interactive prompts
        self.input_bridges: Dict[str, FeishuInputBridge] = {}

    def _get_lock(self, chat_id: str) -> asyncio.Lock:
        if chat_id not in self.message_locks:
            self.message_locks[chat_id] = asyncio.Lock()
        return self.message_locks[chat_id]

    async def _send_text(self, receive_id_type: str, receive_id: str, content: str, reply_to_message_id: str = None):
        """Sends a plain text / markdown text message"""
        from lark_oapi.api.im.v1 import (
            CreateMessageRequest, CreateMessageRequestBody
        )
        import json
        payload = {"text": content}
        req_body = CreateMessageRequestBody.builder() \
            .receive_id(receive_id) \
            .msg_type("text") \
            .content(json.dumps(payload)) \
            .build()
            
        req = CreateMessageRequest.builder() \
            .receive_id_type(receive_id_type) \
            .request_body(req_body).build()
            
        def _sync_send():
            if reply_to_message_id:
                return self.client.im.v1.message.reply(
                    lark.api.im.v1.ReplyMessageRequest.builder()
                    .message_id(reply_to_message_id)
                    .request_body(req_body).build()
                )
            else:
                return self.client.im.v1.message.create(req)
                
        resp = await self.loop.run_in_executor(None, _sync_send)
        return resp

    def _on_message_sync(self, data: Any) -> None:
        """Sync callback from Lark WS thread. Hand off to Asyncio Event Loop."""
        asyncio.run_coroutine_threadsafe(self._handle_message(data), self.loop)

    async def _handle_message(self, data: Any):
        event = data.header.event_type
        if event != "im.message.receive_v1":
            return
            
        msg = data.event.message
        sender = data.event.sender
        chat_id = msg.chat_id
        msg_id = msg.message_id
        
        content_type = msg.message_type
        content_text = ""
        if content_type == "text":
            root = json.loads(msg.content)
            content_text = root.get("text", "")
            
        content_text = content_text.strip()
        if not content_text:
            return
            
        async with self._get_lock(chat_id):
            # Check if the engine is waiting for interactive input
            if chat_id in self.input_bridges and self.input_bridges[chat_id].is_waiting:
                self.input_bridges[chat_id].provide_input(content_text)
                return
            
            if chat_id in self.active_sessions:
                await self._handle_hijacked_message(chat_id, msg_id, content_text)
            else:
                await self._handle_idle_message(chat_id, msg_id, content_text, sender)

    async def _handle_idle_message(self, chat_id: str, msg_id: str, content: str, sender: Any):
        logger.info(f"Idle Bot received: {content}")
        reply, path_to_mount = await self.idle_bot.get_response(chat_id, content)
        
        await self._send_text("chat_id", chat_id, reply, reply_to_message_id=msg_id)
        
        if path_to_mount and os.path.exists(path_to_mount):
            self._start_codeclaw_session(chat_id, path_to_mount)
        elif path_to_mount:
            await self._send_text("chat_id", chat_id, f"⚠️ 找不到路径: {path_to_mount}，挂载失败。")

    def _start_codeclaw_session(self, chat_id: str, project_dir: str):
        logger.info(f"Mounting CodeClaw session for {chat_id} at {project_dir}")
        try:
            os.chdir(project_dir)
            engine = QueryEngine(
                cwd=project_dir,
                permission_handler=self.permission_handler,
            )
            self.active_sessions[chat_id] = engine
            logger.info(f"✅ CodeClaw session active for {chat_id} at {project_dir}, total active: {len(self.active_sessions)}")
        except Exception as e:
            logger.error(f"❌ Failed to create QueryEngine for {chat_id}: {e}")

    def permission_handler(self, req: dict):
        return "allow"

    async def _handle_hijacked_message(self, chat_id: str, msg_id: str, content: str):
        normalized = content.strip().lower()
        engine = self.active_sessions[chat_id]
        
        # ── Slash commands (intercepted locally, never sent to engine) ──
        if normalized == "/exit":
            del self.active_sessions[chat_id]
            logger.info(f"Exited hijacked session for {chat_id}")
            await self._send_text("chat_id", chat_id, "🚪 已结束代码模式，返回普通闲聊。")
            return
        
        if normalized == "/help":
            help_text = (
                "📖 CodeClaw 飞书指令列表\n\n"
                "【会话控制】\n"
                "  /exit           退出代码模式，返回闲聊\n"
                "  /new            开启新会话\n"
                "  /sessions       查看历史会话列表\n"
                "  /resume <id>    恢复指定会话\n\n"
                "【规划模式】\n"
                "  /plan           开启 Plan 模式\n"
                "  /plan off       关闭 Plan 模式\n"
                "  /plan show      查看当前计划\n"
                "  /plan clear     清空当前计划\n\n"
                "【多智能体】\n"
                "  /team           查看多智能体状态\n"
                "  /coordinator    开启协调者模式\n"
                "  /coordinator off 关闭协调者模式\n\n"
                "【检查】\n"
                "  /status         查看引擎运行状态\n"
                "  /model status   查看当前模型配置\n"
                "  /mode           查看当前模式\n"
                "  /todos          查看结构化任务\n"
                "  /tools          查看工具活动\n"
                "  /agents         查看子智能体\n"
                "  /sandbox        查看沙盒状态\n\n"
                "【其他】\n"
                "  直接发送自然语言即可与 CodeClaw 引擎交互"
            )
            await self._send_text("chat_id", chat_id, help_text)
            return
        
        if normalized == "/status":
            await self._send_text("chat_id", chat_id, strip_rich_markup(engine.get_runtime_summary()))
            return
        if normalized == "/mode":
            await self._send_text("chat_id", chat_id, strip_rich_markup(engine.get_mode_summary()))
            return
        if normalized == "/todos":
            await self._send_text("chat_id", chat_id, strip_rich_markup(engine.get_todo_summary()))
            return
        if normalized == "/tools":
            await self._send_text("chat_id", chat_id, strip_rich_markup(engine.get_tools_summary()))
            return
        if normalized == "/agents":
            await self._send_text("chat_id", chat_id, strip_rich_markup(engine.get_agents_summary()))
            return
        if normalized == "/model status":
            info = (
                f"Provider: {engine.model_provider}\n"
                f"Model: {engine.primary_model}\n"
                f"Fallback: {engine.fallback_model or '(same)'}\n"
                f"API Base URL: {engine.api_base_url or '(default)'}"
            )
            await self._send_text("chat_id", chat_id, info)
            return
        if normalized == "/plan":
            engine.set_mode("plan")
            await self._send_text("chat_id", chat_id, "✅ Plan 模式已开启，写入类工具将被禁止。")
            return
        if normalized == "/plan off":
            engine.set_mode("normal")
            await self._send_text("chat_id", chat_id, "✅ 已恢复 Normal 模式。")
            return
        if normalized == "/plan show":
            plan = strip_rich_markup(engine.get_plan() or "(当前无计划)")
            await self._send_text("chat_id", chat_id, plan)
            return
        if normalized == "/plan clear":
            engine.plan_manager.clear_plan()
            await self._send_text("chat_id", chat_id, "✅ 当前计划已清空。")
            return
        if normalized == "/sessions":
            await self._send_text("chat_id", chat_id, strip_rich_markup(engine.session_manager.get_recent_sessions()))
            return
        if normalized == "/new":
            engine.persist_session_state()
            import uuid as _uuid
            old_sid = engine.session_id
            engine.session_id = str(_uuid.uuid4())
            engine.messages = []
            engine.plan_manager.clear_plan()
            engine.todo_manager.load([])
            await self._send_text("chat_id", chat_id, f"✅ 已开启新会话。旧会话已保存：{old_sid[:8]}...")
            return
        if normalized.startswith("/resume "):
            parts = content.strip().split(maxsplit=1)
            if len(parts) > 1:
                sid = parts[1].strip()
                if engine.load_session(sid):
                    await self._send_text("chat_id", chat_id, f"✅ 会话 {sid[:8]}... 已恢复。\n{strip_rich_markup(engine.get_resume_summary())}")
                else:
                    await self._send_text("chat_id", chat_id, f"❌ 会话 '{sid}' 未找到。")
            return
        if normalized == "/sandbox":
            await self._send_text("chat_id", chat_id, strip_rich_markup(engine.get_sandbox_summary()))
            return
        if normalized == "/team":
            await self._send_text("chat_id", chat_id, strip_rich_markup(engine.get_team_summary()))
            return
        if normalized == "/coordinator":
            engine.enable_coordinator_mode()
            await self._send_text("chat_id", chat_id, "✅ 协调者模式已开启，工具集限制为编排工具。")
            return
        if normalized == "/coordinator off":
            engine.disable_coordinator_mode()
            await self._send_text("chat_id", chat_id, "✅ 协调者模式已关闭，完整工具集已恢复。")
            return
        if normalized == "/clear sessions":
            count = engine.session_manager.clear_all_sessions()
            await self._send_text("chat_id", chat_id, f"✅ 已清除 {count} 个历史会话。")
            return
        if normalized.startswith("/delete "):
            parts = content.strip().split(maxsplit=1)
            if len(parts) > 1:
                target_sid = parts[1].strip()
                if target_sid == engine.session_id:
                    await self._send_text("chat_id", chat_id, "❌ 无法删除当前活跃会话，请先 /new 创建新会话。")
                elif engine.session_manager.delete_session(target_sid):
                    await self._send_text("chat_id", chat_id, f"✅ 会话 {target_sid[:8]}... 已删除。")
                else:
                    await self._send_text("chat_id", chat_id, f"❌ 会话 '{target_sid}' 未找到。")
            return
            
        # ── Normal message → send to engine ──
        await self._send_text("chat_id", chat_id, "⏳ CodeClaw 思考中...")
        
        text_chunks = []
        
        # Create input bridge for this chat so interactive prompts work
        bridge = FeishuInputBridge()
        self.input_bridges[chat_id] = bridge
        
        def send_to_feishu_sync(text: str):
            """Send text to Feishu from the sync engine thread."""
            asyncio.run_coroutine_threadsafe(
                self._send_text("chat_id", chat_id, text),
                self.loop
            )
        
        def feishu_print(*args, **kwargs):
            text = " ".join(str(a) for a in args)
            text_chunks.append(text)
            # Also send immediately to Feishu for real-time feedback
            if text.strip():
                send_to_feishu_sync(strip_rich_markup(text))
        
        bridge.install(send_to_feishu_sync)
        
        try:
            final_text = ""
            async for event in engine.submit_message(content, sys_print_callback=feishu_print):
                event_type = event.get("type", "")
                
                if event_type == "final":
                    final_text = event.get("text", "")
                elif event_type == "text_delta":
                    delta = event.get("text", "")
                    if delta:
                        text_chunks.append(delta)
                        
        except Exception as e:
            logger.error(f"Engine error: {e}", exc_info=True)
            await self._send_text("chat_id", chat_id, f"🔴 引擎报错: {str(e)}")
            return
        finally:
            bridge.uninstall()
            self.input_bridges.pop(chat_id, None)
        
        # Final answer takes priority; fallback to accumulated text_delta chunks
        reply = final_text or "".join(text_chunks).strip()
        if not reply:
            reply = "✅ 执行完成"
        
        # Don't re-send if feishu_print already sent everything in real-time
        # Only send the final consolidated reply
        MAX_LEN = 3500
        if len(reply) <= MAX_LEN:
            await self._send_text("chat_id", chat_id, reply)
        else:
            chunks = [reply[i:i+MAX_LEN] for i in range(0, len(reply), MAX_LEN)]
            for chunk in chunks:
                await self._send_text("chat_id", chat_id, chunk)
            

    def start(self):
        self._running = True
        builder = lark.EventDispatcherHandler.builder(self.encrypt_key, self.verification_token)
        builder.register_p2_im_message_receive_v1(self._on_message_sync)
        handler = builder.build()
        
        self.ws_client = lark.ws.Client(
            self.app_id, self.app_secret,
            event_handler=handler,
            log_level=lark.LogLevel.WARNING
        )
        
        logger.info("Feishu WebSocket Gateway started.")
        self.ws_client.start()
