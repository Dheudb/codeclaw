"""JSON-lines RPC over stdio. One process owns one project session."""
import asyncio
import dataclasses
import json
import os
import re
import sys
import uuid
from pathlib import Path

from codeclaw.core.config import load_config, load_config_into_env, save_config
from codeclaw.skills.registry import SkillRegistry
from codeclaw.agents.registry import AgentRegistry


def pipe_reader(fd):
    if os.name != "nt":
        return lambda: os.read(fd, 65536)
    # A blocking CRT _read can hold descriptor locks needed by native DLL
    # initialization. ReadFile keeps the idle reader outside those CRT locks.
    import ctypes
    import msvcrt
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    read_file = kernel.ReadFile
    read_file.argtypes = [wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID]
    read_file.restype = wintypes.BOOL
    handle = msvcrt.get_osfhandle(fd)
    def read():
        buffer = ctypes.create_string_buffer(65536)
        count = wintypes.DWORD()
        if not read_file(handle, buffer, len(buffer), ctypes.byref(count), None):
            error = ctypes.get_last_error()
            if error in {109, 232}:
                return b""
            raise ctypes.WinError(error)
        return buffer.raw[:count.value]
    return read


class DesktopBridge:
    def __init__(self, emit, cwd=None):
        self.emit = emit
        self.cwd = Path(cwd or os.getcwd()).resolve()
        self.skills = SkillRegistry(self.cwd)
        self.agents = AgentRegistry(self.cwd)
        self.engine = None
        self.turn_task = None
        self.pending = {}
        self.resume_id = None
        self.loaded_skills = []
        self.protocol_task = None
        self.protocol_ready = None
        self.protocol_shutdown = asyncio.Event()

    def busy(self):
        return self.turn_task is not None and not self.turn_task.done()

    def require_idle(self):
        workers = self.engine and self.engine.tool_context["subagent_tasks"].tasks.values()
        if self.busy() or (workers and any(not task.done() for task in workers)):
            raise ValueError("Wait for running tasks or stop them before changing this session")

    def config(self):
        cfg = load_config()
        provider = cfg.get("model_provider") or os.environ.get("CODECLAW_MODEL_PROVIDER", "anthropic")
        prefix = "openai" if provider == "openai" else "anthropic"
        return {"provider": provider, "model": cfg.get(prefix + "_model") or os.environ.get(prefix.upper() + "_MODEL", ""), "baseUrl": cfg.get(prefix + "_base_url") or os.environ.get(prefix.upper() + "_BASE_URL", ""), "hasKey": bool(cfg.get(prefix + "_api_key") or os.environ.get(prefix.upper() + "_API_KEY"))}

    def capabilities(self):
        definitions = self.agents.definitions()
        builtins = [{"name": name, "description": item.get("when_to_use", ""), "scope": "builtin", "valid": True, "shadowed": False} for name, item in definitions.items() if "path" not in item]
        return {"skills": self.skills.discover(), "agents": builtins + [{k: v for k, v in item.items() if k != "disallowed_tools"} for item in self.agents.discover()], "settings": self.skills.settings()}

    def sessions(self):
        root = self.cwd / ".codeclaw" / "sessions"
        items = []
        if not root.exists():
            return items
        for path in sorted(root.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:50]:
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                messages = raw if isinstance(raw, list) else raw.get("messages", [])
                history = self.history(messages)
                title = next((m["text"][:65] for m in history if m["role"] == "user"), "Untitled session")
                items.append({"id": path.stem, "title": title, "updated": path.stat().st_mtime})
            except (OSError, ValueError, AttributeError):
                continue
        return items

    @staticmethod
    def history(messages):
        items = []
        for message in messages:
            if message.get("_is_meta") or message.get("role") not in {"user", "assistant"}:
                continue
            if message.get("_origin") in {"team", "subagent"}:
                continue
            content = message.get("content", "")
            text = content if isinstance(content, str) else "\n".join(block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text")
            text = text.split("\n\n<selected-skills>")[0]
            if text.strip():
                items.append({"role": message["role"], "text": text})
        return items

    async def ask(self, kind, payload):
        request_id = str(uuid.uuid4())
        future = asyncio.get_running_loop().create_future()
        self.pending[request_id] = future
        self.emit({"type": kind, "requestId": request_id, **payload})
        try:
            return await future
        finally:
            self.pending.pop(request_id, None)
            self.emit({"type": "interaction_resolved", "requestId": request_id})

    async def permission(self, request):
        return await self.ask("permission", {"request": dataclasses.asdict(request)})

    async def ensure_engine(self):
        if self.engine is not None:
            if self.protocol_ready is not None:
                await asyncio.shield(self.protocol_ready)
            return self.engine
        load_config_into_env()
        from codeclaw.core.engine import QueryEngine
        from codeclaw.tools.lsp_tool import LspTool
        from codeclaw.protocols.lsp_manager import LSPManager
        engine = QueryEngine(cwd=str(self.cwd), permission_handler=self.permission)
        if self.resume_id and not engine.load_session(self.resume_id):
            await engine.client.close()
            raise ValueError("Session could not be resumed")
        for record in engine.subagent_registry:
            if record.get("status") == "running":
                record["status"] = "interrupted"
        for member in engine.team_manager.get_non_lead_members():
            member.is_active = False
        engine.set_permission_handler(self.permission)
        engine.tool_context.update(skill_registry=self.skills, agent_registry=self.agents, capability_event=self.emit, ask_user_callback=lambda questions: self.ask("questions", {"questions": questions}), plan_approval_callback=lambda content: self.ask("plan_approval", {"content": content}))
        lsp = LspTool()
        engine.lsp_manager = LSPManager()
        engine.tool_context["lsp_manager"] = engine.lsp_manager
        lsp.context = engine.tool_context
        engine.available_tools[lsp.name] = lsp
        self.engine = engine
        self.protocol_ready = asyncio.get_running_loop().create_future()
        self.protocol_task = asyncio.create_task(self.protocol_lifetime(engine))
        await asyncio.shield(self.protocol_ready)
        self.emit({"type": "session_started", "sessionId": engine.session_id, "model": engine.primary_model})
        return engine

    async def protocol_lifetime(self, engine):
        # MCP's context managers must be entered and exited by the same task.
        from codeclaw.protocols.mcp_bridge import MCPBridge
        engine.mcp_bridge = MCPBridge(str(self.cwd / "mcp_servers.json"))
        try:
            try:
                tools = await engine.mcp_bridge.load_and_connect()
                for name, tool in tools.items():
                    tool.context = engine.tool_context
                    engine.available_tools[name] = tool
                self.emit({"type": "log", "text": "MCP tools connected: " + str(len(tools))})
            except Exception as exc:
                self.emit({"type": "log", "text": "MCP connection failed: " + str(exc)})
            finally:
                if not self.protocol_ready.done():
                    self.protocol_ready.set_result(True)
            await self.protocol_shutdown.wait()
        finally:
            await engine.mcp_bridge.cleanup()

    async def turn(self, text, selected_skills, mode):
        try:
            # Validate all selections before creating a turn or emitting loaded events.
            selections = [(name, self.skills.read(name)) for name in selected_skills]
            engine = await self.ensure_engine()
            engine.set_mode(mode)
            if selections:
                text += "\n\n<selected-skills>\n" + "\n\n".join(body for _, body in selections) + "\n</selected-skills>"
                for name, _ in selections:
                    self.emit({"type": "skill_loaded", "name": name})
            async for event in engine.submit_message(text, sys_print_callback=lambda text: self.emit({"type": "log", "text": str(text)})):
                if event.get("type") == "final" and str(event.get("text", "")).startswith(("[bold red]API Loop Exception:", "[bold red]System Halt:")):
                    self.emit({"type": "turn_error", "message": event["text"]})
                    continue
                self.emit(event)
            self.emit({"type": "sessions_changed", "sessions": self.sessions()})
        except asyncio.CancelledError:
            self.emit({"type": "turn_cancelled"})
        except Exception as exc:
            self.emit({"type": "turn_error", "message": str(exc)})
        finally:
            self.emit({"type": "turn_finished"})

    async def dispatch(self, method, params):
        if method == "initialize":
            sid = params.get("sessionId")
            if sid and not re.fullmatch(r"[a-zA-Z0-9_-]+", sid):
                raise ValueError("Invalid session ID")
            self.resume_id = sid
            messages = []
            if sid:
                raw = json.loads((self.cwd / ".codeclaw" / "sessions" / (sid + ".json")).read_text(encoding="utf-8"))
                messages = raw if isinstance(raw, list) else raw.get("messages", [])
            return {"project": str(self.cwd), "config": self.config(), "capabilities": self.capabilities(), "sessions": self.sessions(), "history": self.history(messages)}
        if method == "capabilities.list":
            return self.capabilities()
        if method == "capabilities.import":
            self.require_idle()
            kind = params.get("kind")
            if kind not in {"skills", "agents"}:
                raise ValueError("Unknown capability kind")
            path = Path(params["path"]).resolve()
            if not path.is_dir():
                raise ValueError("Choose an existing directory")
            if kind == "skills" and not any(path.rglob("SKILL.md")):
                raise ValueError("This directory contains no SKILL.md")
            if kind == "agents" and not any(path.glob("*.md")):
                raise ValueError("Choose a directory containing agent Markdown files")
            key = "skill_roots" if kind == "skills" else "agent_roots"
            roots = self.skills.settings().get(key, [])
            if str(path) not in roots:
                self.skills.update(**{key: roots + [str(path)]})
            return self.capabilities()
        if method == "capabilities.toggle":
            self.require_idle()
            skill = next((s for s in self.skills.discover() if s["id"] == params["id"]), None)
            if skill is None:
                raise ValueError("Unknown skill")
            disabled = set(self.skills.settings().get("disabled_skills", []))
            disabled.discard(skill["id"]) if params.get("enabled") else disabled.add(skill["id"])
            self.skills.update(disabled_skills=sorted(disabled))
            return self.capabilities()
        if method == "skills.read":
            return self.skills.read(params["name"])
        if method == "prompt":
            if self.busy():
                raise ValueError("A turn is already running")
            text = str(params.get("text", "")).strip()
            if not text:
                raise ValueError("Message is empty")
            if params.get("mode", "normal") not in {"normal", "plan"}:
                raise ValueError("Invalid mode")
            self.turn_task = asyncio.create_task(self.turn(text, params.get("skills", []), params.get("mode", "normal")))
            return {"started": True}
        if method == "respond":
            future = self.pending.get(params["requestId"])
            if future is None or future.done():
                raise ValueError("This interaction has already ended")
            future.set_result(params.get("answer"))
            return True
        if method == "stop":
            for future in self.pending.values():
                future.cancel()
            if self.engine:
                await self.engine.tool_context["subagent_tasks"].close()
                await self.engine.request_abort()
            if self.busy():
                self.turn_task.cancel()
                await asyncio.gather(self.turn_task, return_exceptions=True)
            return True
        if method == "agents.cancel":
            if not self.engine:
                raise ValueError("No active engine")
            return await self.engine.tool_context["subagent_tasks"].cancel(params["agentId"])
        if method == "config.save":
            self.require_idle()
            provider = params.get("provider")
            if provider not in {"anthropic", "openai"}:
                raise ValueError("Unsupported protocol")
            model = str(params.get("model", "")).strip()
            if not model:
                raise ValueError("Model ID is required")
            base_url = str(params.get("baseUrl", "")).strip()
            if base_url and not base_url.startswith(("https://", "http://")):
                raise ValueError("API URL must use http or https")
            cfg = load_config()
            cfg.update(model_provider=provider, **{provider + "_model": model, provider + "_base_url": base_url})
            if params.get("apiKey"):
                cfg[provider + "_api_key"] = params["apiKey"]
            save_config(cfg)
            for suffix in ("model", "base_url", "api_key"):
                value = cfg.get(provider + "_" + suffix, "")
                if value:
                    os.environ[provider.upper() + "_" + suffix.upper()] = value
                else:
                    os.environ.pop(provider.upper() + "_" + suffix.upper(), None)
            os.environ["CODECLAW_MODEL_PROVIDER"] = provider
            if self.engine:
                previous_client = self.engine.client
                previous_compactor = self.engine.compactor
                self.engine.switch_model_runtime(provider=provider, model=model, api_base_url=base_url, api_key=cfg.get(provider + "_api_key") or os.environ.get(provider.upper() + "_API_KEY"))
                await previous_client.close()
                if previous_compactor.client:
                    await previous_compactor.client.close()
            return self.config()
        if method == "changes":
            async def git(*args):
                process = await asyncio.create_subprocess_exec("git", "-c", "core.quotepath=false", *args, cwd=str(self.cwd), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                try:
                    output, _ = await asyncio.wait_for(process.communicate(), timeout=10)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.communicate()
                    raise ValueError("Git command timed out")
                return output.decode("utf-8", errors="replace") if process.returncode == 0 else ""
            return {"status": await git("status", "--short"), "diff": (await git("diff", "--no-ext-diff", "--no-color", "HEAD"))[:200000]}
        raise ValueError("Unknown method: " + method)

    async def close(self):
        await self.dispatch("stop", {})
        self.protocol_shutdown.set()
        if self.protocol_task:
            if self.protocol_ready is not None and not self.protocol_ready.done():
                self.protocol_task.cancel()
            await asyncio.gather(self.protocol_task, return_exceptions=True)
        if self.engine:
            self.engine.persist_session_state()
            for server in self.engine.lsp_manager.servers.values():
                await server.stop()
            await self.engine.browser_manager.close()
            for identity in self.engine.shell_task_manager.tasks:
                await self.engine.shell_task_manager.terminate_task(identity)
            await self.engine.client.close()
            if getattr(self.engine.compactor, "client", None):
                await self.engine.compactor.client.close()


async def serve():
    wire = sys.stdout
    sys.stdout = sys.stderr
    def emit(payload):
        wire.write(json.dumps({"event": payload}, ensure_ascii=False, default=str) + "\n")
        wire.flush()
    bridge = DesktopBridge(emit)
    requests = set()
    stdin_fd = sys.stdin.fileno()
    read_chunk = pipe_reader(stdin_fd)
    buffered = b""
    async def handle(request):
        try:
            result = await bridge.dispatch(request["method"], request.get("params") or {})
            response = {"id": request["id"], "result": result}
        except Exception as exc:
            response = {"id": request.get("id"), "error": str(exc)}
        wire.write(json.dumps(response, ensure_ascii=False, default=str) + "\n")
        wire.flush()
    try:
        while True:
            # TextIOWrapper.readline holds its buffer lock while idle. Library
            # imports checking stdin.isatty() would then deadlock the agent loop.
            chunk = await asyncio.to_thread(read_chunk)
            if not chunk:
                break
            buffered += chunk
            while b"\n" in buffered:
                line, buffered = buffered.split(b"\n", 1)
                try:
                    request = json.loads(line.decode("utf-8"))
                    if not isinstance(request, dict) or "id" not in request or not isinstance(request.get("method"), str):
                        raise ValueError("Invalid RPC request")
                    task = asyncio.create_task(handle(request))
                    requests.add(task)
                    task.add_done_callback(requests.discard)
                except (ValueError, UnicodeError) as exc:
                    emit({"type": "protocol_error", "message": str(exc)})
    finally:
        await bridge.close()
        for task in requests:
            task.cancel()
        await asyncio.gather(*requests, return_exceptions=True)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stdin.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    asyncio.run(serve())
