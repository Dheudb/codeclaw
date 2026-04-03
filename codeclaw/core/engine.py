import os
import asyncio
import copy
import inspect
import json
import time
from types import SimpleNamespace
from typing import Optional
from anthropic import (
    AsyncAnthropic,
    APIConnectionError,
    APIStatusError,
    InternalServerError,
    RateLimitError,
)
try:
    from openai import AsyncOpenAI
except Exception:
    AsyncOpenAI = None
try:
    import git
except ImportError:
    git = None
from codeclaw.core.auto_commit import AutoCommitManager
from codeclaw.core.browser import BrowserSessionManager
from codeclaw.core.context import ContextBuilder
from codeclaw.core.content_replacement import ContentReplacementManager
from codeclaw.core.incremental_write_queue import IncrementalWriteQueue
from codeclaw.core.artifact_tracker import ArtifactTracker
from codeclaw.core.file_state_cache import FileStateCache
from codeclaw.core.hooks import HookManager
from codeclaw.core.memory import MemoryCompactor
from codeclaw.core.memory_files import MemoryFileManager
from codeclaw.core.security_classifier import AutoSecurityClassifier
from codeclaw.core.structured_output import StructuredOutputManager
from codeclaw.core.plans import PlanManager
from codeclaw.core.permissions import PermissionManager
from codeclaw.core.shell_tasks import ShellTaskManager
from codeclaw.core.tool_results import build_tool_result, serialize_tool_result
from codeclaw.core.todos import TodoManager
from codeclaw.core.vcr import VCRManager
from codeclaw.core.ultraplan import UltraplanManager
from codeclaw.core.team import TeamManager
from codeclaw.tools.bash_tool import BashTool
from codeclaw.tools.file_read_tool import FileReadTool
from codeclaw.tools.file_edit_tool import FileEditTool
from codeclaw.tools.file_write_tool import FileWriteTool
from codeclaw.tools.grep_tool import GrepTool
from codeclaw.tools.glob_tool import GlobTool
from codeclaw.protocols.mcp_bridge import MCPBridge
from codeclaw.protocols.lsp_manager import LSPManager
from codeclaw.core.sandbox import SandboxManager
from codeclaw.core.session import SessionManager
from codeclaw.tools.lsp_tool import LspTool
from codeclaw.tools.sandbox_tool import SandboxTool
from codeclaw.tools.task_tools import (
    TaskCreateTool,
    TaskKillTool,
    TaskListTool,
    TaskReadTool,
    TaskStatusTool,
)
from codeclaw.tools.web_tool import WebSearchTool, WebFetchTool
from codeclaw.tools.notebook_tool import NotebookTool
from codeclaw.tools.plan_tool import PlanTool
from codeclaw.tools.todo_tool import TodoWriteTool
from codeclaw.tools.agent_tool import AgentTool
from codeclaw.tools.browser_tool import BrowserTool
from codeclaw.tools.repl_tool import ReplTool
from codeclaw.tools.skill_tool import SkillTool
from codeclaw.tools.tool_search_tool import ToolSearchTool
from codeclaw.tools.team_tools import TeamCreateTool, TeamDeleteTool
from codeclaw.tools.send_message_tool import SendMessageTool
from codeclaw.tools.ask_user_question_tool import AskUserQuestionTool
from codeclaw.tools.brief_tool import BriefTool
from codeclaw.tools.plan_mode_tools import EnterPlanModeTool, ExitPlanModeTool
from codeclaw.core.messages import (
    create_user_message,
    create_assistant_message,
    create_tool_result_message,
    get_msg_uuid,
    strip_internal_fields,
    rollback_incomplete_turn,
    export_messages_state,
)
from codeclaw.core.attachments import AttachmentCollector
from codeclaw.core.frc import clear_old_function_results
from codeclaw.core.verification import VerificationManager, run_verification_agent
import uuid


class AbortRequestedError(Exception):
    pass

class QueryEngine:
    """
    The main Agentic loop mapping the Context System and Tools to the Anthropic REST API.
    """
    def __init__(
        self,
        model: Optional[str] = None,
        fallback_model: Optional[str] = None,
        permission_handler=None,
        thinking_config=None,
        mode_thinking_overrides=None,
        agent_id: str = None,
        parent_agent_id: str = None,
        agent_depth: int = 0,
        agent_role: str = "main",
        model_provider: str = None,
        api_base_url: str = None,
        local_tokenizer_path: str = None,
    ):
        self.model_provider = self._resolve_model_provider(model_provider, model)
        self.api_base_url = api_base_url or os.environ.get("OPENAI_BASE_URL", "")
        self.local_tokenizer_path = (
            local_tokenizer_path
            or os.environ.get("CODECLAW_LOCAL_TOKENIZER_PATH", "")
            or os.environ.get("CODECLAW_LOCAL_TOKENIZER_MODEL", "")
        )
        self.primary_model = self._resolve_primary_model(model)
        self.fallback_model = self._resolve_fallback_model(fallback_model)
        self.model = self.primary_model
        self.client = self._build_model_client()
        self.default_thinking_config = self._normalize_thinking_config(
            thinking_config if thinking_config is not None else {"type": "adaptive", "display": "summarized"}
        )
        normalized_mode_overrides = {
            "plan": {"type": "disabled"},
        }
        for mode_name, config in dict(mode_thinking_overrides or {}).items():
            normalized = self._normalize_thinking_config(config)
            if normalized is None:
                normalized_mode_overrides.pop(str(mode_name), None)
            else:
                normalized_mode_overrides[str(mode_name)] = normalized
        self.mode_thinking_overrides = normalized_mode_overrides
        self.agent_id = agent_id or str(uuid.uuid4())
        self.parent_agent_id = parent_agent_id
        self.agent_depth = agent_depth
        self.agent_role = agent_role
        self.context_builder = ContextBuilder()
        self.hook_manager = HookManager()
        self.compactor = MemoryCompactor(
            model=self.fallback_model or self.primary_model,
            provider=self.model_provider,
            api_base_url=self.api_base_url,
            local_tokenizer_path=self.local_tokenizer_path,
        )
        self.plan_manager = PlanManager()
        self.structured_output_manager = StructuredOutputManager()
        self.security_classifier = AutoSecurityClassifier()
        self.permission_manager = PermissionManager(
            permission_handler,
            mode_getter=self.plan_manager.get_mode,
            state_change_callback=self.persist_session_state,
            classifier_manager=self.security_classifier,
        )
        self.todo_manager = TodoManager()
        self.subagent_registry = []
        self.in_progress_tools = {}
        self.recent_tool_activity = []
        self.loop_transition_history = []
        self.token_usage_history = []
        self.post_sampling_history = []
        self.session_token_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }
        
        self.messages = []
        self._compact_boundary_index = 0
        self._tool_result_budget_chars = int(os.environ.get("CODECLAW_TOOL_RESULT_BUDGET_CHARS", "0")) or 400000
        self.session_manager = SessionManager()
        self.session_id = str(uuid.uuid4())
        self.last_persist_ok = True
        self.last_persist_error = ""
        self._last_reported_persist_error = ""
        self.subagent_state_lock = asyncio.Lock()
        self.abort_event = asyncio.Event()
        self.active_run_task = None
        self.active_streaming_tool_entries = []
        self.active_stream = None
        self.sandbox_manager = SandboxManager()
        self.browser_manager = BrowserSessionManager()
        self.shell_task_manager = ShellTaskManager()
        self.memory_file_manager = MemoryFileManager()
        self.file_state_cache = FileStateCache()
        self.artifact_tracker = ArtifactTracker()
        self.incremental_write_queue = IncrementalWriteQueue()
        self.vcr_manager = VCRManager()
        self.auto_commit_manager = AutoCommitManager(state_change_callback=self.persist_session_state)
        self.content_replacement_manager = ContentReplacementManager(session_id=self.session_id)
        self.ultraplan_manager = UltraplanManager()
        self.team_manager = TeamManager()
        self.attachment_collector = AttachmentCollector()
        self.verification_manager = VerificationManager()
        self.coordinator_mode = False

        all_tools = {
            "bash_tool": BashTool(),
            "file_read_tool": FileReadTool(),
            "file_edit_tool": FileEditTool(),
            "file_write_tool": FileWriteTool(),
            "grep_tool": GrepTool(),
            "glob_tool": GlobTool(),
            "sandbox_tool": SandboxTool(),
            "task_create_tool": TaskCreateTool(),
            "task_status_tool": TaskStatusTool(),
            "task_read_tool": TaskReadTool(),
            "task_kill_tool": TaskKillTool(),
            "task_list_tool": TaskListTool(),
            "web_search_tool": WebSearchTool(),
            "web_fetch_tool": WebFetchTool(),
            "notebook_tool": NotebookTool(),
            "plan_tool": PlanTool(),
            "todo_write_tool": TodoWriteTool(),
            "agent_tool": AgentTool(),
            "browser_tool": BrowserTool(),
            "repl_tool": ReplTool(),
            "skill_tool": SkillTool(),
            "tool_search_tool": ToolSearchTool(),
            "team_create_tool": TeamCreateTool(),
            "team_delete_tool": TeamDeleteTool(),
            "send_message_tool": SendMessageTool(),
            "ask_user_question_tool": AskUserQuestionTool(),
            "brief_tool": BriefTool(),
            "enter_plan_mode": EnterPlanModeTool(),
            "exit_plan_mode": ExitPlanModeTool(),
        }
        latent_tool_names = {
            "browser_tool",
            "web_search_tool",
            "web_fetch_tool",
            "notebook_tool",
        }
        self.available_tools = {
            name: tool for name, tool in all_tools.items() if name not in latent_tool_names
        }
        self.latent_tools = {
            name: tool for name, tool in all_tools.items() if name in latent_tool_names
        }
        
        # Share state across instances for locks and cross-tool persistence
        self.tool_context = {
            "read_file_state": {},
            "file_state_cache": self.file_state_cache,
            "artifact_tracker": self.artifact_tracker,
            "verification_manager": self.verification_manager,
            "incremental_write_queue": self.incremental_write_queue,
            "auto_commit_manager": self.auto_commit_manager,
            "content_replacement_manager": self.content_replacement_manager,
            "sandbox_manager": self.sandbox_manager,
            "engine_available_tools": self.available_tools,
            "latent_tools_registry": self.latent_tools,
            "activate_tools": self.activate_tools,
            "permission_manager": self.permission_manager,
            "permission_handler": permission_handler,
            "plan_manager": self.plan_manager,
            "structured_output_manager": self.structured_output_manager,
            "security_classifier": self.security_classifier,
            "todo_manager": self.todo_manager,
            "session_manager": self.session_manager,
            "hook_manager": self.hook_manager,
            "subagent_registry": self.subagent_registry,
            "register_subagent_record": self.register_subagent_record,
            "subagent_state_lock": self.subagent_state_lock,
            "team_manager": self.team_manager,
            "ultraplan_manager": self.ultraplan_manager,
            "browser_manager": self.browser_manager,
            "shell_task_manager": self.shell_task_manager,
            "in_progress_tools": self.in_progress_tools,
            "recent_tool_activity": self.recent_tool_activity,
            "loop_transition_history": self.loop_transition_history,
            "token_usage_history": self.token_usage_history,
            "post_sampling_history": self.post_sampling_history,
            "session_token_usage": self.session_token_usage,
            "thinking_config": self.default_thinking_config,
            "mode_thinking_overrides": self.mode_thinking_overrides,
            "model_provider": self.model_provider,
            "primary_model": self.primary_model,
            "fallback_model": self.fallback_model,
            "api_base_url": self.api_base_url,
            "local_tokenizer_path": self.local_tokenizer_path,
            "memory_files": self.memory_file_manager.export_state(),
            "file_state_cache_state": self.file_state_cache.export_state(),
            "artifact_tracker_state": self.artifact_tracker.export_state(),
            "incremental_write_queue_state": self.incremental_write_queue.export_state(),
            "auto_commit_state": self.auto_commit_manager.export_state(),
            "content_replacement_state": self.content_replacement_manager.export_state(),
            "structured_output_state": self.structured_output_manager.export_state(),
            "security_classifier_state": self.security_classifier.export_state(),
            "vcr_state": self.vcr_manager.export_state(),
            "attachment_collector": self.attachment_collector,
            "abort_event": self.abort_event,
            "messages": self.messages,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "parent_agent_id": self.parent_agent_id,
            "agent_depth": self.agent_depth,
            "agent_role": self.agent_role,
        }
        self._sync_model_runtime_context()
        for t in self.available_tools.values():
            t.context = self.tool_context
        for t in self.latent_tools.values():
            t.context = self.tool_context

    def set_permission_handler(self, permission_handler):
        self.permission_manager.set_prompt_handler(permission_handler)
        self.tool_context["permission_handler"] = permission_handler

    def switch_model_runtime(
        self,
        *,
        provider: str = None,
        model: str = None,
        api_base_url: str = None,
        api_key: str = None,
    ) -> dict:
        """
        Hot-swap the model provider, model name, API base URL, and/or API key at runtime.
        Returns a dict with the new effective configuration.
        """
        if provider:
            self.model_provider = self._resolve_model_provider(provider, model or self.primary_model)
        if model:
            self.primary_model = model
            self.model = model
        if api_base_url is not None:
            self.api_base_url = api_base_url
        if api_key:
            if self.model_provider == "openai":
                os.environ["OPENAI_API_KEY"] = api_key
            else:
                os.environ["ANTHROPIC_API_KEY"] = api_key

        self.client = self._build_model_client()
        self.compactor = MemoryCompactor(
            model=self.fallback_model or self.primary_model,
            provider=self.model_provider,
            api_base_url=self.api_base_url,
            local_tokenizer_path=self.local_tokenizer_path,
        )
        self._sync_model_runtime_context()
        self.persist_session_state()

        return {
            "provider": self.model_provider,
            "model": self.primary_model,
            "api_base_url": self.api_base_url or "(default)",
            "has_api_key": bool(
                os.environ.get("OPENAI_API_KEY") if self.model_provider == "openai"
                else os.environ.get("ANTHROPIC_API_KEY")
            ),
        }

    def _sync_model_runtime_context(self):
        self.tool_context["model_provider"] = self.model_provider
        self.tool_context["primary_model"] = self.primary_model
        self.tool_context["fallback_model"] = self.fallback_model
        self.tool_context["api_base_url"] = self.api_base_url
        self.tool_context["local_tokenizer_path"] = self.local_tokenizer_path

    def _resolve_model_provider(self, provider: str = None, model: str = None) -> str:
        resolved = str(
            provider
            or os.environ.get("CODECLAW_MODEL_PROVIDER", "")
            or ""
        ).strip().lower()
        if resolved in {"anthropic", "openai"}:
            return resolved
        model_name = str(model or "").strip().lower()
        if model_name.startswith(("gpt-", "o1", "o3", "o4")):
            return "openai"
        if os.environ.get("OPENAI_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"):
            return "openai"
        if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_BASE_URL"):
            return "anthropic"
        return ""

    def _resolve_primary_model(self, model: Optional[str]) -> str:
        candidate = str(model or "").strip()
        if self.model_provider == "openai":
            if not candidate or candidate.startswith("claude-"):
                return os.environ.get("OPENAI_MODEL", "") or os.environ.get("CODECLAW_OPENAI_MODEL", "") or ""
            return candidate
        if not candidate:
            return os.environ.get("ANTHROPIC_MODEL", "") or os.environ.get("CODECLAW_ANTHROPIC_MODEL", "") or ""
        return candidate

    def _resolve_fallback_model(self, fallback_model: Optional[str]) -> str:
        candidate = str(fallback_model or "").strip()
        if self.model_provider == "openai":
            if not candidate or candidate.startswith("claude-"):
                return os.environ.get("OPENAI_FALLBACK_MODEL", "") or os.environ.get("CODECLAW_OPENAI_FALLBACK_MODEL", "") or self.primary_model
            return candidate
        if not candidate:
            return os.environ.get("ANTHROPIC_FALLBACK_MODEL", "") or self.primary_model
        return candidate

    def _build_model_client(self):
        if self.model_provider == "openai":
            if AsyncOpenAI is None:
                raise RuntimeError(
                    "OpenAI provider requested but openai SDK is not installed."
                )
            kwargs = {}
            if self.api_base_url:
                kwargs["base_url"] = self.api_base_url
            return AsyncOpenAI(**kwargs)
        return AsyncAnthropic()

    @property
    def is_configured(self) -> bool:
        """Check if minimum configuration (provider + model) is present."""
        return bool(self.model_provider and self.primary_model)

    def _openai_supports_tools(self) -> bool:
        return self.model_provider == "openai"

    def _openai_supports_reasoning(self) -> bool:
        return False

    def _map_openai_finish_reason(self, finish_reason: str) -> str:
        mapped = str(finish_reason or "").strip().lower()
        if mapped == "tool_calls":
            return "tool_use"
        if mapped == "length":
            return "max_tokens"
        if mapped in {"stop", "content_filter"}:
            return "end_turn"
        return mapped or "end_turn"

    def _make_text_block(self, text: str):
        return SimpleNamespace(type="text", text=text)

    def _make_tool_use_block(self, *, tool_id: str, name: str, input_payload: dict):
        return SimpleNamespace(type="tool_use", id=tool_id, name=name, input=input_payload or {})

    def _make_usage_block(self, *, input_tokens: int = 0, output_tokens: int = 0):
        return SimpleNamespace(
            input_tokens=int(input_tokens or 0),
            output_tokens=int(output_tokens or 0),
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )

    def _assistant_text_from_blocks(self, blocks: list) -> str:
        return "".join(
            str(block.get("text", ""))
            for block in blocks or []
            if isinstance(block, dict) and block.get("type") == "text"
        )

    def _openai_tool_definition(self, tool_def: dict) -> dict:
        return {
            "type": "function",
            "function": {
                "name": tool_def.get("name", ""),
                "description": tool_def.get("description", ""),
                "parameters": copy.deepcopy(tool_def.get("input_schema", {}) or {"type": "object", "properties": {}}),
            },
        }

    def _clean_roles_for_openai_api(self, messages, system_prompt: str):
        normalized_messages = self._clean_roles_for_api(messages)
        rendered = []
        if system_prompt:
            rendered.append({"role": "system", "content": system_prompt})

        for msg in normalized_messages:
            role = msg.get("role")
            content = list(msg.get("content", []) or [])
            if role not in {"user", "assistant"}:
                continue

            if role == "assistant":
                text_parts = []
                tool_calls = []
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    block_type = block.get("type")
                    if block_type == "text":
                        text_parts.append(block.get("text", ""))
                    elif block_type == "tool_use":
                        tool_calls.append({
                            "id": block.get("id"),
                            "type": "function",
                            "function": {
                                "name": block.get("name", ""),
                                "arguments": json.dumps(block.get("input", {}) or {}, ensure_ascii=False),
                            },
                        })
                if text_parts or tool_calls:
                    rendered.append({
                        "role": "assistant",
                        "content": "".join(text_parts) if text_parts else None,
                        **({"tool_calls": tool_calls} if tool_calls else {}),
                    })
                continue

            text_parts = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type == "text":
                    text_parts.append(block.get("text", ""))
                elif block_type in {"image", "document"}:
                    text_parts.append(f"[{block_type} attached]")
                elif block_type == "tool_result":
                    if text_parts:
                        rendered.append({"role": "user", "content": "".join(text_parts)})
                        text_parts = []
                    rendered.append({
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id", ""),
                        "content": str(block.get("content", "") or ""),
                    })
            if text_parts:
                rendered.append({"role": "user", "content": "".join(text_parts)})

        return rendered

    def get_pending_auto_commit_proposal(self):
        return copy.deepcopy(self.auto_commit_manager.pending_proposal)

    def _get_git_repo(self, start_path: str = None):
        if git is None:
            return None
        try:
            return git.Repo(start_path or os.getcwd(), search_parent_directories=True)
        except Exception:
            return None

    def _safe_branch_name(self, repo) -> str:
        try:
            return repo.active_branch.name
        except Exception:
            return ""

    def _repo_relative_path(self, repo, abs_path: str) -> str:
        try:
            rel = os.path.relpath(os.path.abspath(abs_path), repo.working_tree_dir)
        except Exception:
            return ""
        if rel.startswith(".."):
            return ""
        return rel.replace("\\", "/")

    def _status_paths_for_repo(self, repo) -> set:
        if repo is None:
            return set()
        try:
            rendered = repo.git.status("--porcelain")
        except Exception:
            return set()
        paths = set()
        for line in rendered.splitlines():
            entry = line[3:].strip()
            if not entry:
                continue
            if "->" in entry:
                entry = entry.split("->", 1)[1].strip()
            abs_path = os.path.abspath(os.path.join(repo.working_tree_dir, entry))
            paths.add(abs_path)
        return paths

    def _capture_auto_commit_baseline(self) -> dict:
        repo = self._get_git_repo()
        if repo is None:
            return {}
        return {
            "repo_root": os.path.abspath(repo.working_tree_dir),
            "branch": self._safe_branch_name(repo),
            "dirty_paths": sorted(self._status_paths_for_repo(repo)),
        }

    def _collect_run_file_changes(self, started_at: float) -> list:
        changes = []
        for abs_path, entries in dict(self.artifact_tracker.file_history or {}).items():
            matching = [
                item for item in list(entries or [])
                if float(item.get("timestamp", 0) or 0) >= float(started_at)
                and item.get("session_id") == self.session_id
            ]
            if not matching:
                continue
            latest = matching[0]
            changes.append({
                "path": os.path.abspath(abs_path),
                "operation": latest.get("operation", "edit"),
                "timestamp": latest.get("timestamp", 0),
            })
        changes.sort(key=lambda item: item.get("path", ""))
        return changes

    def _heuristic_auto_commit_message(self, files: list, final_answer: str) -> str:
        relpaths = [item.get("relative_path") for item in files if item.get("relative_path")]
        if len(relpaths) == 1:
            subject = f"update {os.path.basename(relpaths[0])}"
        else:
            subject = f"update {len(relpaths)} files from agent run"
        details = final_answer.strip().splitlines()
        body = details[0][:160] if details else ""
        if body:
            return subject + "\n\n" + body
        return subject

    async def _generate_auto_commit_message(self, files: list, final_answer: str) -> tuple[str, str]:
        file_lines = [
            f"- {item.get('relative_path')} [{item.get('operation')}]"
            for item in files[:20]
        ]
        prompt = (
            "Generate a concise git commit message for the following agent-produced code changes.\n"
            "Return plain text only. Use a short subject line, optionally followed by a blank line and one short body sentence.\n\n"
            "Changed files:\n"
            + "\n".join(file_lines)
            + "\n\nAgent completion summary:\n"
            + (final_answer or "")[:1000]
        )
        try:
            if self.model_provider == "openai":
                response = await self.client.chat.completions.create(
                    model=self.primary_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=120,
                )
                rendered = str(
                    getattr(getattr(response.choices[0], "message", None), "content", "") or ""
                ).strip()
                if rendered:
                    return rendered, "model"
                raise RuntimeError("OpenAI commit message generation returned empty content.")

            response = await self.client.messages.create(
                model=self.primary_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=120,
            )
            texts = [getattr(block, "text", "") for block in getattr(response, "content", []) if getattr(block, "type", "") == "text"]
            rendered = "\n".join(texts).strip()
            if rendered:
                return rendered, "model"
        except Exception:
            pass
        return self._heuristic_auto_commit_message(files, final_answer), "heuristic"

    async def _maybe_prepare_auto_commit_proposal(
        self,
        *,
        final_answer: str,
        started_at: float,
        git_baseline: dict,
        turn: int,
        event_callback=None,
    ):
        if self.auto_commit_manager.pending_proposal:
            return None

        repo = self._get_git_repo(git_baseline.get("repo_root"))
        if repo is None:
            return None

        run_changes = self._collect_run_file_changes(started_at)
        if not run_changes:
            return None

        baseline_dirty_paths = set(git_baseline.get("dirty_paths", []) or [])
        current_dirty_paths = self._status_paths_for_repo(repo)
        eligible_files = []
        skipped_overlap = 0
        for item in run_changes:
            abs_path = item.get("path")
            if abs_path in baseline_dirty_paths:
                skipped_overlap += 1
                continue
            if abs_path not in current_dirty_paths and not os.path.exists(abs_path):
                continue
            relative_path = self._repo_relative_path(repo, abs_path)
            if not relative_path:
                continue
            eligible_files.append({
                **item,
                "relative_path": relative_path,
            })

        if not eligible_files:
            return None

        commit_message, message_source = await self._generate_auto_commit_message(
            eligible_files,
            final_answer,
        )
        proposal = await self.auto_commit_manager.create_proposal({
            "repo_root": os.path.abspath(repo.working_tree_dir),
            "branch": self._safe_branch_name(repo),
            "files": eligible_files,
            "message": commit_message,
            "message_preview": commit_message.splitlines()[0][:120] if commit_message else "",
            "message_source": message_source,
            "final_answer_preview": (final_answer or "")[:240],
            "skipped_dirty_overlap_count": skipped_overlap,
        })
        self._sync_message_context()
        await self._record_loop_transition(
            "auto_commit_proposed",
            turn=turn,
            event_callback=event_callback,
            file_count=len(eligible_files),
            message_source=message_source,
        )
        await self._emit_stream_event(event_callback, {
            "type": "auto_commit_proposal",
            "proposal": copy.deepcopy(proposal),
        })
        return proposal

    async def resolve_auto_commit_proposal(
        self,
        proposal_id: str,
        decision: str = "skip",
        message_override: str = None,
    ) -> dict:
        proposal = self.auto_commit_manager.pending_proposal
        if not proposal or proposal.get("proposal_id") != proposal_id:
            return build_tool_result(
                ok=False,
                content="No matching auto-commit proposal is pending.",
                metadata={"proposal_id": proposal_id},
                is_error=True,
            )

        normalized = str(decision or "skip").strip().lower()
        if normalized not in {"commit", "approve", "yes", "y", "skip", "no", "n"}:
            normalized = "skip"
        if normalized in {"skip", "no", "n"}:
            resolved = await self.auto_commit_manager.resolve_pending(
                status="skipped",
                metadata={"decision": normalized},
            )
            self._sync_message_context()
            self.persist_session_state()
            return build_tool_result(
                ok=True,
                content="Auto-commit proposal skipped.",
                metadata={"proposal": resolved},
            )

        repo = self._get_git_repo(proposal.get("repo_root"))
        if repo is None:
            resolved = await self.auto_commit_manager.resolve_pending(
                status="failed",
                metadata={"reason": "git repo unavailable"},
            )
            self._sync_message_context()
            self.persist_session_state()
            return build_tool_result(
                ok=False,
                content="Unable to resolve git repository for auto-commit.",
                metadata={"proposal": resolved},
                is_error=True,
            )

        commit_message = str(message_override or proposal.get("message") or "").strip()
        files = proposal.get("files", []) or []
        relpaths = [item.get("relative_path") for item in files if item.get("relative_path")]
        if not relpaths:
            resolved = await self.auto_commit_manager.resolve_pending(
                status="failed",
                metadata={"reason": "no eligible files"},
            )
            self._sync_message_context()
            self.persist_session_state()
            return build_tool_result(
                ok=False,
                content="Auto-commit proposal has no eligible files to commit.",
                metadata={"proposal": resolved},
                is_error=True,
            )

        head_before = ""
        try:
            head_before = repo.head.commit.hexsha
        except Exception:
            head_before = ""

        try:
            repo.index.add(relpaths)
            commit_obj = repo.index.commit(commit_message)
            commit_hash = str(getattr(commit_obj, "hexsha", "") or "")
            self.artifact_tracker.record_commit(
                cwd=proposal.get("repo_root") or repo.working_tree_dir,
                command="auto_commit_orchestrator",
                success=True,
                status="created",
                commit_hash=commit_hash,
                message_preview=commit_message.splitlines()[0][:240] if commit_message else "",
                head_before=head_before,
                head_after=commit_hash,
                branch=proposal.get("branch") or self._safe_branch_name(repo),
                repo_root=repo.working_tree_dir,
                author_name=getattr(getattr(commit_obj, "author", None), "name", "") or "",
                author_email=getattr(getattr(commit_obj, "author", None), "email", "") or "",
                tool_name="auto_commit_orchestrator",
                agent_role=self.agent_role,
                parent_agent_id=self.parent_agent_id,
                agent_depth=self.agent_depth,
                exit_code=0,
                agent_id=self.agent_id,
                session_id=self.session_id,
            )
            resolved = await self.auto_commit_manager.resolve_pending(
                status="committed",
                metadata={"commit_hash": commit_hash, "decision": normalized},
            )
            self._sync_message_context()
            self.persist_session_state()
            return build_tool_result(
                ok=True,
                content=f"Auto-commit created: {commit_hash[:12] if commit_hash else '<unknown>'}",
                metadata={"proposal": resolved, "commit_hash": commit_hash, "files": relpaths},
            )
        except Exception as e:
            self.artifact_tracker.record_commit(
                cwd=proposal.get("repo_root") or repo.working_tree_dir,
                command="auto_commit_orchestrator",
                success=False,
                status="failed",
                commit_hash="",
                message_preview=commit_message.splitlines()[0][:240] if commit_message else "",
                head_before=head_before,
                head_after="",
                branch=proposal.get("branch") or self._safe_branch_name(repo),
                repo_root=repo.working_tree_dir,
                tool_name="auto_commit_orchestrator",
                agent_role=self.agent_role,
                parent_agent_id=self.parent_agent_id,
                agent_depth=self.agent_depth,
                exit_code=None,
                agent_id=self.agent_id,
                session_id=self.session_id,
            )
            resolved = await self.auto_commit_manager.resolve_pending(
                status="failed",
                metadata={"error": str(e), "decision": normalized},
            )
            self._sync_message_context()
            self.persist_session_state()
            return build_tool_result(
                ok=False,
                content=f"Auto-commit failed: {str(e)}",
                metadata={"proposal": resolved},
                is_error=True,
            )

    def has_active_turn(self) -> bool:
        return self.active_run_task is not None and not self.active_run_task.done()

    async def request_abort(self) -> bool:
        if not self.has_active_turn():
            return False

        self.abort_event.set()

        if self.active_stream is not None:
            try:
                await self.active_stream.close()
            except Exception:
                pass

        if self.active_streaming_tool_entries:
            await self._cancel_streaming_tool_tasks(self.active_streaming_tool_entries)

        try:
            await asyncio.wait_for(asyncio.shield(self.active_run_task), timeout=1.5)
        except asyncio.TimeoutError:
            self.active_run_task.cancel()
            await asyncio.gather(self.active_run_task, return_exceptions=True)
        except Exception:
            pass

        return True

    def activate_tools(self, tool_names):
        activated = []
        for name in tool_names or []:
            tool = self.latent_tools.pop(name, None)
            if not tool:
                continue
            tool.context = self.tool_context
            self.available_tools[name] = tool
            activated.append(name)

        self.tool_context["engine_available_tools"] = self.available_tools
        self.tool_context["latent_tools_registry"] = self.latent_tools
        return activated

    async def _maybe_run_verification(self, sys_print_callback=None) -> str:
        """Run verification agent if enough files were modified."""
        if not self.verification_manager.should_verify():
            return ""
        if self.agent_role == "verifier":
            return ""
        files_to_verify = self.verification_manager.get_files_to_verify()
        if not files_to_verify:
            return ""
        if sys_print_callback:
            sys_print_callback(
                f"[bold cyan]🔍 Verification agent reviewing {len(files_to_verify)} modified files...[/bold cyan]"
            )
        report = await run_verification_agent(
            files_to_verify=files_to_verify,
            parent_engine=self,
            sys_print_callback=lambda x: None,
        )
        self.verification_manager.mark_verified(files_to_verify, report or "")
        return report or ""

    def _apply_frc_if_needed(self):
        """Apply Function Result Clearing to keep context window manageable."""
        if not hasattr(self, "_frc_cleared_ids"):
            self._frc_cleared_ids = set()
        if len(self.messages) < 10:
            return
        self.messages, num_cleared, self._frc_cleared_ids = clear_old_function_results(
            self.messages,
            preserve_recent_results=6,
            min_content_chars=200,
            cleared_ids=self._frc_cleared_ids,
        )

    # ── Compact Boundary Slicing ──────────────────────────────────────
    def _find_last_compact_boundary_index(self) -> int:
        """Find the index of the last compaction boundary message."""
        for i in range(len(self.messages) - 1, -1, -1):
            msg = self.messages[i]
            content = msg.get("content", "")
            if isinstance(content, str) and "[CompactionBoundary:" in content:
                return i
        return 0

    def _get_messages_after_compact_boundary(self) -> list:
        """Return only messages from the last compact boundary onward for API calls.

        This avoids sending both the summary and the original messages that
        were summarized, which wastes tokens and degrades compact effectiveness.
        """
        boundary = self._compact_boundary_index
        if boundary <= 0 or boundary >= len(self.messages):
            return self.messages
        return self.messages[boundary:]

    def _update_compact_boundary(self):
        """Refresh the boundary index after compaction modifies self.messages."""
        self._compact_boundary_index = self._find_last_compact_boundary_index()

    # ── Tool Result Budget Pool ───────────────────────────────────────
    def _apply_tool_result_budget(self, messages: list) -> list:
        """Enforce a total character budget across all tool_result content blocks.

        Scans messages from newest to oldest.  Newest results are kept intact;
        once the cumulative size exceeds the budget, older results are replaced
        with a short placeholder.
        """
        budget = self._tool_result_budget_chars
        if budget <= 0:
            return messages

        indexed = []
        for idx, msg in enumerate(messages):
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    text = block.get("content", "")
                    if isinstance(text, str):
                        indexed.append((idx, block, len(text)))

        total = 0
        over_budget_blocks = []
        for idx, block, size in reversed(indexed):
            total += size
            if total > budget:
                over_budget_blocks.append((idx, block))

        if not over_budget_blocks:
            return messages

        for idx, block in over_budget_blocks:
            original_len = len(block.get("content", ""))
            tool_id = block.get("tool_use_id", "unknown")
            block["content"] = (
                f"[Tool result truncated by budget — original {original_len} chars, "
                f"tool_use_id={tool_id}. Re-run the tool if you need the full output.]"
            )

        return messages

    def _is_native_anthropic(self) -> bool:
        """True only when talking to Anthropic's own API, not a third-party proxy."""
        base = (self.api_base_url or os.environ.get("ANTHROPIC_BASE_URL", "")).lower()
        if not base or "anthropic.com" in base:
            return True
        return False

    def _build_cached_system_blocks(self, system_prompt: str) -> list:
        """
        Split system prompt into cached blocks for Anthropic's prompt caching.
        
        The static prefix (identity, rules, tool guidance) is marked with
        cache_control so it's reused across turns. The dynamic suffix
        (environment, git, todos, memory) changes each turn.
        Only used for native Anthropic API; third-party proxies (MiniMax etc.)
        get a plain string to avoid incompatible parameters.
        """
        boundary = self.context_builder.DYNAMIC_BOUNDARY
        if self.model_provider != "anthropic" or not self._is_native_anthropic():
            return system_prompt

        static, dynamic = self.context_builder.generate_system_prompt_split(
            session_summary=self.plan_manager.render_prompt_summary(),
            todo_summary=self.todo_manager.render_prompt_summary(),
            memory_summary=self.get_memory_summary(),
            structured_output_summary=self.structured_output_manager.render_prompt_summary(),
            tool_prompt_summary=self._build_tool_prompt_summary(),
            mcp_instructions=self._build_mcp_instructions(),
        )
        blocks = [
            {
                "type": "text",
                "text": static,
                "cache_control": {"type": "ephemeral"},
            },
            {
                "type": "text",
                "text": dynamic,
            },
        ]
        return blocks

    def _apply_message_cache_control(self, messages: list) -> list:
        """Add cache_control to the last content block of each message.

        Mirrors Claude Code's per-message prompt caching strategy: the last
        content block of every user/assistant message is tagged with
        ``cache_control: {"type": "ephemeral"}`` so that Anthropic can reuse
        the KV cache across turns.  Only applied when talking to native
        Anthropic API.
        """
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, str) and content:
                msg["content"] = [
                    {
                        "type": "text",
                        "text": content,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            elif isinstance(content, list) and content:
                last_block = content[-1]
                if isinstance(last_block, dict):
                    last_block["cache_control"] = {"type": "ephemeral"}
        return messages

    def _build_mcp_instructions(self) -> str:
        """Collect instructions from connected MCP servers for system prompt injection."""
        if not hasattr(self, "mcp_bridge") or self.mcp_bridge is None:
            return ""
        sessions = getattr(self.mcp_bridge, "sessions", {})
        if not sessions:
            return ""
        instruction_blocks = []
        for server_name, session_info in sessions.items():
            instructions = ""
            if isinstance(session_info, dict):
                instructions = session_info.get("instructions", "")
            elif hasattr(session_info, "instructions"):
                instructions = getattr(session_info, "instructions", "")
            if instructions:
                instruction_blocks.append(f"## {server_name}\n{instructions}")
        if not instruction_blocks:
            return ""
        return (
            "# MCP Server Instructions\n\n"
            "The following MCP servers have provided instructions for how to use their tools:\n\n"
            + "\n\n".join(instruction_blocks)
        )

    def _inject_turn_attachments(self, turn_count: int):
        """Collect and inject contextual attachment messages before each turn."""
        try:
            changed_files = self.attachment_collector.get_changed_files_from_git(
                os.getcwd()
            )
            lsp_diags = ""
            if hasattr(self, "lsp_manager"):
                lsp_diags = self.lsp_manager.consume_diagnostics() or ""

            attachment_messages = self.attachment_collector.collect_attachments(
                cwd=os.getcwd(),
                plan_mode=self.plan_manager.get_mode(),
                plan_content=self.plan_manager.get_plan(),
                todo_summary=self.todo_manager.render_prompt_summary(),
                todo_count=len(self.todo_manager.get_items()),
                active_tools=list(self.available_tools.keys()),
                changed_files=changed_files,
                lsp_diagnostics=lsp_diags,
                turn_count=turn_count,
                recent_tool_names=self.attachment_collector._recent_tool_names,
            )
            for att_msg in attachment_messages:
                self.messages.append(att_msg)
            if attachment_messages:
                self._sync_message_context()
        except Exception:
            pass

    def _build_tool_prompt_summary(self) -> str:
        sections = []
        for name, tool in self.available_tools.items():
            prompt_builder = getattr(tool, "prompt", None)
            if not callable(prompt_builder):
                continue
            content = str(prompt_builder() or "").strip()
            if not content:
                continue
            sections.append(f"[{name}] {content}")
        return "\n".join(sections)

    def set_mode(self, mode: str) -> str:
        return self.plan_manager.set_mode(mode)

    def get_mode(self) -> str:
        return self.plan_manager.get_mode()

    # --- Coordinator mode ---

    COORDINATOR_ALLOWED_TOOLS = {
        "agent_tool", "send_message_tool", "team_create_tool",
        "team_delete_tool", "plan_tool", "todo_write_tool",
        "file_read_tool", "grep_tool", "glob_tool",
    }

    PLAN_MODE_ALLOWED_TOOLS = {
        "file_read_tool", "grep_tool", "glob_tool",
        "plan_tool", "todo_write_tool", "agent_tool",
        "web_search_tool", "web_fetch_tool",
        "lsp_tool", "tool_search_tool",
    }

    PLAN_MODE_BLOCKED_TOOLS = {
        "bash_tool", "file_edit_tool", "file_write_tool",
        "notebook_tool", "sandbox_tool",
        "task_create_tool", "task_kill_tool",
    }

    def enable_coordinator_mode(self):
        self.coordinator_mode = True

    def disable_coordinator_mode(self):
        self.coordinator_mode = False

    def _get_effective_tools(self) -> dict:
        """Return tool set filtered by current mode (plan / coordinator)."""
        if self.coordinator_mode:
            return {
                name: tool for name, tool in self.available_tools.items()
                if name in self.COORDINATOR_ALLOWED_TOOLS
            }
        if self.plan_manager.get_mode() == "plan":
            return {
                name: tool for name, tool in self.available_tools.items()
                if name not in self.PLAN_MODE_BLOCKED_TOOLS
            }
        return self.available_tools

    def _get_coordinator_system_prompt_addon(self) -> str:
        if not self.coordinator_mode:
            return ""
        return (
            "\n\n--- COORDINATOR MODE ACTIVE ---\n"
            "You are operating as a **coordinator agent**. Your role:\n"
            "1. Break complex tasks into subtasks and delegate to workers via agent_tool.\n"
            "2. Use send_message_tool to send follow-up instructions to running workers.\n"
            "3. Track progress with todo_write_tool and plan_tool.\n"
            "4. You should NOT directly modify code/files — delegate that to workers.\n"
            "5. Synthesize worker results into a coherent final answer.\n"
            "Available worker-facing tools: agent_tool, send_message_tool\n"
            "Available planning tools: plan_tool, todo_write_tool\n"
            "Available read-only tools: file_read_tool, grep_tool, glob_tool\n"
        )

    def get_team_summary(self) -> str:
        return self.team_manager.get_summary()

    # --- Ultraplan ---

    def _create_ultraplan_sub_engine(self, agent_role="ultraplan", forced_mode="plan"):
        """Factory for creating a plan-mode sub-engine for ultraplan."""
        sub = QueryEngine(
            model=self.primary_model,
            fallback_model=self.fallback_model,
            permission_handler=self.tool_context.get("permission_handler"),
            thinking_config={"type": "disabled"},
            agent_id=None,
            parent_agent_id=self.agent_id,
            agent_depth=self.agent_depth + 1,
            agent_role=agent_role,
            model_provider=self.model_provider,
            api_base_url=self.api_base_url,
            local_tokenizer_path=self.local_tokenizer_path,
        )
        if forced_mode:
            sub.set_mode(forced_mode)
        return sub

    async def launch_ultraplan(self, description: str, seed_plan: str = "", on_phase_change=None) -> dict:
        result = await self.ultraplan_manager.launch(
            description=description,
            engine_factory=self._create_ultraplan_sub_engine,
            seed_plan=seed_plan,
            on_phase_change=on_phase_change,
        )
        if result.get("ok") and result.get("plan"):
            self.plan_manager.write_plan(result["plan"])
            self.set_mode("normal")
            self.persist_session_state()
        return result

    def get_ultraplan_summary(self) -> str:
        return self.ultraplan_manager.get_status_summary()

    def _normalize_thinking_config(self, config):
        if config is None or config is False or config == "disabled":
            return {"type": "disabled"}
        if config == "adaptive":
            return {"type": "adaptive", "display": "summarized"}
        if config == "enabled":
            return {"type": "enabled", "budget_tokens": 2048, "display": "summarized"}
        if not isinstance(config, dict):
            return None

        thinking_type = str(config.get("type", "")).strip().lower()
        display = str(config.get("display", "summarized")).strip().lower()
        if display not in {"summarized", "omitted"}:
            display = "summarized"

        if thinking_type == "disabled":
            return {"type": "disabled"}
        if thinking_type == "adaptive":
            return {"type": "adaptive", "display": display}
        if thinking_type == "enabled":
            budget_tokens = config.get("budget_tokens", 2048)
            try:
                budget_tokens = int(budget_tokens)
            except (TypeError, ValueError):
                budget_tokens = 2048
            budget_tokens = max(1024, budget_tokens)
            return {
                "type": "enabled",
                "budget_tokens": budget_tokens,
                "display": display,
            }
        return None

    def _thinking_config_to_text(self, config) -> str:
        normalized = self._normalize_thinking_config(config)
        if not normalized:
            return "disabled"
        if normalized["type"] == "enabled":
            return (
                f"enabled (budget_tokens={normalized.get('budget_tokens')}, "
                f"display={normalized.get('display', 'summarized')})"
            )
        if normalized["type"] == "adaptive":
            return f"adaptive (display={normalized.get('display', 'summarized')})"
        return "disabled"

    def set_thinking_config(self, config) -> dict:
        normalized = self._normalize_thinking_config(config)
        if normalized is None:
            raise ValueError("Unsupported thinking config.")
        self.default_thinking_config = normalized
        self.tool_context["thinking_config"] = self.default_thinking_config
        return copy.deepcopy(self.default_thinking_config)

    def set_mode_thinking_config(self, mode: str, config) -> dict:
        mode_name = str(mode or "").strip()
        if not mode_name:
            raise ValueError("mode is required.")
        normalized = self._normalize_thinking_config(config)
        if normalized is None:
            self.mode_thinking_overrides.pop(mode_name, None)
        else:
            self.mode_thinking_overrides[mode_name] = normalized
        self.tool_context["mode_thinking_overrides"] = self.mode_thinking_overrides
        return copy.deepcopy(self.mode_thinking_overrides)

    def get_thinking_config(self, mode: str = None) -> dict:
        if mode is None:
            return copy.deepcopy(self._resolve_thinking_config())
        return copy.deepcopy(
            self.mode_thinking_overrides.get(str(mode), self.default_thinking_config)
        )

    def _resolve_thinking_config(self) -> dict:
        mode_name = self.get_mode()
        effective = self.mode_thinking_overrides.get(mode_name, self.default_thinking_config)
        normalized = self._normalize_thinking_config(effective)
        return normalized or {"type": "disabled"}

    def refresh_memory_files(self):
        self.memory_file_manager.set_cwd(os.getcwd())
        loaded = self.memory_file_manager.refresh()
        self.tool_context["memory_files"] = self.memory_file_manager.export_state()
        return loaded

    def get_memory_summary(self) -> str:
        return self.memory_file_manager.render_prompt_summary()

    def set_structured_output_request(self, request: Optional[dict]):
        result = self.structured_output_manager.set_request(request)
        self.tool_context["structured_output_state"] = self.structured_output_manager.export_state()
        return result

    def clear_structured_output_request(self):
        self.structured_output_manager.clear()
        self.tool_context["structured_output_state"] = self.structured_output_manager.export_state()

    def _build_structured_output_feedback(self, reason: str) -> str:
        return (
            "<system-reminder>\n"
            "The previous final answer failed the structured output requirement.\n"
            "Retry and return only a valid structured response.\n"
            f"Validation failure:\n{reason}\n"
            "</system-reminder>"
        )

    async def _apply_post_sampling_hooks(
        self,
        *,
        final_answer: str,
        turn: int,
        event_callback=None,
    ):
        result = self.hook_manager.evaluate_post_sampling_hooks({
            "cwd": os.getcwd(),
            "mode": self.get_mode(),
            "agent_id": self.agent_id,
            "agent_role": self.agent_role,
            "final_answer": final_answer,
            "session_token_usage": dict(self.session_token_usage),
        })
        if not result.outputs:
            return

        appended_messages = []
        for item in result.outputs:
            severity = item.get("severity", "info")
            rendered = item.get("rendered", "")
            message = {
                "role": "user",
                "content": (
                    "<system-reminder>\n"
                    f"[Hook:PostSamplingHook #{item.get('hook_index')} severity={severity}]\n"
                    f"{rendered}\n"
                    "</system-reminder>"
                ),
            }
            appended_messages.append(message)
            self.post_sampling_history.insert(0, {
                "turn": turn,
                "hook_index": item.get("hook_index"),
                "severity": severity,
                "behavior": item.get("behavior", "observe"),
                "rendered": rendered,
            })

        self.post_sampling_history = self.post_sampling_history[:20]
        self.messages.extend(appended_messages)
        self._sync_message_context()
        await self._record_loop_transition(
            "post_sampling_hook",
            turn=turn,
            event_callback=event_callback,
            count=len(result.outputs),
        )
        await self._emit_stream_event(event_callback, {
            "type": "post_sampling_hook",
            "turn": turn,
            "outputs": result.outputs,
        })

    def _build_cache_compaction_signal(self, recent_turns: int = 5) -> dict:
        recent_records = self.token_usage_history[:recent_turns]
        recent_cache_creation = sum(
            int(item.get("cache_creation_input_tokens", 0) or 0)
            for item in recent_records
        )
        recent_cache_read = sum(
            int(item.get("cache_read_input_tokens", 0) or 0)
            for item in recent_records
        )
        cache_pressure_tokens = max(0, recent_cache_creation - recent_cache_read)
        cache_hit_ratio = (
            recent_cache_read / max(1, recent_cache_creation + recent_cache_read)
        )
        return {
            "recent_turns": len(recent_records),
            "recent_cache_creation_input_tokens": recent_cache_creation,
            "recent_cache_read_input_tokens": recent_cache_read,
            "session_cache_creation_input_tokens": int(
                self.session_token_usage.get("cache_creation_input_tokens", 0) or 0
            ),
            "session_cache_read_input_tokens": int(
                self.session_token_usage.get("cache_read_input_tokens", 0) or 0
            ),
            "cache_pressure_tokens": cache_pressure_tokens,
            "cache_hit_ratio": round(cache_hit_ratio, 3),
        }

    def _build_spill_recorder(self, *, tool_name: str, tool_use_id: str):
        def _record_spill(*, file_path: str, original_content_chars: int = 0, metadata: dict = None):
            self.content_replacement_manager.register_spill(
                file_path=file_path,
                tool_name=tool_name,
                tool_use_id=tool_use_id,
                original_content_chars=original_content_chars,
                metadata=metadata,
            )
            self.tool_context["content_replacement_state"] = self.content_replacement_manager.export_state()
        return _record_spill

    def get_plan(self) -> str:
        return self.plan_manager.get_plan()

    def get_todo_summary(self) -> str:
        items = self.todo_manager.export()
        if not items:
            return "Current structured todo list is empty."

        lines = [f"Structured Todos ({len(items)})"]
        for item in items:
            lines.append(f"- [{item.get('status')}] {item.get('id')}: {item.get('content')}")
        return "\n".join(lines)

    def get_sandbox_summary(self) -> str:
        status = self.sandbox_manager.get_status()
        if not status.get("active"):
            return "No active sandbox."

        lines = [
            f"Sandbox mode: {status.get('mode')}",
            f"Branch: {status.get('branch')}",
            f"Path: {status.get('path')}",
            f"Root: {status.get('root')}",
        ]
        if "dirty" in status:
            lines.append(f"Dirty: {status.get('dirty')}")
        if "untracked_files" in status:
            lines.append(f"Untracked files: {status.get('untracked_files')}")
        if "head_commit" in status:
            lines.append(f"Head commit: {status.get('head_commit')}")
        return "\n".join(lines)

    def get_agents_summary(self) -> str:
        records = self.subagent_registry or []
        if not records:
            return "No subagents tracked in this session."

        lines = [f"Tracked Subagents ({len(records)})"]
        for item in records:
            lines.append(
                f"- {item.get('agent_id')} [{item.get('status', 'unknown')}] depth={item.get('agent_depth', '?')} task={item.get('task', '')}"
            )
        return "\n".join(lines)

    def get_mode_summary(self) -> str:
        return (
            f"Current mode: {self.get_mode()}\n"
            f"Plan present: {'yes' if bool(self.get_plan()) else 'no'}\n"
            f"Todos tracked: {len(self.todo_manager.export())}\n"
            f"Sandbox active: {'yes' if self.sandbox_manager.get_status().get('active') else 'no'}"
        )

    def get_tools_summary(self) -> str:
        lines = []
        if self.in_progress_tools:
            lines.append(f"In-progress tools ({len(self.in_progress_tools)})")
            for item in self.in_progress_tools.values():
                lines.append(
                    f"- {item.get('tool_name')} [{item.get('status')}] {item.get('input_preview')}"
                )
        else:
            lines.append("In-progress tools: none")

        lines.append("")
        if self.recent_tool_activity:
            lines.append(f"Recent tool activity ({len(self.recent_tool_activity)})")
            for item in self.recent_tool_activity[:8]:
                lines.append(
                    f"- {item.get('tool_name')} [{item.get('status')}] {item.get('result_preview')}"
                )
        else:
            lines.append("Recent tool activity: none")

        return "\n".join(lines)

    def get_runtime_summary(self) -> str:
        self.refresh_memory_files()
        self.content_replacement_manager.cleanup_orphans()
        shell_tasks = self.shell_task_manager.list_tasks()
        running_shell_tasks = [item for item in shell_tasks if item.get("status") == "running"]
        running_subagents = [item for item in self.subagent_registry if item.get("status") == "running"]
        in_progress_todos = [
            item for item in self.todo_manager.export()
            if item.get("status") == "in_progress"
        ]
        browser_state = self.browser_manager.export_state()
        estimated_tokens = self.compactor.estimate_tokens(self.messages)
        last_transition = self.loop_transition_history[0] if self.loop_transition_history else None
        permission_state = self.permission_manager.export_state()
        denial_history = permission_state.get("denial_history", [])
        pending_permission = permission_state.get("pending_request") or {}
        active_denials = {
            key: value
            for key, value in dict(permission_state.get("consecutive_denials", {}) or {}).items()
            if int(value or 0) > 0
        }
        memory_state = self.memory_file_manager.export_state()
        loaded_memory_files = memory_state.get("loaded_files", [])
        file_state = self.file_state_cache.export_state()
        cached_files = file_state.get("entries", {})
        artifact_state = self.artifact_tracker.export_state()
        commit_history = artifact_state.get("commit_history", []) or []
        incremental_write_state = self.incremental_write_queue.export_state()
        auto_commit_state = self.auto_commit_manager.export_state()
        vcr_state = self.vcr_manager.export_state()
        security_classifier_state = self.security_classifier.export_state()
        file_change_count = sum(
            len(items) for items in dict(artifact_state.get("file_history", {}) or {}).values()
        )
        content_replacement_state = self.content_replacement_manager.export_state()
        replacement_registry = dict(content_replacement_state.get("registry", {}) or {})
        cache_signal = self._build_cache_compaction_signal()

        lines = [
            f"Mode: {self.get_mode()}",
            f"Model provider: {self.model_provider}",
            f"Primary model: {self.primary_model}",
            f"Fallback model: {self.fallback_model}",
            f"Thinking config: {self._thinking_config_to_text(self._resolve_thinking_config())}",
            f"Token estimator: {self.compactor.token_estimator_mode()}",
            f"Local tokenizer path: {self.local_tokenizer_path or 'none'}",
            f"Structured output active: {'yes' if self.structured_output_manager.request else 'no'}",
            f"Memory files loaded: {len(loaded_memory_files)}",
            f"Cached file states: {len(cached_files)}",
            f"Tracked attachments: {len(artifact_state.get('attachments', []) or [])}",
            f"Tracked prefetches: {len(artifact_state.get('prefetch_history', []) or [])}",
            f"Tracked file changes: {file_change_count}",
            f"Tracked commits: {len(commit_history)}",
            f"Incremental write queue events: {len(incremental_write_state.get('history', []) or [])}",
            f"Auto-commit proposals tracked: {len(auto_commit_state.get('proposal_history', []) or [])}",
            f"Content replacements tracked: {len(replacement_registry)}",
            f"Content replacement cleanups: {len(content_replacement_state.get('cleanup_history', []) or [])}",
            f"Post-sampling reports: {len(self.post_sampling_history)}",
            f"Security classifier mode: {security_classifier_state.get('mode', 'off')}",
            f"Security classifier hits: {len(security_classifier_state.get('history', []) or [])}",
            f"VCR mode: {vcr_state.get('mode', 'off')}",
            f"VCR cassettes touched: {len(vcr_state.get('history', []) or [])}",
            f"Estimated context tokens: {estimated_tokens}",
            f"Recent cache creation tokens: {cache_signal.get('recent_cache_creation_input_tokens', 0)}",
            f"Recent cache read tokens: {cache_signal.get('recent_cache_read_input_tokens', 0)}",
            f"Cache-aware compaction pressure: {cache_signal.get('cache_pressure_tokens', 0)}",
            f"Session input tokens: {self.session_token_usage.get('input_tokens', 0)}",
            f"Session output tokens: {self.session_token_usage.get('output_tokens', 0)}",
            f"Remaining output budget: {self._remaining_output_budget(self._build_loop_state())}",
            f"In-progress tools: {len(self.in_progress_tools)}",
            f"Running shell tasks: {len(running_shell_tasks)}",
            f"Running subagents: {len(running_subagents)}",
            f"In-progress todos: {len(in_progress_todos)}",
            f"Permission denials tracked: {len(denial_history)}",
            f"Tools with active denial counts: {len(active_denials)}",
            f"Pending permission request: {'yes' if pending_permission else 'no'}",
            f"Pending auto-commit proposal: {'yes' if auto_commit_state.get('pending_proposal') else 'no'}",
            f"Browser available: {'yes' if browser_state.get('available') else 'no'}",
            f"Browser last URL: {browser_state.get('last_url') or 'none'}",
            f"Session persistence: {'ok' if self.last_persist_ok else 'error'}",
            f"Last session file: {self.session_manager.last_save_path or 'unknown'}",
            (
                f"Persist error: {self.last_persist_error}"
                if self.last_persist_error
                else "Persist error: none"
            ),
        ]
        if last_transition:
            lines.append(
                f"Last loop transition: {last_transition.get('reason')} (turn {last_transition.get('turn')})"
            )
        if denial_history:
            last_denial = denial_history[0]
            lines.append(
                "Last permission denial: "
                f"{last_denial.get('tool_name')} via {last_denial.get('source')} "
                f"(count={last_denial.get('count')})"
            )
        if pending_permission:
            lines.append(
                "Pending permission detail: "
                f"{pending_permission.get('tool_name')} "
                f"(risk={pending_permission.get('risk_level')}, mode={pending_permission.get('mode')})"
            )
        if commit_history:
            last_commit = commit_history[0]
            lines.append(
                "Last tracked commit: "
                f"{last_commit.get('status')} "
                f"{last_commit.get('commit_hash') or '<none>'} "
                f"on {last_commit.get('branch') or '<detached>'} "
                f"via {last_commit.get('agent_role') or 'agent'}"
            )
        pending_auto_commit = auto_commit_state.get("pending_proposal") or {}
        if pending_auto_commit:
            lines.append(
                "Pending auto-commit detail: "
                f"{pending_auto_commit.get('branch') or '<detached>'} "
                f"{pending_auto_commit.get('message_preview') or '<no message>'}"
            )
        return "\n".join(lines)

    def _sync_message_context(self):
        self.tool_context["messages"] = self.messages
        self.tool_context["subagent_registry"] = self.subagent_registry
        self.tool_context["in_progress_tools"] = self.in_progress_tools
        self.tool_context["recent_tool_activity"] = self.recent_tool_activity
        self.tool_context["loop_transition_history"] = self.loop_transition_history
        self.tool_context["post_sampling_history"] = self.post_sampling_history
        self.tool_context["permission_state"] = self.permission_manager.export_state()
        self.tool_context["thinking_config"] = self.default_thinking_config
        self.tool_context["mode_thinking_overrides"] = self.mode_thinking_overrides
        self.tool_context["memory_files"] = self.memory_file_manager.export_state()
        self.tool_context["structured_output_state"] = self.structured_output_manager.export_state()
        self.tool_context["security_classifier_state"] = self.security_classifier.export_state()
        self.tool_context["session_id"] = self.session_id
        self.tool_context["model_provider"] = self.model_provider
        self.tool_context["primary_model"] = self.primary_model
        self.tool_context["fallback_model"] = self.fallback_model
        self.tool_context["api_base_url"] = self.api_base_url
        self.tool_context["local_tokenizer_path"] = self.local_tokenizer_path
        self.tool_context["read_file_state"] = {
            path: True for path, entry in self.file_state_cache.export_state().get("entries", {}).items()
            if entry.get("has_been_read")
        }
        self.tool_context["file_state_cache_state"] = self.file_state_cache.export_state()
        self.tool_context["artifact_tracker_state"] = self.artifact_tracker.export_state()
        self.tool_context["incremental_write_queue_state"] = self.incremental_write_queue.export_state()
        self.tool_context["auto_commit_state"] = self.auto_commit_manager.export_state()
        self.tool_context["content_replacement_state"] = self.content_replacement_manager.export_state()
        self.tool_context["vcr_state"] = self.vcr_manager.export_state()

    def _record_tool_start(self, tool_name: str, tool_use_id: str, tool_inputs: dict):
        self.in_progress_tools[tool_use_id] = {
            "tool_name": tool_name,
            "tool_use_id": tool_use_id,
            "status": "running",
            "input_preview": str(tool_inputs)[:220],
        }
        self._sync_message_context()

    def _record_tool_completion(self, tool_name: str, tool_use_id: str, result_preview: str, status: str):
        self.attachment_collector.track_tool_use(tool_name)
        if tool_use_id in self.in_progress_tools:
            del self.in_progress_tools[tool_use_id]

        self.recent_tool_activity.insert(0, {
            "tool_name": tool_name,
            "tool_use_id": tool_use_id,
            "status": status,
            "result_preview": str(result_preview)[:280],
        })
        self.recent_tool_activity = self.recent_tool_activity[:12]
        self._sync_message_context()

    def inherit_state_from_parent(
        self,
        parent_messages=None,
        todo_payload=None,
        plan_payload=None,
    ):
        if parent_messages:
            self.messages = copy.deepcopy(parent_messages)
            self._sync_message_context()
        if todo_payload is not None:
            self.todo_manager.load(copy.deepcopy(todo_payload))
        if plan_payload is not None:
            self.plan_manager.load(
                mode=plan_payload.get("mode", "normal"),
                content=plan_payload.get("content", ""),
            )

    def register_subagent_record(self, record: dict):
        if not isinstance(record, dict):
            return

        agent_id = record.get("agent_id")
        if not agent_id:
            return

        replaced = False
        for index, item in enumerate(self.subagent_registry):
            if item.get("agent_id") == agent_id:
                merged = dict(item)
                merged.update(record)
                self.subagent_registry[index] = merged
                replaced = True
                break

        if not replaced:
            self.subagent_registry.append(record)

        self._sync_message_context()
        self.persist_session_state()

    def _build_message_statistics(self) -> dict:
        stats = {
            "user_messages": 0,
            "assistant_messages": 0,
            "tool_result_blocks": 0,
            "system_observations": 0,
        }

        for msg in self.messages:
            role = msg.get("role")
            if role == "user":
                stats["user_messages"] += 1
            elif role == "assistant":
                stats["assistant_messages"] += 1

            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        stats["tool_result_blocks"] += 1
            elif isinstance(content, str) and content.startswith("[System Observation]"):
                stats["system_observations"] += 1

        return stats

    async def _apply_layered_compaction_if_needed(
        self,
        sys_print_callback=print,
        *,
        force: bool = False,
        aggressive: bool = False,
    ):
        """
        Apply a staged compaction strategy:
        1. Micro-compact older oversized tool results.
        2. Snip stale historical compaction boundaries.
        3. Trim very large historical tool results.
        4. Summarize only the middle history while preserving the first message
           and a recent raw window.
        """
        self.refresh_memory_files()
        effective_thinking_config = self._resolve_thinking_config()
        system_prompt = self.context_builder.generate_system_prompt(
            session_summary=self.plan_manager.render_prompt_summary(),
            todo_summary=self.todo_manager.render_prompt_summary(),
            memory_summary=self.get_memory_summary(),
            structured_output_summary=self.structured_output_manager.render_prompt_summary(),
            tool_prompt_summary=self._build_tool_prompt_summary(),
        )
        tools_schema = self._get_anthropic_tools_schema()
        preserve_last_messages = 8 if aggressive else 12
        micro_compacted_messages, micro_count = self.compactor.micro_compact_tool_results(
            self.messages,
            preserve_last_messages=preserve_last_messages,
            max_tool_result_chars=300 if aggressive else 450,
            preview_chars=120 if aggressive else 160,
        )
        if micro_count > 0:
            self.messages = micro_compacted_messages
            self._sync_message_context()
            self.persist_session_state()
            sys_print_callback(
                f"[dim magenta]↳ Micro-compacted {micro_count} historical tool result(s).[/dim magenta]"
            )

        snipped_messages, snipped_count = self.compactor.snip_old_compaction_boundaries(
            self.messages,
            preserve_last_messages=preserve_last_messages,
            keep_recent_boundaries=1,
        )
        if snipped_count > 0:
            self.messages = snipped_messages
            self._sync_message_context()
            self.persist_session_state()
            sys_print_callback(
                f"[dim magenta]↳ Snipped {snipped_count} stale compaction boundary message(s).[/dim magenta]"
            )

        max_tool_result_chars = 800 if aggressive else 1200
        trimmed_messages, trimmed_count = self.compactor.prune_large_tool_results(
            self.messages,
            preserve_last_messages=preserve_last_messages,
            max_tool_result_chars=max_tool_result_chars,
        )
        if trimmed_count > 0:
            self.messages = trimmed_messages
            self._sync_message_context()
            self.persist_session_state()
            sys_print_callback(
                f"[dim magenta]↳ Trimmed {trimmed_count} oversized historical tool result(s) before compaction.[/dim magenta]"
            )

        cache_signal = self._build_cache_compaction_signal()
        api_messages = self._clean_roles_for_api(self.messages)
        context_window = int(os.environ.get("CODECLAW_CONTEXT_WINDOW", "0")) or 200000
        budget = int(context_window * 0.75) if aggressive else int(context_window * 0.80)
        reserve = int(context_window * 0.08) if aggressive else int(context_window * 0.06)
        should_compact, estimated_tokens, estimate_source = await self.compactor.should_compact_precise(
            api_messages,
            model=self.primary_model,
            system=system_prompt,
            tools=tools_schema,
            thinking=effective_thinking_config,
            token_budget=budget,
            reserve_tokens=reserve,
            hard_message_count=80 if aggressive else 100,
            cache_read_input_tokens=cache_signal.get("recent_cache_read_input_tokens", 0),
            cache_creation_input_tokens=cache_signal.get("recent_cache_creation_input_tokens", 0),
            max_cache_penalty_tokens=14000 if aggressive else 10000,
        )
        if not force and not should_compact:
            return False

        sys_print_callback(
            "\n[bold magenta]⎔ Context Window Critical: Triggering layered context compaction...[/bold magenta]\n"
            f"[dim magenta]↳ Estimated context tokens: {estimated_tokens} (source={estimate_source})[/dim magenta]\n"
            f"[dim magenta]↳ Cache-aware pressure: {cache_signal.get('cache_pressure_tokens', 0)} "
            f"(creation={cache_signal.get('recent_cache_creation_input_tokens', 0)}, "
            f"read={cache_signal.get('recent_cache_read_input_tokens', 0)})[/dim magenta]"
        )
        head, middle, tail = self.compactor.split_for_layered_compaction(
            self.messages,
            preserve_first_messages=1,
            preserve_last_messages=preserve_last_messages,
        )
        if not middle:
            return False

        summary = await self.compactor.compact_history(middle)
        stage_label = "aggressive_layered_summary" if aggressive else "layered_summary"
        reminder = {
            "role": "user",
            "content": (
                "<system-reminder>\n"
                f"[CompactionBoundary:{stage_label}]\n"
                f"estimated_tokens_before: {estimated_tokens}\n"
                f"cache_pressure_tokens: {cache_signal.get('cache_pressure_tokens', 0)}\n"
                f"cache_creation_input_tokens: {cache_signal.get('recent_cache_creation_input_tokens', 0)}\n"
                f"cache_read_input_tokens: {cache_signal.get('recent_cache_read_input_tokens', 0)}\n"
                f"preserved_first_messages: 1\n"
                f"preserved_last_messages: {preserve_last_messages}\n"
                "Compaction summary:\n"
                f"{summary}\n"
                "</system-reminder>"
            ),
        }
        self.messages = head + [reminder] + tail
        self._update_compact_boundary()
        self._sync_message_context()
        self.persist_session_state()
        sys_print_callback(
            "[dim magenta]↳ Layered compaction complete. Preserved recent raw context and summarized older history.[/dim magenta]\n"
        )

        post_summary_api_messages = self._clean_roles_for_api(self.messages)
        collapse_needed, collapsed_estimated_tokens, collapsed_estimate_source = await self.compactor.should_compact_precise(
            post_summary_api_messages,
            model=self.primary_model,
            system=system_prompt,
            tools=tools_schema,
            thinking=effective_thinking_config,
            token_budget=44000 if aggressive else 52000,
            reserve_tokens=12000 if aggressive else 10000,
            hard_message_count=18 if aggressive else 24,
            cache_read_input_tokens=cache_signal.get("recent_cache_read_input_tokens", 0),
            cache_creation_input_tokens=cache_signal.get("recent_cache_creation_input_tokens", 0),
            max_cache_penalty_tokens=12000 if aggressive else 8000,
        )
        if collapse_needed:
            collapsed_messages, collapse_meta = self.compactor.context_collapse(
                self.messages,
                preserve_first_messages=1,
                preserve_last_messages=preserve_last_messages,
                max_projection_items=8 if aggressive else 10,
                text_chars=140 if aggressive else 180,
            )
            if collapse_meta.get("collapsed"):
                self.messages = collapsed_messages
                self._update_compact_boundary()
                self._sync_message_context()
                self.persist_session_state()
                sys_print_callback(
                    "[dim magenta]↳ Context collapse applied after layered summary to preserve only a projection of older history.[/dim magenta]\n"
                    f"[dim magenta]↳ Post-summary estimated tokens before collapse: {collapsed_estimated_tokens} "
                    f"(source={collapsed_estimate_source})[/dim magenta]"
                )

        restored = self._post_compact_restore_state(sys_print_callback)
        if restored:
            self._sync_message_context()
            self.persist_session_state()
        return True

    POST_COMPACT_MAX_FILES = 5
    POST_COMPACT_TOKEN_BUDGET = 50_000
    POST_COMPACT_MAX_TOKENS_PER_FILE = 5_000

    def _post_compact_restore_state(self, sys_print_callback=print) -> bool:
        """
        Re-inject critical context after compaction so the agent doesn't lose
        awareness of recently-read files, the active plan, or loaded skills.
        Mirrors Claude Code's post-compact state recovery in compact.ts.
        """
        restore_parts = []
        budget_remaining = self.POST_COMPACT_TOKEN_BUDGET

        recently_read = self._collect_recently_read_files()
        files_restored = 0
        for path, content in recently_read:
            if files_restored >= self.POST_COMPACT_MAX_FILES:
                break
            token_est = len(content) // 4
            capped = min(token_est, self.POST_COMPACT_MAX_TOKENS_PER_FILE, budget_remaining)
            if capped <= 0:
                break
            char_limit = capped * 4
            snippet = content[:char_limit]
            if len(content) > char_limit:
                snippet += "\n... [truncated for post-compact budget]"
            restore_parts.append(f"[Recently read file: {path}]\n{snippet}")
            budget_remaining -= capped
            files_restored += 1

        plan_content = self.plan_manager.render_prompt_summary()
        if plan_content:
            plan_tokens = len(plan_content) // 4
            if plan_tokens <= budget_remaining:
                restore_parts.append(f"[Active plan state]\n{plan_content}")
                budget_remaining -= plan_tokens

        todo_content = self.todo_manager.render_prompt_summary()
        if todo_content:
            todo_tokens = len(todo_content) // 4
            if todo_tokens <= budget_remaining:
                restore_parts.append(f"[Active todo state]\n{todo_content}")
                budget_remaining -= todo_tokens

        mcp_instructions = self._build_mcp_instructions()
        if mcp_instructions:
            mcp_tokens = len(mcp_instructions) // 4
            if mcp_tokens <= budget_remaining:
                restore_parts.append(f"[MCP tool instructions]\n{mcp_instructions}")
                budget_remaining -= mcp_tokens

        if not restore_parts:
            return False

        restore_text = (
            "<system-reminder>\n"
            "[Post-compaction context restoration]\n"
            "The conversation was compacted. The following recently-accessed context "
            "is re-injected so you can continue without re-reading these resources.\n\n"
            + "\n\n".join(restore_parts)
            + "\n</system-reminder>"
        )
        self.messages.append(create_user_message(
            restore_text,
            is_meta=True,
            origin="post_compact_restore",
        ))
        sys_print_callback(
            f"[dim magenta]↳ Post-compact: restored {files_restored} file(s), "
            f"plan={'yes' if plan_content else 'no'}, "
            f"todos={'yes' if todo_content else 'no'}, "
            f"mcp={'yes' if mcp_instructions else 'no'}.[/dim magenta]"
        )
        return True

    def _collect_recently_read_files(self) -> list:
        """Return (path, content) pairs for files tracked by file_state_cache, most recent first."""
        entries = self.file_state_cache.entries
        if not entries:
            return []

        candidates = []
        for abs_path, entry in entries.items():
            if not entry.get("has_been_read"):
                continue
            if not os.path.isfile(abs_path):
                continue
            mtime = entry.get("mtime", 0)
            candidates.append((mtime, abs_path))

        candidates.sort(key=lambda x: x[0], reverse=True)

        result = []
        for _, abs_path in candidates[:self.POST_COMPACT_MAX_FILES * 2]:
            try:
                with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                result.append((abs_path, content))
            except Exception:
                continue
        return result

    def _is_context_overflow_error(self, error: Exception) -> bool:
        rendered = str(error).lower()
        markers = [
            "prompt is too long",
            "context length",
            "too many tokens",
            "maximum context length",
            "input is too long",
            "context window exceeds",
            "exceed context limit",
            "context_length_exceeded",
            "input length and `max_tokens` exceed",
        ]
        return any(marker in rendered for marker in markers)

    MAX_TOOL_CONCURRENCY = 10
    SIBLING_CANCEL_TOOLS = frozenset({"bash_tool"})

    def _is_concurrency_safe(self, tool_name: str, tool_inputs: dict) -> bool:
        handler = self.available_tools.get(tool_name)
        if not handler:
            return False

        inspector = getattr(handler, "is_concurrency_safe_call", None)
        if callable(inspector):
            try:
                return bool(inspector(**tool_inputs))
            except Exception:
                return False

        return False

    @staticmethod
    def _tb_get(tool_block, key, default=""):
        if isinstance(tool_block, dict):
            return tool_block.get(key, default)
        return getattr(tool_block, key, default)

    def _partition_tool_batches(self, tool_calls):
        batches = []
        for tool_block in tool_calls:
            name = self._tb_get(tool_block, "name", "")
            inp = self._tb_get(tool_block, "input", {}) or {}
            is_safe = self._is_concurrency_safe(name, inp)
            if is_safe and batches and batches[-1]["is_concurrency_safe"]:
                batches[-1]["blocks"].append(tool_block)
            else:
                batches.append({
                    "is_concurrency_safe": is_safe,
                    "blocks": [tool_block],
                })
        return batches

    async def _execute_tool_wrapper(self, tool_block, sys_print_callback=print, *, turn: int = 0, event_callback=None):
        tool_name = self._tb_get(tool_block, "name", "")
        tool_id = self._tb_get(tool_block, "id", "")
        tool_inputs = copy.deepcopy(self._tb_get(tool_block, "input", {}) or {})
        spill_recorder = self._build_spill_recorder(tool_name=tool_name, tool_use_id=tool_id)

        sys_print_callback(f"  [dim cyan]↳ calling {tool_name}[/dim cyan]")
        self._record_tool_start(tool_name, tool_id, tool_inputs)
        before_tool_decision = self.hook_manager.evaluate_before_tool_hooks({
            "tool_name": tool_name,
            "tool_use_id": tool_id,
            "tool_input": tool_inputs,
            "mode": self.get_mode(),
            "agent_id": self.agent_id,
            "agent_role": self.agent_role,
        })
        before_tool_prefix = ""
        if before_tool_decision.outputs:
            before_tool_prefix = "\n".join(
                f"[Hook:BeforeToolUse #{index}] {content}"
                for index, content in enumerate(before_tool_decision.outputs, start=1)
            )
            before_tool_prefix += "\n\n"
        tool_inputs = copy.deepcopy(before_tool_decision.updated_input or tool_inputs)

        if before_tool_decision.history:
            await self._record_loop_transition(
                "before_tool_hook",
                turn=turn,
                event_callback=event_callback,
                tool_name=tool_name,
                tool_use_id=tool_id,
                behavior=before_tool_decision.behavior,
                hook_count=len(before_tool_decision.history),
                input_changed=any(item.get("input_changed") for item in before_tool_decision.history),
                reason=before_tool_decision.reason,
            )

        if tool_name not in self.available_tools:
            res_text = f"System Error: Tool '{tool_name}' not found."
        elif before_tool_decision.behavior == "reject":
            res_text = serialize_tool_result(
                build_tool_result(
                    ok=False,
                    content=before_tool_decision.reason or f"Tool '{tool_name}' rejected by BeforeToolUse hook.",
                    metadata={
                        "tool_name": tool_name,
                        "tool_use_id": tool_id,
                        "hook_behavior": before_tool_decision.behavior,
                        "hook_count": len(before_tool_decision.history or []),
                        "hook_updated_input": tool_inputs,
                    },
                    is_error=True,
                ),
                artifact_dir=os.path.join(".codeclaw", "tool-results", self.session_id),
                spill_recorder=spill_recorder,
            )
        else:
            try:
                effective_tools = self._get_effective_tools()
                if tool_name not in effective_tools:
                    raise KeyError(f"Tool '{tool_name}' is not available in current mode.")
                handler = effective_tools[tool_name]
                permission = await self.permission_manager.authorize(
                    handler,
                    tool_inputs,
                    context={
                        "turn": turn,
                        "tool_use_id": tool_id,
                        "agent_id": self.agent_id,
                        "agent_role": self.agent_role,
                    },
                )
                if permission.behavior != "allow":
                    await self._record_loop_transition(
                        "permission_denied",
                        turn=turn,
                        event_callback=event_callback,
                        tool_name=tool_name,
                        tool_use_id=tool_id,
                        source=permission.source,
                        reason=permission.reason,
                        denial_count=(permission.metadata or {}).get("denial_count_after"),
                    )
                    res_text = serialize_tool_result(
                        build_tool_result(
                            ok=False,
                            content=(
                                permission.reason
                                or f"Permission denied for tool '{tool_name}'."
                            ),
                            metadata={
                                "tool_name": tool_name,
                                "tool_use_id": tool_id,
                                "permission_behavior": permission.behavior,
                                "permission_source": permission.source,
                                **dict(permission.metadata or {}),
                            },
                            is_error=True,
                        ),
                        artifact_dir=os.path.join(".codeclaw", "tool-results", self.session_id),
                        spill_recorder=spill_recorder,
                    )
                else:
                    raw_result = await handler(**tool_inputs)
                    res_text = serialize_tool_result(
                        raw_result,
                        artifact_dir=os.path.join(".codeclaw", "tool-results", self.session_id),
                        spill_recorder=spill_recorder,
                    )
            except Exception as e:
                res_text = f"Tool Exception Caught: {str(e)}"

        if before_tool_prefix:
            res_text = before_tool_prefix + str(res_text)

        status = "error" if str(res_text).startswith("Permission denied") or "Status: error" in str(res_text) or "Tool Exception Caught" in str(res_text) else "ok"
        self._record_tool_completion(tool_name, tool_id, res_text, status)
        sys_print_callback(f"    [dim]{tool_name} -> {status}[/dim]")

        return {
            "type": "tool_result",
            "tool_use_id": tool_id,
            "content": res_text,
            "is_error": status == "error",
        }

    async def _execute_tool_batches(self, tool_calls, sys_print_callback=print, *, turn: int = 0, event_callback=None):
        results = []
        batches = self._partition_tool_batches(tool_calls)
        semaphore = asyncio.Semaphore(self.MAX_TOOL_CONCURRENCY)

        for batch in batches:
            blocks = batch["blocks"]
            if batch["is_concurrency_safe"]:
                sys_print_callback(
                    f"[bold dim yellow]❯ Executing {len(blocks)} concurrency-safe tool call(s) in parallel "
                    f"(max concurrency={self.MAX_TOOL_CONCURRENCY})...[/bold dim yellow]"
                )
                sibling_cancel = asyncio.Event()

                async def _run_with_sibling_cancel(tc):
                    async with semaphore:
                        if sibling_cancel.is_set():
                            tool_id = self._tb_get(tc, "id", "")
                            tool_name = self._tb_get(tc, "name", "")
                            return {
                                "type": "tool_result",
                                "tool_use_id": tool_id,
                                "content": f"Cancelled: sibling tool in batch failed ({tool_name}).",
                                "is_error": True,
                            }
                        res = await self._execute_tool_wrapper(
                            tc, sys_print_callback, turn=turn, event_callback=event_callback,
                        )
                        if res.get("is_error") and self._tb_get(tc, "name", "") in self.SIBLING_CANCEL_TOOLS:
                            sibling_cancel.set()
                        return res

                batch_results = await asyncio.gather(
                    *(_run_with_sibling_cancel(tc) for tc in blocks)
                )
                results.extend(batch_results)
            else:
                sys_print_callback(
                    f"[bold dim yellow]❯ Executing {len(blocks)} mutating/high-risk tool call(s) serially...[/bold dim yellow]"
                )
                for tool_block in blocks:
                    results.append(
                        await self._execute_tool_wrapper(
                            tool_block,
                            sys_print_callback,
                            turn=turn,
                            event_callback=event_callback,
                        )
                    )

        return results
            
    async def start_protocols(self, sys_print=print):
        # 1. Start LSP Daemon Support
        self.lsp_manager = LSPManager()
        self.tool_context["lsp_manager"] = self.lsp_manager
        lsp_t = LspTool()
        lsp_t.context = self.tool_context
        self.available_tools["lsp_tool"] = lsp_t
        sys_print("[dim green]↳ LSP Manager Established (Passive Telemetry & AST active)[/dim green]")
        
        # 2. Start MCP Background Connections
        self.mcp_bridge = MCPBridge()
        remote_tools = await self.mcp_bridge.load_and_connect()
        if remote_tools:
            for name, tool in remote_tools.items():
                tool.context = self.tool_context
                self.available_tools[name] = tool
            sys_print(f"[dim green]↳ Protocol Matrix Online: Loaded {len(remote_tools)} Remote MCP Tools.[/dim green]")
        
    def _get_anthropic_tools_schema(self):
        """Converts registered Pydantic tool classes into provider-native schemas."""
        effective_tools = self._get_effective_tools()
        schemas = []
        for tool in effective_tools.values():
            definition = tool.get_tool_definition()
            if self.model_provider == "openai":
                schemas.append(self._openai_tool_definition(definition))
            else:
                schemas.append(definition)
        return schemas

    def _block_to_dict(self, block):
        if isinstance(block, dict):
            return copy.deepcopy(block)
        if isinstance(block, str):
            return {"type": "text", "text": block}

        block_type = getattr(block, "type", None)
        if block_type == "text":
            return {
                "type": "text",
                "text": getattr(block, "text", ""),
            }
        if block_type == "tool_use":
            return {
                "type": "tool_use",
                "id": getattr(block, "id", ""),
                "name": getattr(block, "name", ""),
                "input": copy.deepcopy(getattr(block, "input", {}) or {}),
            }
        if block_type == "thinking":
            return {
                "type": "thinking",
                "thinking": getattr(block, "thinking", ""),
                "signature": getattr(block, "signature", ""),
            }
        if block_type == "redacted_thinking":
            return {
                "type": "redacted_thinking",
                "data": getattr(block, "data", ""),
            }

        text_value = getattr(block, "text", None)
        if isinstance(text_value, str):
            return {"type": "text", "text": text_value}
        return None

    def _merge_adjacent_text_blocks(self, blocks: list) -> list:
        merged = []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            if (
                merged
                and merged[-1].get("type") == "text"
                and block.get("type") == "text"
            ):
                merged[-1]["text"] = (
                    f"{merged[-1].get('text', '')}{block.get('text', '')}"
                )
                continue
            merged.append(block)
        return merged

    def _normalize_message_content_for_api(self, role: str, content):
        if isinstance(content, str):
            normalized = [{"type": "text", "text": content}]
        elif isinstance(content, list):
            normalized = []
            for raw_block in content:
                block = self._block_to_dict(raw_block)
                if not block:
                    continue

                block_type = block.get("type")
                if block_type in {"thinking", "redacted_thinking"}:
                    # Do not feed internal reasoning or signatures back into later turns.
                    continue

                if block_type == "text":
                    text = block.get("text", "")
                    if text:
                        normalized.append({"type": "text", "text": text})
                    continue

                if block_type == "tool_use" and role == "assistant":
                    tool_name = block.get("name")
                    tool_use_id = block.get("id")
                    if tool_name and tool_use_id:
                        normalized.append({
                            "type": "tool_use",
                            "id": tool_use_id,
                            "name": tool_name,
                            "input": copy.deepcopy(block.get("input", {}) or {}),
                        })
                    continue

                if block_type == "tool_result" and role == "user":
                    tool_use_id = block.get("tool_use_id")
                    if not tool_use_id:
                        continue

                    block_content = block.get("content", "")
                    if isinstance(block_content, list):
                        parts = []
                        for part in block_content:
                            part_block = self._block_to_dict(part)
                            if isinstance(part_block, dict) and part_block.get("type") == "text":
                                parts.append(part_block.get("text", ""))
                        block_content = "".join(parts)
                    elif block_content is None:
                        block_content = ""
                    else:
                        block_content = str(block_content)

                    normalized.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": block_content,
                        "is_error": bool(block.get("is_error", False)),
                    })
                    continue

                if block_type in {"image", "document"} and role == "user":
                    normalized.append(block)
                    continue

                fallback_text = block.get("text") or block.get("content")
                if fallback_text:
                    normalized.append({"type": "text", "text": str(fallback_text)})
        else:
            normalized = [{"type": "text", "text": str(content)}]

        normalized = self._merge_adjacent_text_blocks(normalized)
        return [block for block in normalized if isinstance(block, dict)]

    def _filter_incomplete_tool_messages_for_api(self, messages):
        """
        Ensure tool_use / tool_result pairs are always complete.
        Drop assistant messages whose tool_use has no matching tool_result,
        AND strip tool_result blocks whose tool_use_id has no matching tool_use.
        """
        if not messages:
            return []

        tool_use_ids_all = set()
        tool_result_ids_all = set()
        for msg in messages:
            for block in (msg.get("content", []) if isinstance(msg.get("content"), list) else []):
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use" and block.get("id"):
                    tool_use_ids_all.add(block["id"])
                if block.get("type") == "tool_result" and block.get("tool_use_id"):
                    tool_result_ids_all.add(block["tool_use_id"])

        filtered = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", [])

            if role == "assistant" and isinstance(content, list):
                use_ids = [
                    b.get("id") for b in content
                    if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("id")
                ]
                if use_ids and any(uid not in tool_result_ids_all for uid in use_ids):
                    continue

            if role == "user" and isinstance(content, list):
                cleaned_content = []
                for block in content:
                    if (isinstance(block, dict)
                            and block.get("type") == "tool_result"
                            and block.get("tool_use_id")
                            and block["tool_use_id"] not in tool_use_ids_all):
                        continue
                    cleaned_content.append(block)
                if not cleaned_content:
                    continue
                msg = {**msg, "content": cleaned_content}

            filtered.append(msg)

        return filtered

    def _clean_roles_for_api(self, messages):
        """Normalize message content and collapse adjacent roles for Anthropic API calls."""
        normalized_messages = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue

            role = msg.get("role")
            if role not in {"user", "assistant"}:
                continue

            normalized_content = self._normalize_message_content_for_api(
                role,
                msg.get("content", ""),
            )
            if not normalized_content:
                continue

            normalized_messages.append({
                "role": role,
                "content": normalized_content,
            })

        normalized_messages = self._filter_incomplete_tool_messages_for_api(normalized_messages)
        normalized_messages = self._strip_error_causing_attachments(normalized_messages)

        collapsed = []
        for msg in normalized_messages:
            if not collapsed:
                collapsed.append(msg)
                continue
            if collapsed[-1]["role"] == msg["role"]:
                collapsed[-1]["content"] = self._merge_adjacent_text_blocks(
                    collapsed[-1]["content"] + msg["content"]
                )
            else:
                collapsed.append(msg)

        collapsed = self._filter_orphaned_thinking_only_messages(collapsed)
        collapsed = self._strip_trailing_thinking_from_last_assistant(collapsed)
        collapsed = self._filter_whitespace_only_assistant_messages(collapsed)
        collapsed = self._ensure_non_empty_assistant_content(collapsed)
        return collapsed

    # ------------------------------------------------------------------
    # Post-normalization sanitizers (prevent API 400 in long conversations)
    # ------------------------------------------------------------------

    _ERROR_ATTACHMENT_MARKERS = {
        "pdf is too large": {"document"},
        "document is too large": {"document"},
        "image exceeds": {"image"},
        "image is too large": {"image"},
        "request too large": {"document", "image"},
        "request_too_large": {"document", "image"},
        "payload too large": {"document", "image"},
    }

    def _strip_error_causing_attachments(self, messages: list) -> list:
        """Remove document/image blocks from user messages that triggered 413.

        When a large PDF or image causes a prompt-too-large error, the API
        returns a synthetic error assistant message.  If the offending
        attachment stays in history, every subsequent turn will also 413.
        This method back-tracks from each error message and strips the
        block types that caused it from the nearest prior user message.
        """
        strip_targets: dict[int, set[str]] = {}
        for i, msg in enumerate(messages):
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "text":
                    continue
                text_lower = block.get("text", "").lower()
                for marker, block_types in self._ERROR_ATTACHMENT_MARKERS.items():
                    if marker in text_lower:
                        for j in range(i - 1, -1, -1):
                            if messages[j].get("role") == "user":
                                strip_targets.setdefault(j, set()).update(block_types)
                                break

        if not strip_targets:
            return messages

        result = []
        for i, msg in enumerate(messages):
            if i not in strip_targets:
                result.append(msg)
                continue
            types_to_strip = strip_targets[i]
            content = msg.get("content", [])
            if not isinstance(content, list):
                result.append(msg)
                continue
            cleaned = [
                b for b in content
                if not (isinstance(b, dict) and b.get("type") in types_to_strip)
            ]
            if cleaned:
                result.append({**msg, "content": cleaned})
        return result

    def _filter_orphaned_thinking_only_messages(self, messages: list) -> list:
        """Remove assistant messages that contain no substantive content.

        After compaction or streaming fallback, the history may contain
        assistant messages that originally had thinking + text/tool_use but
        lost their non-thinking blocks.  Consecutive assistant messages with
        mismatched thinking signatures cause API 400 errors.
        """
        filtered = []
        for msg in messages:
            if msg.get("role") != "assistant":
                filtered.append(msg)
                continue
            content = msg.get("content", [])
            if not isinstance(content, list):
                filtered.append(msg)
                continue
            has_substantive = any(
                isinstance(b, dict) and b.get("type") in {"text", "tool_use"}
                for b in content
            )
            if has_substantive:
                filtered.append(msg)
        return filtered

    def _strip_trailing_thinking_from_last_assistant(self, messages: list) -> list:
        """Strip thinking/redacted_thinking blocks from the last assistant message.

        The API rejects trailing thinking blocks that lack a valid signature
        continuation.  Order matters: this runs BEFORE the whitespace-only
        filter so that a message like [text("\\n"), thinking("...")] doesn't
        survive the whitespace check only to be rejected later.
        """
        if not messages:
            return messages
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "assistant":
                content = messages[i].get("content", [])
                if isinstance(content, list):
                    cleaned = [
                        b for b in content
                        if not (isinstance(b, dict)
                                and b.get("type") in {"thinking", "redacted_thinking"})
                    ]
                    if cleaned != content:
                        messages[i] = {**messages[i], "content": cleaned}
                break
        return messages

    def _filter_whitespace_only_assistant_messages(self, messages: list) -> list:
        """Drop assistant messages whose text content is only whitespace."""
        filtered = []
        for msg in messages:
            if msg.get("role") != "assistant":
                filtered.append(msg)
                continue
            content = msg.get("content", [])
            if not isinstance(content, list):
                filtered.append(msg)
                continue
            has_non_whitespace = False
            for b in content:
                if not isinstance(b, dict):
                    continue
                btype = b.get("type")
                if btype == "tool_use":
                    has_non_whitespace = True
                    break
                if btype == "text" and b.get("text", "").strip():
                    has_non_whitespace = True
                    break
            if has_non_whitespace:
                filtered.append(msg)
        return filtered

    def _ensure_non_empty_assistant_content(self, messages: list) -> list:
        """Guarantee every assistant message has at least one content block."""
        filtered = []
        for msg in messages:
            if msg.get("role") == "assistant":
                content = msg.get("content", [])
                if isinstance(content, list) and not content:
                    continue
            filtered.append(msg)
        return filtered

    def _strip_thinking_signatures_from_history(self):
        """Remove thinking/redacted_thinking blocks from all assistant messages.

        Thinking signatures are model-bound: replaying a protected-thinking
        block produced by one model (e.g. sonnet) to a different fallback
        model (e.g. opus) causes API 400 errors.  This must be called before
        retrying with a different model.
        """
        changed = False
        for msg in self.messages:
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            cleaned = [
                b for b in content
                if not (isinstance(b, dict)
                        and b.get("type") in {"thinking", "redacted_thinking"})
            ]
            if len(cleaned) != len(content):
                msg["content"] = cleaned
                changed = True
        if changed:
            self._sync_message_context()

    def load_session(self, sid: str) -> bool:
        state = self.session_manager.load_session_state(sid)
        msgs = state.get("messages", [])
        if msgs:
            self.session_id = sid
            self.last_persist_ok = self.session_manager.last_save_ok
            self.last_persist_error = self.session_manager.last_save_error
            self.messages = msgs
            self._sync_message_context()
            metadata = state.get("metadata", {})
            model_runtime = metadata.get("model_runtime", {}) or {}
            if model_runtime:
                self.model_provider = self._resolve_model_provider(
                    model_runtime.get("provider"),
                    model_runtime.get("primary_model"),
                )
                self.api_base_url = str(model_runtime.get("api_base_url", self.api_base_url) or "")
                self.local_tokenizer_path = str(
                    model_runtime.get("local_tokenizer_path", self.local_tokenizer_path) or ""
                )
                self.primary_model = str(
                    model_runtime.get("primary_model", self.primary_model) or self.primary_model
                )
                self.fallback_model = str(
                    model_runtime.get("fallback_model", self.fallback_model) or self.fallback_model
                )
                self.model = self.primary_model
                self.client = self._build_model_client()
                self.compactor = MemoryCompactor(
                    model=self.fallback_model or self.primary_model,
                    provider=self.model_provider,
                    api_base_url=self.api_base_url,
                    local_tokenizer_path=self.local_tokenizer_path,
                )
                self._sync_model_runtime_context()
            self.todo_manager.load(metadata.get("todos", []))
            plan_state = metadata.get("plan", {})
            self.plan_manager.load(
                mode=plan_state.get("mode", "normal"),
                content=plan_state.get("content", ""),
            )
            self.sandbox_manager.load_state(metadata.get("sandbox", {}))
            self.browser_manager.load_state(metadata.get("browser", {}))
            self.shell_task_manager.load_state(metadata.get("shell_tasks", []))
            self.permission_manager.load_state(metadata.get("permissions", {}))
            self.memory_file_manager.load_state(metadata.get("memory_files", {}))
            self.file_state_cache.load_state(metadata.get("file_state_cache", {}))
            self.artifact_tracker.load_state(metadata.get("artifacts", {}))
            self.incremental_write_queue.load_state(metadata.get("incremental_write_queue", {}))
            self.auto_commit_manager.load_state(metadata.get("auto_commit", {}))
            self.content_replacement_manager.set_session(self.session_id)
            self.content_replacement_manager.load_state(metadata.get("content_replacements", {}))
            self.content_replacement_manager.cleanup_orphans()
            self.structured_output_manager.load_state(metadata.get("structured_output", {}))
            self.security_classifier.load_state(metadata.get("security_classifier", {}))
            self.vcr_manager.load_state(metadata.get("vcr", {}))
            thinking_state = metadata.get("thinking", {})
            self.default_thinking_config = self._normalize_thinking_config(
                thinking_state.get("default", self.default_thinking_config)
            ) or {"type": "disabled"}
            loaded_mode_overrides = {}
            for mode_name, config in dict(thinking_state.get("mode_overrides", {}) or {}).items():
                normalized = self._normalize_thinking_config(config)
                if normalized is not None:
                    loaded_mode_overrides[str(mode_name)] = normalized
            if loaded_mode_overrides:
                self.mode_thinking_overrides = loaded_mode_overrides
            self.subagent_registry = metadata.get("subagents", [])
            self.team_manager.load_state(metadata.get("team", {}))
            self.ultraplan_manager.load_state(metadata.get("ultraplan", {}))
            self.coordinator_mode = metadata.get("coordinator_mode", False)
            self.loop_transition_history = metadata.get("loop_transitions", [])
            self.token_usage_history = metadata.get("token_usage_history", [])
            self.post_sampling_history = metadata.get("post_sampling_history", [])
            self.session_token_usage = metadata.get("session_token_usage", {
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            })
            self.tool_context["token_usage_history"] = self.token_usage_history
            self.tool_context["session_token_usage"] = self.session_token_usage
            self._update_compact_boundary()
            self._sync_message_context()
            self.session_started = metadata.get("hooks", {}).get("session_started", False)
            return True
        return False

    def _build_session_metadata(self) -> dict:
        return {
            "todos": self.todo_manager.export(),
            "plan": self.plan_manager.export(),
            "sandbox": self.sandbox_manager.export_state(),
            "browser": self.browser_manager.export_state(),
            "shell_tasks": self.shell_task_manager.export_state(),
            "permissions": self.permission_manager.export_state(),
            "memory_files": self.memory_file_manager.export_state(),
            "file_state_cache": self.file_state_cache.export_state(),
            "artifacts": self.artifact_tracker.export_state(),
            "incremental_write_queue": self.incremental_write_queue.export_state(),
            "auto_commit": self.auto_commit_manager.export_state(),
            "content_replacements": self.content_replacement_manager.export_state(),
            "structured_output": self.structured_output_manager.export_state(),
            "security_classifier": self.security_classifier.export_state(),
            "vcr": self.vcr_manager.export_state(),
            "thinking": {
                "default": self.default_thinking_config,
                "mode_overrides": self.mode_thinking_overrides,
            },
            "subagents": self.subagent_registry,
            "team": self.team_manager.export_state(),
            "ultraplan": self.ultraplan_manager.export_state(),
            "coordinator_mode": self.coordinator_mode,
            "loop_transitions": self.loop_transition_history,
            "token_usage_history": self.token_usage_history,
            "post_sampling_history": self.post_sampling_history,
            "session_token_usage": self.session_token_usage,
            "message_statistics": self._build_message_statistics(),
            "hooks": {
                "session_started": getattr(self, "session_started", False),
            },
            "agent": {
                "agent_id": self.agent_id,
                "parent_agent_id": self.parent_agent_id,
                "agent_depth": self.agent_depth,
                "agent_role": self.agent_role,
            },
            "model_runtime": {
                "provider": self.model_provider,
                "primary_model": self.primary_model,
                "fallback_model": self.fallback_model,
                "api_base_url": self.api_base_url,
                "local_tokenizer_path": self.local_tokenizer_path,
            },
        }

    def persist_session_state(self):
        if self.agent_depth > 0:
            self.session_manager.save_subagent_transcript(
                self.agent_id or self.session_id,
                {"messages": self.messages, "metadata": self._build_session_metadata()},
            )
            return

        ok = self.session_manager.save_session_state(
            self.session_id,
            self.messages,
            metadata=self._build_session_metadata(),
        )
        self.last_persist_ok = ok
        self.last_persist_error = self.session_manager.last_save_error

        if not ok and self.last_persist_error != self._last_reported_persist_error:
            print(f"Warning: failed to persist session state: {self.last_persist_error}")
            self._last_reported_persist_error = self.last_persist_error

        if ok:
            self._last_reported_persist_error = ""

        return ok

    def get_resume_summary(self) -> str:
        self.refresh_memory_files()
        self.content_replacement_manager.cleanup_orphans()
        metadata = self._build_session_metadata()
        plan = metadata.get("plan", {})
        todos = metadata.get("todos", [])
        sandbox = metadata.get("sandbox", {})
        browser = metadata.get("browser", {})
        subagents = metadata.get("subagents", [])
        shell_tasks = metadata.get("shell_tasks", [])
        loop_transitions = metadata.get("loop_transitions", [])
        post_sampling_history = metadata.get("post_sampling_history", [])
        permissions = metadata.get("permissions", {})
        pending_permission = permissions.get("pending_request") or {}
        memory_files = metadata.get("memory_files", {})
        file_state_cache = metadata.get("file_state_cache", {})
        artifacts = metadata.get("artifacts", {})
        commit_history = artifacts.get("commit_history", []) or []
        incremental_write_queue = metadata.get("incremental_write_queue", {})
        auto_commit = metadata.get("auto_commit", {})
        content_replacements = metadata.get("content_replacements", {})
        structured_output = metadata.get("structured_output", {})
        security_classifier = metadata.get("security_classifier", {})
        vcr = metadata.get("vcr", {})
        thinking = metadata.get("thinking", {})
        session_token_usage = metadata.get("session_token_usage", {})
        stats = metadata.get("message_statistics", {})
        effective_resume_thinking = thinking.get(
            "mode_overrides",
            {},
        ).get(
            plan.get("mode", "normal"),
            thinking.get("default", self.default_thinking_config),
        )
        artifact_file_change_count = sum(
            len(items) for items in dict(artifacts.get("file_history", {}) or {}).values()
        )

        lines = [
            f"Session ID: {self.session_id}",
            f"Mode: {plan.get('mode', 'normal')}",
            f"Model provider: {self.model_provider}",
            f"Primary model: {self.primary_model}",
            f"Fallback model: {self.fallback_model}",
            f"Thinking config: {self._thinking_config_to_text(effective_resume_thinking)}",
            f"Token estimator: {self.compactor.token_estimator_mode()}",
            f"Local tokenizer path: {self.local_tokenizer_path or 'none'}",
            f"Memory files loaded: {len(memory_files.get('loaded_files', []) or [])}",
            f"Cached file states: {len(file_state_cache.get('entries', {}) or {})}",
            f"Tracked attachments: {len(artifacts.get('attachments', []) or [])}",
            f"Tracked prefetches: {len(artifacts.get('prefetch_history', []) or [])}",
            f"Tracked file changes: {artifact_file_change_count}",
            f"Tracked commits: {len(commit_history)}",
            f"Incremental write queue events: {len(incremental_write_queue.get('history', []) or [])}",
            f"Auto-commit proposals tracked: {len(auto_commit.get('proposal_history', []) or [])}",
            f"Content replacements tracked: {len(content_replacements.get('registry', {}) or {})}",
            f"Structured output active: {'yes' if structured_output.get('request') else 'no'}",
            f"Post-sampling reports: {len(post_sampling_history)}",
            f"Security classifier mode: {security_classifier.get('mode', 'off')}",
            f"Security classifier hits: {len(security_classifier.get('history', []) or [])}",
            f"VCR mode: {vcr.get('mode', 'off')}",
            f"VCR cassettes touched: {len(vcr.get('history', []) or [])}",
            f"Session cache creation tokens: {session_token_usage.get('cache_creation_input_tokens', 0)}",
            f"Session cache read tokens: {session_token_usage.get('cache_read_input_tokens', 0)}",
            f"Messages loaded: {len(self.messages)}",
            f"Todos: {len(todos)}",
            f"Subagents tracked: {len(subagents)}",
            f"Shell tasks tracked: {len(shell_tasks)}",
            f"Loop transitions tracked: {len(loop_transitions)}",
            f"Permission denials tracked: {len(permissions.get('denial_history', []) or [])}",
            f"Pending permission request: {'yes' if pending_permission else 'no'}",
            f"Pending auto-commit proposal: {'yes' if auto_commit.get('pending_proposal') else 'no'}",
            f"Session input tokens: {session_token_usage.get('input_tokens', 0)}",
            f"Session output tokens: {session_token_usage.get('output_tokens', 0)}",
            f"Remaining output budget: {max(0, 64000 - int(session_token_usage.get('output_tokens', 0) or 0))}",
            f"Persistence status: {'ok' if self.last_persist_ok else 'error'}",
        ]
        if sandbox:
            lines.append(
                f"Active sandbox: {sandbox.get('branch')} @ {sandbox.get('path')}"
            )
        else:
            lines.append("Active sandbox: none")

        lines.append(
            f"Browser last URL: {browser.get('last_url') or 'none'}"
        )
        lines.append(
            f"Browser available: {'yes' if browser.get('available') else 'no'}"
        )
        if pending_permission:
            lines.append(
                "Recovered pending permission: "
                f"{pending_permission.get('tool_name')} "
                f"(risk={pending_permission.get('risk_level')}, mode={pending_permission.get('mode')})"
            )
        if commit_history:
            last_commit = commit_history[0]
            lines.append(
                "Recovered last commit attribution: "
                f"{last_commit.get('status')} "
                f"{last_commit.get('commit_hash') or '<none>'} "
                f"on {last_commit.get('branch') or '<detached>'} "
                f"via {last_commit.get('agent_role') or 'agent'}"
            )
        pending_auto_commit = auto_commit.get("pending_proposal") or {}
        if pending_auto_commit:
            lines.append(
                "Recovered pending auto-commit: "
                f"{pending_auto_commit.get('branch') or '<detached>'} "
                f"{pending_auto_commit.get('message_preview') or '<no message>'}"
            )

        lines.append(
            "Message stats: "
            f"user={stats.get('user_messages', 0)}, "
            f"assistant={stats.get('assistant_messages', 0)}, "
            f"tool_results={stats.get('tool_result_blocks', 0)}, "
            f"diagnostics={stats.get('system_observations', 0)}"
        )
        if self.last_persist_error:
            lines.append(f"Last persistence error: {self.last_persist_error}")
        return "\n".join(lines)

    def _run_hooks(self, event: str, payload: dict) -> list:
        try:
            return self.hook_manager.execute(event, payload)
        except Exception:
            return []

    def _append_hook_messages(self, event: str, outputs: list):
        if not outputs:
            return
        for index, content in enumerate(outputs, start=1):
            self.messages.append(create_user_message(
                f"<system-reminder>\n"
                f"[Hook:{event} #{index}]\n"
                f"{content}\n"
                f"</system-reminder>",
                is_meta=True,
                origin="hook",
            ))
        self._sync_message_context()

    def _rollback_incomplete_tool_turn(self, assistant_message: dict) -> bool:
        if not assistant_message or not self.messages:
            return False

        target_uuid = get_msg_uuid(assistant_message)
        if target_uuid:
            from codeclaw.core.messages import rollback_to_uuid
            removed = rollback_to_uuid(self.messages, target_uuid)
            if removed:
                self._sync_message_context()
                self.persist_session_state()
                return True

        if self.messages and self.messages[-1] is assistant_message:
            content = assistant_message.get("content")
            if isinstance(content, list):
                has_tool_use = any(
                    (block.get("type") if isinstance(block, dict) else getattr(block, "type", None)) == "tool_use"
                    for block in content
                )
                if has_tool_use:
                    self.messages.pop()
                    self._sync_message_context()
                    self.persist_session_state()
                    return True

        removed = rollback_incomplete_turn(self.messages)
        if removed:
            self._sync_message_context()
            self.persist_session_state()
            return True

        return False

    async def _emit_stream_event(self, event_callback, payload: dict):
        if not callable(event_callback):
            return

        result = event_callback(payload)
        if inspect.isawaitable(result):
            await result

    def _serialize_response_usage(self, response) -> dict:
        usage = getattr(response, "usage", None)
        if usage is None:
            return {}
        fields = [
            "input_tokens",
            "output_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
        ]
        payload = {}
        for field in fields:
            value = getattr(usage, field, None)
            if value is not None:
                payload[field] = value
        return payload

    def _serialize_response_for_vcr(self, response) -> dict:
        return {
            "stop_reason": getattr(response, "stop_reason", None),
            "content": [
                block
                for block in [self._block_to_dict(item) for item in getattr(response, "content", []) or []]
                if block
            ],
            "usage": self._serialize_response_usage(response),
        }

    async def _replay_vcr_response_events(self, response, *, turn_count: int, event_callback=None):
        await self._emit_stream_event(event_callback, {
            "type": "message_start",
            "turn": turn_count,
            "replayed": True,
        })
        for block in getattr(response, "content", []) or []:
            if getattr(block, "type", "") == "text":
                text = getattr(block, "text", "")
                if text:
                    await self._emit_stream_event(event_callback, {
                        "type": "text_delta",
                        "turn": turn_count,
                        "text": text,
                        "replayed": True,
                    })
            elif getattr(block, "type", "") == "tool_use":
                await self._emit_stream_event(event_callback, {
                    "type": "tool_scheduled",
                    "turn": turn_count,
                    "tool_name": getattr(block, "name", ""),
                    "tool_use_id": getattr(block, "id", None),
                    "replayed": True,
                })
        await self._emit_stream_event(event_callback, {
            "type": "message_stop",
            "turn": turn_count,
            "stop_reason": getattr(response, "stop_reason", None),
            "replayed": True,
        })

    def _get_max_output_tokens(self) -> int:
        """
        Resolve max output tokens: env override > 32000 default.
        Escalate to 64k on overflow (see max_tokens recovery in _run_loop).
        """
        env_val = os.environ.get("CODECLAW_MAX_OUTPUT_TOKENS", "")
        if env_val:
            try:
                parsed = int(env_val)
                if parsed > 0:
                    return parsed
            except ValueError:
                pass
        return 32000

    def _build_loop_state(self) -> dict:
        env_max = os.environ.get("CODECLAW_MAX_TURNS", "")
        max_turns = int(env_max) if env_max.isdigit() else 1000
        return {
            "max_turns": max_turns,
            "turn_count": 0,
            "max_tokens": self._get_max_output_tokens(),
            "active_model": self.primary_model,
            "fallback_count": 0,
            "session_output_budget": 200000,
            "session_input_budget": 200000,
            "turn_input_tokens": 0,
            "turn_output_tokens": 0,
            "max_tokens_recovery_attempts": 0,
            "max_tokens_escalations": 0,
            "reactive_compact_retries": 0,
            "stop_hook_retries": 0,
            "structured_output_retries": 0,
            "continue_reason": "initial_turn",
        }

    async def _record_loop_transition(self, reason: str, *, turn: int, event_callback=None, **details):
        record = {
            "reason": reason,
            "turn": turn,
            **details,
        }
        self.loop_transition_history.insert(0, record)
        self.loop_transition_history = self.loop_transition_history[:20]
        self.tool_context["loop_transition_history"] = self.loop_transition_history
        await self._emit_stream_event(event_callback, {
            "type": "loop_transition",
            **record,
        })

    def _build_continuation_prompt(self) -> str:
        return (
            "<system-reminder>\n"
            "The previous assistant response stopped because the output token budget was exhausted.\n"
            "Continue immediately from where you left off. Do not repeat completed content.\n"
            "</system-reminder>"
        )

    def _build_stop_hook_feedback(self, reason: str) -> str:
        return (
            "<system-reminder>\n"
            "Stop hook blocked the previous answer.\n"
            "Revise the answer and try again.\n"
            f"Blocking reason:\n{reason}\n"
            "</system-reminder>"
        )

    def _record_token_usage(
        self,
        response,
        loop_state: dict,
        turn: int,
        *,
        system_prompt: str = "",
        api_messages: list = None,
        tools_schema: list = None,
        thinking_config: dict = None,
    ):
        usage = getattr(response, "usage", None)
        turn_record = {
            "turn": turn,
            "model": loop_state.get("active_model"),
            "input_tokens": int(getattr(usage, "input_tokens", 0) or 0) if usage is not None else 0,
            "output_tokens": int(getattr(usage, "output_tokens", 0) or 0) if usage is not None else 0,
            "cache_creation_input_tokens": int(getattr(usage, "cache_creation_input_tokens", 0) or 0) if usage is not None else 0,
            "cache_read_input_tokens": int(getattr(usage, "cache_read_input_tokens", 0) or 0) if usage is not None else 0,
        }
        source = "api_usage"
        if turn_record["input_tokens"] <= 0 and turn_record["output_tokens"] <= 0:
            turn_record["input_tokens"] = self.compactor.estimate_request_tokens(
                api_messages or [],
                system=system_prompt,
                tools=tools_schema,
                thinking=thinking_config,
            )
            turn_record["output_tokens"] = self.compactor.estimate_content_tokens(
                "assistant",
                getattr(response, "content", []),
            )
            source = self.compactor._local_estimator_mode()
        turn_record["source"] = source

        self.token_usage_history.insert(0, turn_record)
        self.token_usage_history = self.token_usage_history[:50]
        self.tool_context["token_usage_history"] = self.token_usage_history

        for key in [
            "input_tokens",
            "output_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
        ]:
            self.session_token_usage[key] = int(self.session_token_usage.get(key, 0) or 0) + turn_record[key]

        loop_state["turn_input_tokens"] = turn_record["input_tokens"]
        loop_state["turn_output_tokens"] = turn_record["output_tokens"]

    def _remaining_output_budget(self, loop_state: dict) -> int:
        return max(
            0,
            int(loop_state.get("session_output_budget", 0) or 0)
            - int(self.session_token_usage.get("output_tokens", 0) or 0),
        )

    def _is_model_fallback_error(self, error: Exception) -> bool:
        if isinstance(error, (APIConnectionError, RateLimitError, InternalServerError)):
            return True

        if isinstance(error, APIStatusError):
            status_code = getattr(error, "status_code", None)
            if status_code in {500, 502, 503, 504, 529}:
                return True

        rendered = str(error).lower()
        markers = [
            "overloaded",
            "service unavailable",
            "temporarily unavailable",
            "rate limit",
            "connection error",
        ]
        return any(marker in rendered for marker in markers)

    async def _await_gate_and_execute_tool(
        self, gate, tool_block, sys_print_callback=print, *,
        turn: int = 0, event_callback=None,
        semaphore: asyncio.Semaphore = None,
        sibling_cancel: asyncio.Event = None,
    ):
        if gate is not None:
            await gate
        if self.abort_event.is_set():
            raise AbortRequestedError("Abort requested before tool execution.")
        if sibling_cancel is not None and sibling_cancel.is_set():
            tool_id = getattr(tool_block, "id", "") or (tool_block.get("id", "") if isinstance(tool_block, dict) else "")
            return {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": "Cancelled: sibling tool in batch failed.",
                "is_error": True,
            }

        async def _do_execute():
            res = await self._execute_tool_wrapper(
                tool_block, sys_print_callback, turn=turn, event_callback=event_callback,
            )
            if (
                res.get("is_error")
                and sibling_cancel is not None
                and (getattr(tool_block, "name", "") or (tool_block.get("name", "") if isinstance(tool_block, dict) else ""))
                in self.SIBLING_CANCEL_TOOLS
            ):
                sibling_cancel.set()
            return res

        if semaphore is not None:
            async with semaphore:
                return await _do_execute()
        return await _do_execute()

    async def _cancel_streaming_tool_tasks(self, entries: list):
        pending = [
            item.get("task")
            for item in entries or []
            if item.get("task") is not None and not item["task"].done()
        ]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    async def _stream_model_response(
        self,
        *,
        system_prompt: str,
        api_messages: list,
        tools_schema: list,
        turn_count: int,
        model_name: str,
        max_tokens: int,
        thinking_config: dict = None,
        sys_print_callback=print,
        event_callback=None,
    ):
        """Returns (response, tool_entries, streaming_fallback_occurred)."""
        tool_entries = []
        self.active_streaming_tool_entries = tool_entries
        previous_batch_gate = None
        current_safe_batch_gate = None
        current_safe_batch_tasks = None
        streaming_semaphore = asyncio.Semaphore(self.MAX_TOOL_CONCURRENCY)
        streaming_sibling_cancel = asyncio.Event()

        try:
            if self.model_provider == "openai":
                openai_messages = self._clean_roles_for_openai_api(
                    api_messages,
                    system_prompt,
                )
                create_kwargs = {
                    "model": model_name,
                    "messages": openai_messages,
                    "max_tokens": max_tokens,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                }
                if tools_schema:
                    create_kwargs["tools"] = tools_schema

                text_fragments = []
                tool_calls = {}
                emitted_tool_ids = set()
                finish_reason = "end_turn"
                usage_payload = self._make_usage_block()

                stream = await self.client.chat.completions.create(**create_kwargs)
                self.active_stream = stream
                await self._emit_stream_event(event_callback, {
                    "type": "message_start",
                    "turn": turn_count,
                })

                async for chunk in stream:
                    if self.abort_event.is_set():
                        raise AbortRequestedError("Abort requested during OpenAI streaming response.")
                    chunk_usage = getattr(chunk, "usage", None)
                    if chunk_usage is not None:
                        usage_payload = self._make_usage_block(
                            input_tokens=getattr(chunk_usage, "prompt_tokens", 0) or 0,
                            output_tokens=getattr(chunk_usage, "completion_tokens", 0) or 0,
                        )

                    choices = list(getattr(chunk, "choices", []) or [])
                    if not choices:
                        continue

                    choice = choices[0]
                    delta = getattr(choice, "delta", None)
                    if delta is not None:
                        delta_text = getattr(delta, "content", None)
                        if delta_text:
                            text_fragments.append(delta_text)
                            await self._emit_stream_event(event_callback, {
                                "type": "text_delta",
                                "turn": turn_count,
                                "text": delta_text,
                            })

                        for tool_delta in list(getattr(delta, "tool_calls", []) or []):
                            index = int(getattr(tool_delta, "index", 0) or 0)
                            record = tool_calls.setdefault(index, {
                                "id": "",
                                "name": "",
                                "arguments": [],
                            })
                            tool_id = getattr(tool_delta, "id", None)
                            if tool_id:
                                record["id"] = tool_id
                            fn = getattr(tool_delta, "function", None)
                            if fn is not None:
                                fn_name = getattr(fn, "name", None)
                                fn_arguments = getattr(fn, "arguments", None)
                                if fn_name:
                                    record["name"] = fn_name
                                if fn_arguments:
                                    record["arguments"].append(fn_arguments)
                            if record["id"] and record["name"] and record["id"] not in emitted_tool_ids:
                                emitted_tool_ids.add(record["id"])
                                await self._emit_stream_event(event_callback, {
                                    "type": "tool_scheduled",
                                    "turn": turn_count,
                                    "tool_name": record["name"],
                                    "tool_use_id": record["id"],
                                })

                    if getattr(choice, "finish_reason", None):
                        finish_reason = self._map_openai_finish_reason(choice.finish_reason)

                content_blocks = []
                full_text = "".join(text_fragments)
                if full_text:
                    content_blocks.append(self._make_text_block(full_text))
                for record in [tool_calls[idx] for idx in sorted(tool_calls.keys())]:
                    tool_id = record.get("id") or f"call_{len(content_blocks) + 1}"
                    raw_arguments = "".join(record.get("arguments", []) or [])
                    try:
                        parsed_arguments = json.loads(raw_arguments) if raw_arguments.strip() else {}
                    except Exception:
                        parsed_arguments = {}
                    content_blocks.append(
                        self._make_tool_use_block(
                            tool_id=tool_id,
                            name=record.get("name", ""),
                            input_payload=parsed_arguments,
                        )
                    )

                await self._emit_stream_event(event_callback, {
                    "type": "message_stop",
                    "turn": turn_count,
                    "stop_reason": finish_reason,
                })
                return SimpleNamespace(
                    stop_reason=finish_reason,
                    content=content_blocks,
                    usage=usage_payload,
                ), [], False

            system_blocks = self._build_cached_system_blocks(system_prompt)
            stream_kwargs = {
                "model": model_name,
                "system": system_blocks,
                "messages": api_messages,
                "tools": tools_schema,
                "max_tokens": max_tokens,
            }
            if thinking_config is not None and self._is_native_anthropic():
                stream_kwargs["thinking"] = thinking_config

            try:
                async with self.client.messages.stream(**stream_kwargs) as stream:
                    self.active_stream = stream
                    while True:
                        next_event_task = asyncio.create_task(stream.__anext__())
                        abort_wait_task = asyncio.create_task(self.abort_event.wait())
                        done, pending = await asyncio.wait(
                            {next_event_task, abort_wait_task},
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        for task in pending:
                            task.cancel()
                        await asyncio.gather(*pending, return_exceptions=True)

                        if abort_wait_task in done and self.abort_event.is_set():
                            next_event_task.cancel()
                            await asyncio.gather(next_event_task, return_exceptions=True)
                            raise AbortRequestedError("Abort requested during streaming response.")

                        try:
                            event = next_event_task.result()
                        except StopAsyncIteration:
                            break

                        event_type = getattr(event, "type", "")
                        if event_type == "message_start":
                            await self._emit_stream_event(event_callback, {
                                "type": "message_start",
                                "turn": turn_count,
                            })
                        elif event_type == "content_block_delta":
                            delta = getattr(event, "delta", None)
                            if getattr(delta, "type", "") == "text_delta":
                                await self._emit_stream_event(event_callback, {
                                    "type": "text_delta",
                                    "turn": turn_count,
                                    "text": getattr(delta, "text", ""),
                                })
                        elif event_type == "content_block_stop":
                            content_block = getattr(event, "content_block", None)
                            if getattr(content_block, "type", "") == "tool_use":
                                is_safe = self._is_concurrency_safe(
                                    getattr(content_block, "name", ""),
                                    getattr(content_block, "input", {}) or {},
                                )
                                if is_safe:
                                    if current_safe_batch_tasks is None:
                                        current_safe_batch_gate = previous_batch_gate
                                        current_safe_batch_tasks = []
                                    task = asyncio.create_task(
                                        self._await_gate_and_execute_tool(
                                            current_safe_batch_gate,
                                            content_block,
                                            sys_print_callback,
                                            turn=turn_count,
                                            event_callback=event_callback,
                                            semaphore=streaming_semaphore,
                                            sibling_cancel=streaming_sibling_cancel,
                                        )
                                    )
                                    current_safe_batch_tasks.append(task)
                                else:
                                    if current_safe_batch_tasks is not None:
                                        previous_batch_gate = asyncio.gather(*current_safe_batch_tasks)
                                        current_safe_batch_tasks = None
                                        current_safe_batch_gate = None
                                    task = asyncio.create_task(
                                        self._await_gate_and_execute_tool(
                                            previous_batch_gate,
                                            content_block,
                                            sys_print_callback,
                                            turn=turn_count,
                                            event_callback=event_callback,
                                            semaphore=streaming_semaphore,
                                        )
                                    )
                                    previous_batch_gate = task

                                tool_entries.append({
                                    "tool_use_id": getattr(content_block, "id", None),
                                    "task": task,
                                })
                                await self._emit_stream_event(event_callback, {
                                    "type": "tool_scheduled",
                                    "turn": turn_count,
                                    "tool_name": getattr(content_block, "name", ""),
                                    "tool_use_id": getattr(content_block, "id", None),
                                })

                            await self._emit_stream_event(event_callback, {
                                "type": "content_block_stop",
                                "turn": turn_count,
                                "index": getattr(event, "index", None),
                            })
                        elif event_type == "message_stop":
                            snapshot = stream.current_message_snapshot
                            await self._emit_stream_event(event_callback, {
                                "type": "message_stop",
                                "turn": turn_count,
                                "stop_reason": getattr(snapshot, "stop_reason", None),
                            })

                    if current_safe_batch_tasks is not None:
                        previous_batch_gate = asyncio.gather(*current_safe_batch_tasks)

                    final_message = await stream.get_final_message()
                    return final_message, tool_entries, False
            except AbortRequestedError:
                raise
            except Exception as stream_err:
                if self._is_context_overflow_error(stream_err):
                    raise
                await self._cancel_streaming_tool_tasks(tool_entries)
                tool_entries.clear()
                previous_batch_gate = None
                current_safe_batch_gate = None
                current_safe_batch_tasks = None

                disable_fallback = os.environ.get(
                    "CODECLAW_DISABLE_NONSTREAMING_FALLBACK", ""
                )
                if disable_fallback:
                    raise

                sys_print_callback(
                    "[dim yellow]↳ Streaming request failed mid-stream; "
                    "retrying as non-streaming request.[/dim yellow]"
                )
                create_kwargs = {
                    k: v for k, v in stream_kwargs.items()
                }
                response = await self.client.messages.create(**create_kwargs)
                await self._emit_stream_event(event_callback, {
                    "type": "message_start",
                    "turn": turn_count,
                })
                for block in response.content:
                    btype = getattr(block, "type", "")
                    if btype == "text":
                        await self._emit_stream_event(event_callback, {
                            "type": "text_delta",
                            "turn": turn_count,
                            "text": getattr(block, "text", ""),
                        })
                    elif btype == "tool_use":
                        await self._emit_stream_event(event_callback, {
                            "type": "tool_scheduled",
                            "turn": turn_count,
                            "tool_name": getattr(block, "name", ""),
                            "tool_use_id": getattr(block, "id", None),
                        })
                await self._emit_stream_event(event_callback, {
                    "type": "message_stop",
                    "turn": turn_count,
                    "stop_reason": getattr(response, "stop_reason", None),
                })
                return response, [], True
        finally:
            self.active_stream = None
            self.active_streaming_tool_entries = []

    async def _run_loop(self, user_input: str, sys_print_callback=print, event_callback=None):
        """
        The Core Action Loop:
        1. Contextualize System Prompt.
        2. Feed user text to the LLM.
        3. If tool calls -> Execute in parallel via asyncio & Append results to loop.
        4. If stop -> Return final textual output.
        """
        self.abort_event.clear()

        if not self.is_configured:
            return (
                "Model not configured. Use /model to set protocol, model name, "
                "base URL, and API key before sending messages."
            )

        run_started_at = time.time()
        auto_commit_git_baseline = self._capture_auto_commit_baseline()
        user_msg = create_user_message(user_input, origin="human")
        self.messages.append(user_msg)
        self._sync_message_context()

        if not hasattr(self, "session_started"):
            self.session_started = False
        if not self.session_started:
            session_start_outputs = self._run_hooks("SessionStart", {
                "cwd": os.getcwd(),
                "mode": self.get_mode(),
                "agent_id": self.agent_id,
                "agent_role": self.agent_role,
            })
            self._append_hook_messages("SessionStart", session_start_outputs)
            self.session_started = True
            self.persist_session_state()
        
        tools_schema = self._get_anthropic_tools_schema()
        
        loop_state = self._build_loop_state()

        while loop_state["turn_count"] < loop_state["max_turns"]:
            loop_state["turn_count"] += 1
            turn_count = loop_state["turn_count"]
            assistant_message = None
            streaming_tool_entries = []
            loop_state["turn_input_tokens"] = 0
            loop_state["turn_output_tokens"] = 0
            
            try:
                if self.abort_event.is_set():
                    raise AbortRequestedError("Abort requested before turn start.")
                await self._apply_layered_compaction_if_needed(sys_print_callback)
                remaining_output_budget = self._remaining_output_budget(loop_state)
                if remaining_output_budget <= 0:
                    await self._record_loop_transition(
                        "session_output_budget_exhausted",
                        turn=turn_count,
                        event_callback=event_callback,
                        session_output_budget=loop_state["session_output_budget"],
                        output_tokens_used=self.session_token_usage.get("output_tokens", 0),
                    )
                    self.persist_session_state()
                    return "Session output token budget exhausted before another turn could start."
                self._inject_turn_attachments(turn_count)
                self._apply_frc_if_needed()
                boundary_messages = self._get_messages_after_compact_boundary()
                self._apply_tool_result_budget(boundary_messages)
                api_messages = self._clean_roles_for_api(boundary_messages)
                if self.model_provider == "anthropic" and self._is_native_anthropic():
                    api_messages = self._apply_message_cache_control(api_messages)
                loaded_memory_files = self.refresh_memory_files()
                effective_thinking_config = self._resolve_thinking_config()
                system_prompt = self.context_builder.generate_system_prompt(
                    session_summary=self.plan_manager.render_prompt_summary(),
                    todo_summary=self.todo_manager.render_prompt_summary(),
                    memory_summary=self.get_memory_summary(),
                    structured_output_summary=self.structured_output_manager.render_prompt_summary(),
                    tool_prompt_summary=self._build_tool_prompt_summary(),
                    mcp_instructions=self._build_mcp_instructions(),
                )
                system_prompt += self._get_coordinator_system_prompt_addon()
                await self._emit_stream_event(event_callback, {
                    "type": "memory_files",
                    "turn": turn_count,
                    "count": len(loaded_memory_files),
                    "files": [
                        {
                            "scope": item.get("scope"),
                            "path": item.get("path"),
                            "truncated": item.get("truncated", False),
                        }
                        for item in loaded_memory_files
                    ],
                })

                vcr_request_payload = self.vcr_manager.build_request_payload(
                    provider=self.model_provider,
                    model=loop_state["active_model"],
                    system=system_prompt,
                    messages=api_messages,
                    tools=tools_schema,
                    max_tokens=min(loop_state["max_tokens"], remaining_output_budget),
                    thinking=effective_thinking_config,
                )
                streaming_fallback = False
                replayed_response = self.vcr_manager.try_replay(vcr_request_payload)
                if replayed_response is not None:
                    response = replayed_response
                    streaming_tool_entries = []
                    await self._replay_vcr_response_events(
                        response,
                        turn_count=turn_count,
                        event_callback=event_callback,
                    )
                else:
                    response, streaming_tool_entries, streaming_fallback = await self._stream_model_response(
                        system_prompt=system_prompt,
                        api_messages=api_messages,
                        tools_schema=tools_schema,
                        turn_count=turn_count,
                        model_name=loop_state["active_model"],
                        max_tokens=min(loop_state["max_tokens"], remaining_output_budget),
                        thinking_config=effective_thinking_config,
                        sys_print_callback=sys_print_callback,
                        event_callback=event_callback,
                    )
                    if streaming_fallback:
                        await self._emit_stream_event(event_callback, {
                            "type": "streaming_fallback",
                            "turn": turn_count,
                        })
                    self.vcr_manager.record(
                        vcr_request_payload,
                        self._serialize_response_for_vcr(response),
                    )
                self._record_token_usage(
                    response,
                    loop_state,
                    turn_count,
                    system_prompt=system_prompt,
                    api_messages=api_messages,
                    tools_schema=tools_schema,
                    thinking_config=effective_thinking_config,
                )
                await self._emit_stream_event(event_callback, {
                    "type": "token_usage",
                    "turn": turn_count,
                    "input_tokens": loop_state["turn_input_tokens"],
                    "output_tokens": loop_state["turn_output_tokens"],
                    "session_input_tokens": self.session_token_usage.get("input_tokens", 0),
                    "session_output_tokens": self.session_token_usage.get("output_tokens", 0),
                })
                await self._emit_stream_event(event_callback, {
                    "type": "thinking_config",
                    "turn": turn_count,
                    "config": effective_thinking_config,
                    "mode": self.get_mode(),
                })
                
                content_as_dicts = [self._block_to_dict(b) for b in response.content]
                content_as_dicts = [b for b in content_as_dicts if b is not None]
                assistant_message = create_assistant_message(
                    content_as_dicts,
                    model=loop_state.get("active_model", ""),
                    stop_reason=getattr(response, "stop_reason", ""),
                )
                self.messages.append(assistant_message)
                self._sync_message_context()
                # Checkpointing memory auto-flush
                self.persist_session_state()
                
                if response.stop_reason == "tool_use":
                    if streaming_tool_entries:
                        tool_results = []
                        for item in streaming_tool_entries:
                            if self.abort_event.is_set():
                                raise AbortRequestedError("Abort requested while awaiting tool results.")
                            tool_results.append(await item["task"])
                    else:
                        tool_calls = [b for b in content_as_dicts if isinstance(b, dict) and b.get("type") == "tool_use"]
                        tool_results = []
                        results = await self._execute_tool_batches(
                            tool_calls,
                            sys_print_callback,
                            turn=turn_count,
                            event_callback=event_callback,
                        )
                        tool_results.extend(results)
                    
                    tool_result_msg = create_tool_result_message(
                        tool_results,
                        source_assistant_uuid=get_msg_uuid(assistant_message),
                    )
                    self.messages.append(tool_result_msg)
                    self._sync_message_context()
                    
                    # Capture lagging compiler diagnostics from background daemon
                    await asyncio.sleep(0.4)
                    if hasattr(self, 'lsp_manager'):
                        diags_msg = self.lsp_manager.consume_diagnostics()
                        if diags_msg:
                            self.messages.append(create_user_message(
                                f"[System Observation]\\n{diags_msg}",
                                is_meta=True,
                                origin="lsp_diagnostics",
                            ))
                            self._sync_message_context()
                            sys_print_callback("[bold red]⚠ Active Compiler Diagnostics intercepted and fed to agent loop![/bold red]")
                    
                    # Tool cycle complete, checkpoint again
                    self.persist_session_state()
                    loop_state["continue_reason"] = "next_turn"
                    await self._record_loop_transition(
                        "next_turn",
                        turn=turn_count,
                        event_callback=event_callback,
                        stop_reason="tool_use",
                    )
                    
                elif response.stop_reason == "max_tokens":
                    if streaming_tool_entries:
                        await self._cancel_streaming_tool_tasks(streaming_tool_entries)
                    loop_state["max_tokens_recovery_attempts"] += 1
                    remaining_output_budget = self._remaining_output_budget(loop_state)
                    if remaining_output_budget <= 0:
                        await self._record_loop_transition(
                            "session_output_budget_exhausted",
                            turn=turn_count,
                            event_callback=event_callback,
                            session_output_budget=loop_state["session_output_budget"],
                            output_tokens_used=self.session_token_usage.get("output_tokens", 0),
                        )
                        self.persist_session_state()
                        return "Session output token budget exhausted before continuation."
                    transition_reason = "max_output_tokens_recovery"
                    if (
                        loop_state["max_tokens_recovery_attempts"] >= 2
                        and loop_state["max_tokens_escalations"] < 1
                    ):
                        loop_state["max_tokens"] = 64000
                        loop_state["max_tokens_escalations"] += 1
                        transition_reason = "max_output_tokens_escalate"
                    loop_state["max_tokens"] = min(loop_state["max_tokens"], remaining_output_budget)
                    self.messages.append(create_user_message(
                        self._build_continuation_prompt(),
                        is_continuation=True,
                        origin="system_continuation",
                    ))
                    self._sync_message_context()
                    self.persist_session_state()
                    await self._record_loop_transition(
                        transition_reason,
                        turn=turn_count,
                        event_callback=event_callback,
                        max_tokens=loop_state["max_tokens"],
                        recovery_attempts=loop_state["max_tokens_recovery_attempts"],
                        remaining_output_budget=remaining_output_budget,
                    )
                    sys_print_callback(
                        f"[dim yellow]↳ Continuing after max_tokens stop ({transition_reason}, budget={loop_state['max_tokens']}).[/dim yellow]"
                    )
                    continue
                else:
                    if streaming_tool_entries:
                        await self._cancel_streaming_tool_tasks(streaming_tool_entries)
                    # Normal completion without Tool calls => Return final answer
                    texts = [b.get("text", "") for b in content_as_dicts if isinstance(b, dict) and b.get("type") == "text"]
                    final_answer = "\n".join(texts)
                    structured_ok, structured_reason = self.structured_output_manager.validate(final_answer)
                    if not structured_ok:
                        loop_state["structured_output_retries"] += 1
                        self.messages.append(create_user_message(
                            self._build_structured_output_feedback(structured_reason),
                            is_meta=True,
                            origin="structured_output_retry",
                        ))
                        self._sync_message_context()
                        self.persist_session_state()
                        await self._record_loop_transition(
                            "structured_output_retry",
                            turn=turn_count,
                            event_callback=event_callback,
                            retries=loop_state["structured_output_retries"],
                            reason=structured_reason,
                        )
                        sys_print_callback(
                            "[dim yellow]↳ Structured output validation failed; retrying with feedback.[/dim yellow]"
                        )
                        continue
                    stop_hook_decision = self.hook_manager.evaluate_stop_hooks({
                        "cwd": os.getcwd(),
                        "mode": self.get_mode(),
                        "agent_id": self.agent_id,
                        "agent_role": self.agent_role,
                        "final_answer": final_answer,
                        "turn_input_tokens": loop_state["turn_input_tokens"],
                        "turn_output_tokens": loop_state["turn_output_tokens"],
                        "session_token_usage": dict(self.session_token_usage),
                    })
                    if stop_hook_decision.behavior in {"block", "retry"}:
                        loop_state["stop_hook_retries"] += 1
                        self.messages.append(create_user_message(
                            self._build_stop_hook_feedback(
                                stop_hook_decision.reason or "Blocked by stop hook."
                            ),
                            is_meta=True,
                            origin="stop_hook",
                        ))
                        self._sync_message_context()
                        self.persist_session_state()
                        await self._record_loop_transition(
                            "stop_hook_blocking",
                            turn=turn_count,
                            event_callback=event_callback,
                            retries=loop_state["stop_hook_retries"],
                            behavior=stop_hook_decision.behavior,
                        )
                        sys_print_callback(
                            "[dim yellow]↳ StopHook blocked the current answer; retrying with feedback.[/dim yellow]"
                        )
                        continue
                    turn_end_outputs = self._run_hooks("TurnEnd", {
                        "cwd": os.getcwd(),
                        "mode": self.get_mode(),
                        "agent_id": self.agent_id,
                        "agent_role": self.agent_role,
                        "final_answer": final_answer,
                        "turn_input_tokens": loop_state["turn_input_tokens"],
                        "turn_output_tokens": loop_state["turn_output_tokens"],
                        "session_token_usage": dict(self.session_token_usage),
                    })
                    self._append_hook_messages("TurnEnd", turn_end_outputs)
                    await self._apply_post_sampling_hooks(
                        final_answer=final_answer,
                        turn=turn_count,
                        event_callback=event_callback,
                    )
                    await self._maybe_prepare_auto_commit_proposal(
                        final_answer=final_answer,
                        started_at=run_started_at,
                        git_baseline=auto_commit_git_baseline,
                        turn=turn_count,
                        event_callback=event_callback,
                    )
                    verification_report = await self._maybe_run_verification(
                        sys_print_callback=sys_print_callback,
                    )
                    if verification_report:
                        final_answer += f"\n\n---\n**Verification Report:**\n{verification_report}"
                    self.persist_session_state()
                    return final_answer
                    
            except Exception as e:
                if streaming_tool_entries:
                    await self._cancel_streaming_tool_tasks(streaming_tool_entries)
                if isinstance(e, AbortRequestedError):
                    rolled_back = self._rollback_incomplete_tool_turn(assistant_message)
                    await self._record_loop_transition(
                        "user_abort",
                        turn=turn_count,
                        event_callback=event_callback,
                        rolled_back=rolled_back,
                    )
                    self.active_stream = None
                    self.active_streaming_tool_entries = []
                    self.persist_session_state()
                    return (
                        "[bold yellow]Turn aborted.[/bold yellow] "
                        "The current agent turn was interrupted and cleanup has completed."
                    )
                if (
                    self._is_model_fallback_error(e)
                    and loop_state["active_model"] != self.fallback_model
                    and self.fallback_model
                ):
                    self._strip_thinking_signatures_from_history()
                    loop_state["active_model"] = self.fallback_model
                    loop_state["fallback_count"] += 1
                    await self._record_loop_transition(
                        "model_fallback",
                        turn=turn_count,
                        event_callback=event_callback,
                        fallback_model=self.fallback_model,
                        error=str(e),
                    )
                    sys_print_callback(
                        f"[dim yellow]↳ Primary model failed; retrying current turn with fallback model {self.fallback_model}.[/dim yellow]"
                    )
                    continue
                if self._is_context_overflow_error(e):
                    retries = loop_state.get("reactive_compact_retries", 0)

                    # Withhold-then-recover: attempt transparent recovery
                    # without surfacing the error to the user. Each stage
                    # is progressively more aggressive; if any succeeds the
                    # loop continues seamlessly.

                    # Stage 1: reduce max_tokens (zero-cost, no data loss)
                    current_max = loop_state["max_tokens"]
                    if retries == 0 and current_max > 4096:
                        new_max = max(4096, current_max // 2)
                        loop_state["max_tokens"] = new_max
                        loop_state["reactive_compact_retries"] = 1
                        await self._record_loop_transition(
                            "reactive_withhold_stage1",
                            turn=turn_count,
                            event_callback=event_callback,
                            old_max=current_max,
                            new_max=new_max,
                        )
                        sys_print_callback(
                            f"[dim yellow]↳ Context overflow withheld (stage 1). "
                            f"Reducing max_tokens {current_max} → {new_max}.[/dim yellow]"
                        )
                        continue

                    # Stage 2: normal compaction + post-compact restore
                    if retries <= 1:
                        compacted = await self._apply_layered_compaction_if_needed(
                            sys_print_callback,
                            force=True,
                            aggressive=False,
                        )
                        if compacted:
                            loop_state["reactive_compact_retries"] = 2
                            loop_state["max_tokens"] = self._get_max_output_tokens()
                            await self._record_loop_transition(
                                "reactive_withhold_stage2",
                                turn=turn_count,
                                event_callback=event_callback,
                                retries=2,
                                stage="normal",
                                max_tokens_restored=loop_state["max_tokens"],
                            )
                            sys_print_callback(
                                "[dim magenta]↳ Context overflow withheld (stage 2). "
                                "Retrying after normal compaction + state restore.[/dim magenta]"
                            )
                            continue

                    # Stage 3: aggressive compaction + restore
                    if retries <= 2:
                        compacted = await self._apply_layered_compaction_if_needed(
                            sys_print_callback,
                            force=True,
                            aggressive=True,
                        )
                        if compacted:
                            loop_state["reactive_compact_retries"] = 3
                            loop_state["max_tokens"] = self._get_max_output_tokens()
                            await self._record_loop_transition(
                                "reactive_withhold_stage3",
                                turn=turn_count,
                                event_callback=event_callback,
                                retries=3,
                                stage="aggressive",
                                max_tokens_restored=loop_state["max_tokens"],
                            )
                            sys_print_callback(
                                "[dim magenta]↳ Context overflow withheld (stage 3). "
                                "Retrying after aggressive compaction + state restore.[/dim magenta]"
                            )
                            continue
                rolled_back = self._rollback_incomplete_tool_turn(assistant_message)
                if rolled_back:
                    return (
                        "[bold red]API Loop Exception:[/bold red] "
                        f"{str(e)}\n"
                        "[dim]Rolled back the incomplete assistant tool_use turn to keep session history resumable.[/dim]"
                    )
                # Catch Anthropic Auth errors or general crashes
                return f"[bold red]API Loop Exception:[/bold red] {str(e)}"
                
        return (
            "[bold red]System Halt:[/bold red] "
            f"Max internal reasoning turns ({loop_state['max_turns']}) reached without returning an answer."
        )

    async def run(self, user_input: str, sys_print_callback=print):
        return await self._run_loop(user_input, sys_print_callback=sys_print_callback)

    async def submit_message(self, user_input: str, sys_print_callback=print):
        queue = asyncio.Queue()

        async def emit(payload: dict):
            await queue.put(payload)

        task = asyncio.create_task(
            self._run_loop(
                user_input,
                sys_print_callback=sys_print_callback,
                event_callback=emit,
            )
        )
        self.active_run_task = task

        try:
            while True:
                if task.done() and queue.empty():
                    break
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=0.05)
                    yield payload
                except asyncio.TimeoutError:
                    continue

            final_answer = await task
            yield {
                "type": "final",
                "text": final_answer,
            }
        finally:
            self.active_run_task = None
            self.active_stream = None
            self.active_streaming_tool_entries = []
