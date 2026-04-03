"""
Verification Agent

Automatically spawns a read-only sub-agent to review complex implementations
when 3+ files have been modified in the current run. Uses the full adversarial
verification prompt aligned with the original Claude Code verification agent.
"""
import asyncio
import copy
import uuid
import os
from typing import Optional

from codeclaw.tools.builtin_agents import VERIFICATION_SYSTEM_PROMPT, VERIFICATION_CRITICAL_REMINDER


VERIFICATION_THRESHOLD = 3


class VerificationManager:
    """Tracks file modifications and decides when to trigger verification."""

    def __init__(self):
        self.modified_files: dict[str, dict] = {}
        self.verification_history: list[dict] = []
        self._last_verified_set: set[str] = set()
        self.enabled = True

    def record_file_modification(self, path: str, operation: str = "edit"):
        abs_path = os.path.abspath(path)
        if abs_path not in self.modified_files:
            self.modified_files[abs_path] = {
                "path": abs_path,
                "operations": [],
                "count": 0,
            }
        entry = self.modified_files[abs_path]
        entry["operations"].append(operation)
        entry["count"] += 1

    def should_verify(self) -> bool:
        if not self.enabled:
            return False
        current_set = set(self.modified_files.keys())
        new_modifications = current_set - self._last_verified_set
        return len(new_modifications) >= VERIFICATION_THRESHOLD

    def get_files_to_verify(self) -> list[str]:
        current_set = set(self.modified_files.keys())
        new_files = current_set - self._last_verified_set
        return sorted(new_files)

    def mark_verified(self, files: list[str], result: str):
        self._last_verified_set.update(files)
        self.verification_history.append({
            "files": files,
            "result_preview": result[:500] if result else "",
            "file_count": len(files),
        })
        if len(self.verification_history) > 10:
            self.verification_history = self.verification_history[-10:]

    def reset(self):
        self.modified_files.clear()
        self._last_verified_set.clear()

    def export_state(self) -> dict:
        return {
            "modified_files": copy.deepcopy(self.modified_files),
            "verification_history": copy.deepcopy(self.verification_history),
            "last_verified_set": list(self._last_verified_set),
            "enabled": self.enabled,
        }

    def load_state(self, payload: dict):
        payload = payload or {}
        self.modified_files = dict(payload.get("modified_files", {}))
        self.verification_history = list(payload.get("verification_history", []))
        self._last_verified_set = set(payload.get("last_verified_set", []))
        self.enabled = payload.get("enabled", True)


async def run_verification_agent(
    *,
    files_to_verify: list[str],
    parent_engine,
    sys_print_callback=None,
) -> Optional[str]:
    """
    Spawn a read-only verification sub-agent to review the given files.
    
    Returns the verification report text, or None if spawning fails.
    """
    from codeclaw.core.engine import QueryEngine

    if sys_print_callback is None:
        sys_print_callback = lambda x: None

    file_list_str = "\n".join(f"  - {f}" for f in files_to_verify)
    task_prompt = (
        f"{VERIFICATION_SYSTEM_PROMPT}\n\n"
        f"{VERIFICATION_CRITICAL_REMINDER}\n\n"
        f"---\n\n"
        f"## Files to verify\n{file_list_str}\n\n"
        "Read each file, run any available build/lint checks, "
        "then produce your verification report ending with VERDICT: PASS/FAIL/PARTIAL."
    )

    child_agent_id = f"verifier-{uuid.uuid4()}"

    try:
        sub_engine = QueryEngine(
            model=getattr(parent_engine, "primary_model", None),
            fallback_model=getattr(parent_engine, "fallback_model", None),
            permission_handler=getattr(parent_engine, "permission_handler", None),
            agent_id=child_agent_id,
            parent_agent_id=getattr(parent_engine, "agent_id", "root"),
            agent_depth=getattr(parent_engine, "agent_depth", 0) + 1,
            agent_role="verifier",
            model_provider=getattr(parent_engine, "model_provider", "anthropic"),
            api_base_url=getattr(parent_engine, "api_base_url", None),
            local_tokenizer_path=getattr(parent_engine, "local_tokenizer_path", None),
        )

        read_only_tools = {"file_read_tool", "grep_tool", "glob_tool"}
        parent_tools = getattr(parent_engine, "available_tools", {})
        sub_engine.available_tools = {
            name: copy.copy(tool)
            for name, tool in parent_tools.items()
            if name in read_only_tools
        }
        sub_engine.tool_context["engine_available_tools"] = sub_engine.available_tools
        for tool in sub_engine.available_tools.values():
            tool.context = sub_engine.tool_context

        shared_fields = [
            "lsp_manager", "file_state_cache", "memory_file_manager",
            "artifact_tracker", "content_replacement_manager",
        ]
        for field in shared_fields:
            manager = getattr(parent_engine, field, None)
            if manager is not None:
                setattr(sub_engine, field, manager)
                sub_engine.tool_context[field] = manager

        result = await sub_engine.run(
            user_input=task_prompt,
            sys_print_callback=sys_print_callback,
        )
        return str(result) if result else None

    except Exception as e:
        err_str = str(e)
        if any(k in err_str.lower() for k in (
            "context window", "context length", "too many tokens",
            "maximum context", "token limit", "context_length",
        )):
            return (
                f"[Verification skipped: context window too small for "
                f"{len(files_to_verify)} files. Consider using a model "
                f"with a larger context window for verification.]"
            )
        return f"[Verification agent error: {e}]"
