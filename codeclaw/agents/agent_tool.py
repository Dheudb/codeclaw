import asyncio
import copy
import uuid
from typing import List, Optional
from pydantic import BaseModel, Field
from codeclaw.tools.base import BaseAgenticTool
from codeclaw.agents.builtin_agents import (
    BUILTIN_AGENTS,
    get_builtin_agent,
    list_builtin_agent_descriptions,
)

MAX_SUBAGENT_DEPTH = 2

ALL_AGENT_DISALLOWED_TOOLS = frozenset({
    "agent_tool",
    "ask_user_question_tool",
    "send_message_tool",
    "team_create_tool",
    "team_delete_tool",
    "enter_plan_mode",
    "exit_plan_mode",
})

class AgentToolInput(BaseModel):
    description: str = Field(..., description="A short (3-5 word) description of the task")
    prompt: str = Field(..., description="The task for the agent to perform")
    subagent_type: Optional[str] = Field(
        None,
        description=(
            "The type of specialized agent to use for this task. "
            "Available types: explore, plan, verification, general-purpose. "
            "If omitted, defaults to general-purpose behavior."
        ),
    )
    model: Optional[str] = Field(
        None,
        description="Optional model override for this agent. If omitted, inherits from the parent.",
    )
    run_in_background: bool = Field(
        False,
        description="Set to true to run this agent in the background. You will be notified when it completes.",
    )

class AgentTool(BaseAgenticTool):
    name = "agent_tool"
    description = "Launch a new agent to handle complex, multi-step tasks autonomously. Each agent type has specific capabilities and tools available to it."
    input_schema = AgentToolInput
    risk_level = "medium"

    def prompt(self) -> str:
        agent_list = list_builtin_agent_descriptions()
        return f"""Use the agent_tool with specialized agents when the task at hand matches the agent's description. Subagents are valuable for parallelizing independent queries or for protecting the main context window from excessive results, but they should not be used excessively when not needed. Importantly, avoid duplicating work that subagents are already doing.

## Built-in Agent Types
Set `agent_type` to one of the following to get a specialized system prompt and tool restrictions:

{agent_list}

If `agent_type` is omitted, the agent defaults to general-purpose behavior.

## When NOT to use agent_tool
- If you want to read a specific file path, use file_read_tool or glob_tool instead — they find the match more quickly.
- If you are searching for a specific class definition like "class Foo", use glob_tool instead.
- If you are searching for code within a specific file or set of 2-3 files, use file_read_tool instead.
- Other tasks that are not related to the available agent descriptions.

For simple, directed codebase searches (e.g. for a specific file/class/function) use grep_tool or glob_tool directly. For broader codebase exploration and deep research, use agent_tool with agent_type="explore" — this is slower than direct search, so use it only when a simple directed search is insufficient.

## Usage notes
- Always include a short description (3-5 words) summarizing what the agent will do.
- Launch multiple agents concurrently whenever possible — use a single message with multiple tool uses.
- When the agent is done, it returns a single message back to you. The result is not visible to the user. To show the user the result, send a text message with a concise summary.
- Each Agent invocation starts fresh — provide a complete task description with all necessary context.
- The agent's outputs should generally be trusted.
- Clearly tell the agent whether you expect it to write code or just do research (search, file reads, web fetches, etc.).
- If the agent description mentions that it should be used proactively, try your best to use it without the user having to ask for it first.
- If the user specifies that they want you to run agents "in parallel", you MUST send a single message with multiple agent_tool use content blocks.
- Use agent_type="verification" proactively after non-trivial implementations (3+ file edits).

## Writing the prompt
Brief the agent like a smart colleague who just walked into the room — it hasn't seen this conversation, doesn't know what you've tried, doesn't understand why this task matters.
- Explain what you're trying to accomplish and why.
- Describe what you've already learned or ruled out.
- Give enough context about the surrounding problem that the agent can make judgment calls rather than just following a narrow instruction.
- If you need a short response, say so ("report in under 200 words").
- Lookups: hand over the exact command. Investigations: hand over the question.

**Never delegate understanding.** Don't write "based on your findings, fix the bug" — write prompts that prove you understood: include file paths, line numbers, what specifically to change.

Terse command-style prompts produce shallow, generic work."""

    def build_permission_summary(
        self,
        description: str = "",
        prompt: str = "",
        subagent_type: str = None,
        model: str = None,
        run_in_background: bool = False,
    ) -> str:
        preview = prompt[:240] + ("..." if len(prompt) > 240 else "") if prompt else "<empty>"
        agent_type_str = subagent_type or "general-purpose"
        return (
            "Sub-agent delegation requested.\n"
            f"description: {description}\n"
            f"subagent_type: {agent_type_str}\n"
            f"model: {model or '<inherit>'}\n"
            f"run_in_background: {run_in_background}\n"
            f"prompt_preview: {preview}"
        )

    def _filter_incomplete_tool_messages(self, messages):
        if not messages:
            return []

        tool_result_ids = set()
        for msg in messages:
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result" and block.get("tool_use_id"):
                    tool_result_ids.add(block["tool_use_id"])

        filtered = []
        for msg in messages:
            if msg.get("role") != "assistant":
                filtered.append(msg)
                continue

            content = msg.get("content")
            if not isinstance(content, list):
                filtered.append(msg)
                continue

            has_incomplete_tool_use = False
            for block in content:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "tool_use"
                    and block.get("id")
                    and block.get("id") not in tool_result_ids
                ):
                    has_incomplete_tool_use = True
                    break

            if not has_incomplete_tool_use:
                filtered.append(msg)

        return filtered
    
    def _get_effective_tasks(self, task: str, tasks: List[str] = None) -> List[str]:
        normalized = [item.strip() for item in (tasks or []) if str(item).strip()]
        if normalized:
            return normalized
        if task and task.strip():
            return [task.strip()]
        return []

    def _filter_tools(
        self,
        parent_tools,
        allowed_tools: List[str],
        disallowed_tools: set = None,
    ):
        always_blocked = ALL_AGENT_DISALLOWED_TOOLS
        merged_disallowed = set(always_blocked)
        if disallowed_tools:
            merged_disallowed |= set(disallowed_tools)

        filtered_tools = {
            name: copy.copy(tool)
            for name, tool in parent_tools.items()
            if name not in merged_disallowed
        }
        if allowed_tools and "*" not in allowed_tools:
            allowset = set(allowed_tools)
            filtered_tools = {
                name: tool
                for name, tool in filtered_tools.items()
                if name in allowset
            }
        return filtered_tools

    def _share_runtime_managers(self, sub_engine):
        shared_manager_fields = [
            "lsp_manager",
            "sandbox_manager",
            "browser_manager",
            "shell_task_manager",
            "file_state_cache",
            "memory_file_manager",
            "artifact_tracker",
            "content_replacement_manager",
        ]

        for field in shared_manager_fields:
            manager = self.context.get(field)
            if manager is None:
                continue
            setattr(sub_engine, field, manager)
            sub_engine.tool_context[field] = manager

    def _configure_subagent_permissions(self, sub_engine):
        parent_permission_manager = self.context.get("permission_manager")
        if parent_permission_manager is None:
            return

        # Child agents keep the same approval UX, but start with a clean
        # permission memory so parent session-wide allow rules do not leak.
        sub_engine.permission_manager.always_allow_tools = set()
        sub_engine.tool_context["permission_manager"] = sub_engine.permission_manager
        sub_engine.tool_context["permission_handler"] = self.context.get("permission_handler")

    def _merge_state_back_to_parent(self, sub_engine):
        parent_todo_manager = self.context.get("todo_manager")
        if parent_todo_manager is not None and getattr(sub_engine, "todo_manager", None) is not None:
            parent_todo_manager.load(copy.deepcopy(sub_engine.todo_manager.export()))

        parent_plan_manager = self.context.get("plan_manager")
        if parent_plan_manager is not None and getattr(sub_engine, "plan_manager", None) is not None:
            plan_state = copy.deepcopy(sub_engine.plan_manager.export())
            parent_plan_manager.load(
                mode=plan_state.get("mode", "normal"),
                content=plan_state.get("content", ""),
            )

    async def _write_subagent_registry_record(self, record: dict):
        registry_writer = self.context.get("register_subagent_record")
        if not callable(registry_writer):
            return

        lock = self.context.get("subagent_state_lock")
        if lock is None:
            registry_writer(record)
            return

        async with lock:
            registry_writer(record)

    async def _run_single_subagent(
        self,
        task: str,
        allowed_tools: List[str],
        inherit_context: bool,
        parent_depth: int,
        parent_agent_id: str,
        agent_role: str = "subagent",
        builtin_agent_def: dict = None,
    ) -> dict:
        from codeclaw.core.engine import QueryEngine

        effective_role = agent_role
        if builtin_agent_def:
            effective_role = builtin_agent_def.get("agent_role", agent_role)

        child_agent_id = f"{effective_role}-{uuid.uuid4()}"
        sub_engine = QueryEngine(
            model=self.context.get("primary_model"),
            fallback_model=self.context.get("fallback_model"),
            permission_handler=self.context.get("permission_handler"),
            agent_id=child_agent_id,
            parent_agent_id=parent_agent_id,
            agent_depth=parent_depth + 1,
            agent_role=effective_role,
            model_provider=self.context.get("model_provider"),
            api_base_url=self.context.get("api_base_url"),
            local_tokenizer_path=self.context.get("local_tokenizer_path"),
        )
        self._configure_subagent_permissions(sub_engine)
        if self.context.get("plan_manager"):
            sub_engine.plan_manager = copy.deepcopy(self.context.get("plan_manager"))
            sub_engine.tool_context["plan_manager"] = sub_engine.plan_manager
            sub_engine.permission_manager.set_mode_getter(sub_engine.plan_manager.get_mode)
        if self.context.get("todo_manager"):
            sub_engine.todo_manager = copy.deepcopy(self.context.get("todo_manager"))
            sub_engine.tool_context["todo_manager"] = sub_engine.todo_manager

        self._share_runtime_managers(sub_engine)

        disallowed = set()
        if builtin_agent_def:
            disallowed = set(builtin_agent_def.get("disallowed_tools", []))

        parent_tools = self.context.get("engine_available_tools") 
        if parent_tools:
            sub_engine.available_tools = self._filter_tools(
                parent_tools, allowed_tools, disallowed_tools=disallowed
            )
            sub_engine.tool_context["engine_available_tools"] = sub_engine.available_tools
            for tool in sub_engine.available_tools.values():
                tool.context = sub_engine.tool_context

        skip_context = False
        if builtin_agent_def and builtin_agent_def.get("omit_claude_md"):
            skip_context = True

        if inherit_context and not skip_context:
            sanitized_parent_messages = self._filter_incomplete_tool_messages(
                self.context.get("messages", [])
            )
            sub_engine.inherit_state_from_parent(
                parent_messages=sanitized_parent_messages,
                todo_payload=self.context.get("todo_manager").export() if self.context.get("todo_manager") else [],
                plan_payload=self.context.get("plan_manager").export() if self.context.get("plan_manager") else {},
            )

        specialized_prompt = ""
        if builtin_agent_def:
            specialized_prompt = builtin_agent_def.get("system_prompt", "")
            critical_reminder = builtin_agent_def.get("critical_reminder", "")
            if critical_reminder:
                specialized_prompt += f"\n\n{critical_reminder}"

        if specialized_prompt:
            user_input_text = f"{specialized_prompt}\n\n---\n\nTask:\n{task}"
        else:
            user_input_text = (
                f"[SYSTEM: You are a specialized Sub-Agent. Complete the task below "
                f"using your tools, then report your findings concisely.]\n\nTask:\n{task}"
            )

        try:
            session_manager = self.context.get("session_manager")
            transcript_path = (
                session_manager.get_subagent_transcript_path(child_agent_id)
                if session_manager else None
            )
            registry_record = {
                "agent_id": child_agent_id,
                "parent_agent_id": parent_agent_id,
                "agent_depth": parent_depth + 1,
                "agent_role": effective_role,
                "agent_type": builtin_agent_def.get("agent_type") if builtin_agent_def else None,
                "status": "running",
                "task": task[:240] + ("..." if len(task) > 240 else ""),
                "transcript_path": transcript_path,
            }
            await self._write_subagent_registry_record(registry_record)
            if session_manager:
                session_manager.save_subagent_transcript(child_agent_id, {
                    "agent_id": child_agent_id,
                    "parent_agent_id": parent_agent_id,
                    "agent_depth": parent_depth + 1,
                    "agent_role": effective_role,
                    "task": task,
                    "allowed_tools": allowed_tools,
                    "inherit_context": inherit_context,
                    "status": "running",
                    "messages": sub_engine.messages,
                    "metadata": sub_engine._build_session_metadata(),
                })

            res = await sub_engine.run(
                user_input=user_input_text,
                sys_print_callback=lambda x: None
            )
            self._merge_state_back_to_parent(sub_engine)
            if session_manager:
                session_manager.save_subagent_transcript(child_agent_id, {
                    "agent_id": child_agent_id,
                    "parent_agent_id": parent_agent_id,
                    "agent_depth": parent_depth + 1,
                    "agent_role": agent_role,
                    "task": task,
                    "allowed_tools": allowed_tools,
                    "inherit_context": inherit_context,
                    "status": "completed",
                    "transcript_path": transcript_path,
                    "messages": sub_engine.messages,
                    "metadata": sub_engine._build_session_metadata(),
                    "result": res,
                })
            await self._write_subagent_registry_record({
                **registry_record,
                "status": "completed",
            })
            return {
                "task": task,
                "agent_id": child_agent_id,
                "agent_role": agent_role,
                "status": "completed",
                "result": res,
            }
        except Exception as e:
            session_manager = self.context.get("session_manager")
            if session_manager:
                session_manager.save_subagent_transcript(child_agent_id, {
                    "agent_id": child_agent_id,
                    "parent_agent_id": parent_agent_id,
                    "agent_depth": parent_depth + 1,
                    "agent_role": agent_role,
                    "task": task,
                    "allowed_tools": allowed_tools,
                    "inherit_context": inherit_context,
                    "status": "crashed",
                    "error": str(e),
                    "messages": sub_engine.messages,
                    "metadata": sub_engine._build_session_metadata(),
                })
            await self._write_subagent_registry_record({
                **registry_record,
                "status": "crashed",
            })
            return {
                "task": task,
                "agent_id": child_agent_id,
                "agent_role": agent_role,
                "status": "crashed",
                "result": f"Exception Fault: Sub-Agent terminal crash: {e}",
            }

    def _format_single_report(self, payload: dict) -> str:
        return (
            "=======[ SUB-AGENT COMPLETION REPORT ]=======\n"
            f"Agent ID: {payload.get('agent_id')}\n"
            f"Role: {payload.get('agent_role', 'subagent')}\n"
            f"Task Handled: {payload.get('task')}\n"
            f"Status: {payload.get('status')}\n\n"
            f"Outcome / Findings:\n{payload.get('result')}\n"
            "==============================================="
        )

    def _format_batch_report(self, payloads: List[dict], parallel: bool) -> str:
        completed = sum(1 for item in payloads if item.get("status") == "completed")
        crashed = sum(1 for item in payloads if item.get("status") == "crashed")
        header = (
            "=======[ SUB-AGENT BATCH REPORT ]=======\n"
            f"Execution mode: {'parallel' if parallel else 'serial'}\n"
            f"Task count: {len(payloads)}\n"
            f"Completed: {completed}\n"
            f"Crashed: {crashed}\n"
            "========================================\n"
        )
        body = "\n\n".join(self._format_single_report(item) for item in payloads)
        return header + "\n" + body

    def _build_coordinator_prompt(self, parent_task: str, payloads: List[dict], parallel: bool) -> str:
        lines = [
            "[SYSTEM: Coordinator synthesis mode is active.]",
            "You are the coordinator agent. Synthesize the child sub-agent results into one concise final report.",
            f"Parent objective:\n{parent_task or '<none>'}",
            f"Dispatch mode: {'parallel' if parallel else 'serial'}",
            "",
            "Child agent results:",
        ]
        for index, payload in enumerate(payloads, start=1):
            lines.append(f"## Child {index}")
            lines.append(f"Agent ID: {payload.get('agent_id')}")
            lines.append(f"Status: {payload.get('status')}")
            lines.append(f"Task: {payload.get('task')}")
            lines.append("Result:")
            lines.append(str(payload.get("result", "")))
            lines.append("")

        lines.append("Return a unified coordination report with: overall status, key findings, unresolved risks, and recommended next steps.")
        return "\n".join(lines)

    async def _run_coordinator_mode(
        self,
        *,
        parent_task: str,
        payloads: List[dict],
        parallel: bool,
        parent_depth: int,
        parent_agent_id: str,
        inherit_context: bool,
    ) -> dict:
        return await self._run_single_subagent(
            task=self._build_coordinator_prompt(parent_task, payloads, parallel),
            allowed_tools=["plan_tool", "todo_write_tool"],
            inherit_context=inherit_context,
            parent_depth=parent_depth,
            parent_agent_id=parent_agent_id,
            agent_role="coordinator",
        )

    def _format_coordinator_report(self, coordinator_payload: dict, payloads: List[dict], parallel: bool) -> str:
        completed = sum(1 for item in payloads if item.get("status") == "completed")
        crashed = sum(1 for item in payloads if item.get("status") == "crashed")
        lines = [
            "=======[ COORDINATOR REPORT ]=======",
            f"Execution mode: {'parallel' if parallel else 'serial'}",
            f"Child task count: {len(payloads)}",
            f"Completed: {completed}",
            f"Crashed: {crashed}",
            f"Coordinator agent: {coordinator_payload.get('agent_id')}",
            "",
            "Coordinator synthesis:",
            str(coordinator_payload.get("result", "")),
            "====================================",
        ]
        return "\n".join(lines)

    async def _run_fork_subagent(
        self,
        task: str,
        allowed_tools: List[str],
        parent_depth: int,
        parent_agent_id: str,
    ) -> dict:
        """
        Fork mode: child inherits the full parent context (messages, system prompt,
        plans, todos) and shares the prompt cache prefix for token efficiency.
        """
        from codeclaw.core.engine import QueryEngine

        child_agent_id = f"fork-{uuid.uuid4()}"
        sub_engine = QueryEngine(
            model=self.context.get("primary_model"),
            fallback_model=self.context.get("fallback_model"),
            permission_handler=self.context.get("permission_handler"),
            agent_id=child_agent_id,
            parent_agent_id=parent_agent_id,
            agent_depth=parent_depth + 1,
            agent_role="fork",
            model_provider=self.context.get("model_provider"),
            api_base_url=self.context.get("api_base_url"),
            local_tokenizer_path=self.context.get("local_tokenizer_path"),
        )
        self._configure_subagent_permissions(sub_engine)

        if self.context.get("plan_manager"):
            sub_engine.plan_manager = copy.deepcopy(self.context.get("plan_manager"))
            sub_engine.tool_context["plan_manager"] = sub_engine.plan_manager
            sub_engine.permission_manager.set_mode_getter(sub_engine.plan_manager.get_mode)
        if self.context.get("todo_manager"):
            sub_engine.todo_manager = copy.deepcopy(self.context.get("todo_manager"))
            sub_engine.tool_context["todo_manager"] = sub_engine.todo_manager

        self._share_runtime_managers(sub_engine)

        parent_tools = self.context.get("engine_available_tools")
        if parent_tools:
            sub_engine.available_tools = self._filter_tools(parent_tools, allowed_tools)
            sub_engine.tool_context["engine_available_tools"] = sub_engine.available_tools
            for tool in sub_engine.available_tools.values():
                tool.context = sub_engine.tool_context

        parent_messages = self._filter_incomplete_tool_messages(
            self.context.get("messages", [])
        )
        sub_engine.messages = copy.deepcopy(parent_messages)

        parent_context_builder = self.context.get("context_builder")
        if parent_context_builder is not None:
            sub_engine.context_builder = parent_context_builder

        sub_engine._sync_message_context()

        registry_record = {
            "agent_id": child_agent_id,
            "parent_agent_id": parent_agent_id,
            "agent_depth": parent_depth + 1,
            "agent_role": "fork",
            "status": "running",
            "task": task[:240] + ("..." if len(task) > 240 else ""),
        }
        await self._write_subagent_registry_record(registry_record)

        try:
            res = await sub_engine.run(
                user_input=(
                    "[SYSTEM: Fork mode active — you have the parent agent's full context. "
                    "Complete the task below and report findings concisely.]\n\n"
                    f"Task:\n{task}"
                ),
                sys_print_callback=lambda x: None,
            )
            self._merge_state_back_to_parent(sub_engine)
            await self._write_subagent_registry_record({**registry_record, "status": "completed"})
            return {
                "task": task,
                "agent_id": child_agent_id,
                "agent_role": "fork",
                "status": "completed",
                "result": res,
            }
        except Exception as e:
            await self._write_subagent_registry_record({**registry_record, "status": "crashed"})
            return {
                "task": task,
                "agent_id": child_agent_id,
                "agent_role": "fork",
                "status": "crashed",
                "result": f"Fork agent crash: {e}",
            }

    async def execute(
        self,
        description: str = "",
        prompt: str = "",
        subagent_type: str = None,
        model: str = None,
        run_in_background: bool = False,
    ) -> str:
        if not prompt:
            return "Sub-agent spawn denied: no prompt was provided."

        parent_depth = int(self.context.get("agent_depth", 0) or 0)
        if parent_depth >= MAX_SUBAGENT_DEPTH:
            return (
                f"Sub-agent spawn denied: maximum depth {MAX_SUBAGENT_DEPTH} reached. "
                "Complete the remaining work in the current agent."
            )

        builtin_def = get_builtin_agent(subagent_type) if subagent_type else None
        parent_agent_id = self.context.get("agent_id")

        payload = await self._run_single_subagent(
            task=prompt,
            allowed_tools=[],
            inherit_context=True,
            parent_depth=parent_depth,
            parent_agent_id=parent_agent_id,
            builtin_agent_def=builtin_def,
        )
        return self._format_single_report(payload)
