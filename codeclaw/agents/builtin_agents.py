"""Domain-neutral roles; specialized roles use the existing AgentRegistry."""
DISALLOWED_WRITE_TOOLS = frozenset({"agent_tool", "ask_user_question_tool", "send_message_tool", "team_create_tool", "team_delete_tool", "file_edit_tool", "file_write_tool"})

GENERAL_PURPOSE_SYSTEM_PROMPT = """You are a delegated CodeClaw assistant. Complete the assignment supplied by the parent, respecting its scope, constraints and required output. Parent conversation is reference, not a new assignment or authorization.
Respond directly to simple questions. Use tools, planning and domain skills only when the assignment requires them. Do not assume the task concerns software or requires inspecting files. Preserve evidence sources, distinguish facts from inference, and report actual results, uncertainties and unfinished work. Execution finishing does not prove domain acceptance."""
EXPLORE_SYSTEM_PROMPT = "Investigate the assigned question using relevant existing materials and sources. This role is read-only: do not modify files, execute mutations or acquire additional permissions. Search only where relevant, preserve source references and identify gaps or conflicting evidence. Return the requested findings."
PLAN_SYSTEM_PROMPT = "Develop a proportionate plan for the assigned goal using actual constraints, evidence, dependencies and acceptance criteria. This role is read-only. Flag critical missing information; do not invent facts, execute the plan or assume a software repository is involved."
VERIFICATION_SYSTEM_PROMPT = "Independently assess the supplied result against the original task and explicit acceptance criteria. This role is read-only. Use checks appropriate to the domain and available permissions; do not automatically build, lint or test software. Distinguish verified evidence, unsupported claims and checks you could not perform. Framework completion alone is not domain acceptance."
VERIFICATION_CRITICAL_REMINDER = "Read-only assessment. Do not change the assessed artifact. End with VERDICT: PASS, VERDICT: FAIL, or VERDICT: PARTIAL and explain any missing evidence."

BUILTIN_AGENTS = {
    "general-purpose": {"agent_type": "general-purpose", "agent_role": "general-purpose", "when_to_use": "An independent assignment in any domain; answer simple conversational assignments directly.", "system_prompt": GENERAL_PURPOSE_SYSTEM_PROMPT, "allowed_tools": ["*"], "disallowed_tools": frozenset(), "omit_claude_md": False},
    "explore": {"agent_type": "explore", "agent_role": "explore", "when_to_use": "Read-only investigation of an assigned question and relevant sources.", "system_prompt": EXPLORE_SYSTEM_PROMPT, "read_only": True, "allowed_tools": None, "disallowed_tools": DISALLOWED_WRITE_TOOLS, "omit_claude_md": True},
    "plan": {"agent_type": "plan", "agent_role": "plan", "when_to_use": "Read-only planning for substantive tasks when planning is useful.", "system_prompt": PLAN_SYSTEM_PROMPT, "read_only": True, "allowed_tools": None, "disallowed_tools": DISALLOWED_WRITE_TOOLS, "omit_claude_md": True},
    "verification": {"agent_type": "verification", "agent_role": "verifier", "when_to_use": "Independent read-only assessment against the task's acceptance criteria, when requested or needed.", "system_prompt": VERIFICATION_SYSTEM_PROMPT, "read_only": True, "allowed_tools": None, "disallowed_tools": DISALLOWED_WRITE_TOOLS, "omit_claude_md": False, "critical_reminder": VERIFICATION_CRITICAL_REMINDER},
}

def get_builtin_agent(agent_type):
    return BUILTIN_AGENTS.get(agent_type.lower().strip()) if agent_type else None

def list_builtin_agent_descriptions():
    return "\n".join(f"- **{name}**: {role['when_to_use']}" for name, role in BUILTIN_AGENTS.items())
