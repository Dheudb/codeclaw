"""Execution receipts describe harness termination, not domain-level acceptance."""
from dataclasses import asdict, dataclass, replace


class PersistenceError(OSError):
    """Execution cannot safely continue without its durable checkpoint."""


@dataclass(frozen=True)
class RunResult:
    text: str
    reason: str
    persisted: bool = False
    execution_reason: str = ""
    session_id: str = ""
    agent_id: str = ""
    user_message_id: str = ""

    @property
    def status(self):
        if self.reason == "completed" and self.persisted:
            return "completed"
        if self.reason == "cancelled":
            return "cancelled"
        return "crashed"

    @property
    def ok(self):
        return self.reason == "completed" and self.persisted

    def to_dict(self):
        return {**asdict(self), "status": self.status, "ok": self.ok}

    def __str__(self):
        return self.text


def finish_run(engine, text, reason):
    from xuloom.core.tool_execution import tool_identity
    user_message_id, _ = tool_identity(engine, "", {})
    owner = engine.tool_context.get("delegation", {}).get("owner_session_id")
    result = RunResult(str(text), reason, persisted=True, execution_reason=reason,
        session_id=owner or engine.session_id, agent_id=engine.agent_id or "", user_message_id=user_message_id)
    engine.last_run_result = result.to_dict()
    try:
        saved = engine.persist_session_state()
    except Exception:
        saved = False
    if not saved:
        result = replace(result, reason="persistence_error", persisted=False)
        engine.last_run_result = result.to_dict()
    return result
