import copy
import inspect
import time
import uuid


class AutoCommitManager:
    def __init__(self, state_change_callback=None):
        self.state_change_callback = state_change_callback
        self.pending_proposal = None
        self.proposal_history = []

    def set_state_change_callback(self, callback):
        self.state_change_callback = callback

    async def _notify_state_change(self):
        if not callable(self.state_change_callback):
            return
        result = self.state_change_callback()
        if inspect.isawaitable(result):
            await result

    async def create_proposal(self, payload: dict) -> dict:
        proposal = {
            "proposal_id": uuid.uuid4().hex,
            "created_at": time.time(),
            **copy.deepcopy(payload or {}),
        }
        self.pending_proposal = proposal
        self.proposal_history.insert(0, {
            "proposal_id": proposal["proposal_id"],
            "status": "pending",
            "created_at": proposal["created_at"],
            "repo_root": proposal.get("repo_root"),
            "branch": proposal.get("branch"),
            "file_count": len(proposal.get("files", []) or []),
            "message_source": proposal.get("message_source"),
        })
        self.proposal_history = self.proposal_history[:20]
        await self._notify_state_change()
        return proposal

    async def resolve_pending(self, *, status: str, metadata: dict = None):
        if self.pending_proposal is None:
            return None
        proposal = copy.deepcopy(self.pending_proposal)
        proposal["resolved_at"] = time.time()
        proposal["status"] = status
        proposal["resolution_metadata"] = copy.deepcopy(metadata or {})
        self.proposal_history.insert(0, {
            "proposal_id": proposal.get("proposal_id"),
            "status": status,
            "created_at": proposal.get("created_at"),
            "resolved_at": proposal.get("resolved_at"),
            "repo_root": proposal.get("repo_root"),
            "branch": proposal.get("branch"),
            "file_count": len(proposal.get("files", []) or []),
            "message_source": proposal.get("message_source"),
        })
        self.proposal_history = self.proposal_history[:20]
        self.pending_proposal = None
        await self._notify_state_change()
        return proposal

    def export_state(self) -> dict:
        return {
            "pending_proposal": copy.deepcopy(self.pending_proposal),
            "proposal_history": copy.deepcopy(self.proposal_history),
        }

    def load_state(self, payload):
        payload = payload or {}
        pending = payload.get("pending_proposal")
        self.pending_proposal = copy.deepcopy(pending) if isinstance(pending, dict) else None
        self.proposal_history = list(payload.get("proposal_history", []) or [])[:20]
