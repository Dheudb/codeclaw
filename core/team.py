"""
Team management for multi-agent coordination.

Provides in-process team state, member registry, and async message queues
so a coordinator agent can spawn workers and exchange messages with them.
"""

import asyncio
import time
import uuid
from typing import Optional


TEAM_LEAD_NAME = "team-lead"


class TeamMember:
    __slots__ = (
        "agent_id", "name", "agent_type", "joined_at",
        "is_active", "cwd", "model", "pending_messages",
    )

    def __init__(
        self,
        agent_id: str,
        name: str,
        agent_type: str = "",
        cwd: str = "",
        model: str = "",
    ):
        self.agent_id = agent_id
        self.name = name
        self.agent_type = agent_type or name
        self.joined_at = time.time()
        self.is_active = True
        self.cwd = cwd
        self.model = model
        self.pending_messages: asyncio.Queue = asyncio.Queue()

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "agent_type": self.agent_type,
            "joined_at": self.joined_at,
            "is_active": self.is_active,
            "cwd": self.cwd,
            "model": self.model,
            "pending_message_count": self.pending_messages.qsize(),
        }


class TeamMessage:
    __slots__ = ("id", "from_name", "to_name", "text", "timestamp", "msg_type")

    def __init__(self, from_name: str, to_name: str, text: str, msg_type: str = "text"):
        self.id = str(uuid.uuid4())
        self.from_name = from_name
        self.to_name = to_name
        self.text = text
        self.timestamp = time.time()
        self.msg_type = msg_type

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "from": self.from_name,
            "to": self.to_name,
            "text": self.text,
            "timestamp": self.timestamp,
            "type": self.msg_type,
        }


class TeamManager:
    """
    In-process team state for one coordinator session.
    Manages member registry and async message routing.
    """

    def __init__(self):
        self.team_name: str = ""
        self.description: str = ""
        self.lead_agent_id: str = ""
        self.created_at: float = 0
        self.members: dict[str, TeamMember] = {}
        self.message_log: list[dict] = []
        self._active = False

    @property
    def is_active(self) -> bool:
        return self._active

    def create_team(self, team_name: str, lead_agent_id: str, description: str = "", lead_model: str = "") -> dict:
        if self._active:
            return {
                "ok": False,
                "error": f'Already leading team "{self.team_name}". Delete current team first.',
            }
        self.team_name = team_name
        self.description = description
        self.lead_agent_id = lead_agent_id
        self.created_at = time.time()
        self._active = True

        lead = TeamMember(
            agent_id=lead_agent_id,
            name=TEAM_LEAD_NAME,
            agent_type=TEAM_LEAD_NAME,
            model=lead_model,
        )
        self.members = {lead_agent_id: lead}
        self.message_log = []

        return {
            "ok": True,
            "team_name": self.team_name,
            "lead_agent_id": lead_agent_id,
        }

    def delete_team(self) -> dict:
        if not self._active:
            return {"ok": True, "message": "No active team."}

        active_non_lead = [
            m for m in self.members.values()
            if m.name != TEAM_LEAD_NAME and m.is_active
        ]
        if active_non_lead:
            names = ", ".join(m.name for m in active_non_lead)
            return {
                "ok": False,
                "error": f"Cannot delete team with {len(active_non_lead)} active member(s): {names}. "
                         "Mark them inactive first.",
            }

        old_name = self.team_name
        self.team_name = ""
        self.description = ""
        self.lead_agent_id = ""
        self.members.clear()
        self.message_log.clear()
        self._active = False
        return {"ok": True, "message": f'Team "{old_name}" deleted.'}

    def register_member(
        self,
        agent_id: str,
        name: str,
        agent_type: str = "",
        cwd: str = "",
        model: str = "",
    ) -> TeamMember:
        member = TeamMember(
            agent_id=agent_id,
            name=name,
            agent_type=agent_type,
            cwd=cwd,
            model=model,
        )
        self.members[agent_id] = member
        return member

    def deactivate_member(self, agent_id: str):
        member = self.members.get(agent_id)
        if member:
            member.is_active = False

    def find_member_by_name(self, name: str) -> Optional[TeamMember]:
        for member in self.members.values():
            if member.name == name:
                return member
        return None

    def find_member_by_id(self, agent_id: str) -> Optional[TeamMember]:
        return self.members.get(agent_id)

    def send_message(self, from_name: str, to_name: str, text: str, msg_type: str = "text") -> dict:
        """Route a message to a team member's async queue."""
        if not self._active:
            return {"ok": False, "error": "No active team."}

        msg = TeamMessage(from_name=from_name, to_name=to_name, text=text, msg_type=msg_type)
        self.message_log.append(msg.to_dict())

        if to_name == "*":
            count = 0
            for member in self.members.values():
                if member.name != from_name and member.is_active:
                    member.pending_messages.put_nowait(msg)
                    count += 1
            return {"ok": True, "delivered_to": count, "broadcast": True}

        target = self.find_member_by_name(to_name)
        if not target:
            return {"ok": False, "error": f'No team member named "{to_name}".'}
        if not target.is_active:
            return {"ok": False, "error": f'Team member "{to_name}" is not active.'}

        target.pending_messages.put_nowait(msg)
        return {"ok": True, "delivered_to": 1, "broadcast": False}

    async def receive_message(self, agent_id: str, timeout: float = 0) -> Optional[TeamMessage]:
        """Receive next message for a member, optionally with timeout."""
        member = self.members.get(agent_id)
        if not member:
            return None
        try:
            if timeout > 0:
                return await asyncio.wait_for(member.pending_messages.get(), timeout=timeout)
            else:
                return member.pending_messages.get_nowait()
        except (asyncio.TimeoutError, asyncio.QueueEmpty):
            return None

    def get_active_members(self) -> list[TeamMember]:
        return [m for m in self.members.values() if m.is_active]

    def get_non_lead_members(self) -> list[TeamMember]:
        return [m for m in self.members.values() if m.name != TEAM_LEAD_NAME]

    def export_state(self) -> dict:
        return {
            "team_name": self.team_name,
            "description": self.description,
            "lead_agent_id": self.lead_agent_id,
            "active": self._active,
            "created_at": self.created_at,
            "members": {aid: m.to_dict() for aid, m in self.members.items()},
            "message_log": self.message_log[-50:],
        }

    def load_state(self, state: dict):
        self.team_name = state.get("team_name", "")
        self.description = state.get("description", "")
        self.lead_agent_id = state.get("lead_agent_id", "")
        self._active = state.get("active", False)
        self.created_at = state.get("created_at", 0)
        self.message_log = state.get("message_log", [])
        self.members.clear()
        for aid, mdata in state.get("members", {}).items():
            member = TeamMember(
                agent_id=aid,
                name=mdata.get("name", ""),
                agent_type=mdata.get("agent_type", ""),
                cwd=mdata.get("cwd", ""),
                model=mdata.get("model", ""),
            )
            member.is_active = mdata.get("is_active", True)
            member.joined_at = mdata.get("joined_at", 0)
            self.members[aid] = member

    def get_summary(self) -> str:
        if not self._active:
            return "No active team."
        lines = [
            f"Team: {self.team_name}",
            f"Description: {self.description or '(none)'}",
            f"Lead: {self.lead_agent_id[:8]}...",
            f"Members ({len(self.members)}):",
        ]
        for m in self.members.values():
            status = "active" if m.is_active else "inactive"
            pending = m.pending_messages.qsize()
            lines.append(f"  - {m.name} ({m.agent_type}) [{status}] msgs_pending={pending}")
        lines.append(f"Messages exchanged: {len(self.message_log)}")
        return "\n".join(lines)
