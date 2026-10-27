"""Session-owned, resumable workers. Host liveness is distinct from saved output."""
import asyncio
import copy
import json
import re
import time
import uuid
from pathlib import Path


def concurrency_limit(context):
    registry = context.get("agent_registry")
    value = registry.skills.settings().get("agent_concurrency", 10) if registry else 10
    if type(value) is not int or not 1 <= value <= 80:
        raise ValueError("agent_concurrency must be an integer from 1 to 80")
    return value


class AgentLifecycle:
    def _worker(self, identity):
        if not isinstance(identity, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", identity):
            raise ValueError("A valid agent_id is required")
        row = next((r for r in self.context.get("subagent_registry", []) if r.get("agent_id") == identity), None)
        if row is None:
            raise ValueError("Unknown subagent in this conversation")
        return row

    def _inventory(self):
        tasks = self.context["subagent_tasks"]
        rows = []
        for saved in self.context.get("subagent_registry", []):
            row = copy.deepcopy(saved)
            task = tasks.tasks.get(row["agent_id"])
            row["executing"] = bool(task and not task.done())
            if row.get("status") == "running" and not row["executing"]:
                row["status"] = "interrupted"
            row["pending_message_count"] = len(row.pop("pending_messages", []))
            commands = self.context.get("command_sessions")
            row["running_commands"] = sum(r["owner"] == row["agent_id"] and r["status"] == "running" for r in commands.records.values()) if commands else 0
            row.pop("role_definition", None)
            rows.append(row)
        return rows

    def _resume_state(self, row):
        manager = self.context.get("session_manager")
        if manager is None:
            raise ValueError("Subagent transcript storage is unavailable")
        path = Path(manager.get_subagent_transcript_path(row["agent_id"]))
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or not isinstance(raw.get("messages"), list):
            raise ValueError("Invalid subagent transcript")
        # Resolve from our own central storage, never a model-supplied path.
        owner = raw.get("session_id") or raw.get("metadata", {}).get("delegation", {}).get("owner_session_id")
        if owner and owner != self.context.get("session_id"):
            raise ValueError("Subagent belongs to a different conversation")
        runtime = raw.get("metadata", {}).get("model_runtime", {})
        for field, key in (("provider", "model_provider"), ("api_base_url", "api_base_url")):
            if field in runtime and (runtime[field] or "") != (self.context.get(key) or ""):
                raise ValueError("Restore the worker's original provider and API address before continuing")
        return raw

    async def _manage_worker(self, action, agent_id, prompt, timeout_ms, agent_ids, run_in_background):
        tasks = self.context["subagent_tasks"]
        encode = lambda data: json.dumps(data, ensure_ascii=False)
        if action == "capacity":
            active = sum(not t.done() for t in tasks.tasks.values())
            return encode({"limit": tasks.limit, "active": active, "available": max(0, tasks.limit - active),
                           "max_batch_size": 10, "resume": True, "messages": True, "wait_timeout_ms": 60000})
        if action == "list":
            return encode({"agents": self._inventory()})
        if action == "wait":
            ids = agent_ids or ([agent_id] if agent_id else list(tasks.tasks))
            for identity in ids:
                self._worker(identity)
            live = [tasks.tasks[i] for i in ids if i in tasks.tasks and not tasks.tasks[i].done()]
            if live:
                await asyncio.wait(live, timeout=timeout_ms / 1000, return_when=asyncio.FIRST_COMPLETED)
            rows = [r for r in self._inventory() if r["agent_id"] in ids]
            tasks.delivered.update(r["agent_id"] for r in rows if not r["executing"])
            return encode({"agents": rows, "timed_out": bool(live) and all(not t.done() for t in live)})
        row = self._worker(agent_id)
        active = tasks.tasks.get(agent_id)
        running = bool(active and not active.done())
        if action == "result":
            return encode(next(r for r in self._inventory() if r["agent_id"] == agent_id))
        if action == "cancel":
            if running:
                await tasks.cancel(agent_id)
            if self.context.get("command_sessions"):
                await self.context["command_sessions"].close(owner=agent_id)
            current = self._worker(agent_id)
            if current.get("status") not in {"cancelled", "completed", "crashed", "closed"}:
                await self._write_subagent_registry_record({"agent_id": agent_id, "status": "cancelled", "finished_at": int(time.time() * 1000)})
            return encode({"agent_id": agent_id, "status": self._worker(agent_id)["status"], "executing": False})
        if action == "close":
            if running:
                raise ValueError("Cannot close a running worker; cancel or wait first")
            if self.context.get("command_sessions"):
                await self.context["command_sessions"].close(owner=agent_id)
            await self._write_subagent_registry_record({"agent_id": agent_id, "status": "closed"})
            return encode({"agent_id": agent_id, "status": "closed", "history_retained": True})
        if action == "send":
            if row.get("status") == "closed":
                raise ValueError("Worker is closed")
            if not prompt.strip():
                raise ValueError("A message prompt is required")
            pending = list(row.get("pending_messages", []))
            if len(pending) >= 100:
                raise ValueError("Worker mailbox is full")
            message = {"id": str(uuid.uuid4()), "text": prompt, "at": int(time.time() * 1000)}
            await self._write_subagent_registry_record({"agent_id": agent_id, "pending_messages": pending + [message]})
            return encode({"agent_id": agent_id, "message_id": message["id"], "queued": True,
                           "executing": running, "requires_followup": not running})
        if action != "followup":
            raise ValueError("Unknown subagent action")
        if running:
            raise ValueError("Worker is running; use send to queue a message")
        if row.get("status") == "closed":
            raise ValueError("Worker is closed")
        if not prompt.strip():
            raise ValueError("A followup prompt is required")
        tasks.check_capacity(1)
        state = self._resume_state(row)
        definition = row.get("role_definition") or state.get("metadata", {}).get("delegation", {}).get("role_definition")
        if not definition:
            raise ValueError("This legacy worker has no frozen role; start a new worker")
        record = {**row, "status": "running", "phase": "starting", "finished_at": None,
                  "turn": row.get("turn", 1) + 1, "background": run_in_background}
        await self._write_subagent_registry_record(record)
        work = self._run_registered_subagent(record, task=prompt, allowed_tools=definition.get("allowed_tools"),
            inherit_context=False, parent_depth=row.get("agent_depth", 1) - 1,
            parent_agent_id=row.get("parent_agent_id"), builtin_agent_def=definition,
            model=row.get("model"), child_agent_id=agent_id, background=run_in_background,
            description=row.get("description", ""), resume_state=state)
        tasks.launch(agent_id, work, on_cancelled_before_start=self.context.get("register_subagent_record"))
        if not run_in_background:
            await tasks.tasks[agent_id]
            tasks.delivered.add(agent_id)
        return encode(next(r for r in self._inventory() if r["agent_id"] == agent_id))

    def _mailbox(self, identity):
        return copy.deepcopy(self._worker(identity).get("pending_messages", []))

    def _ack_mailbox(self, identity, message_ids):
        row = self._worker(identity)
        pending = [m for m in row.get("pending_messages", []) if m["id"] not in message_ids]
        writer = self.context.get("register_subagent_record")
        if writer:
            writer({"agent_id": identity, "pending_messages": pending})
