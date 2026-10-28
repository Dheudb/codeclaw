"""Durable timed messages; execution belongs to the existing session owner."""
import copy
import json
import math
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from xuloom.core.session_lock import SessionLock
from xuloom.core.storage import atomic_json, data_directory
from xuloom.features.goals import save


@contextmanager
def store(write=True):
    lock = SessionLock(data_directory() / "session-locks", "scheduled_messages")
    lock.acquire()
    try:
        path = data_directory() / "schedules.json"
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"version": 1, "items": {}}
        if data.get("version") != 1 or not isinstance(data.get("items"), dict): raise ValueError("Invalid schedules file; preserved")
        yield data["items"]
        if write: atomic_json(path, data)
    finally: lock.close()


def list_items(*, due=False, now=None):
    now = time.time() if now is None else now
    with store(write=False) as items:
        return sorted([copy.deepcopy(row) for row in items.values() if not due or row["status"] == "active" and row["due_at"] <= now], key=lambda row: (row["due_at"], row["id"]))


def add(engine, prompt, *, due_at, interval=None, skills=None):
    engine.require_writer()
    if not isinstance(prompt, str) or not prompt.strip(): raise ValueError("Scheduled message cannot be empty")
    if type(due_at) not in (int, float) or not math.isfinite(due_at) or due_at < 0: raise ValueError("Invalid due time")
    if interval is not None and (type(interval) not in (int, float) or not math.isfinite(interval) or interval < 60):
        raise ValueError("Repeat interval must be at least 60 seconds")
    for skill in skills or []: engine.tool_context["skill_registry"].resolve(skill)
    save(engine)  # A schedule must target a real, durable session.
    row = dict(id=str(uuid.uuid4()), session_id=engine.session_id, cwd=engine.session_cwd, prompt=prompt,
        due_at=due_at, interval=interval, skills=list(skills or []), status="active", last_delivery=None, reason="")
    with store() as items: items[row["id"]] = row
    return copy.deepcopy(row)


def change(identity, action):
    with store() as items:
        row = items[identity]
        if action == "cancel": row["status"] = "cancelled"
        elif action == "pause": row["status"] = "paused"
        elif action == "resume":
            if row["status"] in {"completed", "cancelled", "blocked"}:
                raise ValueError("Completed, cancelled or interrupted deliveries cannot be replayed; create a new schedule after inspecting effects")
            row["status"] = "active"
        else: raise ValueError("Unknown schedule action")
        return copy.deepcopy(row)


async def deliver(engine, identity, execute):
    """Claim only while executing. Lost acknowledgement is reconciled from the session receipt."""
    if engine.has_active_turn() or getattr(engine, "_goal_task", None) and not engine._goal_task.done():
        return {"status": "queued", "reason": "Existing work owns this session"}
    engine.require_writer()
    lease = SessionLock(data_directory() / "session-locks", "wake_" + identity)
    lease.acquire()
    try:
        with store(write=False) as items:
            row = copy.deepcopy(items[identity])
        if row["session_id"] != engine.session_id or os.path.normcase(os.path.realpath(row["cwd"])) != os.path.normcase(engine.session_cwd):
            raise ValueError("Schedule target moved; cancel and recreate it for the current conversation directory")
        if row["status"] != "active" or row["due_at"] > time.time(): return {"status": "not_due"}
        if engine.goal and engine.goal["status"] == "active": return {"status": "queued", "reason": "Goal owns this session"}
        occurrence = identity + ":" + str(row["due_at"])
        receipt = engine.scheduled_deliveries.get(identity)
        if receipt and receipt["occurrence"] == occurrence:
            if receipt["status"] != "completed":
                with store() as items:
                    items[identity]["status"], items[identity]["reason"] = "blocked", "Delivery was interrupted or failed; inspect session effects. It was not replayed."
                return {"status": "blocked", "receipt": receipt}
            save(engine)  # In-memory completion may follow a failed final checkpoint; never acknowledge it unsaved.
        else:
            receipt = dict(occurrence=occurrence, status="executing", started_at=time.time(), result=None)
            engine.scheduled_deliveries[identity] = receipt
            save(engine)
            try:
                result = await execute(row["prompt"], row["skills"])
                receipt.update(status="completed" if result.get("ok") else "failed", result=result, finished_at=time.time())
            except BaseException:
                receipt["status"] = "interrupted"
                save(engine)
                raise
            save(engine)
        with store() as items:
            current = items[identity]
            current["last_delivery"] = copy.deepcopy(receipt)
            if current["status"] == "active":
                if receipt["status"] != "completed": current["status"], current["reason"] = "blocked", "Framework execution failed; inspect the saved result"
                elif row["interval"]:
                    # Coalesce missed intervals; never burst-replay past occurrences.
                    current["due_at"] = row["due_at"] + (math.floor(max(0, time.time() - row["due_at"]) / row["interval"]) + 1) * row["interval"]
                else: current["status"] = "completed"
        return copy.deepcopy(receipt)
    finally: lease.close()
