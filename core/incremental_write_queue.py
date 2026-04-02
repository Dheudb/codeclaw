import asyncio
import copy
import inspect
import os
import time


class IncrementalWriteQueue:
    def __init__(self):
        self._locks = {}
        self.history = []

    def _lock_for(self, path: str):
        abs_path = os.path.abspath(path)
        lock = self._locks.get(abs_path)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[abs_path] = lock
        return abs_path, lock

    async def run(self, path: str, operation: str, callback):
        abs_path, lock = self._lock_for(path)
        entry = {
            "path": abs_path,
            "operation": operation,
            "status": "queued",
            "queued_at": time.time(),
        }
        self.history.insert(0, entry)
        self.history = self.history[:40]
        async with lock:
            entry["status"] = "running"
            entry["started_at"] = time.time()
            try:
                result = callback()
                if inspect.isawaitable(result):
                    result = await result
                entry["finished_at"] = time.time()
                if isinstance(result, dict) and result.get("is_error"):
                    entry["status"] = "error"
                else:
                    entry["status"] = "ok"
                return result
            except Exception as e:
                entry["status"] = "exception"
                entry["finished_at"] = time.time()
                entry["error"] = str(e)
                raise

    def export_state(self) -> dict:
        return {
            "history": copy.deepcopy(self.history),
        }

    def load_state(self, payload):
        payload = payload or {}
        self.history = list(payload.get("history", []) or [])[:40]

    def render_summary(self) -> str:
        if not self.history:
            return "Incremental write queue: idle"
        latest = self.history[0]
        return (
            f"Incremental write queue: {len(self.history)} recorded writes, "
            f"last={latest.get('operation')} {latest.get('status')} {latest.get('path')}"
        )
