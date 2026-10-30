import asyncio
import copy
import inspect
import os
import time
import hashlib
from weakref import WeakValueDictionary
from xuloom.core.session_lock import SessionLock, SessionBusyError
from xuloom.core.storage import data_directory


class IncrementalWriteQueue:
    def __init__(self, lock_directory=None, lock_timeout=30):
        self._locks = WeakValueDictionary()
        self.history = []
        self.lock_directory = lock_directory or data_directory() / "file-locks"
        self.lock_timeout = lock_timeout

    def _lock_for(self, path: str):
        abs_path = os.path.normcase(os.path.realpath(path))
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
        lease = SessionLock(self.lock_directory, hashlib.sha256(abs_path.encode()).hexdigest())
        try:
            async with lock:
                deadline = time.monotonic() + self.lock_timeout
                while True:
                    try:
                        lease.acquire()
                        break
                    except SessionBusyError:
                        if time.monotonic() >= deadline:
                            raise TimeoutError("File is being modified by another process; no write was performed.") from None
                        await asyncio.sleep(.02)
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
                except asyncio.CancelledError:
                    entry["status"] = "cancelled"
                    entry["finished_at"] = time.time()
                    raise
                except Exception as e:
                    entry["status"] = "exception"
                    entry["finished_at"] = time.time()
                    entry["error"] = str(e)
                    raise
        except asyncio.CancelledError:
            entry["status"] = "cancelled"
            entry["finished_at"] = time.time()
            raise
        except Exception as error:
            entry.update(status="exception", finished_at=time.time(), error=str(error))
            raise
        finally:
            lease.close()

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
