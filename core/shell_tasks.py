import asyncio
import os
import time
import uuid
from typing import Optional


class ShellTaskManager:
    """
    Manages long-running shell tasks with file-backed logs.
    """

    def __init__(self, log_dir=".codeclaw/shell-tasks"):
        self.log_dir = log_dir
        self.tasks = {}

    async def start_task(self, command: str, cwd: str = None) -> dict:
        os.makedirs(self.log_dir, exist_ok=True)
        task_id = f"task-{uuid.uuid4().hex[:12]}"
        log_path = os.path.abspath(os.path.join(self.log_dir, f"{task_id}.log"))
        normalized_cwd = os.path.abspath(cwd or os.getcwd())

        log_handle = open(log_path, "ab")
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=normalized_cwd,
            stdout=log_handle,
            stderr=log_handle,
        )

        self.tasks[task_id] = {
            "task_id": task_id,
            "command": command,
            "cwd": normalized_cwd,
            "log_path": log_path,
            "started_at": time.time(),
            "status": "running",
            "pid": process.pid,
            "exit_code": None,
            "process": process,
            "log_handle": log_handle,
        }
        return self._export_record(task_id)

    def get_task(self, task_id: str) -> Optional[dict]:
        if task_id not in self.tasks:
            return None
        self._refresh_task(task_id)
        return self._export_record(task_id)

    def list_tasks(self) -> list[dict]:
        for task_id in list(self.tasks.keys()):
            self._refresh_task(task_id)
        return [self._export_record(task_id) for task_id in self.tasks]

    def read_output(self, task_id: str, tail_chars: int = 4000) -> Optional[dict]:
        record = self.get_task(task_id)
        if not record:
            return None

        try:
            with open(record["log_path"], "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except FileNotFoundError:
            content = ""

        if tail_chars and len(content) > tail_chars:
            content = content[-tail_chars:]

        record["output"] = content
        record["output_chars"] = len(content)
        return record

    async def terminate_task(self, task_id: str) -> Optional[dict]:
        if task_id not in self.tasks:
            return None

        record = self.tasks[task_id]
        process = record.get("process")
        if process and process.returncode is None:
            process.kill()
            await process.wait()
        self._refresh_task(task_id)
        return self._export_record(task_id)

    def export_state(self) -> list[dict]:
        return self.list_tasks()

    def load_state(self, payload: list):
        self.tasks = {}
        for item in payload or []:
            task_id = item.get("task_id")
            if not task_id:
                continue
            self.tasks[task_id] = {
                "task_id": task_id,
                "command": item.get("command", ""),
                "cwd": item.get("cwd", os.getcwd()),
                "log_path": item.get("log_path", ""),
                "started_at": item.get("started_at", time.time()),
                "status": item.get("status", "unknown"),
                "pid": item.get("pid"),
                "exit_code": item.get("exit_code"),
                "process": None,
                "log_handle": None,
            }

    def _refresh_task(self, task_id: str):
        record = self.tasks[task_id]
        process = record.get("process")
        if process is None:
            if record["status"] == "running":
                record["status"] = "detached"
            return

        if process.returncode is None:
            record["status"] = "running"
            return

        record["status"] = "completed" if process.returncode == 0 else "failed"
        record["exit_code"] = process.returncode
        log_handle = record.get("log_handle")
        if log_handle and not log_handle.closed:
            log_handle.close()
        record["log_handle"] = None

    def _export_record(self, task_id: str) -> dict:
        record = self.tasks[task_id]
        self._refresh_task(task_id)
        return {
            "task_id": record["task_id"],
            "command": record["command"],
            "cwd": record["cwd"],
            "log_path": record["log_path"],
            "started_at": record["started_at"],
            "status": record["status"],
            "pid": record["pid"],
            "exit_code": record["exit_code"],
        }
