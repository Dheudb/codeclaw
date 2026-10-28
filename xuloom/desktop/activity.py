"""Rebuild desktop activity from the shared CLI/desktop transcripts."""
import json
import re
from datetime import datetime
from xuloom.core.storage import session_directory


def timestamp(message):
    try:
        return int(datetime.fromisoformat(message.get("_timestamp", "").replace("Z", "+00:00")).timestamp() * 1000)
    except (ValueError, TypeError):
        return None


def transcript_activity(messages, agent_id=None, agent_role="main"):
    tools = {}
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use" and block.get("id"):
                tools[block["id"]] = {"id": f"{agent_id or 'main'}:{block['id']}", "name": block.get("name", "tool"),
                                      "input": block.get("input", {}), "status": "interrupted",
                                      "agentId": agent_id, "agentRole": agent_role, "startedAt": timestamp(message)}
            elif block.get("type") == "tool_result" and block.get("tool_use_id") in tools:
                tools[block["tool_use_id"]].update(status="error" if block.get("is_error") else "completed",
                                                  result=str(block.get("content", ""))[:16000], finishedAt=timestamp(message))
    return list(tools.values())


def restore_activity(raw, cwd):
    if isinstance(raw, list):
        return transcript_activity(raw)[-200:], []
    records = raw.get("metadata", {}).get("subagents", [])
    activity = transcript_activity(raw.get("messages", []))
    workers = []
    for record in records:
        agent_id = record.get("agent_id", "")
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", agent_id):
            continue
        worker = dict(record)
        for key in ("dispatch", "api_base_url", "role_definition"):
            worker.pop(key, None)
        if worker.get("status") == "running":
            worker["status"] = "interrupted"
        elif worker.get("status") == "queued":
            worker["phase"] = "awaiting_resume"
        workers.append(worker)
        activity.append({"id": "agent:" + agent_id, "name": "subagent", "agentId": agent_id,
                         "agentRole": worker.get("agent_role"), "status": worker.get("status", "interrupted"),
                         "input": {"description": worker.get("description"), "prompt": worker.get("task")},
                         "startedAt": worker.get("started_at"), "finishedAt": worker.get("finished_at"),
                         "result": str(worker.get("result", ""))[:16000]})
        try:
            child = json.loads((session_directory(cwd) / "subagents" / (agent_id + ".json")).read_text(encoding="utf-8"))
            messages = child.get("messages", [])
            task = child.get("task", "")
            for index, message in enumerate(messages):
                content = message.get("content")
                if message.get("role") == "user" and isinstance(content, str) and (content == task or content.endswith("Task:\n" + task)):
                    messages = messages[index:]
                    break
            activity.extend(transcript_activity(messages, agent_id, worker.get("agent_role", "subagent")))
        except (OSError, ValueError, TypeError, AttributeError):
            continue
    activity.sort(key=lambda item: item.get("startedAt") or 0)
    return activity[-200:], workers
