"""File-tool changes, independent of Git and kept out of model messages."""
import difflib
import os
import time
from pathlib import Path

SNAPSHOT_LIMIT = 200_000


def describe_change(path, before, after, operation="edit"):
    available = before is not None and after is not None
    record = {"path": str(path), "operation": operation, "updatedAt": int(time.time() * 1000),
              "added": None, "removed": None, "diff": "", "statsAvailable": available, "truncated": False}
    if available:
        left, right = before.splitlines(), after.splitlines()
        matcher = difflib.SequenceMatcher(None, left, right)
        added = removed = 0
        for tag, i, j, k, l in matcher.get_opcodes():
            if tag in {"replace", "delete"}: removed += j - i
            if tag in {"replace", "insert"}: added += l - k
        diff = "\n".join(difflib.unified_diff(left, right, fromfile="修改前", tofile="修改后", lineterm=""))
        record.update(added=added, removed=removed, diff=diff[:SNAPSHOT_LIMIT], truncated=len(diff) > SNAPSHOT_LIMIT)
    return record


def notify_change(context, path, before, after, operation):
    callback = context.get("capability_event")
    if not callable(callback): return
    # Large files remain clickable, but unbounded snapshots never cross IPC.
    before = before if before is not None and len(before) <= SNAPSHOT_LIMIT else None
    after = after if after is not None and len(after) <= SNAPSHOT_LIMIT else None
    try:
        callback({"type": "file_changed", "change": describe_change(os.path.realpath(path), before, after, operation),
                  "before": before, "after": after, "agent_id": context.get("agent_id")})
    except Exception:
        # A UI observer must never turn an already successful write into a tool failure.
        pass


def read_snapshot(path):
    try:
        with open(path, "r", encoding="utf-8") as stream:
            value = stream.read(SNAPSHOT_LIMIT + 1)
        return value if len(value) <= SNAPSHOT_LIMIT else None
    except (OSError, UnicodeError): return None


def legacy_changes(activity, cwd):
    files = {}
    for tool in activity:
        inputs = tool.get("input", {})
        path = inputs.get("file_path") or inputs.get("notebook_path")
        if tool.get("status") != "completed" or tool.get("name") not in {"file_write_tool", "file_edit_tool", "notebook_edit_tool"} or not path:
            continue
        path = str((Path(cwd) / path).resolve())
        created = "operation: create" in tool.get("result", "") or "Successfully created" in tool.get("result", "")
        after = inputs.get("content", inputs.get("new_string")) if created else None
        record = describe_change(path, "" if created else None, after, "create" if created else "edit")
        record["updatedAt"] = tool.get("finishedAt") or tool.get("startedAt") or 0
        record["legacy"] = True
        files[os.path.normcase(path)] = record
    return list(files.values())


def preview_file(cwd, requested, known):
    if not isinstance(requested, str) or not requested.strip() or "\0" in requested or "://" in requested:
        raise ValueError("请选择本地文件路径")
    path = (Path(cwd) / requested).resolve()
    allowed = {os.path.normcase(os.path.abspath(item["path"])) for item in known if item.get("path")}
    try: path.relative_to(Path(cwd).resolve())
    except ValueError:
        if os.path.normcase(str(path)) not in allowed: raise ValueError("只能查看当前工作目录或本会话实际修改的文件")
    if not path.is_file(): raise ValueError("文件已被移动、删除，或路径不是文件")
    with path.open("rb") as stream: data = stream.read(500_001)
    truncated = len(data) > 500_000
    data = data[:500_000]
    if b"\0" in data: raise ValueError("此文件是二进制文件，暂不支持文本预览")
    try: content = data.decode("utf-8-sig", errors="strict" if not truncated else "replace")
    except UnicodeError: raise ValueError("此文件不是 UTF-8 文本，暂不支持预览")
    return {"path": str(path), "content": content, "truncated": truncated, "size": path.stat().st_size,
            "modifiedAt": int(path.stat().st_mtime * 1000)}
