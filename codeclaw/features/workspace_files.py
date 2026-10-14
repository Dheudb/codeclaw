"""Read-only workspace navigation and Git review, shared by desktop RPCs."""
import asyncio
import os
from pathlib import Path

from codeclaw.features.file_changes import SNAPSHOT_LIMIT, describe_change, read_snapshot


def workspace_path(cwd, requested=""):
    root = Path(cwd).resolve()
    if not isinstance(requested, str) or "\0" in requested or "://" in requested:
        raise ValueError("请选择工作区内的本地路径")
    path = (root / requested).resolve()
    if not path.is_relative_to(root):
        raise ValueError("只能浏览当前工作目录")
    return path


def list_directory(cwd, requested=""):
    directory = workspace_path(cwd, requested)
    if not directory.is_dir():
        raise ValueError("目录已被移动或删除")
    entries = []
    with os.scandir(directory) as children:
        for child in children:
            if child.name == ".git":
                continue
            try:
                resolved = Path(child.path).resolve()
                inside = resolved.is_relative_to(Path(cwd).resolve())
                entries.append({"name": child.name, "path": child.path,
                                "directory": child.is_dir(), "available": inside})
            except (OSError, RuntimeError):
                continue
    entries.sort(key=lambda item: (not item["directory"], item["name"].casefold()))
    return {"path": str(directory), "entries": entries[:1000], "truncated": len(entries) > 1000}


async def git(cwd, *args):
    process = await asyncio.create_subprocess_exec(
        "git", "-c", "core.quotepath=false", "-c", "diff.relative=false", *args, cwd=str(cwd),
        stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        creationflags=0x08000000 if os.name == "nt" else 0)
    try:
        output, error = await asyncio.wait_for(process.communicate(), timeout=10)
    except asyncio.TimeoutError:
        process.kill()
        await process.communicate()
        raise ValueError("读取 Git 变更超时")
    return process.returncode, output.decode("utf-8", errors="replace"), error.decode("utf-8", errors="replace")


async def git_changes(cwd):
    try:
        code, root, _ = await git(cwd, "rev-parse", "--show-toplevel")
    except FileNotFoundError:
        return {"available": False, "reason": "未安装 Git", "files": []}
    if code:
        return {"available": False, "reason": "当前工作目录不是 Git 仓库", "files": []}
    code, status, error = await git(cwd, "status", "--porcelain=v1", "-z", "--untracked-files=all", "--no-renames", "--", ".")
    if code:
        raise ValueError(error or "无法读取 Git 状态")
    head = (await git(cwd, "rev-parse", "--verify", "HEAD"))[0] == 0
    stats = {}
    if head:
        code, numbers, error = await git(cwd, "diff", "--no-ext-diff", "--no-textconv", "--no-renames", "--numstat", "-z", "HEAD", "--", ".")
        if code:
            raise ValueError(error or "无法读取 Git 行数")
        for row in numbers.split("\0"):
            parts = row.split("\t", 2)
            if len(parts) == 3:
                stats[parts[2]] = (int(parts[0]), int(parts[1])) if parts[0].isdigit() and parts[1].isdigit() else (None, None)
    files = []
    for row in status.split("\0"):
        if len(row) < 4:
            continue
        state, relative = row[:2], row[3:]
        path = Path(root.strip()) / relative
        if not path.resolve().is_relative_to(Path(cwd).resolve()):
            continue
        added, removed = stats.get(relative, (0, 0))
        if not head or state == "??":
            snapshot = read_snapshot(path)
            if snapshot is not None and "\0" not in snapshot:
                added, removed = len(snapshot.splitlines()), 0
            else:
                added = removed = None
        record = describe_change(path, None, None, "delete" if "D" in state else "create" if "A" in state or state == "??" else "edit")
        record.update(added=added, removed=removed, statsAvailable=added is not None, source="git", status=state.strip())
        files.append(record)
    return {"available": True, "root": root.strip(), "files": files[:500], "truncated": len(files) > 500}


async def git_file_diff(cwd, requested):
    workspace_path(cwd, requested)
    # Keep Git's lexical path: a tracked symlink and its target have different diffs.
    path = Path(os.path.abspath(Path(cwd) / requested))
    snapshot = await git_changes(cwd)
    entry = next((item for item in snapshot["files"] if os.path.normcase(os.path.abspath(item["path"])) == os.path.normcase(str(path))), None)
    if not entry:
        raise ValueError("文件当前没有 Git 工作区差异，请刷新变更列表")
    head = (await git(cwd, "rev-parse", "--verify", "HEAD"))[0] == 0
    if not head or entry["status"] == "??":
        content = read_snapshot(path)
        if content is not None and "\0" in content:
            content = None
        return {**entry, **describe_change(path, "", content, entry["operation"]), "source": "git"}
    relative = path.relative_to(Path(snapshot["root"]).resolve()).as_posix()
    code, diff, error = await git(snapshot["root"], "diff", "--no-ext-diff", "--no-textconv", "--no-color", "--no-renames", "HEAD", "--", relative)
    if code:
        raise ValueError(error or "无法读取文件差异")
    return {**entry, "diff": diff[:SNAPSHOT_LIMIT], "truncated": len(diff) > SNAPSHOT_LIMIT}
