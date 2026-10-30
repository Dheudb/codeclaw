"""Durable file-tool revisions, with compare-before-restore and no model instructions."""
import hashlib
import json
import os
import re
import time
import uuid
from pathlib import Path
from xuloom.core.storage import artifact_directory, atomic_copy, atomic_json


def revision_root(context):
    owner = context.get("delegation", {}).get("owner_session_id") or context.get("session_id")
    if not owner:
        return None
    if not isinstance(owner, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,160}", owner):
        raise ValueError("Invalid revision session")
    return artifact_directory(context.get("session_cwd")) / owner / "file-revisions"


def digest_file(path):
    try:
        with Path(path).open("rb") as stream:
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(65536), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except FileNotFoundError:
        return None


def prepare_revision(context, path, content):
    if context.get("revision_tracking") is False:
        return None  # A confined worker cannot write host history; it never falls back to host writes.
    root = revision_root(context)
    if root is None:
        return None
    # Never create recursive snapshots of the harness's own history.
    from xuloom.core.storage import data_directory
    path = Path(os.path.realpath(path))
    if path.is_relative_to((data_directory() / "history").resolve()):
        raise ValueError("File tools cannot overwrite execution history")
    identity = uuid.uuid4().hex
    folder = root / identity
    before = digest_file(path)
    after = hashlib.sha256(content).hexdigest()
    if before is not None:
        atomic_copy(folder / "before.bin", path, before)
    record = {"id": identity, "path": str(path), "before": before, "after": after,
        "createdAt": int(time.time() * 1000), "agentId": context.get("agent_id", "")}
    # Intent and original bytes are durable before the workspace is mutated.
    atomic_json(folder / "revision.json", record)
    return identity


async def restore_revision(context, identity):
    if not isinstance(identity, str) or not re.fullmatch(r"[a-f0-9]{32}", identity):
        raise ValueError("Invalid file revision")
    root = revision_root(context)
    if root is None:
        raise ValueError("No revision storage for this session")
    folder = root / identity
    for item in (root, folder, folder / "revision.json", folder / "before.bin"):
        if item.is_symlink():
            raise ValueError("Linked revision storage is not supported")
    record = json.loads((folder / "revision.json").read_text(encoding="utf-8"))
    if record.get("id") != identity or not isinstance(record.get("path"), str):
        raise ValueError("Invalid file revision metadata")
    path = Path(record["path"])
    if not path.is_absolute() or str(path) != os.path.realpath(path):
        raise ValueError("File path changed or now follows a link; undo was refused")
    def restore():
        current = digest_file(path)
        if current == record["before"]:
            return {"restored": True, "alreadyRestored": True, "path": str(path), "revisionId": identity}
        if current != record["after"]:
            raise ValueError("文件在这次写入后又被修改，拒绝覆盖；请先查看当前差异。")
        if record["before"] is None:
            path.unlink()
        else:
            # atomic_copy verifies the original snapshot hash before publication.
            atomic_copy(path, folder / "before.bin", record["before"], preserve_mode=True)
        cache = context.get("file_state_cache")
        if cache:
            cache.invalidate(str(path))
        reads = context.get("read_file_state", {})
        for key in list(reads):
            if os.path.normcase(os.path.realpath(key)) == os.path.normcase(str(path)):
                reads.pop(key, None)
        return {"restored": True, "alreadyRestored": False, "path": str(path), "revisionId": identity}
    queue = context.get("incremental_write_queue")
    if queue is None:
        from xuloom.features.incremental_write_queue import IncrementalWriteQueue
        queue = IncrementalWriteQueue()
    return await queue.run(str(path), "undo", restore)
