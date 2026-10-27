"""Disposable actor process: exit at real filesystem commit boundaries."""
import json
import os
from pathlib import Path
import sys

from codeclaw.core import session_mutation as actor

spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
phase = sys.argv[2]
original_json, original_bytes, original_unlink = actor.atomic_json, actor.atomic_bytes, Path.unlink


def write_json(path, value):
    original_json(path, value)
    if phase == "prepared" and Path(path).name == spec["session_id"] + ".json":
        os._exit(87)
    if phase == "completed" and value.get("phase") == "completed":
        os._exit(87)


def write_bytes(path, value):
    original_bytes(path, value)
    path = Path(path)
    if phase == "workspace" and path.name == "capabilities.json":
        os._exit(87)
    if phase == "output" and path.name == "result.txt":
        os._exit(87)
    if phase == "checkpoint" and path.name == spec.get("new_id", spec["session_id"]) + ".json":
        os._exit(87)


def unlink(path, *args, **kwargs):
    result = original_unlink(path, *args, **kwargs)
    if phase == "source_removed" and path.name == spec["session_id"] + ".json":
        os._exit(87)
    return result


actor.atomic_json, actor.atomic_bytes, Path.unlink = write_json, write_bytes, unlink
actor.mutate(spec)
raise SystemExit("Expected crash boundary was not reached")
