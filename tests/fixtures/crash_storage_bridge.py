"""The production JSON-lines bridge, exiting after a real checkpoint write."""
import asyncio
import os
from pathlib import Path
import sys
from xuloom.core import session_mutation as actor
from xuloom.desktop.bridge import serve

target = Path(sys.argv[1])
original = actor.atomic_bytes


def write(path, content):
    original(path, content)
    if Path(path) == target:
        os._exit(87)


actor.atomic_bytes = write
asyncio.run(serve())
