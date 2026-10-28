"""Hold a real presentation-index lease without changing any conversation."""
from pathlib import Path
import sys
import time
from xuloom.core.session_lock import SessionLock, SessionBusyError
from xuloom.core.storage import data_directory, history_root

lock = SessionLock(data_directory() / "session-locks", "index_" + history_root(Path(sys.argv[1])).name)
for attempt in range(201):
    try:
        lock.acquire()
        break
    except SessionBusyError:
        if attempt == 200: raise
        time.sleep(0.01)
print("OWNED", flush=True)
try:
    sys.stdin.readline()
finally:
    lock.close()
