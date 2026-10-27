"""A real competing index writer; only writes its requested presentation field."""
import json
from pathlib import Path
import sys
from codeclaw.core.session_mutation import mutate

spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print("READY", flush=True)
sys.stdin.readline()
field = spec.pop("field")
for sequence in range(30):
    mutate({**spec, "patch": {field: str(sequence)}})
print("DONE", flush=True)
