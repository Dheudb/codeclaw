"""Fail on undefined Python names without treating unused imports as runtime faults."""
import ast
from pathlib import Path
from pyflakes.checker import Checker
from pyflakes.messages import UndefinedName, UndefinedLocal

root = Path(__file__).resolve().parents[1]
failures = []
paths = sorted((root / "xuloom").rglob("*.py"))
for path in paths:
    source = path.read_text(encoding="utf-8-sig")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as error:
        failures.append(f"{path.relative_to(root)}:{error.lineno}: {error.msg}")
        continue
    checker = Checker(tree, filename=str(path.relative_to(root)))
    failures.extend(str(message) for message in checker.messages if isinstance(message, (UndefinedName, UndefinedLocal)))
if failures:
    print("\n".join(failures))
    raise SystemExit(1)
print(f"Undefined-name and syntax checks passed: {len(paths)} Python modules")
