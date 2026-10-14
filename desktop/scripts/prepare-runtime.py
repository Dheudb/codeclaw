"""Build a relocatable Windows CPython runtime with CodeClaw and MCP installed."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "desktop" / "build"
RUNTIME = BUILD / "python"


def reset_generated(path):
    # Only this script's fixed generated directory may be recursively removed.
    if path.parent.resolve() != BUILD.resolve() or path.name not in {"python", "python-staging"} or path.is_symlink():
        raise RuntimeError("Refusing to replace a path outside desktop/build")
    if path.exists():
        shutil.rmtree(path)


def verify(runtime):
    with tempfile.TemporaryDirectory(prefix="codeclaw-runtime-check-") as directory:
        subprocess.run([str(runtime / "python.exe"), "-I", "-c", "import sys,codeclaw.core.engine; from codeclaw.protocols.mcp_bridge import MCP_AVAILABLE; assert MCP_AVAILABLE; assert sys.version_info >= (3,12); print('Bundled Python and MCP verified')"], cwd=directory, check=True)


def main():
    if sys.platform != "win32" or sys.version_info < (3, 12):
        raise SystemExit("Desktop packaging requires Windows and a build-time Python 3.12+ interpreter.")
    base = Path(sys.base_prefix)
    if not (base / "python.exe").is_file() or not (base / "Lib" / "os.py").is_file():
        raise SystemExit("Use a standard Windows CPython installation to build the bundled runtime.")
    inputs = [ROOT / "setup.py", ROOT / "pyproject.toml"] + sorted((ROOT / "codeclaw").rglob("*.py"))
    inputs += sorted((ROOT / "codeclaw" / "skills" / "builtin").rglob("SKILL.md"))
    fingerprint = hashlib.sha256(sys.version.encode() + b"".join(path.relative_to(ROOT).as_posix().encode() + path.read_bytes() for path in inputs)).hexdigest()
    manifest = RUNTIME / "codeclaw-runtime.json"
    if manifest.exists() and json.loads(manifest.read_text())["fingerprint"] == fingerprint:
        verify(RUNTIME)
        return
    BUILD.mkdir(parents=True, exist_ok=True)
    stage = BUILD / "python-staging"
    reset_generated(stage)
    stage.mkdir()
    for path in base.iterdir():
        if path.is_file() and (path.suffix.lower() == ".dll" or path.name in {"python.exe", "LICENSE.txt"}):
            shutil.copy2(path, stage / path.name)
    shutil.copytree(base / "DLLs", stage / "DLLs", ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "_tkinter.pyd", "tcl*.dll", "tk*.dll"))
    shutil.copytree(base / "Lib", stage / "Lib", ignore=shutil.ignore_patterns("site-packages", "__pycache__", "*.pyc", "test", "tests", "idlelib", "tkinter", "turtledemo"))
    target = stage / "Lib" / "site-packages"
    subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--no-compile", "--target", str(target), str(ROOT) + "[mcp]"], cwd=ROOT, check=True)
    # A normal wheel installation, rather than a copied virtual environment,
    # keeps project and interpreter paths out of the distributed runtime.
    verify(stage)
    installed = []
    for metadata in sorted(target.glob("*.dist-info/METADATA")):
        fields = metadata.read_text(encoding="utf-8", errors="replace").splitlines()
        installed.append({key: next((line.split(": ", 1)[1] for line in fields if line.startswith(key + ": ")), "") for key in ("Name", "Version")})
    (stage / "codeclaw-runtime.json").write_text(json.dumps({"python": sys.version.split()[0], "fingerprint": fingerprint, "packages": installed}, indent=2), encoding="utf-8")
    reset_generated(RUNTIME)
    stage.rename(RUNTIME)
    print("Prepared bundled runtime: " + str(RUNTIME))


if __name__ == "__main__":
    main()
