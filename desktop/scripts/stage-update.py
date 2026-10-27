"""Stage public application resources for the existing installation; never replace its EXE."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[2]
DESKTOP = ROOT / "desktop"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-directory", type=Path, default=DESKTOP / "release/CodeClaw-win32-x64")
    args = parser.parse_args()
    package = args.package_directory.resolve()
    executable = package / "CodeClaw.exe"
    if not executable.is_file() or not (package / "resources/python/python.exe").is_file():
        raise SystemExit("The existing installation and its bundled runtime are required")
    if not (DESKTOP / "dist/index.html").is_file():
        raise SystemExit("Run npm run build before staging resources")
    build_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    stage = DESKTOP / "build" / ("pending-update-general-" + build_id + "-" + uuid.uuid4().hex[:6])
    stage.mkdir(parents=True)
    wheels = stage / "wheels"; wheels.mkdir()
    subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-index", "--no-deps", "--no-build-isolation",
        "--wheel-dir", str(wheels), str(ROOT)], cwd=ROOT, check=True)
    wheel, = wheels.glob("codeclaw-*.whl")
    with zipfile.ZipFile(wheel) as archive:
        metadata, = {name.split("/")[0] for name in archive.namelist() if ".dist-info/" in name}
        for member in archive.infolist():
            relative = Path(member.filename)
            if member.is_dir(): continue
            if relative.is_absolute() or ".." in relative.parts:
                raise SystemExit("Invalid wheel member")
            if relative.parts[0] == "codeclaw": target = stage / relative
            elif relative.parts[0] == metadata: target = stage / "metadata" / Path(*relative.parts[1:])
            else: raise SystemExit("Unexpected distribution root: " + relative.parts[0])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(member))
    source = ROOT / "codeclaw"
    for pattern in ("*.py", "*.md"):
        expected = {path.relative_to(source) for path in source.rglob(pattern)}
        actual = {path.relative_to(stage / "codeclaw") for path in (stage / "codeclaw").rglob(pattern)}
        if expected != actual or any((source / path).read_bytes() != (stage / "codeclaw" / path).read_bytes() for path in expected):
            raise SystemExit("Wheel resources do not match public source")
    skills = sorted(path.parent.name for path in (stage / "codeclaw/skills/builtin").glob("*/SKILL.md"))
    if skills != ["code-development", "code-review", "debug-issue", "web-research"]:
        raise SystemExit("Unexpected bundled skills")
    app = stage / "app"; app.mkdir()
    for directory in ("dist", "electron"):
        shutil.copytree(DESKTOP / directory, app / directory)
    version = json.loads((DESKTOP / "package.json").read_text(encoding="utf-8"))["version"]
    (app / "package.json").write_text(json.dumps({"name": "codeclaw-desktop", "productName": "CodeClaw",
        "version": version, "main": "electron/main.cjs", "buildId": build_id}, indent=2), encoding="utf-8")
    subprocess.run([str(package / "resources/python/python.exe"), "-I", "-B", "-c",
        "import sys;sys.path.insert(0,sys.argv[1]);import codeclaw.core.session_mutation,codeclaw.core.engine;from codeclaw.protocols.mcp_bridge import MCP_AVAILABLE;assert MCP_AVAILABLE;print('Staged engine and bundled dependencies verified')",
        str(stage)], cwd=stage, check=True)
    manifest = {"metadataDirectory": metadata, "buildId": build_id,
        "originalExeSha256": hashlib.sha256(executable.read_bytes()).hexdigest(), "bundledSkills": skills}
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if list(stage.rglob("*.exe")): raise SystemExit("Resource updates must not contain another executable")
    print(json.dumps({"stage": str(stage), "target": str(package), **manifest}))


if __name__ == "__main__":
    main()
