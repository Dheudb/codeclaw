"""Archive the verified, existing portable installation; never rebuild or replace its EXE."""
import argparse
import hashlib
import json
import re
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DESKTOP = ROOT / "desktop"


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"v\d+\.\d+\.\d+(?:-[a-z0-9.]+)?", args.tag):
        raise SystemExit("Use a version tag such as v0.1.0-beta.1")
    if git("status", "--porcelain", "--untracked-files=normal"):
        raise SystemExit("Commit the release sources before creating a distributable")
    package = DESKTOP / "release/Xuloom-win32-x64"
    app = package / "resources/app"
    site = package / "resources/python/Lib/site-packages/xuloom"
    metadata = json.loads((app / "package.json").read_text(encoding="utf-8"))
    if metadata["version"] != args.tag[1:]:
        raise SystemExit("Installed application version must match the release tag")
    # Reject old resources, missing new modules, and additional private skill payloads.
    for directory, installed, patterns in [
        (ROOT / "xuloom", site, ("*.py", "*.md")),
        (DESKTOP / "dist", app / "dist", ("*",)),
        (DESKTOP / "electron", app / "electron", ("*",)),
    ]:
        source = {p.relative_to(directory) for pattern in patterns for p in directory.rglob(pattern) if p.is_file()}
        actual = {p.relative_to(installed) for pattern in patterns for p in installed.rglob(pattern) if p.is_file()}
        if source != actual or any((directory / name).read_bytes() != (installed / name).read_bytes() for name in source):
            raise SystemExit("Installed resources do not match source: " + directory.name)
    if sorted(p.parent.name for p in (site / "skills/builtin").glob("*/SKILL.md")) != ["code-development", "code-review", "debug-issue", "web-research"]:
        raise SystemExit("Unexpected bundled skills")
    # Only known application roots may be published. User state stays outside this package.
    allowed = {p.name for p in (DESKTOP / "node_modules/electron/dist").iterdir()} - {"electron.exe"}
    allowed.update({"Xuloom.exe", "START-HERE.txt", "LICENSE-Xuloom.txt"})
    if any(p.name not in allowed for p in package.iterdir()):
        raise SystemExit("Unexpected file in portable installation")
    if {p.name for p in (package / "resources").iterdir()} != {"app", "python", "default_app.asar"}:
        raise SystemExit("Unexpected resource directory")
    output = DESKTOP / "release" / f"Xuloom-{args.tag}-windows-x64.zip"
    checksum = output.with_name("SHA256SUMS.txt")
    if output.exists():
        raise SystemExit("Release archive already exists; never overwrite a published build")
    revision = git("rev-parse", "HEAD")
    record = {"tag": args.tag, "sourceCommit": revision, "platform": "windows-x64",
              "version": metadata["version"], "buildId": metadata.get("buildId"),
              "bundledSkills": ["code-development", "code-review", "debug-issue", "web-research"]}
    skipped = 0
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(package.rglob("*")):
            if path.is_symlink() or getattr(path.lstat(), "st_file_attributes", 0) & 0x400:
                raise SystemExit("Refusing a linked package path")
            if not path.is_file():
                continue
            relative = path.relative_to(package)
            # pip's generated launcher embeds the build interpreter's absolute path.
            local_launcher = relative.as_posix() == "resources/python/Lib/site-packages/bin/xuloom.exe"
            if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"} or path.name == "direct_url.json" or local_launcher:
                skipped += 1
                continue
            dependency_workspace = relative.as_posix().startswith("resources/python/Lib/site-packages/anthropic/")
            forbidden = {".git", ".xuloom", ".env", "history"}
            if not dependency_workspace:
                forbidden.add("workspaces")
            if any(part.lower() in forbidden for part in relative.parts):
                raise SystemExit("Refusing user/private data in archive")
            archive.write(path, "Xuloom-win32-x64/" + relative.as_posix())
        archive.writestr("Xuloom-win32-x64/RELEASE.json", json.dumps(record, indent=2) + "\n")
    with zipfile.ZipFile(output) as archive:
        bad = archive.testzip()
        if bad:
            raise SystemExit("Archive integrity check failed: " + bad)
    hasher = hashlib.sha256()
    with output.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    checksum.write_text(digest + "  " + output.name + "\n", encoding="utf-8")
    print(json.dumps({"archive": str(output), "bytes": output.stat().st_size, "sha256": digest,
                      "checksum": str(checksum), "sourceCommit": revision, "excludedCachesAndLocalMetadata": skipped}))


if __name__ == "__main__":
    main()
