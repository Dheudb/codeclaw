"""Shared per-user history storage; working directories contain configuration and user files."""
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path


def data_directory():
    override = os.environ.get("CODECLAW_DATA_DIR")
    if override and Path(override).is_absolute():
        return Path(override)
    home = Path.home()
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        return (Path(base) if base and Path(base).is_absolute() else home / "AppData/Local") / "CodeClaw"
    if sys.platform == "darwin":
        return home / "Library/Application Support/CodeClaw"
    base = os.environ.get("XDG_DATA_HOME")
    return (Path(base) if base and Path(base).is_absolute() else home / ".local/share") / "codeclaw"


def history_root(cwd=None):
    project = os.path.realpath(cwd or os.getcwd())
    identity = project.lower() if os.name == "nt" else project
    return data_directory() / "history" / hashlib.sha256(identity.encode()).hexdigest()[:16]


def atomic_json(path, payload):
    atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2))


def atomic_text(path, text):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".migration-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(6):
            try:
                os.replace(temporary, path)
                break
            except PermissionError as error:
                # Windows readers/scanners can briefly hold a destination without delete sharing.
                if os.name != "nt" or getattr(error, "winerror", None) not in {5, 32, 33} or attempt == 5:
                    raise
                time.sleep(0.02 * (2 ** attempt))
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def rebase(value, source, target):
    if isinstance(value, str):
        converted = value.replace(str(source), str(target))
        return converted.replace(str(source).replace("\\", "/"), str(target).replace("\\", "/")) if "\\" in str(source) else converted
    if isinstance(value, list): return [rebase(item, source, target) for item in value]
    if isinstance(value, dict): return {name: rebase(item, source, target) for name, item in value.items()}
    return value


def migrate_project(cwd=None):
    project = Path(os.path.realpath(cwd or os.getcwd())); target = history_root(project); legacy = project / ".codeclaw"
    journal_path = target / "migration.json"
    journal = json.loads(journal_path.read_text(encoding="utf-8")) if journal_path.exists() else {}
    artifacts = legacy / "tool-results"
    runtime_kinds = ["browser", "shell-tasks", "vcr", "notebooks"]
    def transfer(source, destination, transform=False):
        if not source.exists(): return
        if source.is_symlink() or (hasattr(source, "is_junction") and source.is_junction()):
            raise ValueError("不能迁移链接形式的会话资源")
        if source.is_dir():
            for child in source.iterdir(): transfer(child, destination / child.name, transform)
            try: source.rmdir()
            except OSError: pass
            return
        original = source.read_bytes(); digest = hashlib.sha256(original).hexdigest(); content = original; stat = source.stat()
        if transform and source.suffix == ".json":
            try:
                value = json.loads(original)
                for name in ["tool-results", *runtime_kinds]:
                    value = rebase(value, legacy / name, target / name)
                    value = rebase(value, Path(".codeclaw") / name, target / name)
                content = json.dumps(value, ensure_ascii=False, indent=2).encode()
            except (ValueError, UnicodeError): pass
        if journal.get(str(source)) != digest:
            if destination.exists():
                try: same = json.loads(destination.read_bytes()) == json.loads(content) if transform else destination.read_bytes() == content
                except (ValueError, UnicodeError): same = destination.read_bytes() == content
                if not same: raise ValueError("旧会话与集中存储冲突，双方记录已保留：" + str(source))
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("xb") as stream: stream.write(content)
                os.utime(destination, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        journal[str(source)] = digest; atomic_json(journal_path, journal)
        if source.read_bytes() != original: raise ValueError("迁移过程中旧会话被其他程序修改，原文件保留")
        try: source.unlink()
        except PermissionError: pass
    if any((legacy / name).exists() for name in ["sessions", "session-index.json", "tool-results", *runtime_kinds]):
        if legacy.is_symlink() or (hasattr(legacy, "is_junction") and legacy.is_junction()): raise ValueError("不能迁移链接形式的历史目录")
        transfer(artifacts, target / "tool-results")
        for name in runtime_kinds: transfer(legacy / name, target / name)
        transfer(legacy / "sessions", target / "sessions", True)
        old, current = legacy / "session-index.json", target / "index.json"
        if old.exists() and current.exists():
            previous = json.loads(old.read_text(encoding="utf-8")); present = json.loads(current.read_text(encoding="utf-8"))
            if previous.get("version") != 1 or present.get("version") != 1: raise ValueError("旧会话索引格式不正确")
            atomic_json(current, {"version": 1, "sessions": {**previous.get("sessions", {}), **present.get("sessions", {})}})
            try: old.unlink()
            except PermissionError: pass
        else: transfer(old, current)
    target.mkdir(parents=True, exist_ok=True)
    manifest = target / "project.json"
    if not manifest.exists(): atomic_json(manifest, {"version": 1, "project": str(project)})
    return target


def session_directory(cwd=None):
    return migrate_project(cwd) / "sessions"


def artifact_directory(cwd=None):
    return history_root(cwd) / "tool-results"


def migrate_user_file(old, new):
    old, new = Path(old), Path(new)
    if old.is_file() and not old.is_symlink():
        if not new.exists():
            new.parent.mkdir(parents=True, exist_ok=True)
            with new.open("xb") as stream: stream.write(old.read_bytes())
        if new.read_bytes() == old.read_bytes():
            try: old.unlink()
            except PermissionError: pass
    return new
