import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from codeclaw.core import session_mutation as actor
from codeclaw.core.session import SessionManager
from codeclaw.core.session_lock import SessionBusyError, SessionConflictError
from codeclaw.core.storage import atomic_json, history_root


class SessionMutationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.env = {**os.environ, "CODECLAW_DATA_DIR": str(self.base / "data"),
            "PYTHONPATH": str(Path(__file__).resolve().parents[1])}
        environment = patch.dict(os.environ, self.env)
        environment.start(); self.addCleanup(environment.stop)
        self.managers = []
        self.addCleanup(lambda: [manager.release_writer() for manager in self.managers])
        self.source, self.target = self.base / "source", self.base / "target"
        self.source.mkdir(); self.target.mkdir()
        self.origin, self.destination = history_root(self.source), history_root(self.target)
        self.checkpoint = self.origin / "sessions/shared.json"
        self.artifact = self.origin / "tool-results/shared/result.txt"
        self.child = self.origin / "sessions/subagents/owned.json"
        self.spec = {"action": "move", "project": str(self.source), "target": str(self.target),
            "session_id": "shared", "scope": "project", "title": "keep title", "created": 123}
        self.seed()

    def seed(self):
        self.artifact.parent.mkdir(parents=True, exist_ok=True)
        self.artifact.write_text("complete output", encoding="utf-8")
        atomic_json(self.checkpoint, {"revision": 1, "messages": [{"role": "user", "content": str(self.artifact)}],
            "metadata": {"permissions": {"full": True}, "subagents": [{"agent_id": "owned"}]}})
        atomic_json(self.child, {"session_id": "shared", "messages": [{"role": "assistant", "content": str(self.artifact)}]})
        atomic_json(self.origin / "index.json", {"version": 1, "sessions": {"shared": {"title": "keep title", "draft": "keep draft"}, "other": {"title": "unrelated"}}})
        (self.source / "user.txt").write_text("never move", encoding="utf-8")

    def manager(self, project):
        manager = SessionManager(cwd=project)
        self.managers.append(manager)
        return manager

    def assert_moved(self):
        self.assertFalse(self.checkpoint.exists())
        self.assertFalse(self.artifact.exists())
        self.assertFalse(self.child.exists())
        raw = json.loads((self.destination / "sessions/shared.json").read_bytes())
        self.assertEqual(raw["revision"], 2)
        self.assertNotIn("permissions", raw["metadata"])
        expected = self.destination / "tool-results/shared/result.txt"
        self.assertEqual(os.path.normcase(raw["messages"][0]["content"]), os.path.normcase(str(expected)))
        self.assertEqual(expected.read_text(), "complete output")
        self.assertTrue((self.destination / "sessions/subagents/owned.json").exists())
        self.assertEqual(json.loads((self.destination / "index.json").read_bytes())["sessions"]["shared"]["draft"], "keep draft")
        self.assertEqual(json.loads((self.origin / "index.json").read_bytes())["sessions"], {"other": {"title": "unrelated"}})
        self.assertEqual((self.source / "user.txt").read_text(), "never move")
        self.assertEqual(list((self.base / "data/session-operations").glob("*.json")), [])

    def test_busy_actor_rejected_then_move_fork_delete_preserve_owned_results(self):
        owner = self.manager(self.source); owner.acquire_writer("shared")
        with self.assertRaises(SessionBusyError):
            actor.mutate(self.spec)
        self.assertTrue(self.checkpoint.exists())
        self.assertFalse((self.destination / "sessions/shared.json").exists())
        owner.release_writer()
        self.assertTrue(actor.mutate(self.spec)["completed"])
        self.assert_moved()
        fork = {**self.spec, "action": "fork", "project": str(self.target), "new_id": "copy"}
        actor.mutate(fork)
        raw = json.loads((self.destination / "sessions/copy.json").read_bytes())
        self.assertEqual(raw["metadata"], {})
        self.assertIn("copy", raw["messages"][0]["content"])
        self.assertEqual((self.destination / "tool-results/copy/result.txt").read_text(), "complete output")
        actor.mutate({"action": "delete", "sessions": [{"project": str(self.target), "session_id": "shared"}]})
        self.assertFalse((self.destination / "sessions/shared.json").exists())
        self.assertFalse((self.destination / "sessions/subagents/owned.json").exists())
        self.assertTrue((self.destination / "sessions/copy.json").exists())

    def test_partial_copy_recovers_before_another_writer_proceeds(self):
        original = actor.atomic_bytes
        def fail(path, content):
            original(path, content)
            if Path(path).name == "result.txt":
                raise OSError("injected crash after output commit")
        with patch.object(actor, "atomic_bytes", side_effect=fail):
            with self.assertRaisesRegex(OSError, "injected crash"):
                actor.mutate(self.spec)
        self.assertTrue(self.checkpoint.exists())
        self.assertFalse((self.destination / "sessions/shared.json").exists())
        self.manager(self.target).acquire_writer("shared")
        self.assert_moved()

    def legacy_workspace(self):
        base = self.base / "workspaces"; base.mkdir(exist_ok=True)
        target = base / "chat-legacy"
        raw = json.loads(self.checkpoint.read_bytes())
        raw["messages"].append({"role": "assistant", "content": [
            {"type": "tool_use", "name": "file_write_tool", "input": {"file_path": name}}
            for name in ("nested/result.bin", "user.txt", ".hidden", "../external.txt")]})
        atomic_json(self.checkpoint, raw)
        atomic_json(self.origin / "index.json", {"version": 1, "sessions": {"shared": {"scope": "standalone", "draft": "keep draft"}}})
        binary = self.source / "nested/result.bin"; binary.parent.mkdir(exist_ok=True)
        with binary.open("wb") as stream:
            for _ in range(16): stream.write(bytes(range(256)) * 4096)
        (self.source / ".hidden").write_bytes(b"private")
        (self.base / "external.txt").write_bytes(b"outside")
        skill = self.source / ".codeclaw/skills/private/SKILL.md"
        skill.parent.mkdir(parents=True, exist_ok=True); skill.write_text("private skill", encoding="utf-8")
        atomic_json(self.source / "mcp_servers.json", {"mcpServers": {"fixture": {"command": "fixture"}}})
        return target, {**self.spec, "target": str(target), "scope": "standalone", "legacy_workspace": {"base": str(base), "id": "legacy"}}, binary

    def test_legacy_upgrade_preflights_writer_then_streams_binary_without_copying_private_skills(self):
        target, spec, binary = self.legacy_workspace()
        owner = self.manager(self.source); owner.acquire_writer("shared")
        with self.assertRaises(SessionBusyError): actor.mutate(spec)
        self.assertFalse(target.exists()); self.assertTrue(self.checkpoint.exists())
        owner.release_writer()
        original = Path.read_bytes
        def bounded_read(path):
            if path == binary: raise AssertionError("Large workspace outputs must be streamed")
            return original(path)
        with patch.object(Path, "read_bytes", bounded_read): actor.mutate(spec)
        self.assertEqual(actor.file_hash(binary), actor.file_hash(target / "nested/result.bin"))
        self.assertTrue(binary.exists()); self.assertEqual((self.source / "user.txt").read_text(), "never move")
        self.assertFalse((target / ".hidden").exists()); self.assertFalse((target / "external.txt").exists())
        self.assertFalse((target / ".codeclaw/skills").exists())
        config = json.loads((target / ".codeclaw/capabilities.json").read_bytes())
        self.assertIn(os.path.normcase(str(self.source / ".codeclaw/skills")), [os.path.normcase(item) for item in config["skill_roots"]])
        self.assertEqual((target / "mcp_servers.json").read_bytes(), (self.source / "mcp_servers.json").read_bytes())

    def test_legacy_upgrade_conflict_preserves_user_edit_and_missing_source_is_not_success(self):
        target, spec, binary = self.legacy_workspace()
        original = actor.atomic_copy
        def fail(path, source, digest):
            original(path, source, digest)
            raise OSError("interrupt after output")
        with patch.object(actor, "atomic_copy", side_effect=fail):
            with self.assertRaises(OSError): actor.mutate(spec)
        output = target / "nested/result.bin"; output.write_bytes(b"user edit")
        with self.assertRaises(SessionConflictError): actor.mutate({"action": "recover"})
        self.assertEqual(output.read_bytes(), b"user edit"); self.assertTrue(self.checkpoint.exists())
        output.unlink(); binary.unlink()
        with self.assertRaises(FileNotFoundError): actor.mutate({"action": "recover"})
        self.assertTrue(self.checkpoint.exists())
        self.assertTrue(list((self.base / "data/session-operations").glob("transaction-*.json")))

    def test_legacy_upgrade_process_crashes_recover_at_startup_without_repeating_user_operations(self):
        import shutil
        from codeclaw.core.session_lock import SessionLock
        target, spec, binary = self.legacy_workspace()
        digest = actor.file_hash(binary)
        fixture = Path(__file__).with_name("fixtures") / "crash_session_mutation.py"
        request = self.base / "legacy.json"; atomic_json(request, spec)
        for phase in ("intent", "prepared", "workspace", "workspace_output", "checkpoint", "source_removed", "completed"):
            with self.subTest(phase=phase):
                process = subprocess.run([sys.executable, str(fixture), str(request), phase], env=self.env,
                    capture_output=True, text=True, timeout=30)
                self.assertEqual(process.returncode, 87, process.stdout + process.stderr)
                lock = SessionLock(self.base / "data/session-locks", "shared"); lock.acquire()
                try:
                    busy = actor.mutate({"action": "recover"})
                    self.assertFalse(busy["completed"]); self.assertEqual(len(busy["busy"]), 1)
                finally: lock.close()
                self.assertTrue(actor.mutate({"action": "recover"})["completed"])
                if phase in {"intent", "prepared"}:
                    self.assertFalse(target.exists()); self.assertTrue(self.checkpoint.exists())
                else:
                    self.assertFalse(self.checkpoint.exists())
                    self.assertEqual(actor.file_hash(target / "nested/result.bin"), digest)
                    self.assertEqual((target / "user.txt").read_text(), "never move")
                    self.assertTrue((history_root(target) / "sessions/shared.json").is_file())
                self.assertEqual(actor.file_hash(binary), digest)
                self.assertEqual(list((self.base / "data/session-operations").glob("*.json")), [])
                destination = history_root(target)
                if destination.exists(): shutil.rmtree(destination)
                if target.exists():
                    self.assertEqual(target.resolve().parent, (self.base / "workspaces").resolve())
                    shutil.rmtree(target)
                self.seed(); self.legacy_workspace()

    def test_real_process_crash_at_each_move_commit_boundary(self):
        fixture = Path(__file__).with_name("fixtures") / "crash_session_mutation.py"
        request = self.base / "request.json"; atomic_json(request, self.spec)
        for phase in ("prepared", "output", "checkpoint", "source_removed", "completed"):
            with self.subTest(phase=phase):
                process = subprocess.run([sys.executable, str(fixture), str(request), phase], env=self.env,
                    capture_output=True, text=True, timeout=30)
                self.assertEqual(process.returncode, 87, process.stdout + process.stderr)
                manager = self.manager(self.target)
                manager.acquire_writer("shared")
                if phase == "prepared":
                    self.assertTrue(self.checkpoint.exists())
                    self.assertFalse((self.destination / "sessions/shared.json").exists())
                else:
                    self.assert_moved()
                manager.release_writer()
                # Reset only this disposable fixture, outside production recovery.
                import shutil
                if self.destination.exists(): shutil.rmtree(self.destination)
                self.seed()

    def test_changed_destination_is_preserved_and_blocks_recovery(self):
        original = actor.atomic_bytes
        def fail(path, content):
            original(path, content)
            if Path(path).name == "result.txt": raise OSError("interrupt")
        with patch.object(actor, "atomic_bytes", side_effect=fail):
            with self.assertRaises(OSError): actor.mutate(self.spec)
        target = self.destination / "tool-results/shared/result.txt"
        target.write_text("external edit", encoding="utf-8")
        with self.assertRaises(SessionConflictError):
            self.manager(self.target).acquire_writer("shared")
        self.assertEqual(target.read_text(), "external edit")
        self.assertTrue(self.checkpoint.exists())

    def test_real_process_crash_during_fork_never_loses_source_or_duplicates_results(self):
        import shutil
        fixture = Path(__file__).with_name("fixtures") / "crash_session_mutation.py"
        spec = {**self.spec, "action": "fork", "new_id": "copy"}
        request = self.base / "fork.json"; atomic_json(request, spec)
        for phase in ("prepared", "output", "checkpoint", "completed"):
            with self.subTest(phase=phase):
                process = subprocess.run([sys.executable, str(fixture), str(request), phase], env=self.env,
                    capture_output=True, text=True, timeout=30)
                self.assertEqual(process.returncode, 87, process.stdout + process.stderr)
                manager = self.manager(self.target); manager.acquire_writer("copy")
                self.assertTrue(self.checkpoint.exists()); self.assertTrue(self.artifact.exists())
                destination = self.destination / "sessions/copy.json"
                if phase == "prepared":
                    self.assertFalse(destination.exists())
                else:
                    raw = json.loads(destination.read_bytes())
                    self.assertEqual(raw["metadata"], {}); self.assertEqual(raw["revision"], 1)
                    self.assertEqual((self.destination / "tool-results/copy/result.txt").read_text(), "complete output")
                manager.release_writer()
                if self.destination.exists(): shutil.rmtree(self.destination)

    def test_real_process_crash_during_delete_recovers_without_recreating_history(self):
        fixture = Path(__file__).with_name("fixtures") / "crash_session_mutation.py"
        spec = {"action": "delete", "session_id": "shared", "sessions": [{"project": str(self.source), "session_id": "shared"}]}
        request = self.base / "delete.json"; atomic_json(request, spec)
        for phase in ("prepared", "source_removed", "completed"):
            with self.subTest(phase=phase):
                process = subprocess.run([sys.executable, str(fixture), str(request), phase], env=self.env,
                    capture_output=True, text=True, timeout=30)
                self.assertEqual(process.returncode, 87, process.stdout + process.stderr)
                manager = self.manager(self.source); manager.acquire_writer("shared")
                if phase == "prepared":
                    self.assertTrue(self.checkpoint.exists()); self.assertTrue(self.artifact.exists())
                else:
                    self.assertFalse(self.checkpoint.exists()); self.assertFalse(self.artifact.exists()); self.assertFalse(self.child.exists())
                    self.assertEqual(json.loads((self.origin / "index.json").read_bytes())["sessions"], {"other": {"title": "unrelated"}})
                self.assertEqual((self.source / "user.txt").read_text(), "never move")
                manager.release_writer()
                self.seed()

    def test_locked_attachment_keeps_checkpoint_and_metadata_then_delete_recovers(self):
        original = Path.unlink
        def denied(path, *args, **kwargs):
            if path == self.artifact: raise PermissionError("injected locked file")
            return original(path, *args, **kwargs)
        with patch.object(Path, "unlink", denied):
            with self.assertRaisesRegex(PermissionError, "会话附件"):
                actor.mutate({"action": "delete", "sessions": [{"project": str(self.source), "session_id": "shared"}]})
        self.assertTrue(self.checkpoint.exists())
        self.assertEqual(json.loads((self.origin / "index.json").read_bytes())["sessions"]["shared"]["title"], "keep title")
        self.manager(self.source).acquire_writer("shared")
        self.assertFalse(self.checkpoint.exists())
        self.assertFalse(self.artifact.exists())
        self.assertEqual((self.source / "user.txt").read_text(), "never move")

    def test_batch_delete_preflights_all_writers_before_any_removal(self):
        atomic_json(self.origin / "sessions/busy.json", {"messages": []})
        manager = self.manager(self.source); manager.acquire_writer("busy")
        rows = [{"project": str(self.source), "session_id": sid} for sid in ("shared", "busy")]
        with self.assertRaises(SessionBusyError): actor.mutate({"action": "delete", "sessions": rows})
        self.assertTrue(self.checkpoint.exists()); self.assertTrue(self.artifact.exists())
        manager.release_writer()
        actor.mutate({"action": "delete", "sessions": rows})
        self.assertFalse(self.checkpoint.exists())
        self.assertFalse((self.origin / "sessions/busy.json").exists())

    def test_cli_manager_delete_uses_same_complete_storage_commit(self):
        manager = self.manager(self.source)
        self.assertTrue(manager.delete_session("shared"))
        self.assertFalse(self.checkpoint.exists())
        self.assertFalse(self.artifact.exists())
        self.assertFalse(self.child.exists())
        self.assertEqual(json.loads((self.origin / "index.json").read_bytes())["sessions"], {"other": {"title": "unrelated"}})
        self.assertEqual((self.source / "user.txt").read_text(), "never move")

    def test_metadata_does_not_take_engine_writer_or_replace_other_fields(self):
        manager = self.manager(self.source); manager.acquire_writer("shared")
        before = self.checkpoint.read_bytes()
        actor.mutate({"action": "metadata", "project": str(self.source), "session_id": "shared", "patch": {"pinned": True}})
        actor.mutate({"action": "metadata", "project": str(self.source), "session_id": "shared", "patch": {"draft": "new draft"}})
        row = json.loads((self.origin / "index.json").read_bytes())["sessions"]["shared"]
        self.assertEqual(row, {"title": "keep title", "draft": "new draft", "pinned": True})
        self.assertEqual(self.checkpoint.read_bytes(), before)

    def test_late_metadata_cannot_recreate_moved_or_deleted_history(self):
        actor.mutate(self.spec)
        with self.assertRaises(SessionConflictError):
            actor.mutate({"action": "metadata", "project": str(self.source), "session_id": "shared", "patch": {"draft": "late draft"}})
        self.assert_moved()
        actor.mutate({"action": "delete", "sessions": [{"project": str(self.target), "session_id": "shared"}]})
        with self.assertRaises(SessionConflictError):
            actor.mutate({"action": "metadata", "project": str(self.target), "session_id": "shared", "patch": {"draft": "late draft"}})
        self.assertFalse((self.destination / "sessions/shared.json").exists())
        self.assertNotIn("shared", json.loads((self.destination / "index.json").read_bytes())["sessions"])

    def test_metadata_recovers_pending_move_before_merging_new_fields(self):
        original = actor.atomic_bytes
        def fail(path, content):
            original(path, content)
            if Path(path).name == "result.txt": raise OSError("interrupted move")
        with patch.object(actor, "atomic_bytes", side_effect=fail):
            with self.assertRaises(OSError): actor.mutate(self.spec)
        actor.mutate({"action": "metadata", "project": str(self.target), "session_id": "shared", "patch": {"title": "new title"}})
        row = json.loads((self.destination / "index.json").read_bytes())["sessions"]["shared"]
        self.assertEqual(row["title"], "new title"); self.assertEqual(row["draft"], "keep draft")
        self.assertFalse(self.checkpoint.exists())

    def test_real_concurrent_metadata_processes_merge_without_lost_updates(self):
        fixture = Path(__file__).with_name("fixtures") / "metadata_writer.py"
        processes = []
        try:
            for field in ("title", "draft"):
                request = self.base / (field + ".json")
                atomic_json(request, {"action": "metadata", "project": str(self.source), "session_id": "shared", "field": field})
                process = subprocess.Popen([sys.executable, str(fixture), str(request)], env=self.env,
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                processes.append(process)
                self.assertEqual(process.stdout.readline().strip(), "READY")
            for process in processes:
                process.stdin.write("GO\n"); process.stdin.flush()
            for process in processes:
                stdout, stderr = process.communicate(timeout=30)
                self.assertEqual(process.returncode, 0, stderr); self.assertIn("DONE", stdout)
        finally:
            for process in processes:
                if process.poll() is None: process.communicate("GO\n", timeout=30)
        row = json.loads((self.origin / "index.json").read_bytes())["sessions"]["shared"]
        self.assertEqual(row["title"], "29"); self.assertEqual(row["draft"], "29")
        self.assertEqual(json.loads((self.origin / "index.json").read_bytes())["sessions"]["other"], {"title": "unrelated"})

    def test_create_never_overwrites_an_existing_checkpoint(self):
        spec = {"action": "create", "project": str(self.source), "session_id": "fresh", "created": 123, "scope": "standalone", "messages": []}
        actor.mutate(spec)
        path = self.origin / "sessions/fresh.json"; before = path.read_bytes()
        with self.assertRaises(SessionConflictError): actor.mutate({**spec, "messages": [{"role": "user", "content": "overwrite"}]})
        self.assertEqual(path.read_bytes(), before)
        row = json.loads((self.origin / "index.json").read_bytes())["sessions"]["fresh"]
        self.assertTrue(row["workspaceIsolated"]); self.assertEqual(row["scope"], "standalone")

    def test_real_process_create_crash_recovers_checkpoint_and_index_together(self):
        fixture = Path(__file__).with_name("fixtures") / "crash_session_mutation.py"
        spec = {"action": "create", "project": str(self.target), "session_id": "fresh", "created": 123, "scope": "project", "messages": []}
        request = self.base / "create.json"; atomic_json(request, spec)
        for phase in ("prepared", "checkpoint", "completed"):
            with self.subTest(phase=phase):
                process = subprocess.run([sys.executable, str(fixture), str(request), phase], env=self.env,
                    capture_output=True, text=True, timeout=30)
                self.assertEqual(process.returncode, 87, process.stdout + process.stderr)
                manager = self.manager(self.target); manager.acquire_writer("fresh")
                checkpoint = self.destination / "sessions/fresh.json"
                if phase == "prepared":
                    self.assertFalse(checkpoint.exists())
                else:
                    self.assertEqual(json.loads(checkpoint.read_bytes())["revision"], 1)
                    self.assertEqual(json.loads((self.destination / "index.json").read_bytes())["sessions"]["fresh"]["created"], 123)
                manager.release_writer()
                if checkpoint.exists():
                    actor.mutate({"action": "delete", "sessions": [{"project": str(self.target), "session_id": "fresh"}]})

    def test_real_bridge_exit_before_reply_can_recover_the_same_move_without_replaying_it(self):
        fixture = Path(__file__).with_name("fixtures") / "crash_storage_bridge.py"
        request = json.dumps({"id": 1, "method": "storage.mutate", "params": self.spec}) + "\n"
        env = {**self.env, "CODECLAW_CONFIG_DIR": str(self.base / "config"), "HOME": str(self.base / "home"), "USERPROFILE": str(self.base / "home")}
        target = self.destination / "sessions/shared.json"
        process = subprocess.run([sys.executable, str(fixture), str(target)], input=request,
            capture_output=True, text=True, env=env, cwd=self.source, timeout=30)
        self.assertEqual(process.returncode, 87, process.stdout + process.stderr)
        self.assertNotIn('"result"', process.stdout)
        self.assertTrue(self.checkpoint.exists()); self.assertTrue(target.exists())
        # Retry through a fresh production bridge. Its reply describes the recovered commit.
        retry = subprocess.run([sys.executable, "-m", "codeclaw.desktop.bridge"], input=request,
            capture_output=True, text=True, env=env, cwd=self.source, timeout=30)
        self.assertEqual(retry.returncode, 0, retry.stderr)
        replies = [json.loads(line) for line in retry.stdout.splitlines() if line.startswith("{")]
        reply = next(row for row in replies if row.get("id") == 1)
        self.assertTrue(reply["result"]["completed"]); self.assertTrue(reply["result"]["recovered"])
        self.assert_moved()

    def test_private_workspace_and_checkpoint_recover_as_one_commit(self):
        import shutil
        fixture = Path(__file__).with_name("fixtures") / "crash_session_mutation.py"
        base = self.base / "workspaces"; base.mkdir()
        skill = base / ".codeclaw/skills/example/SKILL.md"
        skill.parent.mkdir(parents=True); skill.write_text("private fixture skill", encoding="utf-8")
        atomic_json(base / "mcp_servers.json", {"mcpServers": {"fixture": {"command": "fixture"}}})
        target = base / "chat-private"
        spec = {"action": "create", "project": str(target), "session_id": "private", "scope": "standalone", "workspace_base": str(base), "created": 123}
        request = self.base / "independent.json"; atomic_json(request, spec)
        for phase in ("prepared", "workspace", "checkpoint", "completed"):
            with self.subTest(phase=phase):
                process = subprocess.run([sys.executable, str(fixture), str(request), phase], env=self.env,
                    capture_output=True, text=True, timeout=30)
                self.assertEqual(process.returncode, 87, process.stdout + process.stderr)
                if phase == "prepared": self.assertFalse(target.exists())
                elif target.exists(): (target / "user.txt").write_text("preserve during recovery", encoding="utf-8")
                manager = self.manager(target); manager.acquire_writer("private")
                root = history_root(target)
                if phase == "prepared":
                    self.assertFalse(target.exists()); self.assertFalse((root / "sessions/private.json").exists())
                else:
                    self.assertEqual((target / "user.txt").read_text(), "preserve during recovery")
                    self.assertEqual((target / "mcp_servers.json").read_bytes(), (base / "mcp_servers.json").read_bytes())
                    config = json.loads((target / ".codeclaw/capabilities.json").read_bytes())
                    self.assertIn(os.path.normcase(str(skill.parent.parent)), [os.path.normcase(item) for item in config["skill_roots"]])
                    self.assertFalse((target / ".codeclaw/skills").exists())
                    self.assertEqual(json.loads((root / "sessions/private.json").read_bytes())["revision"], 1)
                manager.release_writer()
                if (root / "sessions/private.json").exists():
                    actor.mutate({"action": "delete", "sessions": [{"project": str(target), "session_id": "private"}]})
                if target.exists():
                    self.assertEqual(target.resolve().parent, base.resolve())
                    shutil.rmtree(target)

    def test_actor_json_protocol_uses_utf8_even_with_legacy_windows_code_page(self):
        project = self.base / "中文目录 😀"; project.mkdir()
        spec = {"action": "create", "project": str(project), "session_id": "unicode", "scope": "project", "created": 123,
            "messages": [{"role": "user", "content": "你好，α 😀"}]}
        process = subprocess.run([sys.executable, "-m", "codeclaw.core.session_mutation"],
            input=json.dumps(spec, ensure_ascii=False).encode("utf-8"), capture_output=True,
            env={**self.env, "PYTHONIOENCODING": "gbk"}, timeout=30)
        self.assertEqual(process.returncode, 0, process.stdout.decode("utf-8") + process.stderr.decode("utf-8"))
        self.assertTrue(json.loads(process.stdout)["completed"])
        raw = json.loads((history_root(project) / "sessions/unicode.json").read_bytes())
        self.assertEqual(raw["messages"], spec["messages"])
