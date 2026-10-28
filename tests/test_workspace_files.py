import subprocess
import tempfile
import unittest
from pathlib import Path

from xuloom.features.workspace_files import list_directory, git_changes, git_file_diff


class WorkspaceFileTests(unittest.IsolatedAsyncioTestCase):
    def git(self, cwd, *args):
        return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True).stdout

    async def test_git_staged_unstaged_deleted_untracked_and_subdirectory_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.git(root, "init")
            for name in ["modified.txt", "deleted.txt", "staged.txt", "outside.txt", "sub/中文 space.txt"]:
                path = root / name; path.parent.mkdir(exist_ok=True); path.write_text("old\nkeep\n", encoding="utf-8")
            self.git(root, "add", ".")
            self.git(root, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "baseline")
            for name in ["modified.txt", "staged.txt", "outside.txt", "sub/中文 space.txt"]:
                (root / name).write_text("new\nkeep\nextra\n", encoding="utf-8")
            (root / "deleted.txt").unlink()
            (root / "new.txt").write_text("new file\n", encoding="utf-8")
            self.git(root, "add", "staged.txt")
            records = await git_changes(root)
            self.assertTrue(records["available"])
            files = {Path(item["path"]).name: item for item in records["files"]}
            self.assertEqual((files["modified.txt"]["added"], files["modified.txt"]["removed"]), (2, 1))
            self.assertEqual(files["deleted.txt"]["removed"], 2)
            self.assertEqual(files["staged.txt"]["added"], 2)
            self.assertEqual(files["new.txt"]["added"], 1)
            for name, fragment in [("modified.txt", "+new"), ("deleted.txt", "-old"), ("staged.txt", "+extra"), ("new.txt", "+new file")]:
                self.assertIn(fragment, (await git_file_diff(root, name))["diff"])
            sub = await git_changes(root / "sub")
            self.assertEqual(len(sub["files"]), 1)
            self.assertEqual((sub["files"][0]["added"], sub["files"][0]["removed"]), (2, 1))
            self.assertIn("+new", (await git_file_diff(root / "sub", "中文 space.txt"))["diff"])
            with self.assertRaises(ValueError): await git_file_diff(root / "sub", "../outside.txt")

    async def test_unborn_repository_and_no_git_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertFalse((await git_changes(root))["available"])
            self.git(root, "init")
            (root / "first.txt").write_text("first\nsecond\n", encoding="utf-8")
            self.git(root, "add", "first.txt")
            result = await git_changes(root)
            self.assertEqual(result["files"][0]["added"], 2)
            self.assertIn("+second", (await git_file_diff(root, "first.txt"))["diff"])
            (root / "binary.bin").write_bytes(b"\0\1\2")
            binary = await git_file_diff(root, "binary.bin")
            self.assertFalse(binary["statsAvailable"])
            self.assertIsNone(binary["added"])

    def test_lazy_directory_navigation_and_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); folder = root / "workspace"; folder.mkdir()
            (folder / "nested").mkdir(); (folder / "nested/中文 space.md").write_text("hello", encoding="utf-8")
            (folder / "a.txt").write_text("a", encoding="utf-8"); (folder / ".git").mkdir()
            listing = list_directory(folder)
            self.assertEqual([entry["name"] for entry in listing["entries"]], ["nested", "a.txt"])
            self.assertEqual(list_directory(folder, "nested")["entries"][0]["name"], "中文 space.md")
            for path in ["..", "https://example.org", "\0", "a.txt"]:
                with self.assertRaises(ValueError): list_directory(folder, path)
