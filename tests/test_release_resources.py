"""A release can read the existing runtime while overlaying a staged update."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("release_resources", Path(__file__).resolve().parents[1] / "desktop/scripts/release.py")
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


class ReleaseResourceTests(unittest.TestCase):
    def test_overlay_excludes_old_resources_preserves_runtime_and_never_mutates_installation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package, stage = root / "installed", root / "staged"
            (package / "resources/app/dist").mkdir(parents=True)
            (package / "resources/python").mkdir()
            stage.mkdir()
            (package / "Xuloom.exe").write_bytes(b"STABLE_EXE")
            (package / "resources/python/python.exe").write_bytes(b"STABLE_PYTHON")
            (package / "resources/app/dist/old.js").write_bytes(b"OLD")
            (stage / "new.js").write_bytes(b"NEW")
            entries = dict(release.archive_files(package, [(Path("resources/app/dist"), stage)]))
            self.assertNotIn(Path("resources/app/dist/old.js"), entries)
            self.assertEqual(entries[Path("resources/app/dist/new.js")].read_bytes(), b"NEW")
            self.assertEqual(entries[Path("Xuloom.exe")].read_bytes(), b"STABLE_EXE")
            self.assertEqual(entries[Path("resources/python/python.exe")].read_bytes(), b"STABLE_PYTHON")
            self.assertEqual((package / "resources/app/dist/old.js").read_bytes(), b"OLD")
            self.assertFalse((package / "resources/app/dist/new.js").exists())
            with self.assertRaises(ValueError): release.archive_files(package, [(Path("../escape"), stage)])
