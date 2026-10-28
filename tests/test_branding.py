"""A clean rename must not silently reload the previous namespace's state."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from xuloom.branding import configuration_file
from xuloom.core.storage import data_directory
from xuloom.skills.registry import SkillRegistry


class BrandingTests(unittest.TestCase):
    def test_only_new_skill_directory_is_discovered(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for directory, name in [(".codeclaw", "old-only"), (".xuloom", "new-only")]:
                file = root / directory / "skills" / name / "SKILL.md"
                file.parent.mkdir(parents=True)
                file.write_text(f"---\nname: {name}\ndescription: Test skill\n---\nTask instructions", encoding="utf-8")
            names = {skill["name"] for skill in SkillRegistry(root, root / "home").active()}
            self.assertIn("new-only", names)
            self.assertNotIn("old-only", names)

    def test_old_settings_do_not_override_new_defaults(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = root / ".codeclaw/capabilities.json"
            old.parent.mkdir()
            old.write_text(json.dumps({"skill_roots": ["should-not-load"]}))
            registry = SkillRegistry(root, root / "home")
            self.assertEqual(registry.settings()["skill_roots"], [])
            registry.update(skill_roots=[])
            self.assertTrue((root / ".xuloom/capabilities.json").is_file())
            self.assertEqual(json.loads(old.read_text())["skill_roots"], ["should-not-load"])
            self.assertEqual(configuration_file(root, "permissions.json"), root / ".xuloom/permissions.json")

    def test_old_data_environment_is_not_an_alias(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {}, clear=True), patch("pathlib.Path.home", return_value=Path(temporary)):
            os.environ["CODECLAW_DATA_DIR"] = str(Path(temporary) / "ignored")
            with patch("sys.platform", "linux"):
                self.assertEqual(data_directory(), Path(temporary) / ".local/share/xuloom")
            os.environ["XULOOM_DATA_DIR"] = str(Path(temporary) / "selected")
            self.assertEqual(data_directory(), Path(temporary) / "selected")
