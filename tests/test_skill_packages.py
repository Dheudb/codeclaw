"""Portable external skill packages; no private skill fixtures."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from xuloom.agents.registry import AgentRegistry
from xuloom.protocols.mcp_config import MCPConfigStore
from xuloom.skills.registry import SkillRegistry


class SkillPackageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project = self.root / 'project'
        self.home = self.root / 'home'
        self.project.mkdir()
        self.home.mkdir()
        self.environment = patch.dict(os.environ, {'USERPROFILE': str(self.home), 'HOME': str(self.home), 'XULOOM_SKILLS_PATHS': ''})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.skill = self.home / '.xuloom/skills/fixture-package'
        (self.skill / 'agents').mkdir(parents=True)
        (self.skill / 'SKILL.md').write_text('---\nname: fixture-package\ndescription: Fixture package only\n---\nInspect fixture resources.', encoding='utf-8')
        (self.skill / 'agents/fixture-worker.md').write_text('---\nname: fixture-worker\ndescription: Fixture role\nmodel: inherit\ntools: [Read, Exec]\nweb_search: disabled\nmcp_servers: []\n---\nWork on the assigned fixture.', encoding='utf-8')
        (self.skill / 'mcp_servers.json').write_text(json.dumps({'mcpServers': {'fixture_service': {
            'command': '${python}', 'args': ['${skill}/server.py'], 'enabled': True}}}), encoding='utf-8')
        self.store = MCPConfigStore(self.project / 'mcp_servers.json')

    def test_external_skill_role_and_service_resolve_without_packaging(self):
        self.assertEqual(SkillRegistry(self.project).resolve('fixture-package')['scope'], 'global')
        role = AgentRegistry(self.project).resolve('fixture-worker')
        self.assertEqual(role['scope'], 'global')
        self.assertEqual(role['allowed_tools'], ['file_read_tool', 'command_tool'])
        service = self.store.read(include_skills=True)['mcpServers']['fixture_service']
        self.assertEqual(service['command'], sys.executable)
        self.assertEqual(Path(service['args'][0]).resolve(), (self.skill / 'server.py').resolve())
        self.assertEqual(self.store.read(), {'mcpServers': {}})
        self.assertFalse(self.store.path.exists())

    def test_workspace_overrides_and_disabled_service_survive_reload(self):
        self.store.toggle('fixture_service', False)
        self.assertFalse(MCPConfigStore(self.store.path).read(include_skills=True)['mcpServers']['fixture_service']['enabled'])
        self.store.save('fixture_service', {'command': 'fixture-command', 'args': [], 'enabled': True}, existing=True)
        self.assertEqual(self.store.read(include_skills=True)['mcpServers']['fixture_service']['command'], 'fixture-command')
        self.store.remove('fixture_service')
        self.assertFalse(self.store.read(include_skills=True)['mcpServers']['fixture_service']['enabled'])

    def test_removed_or_invalid_skill_cannot_supply_roles_or_services(self):
        (self.skill / 'SKILL.md').write_text('---\nname: other-name\ndescription: Fixture\n---\nInvalid folder name.', encoding='utf-8')
        self.assertNotIn('fixture-worker', AgentRegistry(self.project).definitions())
        self.assertNotIn('fixture_service', self.store.read(include_skills=True)['mcpServers'])


if __name__ == '__main__':
    unittest.main()
