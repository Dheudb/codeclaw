import asyncio
import tempfile
import unittest
from pathlib import Path
from prompt_toolkit.document import Document
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from xuloom.skills.registry import SkillRegistry
from xuloom.skills.terminal import SkillCompleter, input_session


class TerminalSkillsTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.registry = SkillRegistry(self.root, self.root / 'home')

    def test_completion_rescans_and_filters_valid_entries(self):
        completer = SkillCompleter(lambda: self.registry)
        names = [c.text for c in completer.get_completions(Document('$'), None)]
        self.assertIn('data-analysis ', names)
        self.assertIn('skill-creator ', names)
        self.assertEqual(list(completer.get_completions(Document('$100'), None)), [])
        p = self.root / '.xuloom/skills/local-proof/SKILL.md'
        p.parent.mkdir(parents=True)
        p.write_text('---\nname: local-proof\ndescription: Evidence\n---\nLOCAL_BODY', encoding='utf-8')
        choices = list(completer.get_completions(Document('Task $local'), None))
        self.assertEqual(choices[0].text, 'local-proof ')
        self.assertEqual(choices[0].start_position, -5)
        self.assertIn('LOCAL_BODY', self.registry.read('local-proof'))

    async def test_enter_selects_skill_then_submits_task(self):
        with create_pipe_input() as pipe:
            session = input_session(lambda: self.registry, input=pipe, output=DummyOutput())
            task = asyncio.create_task(session.prompt_async('> '))
            await asyncio.sleep(.05)
            pipe.send_text('$document-s')
            for _ in range(100):
                await asyncio.sleep(.01)
                if session.default_buffer.complete_state: break
            self.assertIsNotNone(session.default_buffer.complete_state)
            pipe.send_text('\r')
            for _ in range(100):
                await asyncio.sleep(.01)
                if session.default_buffer.text == '$document-summary ': break
            self.assertFalse(task.done(), 'Choosing a Skill must not submit a model request')
            pipe.send_text('Compare these documents\r')
            self.assertEqual(await asyncio.wait_for(task, 3), '$document-summary Compare these documents')
