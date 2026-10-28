import json
import unittest
from xuloom.features.workflow_completion import WorkflowCompletion


class CompletionTests(unittest.TestCase):
    def test_signals_are_opt_in_scoped_and_never_inferred_from_prose(self):
        guard = WorkflowCompletion()
        signal = {'workflow_finished': False, 'artifact_locator': {'run_handle': 'run:first'}, 'required_main_action': {'action': 'select'}}
        guard.observe('one', json.dumps(signal))
        self.assertFalse(guard.pending)
        guard.activate({'completion_protocol': 'workflow-json-v1'})
        for invalid in ['I will continue next.', '[]', '{"workflow_finished":false}', '{"run_handle":"one","workflow_finished":"false"}', '{"run_handle":"one","workflow_finished":false']:
            guard.observe('invalid', invalid)
        self.assertFalse(guard.pending)
        guard.observe('one', json.dumps(signal))
        self.assertEqual(guard.pending['run:first']['required_action'], 'select')
        guard.observe('two', '{"run_handle":"run:second","next_command":["fixture","next"]}')
        guard.observe('three', '{"run_handle":"run:first","workflow_finished":true}')
        self.assertEqual(list(guard.pending), ['run:second'])
        guard.observe('one', json.dumps(signal))
        self.assertEqual(list(guard.pending), ['run:second'])
        guard.observe('four', '{"run_handle":"run:second","workflow_finished":false,"workflow_status":"blocked"}')
        self.assertFalse(guard.pending)
        guard.reset()
        self.assertFalse(guard.enabled)
