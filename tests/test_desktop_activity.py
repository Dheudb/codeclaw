import json
import tempfile
import unittest
from pathlib import Path
from codeclaw.desktop.activity import restore_activity, transcript_activity


class ActivityTests(unittest.TestCase):
    def messages(self, finished=True):
        messages = [{"role": "assistant", "_timestamp": "2026-10-14T01:00:00Z", "content": [
            {"type": "tool_use", "id": "same", "name": "bash_tool", "input": {"command": "echo proof"}}]}]
        if finished:
            messages.append({"role": "user", "_timestamp": "2026-10-14T01:00:02Z", "content": [
                {"type": "tool_result", "tool_use_id": "same", "content": "proof", "is_error": False}]})
        return messages

    def test_recover_input_output_timestamps_and_incomplete_status(self):
        activity = transcript_activity(self.messages())[0]
        self.assertEqual(activity["input"], {"command": "echo proof"})
        self.assertEqual(activity["result"], "proof")
        self.assertEqual(activity["finishedAt"] - activity["startedAt"], 2000)
        self.assertEqual(activity["status"], "completed")
        self.assertEqual(transcript_activity(self.messages(False))[0]["status"], "interrupted")

    def test_child_scope_cannot_collide_and_inherited_parent_calls_are_excluded(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = root / ".codeclaw/sessions/subagents"
            directory.mkdir(parents=True)
            (directory / "child.json").write_text(json.dumps({"task": "Inspect", "messages": self.messages() + [
                {"role": "user", "content": "Inspect"}] + self.messages()}), encoding="utf-8")
            raw = {"messages": self.messages(), "metadata": {"subagents": [
                {"agent_id": "child", "status": "completed", "description": "Inspect files", "result": "done"},
                {"agent_id": "missing", "status": "running"}, {"agent_id": "../escape", "status": "running"}]}}
            activity, workers = restore_activity(raw, root)
            self.assertEqual(len(activity), 4)
            self.assertEqual({item["id"] for item in activity}, {"main:same", "child:same", "agent:child", "agent:missing"})
            self.assertEqual(workers[-1]["status"], "interrupted")
            self.assertEqual(len(workers), 2)

    def test_error_and_legacy_histories(self):
        messages = self.messages()
        messages[-1]["content"][0]["is_error"] = True
        with tempfile.TemporaryDirectory() as temp:
            activity, workers = restore_activity(messages, Path(temp))
        self.assertEqual(activity[0]["status"], "error")
        self.assertEqual(workers, [])
