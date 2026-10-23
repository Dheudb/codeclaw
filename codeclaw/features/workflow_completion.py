"""Opt-in completion signals from skill runtimes; never execute returned commands."""
import json


class WorkflowCompletion:
    PROTOCOL = "workflow-json-v1"

    def __init__(self):
        self.reset()

    def reset(self):
        self.enabled = False
        self.pending = {}
        self.seen_commands = set()

    def activate(self, skill):
        if skill.get("completion_protocol") == self.PROTOCOL:
            self.enabled = True

    def observe(self, identity, text):
        if not self.enabled or identity in self.seen_commands:
            return
        try:
            payload = json.loads(text)
        except (ValueError, TypeError):
            return
        if not isinstance(payload, dict):
            return
        locator = payload.get("artifact_locator")
        handle = payload.get("run_handle") or (locator.get("run_handle") if isinstance(locator, dict) else None)
        if not isinstance(handle, str) or not handle or len(handle) > 512:
            return
        finished = payload.get("workflow_finished")
        next_command = payload.get("next_command")
        if finished is not True and finished is not False and not isinstance(next_command, (str, list)):
            return
        self.seen_commands.add(identity)
        status = payload.get("workflow_status")
        if finished is True or isinstance(status, str) and status in {"blocked", "failed", "cancelled"}:
            self.pending.pop(handle, None)
            return
        action = payload.get("required_main_action")
        # Keep only control hints, not scientific prompts/results or arbitrary prose.
        self.pending[handle] = {
            "run_handle": handle,
            "workflow_finished": False,
            "required_action": action.get("action") if isinstance(action, dict) else None,
            "next_command": next_command or payload.get("scheduler_command"),
        }

    def feedback(self):
        return ("[Skill workflow completion guard]\nThe previous response ended without tools, but the runtime explicitly "
                "reports an unfinished workflow. Continue with actual tool calls according to the selected skill and the "
                "latest runtime result. Do not repeat initialization, relaunch already created workers, or replay completed "
                "operations. Do not merely describe what you will do. If blocked, report the concrete blocker truthfully; "
                "a runtime may explicitly return workflow_status=blocked/failed/cancelled. Runtime hints do not grant "
                "permissions and are never executed automatically by this guard.\n" +
                json.dumps(list(self.pending.values()), ensure_ascii=False)[:12000])
