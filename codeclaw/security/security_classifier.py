import copy
import time

from codeclaw.security.permissions import PermissionDecision


class AutoSecurityClassifier:
    def __init__(self, mode: str = "heuristic"):
        self.mode = mode
        self.history = []

    def export_state(self) -> dict:
        return {
            "mode": self.mode,
            "history": copy.deepcopy(self.history),
        }

    def load_state(self, payload):
        payload = payload or {}
        self.mode = str(payload.get("mode", self.mode or "heuristic") or "heuristic")
        self.history = list(payload.get("history", []) or [])[:40]

    def classify(self, request) -> PermissionDecision:
        if str(self.mode or "off").lower() == "off":
            return None

        classifier_input = copy.deepcopy(getattr(request, "classifier_input", {}) or {})
        tool_name = str(classifier_input.get("tool_name", getattr(request, "tool_name", "")) or "")
        input_payload = copy.deepcopy(classifier_input.get("input_payload", {}) or getattr(request, "input_payload", {}) or {})
        summary = str(classifier_input.get("summary", getattr(request, "summary", "")) or "")
        candidate_text = " ".join([
            tool_name,
            summary,
            str(input_payload),
        ]).lower()

        destructive_markers = [
            "rm -rf",
            "git reset --hard",
            "git clean -fd",
            "git push --force",
            "git push -f",
            "del /f /s /q",
            "format c:",
            "mkfs",
            "shutdown /s",
        ]
        if any(marker in candidate_text for marker in destructive_markers):
            decision = PermissionDecision(
                "deny",
                "Heuristic security classifier blocked a destructive-looking action.",
                source="auto_security_classifier",
                metadata={"classifier_mode": self.mode, "severity": "high"},
            )
            self._record(tool_name, decision)
            return decision

        sensitive_markers = [
            ".env",
            "id_rsa",
            ".pem",
            ".key",
            "credentials",
            "secrets",
            "token",
            "password",
        ]
        if any(marker in candidate_text for marker in sensitive_markers):
            decision = PermissionDecision(
                "ask",
                "Heuristic security classifier flagged a potentially sensitive path or secret-bearing input.",
                source="auto_security_classifier",
                metadata={"classifier_mode": self.mode, "severity": "medium"},
            )
            self._record(tool_name, decision)
            return decision

        return None

    def _record(self, tool_name: str, decision: PermissionDecision):
        self.history.insert(0, {
            "tool_name": tool_name,
            "behavior": decision.behavior,
            "reason": decision.reason,
            "source": decision.source,
            "metadata": copy.deepcopy(decision.metadata or {}),
            "timestamp": time.time(),
        })
        self.history = self.history[:40]
