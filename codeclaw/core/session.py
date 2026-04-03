import os
import json
import shutil
import time
import tempfile

class SessionManager:
    """
    Infinite Context Engine Cache Layer. 
    Drops conversation trees into lightweight logs maintaining persistence after crash or terminal exit.
    """
    def __init__(self, session_dir=".codeclaw/sessions"):
        self.session_dir = session_dir
        os.makedirs(self.session_dir, exist_ok=True)
        self.last_save_error = ""
        self.last_save_path = ""
        self.last_save_ok = True

    def _make_json_safe(self, value):
        if value is None or isinstance(value, (str, int, float, bool)):
            return value

        if isinstance(value, dict):
            return {
                str(key): self._make_json_safe(item)
                for key, item in value.items()
            }

        if isinstance(value, (list, tuple, set)):
            return [self._make_json_safe(item) for item in value]

        if hasattr(value, "model_dump") and callable(value.model_dump):
            return self._make_json_safe(value.model_dump())

        if hasattr(value, "dict") and callable(value.dict):
            return self._make_json_safe(value.dict())

        if hasattr(value, "to_dict") and callable(value.to_dict):
            return self._make_json_safe(value.to_dict())

        if hasattr(value, "__dict__"):
            return self._make_json_safe(vars(value))

        return repr(value)

    def _write_json_atomic(self, path: str, payload: dict):
        parent = os.path.dirname(path)
        os.makedirs(parent, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(prefix=".session-", suffix=".tmp", dir=parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(temp_path, path)
        except Exception:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise
        
    def save_session_state(self, session_id: str, messages: list, metadata: dict = None):
        path = os.path.join(self.session_dir, f"{session_id}.json")
        try:
            payload = {
                "messages": self._make_json_safe(messages),
                "metadata": self._make_json_safe(metadata or {}),
            }
            self._write_json_atomic(path, payload)
            self.last_save_ok = True
            self.last_save_error = ""
            self.last_save_path = path
            return True
        except Exception as e:
            self.last_save_ok = False
            self.last_save_error = str(e)
            self.last_save_path = path
            return False

    def save_session(self, session_id: str, messages: list):
        self.save_session_state(session_id, messages, metadata={})
            
    def load_session_state(self, session_id: str) -> dict:
        path = os.path.join(self.session_dir, f"{session_id}.json")
        if not os.path.exists(path):
            return {"messages": [], "metadata": {}}
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
                if isinstance(payload, list):
                    return {"messages": payload, "metadata": {}}
                if isinstance(payload, dict):
                    return {
                        "messages": payload.get("messages", []),
                        "metadata": payload.get("metadata", {}),
                    }
                return {"messages": [], "metadata": {}}
        except Exception:
            return {"messages": [], "metadata": {}}

    def load_session(self, session_id: str) -> list:
        return self.load_session_state(session_id).get("messages", [])

    def save_subagent_transcript(self, agent_id: str, payload: dict):
        subagent_dir = os.path.join(self.session_dir, "subagents")
        os.makedirs(subagent_dir, exist_ok=True)
        path = os.path.join(subagent_dir, f"{agent_id}.json")
        try:
            self._write_json_atomic(path, self._make_json_safe(payload))
            self.last_save_ok = True
            self.last_save_error = ""
            self.last_save_path = path
            return True
        except Exception as e:
            self.last_save_ok = False
            self.last_save_error = str(e)
            self.last_save_path = path
            return False

    def get_subagent_transcript_path(self, agent_id: str) -> str:
        subagent_dir = os.path.join(self.session_dir, "subagents")
        os.makedirs(subagent_dir, exist_ok=True)
        return os.path.join(subagent_dir, f"{agent_id}.json")
            
    def get_recent_sessions(self) -> str:
        """Returns string representation of the last 5 sessions."""
        try:
            files = [f for f in os.listdir(self.session_dir) if f.endswith('.json')]
            if not files:
                return "No existing sessions found."
                
            sorted_files = sorted(
                files, 
                key=lambda x: os.path.getmtime(os.path.join(self.session_dir, x)),
                reverse=True
            )
            
            out = "[bold cyan]Active Sessions Cache[/bold cyan]\n"
            for f in sorted_files[:5]:
                sid = f.replace('.json', '')
                mtime = os.path.getmtime(os.path.join(self.session_dir, f))
                time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))
                state = self.load_session_state(sid)
                metadata = state.get("metadata", {})
                mode = metadata.get("plan", {}).get("mode", "normal")
                todo_count = len(metadata.get("todos", []))
                subagent_count = len(metadata.get("subagents", []))
                sandbox = metadata.get("sandbox", {})
                sandbox_label = sandbox.get("branch", "none") if sandbox else "none"
                out += (
                    f"  - ID: [green]{sid}[/green] (Last Modified: {time_str})\n"
                    f"    mode={mode}, todos={todo_count}, subagents={subagent_count}, sandbox={sandbox_label}\n"
                )
            out += "\n[dim]To resume, type: /resume <ID>[/dim]"
            return out
        except Exception as e:
            return f"Error retrieving sessions: {e}"

    def delete_session(self, session_id: str) -> bool:
        path = os.path.join(self.session_dir, f"{session_id}.json")
        found = False
        if os.path.exists(path):
            os.unlink(path)
            found = True

        self._cleanup_subagents_for_session(session_id)
        self._cleanup_plan_files_for_session(session_id)
        return found

    def _cleanup_subagents_for_session(self, session_id: str):
        """Remove subagent transcripts that belong to the given session."""
        subagent_dir = os.path.join(self.session_dir, "subagents")
        if not os.path.isdir(subagent_dir):
            return
        for fname in os.listdir(subagent_dir):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(subagent_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and data.get("session_id") == session_id:
                    os.unlink(fpath)
            except Exception:
                pass

    def _cleanup_plan_files_for_session(self, session_id: str):
        """Remove plan files generated by the given session."""
        try:
            from codeclaw.core.plans import get_plan_file_path, get_plans_directory
        except ImportError:
            return
        plan_path = get_plan_file_path(session_id)
        if os.path.exists(plan_path):
            os.unlink(plan_path)
        plans_dir = get_plans_directory()
        if not os.path.isdir(plans_dir):
            return
        from codeclaw.core.plans import _make_plan_slug
        slug = _make_plan_slug(session_id)
        for fname in os.listdir(plans_dir):
            if fname.startswith(slug) and fname.endswith(".md"):
                try:
                    os.unlink(os.path.join(plans_dir, fname))
                except Exception:
                    pass

    def list_session_ids(self) -> list:
        try:
            files = [f for f in os.listdir(self.session_dir) if f.endswith('.json')]
            return [f.replace('.json', '') for f in sorted(
                files,
                key=lambda x: os.path.getmtime(os.path.join(self.session_dir, x)),
                reverse=True,
            )]
        except Exception:
            return []

    def clear_all_sessions(self) -> int:
        count = 0
        try:
            for f in os.listdir(self.session_dir):
                fpath = os.path.join(self.session_dir, f)
                if f.endswith('.json'):
                    sid = f.replace('.json', '')
                    self._cleanup_plan_files_for_session(sid)
                    os.unlink(fpath)
                    count += 1

            subagent_dir = os.path.join(self.session_dir, "subagents")
            if os.path.isdir(subagent_dir):
                shutil.rmtree(subagent_dir, ignore_errors=True)
        except Exception:
            pass
        return count
