from dataclasses import asdict, dataclass
from typing import Dict, List


VALID_TODO_STATUSES = {"pending", "in_progress", "completed", "cancelled"}


@dataclass
class TodoItem:
    id: str
    content: str
    status: str


class TodoManager:
    """
    Stores structured task state for the current session.
    """

    def __init__(self):
        self._todos: Dict[str, TodoItem] = {}

    def write(self, todos: List[dict], merge: bool = True) -> str:
        if not isinstance(todos, list) or len(todos) < 1:
            return "Error: At least one todo item is required."

        next_items: Dict[str, TodoItem] = dict(self._todos) if merge else {}

        for raw in todos:
            todo_id = str(raw.get("id", "")).strip()
            content = str(raw.get("content", "")).strip()
            status = str(raw.get("status", "")).strip()

            if not todo_id:
                return "Error: Every todo item must include a non-empty 'id'."
            if not content:
                return f"Error: Todo '{todo_id}' must include non-empty 'content'."
            if status not in VALID_TODO_STATUSES:
                return (
                    f"Error: Todo '{todo_id}' has invalid status '{status}'. "
                    "Valid values: pending, in_progress, completed, cancelled."
                )

            next_items[todo_id] = TodoItem(
                id=todo_id,
                content=content,
                status=status,
            )

        in_progress_count = sum(1 for item in next_items.values() if item.status == "in_progress")
        if in_progress_count > 1:
            return "Error: At most one todo item may be 'in_progress' at a time."

        self._todos = next_items
        return self.render_user_summary()

    def list_items(self) -> List[TodoItem]:
        return list(self._todos.values())

    def export(self) -> List[dict]:
        return [asdict(item) for item in self.list_items()]

    def load(self, payload: List[dict]):
        self._todos = {}
        for raw in payload or []:
            todo_id = str(raw.get("id", "")).strip()
            content = str(raw.get("content", "")).strip()
            status = str(raw.get("status", "")).strip()
            if not todo_id or not content or status not in VALID_TODO_STATUSES:
                continue
            self._todos[todo_id] = TodoItem(id=todo_id, content=content, status=status)

    def render_prompt_summary(self) -> str:
        items = self.list_items()
        if not items:
            return ""

        lines = [
            "--- STRUCTURED TODO STATE ---",
            "Use this task list to track progress explicitly.",
        ]
        for item in items:
            lines.append(f"- [{item.status}] {item.id}: {item.content}")
        return "\n".join(lines)

    def render_user_summary(self) -> str:
        items = self.list_items()
        if not items:
            return "Todo list is currently empty."

        lines = ["Structured todo list updated:"]
        for item in items:
            lines.append(f"- [{item.status}] {item.id}: {item.content}")
        return "\n".join(lines)
