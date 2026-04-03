import os
from pydantic import BaseModel, Field
from codeclaw.tools.base import BaseAgenticTool

class NotebookToolInput(BaseModel):
    action: str = Field(..., description="'read', 'write', or 'append'")
    title: str = Field(..., description="The title or filename of the notebook (e.g. 'architecture_todo' or 'bug_list'). It will be auto-saved as a Markdown file.")
    content: str = Field(None, description="The content to write or append. Required for 'write' and 'append' actions.")

class NotebookTool(BaseAgenticTool):
    name = "notebook_tool"
    description = "Internal memory notebook for logging architectural milestones and hypotheses across multi-agent turns. Saves to .codeclaw/notebooks/ directory."
    input_schema = NotebookToolInput
    risk_level = "medium"

    def is_read_only_call(self, action: str, title: str, content: str = None) -> bool:
        return action == "read"

    def build_permission_summary(self, action: str, title: str, content: str = None) -> str:
        preview = (content or "")[:160] + ("..." if content and len(content) > 160 else "")
        return (
            "Notebook operation requested.\n"
            f"action: {action}\n"
            f"title: {title}\n"
            f"content_preview: {preview or '<empty>'}"
        )

    def _write_queue(self):
        return self.context.get("incremental_write_queue")
    
    async def execute(self, action: str, title: str, content: str = None) -> str:
        nb_dir = ".codeclaw/notebooks"
        os.makedirs(nb_dir, exist_ok=True)
        
        safe_title = "".join([c if c.isalnum() or c in ['-', '_'] else '_' for c in title]).strip()
        if not safe_title.endswith(".md"): safe_title += ".md"
            
        path = os.path.join(nb_dir, safe_title)
        
        if action == "read":
            if not os.path.exists(path):
                return f"Notebook '{title}' is empty or does not exist."
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception as e:
                return f"Error reading notebook: {e}"
                
        elif action == "write":
            if not content: return "Error: Content block required for 'write' operation."
            try:
                def _perform_write():
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(content)
                    return f"Successfully wrote to tactical notebook '{title}'."
                queue = self._write_queue()
                if queue is not None:
                    return await queue.run(path, "notebook_write", _perform_write)
                return _perform_write()
            except Exception as e:
                return f"Error writing notebook: {e}"
                
        elif action == "append":
            if not content: return "Error: Content block required for 'append' operation."
            try:
                def _perform_append():
                    with open(path, "a", encoding="utf-8") as f:
                        f.write("\\n" + content)
                    return f"Successfully appended to tactical notebook '{title}'."
                queue = self._write_queue()
                if queue is not None:
                    return await queue.run(path, "notebook_append", _perform_append)
                return _perform_append()
            except Exception as e:
                return f"Error appending notebook: {e}"
        
        return "Unknown action parameter. Supported: 'read', 'write', or 'append'."
