"""One fixed, local file operation, executed inside the system sandbox."""
import contextlib
import json
import os
import sys


async def execute(request):
    if request.get("provider"):
        os.environ["XULOOM_MODEL_PROVIDER"] = request["provider"]
    from xuloom.tools.file_read_tool import FileReadTool
    from xuloom.tools.file_write_tool import FileWriteTool
    from xuloom.tools.file_edit_tool import FileEditTool
    from xuloom.tools.glob_tool import GlobTool
    from xuloom.tools.grep_tool import GrepTool
    from xuloom.tools.notebook_edit_tool import NotebookEditTool
    from xuloom.context.file_state_cache import FileStateCache
    classes = [FileReadTool, FileWriteTool, FileEditTool, GlobTool, GrepTool, NotebookEditTool]
    tools = {cls.name: cls for cls in classes}
    if request["name"] not in tools:
        raise ValueError("Operation is not a confined file tool")
    events = []
    cache = FileStateCache(); cache.load_state(request.get("cache", {}))
    tool = tools[request["name"]]()
    tool.context = {"session_cwd": os.getcwd(), "confined_sync_worker": True, "session_id": request.get("session_id"), "agent_id": request.get("agent_id"),
        "read_file_state": request.get("read_state", {}), "file_state_cache": cache, "capability_event": events.append}
    with contextlib.redirect_stdout(sys.stderr):
        result = await tool(**request["arguments"])
    if isinstance(result, str) and result.startswith(("Error:", "Error reading file:", "Search execution error:", "Ripgrep error")):
        from xuloom.context.tool_results import build_tool_result
        result = build_tool_result(ok=False, content=result, is_error=True)
    return {"ok": True, "result": result, "read_state": tool.context["read_file_state"], "cache": cache.export_state(), "events": events}


def main():
    try:
        # These file operations are synchronous coroutines. Importing Windows
        # asyncio initializes Winsock, which a network-denied LPAC cannot do.
        # Never grant networking merely to run a local file operation.
        coroutine = execute(json.loads(sys.stdin.buffer.read().decode("utf-8")))
        try:
            coroutine.send(None)
        except StopIteration as done:
            reply = done.value
        else:
            coroutine.close()
            raise RuntimeError("Confined file operation unexpectedly required an event loop")
    except Exception as error:
        reply = {"ok": False, "error": type(error).__name__ + ": " + str(error)}
    sys.stdout.write(json.dumps(reply, ensure_ascii=True))
    sys.stdout.flush()
