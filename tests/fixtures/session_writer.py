"""Disposable cross-process writer/probe used by session ownership tests."""
import asyncio
import json
import os
from pathlib import Path
import sys
from unittest.mock import AsyncMock, patch

from codeclaw.core.session import SessionManager

root, mode, identity = Path(sys.argv[1]), sys.argv[2], sys.argv[3]

if mode == "hold":
    manager = SessionManager(cwd=root)
    manager.load_session_state(identity)
    assert manager.save_session_state(identity, [{"role": "user", "content": "first"}]), manager.last_save_error
    print("OWNED", flush=True)
    for command in sys.stdin:
        if command.strip() == "crash":
            os._exit(87)
        if command.strip() == "save":
            assert manager.save_session_state(identity, [{"role": "user", "content": "latest"}])
            print("SAVED", flush=True)
        else:
            break
    manager.release_writer()
elif mode == "probe":
    from codeclaw.core.engine import QueryEngine
    from codeclaw.memory.memory import MemoryCompactor
    async def probe():
        with patch.object(MemoryCompactor, "_resolve_token_encoder", return_value=None):
            engine = QueryEngine(model="fixture", model_provider="openai", cwd=str(root))
        try:
            assert engine.load_session(identity)
            client = engine.client.chat.completions.create = AsyncMock(side_effect=AssertionError("No request before ownership"))
            result = await engine.run("Do not execute while another writer owns this session", sys_print_callback=lambda _: None)
            agent_result = await engine.agents("capacity")
            target = root / "forbidden-effect.txt"
            tool_result = await engine._execute_tool_wrapper({"id": "forbidden", "name": "file_write_tool", "input": {"file_path": str(target), "content": "wrong"}}, lambda _: None)
            print(json.dumps({"reason": result.reason, "model_calls": client.await_count,
                "tool_error": tool_result["is_error"], "agent_error": agent_result["error"]["code"], "effect_exists": target.exists()}), flush=True)
        finally:
            await engine.aclose()
    asyncio.run(probe())
