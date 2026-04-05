import time
from typing import Callable, Any
import asyncio

class FeishuRenderer:
    """
    Renders stream blocks into buffered text to avoid Feishu API rate limits during streaming.
    Provides batch synchronization ticks (e.g. flushing every 1.5s).
    """
    def __init__(self, update_callback: Callable[[str, bool], Any], min_flush_interval: float = 1.0):
        self.update_callback = update_callback
        self.min_flush_interval = min_flush_interval
        self.buffer = ""
        self.last_flush_time = time.time()
        self.flushing = False
        self.lock = asyncio.Lock()
        
    async def process_event(self, event: dict):
        event_type = event.get("type")
        if event_type == "content_block_delta":
            delta = event.get("delta", {})
            if delta.get("type") == "text_delta":
                self.buffer += delta.get("text", "")
                await self.maybe_flush()
        elif event_type == "message_stop":
            await self.flush(is_final=True)
            
    async def maybe_flush(self):
        now = time.time()
        if now - self.last_flush_time >= self.min_flush_interval:
            await self.flush(is_final=False)
            
    async def flush(self, is_final: bool = False):
        async with self.lock:
            if not self.buffer and not is_final:
                return
                
            self.last_flush_time = time.time()
            try:
                await self.update_callback(self.buffer, is_final)
            except Exception as e:
                print(f"[Renderer] Error updating feishu message: {e}")
