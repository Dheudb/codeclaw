"""Own background subagent tasks and their cancellation lifecycle."""
import asyncio


class SubagentTasks:
    def __init__(self, limit=4):
        self.limit = limit
        self.tasks = {}
        self.results = {}
        self.delivered = set()

    def launch(self, agent_id, coroutine):
        if sum(not task.done() for task in self.tasks.values()) >= self.limit:
            coroutine.close()
            raise ValueError("At most {} background subagents may run at once".format(self.limit))

        started = False
        async def run():
            nonlocal started
            started = True
            try:
                result = await coroutine
                self.results[agent_id] = result
                return result
            except asyncio.CancelledError:
                self.results[agent_id] = {"agent_id": agent_id, "status": "cancelled", "result": "Cancelled by user"}
                raise
            except Exception as exc:
                self.results[agent_id] = {"agent_id": agent_id, "status": "crashed", "result": str(exc)}
                return self.results[agent_id]

        task = asyncio.create_task(run())
        def settled(completed):
            if completed.cancelled() and not started:
                coroutine.close()
                self.results[agent_id] = {"agent_id": agent_id, "status": "cancelled", "result": "Cancelled before execution"}
        task.add_done_callback(settled)
        self.tasks[agent_id] = task

    async def cancel(self, agent_id):
        task = self.tasks.get(agent_id)
        if task is None:
            raise ValueError("Unknown background subagent")
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        return self.results.get(agent_id, {"agent_id": agent_id, "status": "cancelled"})

    async def close(self):
        for task in self.tasks.values():
            task.cancel()
        await asyncio.gather(*self.tasks.values(), return_exceptions=True)

    def completed(self):
        results = [r for key, r in self.results.items() if key not in self.delivered]
        self.delivered.update(self.results)
        return results
