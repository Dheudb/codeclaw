"""Own background subagent tasks and their cancellation lifecycle."""
import asyncio


class SubagentTasks:
    def __init__(self, limit=10):
        self.limit = limit
        self.tasks = {}
        self.results = {}
        self.delivered = set()

    def check_capacity(self, count):
        if sum(not task.done() for task in self.tasks.values()) + count > self.limit:
            raise ValueError("At most {} subagents may run at once".format(self.limit))

    def launch_batch(self, coroutines, on_cancelled_before_start=None):
        try:
            self.check_capacity(len(coroutines))
        except ValueError:
            for coroutine in coroutines.values():
                coroutine.close()
            raise
        # No awaits: capacity validation and allocation are atomic on this loop.
        for agent_id, coroutine in coroutines.items():
            self.launch(agent_id, coroutine, on_cancelled_before_start)

    def launch(self, agent_id, coroutine, on_cancelled_before_start=None):
        if agent_id in self.tasks and not self.tasks[agent_id].done():
            coroutine.close()
            raise ValueError("Subagent is already running")
        try:
            self.check_capacity(1)
        except ValueError:
            coroutine.close()
            raise

        self.results.pop(agent_id, None)
        self.delivered.discard(agent_id)
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
                if on_cancelled_before_start:
                    on_cancelled_before_start(self.results[agent_id])
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
