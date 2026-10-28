import asyncio
import unittest

from xuloom.agents.tasks import SubagentTasks


class AgentQueueTests(unittest.IsolatedAsyncioTestCase):
    async def test_fifo_ten_workers_use_three_slots(self):
        tasks = SubagentTasks(limit=3)
        gates = [asyncio.Event() for _ in range(10)]
        entered, active, peak = [], set(), 0
        async def worker(index):
            nonlocal peak
            entered.append(index); active.add(index); peak = max(peak, len(active))
            await gates[index].wait()
            active.remove(index)
            return {"agent_id": str(index), "status": "completed"}
        tasks.launch_batch({str(i): worker(i) for i in range(10)})
        await asyncio.sleep(0.01)
        self.assertEqual(entered, [0, 1, 2])
        self.assertEqual(len(tasks.pending), 7)
        for index in range(10):
            gates[index].set()
            await asyncio.sleep(0.01)
        await asyncio.gather(*tasks.tasks.values())
        self.assertEqual(entered, list(range(10)))
        self.assertEqual(peak, 3)
        self.assertFalse(tasks.running or tasks.pending)
        self.assertEqual(len(tasks.completed()), 10)
        self.assertFalse(tasks.completed())

    async def test_cancel_queued_never_invokes_worker(self):
        tasks = SubagentTasks(limit=1)
        gate = asyncio.Event()
        entered, cancelled = [], []
        async def worker(index):
            entered.append(index)
            await gate.wait()
            return {"agent_id": str(index), "status": "completed"}
        tasks.launch_batch({str(i): worker(i) for i in range(3)}, cancelled.append)
        await asyncio.sleep(0.01)
        await tasks.cancel("1")
        self.assertEqual(entered, [0])
        self.assertEqual(cancelled[0]["agent_id"], "1")
        gate.set()
        await asyncio.gather(*tasks.tasks.values(), return_exceptions=True)
        self.assertEqual(entered, [0, 2])

    async def test_close_cancels_running_and_pending(self):
        tasks = SubagentTasks(limit=2)
        entered = []
        async def worker(index):
            entered.append(index)
            await asyncio.Event().wait()
        tasks.launch_batch({str(i): worker(i) for i in range(10)})
        await asyncio.sleep(0.01)
        await tasks.close()
        self.assertEqual(entered, [0, 1])
        self.assertFalse(tasks.running or tasks.pending)
        self.assertTrue(all(task.done() for task in tasks.tasks.values()))
        self.assertTrue(all(row["status"] == "cancelled" for row in tasks.results.values()))

    async def test_bounded_queue_rejects_whole_batch(self):
        tasks = SubagentTasks(limit=1, queue_limit=1)
        async def worker():
            await asyncio.Event().wait()
        with self.assertRaisesRegex(ValueError, "queue capacity"):
            tasks.launch_batch({str(i): worker() for i in range(3)})
        self.assertFalse(tasks.tasks)
