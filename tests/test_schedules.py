import asyncio
import copy
import time
import unittest
from unittest.mock import AsyncMock, patch
import test_tool_recovery as recovery
from xuloom.features import schedules
from xuloom.core.session_lock import SessionBusyError, SessionLock
from xuloom.core.storage import data_directory


class ScheduleTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = recovery.RecoveryTests.asyncSetUp
    asyncTearDown = recovery.RecoveryTests.asyncTearDown

    async def test_once_and_lost_ack_reconcile_without_repeating_execution(self):
        row = schedules.add(self.engine, "Check the experiment", due_at=time.time()-1)
        execute = AsyncMock(return_value={"ok": True, "reason": "completed"})
        before = copy.deepcopy(schedules.list_items()[0])
        receipt = await schedules.deliver(self.engine, row["id"], execute)
        self.assertEqual(receipt["status"], "completed")
        with schedules.store() as rows: rows[row["id"]] = before  # lost acknowledgement after durable checkpoint
        await schedules.deliver(self.engine, row["id"], execute)
        execute.assert_awaited_once()
        self.assertEqual(schedules.list_items()[0]["status"], "completed")

    async def test_interrupted_delivery_is_blocked_instead_of_replayed(self):
        row = schedules.add(self.engine, "Mutate once", due_at=time.time()-1)
        execute = AsyncMock(side_effect=asyncio.CancelledError())
        with self.assertRaises(asyncio.CancelledError): await schedules.deliver(self.engine, row["id"], execute)
        next_execution = AsyncMock()
        result = await schedules.deliver(self.engine, row["id"], next_execution)
        self.assertEqual(result["status"], "blocked")
        next_execution.assert_not_awaited()
        with self.assertRaises(ValueError): schedules.change(row["id"], "resume")

    async def test_repeating_messages_coalesce_missed_periods_and_respect_pause_cancel(self):
        row = schedules.add(self.engine, "Daily check", due_at=time.time()-500, interval=60)
        execute = AsyncMock(return_value={"ok": True})
        await schedules.deliver(self.engine, row["id"], execute)
        updated = schedules.list_items()[0]
        self.assertGreater(updated["due_at"], time.time())
        self.assertLess(updated["due_at"], time.time()+61)
        self.assertEqual(schedules.list_items(due=True), [])
        schedules.change(row["id"], "pause")
        self.assertEqual((await schedules.deliver(self.engine, row["id"], execute))["status"], "not_due")
        schedules.change(row["id"], "resume"); schedules.change(row["id"], "cancel")
        self.assertEqual(schedules.list_items()[0]["status"], "cancelled")
        execute.assert_awaited_once()

    async def test_two_workers_cannot_claim_same_occurrence(self):
        row = schedules.add(self.engine, "Check once", due_at=time.time()-1)
        lock = SessionLock(data_directory()/"session-locks", "wake_"+row["id"]); lock.acquire()
        execute = AsyncMock()
        try:
            with self.assertRaises(SessionBusyError): await schedules.deliver(self.engine, row["id"], execute)
            execute.assert_not_awaited()
        finally: lock.close()

    async def test_failed_final_checkpoint_does_not_acknowledge_delivery(self):
        row = schedules.add(self.engine, "Check", due_at=time.time()-1)
        from xuloom.core.run_result import PersistenceError
        real = self.engine.persist_session_state
        def save():
            if self.engine.scheduled_deliveries.get(row["id"], {}).get("status") == "completed": return False
            return real()
        with patch.object(self.engine, "persist_session_state", side_effect=save):
            with self.assertRaises(PersistenceError): await schedules.deliver(self.engine, row["id"], AsyncMock(return_value={"ok": True}))
            retry = AsyncMock()
            with self.assertRaises(PersistenceError): await schedules.deliver(self.engine, row["id"], retry)
            retry.assert_not_awaited()
        self.assertEqual(schedules.list_items()[0]["status"], "active")

    async def test_active_goal_keeps_due_message_queued(self):
        row = schedules.add(self.engine, "Later", due_at=time.time()-1)
        self.engine.goal = {"status": "active"}
        execute = AsyncMock()
        self.assertEqual((await schedules.deliver(self.engine, row["id"], execute))["status"], "queued")
        execute.assert_not_awaited()

    async def test_live_turn_keeps_due_message_queued(self):
        row = schedules.add(self.engine, "Later", due_at=time.time()-1)
        self.engine.active_run_task = asyncio.current_task()
        execute = AsyncMock()
        try:
            self.assertEqual((await schedules.deliver(self.engine, row["id"], execute))["status"], "queued")
            self.assertEqual(schedules.list_items()[0]["status"], "active")
            execute.assert_not_awaited()
        finally: self.engine.active_run_task = None
