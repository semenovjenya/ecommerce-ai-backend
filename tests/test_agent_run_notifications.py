from __future__ import annotations

import asyncio
import inspect
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agi_ecom_app.agent_run_notifications import (
    AGENT_RUN_EVENTS_CHANNEL,
    AgentRunNotificationHub,
    _notification_run_id,
)
from agi_ecom_app.api_models import AgentRunStatusResponse
from agi_ecom_app.services.memory import MEMORY_SCHEMA_SQL
from agi_ecom_app.websocket_chat import _wait_run_until_terminal


class _RecordingConnection:
    def __init__(self) -> None:
        self.events = []

    async def send_event(self, event) -> None:
        self.events.append(event)


class _FakeNotificationHub:
    def __init__(self, message: str = "changed") -> None:
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.queue.put_nowait(message)
        self.subscriptions: list[int] = []
        self.unsubscriptions: list[tuple[int, asyncio.Queue[str]]] = []

    async def subscribe(self, run_id: int) -> asyncio.Queue[str]:
        self.subscriptions.append(run_id)
        return self.queue

    async def unsubscribe(self, run_id: int, queue: asyncio.Queue[str]) -> None:
        self.unsubscriptions.append((run_id, queue))


class AgentRunWaitTests(unittest.IsolatedAsyncioTestCase):
    async def test_wait_uses_notification_then_rereads_durable_terminal_status(self) -> None:
        hub = _FakeNotificationHub(message="reconnecting")
        connection = _RecordingConnection()
        statuses = [
            AgentRunStatusResponse(
                run_id=41,
                chat_id="chat-41",
                status="running",
                intent="sales",
            ),
            AgentRunStatusResponse(
                run_id=41,
                chat_id="chat-41",
                status="completed",
                intent="sales",
                server_message_id=8,
                answer="Готово",
            ),
        ]

        with (
            patch(
                "agi_ecom_app.websocket_chat.get_agent_run_notification_hub",
                return_value=hub,
            ),
            patch(
                "agi_ecom_app.websocket_chat.get_agent_run_status_payload",
                side_effect=statuses,
            ) as durable_read,
            patch(
                "agi_ecom_app.websocket_chat.asyncio.sleep",
                side_effect=AssertionError("agent status wait must not poll"),
            ),
        ):
            response = await _wait_run_until_terminal(
                connection=connection,
                client_message_id="client-41",
                run_id=41,
                chat_id="chat-41",
            )

        self.assertEqual(response.status, "completed")
        self.assertEqual(response.answer, "Готово")
        self.assertEqual(durable_read.call_count, 2)
        self.assertEqual([event.status for event in connection.events], ["running", "completed"])
        self.assertEqual(hub.subscriptions, [41])
        self.assertEqual(hub.unsubscriptions, [(41, hub.queue)])

    async def test_one_listener_task_is_shared_across_run_subscribers(self) -> None:
        hub = AgentRunNotificationHub()
        listener_started = asyncio.Event()
        keep_listener_alive = asyncio.Event()

        async def fake_listener() -> None:
            hub._ready.set()
            listener_started.set()
            await keep_listener_alive.wait()

        with patch.object(hub, "_listen", side_effect=fake_listener) as listener:
            first = await hub.subscribe(1)
            await listener_started.wait()
            second = await hub.subscribe(2)

            self.assertEqual(listener.call_count, 1)
            self.assertIsNot(first, second)

            await hub.unsubscribe(1, first)
            self.assertIsNotNone(hub._task)
            await hub.unsubscribe(2, second)

        self.assertIsNone(hub._task)

    async def test_notification_is_fanned_out_only_to_matching_run(self) -> None:
        hub = AgentRunNotificationHub()
        run_one_a: asyncio.Queue[str] = asyncio.Queue()
        run_one_b: asyncio.Queue[str] = asyncio.Queue()
        run_two: asyncio.Queue[str] = asyncio.Queue()
        hub._subscribers = {1: {run_one_a, run_one_b}, 2: {run_two}}

        hub._broadcast("changed", run_id=1)

        self.assertEqual(run_one_a.get_nowait(), "changed")
        self.assertEqual(run_one_b.get_nowait(), "changed")
        self.assertTrue(run_two.empty())


class AgentRunNotificationContractTests(unittest.TestCase):
    def test_trigger_notifies_run_status_changes(self) -> None:
        normalized_schema = " ".join(MEMORY_SCHEMA_SQL.split()).lower()
        self.assertIn("after insert or update of status, assistant_message_seq", normalized_schema)
        self.assertIn(f"pg_notify( '{AGENT_RUN_EVENTS_CHANNEL}'", normalized_schema)

    def test_notification_payload_parser_is_defensive(self) -> None:
        self.assertEqual(
            _notification_run_id(
                SimpleNamespace(payload=json.dumps({"run_id": 73, "status": "completed"}))
            ),
            73,
        )
        self.assertIsNone(_notification_run_id(SimpleNamespace(payload="not-json")))
        self.assertIsNone(_notification_run_id(SimpleNamespace(payload=json.dumps({"status": "running"}))))

    def test_websocket_status_wait_has_no_two_second_polling(self) -> None:
        source = inspect.getsource(_wait_run_until_terminal)
        self.assertNotIn("sleep(2", source)
        self.assertIn("notifications.get()", source)


if __name__ == "__main__":
    unittest.main()
