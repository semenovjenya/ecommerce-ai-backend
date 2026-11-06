from __future__ import annotations

import asyncio
import json
import logging
from typing import Any
from weakref import WeakKeyDictionary

import psycopg

from agi_ecom_app.config import get_settings


logger = logging.getLogger(__name__)

AGENT_RUN_EVENTS_CHANNEL = "agent_run_trace_events"
LISTENER_IDLE_TIMEOUT_SECONDS = 30.0
LISTENER_RECONNECT_DELAY_SECONDS = 1.0


async def _connect_agent_run_listener() -> psycopg.AsyncConnection[Any]:
    connection = await psycopg.AsyncConnection.connect(
        get_settings().memory_postgres_dsn,
        autocommit=True,
    )
    await connection.execute(f"LISTEN {AGENT_RUN_EVENTS_CHANNEL}")
    return connection


async def _wait_for_agent_run_notification(
    connection: psycopg.AsyncConnection[Any],
) -> Any | None:
    notification = None
    async for notification in connection.notifies(
        timeout=LISTENER_IDLE_TIMEOUT_SECONDS,
        stop_after=1,
    ):
        # Let psycopg finish the async generator. Returning from inside this loop
        # can leave its connection lock held until the generator is collected.
        pass
    return notification


def _notification_run_id(notification: Any) -> int | None:
    raw_payload = getattr(notification, "payload", None)
    if not isinstance(raw_payload, str) or not raw_payload:
        return None
    try:
        payload = json.loads(raw_payload)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    raw_run_id = payload.get("run_id", payload.get("runId"))
    try:
        return int(raw_run_id)
    except (TypeError, ValueError):
        return None


class AgentRunNotificationHub:
    """One LISTEN connection per API event loop, fanned out by agent run."""

    def __init__(self) -> None:
        self._subscribers: dict[int, set[asyncio.Queue[str]]] = {}
        self._task: asyncio.Task[None] | None = None
        self._ready = asyncio.Event()

    async def subscribe(self, run_id: int) -> asyncio.Queue[str]:
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=2)
        self._subscribers.setdefault(run_id, set()).add(queue)
        if self._task is None or self._task.done():
            self._ready.clear()
            self._task = asyncio.create_task(self._listen())
        try:
            await self._ready.wait()
        except BaseException:
            await self.unsubscribe(run_id, queue)
            raise
        return queue

    async def unsubscribe(self, run_id: int, queue: asyncio.Queue[str]) -> None:
        subscribers = self._subscribers.get(run_id)
        if subscribers is not None:
            subscribers.discard(queue)
            if not subscribers:
                self._subscribers.pop(run_id, None)

        if not self._subscribers and self._task is not None:
            task = self._task
            self._task = None
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    @staticmethod
    def _offer(queue: asyncio.Queue[str], message: str) -> None:
        # Coalescing is safe: consumers always reread the durable Postgres row.
        if queue.full():
            queue.get_nowait()
        queue.put_nowait(message)

    def _broadcast(self, message: str, run_id: int | None = None) -> None:
        subscriber_groups = (
            [self._subscribers.get(run_id, set())]
            if run_id is not None
            else list(self._subscribers.values())
        )
        for subscribers in subscriber_groups:
            for queue in tuple(subscribers):
                self._offer(queue, message)

    async def _listen(self) -> None:
        needs_durable_replay = False
        while self._subscribers:
            connection: psycopg.AsyncConnection[Any] | None = None
            try:
                connection = await _connect_agent_run_listener()
                if needs_durable_replay:
                    self._broadcast("replay")
                    needs_durable_replay = False
                self._ready.set()
                while self._subscribers:
                    notification = await _wait_for_agent_run_notification(connection)
                    if notification is None:
                        continue
                    run_id = _notification_run_id(notification)
                    if run_id is not None:
                        self._broadcast("changed", run_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("agent run LISTEN reconnecting error=%s", exc)
                needs_durable_replay = True
                # Do not block websocket setup if Postgres is temporarily down.
                # The consumer performs a durable read after subscribe and again
                # on this reconnect signal.
                self._ready.set()
                self._broadcast("reconnecting")
                await asyncio.sleep(LISTENER_RECONNECT_DELAY_SECONDS)
            finally:
                if connection is not None:
                    try:
                        await connection.close()
                    except Exception as exc:
                        logger.warning("agent run LISTEN close failed error=%s", exc)


_HUBS: WeakKeyDictionary[
    asyncio.AbstractEventLoop,
    AgentRunNotificationHub,
] = WeakKeyDictionary()


def get_agent_run_notification_hub() -> AgentRunNotificationHub:
    loop = asyncio.get_running_loop()
    hub = _HUBS.get(loop)
    if hub is None:
        hub = AgentRunNotificationHub()
        _HUBS[loop] = hub
    return hub
