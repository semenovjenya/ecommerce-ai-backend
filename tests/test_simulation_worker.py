from __future__ import annotations

import threading

from agi_ecom_app.simulation_worker import _build_worker, _start_reconciler_watchdog


def test_build_worker_discards_connections_created_before_rq_timeout_configuration() -> None:
    events: list[str] = []

    class FakePool:
        def disconnect(self) -> None:
            events.append("disconnect")

        def reset(self) -> None:
            events.append("reset")

    class FakeConnection:
        connection_pool = FakePool()

    class FakeQueue:
        connection = FakeConnection()

    class FakeWorker:
        def __init__(self, queues: list[object], *, connection: object) -> None:
            assert queues == [queue]
            assert connection is queue.connection
            events.append("rq-configured")
            self.connection = connection

    queue = FakeQueue()

    worker = _build_worker(FakeWorker, queue)

    assert isinstance(worker, FakeWorker)
    assert events == ["rq-configured", "disconnect", "reset"]


def test_reconciler_watchdog_retries_after_scheduler_failure() -> None:
    queue = object()
    stopped = threading.Event()
    scheduled_with: list[object] = []

    def schedule_reconciliation(*, queue: object) -> str:
        scheduled_with.append(queue)
        if len(scheduled_with) == 1:
            raise RuntimeError("temporary Redis outage")
        stopped.set()
        return "scheduled"

    thread = _start_reconciler_watchdog(
        queue,
        schedule_reconciliation,
        stop_event=stopped,
        interval_seconds=0.01,
    )
    thread.join(timeout=1)

    assert not thread.is_alive()
    assert scheduled_with == [queue, queue]
