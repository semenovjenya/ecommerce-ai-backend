from __future__ import annotations

from collections.abc import Callable
import logging
import os
from pathlib import Path
import threading

from agi_ecom_app.config import get_settings


logger = logging.getLogger(__name__)


def _prepare_multiprocess_metrics() -> object:
    metrics_dir = Path(
        os.getenv("PROMETHEUS_MULTIPROC_DIR", "/tmp/decision-simulation-prometheus")
    )
    metrics_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PROMETHEUS_MULTIPROC_DIR"] = str(metrics_dir)
    for stale_file in metrics_dir.glob("*.db"):
        stale_file.unlink(missing_ok=True)

    from prometheus_client import CollectorRegistry, multiprocess

    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
    return registry


def _start_queue_depth_observer(queue: object) -> None:
    """Publish Redis queue depth from the process that serves /metrics."""

    from agi_ecom_app.decision_simulation.metrics import QUEUED_JOBS

    def observe() -> None:
        while True:
            try:
                QUEUED_JOBS.set(len(queue))  # type: ignore[arg-type]
            except Exception:
                # Redis outages are reflected by the stale last value and the
                # worker's own connectivity logs; they must not kill metrics.
                pass
            threading.Event().wait(5)

    threading.Thread(
        target=observe,
        name="decision-simulation-queue-depth",
        daemon=True,
    ).start()


def _start_reconciler_watchdog(
    queue: object,
    schedule_reconciliation: Callable[..., object],
    *,
    stop_event: threading.Event | None = None,
    interval_seconds: float | None = None,
) -> threading.Thread:
    """Keep the recurring reconciliation chain alive from the worker parent.

    An RQ work horse can be killed without running the reconciler job's
    ``finally`` block. The parent process survives that failure, so it
    periodically re-enqueues the deterministic future slot. Duplicate slots
    are harmless because the scheduler uses a stable job ID.
    """

    stopped = stop_event or threading.Event()
    if interval_seconds is not None:
        watchdog_interval = max(0.01, float(interval_seconds))
    else:
        schedule_interval = max(
            30,
            get_settings().decision_simulation_stale_after_seconds // 2,
        )
        watchdog_interval = max(5.0, min(60.0, schedule_interval / 2))

    def watch() -> None:
        while not stopped.wait(watchdog_interval):
            try:
                schedule_reconciliation(queue=queue)
            except Exception:
                # A transient Redis outage must not terminate the watchdog;
                # the next tick retries the deterministic schedule slot.
                logger.exception("decision simulation reconciler watchdog failed")

    thread = threading.Thread(
        target=watch,
        name="decision-simulation-reconciler-watchdog",
        daemon=True,
    )
    thread.start()
    return thread


def _build_worker(worker_class: type, queue: object) -> object:
    """Create the RQ worker before anything else opens its Redis pool.

    RQ expands the Redis socket timeout so its blocking dequeue can wait for
    ``dequeue_timeout`` seconds.  redis-py 8, however, gives connections opened
    before that adjustment a five-second socket timeout.  Updating the pool's
    kwargs does not retrofit those already-created connections, so an idle
    worker would otherwise quit every five seconds.  Disconnecting immediately
    after RQ configures the pool discards only those stale pooled connections;
    subsequent connections inherit RQ's longer timeout.
    """

    worker = worker_class([queue], connection=queue.connection)  # type: ignore[attr-defined]
    pool = worker.connection.connection_pool  # type: ignore[attr-defined]
    pool.disconnect()
    # ``disconnect()`` closes sockets but retains the Connection objects and
    # their old per-connection timeout. ``reset()`` makes the next checkout
    # construct a new Connection from RQ's updated pool kwargs.
    pool.reset()
    return worker


def main() -> None:
    role = os.getenv("SIMULATION_WORKER_ROLE", "worker").strip().lower()
    if role not in {"worker", "reconciler"}:
        raise ValueError("SIMULATION_WORKER_ROLE must be worker or reconciler")
    registry = _prepare_multiprocess_metrics()
    try:
        from prometheus_client import start_http_server
        from rq import SimpleWorker, SpawnWorker
    except ImportError as exc:  # pragma: no cover - production dependency guard
        raise RuntimeError(
            "Simulation worker dependencies are not installed; run pip install -r requirements.txt."
        ) from exc

    settings = get_settings()
    # Import after PROMETHEUS_MULTIPROC_DIR is configured so work-horse
    # processes write counters into the shared collector directory.
    from agi_ecom_app.decision_simulation.jobs import (
        schedule_decision_simulation_reconciliation,
    )
    from agi_ecom_app.decision_simulation.service import _build_rq_queue
    from agi_ecom_app.services.memory import (
        close_memory_connection_pool,
        ensure_memory_schema,
    )

    # Run DDL once in the parent. Spawned RQ work horses receive an explicit
    # ready marker and create their own process-local connection pool.
    ensure_memory_schema()
    os.environ["MEMORY_SCHEMA_PREINITIALIZED"] = "1"
    close_memory_connection_pool()

    queue_name = (
        settings.decision_simulation_maintenance_queue_name
        if role == "reconciler"
        else settings.decision_simulation_queue_name
    )
    queue = _build_rq_queue(queue_name)
    metrics_port = int(
        os.getenv(
            "DECISION_SIMULATION_METRICS_PORT",
            "9101" if role == "worker" else "9102",
        )
    )
    worker_class = (
        SimpleWorker
        if os.getenv("SIMULATION_WORKER_MODE", "worker").strip().lower() == "simple"
        else SpawnWorker
    )
    close_memory_connection_pool()
    # Construct the worker before the observer or reconciler performs the first
    # Redis command.  RQ must configure blocking-dequeue timeouts first.
    worker = _build_worker(worker_class, queue)
    start_http_server(metrics_port, registry=registry)
    if role == "worker":
        _start_queue_depth_observer(queue)
    elif role == "reconciler":
        schedule_decision_simulation_reconciliation(queue=queue)
        _start_reconciler_watchdog(
            queue,
            schedule_decision_simulation_reconciliation,
        )
    worker.work(with_scheduler=True)


if __name__ == "__main__":
    main()
