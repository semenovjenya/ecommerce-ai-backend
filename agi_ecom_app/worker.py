from __future__ import annotations

import os
from pathlib import Path


def _prepare_agent_worker_metrics() -> object:
    """Create the worker-local multiprocess registry before app imports.

    RQ's regular ``Worker`` executes jobs in forked work horses. The directory
    is deliberately private to the agent-worker container; it is never shared
    with decision-simulation workers.
    """

    metrics_dir = Path(
        os.getenv(
            "AGENT_WORKER_PROMETHEUS_MULTIPROC_DIR",
            "/tmp/agent-worker-prometheus",
        )
    )
    metrics_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PROMETHEUS_MULTIPROC_DIR"] = str(metrics_dir)
    for stale_file in metrics_dir.glob("*.db"):
        stale_file.unlink(missing_ok=True)

    from prometheus_client import CollectorRegistry, multiprocess

    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
    return registry


def _start_agent_worker_metrics(registry: object) -> int:
    from prometheus_client import start_http_server
    from prometheus_client.core import GaugeMetricFamily

    class AgentWorkerHealthCollector:
        def collect(self):
            metric = GaugeMetricFamily(
                "agent_worker_up",
                "Whether the manager/agent RQ worker process completed startup.",
            )
            metric.add_metric([], 1.0)
            yield metric

    port = int(os.getenv("AGENT_WORKER_METRICS_PORT", "9103"))
    registry.register(AgentWorkerHealthCollector())
    start_http_server(port, registry=registry)
    return port


def _ensure_decision_simulation_marts(
    mode: str,
    *,
    pricing_mode: str = "off",
    apply_fn: object | None = None,
) -> dict[str, object] | None:
    """Deploy the analytics contract before a simulation-capable worker starts."""

    if mode == "off" and pricing_mode == "off":
        return None
    if apply_fn is None:
        from scripts.apply_agent_marts import apply_agent_marts

        apply_fn = apply_agent_marts
    if not callable(apply_fn):
        raise TypeError("apply_fn must be callable")
    return apply_fn(schema="agent_marts", commit=True)


def _agent_worker_concurrency() -> int:
    raw_value = os.getenv("AGENT_WORKER_CONCURRENCY", "4")
    try:
        return max(1, int(raw_value))
    except ValueError:
        return 4


def main() -> None:
    registry = _prepare_agent_worker_metrics()
    try:
        from rq import SimpleWorker, Worker
    except ImportError as exc:
        raise RuntimeError(
            "Redis/RQ зависимости не установлены. Выполни `pip install -r requirements.txt`."
        ) from exc

    from agi_ecom_app.agent_queue import (
        _build_rq_queue,
        schedule_agent_question_reconciliation,
    )
    from agi_ecom_app.config import (
        get_settings,
        normalize_decision_simulation_mode,
    )

    settings = get_settings()
    pricing_mode = normalize_decision_simulation_mode(
        os.getenv("DECISION_SIMULATION_PRICING_MODE", "off")
    )
    _ensure_decision_simulation_marts(
        settings.decision_simulation_mode,
        pricing_mode=pricing_mode,
    )
    queue = _build_rq_queue(settings.redis_url, settings.agent_queue_name)
    from agi_ecom_app.manager_daily import schedule_manager_daily_plan
    from agi_ecom_app.manager_lifecycle.advertising_measurement import (
        register_advertising_measurement_provider,
    )
    from agi_ecom_app.manager_lifecycle.pricing_measurement import (
        register_pricing_measurement_provider,
    )
    from agi_ecom_app.manager_lifecycle.jobs import (
        schedule_manager_lifecycle_reconciliation,
    )
    from agi_ecom_app.services.memory import (
        close_memory_connection_pool,
        ensure_memory_schema,
    )

    ensure_memory_schema()
    os.environ["MEMORY_SCHEMA_PREINITIALIZED"] = "1"
    close_memory_connection_pool()
    _start_agent_worker_metrics(registry)
    register_advertising_measurement_provider()
    register_pricing_measurement_provider()

    schedule_manager_daily_plan(queue=queue)
    schedule_manager_lifecycle_reconciliation(queue=queue)
    schedule_agent_question_reconciliation(queue=queue)
    close_memory_connection_pool()
    worker_class = SimpleWorker if os.getenv("AGENT_WORKER_MODE", "").lower() == "simple" else Worker
    concurrency = _agent_worker_concurrency()
    if concurrency == 1:
        worker = worker_class([queue], connection=queue.connection)
        worker.work(with_scheduler=True)
        return

    from rq.worker_pool import WorkerPool

    pool = WorkerPool(
        [queue],
        connection=queue.connection,
        num_workers=concurrency,
        worker_class=worker_class,
    )
    pool.start()


if __name__ == "__main__":
    main()
