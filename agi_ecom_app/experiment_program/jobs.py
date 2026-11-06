from __future__ import annotations

import json
import hashlib
import logging
import socket
import threading
from datetime import datetime, timedelta, timezone
from typing import Any

import psycopg
from redis import Redis
from rq import Queue, Retry

from agi_ecom_app.config import get_settings
from agi_ecom_app.experiment_program.executor_contract import (
    ProgramSimulationLaunch,
    ProgramSimulationUnsupported,
)
from agi_ecom_app.experiment_program.repository import (
    ExperimentOutboxRecord,
    claim_program_outbox,
    fail_program_simulation_dispatch,
    get_program_bundle,
    list_linked_simulation_runs,
    mark_program_outbox_delivered,
    record_program_simulation_dispatch,
    requeue_stale_program_simulations,
    reschedule_program_outbox,
    sync_linked_simulation_run,
)


logger = logging.getLogger(__name__)
_PROGRAM_EVENTS_CHANNEL = "experiment_program_events"


def build_experiment_program_queue() -> Queue:
    settings = get_settings()
    return Queue(
        name=settings.experiment_program_queue_name,
        connection=Redis.from_url(settings.redis_url),
        default_timeout=settings.experiment_program_job_timeout_seconds,
    )


def process_experiment_program_event_job(
    *,
    event_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Idempotent RQ consumer boundary for durable program events.

    This worker only coordinates delivery. NumPy still runs in the isolated
    ``decision-simulations`` queue; the program worker resolves a durable run,
    enqueues it idempotently, or records a fail-closed terminal reason.
    """

    if payload.get("eventId") != event_id:
        raise ValueError("outbox event identity does not match its payload")
    event_type = payload.get("eventType")
    if not isinstance(event_type, str) or not event_type:
        raise ValueError("experiment program event has no eventType")
    program_id = payload.get("programId")
    if not isinstance(program_id, str) or not program_id:
        raise ValueError("experiment program event has no programId")
    dispatch: dict[str, Any] | None = None
    if event_type in {"simulation_requested", "simulation_reconcile_requested"}:
        command_payload = payload.get("payload")
        if not isinstance(command_payload, dict):
            command_payload = {}
        dispatch = coordinate_program_simulation(
            program_id=program_id,
            source_event_id=event_id,
            expected_epoch=_optional_int(
                command_payload.get("epoch", payload.get("epoch"))
            ),
            expected_snapshot_hash=_optional_text(
                command_payload.get("snapshotHash")
            ),
        )
    logger.info(
        "experiment_program_event_consumed program_id=%s event_id=%s event_type=%s",
        program_id,
        event_id,
        event_type,
    )
    result = {
        "status": "acknowledged",
        "programId": program_id,
        "eventId": event_id,
        "eventType": event_type,
    }
    if dispatch is not None:
        result["dispatch"] = dispatch
    return result


def coordinate_program_simulation(
    *,
    program_id: str,
    source_event_id: str,
    executor: Any | None = None,
    expected_epoch: int | None = None,
    expected_snapshot_hash: str | None = None,
    launcher: Any | None = None,
) -> dict[str, Any]:
    """Resolve and deliver every durable simulation already owned by a program."""

    bundle = get_program_bundle(program_id)
    if bundle is None:
        raise LookupError(program_id)
    program = bundle.program
    if (
        (expected_epoch is not None and program["current_epoch"] != expected_epoch)
        or (
            expected_snapshot_hash is not None
            and program["snapshot_hash"] != expected_snapshot_hash
        )
    ):
        return {
            "status": "stale_command",
            "programEpoch": program["current_epoch"],
            "snapshotHash": program["snapshot_hash"],
            "simulationRunIds": [],
        }
    if program["status"] != "simulating":
        return {
            "status": "already_terminal",
            "programStatus": program["status"],
            "simulationRunIds": [],
        }

    effective_epoch = int(program["current_epoch"])
    effective_snapshot_hash = str(program["snapshot_hash"])
    runs = list_linked_simulation_runs(program_id)
    launch: ProgramSimulationLaunch | None = None
    if not runs:
        try:
            launch_callable = launcher or _enqueue_supported_program_simulation
            launch = launch_callable(
                program_id=program_id,
                source_event_id=source_event_id,
                expected_epoch=effective_epoch,
                expected_snapshot_hash=effective_snapshot_hash,
            )
        except ProgramSimulationUnsupported as exc:
            code = exc.code
            message = exc.message
            details = exc.details
        else:
            runs = list_linked_simulation_runs(program_id)
            if not runs:
                record_program_simulation_dispatch(
                    program_id=program_id,
                    source_event_id=source_event_id,
                    simulation_run_ids=[],
                    expected_epoch=effective_epoch,
                    expected_snapshot_hash=effective_snapshot_hash,
                    details={
                        "executor": launch.executor_name,
                        **launch.details,
                    },
                )
                return {
                    "status": "launch_queued",
                    "executor": launch.executor_name,
                    "simulationRunIds": [],
                    **launch.details,
                }
        if not runs:
            changed = fail_program_simulation_dispatch(
                program_id=program_id,
                source_event_id=source_event_id,
                code=code,
                message=message,
                details=details,
                expected_epoch=effective_epoch,
                expected_snapshot_hash=effective_snapshot_hash,
            )
            return {
                "status": "failed_closed" if changed else "already_terminal",
                "code": code,
                "simulationRunIds": [],
            }

    target_executor = executor
    if target_executor is None:
        from agi_ecom_app.decision_simulation.service import RQSimulationExecutor

        target_executor = RQSimulationExecutor()
    delivered: list[str] = []
    observed: list[dict[str, str]] = []
    supported_statuses = {
        "initializing",
        "queued",
        "running",
        "critic_running",
        "completed",
        "blocked",
        "failed",
        "superseded",
    }
    for run in runs:
        run_id = str(run["id"])
        status = str(run["status"])
        if status not in supported_statuses:
            raise ValueError(f"unsupported decision simulation status: {status}")
        if status == "queued":
            target_executor.enqueue(run_id)
            delivered.append(run_id)
        elif status in {"completed", "blocked", "failed", "superseded"}:
            sync_linked_simulation_run(run_id)
        observed.append({"simulationRunId": run_id, "status": status})

    run_ids = [item["simulationRunId"] for item in observed]
    record_program_simulation_dispatch(
        program_id=program_id,
        source_event_id=source_event_id,
        simulation_run_ids=run_ids,
        expected_epoch=effective_epoch,
        expected_snapshot_hash=effective_snapshot_hash,
        details=(
            {"executor": launch.executor_name, **launch.details}
            if launch is not None
            else {}
        ),
    )
    return {
        "status": "dispatched" if delivered else "observed",
        "simulationRunIds": run_ids,
        "enqueuedRunIds": delivered,
        "runs": observed,
    }


def dispatch_program_outbox_once(
    *,
    queue: Queue | None = None,
    lease_owner: str | None = None,
    limit: int = 100,
) -> dict[str, int]:
    settings = get_settings()
    target_queue = queue or build_experiment_program_queue()
    owner = lease_owner or f"{socket.gethostname()}:{threading.get_ident()}"
    records = claim_program_outbox(
        lease_owner=owner,
        limit=limit,
        lease_seconds=settings.experiment_program_outbox_lease_seconds,
    )
    delivered = 0
    retried = 0
    dead = 0
    for record in records:
        try:
            _enqueue_outbox_record(target_queue, record)
            if not mark_program_outbox_delivered(
                outbox_id=record.id,
                lease_owner=owner,
            ):
                raise RuntimeError("outbox lease was lost after RQ delivery")
            delivered += 1
        except Exception as exc:
            logger.exception(
                "experiment_program_outbox_delivery_failed program_id=%s event_id=%s",
                record.program_id,
                record.event_id,
            )
            status = reschedule_program_outbox(
                outbox_id=record.id,
                lease_owner=owner,
                error=str(exc),
                delay_seconds=min(300, 2 ** min(record.attempt_count, 8)),
            )
            if status == "dead":
                dead += 1
            elif status == "pending":
                retried += 1
    return {
        "claimed": len(records),
        "delivered": delivered,
        "retried": retried,
        "dead": dead,
    }


def _enqueue_outbox_record(queue: Queue, record: ExperimentOutboxRecord) -> None:
    job_id = _outbox_job_id(record.event_id)
    existing = queue.fetch_job(job_id)
    if existing is not None:
        if hasattr(existing, "get_status"):
            raw_status = existing.get_status(refresh=True)
            status = str(getattr(raw_status, "value", raw_status)).lower()
            if status == "failed" and hasattr(existing, "requeue"):
                existing.requeue()
        return
    queue.enqueue_call(
        process_experiment_program_event_job,
        kwargs={"event_id": record.event_id, "payload": record.payload},
        job_id=job_id,
        retry=Retry(max=3, interval=[5, 30, 120]),
        result_ttl=86_400,
        failure_ttl=604_800,
    )


def _outbox_job_id(event_id: str) -> str:
    digest = hashlib.sha256(event_id.encode("utf-8")).hexdigest()
    return f"experiment-program-event-{digest}"


def _enqueue_supported_program_simulation(**kwargs: Any) -> ProgramSimulationLaunch:
    from agi_ecom_app.experiment_program.pricing_launcher import (
        enqueue_supported_program_simulation,
    )

    return enqueue_supported_program_simulation(**kwargs)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("experiment program epoch must be an integer")
    return int(value)


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("experiment program snapshotHash must be text")
    return value


def run_program_outbox_dispatcher(
    *,
    stop_event: threading.Event,
    queue: Queue | None = None,
) -> None:
    """Wake on Postgres notifications; timed scans only recover missed hints."""

    settings = get_settings()
    target_queue = queue or build_experiment_program_queue()
    owner = f"{socket.gethostname()}:{threading.get_ident()}"
    while not stop_event.is_set():
        connection: psycopg.Connection[Any] | None = None
        try:
            connection = psycopg.Connection.connect(
                settings.memory_postgres_dsn,
                autocommit=True,
            )
            connection.execute(f"LISTEN {_PROGRAM_EVENTS_CHANNEL}")
            # Install LISTEN before the recovery scan so commits cannot fall
            # into a SELECT-before-subscribe gap.
            _requeue_stale_program_simulations_once()
            dispatch_program_outbox_once(queue=target_queue, lease_owner=owner)
            while not stop_event.is_set():
                notification = None
                for notification in connection.notifies(
                    timeout=settings.experiment_program_outbox_recovery_seconds,
                    stop_after=1,
                ):
                    pass
                if notification is not None:
                    raw_payload = getattr(notification, "payload", "")
                    try:
                        json.loads(raw_payload)
                    except (TypeError, ValueError):
                        logger.warning("ignored malformed experiment program notification")
                else:
                    _requeue_stale_program_simulations_once()
                dispatch_program_outbox_once(queue=target_queue, lease_owner=owner)
        except Exception:
            logger.exception("experiment_program_outbox_dispatcher_reconnecting")
            stop_event.wait(1.0)
        finally:
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    logger.warning("experiment program dispatcher close failed", exc_info=True)


def _requeue_stale_program_simulations_once() -> list[str]:
    settings = get_settings()
    stale_before = datetime.now(timezone.utc) - timedelta(
        seconds=settings.experiment_program_stale_after_seconds
    )
    recovered = requeue_stale_program_simulations(
        stale_before=stale_before,
        limit=100,
        max_reconciles=settings.experiment_program_simulation_max_reconciles,
    )
    if recovered:
        logger.warning(
            "experiment_program_stale_simulations_requeued count=%s program_ids=%s",
            len(recovered),
            recovered,
        )
    return recovered


__all__ = [
    "build_experiment_program_queue",
    "coordinate_program_simulation",
    "dispatch_program_outbox_once",
    "process_experiment_program_event_job",
    "run_program_outbox_dispatcher",
]
