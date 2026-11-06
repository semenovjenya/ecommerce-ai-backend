from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Literal, Protocol
from zoneinfo import ZoneInfo

from agi_ecom_app.config import get_settings
from agi_ecom_app.services.memory import (
    AgentActionExperimentRecord,
    list_pending_agent_action_experiments,
    transition_agent_action_experiment,
)


logger = logging.getLogger(__name__)
MEASUREMENT_RETRY_DELAY = timedelta(hours=6)
MEASUREMENT_CLAIM_LEASE = timedelta(minutes=30)
RECONCILIATION_INTERVAL = timedelta(minutes=5)
_MOSCOW = ZoneInfo("Europe/Moscow")


@dataclass(frozen=True)
class ManagerMeasurementReadout:
    data_complete: bool
    data_as_of: datetime | None = None
    baseline_treated: float | None = None
    post_treated: float | None = None
    baseline_control: float | None = None
    post_control: float | None = None
    metrics: dict[str, Any] | None = None
    disposition: Literal["ready", "waiting_for_data", "blocked"] = "ready"


class ManagerMeasurementProvider(Protocol):
    def collect(
        self,
        experiment: AgentActionExperimentRecord,
        *,
        baseline_start: datetime,
        baseline_end: datetime,
        evaluation_start: datetime,
        evaluation_end: datetime,
    ) -> ManagerMeasurementReadout: ...


_MEASUREMENT_PROVIDERS: dict[str, ManagerMeasurementProvider] = {}


def register_manager_measurement_provider(
    family: str,
    provider: ManagerMeasurementProvider,
) -> None:
    normalized = family.strip().lower()
    if not normalized:
        raise ValueError("measurement provider family is required")
    _MEASUREMENT_PROVIDERS[normalized] = provider


def schedule_manager_action_verification(
    experiment_id: str,
    *,
    version: int,
    delay_seconds: int = 30,
    queue: Any | None = None,
) -> str:
    queue = queue or _manager_queue()
    job_id = f"manager-action-verify-{experiment_id}-v{int(version)}"
    _enqueue_once(
        queue.enqueue_in,
        timedelta(seconds=max(1, min(int(delay_seconds), 300))),
        run_manager_action_verification_job,
        experiment_id,
        job_id=job_id,
        result_ttl=86400,
        failure_ttl=86400,
    )
    return job_id


def run_manager_action_verification_job(experiment_id: str) -> dict[str, Any]:
    from agi_ecom_app.manager_lifecycle.service import verify_manager_action

    experiment = verify_manager_action(experiment_id)
    return {
        "experiment_id": experiment.id,
        "state": experiment.state,
        "version": experiment.version,
    }


def schedule_manager_action_measurement(
    experiment_id: str,
    *,
    version: int,
    due_at: datetime,
    queue: Any | None = None,
) -> str:
    queue = queue or _manager_queue()
    target = _utc(due_at)
    job_id = (
        f"manager-action-measure-{experiment_id}-v{int(version)}-"
        f"{int(target.timestamp())}"
    )
    _enqueue_once(
        queue.enqueue_at,
        target,
        run_manager_action_measurement_job,
        experiment_id,
        expected_version=int(version),
        job_id=job_id,
        result_ttl=86400,
        failure_ttl=86400,
    )
    return job_id


def run_manager_action_measurement_job(
    experiment_id: str,
    *,
    now: datetime | None = None,
    expected_version: int | None = None,
) -> dict[str, Any]:
    from agi_ecom_app.manager_lifecycle.service import (
        get_manager_action,
        record_manager_action_measurement,
    )

    current_time = _utc(now or datetime.now(timezone.utc))
    experiment = get_manager_action(experiment_id)
    if experiment.state not in {"applied", "measuring"}:
        return {"experiment_id": experiment.id, "state": experiment.state, "status": "ignored"}
    if experiment.applied_at is None:
        raise RuntimeError("measurable action has no applied_at timestamp")
    claim_version = (
        int(expected_version)
        if expected_version is not None
        else int(experiment.version)
    )
    if int(experiment.version) != claim_version:
        return {
            "experiment_id": experiment.id,
            "state": experiment.state,
            "status": "ignored",
            "reason": "stale_measurement_attempt",
        }
    claimed, changed = transition_agent_action_experiment(
        experiment_id=experiment.id,
        expected_states={"applied", "measuring"},
        state="measuring",
        event="measurement_attempt_claimed",
        measurement_due_at=current_time + MEASUREMENT_CLAIM_LEASE,
        expected_version=claim_version,
    )
    if claimed is None or not changed:
        return {
            "experiment_id": experiment.id,
            "state": claimed.state if claimed is not None else experiment.state,
            "status": "ignored",
            "reason": "measurement_attempt_already_claimed",
        }
    experiment = claimed

    baseline_days = _bounded_days(
        _pick(experiment.measurement_spec, "baselineDays", "baseline_days"),
        default=7 if experiment.family == "advertising" else 14,
    )
    evaluation_days = _bounded_days(
        _pick(experiment.measurement_spec, "evaluationDays", "evaluation_days"),
        default=7 if experiment.family == "advertising" else 14,
    )
    applied_at = _utc(experiment.applied_at)
    (
        baseline_start,
        baseline_end,
        evaluation_start,
        evaluation_end,
    ) = manager_measurement_windows(
        family=experiment.family,
        applied_at=applied_at,
        baseline_days=baseline_days,
        evaluation_days=evaluation_days,
    )
    provider = _MEASUREMENT_PROVIDERS.get(experiment.family.strip().lower())
    if provider is None:
        readout = ManagerMeasurementReadout(
            data_complete=False,
            metrics={"reason": "measurement_provider_not_configured"},
            disposition="waiting_for_data",
        )
    else:
        readout = provider.collect(
            experiment,
            baseline_start=baseline_start,
            baseline_end=baseline_end,
            evaluation_start=evaluation_start,
            evaluation_end=evaluation_end,
        )

    measurement = record_manager_action_measurement(
        experiment_id=experiment.id,
        evaluation_start=evaluation_start,
        evaluation_end=evaluation_end,
        data_as_of=readout.data_as_of,
        data_complete=readout.data_complete,
        baseline_treated=readout.baseline_treated,
        post_treated=readout.post_treated,
        baseline_control=readout.baseline_control,
        post_control=readout.post_control,
        metrics=readout.metrics,
        now=current_time,
        idempotency_key=_measurement_readout_idempotency_key(
            experiment_id=experiment.id,
            evaluation_start=evaluation_start,
            evaluation_end=evaluation_end,
            readout=readout,
        ),
    )
    effective_status = measurement.status
    current_experiment = None
    if readout.disposition == "blocked":
        reason = str(
            (readout.metrics or {}).get("reason")
            or "measurement_provider_blocked"
        )
        current_experiment, _ = transition_agent_action_experiment(
            experiment_id=experiment.id,
            expected_states={"applied", "measuring"},
            state="blocked",
            event="measurement_blocked",
            last_error=reason,
        )
        if current_experiment is not None and current_experiment.state == "blocked":
            effective_status = "blocked"
    elif measurement.status == "waiting_for_data":
        retry_at = current_time + MEASUREMENT_RETRY_DELAY
        current_experiment, _ = transition_agent_action_experiment(
            experiment_id=experiment.id,
            expected_states={"applied", "measuring"},
            state="measuring",
            event="measurement_retry_scheduled",
            measurement_due_at=retry_at,
        )
        if (
            current_experiment is not None
            and current_experiment.state in {"applied", "measuring"}
        ):
            schedule_manager_action_measurement(
                experiment.id,
                version=current_experiment.version,
                due_at=retry_at,
            )
    return {
        "experiment_id": experiment.id,
        "state": (current_experiment or get_manager_action(experiment.id)).state,
        "measurement_id": measurement.id,
        "status": effective_status,
        "causal_valid": measurement.causal_valid,
    }


def manager_measurement_windows(
    *,
    family: str,
    applied_at: datetime,
    baseline_days: int,
    evaluation_days: int,
) -> tuple[datetime, datetime, datetime, datetime]:
    """Return immutable business windows; retry scheduling never changes them."""

    applied_at = _utc(applied_at)
    if family.strip().lower() == "advertising":
        action_day_start = applied_at.replace(hour=0, minute=0, second=0, microsecond=0)
        baseline_end = action_day_start
        evaluation_start = action_day_start + timedelta(days=1)
        return (
            baseline_end - timedelta(days=baseline_days),
            baseline_end,
            evaluation_start,
            evaluation_start + timedelta(days=evaluation_days),
        )
    if family.strip().lower() == "pricing":
        local_applied_at = applied_at.astimezone(_MOSCOW)
        local_action_day_start = local_applied_at.replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )
        local_baseline_end = local_action_day_start
        local_evaluation_start = local_action_day_start + timedelta(days=1)
        return tuple(
            value.astimezone(timezone.utc)
            for value in (
                local_baseline_end - timedelta(days=baseline_days),
                local_baseline_end,
                local_evaluation_start,
                local_evaluation_start + timedelta(days=evaluation_days),
            )
        )
    return (
        applied_at - timedelta(days=baseline_days),
        applied_at,
        applied_at,
        applied_at + timedelta(days=evaluation_days),
    )


def _measurement_readout_idempotency_key(
    *,
    experiment_id: str,
    evaluation_start: datetime,
    evaluation_end: datetime,
    readout: ManagerMeasurementReadout,
) -> str:
    payload = {
        "experimentId": experiment_id,
        "evaluationStart": _utc(evaluation_start).isoformat(),
        "evaluationEnd": _utc(evaluation_end).isoformat(),
        "dataAsOf": _utc(readout.data_as_of).isoformat() if readout.data_as_of else None,
        "dataComplete": readout.data_complete,
        "baselineTreated": readout.baseline_treated,
        "postTreated": readout.post_treated,
        "baselineControl": readout.baseline_control,
        "postControl": readout.post_control,
        "disposition": readout.disposition,
        "metrics": readout.metrics or {},
    }
    digest = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    return f"manager-measurement:{digest}"


def schedule_manager_lifecycle_reconciliation(
    *,
    queue: Any | None = None,
    due_at: datetime | None = None,
) -> str | None:
    settings = get_settings()
    if not bool(getattr(settings, "manager_decision_enabled", False)):
        return None
    queue = queue or _manager_queue()
    target = _utc(due_at or datetime.now(timezone.utc) + RECONCILIATION_INTERVAL)
    slot = int(target.timestamp()) // int(RECONCILIATION_INTERVAL.total_seconds())
    job_id = f"manager-lifecycle-reconcile-{slot}"
    _enqueue_once(
        queue.enqueue_at,
        target,
        run_manager_lifecycle_reconciliation_job,
        job_id=job_id,
        result_ttl=3600,
        failure_ttl=86400,
    )
    return job_id


def run_manager_lifecycle_reconciliation_job() -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    verification_jobs = 0
    measurement_jobs = 0
    try:
        for experiment in list_pending_agent_action_experiments(now=now, limit=200):
            if experiment.state == "applying":
                schedule_manager_action_verification(
                    experiment.id,
                    version=experiment.version,
                    delay_seconds=1,
                )
                verification_jobs += 1
            elif experiment.state in {"applied", "measuring"}:
                schedule_manager_action_measurement(
                    experiment.id,
                    version=experiment.version,
                    due_at=now,
                )
                measurement_jobs += 1
    finally:
        schedule_manager_lifecycle_reconciliation(due_at=now + RECONCILIATION_INTERVAL)
    return {
        "verification_jobs": verification_jobs,
        "measurement_jobs": measurement_jobs,
    }


def _manager_queue():
    from agi_ecom_app.agent_queue import _build_rq_queue

    settings = get_settings()
    return _build_rq_queue(settings.redis_url, settings.agent_queue_name)


def _enqueue_once(method, *args, **kwargs) -> None:
    try:
        method(*args, **kwargs)
    except Exception as exc:
        if "already exists" not in str(exc).casefold():
            raise
        logger.info("manager_lifecycle_job_already_scheduled job_id=%s", kwargs.get("job_id"))


def _pick(value: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in value:
            return value[key]
    return None


def _bounded_days(value: Any, *, default: int) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        result = default
    return max(1, min(result, 90))


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


__all__ = [
    "ManagerMeasurementProvider",
    "ManagerMeasurementReadout",
    "manager_measurement_windows",
    "register_manager_measurement_provider",
    "run_manager_action_measurement_job",
    "run_manager_action_verification_job",
    "run_manager_lifecycle_reconciliation_job",
    "schedule_manager_action_measurement",
    "schedule_manager_action_verification",
    "schedule_manager_lifecycle_reconciliation",
]
