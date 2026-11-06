from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from agi_ecom_app.config import get_settings


logger = logging.getLogger(__name__)

DAILY_MANAGER_QUERY = (
    "Сформируй единый план действий ИИ-менеджера на сегодня по продажам, финансам, рекламе, "
    "ценам и остаткам. Выбери 3–7 действий по нижней границе добавочной "
    "маржинальной прибыли. Если обязательных свежих данных нет, заблокируй действие "
    "и явно укажи недостающий источник."
)


def next_manager_daily_run(
    *,
    now: datetime | None = None,
    hour: int = 9,
    timezone_name: str = "Europe/Moscow",
) -> datetime:
    zone = ZoneInfo(timezone_name)
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    local = current.astimezone(zone)
    target = local.replace(hour=max(0, min(23, hour)), minute=0, second=0, microsecond=0)
    if target <= local:
        target += timedelta(days=1)
    return target.astimezone(timezone.utc)


def schedule_manager_daily_plan(*, queue: Any | None = None, now: datetime | None = None) -> str | None:
    settings = get_settings()
    if not (
        getattr(settings, "manager_daily_enabled", False)
        and getattr(settings, "manager_decision_enabled", False)
    ):
        return None
    if getattr(settings, "agent_workflow_version", "v1") != "v2_optimized":
        logger.error("manager_daily_not_scheduled workflow_version_must_be_v2_optimized")
        return None

    if queue is None:
        from agi_ecom_app.agent_queue import _build_rq_queue

        queue = _build_rq_queue(settings.redis_url, settings.agent_queue_name)
    run_at = next_manager_daily_run(
        now=now,
        hour=getattr(settings, "manager_daily_hour", 9),
        timezone_name=getattr(settings, "manager_daily_timezone", "Europe/Moscow"),
    )
    local_date = run_at.astimezone(
        ZoneInfo(getattr(settings, "manager_daily_timezone", "Europe/Moscow"))
    ).date()
    job_id = f"manager-daily-trigger-{local_date.isoformat()}"
    try:
        queue.enqueue_at(
            run_at,
            run_manager_daily_plan_job,
            job_id=job_id,
            result_ttl=86400,
            failure_ttl=86400,
        )
    except Exception as exc:
        if "already exists" not in str(exc).casefold():
            raise
        logger.info("manager_daily_already_scheduled job_id=%s", job_id)
    return job_id


def run_manager_daily_plan_job() -> dict[str, Any]:
    settings = get_settings()
    if not (
        getattr(settings, "manager_daily_enabled", False)
        and getattr(settings, "manager_decision_enabled", False)
        and getattr(settings, "agent_workflow_version", "v1") == "v2_optimized"
    ):
        return {"status": "disabled"}

    from agi_ecom_app.agent_queue import enqueue_initial_run

    timezone_name = getattr(settings, "manager_daily_timezone", "Europe/Moscow")
    local_date = datetime.now(timezone.utc).astimezone(ZoneInfo(timezone_name)).date()
    try:
        result = enqueue_initial_run(
            chat_id=getattr(settings, "manager_daily_chat_id", "manager-daily"),
            user_query=DAILY_MANAGER_QUERY,
            client_message_id=f"manager-daily:{local_date.isoformat()}",
        )
    finally:
        # A transient failure today must not silently disable tomorrow's plan.
        schedule_manager_daily_plan()
    return {
        "status": result.status,
        "run_id": result.run_id,
        "chat_id": result.chat_id,
        "date": local_date.isoformat(),
    }


__all__ = [
    "DAILY_MANAGER_QUERY",
    "next_manager_daily_run",
    "run_manager_daily_plan_job",
    "schedule_manager_daily_plan",
]
