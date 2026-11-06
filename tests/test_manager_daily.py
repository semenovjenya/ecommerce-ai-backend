from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

from rq.job import validate_job_id

from agi_ecom_app.manager_daily import (
    DAILY_MANAGER_QUERY,
    next_manager_daily_run,
    schedule_manager_daily_plan,
)
from agi_ecom_app.prompt_pipeline import determine_query_mode


def test_next_daily_run_uses_moscow_wall_clock():
    before = datetime(2026, 7, 13, 5, 0, tzinfo=timezone.utc)
    assert next_manager_daily_run(now=before).isoformat() == "2026-07-13T06:00:00+00:00"
    after = datetime(2026, 7, 13, 7, 0, tzinfo=timezone.utc)
    assert next_manager_daily_run(now=after).isoformat() == "2026-07-14T06:00:00+00:00"


def test_daily_query_is_routed_as_manager_decision():
    assert determine_query_mode(DAILY_MANAGER_QUERY).value == "decision"


def test_daily_schedule_is_fail_closed_outside_v2_optimized():
    settings = SimpleNamespace(
        manager_daily_enabled=True,
        manager_decision_enabled=True,
        agent_workflow_version="v1",
    )
    with patch("agi_ecom_app.manager_daily.get_settings", return_value=settings):
        assert schedule_manager_daily_plan(queue=object()) is None


def test_daily_schedule_uses_stable_per_day_job_id():
    class Queue:
        def __init__(self):
            self.call = None

        def enqueue_at(self, *args, **kwargs):
            self.call = (args, kwargs)

    settings = SimpleNamespace(
        manager_daily_enabled=True,
        manager_decision_enabled=True,
        agent_workflow_version="v2_optimized",
        manager_daily_hour=9,
        manager_daily_timezone="Europe/Moscow",
    )
    queue = Queue()
    now = datetime(2026, 7, 13, 5, 0, tzinfo=timezone.utc)
    with patch("agi_ecom_app.manager_daily.get_settings", return_value=settings):
        job_id = schedule_manager_daily_plan(queue=queue, now=now)
    assert job_id == "manager-daily-trigger-2026-07-13"
    validate_job_id(job_id)
    assert queue.call[1]["job_id"] == job_id
