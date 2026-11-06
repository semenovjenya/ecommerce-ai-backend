from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

from agi_ecom_app.experiment_program import jobs
from agi_ecom_app.experiment_program.repository import ExperimentOutboxRecord


NOW = datetime(2026, 7, 20, 10, 0, tzinfo=timezone.utc)


def _record() -> ExperimentOutboxRecord:
    return ExperimentOutboxRecord(
        id=1,
        event_id="program-1:created",
        program_id="program-1",
        topic="experiment.program.events",
        message_key="program-1:created",
        payload={
            "eventId": "program-1:created",
            "programId": "program-1",
            "eventType": "created",
            "epoch": 1,
            "payload": {},
        },
        status="processing",
        attempt_count=1,
        max_attempts=10,
        available_at=NOW,
        claimed_at=NOW,
        lease_owner="worker-1",
        lease_expires_at=NOW,
        delivered_at=None,
        last_error=None,
        created_at=NOW,
        updated_at=NOW,
    )


def test_dispatcher_enqueues_deterministic_job_and_marks_delivery(monkeypatch) -> None:
    queue = Mock()
    queue.fetch_job.return_value = None
    monkeypatch.setattr(
        jobs,
        "get_settings",
        lambda: SimpleNamespace(experiment_program_outbox_lease_seconds=60),
    )
    monkeypatch.setattr(jobs, "claim_program_outbox", lambda **_: [_record()])
    delivered = Mock(return_value=True)
    monkeypatch.setattr(jobs, "mark_program_outbox_delivered", delivered)

    result = jobs.dispatch_program_outbox_once(
        queue=queue,
        lease_owner="worker-1",
    )

    assert result == {"claimed": 1, "delivered": 1, "retried": 0, "dead": 0}
    job_id = queue.enqueue_call.call_args.kwargs["job_id"]
    assert job_id.startswith("experiment-program-event-")
    assert len(job_id) == len("experiment-program-event-") + 64
    assert all(character.isalnum() or character in "-_" for character in job_id)
    delivered.assert_called_once_with(outbox_id=1, lease_owner="worker-1")


def test_duplicate_rq_job_is_treated_as_delivered_without_second_enqueue(monkeypatch) -> None:
    queue = Mock()
    queue.fetch_job.return_value = object()
    monkeypatch.setattr(
        jobs,
        "get_settings",
        lambda: SimpleNamespace(experiment_program_outbox_lease_seconds=60),
    )
    monkeypatch.setattr(jobs, "claim_program_outbox", lambda **_: [_record()])
    monkeypatch.setattr(jobs, "mark_program_outbox_delivered", lambda **_: True)

    result = jobs.dispatch_program_outbox_once(queue=queue, lease_owner="worker-1")

    assert result["delivered"] == 1
    queue.enqueue_call.assert_not_called()


def test_failed_deterministic_job_is_requeued_instead_of_being_lost(monkeypatch) -> None:
    queue = Mock()
    failed_job = Mock()
    failed_job.get_status.return_value = "failed"
    queue.fetch_job.return_value = failed_job
    monkeypatch.setattr(
        jobs,
        "get_settings",
        lambda: SimpleNamespace(experiment_program_outbox_lease_seconds=60),
    )
    monkeypatch.setattr(jobs, "claim_program_outbox", lambda **_: [_record()])
    monkeypatch.setattr(jobs, "mark_program_outbox_delivered", lambda **_: True)

    result = jobs.dispatch_program_outbox_once(queue=queue, lease_owner="worker-1")

    assert result["delivered"] == 1
    failed_job.requeue.assert_called_once_with()
    queue.enqueue_call.assert_not_called()
