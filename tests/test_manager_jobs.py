from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

from rq.job import validate_job_id

from agi_ecom_app.manager_lifecycle.jobs import (
    ManagerMeasurementReadout,
    run_manager_action_measurement_job,
    schedule_manager_action_measurement,
    schedule_manager_action_verification,
    schedule_manager_lifecycle_reconciliation,
)
from tests.test_manager_lifecycle import _experiment


NOW = datetime(2026, 7, 30, 9, 0, tzinfo=timezone.utc)


class Queue:
    def __init__(self) -> None:
        self.calls = []

    def enqueue_in(self, *args, **kwargs):
        self.calls.append(("in", args, kwargs))

    def enqueue_at(self, *args, **kwargs):
        self.calls.append(("at", args, kwargs))


def test_async_job_ids_are_stable_and_versioned() -> None:
    queue = Queue()
    verification_id = schedule_manager_action_verification(
        "experiment-1",
        version=4,
        delay_seconds=30,
        queue=queue,
    )
    measurement_id = schedule_manager_action_measurement(
        "experiment-1",
        version=5,
        due_at=NOW,
        queue=queue,
    )

    assert verification_id == "manager-action-verify-experiment-1-v4"
    assert measurement_id.endswith(f"-{int(NOW.timestamp())}")
    validate_job_id(verification_id)
    validate_job_id(measurement_id)
    assert queue.calls[0][2]["job_id"] == verification_id
    assert queue.calls[1][2]["job_id"] == measurement_id
    assert "expected_version" not in queue.calls[0][2]
    assert queue.calls[1][2]["expected_version"] == 5


def test_reconciliation_job_id_satisfies_real_rq_contract() -> None:
    queue = Queue()
    settings = SimpleNamespace(manager_decision_enabled=True)
    with patch(
        "agi_ecom_app.manager_lifecycle.jobs.get_settings",
        return_value=settings,
    ):
        job_id = schedule_manager_lifecycle_reconciliation(queue=queue, due_at=NOW)

    assert job_id == f"manager-lifecycle-reconcile-{int(NOW.timestamp()) // 300}"
    validate_job_id(job_id)
    assert queue.calls[0][2]["job_id"] == job_id


def test_measurement_job_uses_provider_and_deterministic_readout() -> None:
    experiment = replace(
        _experiment(state="measuring"),
        applied_at=NOW - timedelta(days=7),
        measurement_due_at=NOW,
    )
    measured = replace(experiment, state="measured", version=experiment.version + 1)
    claimed = replace(
        experiment,
        state="measuring",
        version=experiment.version + 1,
        measurement_due_at=NOW + timedelta(minutes=30),
    )
    collected = {}

    class Provider:
        def collect(self, action, **windows):
            collected.update(windows)
            return ManagerMeasurementReadout(
                data_complete=True,
                data_as_of=NOW,
                baseline_treated=100,
                post_treated=130,
                baseline_control=80,
                post_control=90,
                metrics={"source": "test-mart"},
            )

    measurement = SimpleNamespace(
        id=11,
        status="computed",
        causal_valid=True,
    )
    with (
        patch.dict(
            "agi_ecom_app.manager_lifecycle.jobs._MEASUREMENT_PROVIDERS",
            {"advertising": Provider()},
            clear=True,
        ),
        patch(
            "agi_ecom_app.manager_lifecycle.service.get_manager_action",
            side_effect=[experiment, measured],
        ),
        patch(
            "agi_ecom_app.manager_lifecycle.service.record_manager_action_measurement",
            return_value=measurement,
        ) as record,
        patch(
            "agi_ecom_app.manager_lifecycle.jobs.transition_agent_action_experiment",
            return_value=(claimed, True),
        ),
    ):
        result = run_manager_action_measurement_job(experiment.id, now=NOW)

    assert result["status"] == "computed"
    assert result["causal_valid"] is True
    assert collected["baseline_start"] == datetime(
        2026, 7, 16, tzinfo=timezone.utc
    )
    assert collected["baseline_end"] == datetime(
        2026, 7, 23, tzinfo=timezone.utc
    )
    assert collected["evaluation_start"] == datetime(
        2026, 7, 24, tzinfo=timezone.utc
    )
    assert collected["evaluation_end"] == datetime(
        2026, 7, 31, tzinfo=timezone.utc
    )
    assert record.call_args.kwargs["evaluation_start"] == collected[
        "evaluation_start"
    ]
    assert record.call_args.kwargs["evaluation_end"] == collected["evaluation_end"]
    assert record.call_args.kwargs["baseline_control"] == 80
    assert record.call_args.kwargs["post_control"] == 90
    assert record.call_args.kwargs["idempotency_key"].startswith(
        "manager-measurement:"
    )


def test_measurement_retry_does_not_extend_the_business_evaluation_window() -> None:
    fixed_evaluation_end = datetime(2026, 7, 31, tzinfo=timezone.utc)
    retry_due_at = fixed_evaluation_end + timedelta(hours=6)
    experiment = replace(
        _experiment(state="measuring"),
        applied_at=NOW - timedelta(days=7),
        measurement_due_at=retry_due_at,
    )
    measured = replace(experiment, state="measured", version=experiment.version + 1)
    claimed = replace(
        experiment,
        state="measuring",
        version=experiment.version + 1,
        measurement_due_at=retry_due_at + timedelta(minutes=30),
    )
    collected = {}

    class Provider:
        def collect(self, action, **windows):
            collected.update(windows)
            return ManagerMeasurementReadout(
                data_complete=True,
                data_as_of=retry_due_at,
                baseline_treated=100,
                post_treated=130,
            )

    measurement = SimpleNamespace(id=14, status="computed", causal_valid=False)
    with (
        patch.dict(
            "agi_ecom_app.manager_lifecycle.jobs._MEASUREMENT_PROVIDERS",
            {"advertising": Provider()},
            clear=True,
        ),
        patch(
            "agi_ecom_app.manager_lifecycle.service.get_manager_action",
            side_effect=[experiment, measured],
        ),
        patch(
            "agi_ecom_app.manager_lifecycle.service.record_manager_action_measurement",
            return_value=measurement,
        ) as record,
        patch(
            "agi_ecom_app.manager_lifecycle.jobs.transition_agent_action_experiment",
            return_value=(claimed, True),
        ),
    ):
        run_manager_action_measurement_job(experiment.id, now=retry_due_at)

    assert collected["evaluation_end"] == fixed_evaluation_end
    assert record.call_args.kwargs["evaluation_end"] == fixed_evaluation_end


def test_missing_measurement_provider_waits_and_requeues_without_zeroes() -> None:
    experiment = replace(
        _experiment(state="measuring"),
        applied_at=NOW - timedelta(days=7),
        measurement_due_at=NOW,
    )
    measurement = SimpleNamespace(
        id=12,
        status="waiting_for_data",
        causal_valid=False,
    )
    retry_at = NOW + timedelta(hours=6)
    claimed = replace(
        experiment,
        state="measuring",
        version=experiment.version + 1,
        measurement_due_at=NOW + timedelta(minutes=30),
    )
    refreshed = replace(
        claimed,
        state="measuring",
        version=claimed.version + 1,
        measurement_due_at=retry_at,
    )
    with (
        patch.dict(
            "agi_ecom_app.manager_lifecycle.jobs._MEASUREMENT_PROVIDERS",
            {},
            clear=True,
        ),
        patch(
            "agi_ecom_app.manager_lifecycle.service.get_manager_action",
            side_effect=[experiment, refreshed],
        ),
        patch(
            "agi_ecom_app.manager_lifecycle.service.record_manager_action_measurement",
            return_value=measurement,
        ) as record,
        patch(
            "agi_ecom_app.manager_lifecycle.jobs.schedule_manager_action_measurement"
        ) as requeue,
        patch(
            "agi_ecom_app.manager_lifecycle.jobs.transition_agent_action_experiment",
            side_effect=[(claimed, True), (refreshed, True)],
        ) as transition,
    ):
        result = run_manager_action_measurement_job(experiment.id, now=NOW)

    assert result["status"] == "waiting_for_data"
    assert record.call_args.kwargs["baseline_treated"] is None
    assert record.call_args.kwargs["post_treated"] is None
    assert record.call_args.kwargs["data_complete"] is False
    assert transition.call_args_list[0].kwargs == {
        "experiment_id": experiment.id,
        "expected_states": {"applied", "measuring"},
        "state": "measuring",
        "event": "measurement_attempt_claimed",
        "measurement_due_at": NOW + timedelta(minutes=30),
        "expected_version": experiment.version,
    }
    assert transition.call_args_list[1].kwargs == {
        "experiment_id": experiment.id,
        "expected_states": {"applied", "measuring"},
        "state": "measuring",
        "event": "measurement_retry_scheduled",
        "measurement_due_at": retry_at,
    }
    assert requeue.call_args.kwargs["version"] == refreshed.version
    assert requeue.call_args.kwargs["due_at"] == retry_at


def test_terminal_provider_block_blocks_experiment_without_requeue() -> None:
    experiment = replace(
        _experiment(state="measuring"),
        applied_at=NOW - timedelta(days=7),
        measurement_due_at=NOW,
    )
    claimed = replace(
        experiment,
        state="measuring",
        version=experiment.version + 1,
        measurement_due_at=NOW + timedelta(minutes=30),
    )
    blocked = replace(
        claimed,
        state="blocked",
        version=claimed.version + 1,
        last_error="missing_target_sku",
    )

    class Provider:
        def collect(self, action, **windows):
            return ManagerMeasurementReadout(
                data_complete=False,
                metrics={
                    "reason": "missing_target_sku",
                    "disposition": "blocked",
                },
                disposition="blocked",
            )

    measurement = SimpleNamespace(
        id=13,
        status="waiting_for_data",
        causal_valid=False,
    )
    with (
        patch.dict(
            "agi_ecom_app.manager_lifecycle.jobs._MEASUREMENT_PROVIDERS",
            {"advertising": Provider()},
            clear=True,
        ),
        patch(
            "agi_ecom_app.manager_lifecycle.service.get_manager_action",
            return_value=experiment,
        ),
        patch(
            "agi_ecom_app.manager_lifecycle.service.record_manager_action_measurement",
            return_value=measurement,
        ),
        patch(
            "agi_ecom_app.manager_lifecycle.jobs.transition_agent_action_experiment",
            side_effect=[(claimed, True), (blocked, True)],
        ) as transition,
        patch(
            "agi_ecom_app.manager_lifecycle.jobs.schedule_manager_action_measurement"
        ) as requeue,
    ):
        result = run_manager_action_measurement_job(experiment.id, now=NOW)

    assert result["status"] == "blocked"
    assert result["state"] == "blocked"
    assert transition.call_count == 2
    assert transition.call_args_list[1].kwargs == {
        "experiment_id": experiment.id,
        "expected_states": {"applied", "measuring"},
        "state": "blocked",
        "event": "measurement_blocked",
        "last_error": "missing_target_sku",
    }
    requeue.assert_not_called()


def test_stale_measurement_job_is_ignored_before_provider_or_write() -> None:
    experiment = replace(
        _experiment(state="measuring"),
        applied_at=NOW - timedelta(days=7),
        version=9,
        measurement_due_at=NOW,
    )
    with (
        patch(
            "agi_ecom_app.manager_lifecycle.service.get_manager_action",
            return_value=experiment,
        ),
        patch(
            "agi_ecom_app.manager_lifecycle.jobs.transition_agent_action_experiment"
        ) as transition,
        patch(
            "agi_ecom_app.manager_lifecycle.service.record_manager_action_measurement"
        ) as record,
    ):
        result = run_manager_action_measurement_job(
            experiment.id,
            now=NOW,
            expected_version=8,
        )

    assert result["status"] == "ignored"
    assert result["reason"] == "stale_measurement_attempt"
    transition.assert_not_called()
    record.assert_not_called()


def test_lost_claim_is_ignored_without_appending_a_readout() -> None:
    experiment = replace(
        _experiment(state="measuring"),
        applied_at=NOW - timedelta(days=7),
        measurement_due_at=NOW,
    )
    already_claimed = replace(experiment, version=experiment.version + 1)
    with (
        patch(
            "agi_ecom_app.manager_lifecycle.service.get_manager_action",
            return_value=experiment,
        ),
        patch(
            "agi_ecom_app.manager_lifecycle.jobs.transition_agent_action_experiment",
            return_value=(already_claimed, False),
        ),
        patch(
            "agi_ecom_app.manager_lifecycle.service.record_manager_action_measurement"
        ) as record,
    ):
        result = run_manager_action_measurement_job(
            experiment.id,
            now=NOW,
            expected_version=experiment.version,
        )

    assert result["status"] == "ignored"
    assert result["reason"] == "measurement_attempt_already_claimed"
    record.assert_not_called()
