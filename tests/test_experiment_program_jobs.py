from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

from agi_ecom_app.experiment_program import jobs


def test_simulation_request_fails_closed_when_no_run_is_linked(monkeypatch) -> None:
    monkeypatch.setattr(
        jobs,
        "get_program_bundle",
        lambda _program_id: SimpleNamespace(
            program={
                "status": "simulating",
                "current_epoch": 1,
                "snapshot_hash": "a" * 64,
            }
        ),
    )
    monkeypatch.setattr(jobs, "list_linked_simulation_runs", lambda _program_id: [])
    fail = Mock(return_value=True)
    monkeypatch.setattr(jobs, "fail_program_simulation_dispatch", fail)

    result = jobs.coordinate_program_simulation(
        program_id="program-1",
        source_event_id="program-1:simulation-requested:1",
        launcher=Mock(
            side_effect=jobs.ProgramSimulationUnsupported(
                code="simulation_executor_not_configured",
                message="unsupported",
            )
        ),
    )

    assert result == {
        "status": "failed_closed",
        "code": "simulation_executor_not_configured",
        "simulationRunIds": [],
    }
    fail.assert_called_once()


def test_coordinator_enqueues_only_queued_runs_and_syncs_terminal_runs(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        jobs,
        "get_program_bundle",
        lambda _program_id: SimpleNamespace(
            program={
                "status": "simulating",
                "current_epoch": 1,
                "snapshot_hash": "a" * 64,
            }
        ),
    )
    monkeypatch.setattr(
        jobs,
        "list_linked_simulation_runs",
        lambda _program_id: [
            {"id": "run-queued", "status": "queued"},
            {"id": "run-running", "status": "running"},
            {"id": "run-completed", "status": "completed"},
        ],
    )
    enqueue = Mock()
    sync = Mock()
    monkeypatch.setattr(jobs, "sync_linked_simulation_run", sync)
    record = Mock(return_value=True)
    monkeypatch.setattr(jobs, "record_program_simulation_dispatch", record)

    result = jobs.coordinate_program_simulation(
        program_id="program-1",
        source_event_id="event-1",
        executor=SimpleNamespace(enqueue=enqueue),
    )

    enqueue.assert_called_once_with("run-queued")
    sync.assert_called_once_with("run-completed")
    record.assert_called_once_with(
        program_id="program-1",
        source_event_id="event-1",
        simulation_run_ids=["run-queued", "run-running", "run-completed"],
        expected_epoch=1,
        expected_snapshot_hash="a" * 64,
        details={},
    )
    assert result["status"] == "dispatched"
    assert result["enqueuedRunIds"] == ["run-queued"]


def test_consumer_coordinates_only_simulation_commands(monkeypatch) -> None:
    coordinate = Mock(return_value={"status": "observed"})
    monkeypatch.setattr(jobs, "coordinate_program_simulation", coordinate)

    ordinary = jobs.process_experiment_program_event_job(
        event_id="program-1:created",
        payload={
            "eventId": "program-1:created",
            "programId": "program-1",
            "eventType": "created",
        },
    )
    requested = jobs.process_experiment_program_event_job(
        event_id="program-1:simulation-reconcile:1:2",
        payload={
            "eventId": "program-1:simulation-reconcile:1:2",
            "programId": "program-1",
            "eventType": "simulation_reconcile_requested",
        },
    )

    assert "dispatch" not in ordinary
    assert requested["dispatch"] == {"status": "observed"}
    coordinate.assert_called_once_with(
        program_id="program-1",
        source_event_id="program-1:simulation-reconcile:1:2",
        expected_epoch=None,
        expected_snapshot_hash=None,
    )


def test_stale_recovery_uses_configured_low_frequency_cutoff(monkeypatch) -> None:
    monkeypatch.setattr(
        jobs,
        "get_settings",
        lambda: SimpleNamespace(
            experiment_program_stale_after_seconds=300,
            experiment_program_simulation_max_reconciles=3,
        ),
    )
    requeue = Mock(return_value=["program-1"])
    monkeypatch.setattr(jobs, "requeue_stale_program_simulations", requeue)

    assert jobs._requeue_stale_program_simulations_once() == ["program-1"]
    assert requeue.call_args.kwargs["limit"] == 100
    assert requeue.call_args.kwargs["max_reconciles"] == 3
