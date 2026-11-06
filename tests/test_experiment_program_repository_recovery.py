from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pytest

from agi_ecom_app.experiment_program import repository


PROGRAM_ID = "program-1"
DECISION_ID = "decision-1"
SIMULATION_RUN_ID = "simulation-1"
SNAPSHOT_HASH = "a" * 64
STALE_BEFORE = datetime(2026, 7, 20, 9, 0, tzinfo=timezone.utc)


def _json_value(value: Any) -> Any:
    return getattr(value, "obj", value)


class _BindingCursor:
    def __init__(
        self,
        *,
        program: dict[str, Any],
        missing_simulation_candidate: str | None = None,
    ) -> None:
        self.program = program
        self.missing_simulation_candidate = missing_simulation_candidate
        self._rows: list[dict[str, Any]] = []
        self.executed: list[tuple[str, list[Any]]] = []
        self.mapped_simulation_candidates: list[str] = []

    def __enter__(self) -> "_BindingCursor":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        del exc_type, exc, traceback
        return False

    def execute(self, query: str, params: list[Any] | None = None) -> None:
        values = list(params or [])
        normalized = " ".join(query.split()).lower()
        self.executed.append((normalized, values))

        if normalized.startswith(
            "select * from experiment_programs where id = %s for update"
        ):
            self._rows = [self.program]
            return
        if normalized.startswith(
            "update manager_decisions set experiment_program_id = %s"
        ):
            self._rows = [{"id": values[1]}]
            return
        if normalized.startswith(
            "update experiment_programs set source_decision_id = coalesce"
        ):
            self._rows = []
            return
        if normalized.startswith(
            "update decision_simulation_runs as run "
            "set experiment_program_id = %s"
        ):
            self._rows = [{"id": values[1]}]
            return
        if normalized.startswith(
            "update decision_simulation_candidates as simulation_candidate "
            "set experiment_program_id = %s"
        ):
            simulation_candidate_id = str(values[3])
            self.mapped_simulation_candidates.append(simulation_candidate_id)
            self._rows = (
                []
                if simulation_candidate_id == self.missing_simulation_candidate
                else [{"candidate_id": simulation_candidate_id}]
            )
            return
        raise AssertionError(f"unexpected SQL in binding test: {normalized}")

    def fetchone(self) -> dict[str, Any] | None:
        return self._rows[0] if self._rows else None


class _BindingConnection:
    def __init__(self, cursor: _BindingCursor) -> None:
        self.cursor_instance = cursor
        self.exit_exception_type: type[BaseException] | None = None

    def __enter__(self) -> "_BindingConnection":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        del exc, traceback
        self.exit_exception_type = exc_type
        return False

    def cursor(self) -> _BindingCursor:
        return self.cursor_instance


def _patch_binding_storage(
    monkeypatch: pytest.MonkeyPatch,
    *,
    missing_simulation_candidate: str | None = None,
) -> tuple[_BindingConnection, list[tuple[_BindingCursor, dict[str, Any]]]]:
    cursor = _BindingCursor(
        program={
            "id": PROGRAM_ID,
            "current_epoch": 2,
            "snapshot_hash": SNAPSHOT_HASH,
        },
        missing_simulation_candidate=missing_simulation_candidate,
    )
    connection = _BindingConnection(cursor)
    appended: list[tuple[_BindingCursor, dict[str, Any]]] = []
    monkeypatch.setattr(repository, "ensure_memory_schema", lambda: None)
    monkeypatch.setattr(repository, "get_memory_connection", lambda: connection)
    monkeypatch.setattr(
        repository,
        "_append_event_and_outbox",
        lambda active_cursor, event: appended.append((active_cursor, event)),
    )
    return connection, appended


def test_bind_program_simulation_run_maps_the_complete_frontier_atomically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection, appended = _patch_binding_storage(monkeypatch)
    candidate_links = {
        "simulation-candidate-1": "program-candidate-1",
        "simulation-candidate-2": "program-candidate-2",
        "simulation-candidate-3": "program-candidate-3",
    }

    linked = repository.bind_program_simulation_run(
        program_id=PROGRAM_ID,
        decision_id=DECISION_ID,
        simulation_run_id=SIMULATION_RUN_ID,
        candidate_links=candidate_links,
        expected_epoch=2,
        expected_snapshot_hash=SNAPSHOT_HASH,
    )

    assert linked == 3
    assert connection.cursor_instance.mapped_simulation_candidates == list(
        candidate_links
    )
    assert connection.exit_exception_type is None
    assert appended == [
        (
            connection.cursor_instance,
            {
                "event_id": f"{PROGRAM_ID}:simulation-bound:{SIMULATION_RUN_ID}",
                "program_id": PROGRAM_ID,
                "epoch": 2,
                "event_type": "simulation_bound",
                "payload": {
                    "decisionId": DECISION_ID,
                    "simulationRunId": SIMULATION_RUN_ID,
                    "linkedCandidateCount": 3,
                    "snapshotHash": SNAPSHOT_HASH,
                },
            },
        )
    ]
    assert any(
        "simulation_dispatch_attempts = 0" in query
        for query, _params in connection.cursor_instance.executed
    )


def test_bind_program_simulation_run_never_journals_a_partial_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection, appended = _patch_binding_storage(
        monkeypatch,
        missing_simulation_candidate="simulation-candidate-3",
    )

    with pytest.raises(
        repository.ExperimentProgramStorageConflict,
        match="incomplete or cross-program simulation candidate mapping",
    ):
        repository.bind_program_simulation_run(
            program_id=PROGRAM_ID,
            decision_id=DECISION_ID,
            simulation_run_id=SIMULATION_RUN_ID,
            candidate_links={
                "simulation-candidate-1": "program-candidate-1",
                "simulation-candidate-2": "program-candidate-2",
                "simulation-candidate-3": "program-candidate-3",
            },
            expected_epoch=2,
            expected_snapshot_hash=SNAPSHOT_HASH,
        )

    assert connection.cursor_instance.mapped_simulation_candidates == [
        "simulation-candidate-1",
        "simulation-candidate-2",
        "simulation-candidate-3",
    ]
    assert appended == []
    # psycopg's connection context rolls this one transaction back because the
    # exception exits the same connection that performed every binding write.
    assert connection.exit_exception_type is repository.ExperimentProgramStorageConflict


@dataclass
class _ReconcileStore:
    programs: list[dict[str, Any]]
    linked_runs: dict[str, list[dict[str, Any]]]
    compare_and_swap_succeeds: bool = True
    executed: list[tuple[str, list[Any]]] = field(default_factory=list)
    active_heartbeat_updates: list[str] = field(default_factory=list)
    reconcile_updates: list[dict[str, Any]] = field(default_factory=list)
    failure_updates: list[dict[str, Any]] = field(default_factory=list)
    appended_events: list[dict[str, Any]] = field(default_factory=list)


class _ReconcileCursor:
    def __init__(self, store: _ReconcileStore) -> None:
        self.store = store
        self._rows: list[dict[str, Any]] = []

    def __enter__(self) -> "_ReconcileCursor":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        del exc_type, exc, traceback
        return False

    def execute(self, query: str, params: list[Any] | None = None) -> None:
        values = list(params or [])
        normalized = " ".join(query.split()).lower()
        self.store.executed.append((normalized, values))

        if normalized.startswith(
            "select * from experiment_programs where status = 'simulating'"
        ):
            self._rows = [dict(program) for program in self.store.programs]
            return
        if normalized.startswith(
            "select id, status from decision_simulation_runs "
            "where experiment_program_id = %s"
        ):
            self._rows = [
                dict(run) for run in self.store.linked_runs.get(str(values[0]), [])
            ]
            return
        if (
            normalized.startswith("update experiment_programs set heartbeat_at = now()")
            and "simulation_dispatch_attempts" not in normalized
        ):
            self.store.active_heartbeat_updates.append(str(values[0]))
            self._rows = []
            return
        if normalized.startswith(
            "update experiment_programs set status = 'failed'"
        ):
            self.store.failure_updates.append(
                {
                    "error": _json_value(values[0]),
                    "attempts": values[1],
                    "version": values[2],
                    "program_id": values[3],
                    "expected_version": values[4],
                }
            )
            self._rows = (
                [{"id": values[3]}] if self.store.compare_and_swap_succeeds else []
            )
            return
        if (
            normalized.startswith("update experiment_programs set heartbeat_at = now()")
            and "simulation_dispatch_attempts = %s" in normalized
        ):
            self.store.reconcile_updates.append(
                {
                    "attempts": values[0],
                    "version": values[1],
                    "program_id": values[2],
                    "expected_version": values[3],
                }
            )
            self._rows = (
                [{"id": values[2]}] if self.store.compare_and_swap_succeeds else []
            )
            return
        raise AssertionError(f"unexpected SQL in reconcile test: {normalized}")

    def fetchone(self) -> dict[str, Any] | None:
        return self._rows[0] if self._rows else None

    def fetchall(self) -> list[dict[str, Any]]:
        return list(self._rows)


class _ReconcileConnection:
    def __init__(self, store: _ReconcileStore) -> None:
        self.cursor_instance = _ReconcileCursor(store)

    def __enter__(self) -> "_ReconcileConnection":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        del exc_type, exc, traceback
        return False

    def cursor(self) -> _ReconcileCursor:
        return self.cursor_instance


def _program(*, attempts: int, version: int = 5) -> dict[str, Any]:
    return {
        "id": PROGRAM_ID,
        "status": "simulating",
        "current_epoch": 2,
        "snapshot_hash": SNAPSHOT_HASH,
        "simulation_dispatch_attempts": attempts,
        "version": version,
    }


def _patch_reconcile_storage(
    monkeypatch: pytest.MonkeyPatch,
    store: _ReconcileStore,
) -> None:
    connection = _ReconcileConnection(store)
    monkeypatch.setattr(repository, "ensure_memory_schema", lambda: None)
    monkeypatch.setattr(repository, "get_memory_connection", lambda: connection)
    monkeypatch.setattr(
        repository,
        "_append_event_and_outbox",
        lambda _cursor, event: store.appended_events.append(event),
    )


def test_stale_program_with_active_linked_run_refreshes_without_spending_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _ReconcileStore(
        programs=[_program(attempts=2)],
        linked_runs={PROGRAM_ID: [{"id": "run-active", "status": "running"}]},
    )
    _patch_reconcile_storage(monkeypatch, store)

    recovered = repository.requeue_stale_program_simulations(
        stale_before=STALE_BEFORE,
        max_reconciles=3,
    )

    assert recovered == []
    assert store.active_heartbeat_updates == [PROGRAM_ID]
    assert store.reconcile_updates == []
    assert store.failure_updates == []
    assert store.appended_events == []


def test_stale_program_without_linked_run_spends_one_retry_and_requeues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _ReconcileStore(
        programs=[_program(attempts=0, version=5)],
        linked_runs={PROGRAM_ID: []},
    )
    _patch_reconcile_storage(monkeypatch, store)

    recovered = repository.requeue_stale_program_simulations(
        stale_before=STALE_BEFORE,
        max_reconciles=3,
    )

    assert recovered == [PROGRAM_ID]
    assert store.reconcile_updates == [
        {
            "attempts": 1,
            "version": 6,
            "program_id": PROGRAM_ID,
            "expected_version": 5,
        }
    ]
    assert store.appended_events == [
        {
            "event_id": f"{PROGRAM_ID}:simulation-reconcile:2:6",
            "program_id": PROGRAM_ID,
            "epoch": 2,
            "event_type": "simulation_reconcile_requested",
            "payload": {
                "programId": PROGRAM_ID,
                "epoch": 2,
                "snapshotHash": SNAPSHOT_HASH,
                "reason": "stale_program_heartbeat",
            },
        }
    ]


def test_third_unlinked_reconcile_attempt_fails_program_without_another_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _ReconcileStore(
        programs=[_program(attempts=2, version=8)],
        linked_runs={PROGRAM_ID: []},
    )
    _patch_reconcile_storage(monkeypatch, store)

    recovered = repository.requeue_stale_program_simulations(
        stale_before=STALE_BEFORE,
        max_reconciles=3,
    )

    assert recovered == []
    assert store.reconcile_updates == []
    assert store.failure_updates == [
        {
            "error": {
                "code": "simulation_dispatch_retries_exhausted",
                "message": (
                    "No simulation run was bound before the recovery budget expired."
                ),
                "attempts": 3,
            },
            "attempts": 3,
            "version": 9,
            "program_id": PROGRAM_ID,
            "expected_version": 8,
        }
    ]
    assert store.appended_events[0]["event_type"] == "simulation_dispatch_failed"
    assert store.appended_events[0]["event_id"] == (
        f"{PROGRAM_ID}:simulation-reconcile-exhausted:2"
    )
    assert store.appended_events[0]["payload"]["programStatus"] == "failed"


def test_terminal_linked_run_requeues_sync_without_spending_dispatch_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _ReconcileStore(
        programs=[_program(attempts=2, version=5)],
        linked_runs={PROGRAM_ID: [{"id": "run-completed", "status": "completed"}]},
    )
    _patch_reconcile_storage(monkeypatch, store)

    recovered = repository.requeue_stale_program_simulations(
        stale_before=STALE_BEFORE,
        max_reconciles=3,
    )

    assert recovered == [PROGRAM_ID]
    assert store.reconcile_updates == [
        {
            "attempts": 2,
            "version": 6,
            "program_id": PROGRAM_ID,
            "expected_version": 5,
        }
    ]
    assert store.appended_events[0]["event_type"] == "simulation_reconcile_requested"


def test_stale_reconcile_compare_and_swap_loser_emits_no_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _ReconcileStore(
        programs=[_program(attempts=0, version=5)],
        linked_runs={PROGRAM_ID: []},
        compare_and_swap_succeeds=False,
    )
    _patch_reconcile_storage(monkeypatch, store)

    recovered = repository.requeue_stale_program_simulations(
        stale_before=STALE_BEFORE,
        max_reconciles=3,
    )

    assert recovered == []
    assert store.appended_events == []
    assert store.reconcile_updates[0]["expected_version"] == 5
    first_query = store.executed[0][0]
    assert "for update skip locked" in first_query
