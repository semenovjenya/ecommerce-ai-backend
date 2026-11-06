from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from agi_ecom_app.experiment_program import repository


PROGRAM_ID = "program-1"
SIMULATION_RUN_ID = "simulation-1"


def _json_value(value: Any) -> Any:
    return getattr(value, "obj", value)


def _run(*, status: str = "completed", mode: str = "shadow") -> dict[str, Any]:
    return {
        "id": SIMULATION_RUN_ID,
        "experiment_program_id": PROGRAM_ID,
        "status": status,
        "mode": mode,
        "winner_candidate_id": "simulation-candidate-1",
        "error": {"source": "simulation"} if status in {"blocked", "failed"} else {},
        "critic_result": {"verdict": "blocked"} if status == "blocked" else {},
    }


def _program() -> dict[str, Any]:
    return {
        "id": PROGRAM_ID,
        "current_epoch": 1,
        "status": "simulating",
        "domain_coverage": {"finance": {"coverageStatus": "constraint_only"}},
        "error": {},
        "version": 3,
    }


def _simulation_candidate(
    index: int,
    *,
    status: str = "completed",
    winner: bool = False,
    rank: int | None = None,
) -> dict[str, Any]:
    return {
        "simulation_run_id": SIMULATION_RUN_ID,
        "candidate_id": f"simulation-candidate-{index}",
        "experiment_program_id": PROGRAM_ID,
        "experiment_candidate_id": f"program-candidate-{index}",
        "domain": "pricing",
        "status": status,
        "is_winner": winner,
        "rank": rank,
        "distribution_summary": {"p50": 100 + index},
        "score_breakdown": {"cvar10": 10 - index},
    }


@dataclass
class _SyncStore:
    run: dict[str, Any]
    program: dict[str, Any] = field(default_factory=_program)
    simulation_candidates: list[dict[str, Any]] = field(default_factory=list)
    event_ids: set[str] = field(default_factory=set)
    appended_events: list[dict[str, Any]] = field(default_factory=list)
    candidate_updates: list[dict[str, Any]] = field(default_factory=list)
    frontier_updates: list[dict[str, Any]] = field(default_factory=list)
    program_updates: list[dict[str, Any]] = field(default_factory=list)


class _SyncCursor:
    def __init__(self, store: _SyncStore) -> None:
        self.store = store
        self._rows: list[dict[str, Any]] = []

    def __enter__(self) -> "_SyncCursor":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        del exc_type, exc, traceback
        return False

    def execute(self, query: str, params: list[Any] | None = None) -> None:
        values = list(params or [])
        normalized = " ".join(query.split()).lower()

        if normalized.startswith(
            "select * from decision_simulation_runs where id = %s for update"
        ):
            self._rows = [self.store.run]
            return
        if normalized.startswith(
            "select * from experiment_programs where id = %s for update"
        ):
            self._rows = [self.store.program]
            return
        if normalized.startswith(
            "select 1 from experiment_program_events where event_id = %s"
        ):
            self._rows = (
                [{"exists": 1}] if values[0] in self.store.event_ids else []
            )
            return
        if normalized.startswith(
            "select simulation_candidate.*, candidate.domain "
            "from decision_simulation_candidates as simulation_candidate"
        ):
            self._rows = [dict(candidate) for candidate in self.store.simulation_candidates]
            return
        if normalized.startswith("update experiment_candidates set status = %s"):
            update = {
                "status": values[0],
                "score": _json_value(values[1]),
                "completed_status": values[2],
                "candidate_id": values[3],
                "program_id": values[4],
            }
            self.store.candidate_updates.append(update)
            self._rows = []
            return
        if normalized.startswith("update experiment_domain_frontiers set status = %s"):
            update = {
                "status": values[0],
                "coverage": _json_value(values[1]),
                "program_id": values[3],
                "epoch": values[4],
                "domain": values[5],
            }
            self.store.frontier_updates.append(update)
            self._rows = []
            return
        if normalized.startswith("update experiment_programs set status = %s"):
            update = {
                "status": values[0],
                "domain_coverage": _json_value(values[1]),
                "error": _json_value(values[2]),
                "terminal_status": values[3],
                "program_id": values[4],
            }
            self.store.program_updates.append(update)
            self.store.program.update(
                {
                    "status": update["status"],
                    "domain_coverage": update["domain_coverage"],
                    "error": update["error"],
                    "version": int(self.store.program["version"]) + 1,
                }
            )
            self._rows = []
            return
        raise AssertionError(f"unexpected SQL in sync test: {normalized}")

    def fetchone(self) -> dict[str, Any] | None:
        return self._rows[0] if self._rows else None

    def fetchall(self) -> list[dict[str, Any]]:
        return list(self._rows)


class _SyncConnection:
    def __init__(self, store: _SyncStore) -> None:
        self.cursor_instance = _SyncCursor(store)

    def __enter__(self) -> "_SyncConnection":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        del exc_type, exc, traceback
        return False

    def cursor(self) -> _SyncCursor:
        return self.cursor_instance


def _patch_sync_storage(
    monkeypatch: pytest.MonkeyPatch,
    store: _SyncStore,
) -> None:
    connection = _SyncConnection(store)
    monkeypatch.setattr(repository, "ensure_memory_schema", lambda: None)
    monkeypatch.setattr(repository, "get_memory_connection", lambda: connection)

    def append_event(_cursor: _SyncCursor, event: dict[str, Any]) -> None:
        assert event["event_id"] not in store.event_ids
        store.event_ids.add(event["event_id"])
        store.appended_events.append(event)

    monkeypatch.setattr(repository, "_append_event_and_outbox", append_event)


def test_completed_shadow_sync_projects_winner_ranks_and_selecting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _SyncStore(
        run=_run(status="completed", mode="shadow"),
        simulation_candidates=[
            _simulation_candidate(2, rank=2),
            _simulation_candidate(1, winner=True, rank=1),
        ],
    )
    _patch_sync_storage(monkeypatch, store)

    assert repository.sync_linked_simulation_run(SIMULATION_RUN_ID) == PROGRAM_ID

    by_candidate = {
        update["candidate_id"]: update for update in store.candidate_updates
    }
    assert by_candidate["program-candidate-1"]["status"] == "selected"
    assert by_candidate["program-candidate-1"]["score"] == {
        "simulationRunId": SIMULATION_RUN_ID,
        "simulationCandidateId": "simulation-candidate-1",
        "rank": 1,
        "winner": True,
        "distributionSummary": {"p50": 101},
        "scoreBreakdown": {"cvar10": 9},
    }
    assert by_candidate["program-candidate-2"]["status"] == "simulated"
    assert by_candidate["program-candidate-2"]["score"]["rank"] == 2
    assert by_candidate["program-candidate-2"]["score"]["winner"] is False

    assert store.program_updates == [
        {
            "status": "selecting",
            "domain_coverage": {
                "finance": {"coverageStatus": "constraint_only"},
                "pricing": {
                    "simulationRunId": SIMULATION_RUN_ID,
                    "simulationStatus": "completed",
                    "simulationCompleted": True,
                    "winnerCandidateId": "simulation-candidate-1",
                },
            },
            "error": {},
            "terminal_status": "selecting",
            "program_id": PROGRAM_ID,
        }
    ]
    assert store.frontier_updates[0]["status"] == "ready"
    assert store.appended_events == [
        {
            "event_id": f"{PROGRAM_ID}:simulation:{SIMULATION_RUN_ID}:completed",
            "program_id": PROGRAM_ID,
            "epoch": 1,
            "event_type": "simulation_completed",
            "payload": {
                "simulationRunId": SIMULATION_RUN_ID,
                "simulationStatus": "completed",
                "winnerCandidateId": "simulation-candidate-1",
                "programStatus": "selecting",
                "linkedCandidateCount": 2,
            },
        }
    ]


def test_completed_enforced_sync_moves_program_to_awaiting_approval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _SyncStore(
        run=_run(status="completed", mode="enforced"),
        simulation_candidates=[_simulation_candidate(1, winner=True, rank=1)],
    )
    _patch_sync_storage(monkeypatch, store)

    repository.sync_linked_simulation_run(SIMULATION_RUN_ID)

    assert store.program_updates[0]["status"] == "awaiting_approval"
    assert store.candidate_updates[0]["status"] == "selected"
    assert store.appended_events[0]["payload"]["programStatus"] == "awaiting_approval"


def test_sync_is_idempotent_for_the_same_run_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _SyncStore(
        run=_run(status="completed", mode="shadow"),
        simulation_candidates=[_simulation_candidate(1, winner=True, rank=1)],
    )
    _patch_sync_storage(monkeypatch, store)

    assert repository.sync_linked_simulation_run(SIMULATION_RUN_ID) == PROGRAM_ID
    first_counts = (
        len(store.candidate_updates),
        len(store.frontier_updates),
        len(store.program_updates),
        len(store.appended_events),
    )
    assert repository.sync_linked_simulation_run(SIMULATION_RUN_ID) == PROGRAM_ID

    assert first_counts == (1, 1, 1, 1)
    assert (
        len(store.candidate_updates),
        len(store.frontier_updates),
        len(store.program_updates),
        len(store.appended_events),
    ) == first_counts
    assert store.program["version"] == 4


@pytest.mark.parametrize(
    (
        "run_status",
        "candidate_status",
        "expected_program_status",
        "expected_candidate_status",
        "expected_frontier_status",
    ),
    [
        ("blocked", "needs_data", "blocked_data", "blocked_data", "failed"),
        ("failed", "failed", "failed", "rejected", "failed"),
        ("superseded", "completed", "replanning", "superseded", "failed"),
    ],
)
def test_terminal_simulation_statuses_project_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    run_status: str,
    candidate_status: str,
    expected_program_status: str,
    expected_candidate_status: str,
    expected_frontier_status: str,
) -> None:
    store = _SyncStore(
        run=_run(status=run_status, mode="shadow"),
        simulation_candidates=[
            _simulation_candidate(1, status=candidate_status, winner=False, rank=None)
        ],
    )
    _patch_sync_storage(monkeypatch, store)

    repository.sync_linked_simulation_run(SIMULATION_RUN_ID)

    assert store.program_updates[0]["status"] == expected_program_status
    assert store.candidate_updates[0]["status"] == expected_candidate_status
    assert store.frontier_updates[0]["status"] == expected_frontier_status
    assert store.appended_events[0]["event_type"] == f"simulation_{run_status}"
    if run_status in {"blocked", "failed"}:
        assert store.program_updates[0]["error"] == {
            "code": f"decision_simulation_{run_status}",
            "simulationRunId": SIMULATION_RUN_ID,
            "simulationError": {"source": "simulation"},
            "criticResult": (
                {"verdict": "blocked"} if run_status == "blocked" else {}
            ),
        }


def test_unlinked_simulation_run_is_a_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _run()
    run["experiment_program_id"] = None
    store = _SyncStore(run=run, simulation_candidates=[])
    _patch_sync_storage(monkeypatch, store)

    assert repository.sync_linked_simulation_run(SIMULATION_RUN_ID) is None
    assert store.candidate_updates == []
    assert store.frontier_updates == []
    assert store.program_updates == []
    assert store.appended_events == []
