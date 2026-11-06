from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agi_ecom_app.decision_simulation import jobs
from agi_ecom_app.decision_simulation.engine import (
    DecisionSimulationEngine,
    derive_batch_seed,
)
from agi_ecom_app.services import memory
from agi_ecom_app.services.memory import DecisionSimulationBatchRecord
from tests.test_decision_simulation_service import NOW, _job_run, _ready_workflow


@pytest.fixture(autouse=True)
def _isolate_simulation_recovery_queries(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(jobs, "list_decision_simulation_runs", lambda **_: [])
    monkeypatch.setattr(
        jobs,
        "backfill_decision_simulation_superseded_events",
        lambda **_: [],
    )


@contextmanager
def _acquired_lock(_: str):
    yield True


def test_resume_backfills_missing_batch_event_without_rewriting_durable_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial = _job_run(_ready_workflow())
    state = initial
    candidate_id = initial.evidence_snapshot["simulationCandidates"][0][
        "candidateId"
    ]
    durable_batch = DecisionSimulationBatchRecord(
        id=11,
        simulation_run_id=initial.id,
        candidate_id=candidate_id,
        batch_index=0,
        seed=derive_batch_seed(initial.evidence_hash, candidate_id, 0),
        sample_count=50,
        status="completed",
        aggregates={"durable": True},
        duration_ms=2.5,
        error={},
        completed_at=NOW,
        created_at=NOW,
        updated_at=NOW,
    )
    events_by_id: dict[str, dict] = {}
    batch_write = Mock(side_effect=AssertionError("durable batch was rewritten"))

    class StopAfterBackfillEngine:
        def validate_candidate(self, candidate):  # type: ignore[no-untyped-def]
            del candidate
            return []

        def simulate_candidate(self, request, candidate):  # type: ignore[no-untyped-def]
            del request, candidate
            raise RuntimeError("stop after durable-event backfill")

    def get_run(simulation_run_id: str):
        assert simulation_run_id == initial.id
        return state

    def update_run(simulation_run_id: str, **changes):
        nonlocal state
        assert simulation_run_id == initial.id
        state = replace(state, **changes)
        return state

    def update_progress(simulation_run_id: str, **changes):
        nonlocal state
        assert simulation_run_id == initial.id
        completed = int(changes.get("completed_samples", state.completed_samples))
        total = int(changes.get("total_samples", state.total_samples))
        state = replace(
            state,
            **changes,
            progress=completed / total if total else 0.0,
        )
        return state

    def append_event(**event):
        # This mirrors the durable event table's ON CONFLICT(event_id) DO NOTHING
        # boundary and also proves that the backfill key is deterministic.
        events_by_id.setdefault(event["event_id"], event)

    monkeypatch.setattr(jobs, "get_decision_simulation_run", get_run)
    monkeypatch.setattr(jobs, "update_decision_simulation_run", update_run)
    monkeypatch.setattr(jobs, "update_decision_simulation_progress", update_progress)
    monkeypatch.setattr(jobs, "_simulation_run_lock", _acquired_lock)
    monkeypatch.setattr(jobs, "DecisionSimulationEngine", StopAfterBackfillEngine)
    monkeypatch.setattr(
        jobs,
        "list_decision_simulation_batches",
        lambda *args, **kwargs: [durable_batch],
    )
    monkeypatch.setattr(
        jobs,
        "list_decision_simulation_candidates",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(jobs, "upsert_decision_simulation_batch", batch_write)
    monkeypatch.setattr(
        jobs,
        "upsert_decision_simulation_candidate",
        lambda **kwargs: kwargs,
    )
    monkeypatch.setattr(jobs, "append_decision_simulation_event", append_event)
    monkeypatch.setattr(jobs, "clear_decision_simulation_winner_flags", lambda _: None)
    monkeypatch.setattr(
        jobs,
        "block_manager_decision_for_simulation",
        lambda *args, **kwargs: None,
    )

    outcome = jobs.run_decision_simulation_job(initial.id)

    assert outcome["status"] == "failed"
    backfill_id = (
        f"{initial.id}:candidate:{candidate_id}:batch:0:completed"
    )
    assert events_by_id[backfill_id]["event_type"] == "batch_completed"
    assert events_by_id[backfill_id]["candidate_id"] == candidate_id
    assert events_by_id[backfill_id]["batch_index"] == 0
    assert events_by_id[backfill_id]["payload"] == {
        "seed": durable_batch.seed,
        "sampleCount": 50,
        "completedSamples": 50,
        "totalSamples": 1_500,
        "durationMs": 2.5,
        "resumedFromDurableBatch": True,
    }
    batch_write.assert_not_called()


def test_completed_enforced_repair_rebuilds_ranking_economics_experiments_and_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial = _job_run(_ready_workflow())
    request = jobs._request_from_run(initial)
    numeric = DecisionSimulationEngine().run(request)
    ranked = jobs._ranked(numeric.candidates)
    terminal = replace(
        initial,
        status="completed",
        completed_samples=1_500,
        total_samples=1_500,
        completed_candidates=3,
        progress=1.0,
        provisional_winner_candidate_id=ranked[0].candidate_id,
        winner_candidate_id=ranked[0].candidate_id,
        critic_result={"result": {"verdict": "approved"}},
        completed_at=NOW,
    )
    projected_decisions = []
    projected_experiments: list[str] = []
    links: list[dict] = []
    events: list[tuple] = []

    def persist_projection(decision, **kwargs):
        projected_decisions.append((decision, kwargs))
        projected_experiments.extend(
            action.action_id for action in decision.ranked_actions
        )
        return SimpleNamespace(id=decision.decision_id), []

    monkeypatch.setattr(jobs, "get_manager_decision", lambda _: None)
    monkeypatch.setattr(jobs, "_durable_ranked_results", lambda _: ranked)
    monkeypatch.setattr(jobs, "persist_manager_decision_bundle", persist_projection)
    monkeypatch.setattr(
        jobs,
        "set_manager_decision_simulation_run",
        lambda decision_id, **kwargs: links.append(
            {"decision_id": decision_id, **kwargs}
        ),
    )
    monkeypatch.setattr(
        jobs,
        "_event",
        lambda run, event_type, **kwargs: events.append(
            (run.id, event_type, kwargs)
        ),
    )

    repaired = jobs.repair_terminal_decision_simulation_projection(terminal)

    assert repaired is True
    assert len(projected_decisions) == 1
    decision = projected_decisions[0][0]
    assert decision.status == "ready_for_approval"
    assert [action.action_id for action in decision.ranked_actions] == [
        item.action_id for item in ranked
    ]
    assert projected_experiments == [item.action_id for item in ranked]
    for action, result in zip(decision.ranked_actions, ranked, strict=True):
        assert result.distribution is not None
        per_day = result.distribution.per_day
        assert action.expected_effect.contribution_profit_low_per_day == pytest.approx(
            per_day.p10
        )
        assert action.expected_effect.contribution_profit_base_per_day == pytest.approx(
            per_day.p50
        )
        assert action.expected_effect.contribution_profit_high_per_day == pytest.approx(
            per_day.p90
        )
        assert action.economics.delta_advertising_per_day.low == pytest.approx(
            -per_day.p90
        )
        assert f"decision-simulation-run:{terminal.id}" in (
            action.expected_effect.evidence_refs
        )
    assert links == [
        {
            "decision_id": terminal.decision_id,
            "simulation_run_id": terminal.id,
            "status": "ready_for_approval",
        }
    ]
    assert events == [
        (
            terminal.id,
            "completed",
            {
                "suffix": "completed",
                "progress": 1,
                "payload": {
                    "winnerCandidateId": terminal.winner_candidate_id,
                    "completedSamples": 1_500,
                    "totalSamples": 1_500,
                    "criticVerdict": "approved",
                    "projectionRepaired": True,
                },
            },
        )
    ]


@pytest.mark.parametrize("terminal_status", ["blocked", "failed"])
def test_enforced_terminal_projection_is_incomplete_without_terminal_event_and_repairs_it(
    monkeypatch: pytest.MonkeyPatch,
    terminal_status: str,
) -> None:
    terminal = replace(
        _job_run(_ready_workflow()),
        status=terminal_status,
        progress=0.75,
        error={
            "code": f"test_{terminal_status}",
            "message": f"terminal {terminal_status}",
        },
    )
    durable_events: list[SimpleNamespace] = []
    manager_projection = SimpleNamespace(
        simulation_run_id=terminal.id,
        status="blocked",
    )
    blocked: list[dict] = []
    emitted: list[tuple] = []

    monkeypatch.setattr(jobs, "get_manager_decision", lambda _: manager_projection)
    monkeypatch.setattr(
        jobs,
        "list_decision_simulation_events",
        lambda *args, **kwargs: list(durable_events),
    )
    monkeypatch.setattr(
        jobs,
        "list_agent_action_experiments_for_decision",
        lambda _: [],
    )

    assert jobs._terminal_projection_is_complete(terminal) is False
    durable_events.append(SimpleNamespace(event_type=terminal_status))
    assert jobs._terminal_projection_is_complete(terminal) is True
    durable_events.clear()

    monkeypatch.setattr(
        jobs,
        "block_manager_decision_for_simulation",
        lambda run, **kwargs: blocked.append({"run": run, **kwargs}),
    )
    monkeypatch.setattr(
        jobs,
        "_publish_shadow_decision",
        Mock(side_effect=AssertionError("enforced repair published shadow state")),
    )
    monkeypatch.setattr(
        jobs,
        "_event",
        lambda run, event_type, **kwargs: emitted.append(
            (run.id, event_type, kwargs)
        ),
    )

    assert jobs.repair_terminal_decision_simulation_projection(terminal) is True
    assert blocked == [
        {
            "run": terminal,
            "code": f"test_{terminal_status}",
            "message": f"terminal {terminal_status}",
        }
    ]
    assert emitted == [
        (
            terminal.id,
            terminal_status,
            {
                "suffix": f"{terminal_status}:projection-repair",
                "progress": 0.75,
                "payload": {
                    "code": f"test_{terminal_status}",
                    "projectionRepaired": True,
                },
            },
        )
    ]


@pytest.mark.parametrize("terminal_status", ["blocked", "failed"])
def test_shadow_terminal_repair_restores_public_projection_pointer_and_event(
    monkeypatch: pytest.MonkeyPatch,
    terminal_status: str,
) -> None:
    terminal = replace(
        _job_run(_ready_workflow()),
        status=terminal_status,
        mode="shadow",
        progress=0.4,
        error={"code": f"shadow_{terminal_status}"},
    )
    persisted = []
    links: list[dict] = []
    emitted: list[tuple] = []

    # The terminal row is durable, but the manager projection pointer was lost
    # in the crash window after the run status commit.
    monkeypatch.setattr(
        jobs,
        "get_manager_decision",
        lambda _: SimpleNamespace(
            simulation_run_id=None,
            status="ready_for_approval",
        ),
    )
    monkeypatch.setattr(
        jobs,
        "persist_manager_decision_bundle",
        lambda decision, **kwargs: persisted.append((decision, kwargs)),
    )
    monkeypatch.setattr(
        jobs,
        "set_manager_decision_simulation_run",
        lambda decision_id, **kwargs: links.append(
            {"decision_id": decision_id, **kwargs}
        ),
    )
    monkeypatch.setattr(
        jobs,
        "block_manager_decision_for_simulation",
        Mock(side_effect=AssertionError("shadow repair blocked public decision")),
    )
    monkeypatch.setattr(
        jobs,
        "_event",
        lambda run, event_type, **kwargs: emitted.append(
            (run.id, event_type, kwargs)
        ),
    )

    assert jobs.repair_terminal_decision_simulation_projection(terminal) is True
    assert len(persisted) == 1
    assert persisted[0][0].status == "ready_for_approval"
    assert links == [
        {
            "decision_id": terminal.decision_id,
            "simulation_run_id": terminal.id,
            "status": None,
        }
    ]
    assert emitted[0][1] == terminal_status
    assert emitted[0][2]["payload"] == {
        "code": f"shadow_{terminal_status}",
        "projectionRepaired": True,
    }


def test_provisional_ranking_never_sets_durable_winner_before_critic() -> None:
    initial = _job_run(_ready_workflow())
    numeric = DecisionSimulationEngine().run(jobs._request_from_run(initial))

    assert sum(item.is_winner for item in numeric.candidates) == 1
    provisional = jobs._ranked(numeric.candidates)

    assert [item.rank for item in provisional] == [1, 2, 3]
    assert [item.candidate_id for item in provisional] == numeric.ranked_candidate_ids
    assert not any(item.is_winner for item in provisional)


def test_concurrent_retry_conflict_recovers_the_single_child_for_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    winner = {"id": "child-created-by-concurrent-worker"}

    class FakeCursor:
        def __init__(self) -> None:
            self.executed: list[tuple[str, object]] = []
            self._rows = iter([None, None, winner])

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            del exc_type, exc, traceback

        def execute(self, query, params=None):  # type: ignore[no-untyped-def]
            self.executed.append((" ".join(str(query).split()), params))

        def fetchone(self):
            return next(self._rows)

    class FakeConnection:
        def __init__(self) -> None:
            self.cursor_instance = FakeCursor()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            del exc_type, exc, traceback

        def cursor(self):
            return self.cursor_instance

    connection = FakeConnection()
    monkeypatch.setattr(memory, "ensure_memory_schema", lambda: None)
    monkeypatch.setattr(memory, "get_memory_connection", lambda: connection)
    monkeypatch.setattr(memory, "_row_to_decision_simulation_run", lambda row: row)

    returned = memory.create_decision_simulation_run(
        simulation_run_id="losing-child",
        decision_id="decision-1",
        mode="enforced",
        model_version="advertising-bootstrap-v1",
        evidence_snapshot={},
        evidence_hash="a" * 64,
        master_seed=1,
        candidate_count=3,
        idempotency_key="different-concurrent-key",
        retry_of_run_id="parent-1",
        retry_attempt=1,
    )

    assert returned is winner
    executed = connection.cursor_instance.executed
    assert len(executed) == 3
    assert "insert into decision_simulation_runs" in executed[0][0]
    assert (
        "where retry_of_run_id = %s and idempotency_key = %s"
        in executed[1][0]
    )
    assert executed[1][1] == ["parent-1", "different-concurrent-key"]
    assert "where retry_of_run_id = %s order by created_at, id limit 1" in (
        executed[2][0]
    )
    assert executed[2][1] == ["parent-1"]

    normalized_schema = " ".join(memory.MEMORY_SCHEMA_SQL.split())
    assert (
        "create unique index if not exists "
        "decision_simulation_runs_retry_parent_idx on "
        "decision_simulation_runs(retry_of_run_id) "
        "where retry_of_run_id is not null"
    ) in normalized_schema


def test_terminal_projection_never_restores_parent_after_retry_child_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = replace(
        _job_run(_ready_workflow()),
        id="terminal-parent",
        status="completed",
        winner_candidate_id="candidate-1",
        completed_at=NOW,
    )
    child = replace(
        parent,
        id="retry-child",
        status="queued",
        retry_of_run_id=parent.id,
        winner_candidate_id=None,
        completed_at=None,
    )
    monkeypatch.setattr(
        jobs,
        "list_decision_simulation_runs",
        lambda **_: [child, parent],
    )
    monkeypatch.setattr(
        jobs,
        "_publish_enforced_decision",
        Mock(side_effect=AssertionError("old parent projection was restored")),
    )

    assert jobs.repair_terminal_decision_simulation_projection(parent) is False


def test_terminal_projection_query_models_pricing_shadow_as_needs_data_without_experiments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCursor:
        def __init__(self) -> None:
            self.query = ""
            self.params: object = None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            del exc_type, exc, traceback

        def execute(self, query, params=None):  # type: ignore[no-untyped-def]
            self.query = " ".join(str(query).split())
            self.params = params

        def fetchall(self):
            return []

    class FakeConnection:
        def __init__(self) -> None:
            self.cursor_instance = FakeCursor()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            del exc_type, exc, traceback

        def cursor(self):
            return self.cursor_instance

    connection = FakeConnection()
    monkeypatch.setattr(memory, "ensure_memory_schema", lambda: None)
    monkeypatch.setattr(memory, "get_memory_connection", lambda: connection)

    assert memory.list_unprojected_terminal_decision_simulation_runs(limit=100) == []

    query = connection.cursor_instance.query
    pricing_model = "runs.model_version = 'pricing-elasticity-bootstrap-v1'"
    assert pricing_model in query
    assert (
        "runs.mode = 'shadow' and "
        f"{pricing_model} and ( decisions.status <> 'needs_data' or "
        "( select count(*) from agent_action_experiments experiments "
        "where experiments.decision_id = runs.decision_id ) <> 0 )"
    ) in query
    assert (
        "runs.status in ('blocked', 'failed') and runs.mode = 'shadow'"
    ) in query
    assert connection.cursor_instance.params == [100]
