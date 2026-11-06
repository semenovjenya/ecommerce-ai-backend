from __future__ import annotations

import asyncio
import json
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from rq.job import JobStatus

from agi_ecom_app.config import get_settings
from agi_ecom_app.decision_simulation import jobs, service
from agi_ecom_app.decision_simulation.critic import (
    DecisionCriticExecution,
    DecisionCriticIssue,
    DecisionCriticResult,
)
from agi_ecom_app.manager_lifecycle.auth import ManagerActor
from agi_ecom_app.services.memory import MEMORY_SCHEMA_SQL
from agi_ecom_app.action_board import router as simulation_router
from tests.test_decision_simulation_service import (
    _job_run,
    _patch_schedule_storage,
    _ready_workflow,
    _run_record,
    _settings,
)


@pytest.fixture(autouse=True)
def _isolate_simulation_recovery_queries(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(jobs, "list_decision_simulation_runs", lambda **_: [])
    monkeypatch.setattr(
        jobs,
        "backfill_decision_simulation_superseded_events",
        lambda **_: [],
    )

    def claim_stale(run_id, *, claimed_status, heartbeat_at, error=None, **_):
        changes = {"status": claimed_status, "heartbeat_at": heartbeat_at}
        if error is not None:
            changes["error"] = error
        return jobs.update_decision_simulation_run(run_id, **changes)

    monkeypatch.setattr(jobs, "claim_stale_decision_simulation_run", claim_stale)


@pytest.mark.parametrize(
    ("variable", "invalid_value", "expected"),
    [
        ("DECISION_SIMULATION_SAMPLE_COUNT", "499", 500),
        ("DECISION_SIMULATION_BATCH_SIZE", "25", 50),
    ],
)
def test_v1_fixed_simulation_shape_fails_fast_in_configuration(
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
    invalid_value: str,
    expected: int,
) -> None:
    monkeypatch.setenv("DECISION_SIMULATION_SAMPLE_COUNT", "500")
    monkeypatch.setenv("DECISION_SIMULATION_BATCH_SIZE", "50")
    monkeypatch.setenv(variable, invalid_value)

    with pytest.raises(
        ValueError,
        match=rf"{variable} is fixed at {expected}.*got {invalid_value}",
    ):
        get_settings()


def test_hard_gate_keeps_three_of_four_candidates_and_uses_effective_total(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = _ready_workflow(4)
    actions = list(original.decision.ranked_actions)
    actions[0] = actions[0].model_copy(
        update={
            "evidence": [
                fact
                for fact in actions[0].evidence
                if fact.metric != "daily_ad_spend_distribution"
            ]
        }
    )
    result = original.model_copy(
        update={
            "decision": original.decision.model_copy(
                update={"ranked_actions": actions}
            )
        }
    )

    monkeypatch.setattr(service, "get_settings", lambda: _settings("enforced"))
    persisted, candidates, events, _ = _patch_schedule_storage(monkeypatch)
    created: list[dict[str, object]] = []
    runs = {}

    def create_run(**kwargs):
        created.append(kwargs)
        run = replace(
            _run_record(
                run_id=kwargs["simulation_run_id"],
                decision_id=kwargs["decision_id"],
                status=kwargs.get("status", "queued"),
                mode=kwargs["mode"],
                snapshot=kwargs["evidence_snapshot"],
                evidence_hash=kwargs["evidence_hash"],
            ),
            candidate_count=kwargs["candidate_count"],
            total_samples=kwargs["candidate_count"] * kwargs["sample_count"],
        )
        runs[run.id] = run
        return run

    monkeypatch.setattr(service, "create_decision_simulation_run", create_run)
    monkeypatch.setattr(service, "get_decision_simulation_run", runs.get)
    monkeypatch.setattr(
        service,
        "claim_decision_simulation_initialization",
        lambda run_id: runs.__setitem__(
            run_id, replace(runs[run_id], heartbeat_at=datetime.now(timezone.utc))
        )
        or runs[run_id],
    )
    monkeypatch.setattr(
        service,
        "update_decision_simulation_run",
        lambda run_id, **changes: runs.__setitem__(
            run_id, replace(runs[run_id], **changes)
        )
        or runs[run_id],
    )
    executor = Mock()
    executor.enqueue.return_value = "job-id"

    returned = service.maybe_schedule_decision_simulation(
        result,
        chat_id="chat-1",
        run_id=17,
        source_key="hard-gate-source",
        executor=executor,
    )

    assert returned.decision.status == "evaluating"
    assert persisted[-1][0].status == "evaluating"
    assert len(created) == 1
    assert created[0]["candidate_count"] == 3
    assert len(created[0]["evidence_snapshot"]["simulationCandidates"]) == 3
    rejected = created[0]["evidence_snapshot"]["hardGateRejectedCandidates"]
    assert len(rejected) == 1
    assert rejected[0]["candidate"]["candidateId"] == actions[0].action_id
    assert len(candidates) == 4
    rejected_rows = [item for item in candidates if item.get("status") == "needs_data"]
    assert len(rejected_rows) == 1
    assert rejected_rows[0]["candidate_id"] == actions[0].action_id
    queued_event = next(item for item in events if item["event_type"] == "queued")
    assert queued_event["payload"]["candidateCount"] == 3
    assert queued_event["payload"]["totalSamples"] == 1_500
    assert any(item["event_type"] == "candidate_rejected" for item in events)
    executor.enqueue.assert_called_once()

    run = replace(
        _run_record(
            run_id="critic-hard-gate-run",
            decision_id=result.decision.decision_id,
            snapshot=created[0]["evidence_snapshot"],
            evidence_hash=created[0]["evidence_hash"],
        ),
        candidate_count=3,
        total_samples=1_500,
    )
    numeric = jobs.DecisionSimulationEngine().run(jobs._request_from_run(run))
    critic_input = jobs._critic_input(run, jobs._ranked(numeric.candidates), [])
    assert len(critic_input.candidate_aggregates) == 4
    rejected_aggregate = next(
        item
        for item in critic_input.candidate_aggregates
        if item["candidateId"] == actions[0].action_id
    )
    assert rejected_aggregate["status"] == "needs_data"
    assert rejected_aggregate["constraintViolations"]


def test_source_key_staging_is_resumed_with_durable_decision_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _ready_workflow(3)
    incoming = base.model_copy(
        update={
            "decision": base.decision.model_copy(
                update={"decision_id": "new-volatile-decision-id"}
            )
        }
    )
    staged_decision = base.decision.model_copy(
        update={"decision_id": "durable-staged-decision-id"}
    )
    staged_record = SimpleNamespace(
        id=staged_decision.decision_id,
        simulation_run_id=None,
    )

    monkeypatch.setattr(service, "get_settings", lambda: _settings("enforced"))
    persisted, candidates, _, links = _patch_schedule_storage(monkeypatch)
    monkeypatch.setattr(
        service,
        "get_manager_decision_by_source_key",
        lambda source_key: staged_record if source_key == "stable-source" else None,
    )
    monkeypatch.setattr(
        service,
        "get_manager_decision",
        lambda decision_id: SimpleNamespace(
            decision_json=staged_decision.model_dump(mode="json", by_alias=True)
        )
        if decision_id == staged_decision.decision_id
        else None,
    )
    created: list[dict[str, object]] = []
    runs = {}

    def create_run(**kwargs):
        created.append(kwargs)
        run = _run_record(
            run_id=kwargs["simulation_run_id"],
            decision_id=kwargs["decision_id"],
            status=kwargs.get("status", "queued"),
            mode=kwargs["mode"],
            snapshot=kwargs["evidence_snapshot"],
            evidence_hash=kwargs["evidence_hash"],
        )
        runs[run.id] = run
        return run

    monkeypatch.setattr(service, "create_decision_simulation_run", create_run)
    monkeypatch.setattr(service, "get_decision_simulation_run", runs.get)
    monkeypatch.setattr(
        service,
        "claim_decision_simulation_initialization",
        lambda run_id: runs.__setitem__(
            run_id, replace(runs[run_id], heartbeat_at=datetime.now(timezone.utc))
        )
        or runs[run_id],
    )
    monkeypatch.setattr(
        service,
        "update_decision_simulation_run",
        lambda run_id, **changes: runs.__setitem__(
            run_id, replace(runs[run_id], **changes)
        )
        or runs[run_id],
    )
    executor = Mock()

    returned = service.maybe_schedule_decision_simulation(
        incoming,
        chat_id="chat-1",
        run_id=17,
        source_key="stable-source",
        executor=executor,
    )

    assert returned.decision.decision_id == staged_decision.decision_id
    assert returned.decision.status == "evaluating"
    assert created[0]["decision_id"] == staged_decision.decision_id
    assert (
        created[0]["evidence_snapshot"]["managerDecision"]["decisionId"]
        == staged_decision.decision_id
    )
    assert {item[0].decision_id for item in persisted} == {
        staged_decision.decision_id
    }
    assert {item["decision_id"] for item in links} == {staged_decision.decision_id}
    assert len(candidates) == 3
    executor.enqueue.assert_called_once_with(created[0]["simulation_run_id"])


class _EnumStatusJob:
    def __init__(self, status: JobStatus) -> None:
        self.status = status
        self.cancelled = False
        self.deleted = False

    def get_status(self, *, refresh: bool) -> JobStatus:
        assert refresh is True
        return self.status

    def cancel(self) -> None:
        self.cancelled = True

    def delete(self) -> None:
        self.deleted = True


class _SingleJobQueue:
    def __init__(self, job: _EnumStatusJob) -> None:
        self.job = job
        self.enqueue_calls: list[dict[str, object]] = []

    def fetch_job(self, job_id: str) -> _EnumStatusJob:
        assert job_id.startswith("decision-simulation-")
        return self.job

    def enqueue_call(self, **kwargs) -> None:
        self.enqueue_calls.append(kwargs)

    def __len__(self) -> int:
        return 1


def test_rq_job_status_enum_is_normalized_and_live_delivery_is_not_duplicated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _run_record(run_id="enum-status-run")
    job = _EnumStatusJob(JobStatus.QUEUED)
    queue = _SingleJobQueue(job)
    executor = service.RQSimulationExecutor(queue=queue)
    monkeypatch.setattr(service, "get_decision_simulation_run", lambda _: run)
    monkeypatch.setattr(service, "get_settings", lambda: _settings("enforced"))

    assert executor.delivery_status(run.id) == "queued"
    assert service.enqueue_decision_simulation(run.id, queue=queue) == (
        "decision-simulation-enum-status-run"
    )
    assert queue.enqueue_calls == []
    assert job.cancelled is False
    assert job.deleted is False


def _shadow_critic_execution(verdict: str) -> DecisionCriticExecution:
    if verdict == "recalculate":
        issue = DecisionCriticIssue(
            code="stale_evidence",
            message="Snapshot устарел.",
            evidenceRefs=["evidence:stale"],
            affectedCandidates=["candidate-1"],
        )
    else:
        issue = DecisionCriticIssue(
            code="unsafe_execution",
            message="Исполнение небезопасно.",
            evidenceRefs=["evidence:unsafe"],
            affectedCandidates=["candidate-1"],
        )
    return DecisionCriticExecution(
        mode="shadow",
        result=DecisionCriticResult(
            verdict=verdict,
            issues=[issue],
            summary=issue.message,
        ),
        effectiveAction="record_only",
        llmAvailable=True,
        blocksDecision=False,
        rankingFingerprint="f" * 64,
    )


def _patch_shadow_worker(
    monkeypatch: pytest.MonkeyPatch,
    *,
    critic: DecisionCriticExecution,
):
    initial = replace(_job_run(_ready_workflow()), mode="shadow")
    state = initial
    events: list[dict[str, object]] = []

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

    @contextmanager
    def acquired_lock(simulation_run_id: str):
        assert simulation_run_id == initial.id
        yield True

    monkeypatch.setattr(jobs, "get_decision_simulation_run", get_run)
    monkeypatch.setattr(jobs, "update_decision_simulation_run", update_run)
    monkeypatch.setattr(jobs, "update_decision_simulation_progress", update_progress)
    monkeypatch.setattr(jobs, "_simulation_run_lock", acquired_lock)
    monkeypatch.setattr(jobs, "list_decision_simulation_batches", lambda *a, **k: [])
    monkeypatch.setattr(jobs, "list_decision_simulation_candidates", lambda *a, **k: [])
    monkeypatch.setattr(
        jobs,
        "upsert_decision_simulation_batch",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(jobs, "upsert_decision_simulation_candidate", lambda **_: None)
    monkeypatch.setattr(
        jobs,
        "append_decision_simulation_event",
        lambda **kwargs: events.append(kwargs),
    )
    monkeypatch.setattr(jobs, "run_decision_critic", lambda **_: critic)
    monkeypatch.setattr(
        jobs,
        "get_settings",
        lambda: SimpleNamespace(decision_simulation_llm_critic_enabled=True),
    )
    monkeypatch.setattr(jobs, "clear_decision_simulation_winner_flags", Mock())
    return initial, lambda: state, events


def test_shadow_valid_recalculate_supersedes_run_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial, state, events = _patch_shadow_worker(
        monkeypatch,
        critic=_shadow_critic_execution("recalculate"),
    )
    refreshed = {"schemaVersion": "fresh"}
    monkeypatch.setattr(jobs, "_refreshed_snapshot", lambda _: refreshed)
    replacement = SimpleNamespace(id="replacement-run")
    create_recalculation = Mock(return_value=replacement)
    monkeypatch.setattr(jobs, "create_critic_recalculation", create_recalculation)
    publish_shadow = Mock()
    monkeypatch.setattr(jobs, "_publish_shadow_decision", publish_shadow)
    monkeypatch.setattr(
        jobs,
        "complete_decision_simulation_run",
        Mock(side_effect=AssertionError("recalculation completed the old run")),
    )
    monkeypatch.setattr(
        jobs,
        "block_manager_decision_for_simulation",
        Mock(side_effect=AssertionError("shadow recalculation blocked public decision")),
    )

    outcome = jobs.run_decision_simulation_job(initial.id)

    assert outcome == {
        "simulation_run_id": initial.id,
        "status": "superseded",
        "replacement_run_id": replacement.id,
    }
    create_recalculation.assert_called_once()
    recalculated_run = create_recalculation.call_args.args[0]
    assert recalculated_run.id == initial.id
    assert recalculated_run.mode == "shadow"
    assert recalculated_run.status == "running"
    assert create_recalculation.call_args.kwargs == {"evidence_snapshot": refreshed}
    assert any(item["event_type"] == "critic_completed" for item in events)
    assert state().winner_candidate_id is None
    publish_shadow.assert_not_called()


def test_shadow_valid_blocker_blocks_only_simulation_and_preserves_public_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial, state, events = _patch_shadow_worker(
        monkeypatch,
        critic=_shadow_critic_execution("blocked"),
    )
    publish_shadow = Mock()
    public_block = Mock(
        side_effect=AssertionError("valid shadow blocker changed public ManagerDecision")
    )
    monkeypatch.setattr(jobs, "_publish_shadow_decision", publish_shadow)
    monkeypatch.setattr(jobs, "block_manager_decision_for_simulation", public_block)
    monkeypatch.setattr(
        jobs,
        "create_critic_recalculation",
        Mock(side_effect=AssertionError("blocked verdict started recalculation")),
    )
    monkeypatch.setattr(
        jobs,
        "complete_decision_simulation_run",
        Mock(side_effect=AssertionError("blocked shadow run was completed")),
    )

    outcome = jobs.run_decision_simulation_job(initial.id)

    assert outcome == {"simulation_run_id": initial.id, "status": "blocked"}
    assert state().status == "blocked"
    assert state().winner_candidate_id is None
    assert any(item["event_type"] == "blocked" for item in events)
    publish_shadow.assert_called_once()
    published_run = publish_shadow.call_args.args[0]
    assert published_run.id == initial.id
    assert published_run.mode == "shadow"
    assert published_run.status == "running"
    public_block.assert_not_called()


def test_shadow_recalculation_snapshot_never_replaces_original_public_ranking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = _ready_workflow().decision
    refreshed = original.model_copy(
        update={"ranked_actions": list(reversed(original.ranked_actions))}
    )
    initial = _job_run(_ready_workflow())
    replacement = replace(
        initial,
        mode="shadow",
        evidence_snapshot={
            **initial.evidence_snapshot,
            "managerDecision": refreshed.model_dump(mode="json", by_alias=True),
            "shadowPublicDecision": original.model_dump(mode="json", by_alias=True),
        },
    )
    persisted = []
    monkeypatch.setattr(
        jobs,
        "persist_manager_decision_bundle",
        lambda decision, **kwargs: persisted.append((decision, kwargs)),
    )
    monkeypatch.setattr(
        jobs,
        "set_manager_decision_simulation_run",
        lambda *args, **kwargs: None,
    )

    jobs._publish_shadow_decision(replacement)

    public = persisted[0][0]
    assert [action.action_id for action in public.ranked_actions] == [
        action.action_id for action in original.ranked_actions
    ]
    assert public.ranked_actions != refreshed.ranked_actions


def test_sse_ignores_unrelated_notify_without_querying_event_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeConnection:
        def __init__(self) -> None:
            self.close = AsyncMock()

    async def exercise():
        connection = FakeConnection()
        load_page = AsyncMock(return_value=[])
        unrelated_sent = False
        keep_listener_open = asyncio.Event()

        async def wait_for_notification(_connection):
            nonlocal unrelated_sent
            if not unrelated_sent:
                unrelated_sent = True
                return SimpleNamespace(
                    payload=json.dumps({"simulation_run_id": "another-run"})
                )
            await keep_listener_open.wait()

        wait = AsyncMock(side_effect=wait_for_notification)
        monkeypatch.setattr(
            simulation_router,
            "_connect_simulation_event_listener",
            AsyncMock(return_value=connection),
        )
        monkeypatch.setattr(simulation_router, "_load_simulation_event_page", load_page)
        monkeypatch.setattr(simulation_router, "_wait_for_simulation_notification", wait)
        monkeypatch.setattr(simulation_router, "_SSE_HEARTBEAT_SECONDS", 0.001)
        stream = simulation_router._decision_simulation_event_stream("target-run")
        # Python 3.9 has no built-in ``anext``; call the async iterator
        # protocol directly so this regression test runs on every supported
        # local interpreter.
        chunk = await stream.__anext__()
        await stream.aclose()
        return chunk, load_page, wait, connection

    chunk, load_page, wait, connection = asyncio.run(exercise())

    assert chunk == ": heartbeat\n\n"
    load_page.assert_awaited_once_with("target-run", 0)
    assert wait.await_count >= 2
    connection.close.assert_awaited_once()


def test_two_sequential_notification_waits_fully_release_fake_connection_lock() -> None:
    notification = SimpleNamespace(payload='{"simulation_run_id":"run-1"}')

    class OneNotificationIterator:
        def __init__(self, connection) -> None:
            self.connection = connection
            self.sent = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self.sent:
                self.sent = True
                return notification
            self.connection.locked = False
            raise StopAsyncIteration

    class FakeConnection:
        def __init__(self) -> None:
            self.locked = False
            self.calls = 0

        def notifies(self, *, timeout: int, stop_after: int):
            assert timeout == 15
            assert stop_after == 1
            if self.locked:
                raise RuntimeError("previous notification iterator still owns lock")
            self.locked = True
            self.calls += 1
            return OneNotificationIterator(self)

    async def exercise() -> tuple[object, object, FakeConnection]:
        connection = FakeConnection()
        first = await asyncio.wait_for(
            simulation_router._wait_for_simulation_notification(connection),
            timeout=0.5,
        )
        second = await asyncio.wait_for(
            simulation_router._wait_for_simulation_notification(connection),
            timeout=0.5,
        )
        return first, second, connection

    first, second, connection = asyncio.run(exercise())

    assert first is notification
    assert second is notification
    assert connection.calls == 2
    assert connection.locked is False


def test_reconciler_does_not_requeue_a_live_queued_rq_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)
    run = replace(
        _run_record(run_id="live-queued", status="queued"),
        heartbeat_at=current - timedelta(minutes=30),
    )
    job = _EnumStatusJob(JobStatus.QUEUED)
    queue = _SingleJobQueue(job)
    executor = service.RQSimulationExecutor(queue=queue)
    monkeypatch.setattr(
        jobs,
        "get_settings",
        lambda: SimpleNamespace(decision_simulation_stale_after_seconds=1_200),
    )
    monkeypatch.setattr(
        jobs,
        "list_stale_decision_simulation_runs",
        lambda **_: [run],
    )
    monkeypatch.setattr(
        jobs,
        "list_unprojected_terminal_decision_simulation_runs",
        lambda **_: [],
    )
    monkeypatch.setattr(
        jobs,
        "update_decision_simulation_run",
        Mock(side_effect=AssertionError("live queued job was moved to queue tail")),
    )
    monkeypatch.setattr(
        jobs,
        "append_decision_simulation_event",
        Mock(side_effect=AssertionError("live queued job emitted recovery event")),
    )

    result = jobs.reconcile_decision_simulations(now=current, executor=executor)

    assert result == {"stale": 1, "recovered": []}
    assert queue.enqueue_calls == []
    assert job.cancelled is False
    assert job.deleted is False


def test_retry_idempotency_key_is_scoped_to_direct_parent_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = _run_record(run_id="parent-a", status="blocked")
    wrong_parent_child = _run_record(
        run_id="child-of-b",
        status="queued",
        decision_id=parent.decision_id,
        idempotency_key="same-key",
        retry_of_run_id="parent-b",
        retry_attempt=1,
    )
    correct_child = replace(
        wrong_parent_child,
        id="child-of-a",
        retry_of_run_id=parent.id,
    )
    monkeypatch.setattr(
        service,
        "get_decision_simulation_run",
        lambda simulation_run_id: parent if simulation_run_id == parent.id else None,
    )
    monkeypatch.setattr(
        service,
        "list_decision_simulation_runs",
        lambda **_: [wrong_parent_child, correct_child],
    )
    create_retry = Mock(side_effect=AssertionError("idempotent retry was recreated"))
    monkeypatch.setattr(service, "_create_retry_run", create_retry)

    returned = service.retry_decision_simulation(
        simulation_run_id=parent.id,
        idempotency_key="same-key",
        actor=ManagerActor(user_id=1, username="admin", role="admin"),
    )

    assert returned is correct_child
    create_retry.assert_not_called()
    normalized_schema = " ".join(MEMORY_SCHEMA_SQL.split())
    assert (
        "on decision_simulation_runs(retry_of_run_id, idempotency_key) "
        "where retry_of_run_id is not null and idempotency_key is not null"
        in normalized_schema
    )
