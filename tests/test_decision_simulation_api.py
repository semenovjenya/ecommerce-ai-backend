from __future__ import annotations

import asyncio
import sys
from dataclasses import replace
from datetime import datetime, timezone
from types import ModuleType
from unittest.mock import AsyncMock, patch

import psycopg
from fastapi.testclient import TestClient

from agi_ecom_app.api import app
from agi_ecom_app.manager_lifecycle.auth import ManagerActor
from agi_ecom_app.services.memory import (
    DecisionSimulationCandidateRecord,
    DecisionSimulationEventRecord,
    DecisionSimulationRunRecord,
)
from agi_ecom_app.action_board import router as simulation_router
from tests.test_manager_lifecycle import _decision


client = TestClient(app)
NOW = datetime(2026, 7, 16, 9, 0, tzinfo=timezone.utc)
VIEWER = ManagerActor(user_id=1, username="viewer", role="viewer")
ADMIN = ManagerActor(user_id=3, username="admin", role="admin")


def _run(**overrides) -> DecisionSimulationRunRecord:
    values = {
        "id": "simulation-1",
        "decision_id": "decision-1",
        "chat_id": "chat-1",
        "run_id": 10,
        "mode": "enforced",
        "status": "running",
        "objective": "incremental_contribution_profit",
        "model_version": "advertising-bootstrap-v1",
        "evidence_snapshot": {"candidateIds": ["candidate-1"]},
        "evidence_hash": "a" * 64,
        "master_seed": 101,
        "sample_count": 500,
        "batch_size": 50,
        "candidate_count": 7,
        "completed_samples": 700,
        "total_samples": 3500,
        "completed_candidates": 1,
        "progress": 0.2,
        "provisional_winner_candidate_id": "candidate-1",
        "winner_candidate_id": None,
        "critic_result": {},
        "error": {},
        "idempotency_key": None,
        "retry_of_run_id": None,
        "superseded_by_run_id": None,
        "retry_attempt": 0,
        "heartbeat_at": NOW,
        "started_at": NOW,
        "completed_at": None,
        "created_at": NOW,
        "updated_at": NOW,
    }
    values.update(overrides)
    return DecisionSimulationRunRecord(**values)


def _candidate(**overrides) -> DecisionSimulationCandidateRecord:
    values = {
        "simulation_run_id": "simulation-1",
        "candidate_id": "candidate-1",
        "action_id": "action-1",
        "status": "completed",
        "candidate_payload": {"operation": "pause_campaign"},
        "evidence_refs": ["fact:spend"],
        "constraint_violations": [],
        "distribution_summary": {"per_day": {"p10": 110.0, "p50": 150.0}},
        "score_breakdown": {"cvar10_per_day": 100.0},
        "expected_effect_mean": 155.0,
        "profit_p10": 110.0,
        "cvar_10_per_day": 100.0,
        "execution_risk": 0.1,
        "utility_score": 100.0,
        "rank": 1,
        "is_winner": True,
        "completed_samples": 500,
        "error": {},
        "completed_at": NOW,
        "created_at": NOW,
        "updated_at": NOW,
    }
    values.update(overrides)
    return DecisionSimulationCandidateRecord(**values)


def _event(event_id: int = 18, **overrides) -> DecisionSimulationEventRecord:
    values = {
        "id": event_id,
        "event_id": f"event-{event_id}",
        "simulation_run_id": "simulation-1",
        "event_type": "batch_completed",
        "candidate_id": "candidate-1",
        "batch_index": 2,
        "progress": 0.2,
        "payload": {"completed_samples": 700, "total_samples": 3500},
        "created_at": NOW,
    }
    values.update(overrides)
    return DecisionSimulationEventRecord(**values)


def _auth_headers(**extra: str) -> dict[str, str]:
    return {"Authorization": "Bearer signed-token", **extra}


def test_simulation_run_requires_manager_jwt_and_returns_ranking() -> None:
    missing_auth = client.get("/v1/manager/simulations/simulation-1")
    assert missing_auth.status_code == 401

    with (
        patch(
            "agi_ecom_app.action_board.router.decode_access_token",
            return_value=VIEWER,
        ),
        patch(
            "agi_ecom_app.action_board.router.get_decision_simulation_run",
            return_value=_run(),
        ),
        patch(
            "agi_ecom_app.action_board.router.list_decision_simulation_candidates",
            return_value=[_candidate()],
        ),
    ):
        response = client.get(
            "/v1/manager/simulations/simulation-1",
            headers=_auth_headers(),
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["progressCompleted"] == 700
    assert payload["progressTotal"] == 3500
    assert payload["progressPct"] == 20.0
    assert payload["provisionalLeaderId"] == "candidate-1"
    assert payload["winnerCandidateId"] is None
    assert payload["snapshotHash"] == "a" * 64
    assert payload["candidates"][0]["distribution"]["per_day"]["p10"] == 110.0
    assert payload["candidates"][0]["winner"] is True


def test_manager_decision_includes_linked_simulation_summary() -> None:
    decision = replace(_decision(), simulation_run_id="simulation-1")
    with (
        patch(
            "agi_ecom_app.action_board.router.decode_access_token",
            return_value=VIEWER,
        ),
        patch(
            "agi_ecom_app.action_board.router.get_decision_bundle",
            return_value=(decision, []),
        ),
        patch(
            "agi_ecom_app.action_board.router.get_decision_simulation_run",
            return_value=_run(status="completed", winner_candidate_id="candidate-1"),
        ),
    ):
        response = client.get(
            "/v1/manager/decisions/decision-1",
            headers=_auth_headers(),
        )

    assert response.status_code == 200
    summary = response.json()["simulationSummary"]
    assert summary["runId"] == "simulation-1"
    assert summary["status"] == "completed"
    assert summary["winnerCandidateId"] == "candidate-1"


def test_manager_decision_returns_simulation_eligibility_without_a_run() -> None:
    eligibility = {
        "outcome": "insufficient",
        "mode": "shadow",
        "submittedCandidateCount": 3,
        "eligibleCandidateCount": 0,
        "rejectedCandidateCount": 3,
        "minimumCandidateCount": 3,
        "maximumCandidateCount": 7,
        "reasonCode": "insufficient_eligible_candidates",
    }
    decision = replace(
        _decision(),
        simulation_run_id=None,
        public_json={"simulationEligibility": eligibility},
        decision_json={"simulationEligibility": eligibility},
    )
    with (
        patch(
            "agi_ecom_app.action_board.router.decode_access_token",
            return_value=VIEWER,
        ),
        patch(
            "agi_ecom_app.action_board.router.get_decision_bundle",
            return_value=(decision, []),
        ),
        patch(
            "agi_ecom_app.action_board.router.get_latest_decision_simulation_run_for_decision",
            return_value=None,
        ),
    ):
        response = client.get(
            "/v1/manager/decisions/decision-1",
            headers=_auth_headers(),
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["simulationSummary"] is None
    assert payload["simulationEligibility"] == eligibility


def test_events_endpoint_replays_after_requested_event_id() -> None:
    with (
        patch(
            "agi_ecom_app.action_board.router.decode_access_token",
            return_value=VIEWER,
        ),
        patch(
            "agi_ecom_app.action_board.router.get_decision_simulation_run",
            return_value=_run(),
        ),
        patch(
            "agi_ecom_app.action_board.router.list_decision_simulation_events",
            return_value=[_event()],
        ) as list_events,
    ):
        response = client.get(
            "/v1/manager/simulations/simulation-1/events",
            params={"afterId": 17, "limit": 25},
            headers=_auth_headers(),
        )

    assert response.status_code == 200
    assert response.json()["items"] == [
        {
            "id": 18,
            "runId": "simulation-1",
            "eventType": "batch_completed",
            "candidateId": "candidate-1",
            "batchIndex": 2,
            "progressCompleted": 700,
            "progressTotal": 3500,
            "payload": {"completed_samples": 700, "total_samples": 3500},
            "createdAt": NOW.isoformat().replace("+00:00", "Z"),
        }
    ]
    assert list_events.call_args.kwargs == {"after_id": 17, "limit": 25}


def test_sse_replays_last_event_id_after_listen_without_polling() -> None:
    class FakeConnection:
        def __init__(self) -> None:
            self.close = AsyncMock()

    async def exercise() -> tuple[str, AsyncMock, FakeConnection]:
        connection = FakeConnection()
        load_page = AsyncMock(return_value=[_event(18)])
        connect = AsyncMock(return_value=connection)
        with (
            patch.object(simulation_router, "_connect_simulation_event_listener", connect),
            patch.object(simulation_router, "_load_simulation_event_page", load_page),
        ):
            stream = simulation_router._decision_simulation_event_stream(
                "simulation-1",
                after_id=17,
            )
            chunk = await anext(stream)
            await stream.aclose()
        return chunk, load_page, connection

    chunk, load_page, connection = asyncio.run(exercise())

    assert chunk.startswith("id: 18\nevent: batch_completed\n")
    assert '"progressCompleted":700' in chunk
    load_page.assert_awaited_once_with("simulation-1", 17)
    connection.close.assert_awaited_once()


def test_sse_heartbeats_are_per_stream_and_do_not_poll_postgres() -> None:
    class FakeConnection:
        def __init__(self) -> None:
            self.close = AsyncMock()

    async def exercise() -> tuple[list[str], AsyncMock]:
        connection = FakeConnection()
        load_page = AsyncMock(return_value=[])
        with (
            patch.object(
                simulation_router,
                "_connect_simulation_event_listener",
                AsyncMock(return_value=connection),
            ),
            patch.object(
                simulation_router,
                "_load_simulation_event_page",
                load_page,
            ),
            patch.object(
                simulation_router,
                "_wait_for_simulation_notification",
                AsyncMock(return_value=None),
            ),
            patch.object(simulation_router, "_SSE_HEARTBEAT_SECONDS", 0.001),
        ):
            stream = simulation_router._decision_simulation_event_stream("simulation-1")
            chunks = [await anext(stream), await anext(stream)]
            await stream.aclose()
            return chunks, load_page

    chunks, load_page = asyncio.run(exercise())

    assert chunks == [": heartbeat\n\n", ": heartbeat\n\n"]
    # Initial replay closes the SELECT/LISTEN race. Subsequent idle heartbeats
    # keep the connection alive without turning SSE into a slower DB poller.
    load_page.assert_awaited_once_with("simulation-1", 0)


def test_sse_reconnect_replays_from_durable_cursor() -> None:
    class FakeConnection:
        def __init__(self) -> None:
            self.close = AsyncMock()

    async def exercise() -> tuple[str, str, AsyncMock]:
        first_connection = FakeConnection()
        second_connection = FakeConnection()
        load_page = AsyncMock(side_effect=[[], [_event(18)]])
        with (
            patch.object(
                simulation_router,
                "_connect_simulation_event_listener",
                AsyncMock(side_effect=[first_connection, second_connection]),
            ),
            patch.object(simulation_router, "_load_simulation_event_page", load_page),
            patch.object(
                simulation_router,
                "_wait_for_simulation_notification",
                AsyncMock(side_effect=psycopg.OperationalError("connection lost")),
            ),
            patch.object(simulation_router.asyncio, "sleep", AsyncMock()),
        ):
            stream = simulation_router._decision_simulation_event_stream(
                "simulation-1",
                after_id=17,
            )
            reconnect_chunk = await anext(stream)
            replay_chunk = await anext(stream)
            await stream.aclose()
        return reconnect_chunk, replay_chunk, load_page

    reconnect_chunk, replay_chunk, load_page = asyncio.run(exercise())

    assert reconnect_chunk == ": reconnecting\n\n"
    assert replay_chunk.startswith("id: 18\nevent: batch_completed\n")
    assert load_page.await_args_list[0].args == ("simulation-1", 17)
    assert load_page.await_args_list[1].args == ("simulation-1", 17)


def test_retry_is_admin_only_and_passes_idempotency_key_to_service() -> None:
    retry_module = ModuleType("agi_ecom_app.decision_simulation.service")
    calls: list[dict] = []

    def retry_decision_simulation(**kwargs):
        calls.append(kwargs)
        return _run(
            id="simulation-2",
            status="queued",
            completed_samples=0,
            completed_candidates=0,
            progress=0,
            provisional_winner_candidate_id=None,
            retry_of_run_id="simulation-1",
            retry_attempt=1,
            started_at=None,
            heartbeat_at=None,
        )

    retry_module.retry_decision_simulation = retry_decision_simulation

    with patch(
        "agi_ecom_app.action_board.router.decode_access_token",
        return_value=VIEWER,
    ):
        forbidden = client.post(
            "/v1/manager/simulations/simulation-1/retry",
            headers=_auth_headers(**{"Idempotency-Key": "retry-1"}),
        )
    assert forbidden.status_code == 403

    with (
        patch(
            "agi_ecom_app.action_board.router.decode_access_token",
            return_value=ADMIN,
        ),
        patch.dict(sys.modules, {retry_module.__name__: retry_module}),
        patch(
            "agi_ecom_app.action_board.router.list_decision_simulation_candidates",
            return_value=[],
        ),
    ):
        response = client.post(
            "/v1/manager/simulations/simulation-1/retry",
            headers=_auth_headers(**{"Idempotency-Key": "retry-1"}),
        )

    assert response.status_code == 202
    assert response.json()["runId"] == "simulation-2"
    assert response.json()["status"] == "queued"
    assert calls == [
        {
            "simulation_run_id": "simulation-1",
            "idempotency_key": "retry-1",
            "actor": ADMIN,
        }
    ]
