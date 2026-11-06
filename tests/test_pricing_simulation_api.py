from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

from fastapi.testclient import TestClient

from agi_ecom_app.api import app
from agi_ecom_app.decision_simulation.pricing_source import (
    PricingSimulationSourceError,
)
from agi_ecom_app.decision_simulation.pricing_service import (
    PricingSimulationDeliveryError,
    PricingSimulationIdempotencyConflict,
)
from agi_ecom_app.manager_lifecycle.auth import ManagerActor
from agi_ecom_app.services.memory import DecisionSimulationRunRecord


client = TestClient(app)
NOW = datetime(2026, 7, 17, 9, 0, tzinfo=timezone.utc)
VIEWER = ManagerActor(user_id=1, username="viewer", role="viewer")
MANAGER = ManagerActor(user_id=2, username="manager", role="manager")


def _headers(*, include_key: bool = True) -> dict[str, str]:
    result = {"Authorization": "Bearer signed-token"}
    if include_key:
        result["Idempotency-Key"] = "price-shadow-42"
    return result


def _run(**overrides: object) -> DecisionSimulationRunRecord:
    values: dict[str, object] = {
        "id": "pricing-simulation-1",
        "decision_id": "pricing-decision-1",
        "chat_id": None,
        "run_id": None,
        "mode": "shadow",
        "status": "queued",
        "objective": "incremental_contribution_profit",
        "model_version": "pricing-elasticity-bootstrap-v1",
        "evidence_snapshot": {"domain": "pricing", "selectionMode": "winner_only"},
        "evidence_hash": "a" * 64,
        "master_seed": 42,
        "sample_count": 500,
        "batch_size": 50,
        "candidate_count": 3,
        "completed_samples": 0,
        "total_samples": 1500,
        "completed_candidates": 0,
        "progress": 0.0,
        "provisional_winner_candidate_id": None,
        "winner_candidate_id": None,
        "critic_result": {},
        "error": {},
        "idempotency_key": "price-shadow-42",
        "retry_of_run_id": None,
        "superseded_by_run_id": None,
        "retry_attempt": 0,
        "heartbeat_at": None,
        "started_at": None,
        "completed_at": None,
        "created_at": NOW,
        "updated_at": NOW,
    }
    values.update(overrides)
    return DecisionSimulationRunRecord(**values)  # type: ignore[arg-type]


def test_pricing_simulation_start_requires_jwt_and_manager_role() -> None:
    missing_auth = client.post(
        "/v1/manager/pricing/simulations",
        headers={"Idempotency-Key": "price-shadow-42"},
        json={"nmId": 42},
    )
    assert missing_auth.status_code == 401

    with patch(
        "agi_ecom_app.action_board.router.decode_access_token",
        return_value=VIEWER,
    ):
        forbidden = client.post(
            "/v1/manager/pricing/simulations",
            headers=_headers(),
            json={"nmId": 42},
        )
    assert forbidden.status_code == 403
    assert forbidden.json()["detail"] == "manager role is required"


def test_pricing_simulation_start_passes_explicit_grid_and_returns_observable_run() -> None:
    with (
        patch(
            "agi_ecom_app.action_board.router.decode_access_token",
            return_value=MANAGER,
        ),
        patch(
            "agi_ecom_app.decision_simulation.pricing_service.start_pricing_simulation",
            return_value=_run(),
        ) as start,
        patch(
            "agi_ecom_app.action_board.router.list_decision_simulation_candidates",
            return_value=[],
        ),
    ):
        response = client.post(
            "/v1/manager/pricing/simulations",
            headers=_headers(),
            json={
                "nmId": 42,
                "candidatePrices": [90, 110],
                "horizonDays": 7,
            },
        )

    assert response.status_code == 202
    payload = response.json()
    assert payload["runId"] == "pricing-simulation-1"
    assert payload["status"] == "queued"
    assert payload["mode"] == "shadow"
    assert payload["modelVersion"] == "pricing-elasticity-bootstrap-v1"
    assert payload["progressTotal"] == 1500
    assert payload["evidenceSnapshot"]["selectionMode"] == "winner_only"
    assert start.call_args.kwargs == {
        "nm_id": 42,
        "candidate_prices": [90.0, 110.0],
        "horizon_days": 7,
        "idempotency_key": "price-shadow-42",
        "actor": MANAGER,
    }


def test_pricing_simulation_start_validates_header_and_price_grid() -> None:
    with patch(
        "agi_ecom_app.action_board.router.decode_access_token",
        return_value=MANAGER,
    ):
        missing_key = client.post(
            "/v1/manager/pricing/simulations",
            headers=_headers(include_key=False),
            json={"nmId": 42},
        )
        too_small_grid = client.post(
            "/v1/manager/pricing/simulations",
            headers=_headers(),
            json={"nmId": 42, "candidatePrices": [90]},
        )
        invalid_horizon = client.post(
            "/v1/manager/pricing/simulations",
            headers=_headers(),
            json={"nmId": 42, "horizonDays": 0},
        )
        invalid_price = client.post(
            "/v1/manager/pricing/simulations",
            headers=_headers(),
            json={"nmId": 42, "candidatePrices": [-1, 110]},
        )

    assert missing_key.status_code == 422
    assert too_small_grid.status_code == 422
    assert invalid_horizon.status_code == 422
    assert invalid_price.status_code == 422


def test_pricing_simulation_start_maps_missing_evidence_and_disabled_mode() -> None:
    with (
        patch(
            "agi_ecom_app.action_board.router.decode_access_token",
            return_value=MANAGER,
        ),
        patch(
            "agi_ecom_app.decision_simulation.pricing_service.start_pricing_simulation",
            side_effect=PricingSimulationSourceError("price history is missing"),
        ),
    ):
        missing_evidence = client.post(
            "/v1/manager/pricing/simulations",
            headers=_headers(),
            json={"nmId": 42},
        )

    assert missing_evidence.status_code == 422
    assert missing_evidence.json()["detail"] == "price history is missing"

    with (
        patch(
            "agi_ecom_app.action_board.router.decode_access_token",
            return_value=MANAGER,
        ),
        patch(
            "agi_ecom_app.decision_simulation.pricing_service.start_pricing_simulation",
            side_effect=RuntimeError("pricing decision simulation is disabled"),
        ),
    ):
        disabled = client.post(
            "/v1/manager/pricing/simulations",
            headers=_headers(),
            json={"nmId": 42},
        )

    assert disabled.status_code == 503
    assert disabled.json()["detail"] == "pricing decision simulation is disabled"


def test_pricing_simulation_start_maps_idempotency_conflict() -> None:
    with (
        patch(
            "agi_ecom_app.action_board.router.decode_access_token",
            return_value=MANAGER,
        ),
        patch(
            "agi_ecom_app.decision_simulation.pricing_service.start_pricing_simulation",
            side_effect=PricingSimulationIdempotencyConflict(
                "Idempotency-Key was already used with a different pricing request"
            ),
        ),
    ):
        response = client.post(
            "/v1/manager/pricing/simulations",
            headers=_headers(),
            json={"nmId": 42},
        )

    assert response.status_code == 409
    assert response.json()["detail"] == (
        "Idempotency-Key was already used with a different pricing request"
    )


def test_pricing_simulation_start_exposes_failed_run_id_on_delivery_error() -> None:
    with (
        patch(
            "agi_ecom_app.action_board.router.decode_access_token",
            return_value=MANAGER,
        ),
        patch(
            "agi_ecom_app.decision_simulation.pricing_service.start_pricing_simulation",
            side_effect=PricingSimulationDeliveryError(
                "pricing-failed-run",
                "redis is unavailable",
            ),
        ),
    ):
        response = client.post(
            "/v1/manager/pricing/simulations",
            headers=_headers(),
            json={"nmId": 42},
        )

    assert response.status_code == 503
    assert response.json()["detail"] == {
        "message": "redis is unavailable",
        "simulationRunId": "pricing-failed-run",
    }
