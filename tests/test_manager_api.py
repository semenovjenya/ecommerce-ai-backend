from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from unittest.mock import patch

from fastapi.testclient import TestClient

from agi_ecom_app.api import app
from agi_ecom_app.manager_lifecycle.auth import LoginResult, ManagerActor
from tests.test_manager_lifecycle import _experiment


client = TestClient(app)


def test_manager_read_requires_bearer_token() -> None:
    response = client.get("/v1/manager/actions/action-1")
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_login_returns_public_token_contract() -> None:
    actor = ManagerActor(user_id=2, username="manager", role="manager")
    with patch(
        "agi_ecom_app.action_board.router.authenticate",
        return_value=LoginResult(
            access_token="signed-token",
            token_type="bearer",
            expires_in=900,
            actor=actor,
        ),
    ):
        response = client.post(
            "/v1/auth/login",
            json={"username": "manager", "password": "secret"},
        )
    assert response.status_code == 200
    assert response.json() == {
        "accessToken": "signed-token",
        "tokenType": "bearer",
        "expiresIn": 900,
        "user": {"id": 2, "username": "manager", "role": "manager"},
    }


def test_approval_requires_idempotency_key_and_returns_applying_state() -> None:
    actor = ManagerActor(user_id=2, username="manager", role="manager")
    applying = replace(
        _experiment(),
        state="applying",
        approved_by=2,
        approval_idempotency_key="approve-1",
        external_operation_id="wb-op-1",
    )
    with patch(
        "agi_ecom_app.action_board.router.decode_access_token",
        return_value=actor,
    ):
        missing_key = client.post(
            "/v1/manager/actions/experiment-1/approve",
            headers={"Authorization": "Bearer signed-token"},
        )
    assert missing_key.status_code == 422

    with (
        patch(
            "agi_ecom_app.action_board.router.decode_access_token",
            return_value=actor,
        ),
        patch(
            "agi_ecom_app.action_board.router.approve_manager_action",
            return_value=applying,
        ) as approve,
        patch(
            "agi_ecom_app.action_board.router.list_agent_action_measurements",
            return_value=[],
        ),
    ):
        response = client.post(
            "/v1/manager/actions/experiment-1/approve",
            headers={
                "Authorization": "Bearer signed-token",
                "Idempotency-Key": "approve-1",
            },
        )
    assert response.status_code == 200
    assert response.json()["state"] == "applying"
    assert response.json()["externalOperationId"] == "wb-op-1"
    assert approve.call_args.kwargs["idempotency_key"] == "approve-1"


def test_manual_completion_records_actual_delivery_time() -> None:
    actor = ManagerActor(user_id=2, username="manager", role="manager")
    actual_at = datetime(2026, 7, 13, 8, 0, tzinfo=timezone.utc)
    measuring = replace(
        _experiment(mode="human_task"),
        family="supply",
        state="measuring",
        applied_at=actual_at,
    )
    with (
        patch("agi_ecom_app.action_board.router.decode_access_token", return_value=actor),
        patch(
            "agi_ecom_app.action_board.router.complete_manual_manager_action",
            return_value=measuring,
        ) as complete,
        patch(
            "agi_ecom_app.action_board.router.list_agent_action_measurements",
            return_value=[],
        ),
    ):
        response = client.post(
            "/v1/manager/actions/experiment-1/complete",
            headers={
                "Authorization": "Bearer signed-token",
                "Idempotency-Key": "delivery-1",
            },
            json={"actualAt": actual_at.isoformat()},
        )

    assert response.status_code == 200
    assert response.json()["state"] == "measuring"
    assert complete.call_args.kwargs["actual_at"] == actual_at
