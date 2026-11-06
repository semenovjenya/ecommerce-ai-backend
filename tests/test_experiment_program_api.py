from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

from fastapi.testclient import TestClient

from agi_ecom_app.api import app
from agi_ecom_app.experiment_program.models import (
    ExperimentNegotiationMessageView,
    ExperimentProgramBundleView,
)
from agi_ecom_app.manager_lifecycle.auth import ManagerActor


client = TestClient(app)
NOW = datetime(2026, 7, 20, 10, 0, tzinfo=timezone.utc)


def _bundle() -> ExperimentProgramBundleView:
    return ExperimentProgramBundleView.model_validate(
        {
            "program": {
                "id": "program-1",
                "objective": "incremental_contribution_profit",
                "mode": "shadow",
                "status": "proposing",
                "target": {"nmId": 123},
                "domains": ["pricing"],
                "generatorVersion": "frontier-v1",
                "snapshotHash": "a" * 64,
                "currentEpoch": 1,
                "experimentBudget": {},
                "domainCoverage": {},
                "error": {},
                "version": 1,
                "createdAt": NOW,
                "updatedAt": NOW,
            },
            "snapshot": {
                "id": "snapshot-1",
                "programId": "program-1",
                "epoch": 1,
                "snapshotHash": "a" * 64,
                "status": "ready",
                "evidencePackets": [],
                "qualitySummary": {},
                "createdAt": NOW,
            },
            "frontiers": [],
            "candidates": [],
        }
    )


def test_experiment_program_reads_require_authentication() -> None:
    response = client.get("/v1/manager/experiment-programs/program-1")
    assert response.status_code == 401


def test_create_program_requires_manager_and_preserves_idempotent_status() -> None:
    actor = ManagerActor(user_id=2, username="manager", role="manager")
    with (
        patch(
            "agi_ecom_app.experiment_program.router.decode_access_token",
            return_value=actor,
        ),
        patch(
            "agi_ecom_app.experiment_program.router.create_experiment_program",
            return_value=(_bundle(), False),
        ) as create,
    ):
        response = client.post(
            "/v1/manager/experiment-programs",
            headers={
                "Authorization": "Bearer signed-token",
                "Idempotency-Key": "program-key",
            },
            json={
                "domains": ["pricing"],
                "candidates": [
                    {
                        "domain": "pricing",
                        "family": "pricing",
                        "operation": "set_price",
                    }
                ],
            },
        )

    assert response.status_code == 200
    assert response.json()["program"]["id"] == "program-1"
    assert create.call_args.kwargs["idempotency_key"] == "program-key"


def test_http_message_boundary_is_admin_only() -> None:
    manager = ManagerActor(user_id=2, username="manager", role="manager")
    with patch(
        "agi_ecom_app.experiment_program.router.decode_access_token",
        return_value=manager,
    ):
        response = client.post(
            "/v1/manager/experiment-programs/program-1/messages",
            headers={
                "Authorization": "Bearer signed-token",
                "Idempotency-Key": "message-1",
            },
            json={
                "messageType": "constraint",
                "sourceRole": "finance",
                "payload": {"marginFloor": 0.2},
            },
        )
    assert response.status_code == 403


def test_admin_can_append_typed_message() -> None:
    admin = ManagerActor(user_id=1, username="admin", role="admin")
    stored = ExperimentNegotiationMessageView(
        id=1,
        messageId="message-1",
        programId="program-1",
        epoch=1,
        messageType="constraint",
        sourceRole="finance",
        idempotencyKey="message-key",
        fingerprint="b" * 64,
        payload={"marginFloor": 0.2},
        evidenceRefs=[],
        createdAt=NOW,
    )
    with (
        patch(
            "agi_ecom_app.experiment_program.router.decode_access_token",
            return_value=admin,
        ),
        patch(
            "agi_ecom_app.experiment_program.router.post_experiment_negotiation_message",
            return_value=(stored, True),
        ),
    ):
        response = client.post(
            "/v1/manager/experiment-programs/program-1/messages",
            headers={
                "Authorization": "Bearer signed-token",
                "Idempotency-Key": "message-key",
            },
            json={
                "messageType": "constraint",
                "sourceRole": "finance",
                "payload": {"marginFloor": 0.2},
            },
        )
    assert response.status_code == 201
    assert response.json()["messageType"] == "constraint"
