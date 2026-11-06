from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient
import httpx
import jwt

from agi_ecom_app.api import app
from agi_ecom_app.manager_lifecycle.auth import ManagerActor
from agi_ecom_app.services.memory import AppUserRecord


client = TestClient(app)
THREAD_ID = str(uuid4())


def test_state_read_rejects_missing_or_invalid_user_auth() -> None:
    assert client.get(f"/v2/agent/threads/{THREAD_ID}/state").status_code == 401
    assert client.get(
        f"/v2/agent/threads/{THREAD_ID}/state",
        headers={"Authorization": "not-bearer token"},
    ).status_code == 401


def test_state_read_validates_signed_jwt_before_internal_request(monkeypatch) -> None:
    monkeypatch.setenv("MANAGER_JWT_SECRET", "synthetic-signing-secret-" + "x" * 32)
    monkeypatch.setenv("COMMERCE_AGENT_INTERNAL_URL", "http://agent.internal:8123")
    monkeypatch.setenv("COMMERCE_AGENT_SERVICE_TOKEN", "synthetic-service-token-" + "x" * 32)
    monkeypatch.setenv("COMMERCE_AGENT_TENANT_ID", "configured-tenant")
    monkeypatch.setenv("COMMERCE_AGENT_ENVIRONMENT", "development")
    now = datetime.now(timezone.utc)
    actor_record = AppUserRecord(
        id=47,
        username="owner@example.test",
        password_hash="synthetic-unused-hash",
        role="viewer",
        is_active=True,
        created_at=now,
        updated_at=now,
    )
    token = jwt.encode(
        {
            "sub": "47", "username": actor_record.username, "role": "viewer",
            "iat": now, "exp": now + timedelta(minutes=5), "iss": "ai-ecom-manager",
        },
        "synthetic-signing-secret-" + "x" * 32,
        algorithm="HS256",
    )
    seen: list[httpx.Request] = []
    real_async_client = httpx.AsyncClient

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"values": {}, "checkpoint": None, "next": []})

    def factory(*args, **kwargs):
        return real_async_client(*args, transport=httpx.MockTransport(respond), **kwargs)

    with (
        patch("agi_ecom_app.manager_lifecycle.auth.get_app_user", return_value=actor_record),
        patch("agi_ecom_app.agent_v2_read_api.httpx.AsyncClient", side_effect=factory),
    ):
        accepted = client.get(
            f"/v2/agent/threads/{THREAD_ID}/state",
            headers={"Authorization": f"Bearer {token}"},
        )
        rejected = client.get(
            f"/v2/agent/threads/{THREAD_ID}/state",
            headers={"Authorization": "Bearer invalid-token"},
        )
    assert accepted.status_code == 200
    assert rejected.status_code == 401
    assert len(seen) == 1
    assert seen[0].headers["x-commerce-user-id"] == "47"
    assert "authorization" not in seen[0].headers


def test_state_and_history_use_verified_identity_and_server_only_token(monkeypatch) -> None:
    monkeypatch.setenv("COMMERCE_AGENT_INTERNAL_URL", "http://agent.internal:8123")
    monkeypatch.setenv("COMMERCE_AGENT_SERVICE_TOKEN", "synthetic-service-token-" + "x" * 32)
    monkeypatch.setenv("COMMERCE_AGENT_TENANT_ID", "configured-tenant")
    monkeypatch.setenv("COMMERCE_AGENT_ENVIRONMENT", "development")
    seen: list[httpx.Request] = []
    real_async_client = httpx.AsyncClient

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path.endswith("/state"):
            return httpx.Response(
                200,
                json={
                    "values": {"schema_version": 1, "agent_statuses": {"sales": {"status": "planned"}}},
                    "checkpoint": {"checkpoint_id": "synthetic-checkpoint"},
                    "next": [],
                },
            )
        return httpx.Response(200, json={"items": []})

    def factory(*args, **kwargs):
        assert kwargs["trust_env"] is False
        return real_async_client(*args, transport=httpx.MockTransport(respond), **kwargs)

    actor = ManagerActor(user_id=47, username="owner", role="viewer")
    with (
        patch("agi_ecom_app.agent_v2_read_api.decode_access_token", return_value=actor),
        patch("agi_ecom_app.agent_v2_read_api.httpx.AsyncClient", side_effect=factory),
    ):
        state = client.get(
            f"/v2/agent/threads/{THREAD_ID}/state",
            headers={
                "Authorization": "Bearer synthetic-user-jwt",
                "X-Commerce-Tenant-ID": "forged-tenant",
                "X-Commerce-User-ID": "999",
            },
        )
        history = client.get(
            f"/v2/agent/threads/{THREAD_ID}/history?limit=7",
            headers={"Authorization": "Bearer synthetic-user-jwt"},
        )

    assert state.status_code == 200
    assert state.json()["values"]["agent_statuses"]["sales"]["status"] == "planned"
    assert state.headers["cache-control"] == "no-store"
    assert history.status_code == 200
    assert history.json() == {"items": []}
    assert history.headers["cache-control"] == "no-store"
    assert len(seen) == 2
    for request in seen:
        assert request.headers["x-commerce-tenant-id"] == "configured-tenant"
        assert request.headers["x-commerce-user-id"] == "47"
        assert request.headers["x-commerce-service-token"].startswith("synthetic-service-token-")
        assert "authorization" not in request.headers
        assert request.url.host == "agent.internal"
    assert seen[1].url.params["limit"] == "7"


def test_read_boundary_fails_closed_for_missing_config_and_bad_upstream(monkeypatch) -> None:
    actor = ManagerActor(user_id=47, username="owner", role="viewer")
    with patch("agi_ecom_app.agent_v2_read_api.decode_access_token", return_value=actor):
        monkeypatch.delenv("COMMERCE_AGENT_INTERNAL_URL", raising=False)
        assert client.get(
            f"/v2/agent/threads/{THREAD_ID}/state",
            headers={"Authorization": "Bearer synthetic-user-jwt"},
        ).status_code == 503
        monkeypatch.setenv("COMMERCE_AGENT_INTERNAL_URL", "http://agent.internal:8123")
        monkeypatch.setenv("COMMERCE_AGENT_SERVICE_TOKEN", "synthetic-service-token-" + "x" * 32)
        monkeypatch.setenv("COMMERCE_AGENT_TENANT_ID", "configured-tenant")
        monkeypatch.setenv("COMMERCE_AGENT_ENVIRONMENT", "staging")
        assert client.get(
            f"/v2/agent/threads/{THREAD_ID}/state",
            headers={"Authorization": "Bearer synthetic-user-jwt"},
        ).status_code == 503
        monkeypatch.setenv("COMMERCE_AGENT_ENVIRONMENT", "development")
        assert client.get(
            "/v2/agent/threads/not-a-uuid/state",
            headers={"Authorization": "Bearer synthetic-user-jwt"},
        ).status_code == 422
        assert client.get(
            f"/v2/agent/threads/{THREAD_ID}/history?limit=101",
            headers={"Authorization": "Bearer synthetic-user-jwt"},
        ).status_code == 422

        real_async_client = httpx.AsyncClient

        def factory(*args, **kwargs):
            return real_async_client(
                *args,
                transport=httpx.MockTransport(lambda _: httpx.Response(404)),
                **kwargs,
            )

        with patch("agi_ecom_app.agent_v2_read_api.httpx.AsyncClient", side_effect=factory):
            denied = client.get(
                f"/v2/agent/threads/{THREAD_ID}/state",
                headers={"Authorization": "Bearer synthetic-user-jwt"},
            )
        assert denied.status_code == 404
        assert denied.json()["detail"] == "thread not found"

        def mismatch_factory(*args, **kwargs):
            return real_async_client(
                *args,
                transport=httpx.MockTransport(
                    lambda _: httpx.Response(409, json={"detail": "private upstream details"})
                ),
                **kwargs,
            )

        with patch(
            "agi_ecom_app.agent_v2_read_api.httpx.AsyncClient",
            side_effect=mismatch_factory,
        ):
            mismatch = client.get(
                f"/v2/agent/threads/{THREAD_ID}/state",
                headers={"Authorization": "Bearer synthetic-user-jwt"},
            )
        assert mismatch.status_code == 409
        assert mismatch.json()["detail"] == "state schema mismatch"
