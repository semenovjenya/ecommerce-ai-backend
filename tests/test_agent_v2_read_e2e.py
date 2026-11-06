"""Opt-in M1.5 BFF → internal API → PostgreSQL checkpoint smoke.

Run with both dependency sets and a dedicated development agent database.
The normal legacy suite skips this test because it does not install LangGraph.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import os
from unittest.mock import patch
from uuid import uuid4

import httpx
import jwt
import pytest

pytest.importorskip("commerce_agent")

from langchain_core.messages import HumanMessage

from agi_ecom_app.api import app as backend_app
from agi_ecom_app.services.memory import AppUserRecord
from commerce_agent.context import CommerceRuntimeContext
from commerce_agent.persistence import (
    PersistenceSettings,
    initialize_checkpoint_schema,
    open_postgres_graph,
)
from commerce_agent.protocol_read_api import ReadResources, create_read_app
from commerce_agent.protocol_store import ThreadOwner, open_protocol_store


def test_signed_jwt_bff_internal_api_and_postgres_checkpoint(monkeypatch) -> None:
    if not os.getenv("COMMERCE_AGENT_DATABASE_URI"):
        pytest.skip("dedicated development PostgreSQL is not configured")
    settings = PersistenceSettings.from_environment()
    thread_id = str(uuid4())
    tenant_id = "read-e2e-tenant"
    service_token = "synthetic-internal-service-token-" + "x" * 32
    signing_secret = "synthetic-jwt-signing-secret-" + "y" * 32
    monkeypatch.setenv("MANAGER_JWT_SECRET", signing_secret)
    monkeypatch.setenv("COMMERCE_AGENT_INTERNAL_URL", "http://agent.internal:8123")
    monkeypatch.setenv("COMMERCE_AGENT_SERVICE_TOKEN", service_token)
    monkeypatch.setenv("COMMERCE_AGENT_TENANT_ID", tenant_id)
    monkeypatch.setenv("COMMERCE_AGENT_ENVIRONMENT", "development")
    now = datetime.now(timezone.utc)

    def actor_record(user_id: int) -> AppUserRecord:
        return AppUserRecord(
            id=user_id,
            username=f"user-{user_id}@example.test",
            password_hash="synthetic-unused-hash",
            role="viewer",
            is_active=True,
            created_at=now,
            updated_at=now,
        )

    def user_token(user_id: int) -> str:
        return jwt.encode(
            {
                "sub": str(user_id), "username": f"user-{user_id}@example.test",
                "role": "viewer", "iat": now,
                "exp": now + timedelta(minutes=5), "iss": "ai-ecom-manager",
            },
            signing_secret,
            algorithm="HS256",
        )

    async def cleanup(store) -> None:
        async with store._pool.connection() as conn:
            await conn.execute("DELETE FROM checkpoint_writes WHERE thread_id = %s", (thread_id,))
            await conn.execute("DELETE FROM checkpoint_blobs WHERE thread_id = %s", (thread_id,))
            await conn.execute("DELETE FROM checkpoints WHERE thread_id = %s", (thread_id,))
            await conn.execute("DELETE FROM commerce_agent_threads WHERE thread_id = %s", (thread_id,))

    async def exercise() -> None:
        await initialize_checkpoint_schema(settings)
        async with open_protocol_store(settings) as store:
            await store.setup()
            await store.create_thread(ThreadOwner(tenant_id, "47"), thread_id)
            try:
                async with open_postgres_graph(settings) as graph:
                    await graph.ainvoke(
                        {
                            "schema_version": 1,
                            "messages": [HumanMessage(content="safe synthetic e2e query")],
                        },
                        config={"configurable": {"thread_id": thread_id}},
                        context=CommerceRuntimeContext(
                            tenant_id=tenant_id,
                            user_id="47",
                            roles=("viewer",),
                            request_id="read-e2e-synthetic-request",
                        ),
                    )
                    internal_app = create_read_app(
                        resources=ReadResources(store=store, graph=graph),
                        service_token=service_token,
                    )
                    real_async_client = httpx.AsyncClient

                    def internal_client(*args, **kwargs):
                        return real_async_client(
                            *args,
                            transport=httpx.ASGITransport(app=internal_app),
                            **kwargs,
                        )

                    async with internal_app.router.lifespan_context(internal_app):
                        async with real_async_client(
                            transport=httpx.ASGITransport(app=backend_app),
                            base_url="http://browser.test",
                        ) as browser:
                            with (
                                patch(
                                    "agi_ecom_app.manager_lifecycle.auth.get_app_user",
                                    side_effect=lambda user_id: actor_record(user_id),
                                ),
                                patch(
                                    "agi_ecom_app.agent_v2_read_api.httpx.AsyncClient",
                                    side_effect=internal_client,
                                ),
                            ):
                                state = await browser.get(
                                    f"/v2/agent/threads/{thread_id}/state",
                                    headers={"Authorization": f"Bearer {user_token(47)}"},
                                )
                                history = await browser.get(
                                    f"/v2/agent/threads/{thread_id}/history?limit=2",
                                    headers={"Authorization": f"Bearer {user_token(47)}"},
                                )
                                wrong_user = await browser.get(
                                    f"/v2/agent/threads/{thread_id}/state",
                                    headers={"Authorization": f"Bearer {user_token(48)}"},
                                )
                                monkeypatch.setenv("COMMERCE_AGENT_TENANT_ID", "other-tenant")
                                wrong_tenant = await browser.get(
                                    f"/v2/agent/threads/{thread_id}/state",
                                    headers={"Authorization": f"Bearer {user_token(47)}"},
                                )
                    assert state.status_code == 200, state.text
                    assert state.json()["values"]["schema_version"] == 1
                    assert set(state.json()["values"]["agent_statuses"]) == {
                        "sales", "marketing", "supply", "price", "finance"
                    }
                    assert state.json()["checkpoint"]["checkpoint_id"]
                    assert history.status_code == 200, history.text
                    assert len(history.json()["items"]) == 2
                    assert wrong_user.status_code == 404
                    assert wrong_tenant.status_code == 404
                    assert service_token not in state.text
            finally:
                await cleanup(store)

    asyncio.run(exercise())
