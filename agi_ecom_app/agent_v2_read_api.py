"""Authenticated, read-only BFF projection of commerce-agent checkpoints.

This is the first M1.5 development route, not a legacy-chat fallback or a
generic reverse proxy. User JWTs never reach the internal agent service.
"""

from __future__ import annotations

import os
from urllib.parse import urlsplit
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, Header, HTTPException, Response
import httpx

from agi_ecom_app.manager_lifecycle.auth import (
    AuthConfigurationError,
    AuthenticationError,
    ManagerActor,
    decode_access_token,
)


router = APIRouter(prefix="/v2/agent", tags=["commerce-agent"])


def _current_actor(
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> ManagerActor:
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Bearer token is required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    scheme, separator, token = authorization.partition(" ")
    if separator != " " or scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(status_code=401, detail="invalid Authorization header")
    try:
        return decode_access_token(token.strip())
    except AuthConfigurationError:
        raise HTTPException(status_code=503, detail="authentication unavailable") from None
    except AuthenticationError:
        raise HTTPException(status_code=401, detail="invalid or expired access token") from None


def _runtime_config() -> tuple[str, str, str]:
    url = os.getenv("COMMERCE_AGENT_INTERNAL_URL", "").rstrip("/")
    token = os.getenv("COMMERCE_AGENT_SERVICE_TOKEN", "")
    tenant_id = os.getenv("COMMERCE_AGENT_TENANT_ID", "").strip()
    environment = os.getenv("COMMERCE_AGENT_ENVIRONMENT", "").strip().lower()
    parsed = urlsplit(url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
        or len(token.encode("utf-8")) < 32
        or not tenant_id
        or environment not in {"development", "test", "staging", "production"}
        or (environment in {"staging", "production"} and parsed.scheme != "https")
    ):
        raise HTTPException(status_code=503, detail="agent runtime is not configured")
    return url, token, tenant_id


def _thread_id(value: str) -> str:
    try:
        return str(UUID(value))
    except (TypeError, ValueError):
        raise HTTPException(status_code=422, detail="invalid thread ID") from None


async def _fetch_read(
    actor: ManagerActor,
    thread_id: str,
    suffix: str,
    *,
    limit: int | None = None,
) -> dict:
    runtime_url, service_token, tenant_id = _runtime_config()
    path = f"/threads/{_thread_id(thread_id)}/{suffix}"
    headers = {
        "X-Commerce-Service-Token": service_token,
        "X-Commerce-Tenant-ID": tenant_id,
        "X-Commerce-User-ID": str(actor.user_id),
        "X-Request-ID": str(uuid4()),
    }
    try:
        async with httpx.AsyncClient(
            base_url=runtime_url, timeout=10.0, trust_env=False
        ) as client:
            upstream = await client.get(
                path,
                headers=headers,
                params={"limit": limit} if limit is not None else None,
            )
    except (httpx.TimeoutException, httpx.RequestError):
        raise HTTPException(status_code=503, detail="agent runtime unavailable") from None
    if upstream.status_code == 404:
        raise HTTPException(status_code=404, detail="thread not found")
    if upstream.status_code == 409:
        raise HTTPException(status_code=409, detail="state schema mismatch")
    if upstream.status_code != 200:
        raise HTTPException(status_code=503, detail="agent runtime unavailable")
    try:
        data = upstream.json()
    except ValueError:
        raise HTTPException(status_code=502, detail="invalid agent runtime response") from None
    if not isinstance(data, dict):
        raise HTTPException(status_code=502, detail="invalid agent runtime response")
    if suffix == "state" and not isinstance(data.get("values"), dict):
        raise HTTPException(status_code=502, detail="invalid agent runtime response")
    if suffix == "history" and not isinstance(data.get("items"), list):
        raise HTTPException(status_code=502, detail="invalid agent runtime response")
    return data


@router.get("/threads/{thread_id}/state")
async def get_commerce_agent_state(
    thread_id: str,
    response: Response,
    actor: ManagerActor = Depends(_current_actor),
) -> dict:
    response.headers["Cache-Control"] = "no-store"
    return await _fetch_read(actor, thread_id, "state")


@router.get("/threads/{thread_id}/history")
async def get_commerce_agent_history(
    thread_id: str,
    response: Response,
    limit: int = 20,
    actor: ManagerActor = Depends(_current_actor),
) -> dict:
    if not 1 <= limit <= 100:
        raise HTTPException(status_code=422, detail="invalid history limit")
    response.headers["Cache-Control"] = "no-store"
    return await _fetch_read(actor, thread_id, "history", limit=limit)


__all__ = ["router"]
