"""Owner-scoped read-only HTTP projection of durable graph checkpoints.

This is an internal service boundary. The browser reaches it only through the
authenticated FastAPI BFF; no user bearer token or provider key is accepted.
Schema setup remains an explicit deployment step, not a startup side effect.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
import hmac
import json
import os
from typing import Any
from uuid import UUID

from pydantic import ValidationError
from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from commerce_agent.persistence import PersistenceSettings, open_postgres_graph
from commerce_agent.protocol_events import protocol_json_value
from commerce_agent.protocol_store import (
    ProtocolRecordError,
    ProtocolStore,
    ThreadNotOwned,
    ThreadOwner,
    open_protocol_store,
)
from commerce_agent.safety import SensitiveStateError, assert_secret_free
from commerce_agent.state import CommerceStateSnapshot, STATE_FIELD_NAMES


SERVICE_TOKEN_ENV = "COMMERCE_AGENT_SERVICE_TOKEN"
MAX_STATE_BYTES = 1_048_576
MAX_HISTORY_BYTES = 4_194_304


class StateSchemaMismatch(ProtocolRecordError):
    """The persisted state is not compatible with this API release."""


@dataclass(frozen=True)
class ReadResources:
    store: ProtocolStore
    graph: Any


def _checkpoint_id(config: Any) -> str | None:
    if not isinstance(config, Mapping):
        return None
    configurable = config.get("configurable")
    if not isinstance(configurable, Mapping):
        return None
    value = configurable.get("checkpoint_id")
    return value if isinstance(value, str) else None


def project_snapshot(snapshot: Any) -> dict[str, Any]:
    """Expose state and checkpoint identity, never trace/checkpoint metadata."""

    values = snapshot.values or {}
    assert_secret_free(values, path="checkpoint.values")
    if values:
        if (
            not isinstance(values, Mapping)
            or set(values) - STATE_FIELD_NAMES
            or values.get("schema_version") != 1
        ):
            raise StateSchemaMismatch("checkpoint state schema mismatch")
        try:
            CommerceStateSnapshot.model_validate(values)
        except ValidationError as exc:
            raise StateSchemaMismatch("checkpoint state schema mismatch") from exc
    payload = {
        "values": protocol_json_value(values),
        "checkpoint": (
            {"checkpoint_id": checkpoint_id}
            if (checkpoint_id := _checkpoint_id(snapshot.config)) else None
        ),
        "next": protocol_json_value(list(snapshot.next or ())),
    }
    assert_secret_free(payload, path="http.state")
    if len(json.dumps(payload, ensure_ascii=False).encode("utf-8")) > MAX_STATE_BYTES:
        raise ProtocolRecordError("checkpoint state response exceeds size limit")
    return payload


def _owner(request: Request) -> ThreadOwner:
    expected = request.app.state.service_token
    supplied = request.headers.get("x-commerce-service-token", "")
    if not supplied or not hmac.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="service authentication required")
    try:
        return ThreadOwner(
            tenant_id=request.headers.get("x-commerce-tenant-id", ""),
            user_id=request.headers.get("x-commerce-user-id", ""),
        )
    except (ProtocolRecordError, SensitiveStateError):
        raise HTTPException(status_code=400, detail="verified actor context required") from None


def _thread_id(request: Request) -> str:
    raw = request.path_params["thread_id"]
    try:
        return str(UUID(raw))
    except (TypeError, ValueError):
        raise HTTPException(status_code=422, detail="invalid thread ID") from None


async def _read_state(request: Request) -> JSONResponse:
    owner = _owner(request)
    resources: ReadResources = request.app.state.resources
    thread_id = _thread_id(request)
    try:
        await resources.store.require_owned(owner, thread_id)
        snapshot = await resources.graph.aget_state(
            {"configurable": {"thread_id": thread_id}}
        )
        return JSONResponse(project_snapshot(snapshot), headers={"Cache-Control": "no-store"})
    except ThreadNotOwned:
        raise HTTPException(status_code=404, detail="thread not found") from None
    except StateSchemaMismatch:
        raise HTTPException(status_code=409, detail="state schema mismatch") from None
    except (ProtocolRecordError, SensitiveStateError):
        raise HTTPException(status_code=500, detail="state unavailable") from None


async def _read_history(request: Request) -> JSONResponse:
    owner = _owner(request)
    resources: ReadResources = request.app.state.resources
    thread_id = _thread_id(request)
    try:
        limit = int(request.query_params.get("limit", "20"))
    except ValueError:
        raise HTTPException(status_code=422, detail="invalid history limit") from None
    if not 1 <= limit <= 100:
        raise HTTPException(status_code=422, detail="invalid history limit")
    try:
        await resources.store.require_owned(owner, thread_id)
        snapshots = resources.graph.aget_state_history(
            {"configurable": {"thread_id": thread_id}}, limit=limit
        )
        items: list[dict[str, Any]] = []
        total_bytes = 0
        try:
            async for snapshot in snapshots:
                item = project_snapshot(snapshot)
                total_bytes += len(json.dumps(item, ensure_ascii=False).encode("utf-8"))
                if total_bytes > MAX_HISTORY_BYTES:
                    raise ProtocolRecordError("checkpoint history response exceeds size limit")
                items.append(item)
        finally:
            close = getattr(snapshots, "aclose", None)
            if close is not None:
                await close()
        return JSONResponse({"items": items}, headers={"Cache-Control": "no-store"})
    except ThreadNotOwned:
        raise HTTPException(status_code=404, detail="thread not found") from None
    except StateSchemaMismatch:
        raise HTTPException(status_code=409, detail="state schema mismatch") from None
    except (ProtocolRecordError, SensitiveStateError):
        raise HTTPException(status_code=500, detail="state unavailable") from None


def create_read_app(
    *,
    resources: ReadResources | None = None,
    service_token: str | None = None,
) -> Starlette:
    """Create the internal read API; tests may inject isolated resources."""

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        token = service_token if service_token is not None else os.getenv(SERVICE_TOKEN_ENV, "")
        if len(token.encode("utf-8")) < 32:
            raise RuntimeError(f"{SERVICE_TOKEN_ENV} must have at least 32 bytes")
        app.state.service_token = token
        if resources is not None:
            app.state.resources = resources
            yield
            return
        settings = PersistenceSettings.from_environment()
        async with open_protocol_store(settings) as store:
            async with open_postgres_graph(settings) as graph:
                app.state.resources = ReadResources(store=store, graph=graph)
                yield

    async def http_error(_request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            {"detail": str(exc.detail)},
            status_code=exc.status_code,
            headers={"Cache-Control": "no-store"},
        )

    return Starlette(
        routes=[
            Route("/threads/{thread_id}/state", _read_state, methods=["GET"]),
            Route("/threads/{thread_id}/history", _read_history, methods=["GET"]),
        ],
        lifespan=lifespan,
        exception_handlers={HTTPException: http_error},
    )


app = create_read_app()


__all__ = [
    "ReadResources", "SERVICE_TOKEN_ENV", "StateSchemaMismatch", "app",
    "create_read_app", "project_snapshot",
]
