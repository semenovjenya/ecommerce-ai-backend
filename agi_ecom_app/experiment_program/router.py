from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any
from weakref import WeakKeyDictionary

import psycopg
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Response
from fastapi.responses import StreamingResponse

from agi_ecom_app.config import get_settings
from agi_ecom_app.experiment_program.models import (
    ExperimentNegotiationMessageInput,
    ExperimentNegotiationMessageView,
    ExperimentNegotiationMessagesResponse,
    ExperimentProgramBundleView,
    ExperimentProgramCreate,
    ExperimentProgramEventView,
    ExperimentProgramEventsResponse,
)
from agi_ecom_app.experiment_program.repository import list_program_events
from agi_ecom_app.experiment_program.service import (
    ExperimentProgramConflict,
    ExperimentProgramContractError,
    ExperimentProgramNotFound,
    create_experiment_program,
    get_experiment_negotiation_messages,
    get_experiment_program_bundle,
    get_experiment_program_events,
    post_experiment_negotiation_message,
)
from agi_ecom_app.manager_lifecycle.auth import (
    AuthConfigurationError,
    AuthenticationError,
    AuthorizationError,
    ManagerActor,
    decode_access_token,
    require_role,
)


logger = logging.getLogger(__name__)
router = APIRouter()

_PROGRAM_EVENTS_CHANNEL = "experiment_program_events"
_SSE_HEARTBEAT_SECONDS = 15.0
_SSE_REPLAY_PAGE_SIZE = 5000


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
        raise HTTPException(
            status_code=401,
            detail="invalid Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        return decode_access_token(token.strip())
    except AuthConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
    except AuthenticationError as exc:
        raise HTTPException(
            status_code=401,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from None


@router.post(
    "/v1/manager/experiment-programs",
    response_model=ExperimentProgramBundleView,
    status_code=201,
)
def create_manager_experiment_program(
    payload: ExperimentProgramCreate,
    response: Response,
    idempotency_key: str = Header(..., alias="Idempotency-Key"),
    actor: ManagerActor = Depends(_current_actor),
) -> ExperimentProgramBundleView:
    try:
        require_role(actor, "manager")
        bundle, created = create_experiment_program(
            payload,
            idempotency_key=idempotency_key,
        )
    except AuthorizationError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from None
    except ExperimentProgramConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    except ExperimentProgramContractError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    response.status_code = 201 if created else 200
    return bundle


@router.get(
    "/v1/manager/experiment-programs/{program_id}",
    response_model=ExperimentProgramBundleView,
)
def get_manager_experiment_program(
    program_id: str,
    _: ManagerActor = Depends(_current_actor),
) -> ExperimentProgramBundleView:
    try:
        return get_experiment_program_bundle(program_id)
    except ExperimentProgramNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None


@router.post(
    "/v1/manager/experiment-programs/{program_id}/messages",
    response_model=ExperimentNegotiationMessageView,
    status_code=201,
)
def create_manager_experiment_message(
    program_id: str,
    payload: ExperimentNegotiationMessageInput,
    response: Response,
    idempotency_key: str = Header(..., alias="Idempotency-Key"),
    actor: ManagerActor = Depends(_current_actor),
) -> ExperimentNegotiationMessageView:
    try:
        # This endpoint is an operational/debug boundary. Production agents and
        # Evidence Authority call the service directly, so HTTP clients cannot
        # impersonate either role with only a manager token.
        require_role(actor, "admin")
        message, created = post_experiment_negotiation_message(
            program_id,
            payload,
            idempotency_key=idempotency_key,
        )
    except AuthorizationError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from None
    except ExperimentProgramNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    except ExperimentProgramConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    except ExperimentProgramContractError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    response.status_code = 201 if created else 200
    return message


@router.get(
    "/v1/manager/experiment-programs/{program_id}/messages",
    response_model=ExperimentNegotiationMessagesResponse,
)
def get_manager_experiment_messages(
    program_id: str,
    after_id: int = Query(default=0, alias="afterId", ge=0),
    limit: int = Query(default=500, ge=1, le=5000),
    _: ManagerActor = Depends(_current_actor),
) -> ExperimentNegotiationMessagesResponse:
    try:
        items = get_experiment_negotiation_messages(
            program_id,
            after_id=after_id,
            limit=limit,
        )
    except ExperimentProgramNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    return ExperimentNegotiationMessagesResponse(programId=program_id, items=items)


@router.get(
    "/v1/manager/experiment-programs/{program_id}/events",
    response_model=ExperimentProgramEventsResponse,
)
def get_manager_experiment_events(
    program_id: str,
    after_id: int = Query(default=0, alias="afterId", ge=0),
    limit: int = Query(default=500, ge=1, le=5000),
    _: ManagerActor = Depends(_current_actor),
) -> ExperimentProgramEventsResponse:
    try:
        items = get_experiment_program_events(
            program_id,
            after_id=after_id,
            limit=limit,
        )
    except ExperimentProgramNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    return ExperimentProgramEventsResponse(programId=program_id, items=items)


@router.get("/v1/manager/experiment-programs/{program_id}/events/stream")
async def stream_manager_experiment_events(
    program_id: str,
    last_event_id: int = Header(default=0, alias="Last-Event-ID", ge=0),
    _: ManagerActor = Depends(_current_actor),
) -> StreamingResponse:
    try:
        get_experiment_program_bundle(program_id)
    except ExperimentProgramNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    return StreamingResponse(
        _program_event_stream(program_id, after_id=last_event_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _format_sse_event(row: dict[str, Any]) -> str:
    view = ExperimentProgramEventView.model_validate(row)
    return (
        f"id: {view.id}\n"
        f"event: {view.event_type}\n"
        f"data: {view.model_dump_json(by_alias=True)}\n\n"
    )


async def _load_event_page(program_id: str, after_id: int) -> list[dict[str, Any]]:
    return await asyncio.to_thread(
        list_program_events,
        program_id,
        after_id=after_id,
        limit=_SSE_REPLAY_PAGE_SIZE,
    )


async def _connect_program_listener() -> psycopg.AsyncConnection[Any]:
    connection = await psycopg.AsyncConnection.connect(
        get_settings().memory_postgres_dsn,
        autocommit=True,
    )
    await connection.execute(f"LISTEN {_PROGRAM_EVENTS_CHANNEL}")
    return connection


async def _wait_for_notification(connection: psycopg.AsyncConnection[Any]) -> Any | None:
    notification = None
    async for notification in connection.notifies(
        timeout=_SSE_HEARTBEAT_SECONDS,
        stop_after=1,
    ):
        pass
    return notification


def _notification_program_id(notification: Any) -> str | None:
    raw_payload = getattr(notification, "payload", None)
    if not isinstance(raw_payload, str) or not raw_payload:
        return None
    try:
        payload = json.loads(raw_payload)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    value = payload.get("program_id", payload.get("programId"))
    return str(value) if value is not None else None


class _ProgramNotificationHub:
    """One LISTEN connection per event loop, fanned out by program ID."""

    def __init__(self) -> None:
        self._subscribers: dict[str, set[asyncio.Queue[str]]] = {}
        self._task: asyncio.Task[None] | None = None
        self._ready = asyncio.Event()

    async def subscribe(self, program_id: str) -> asyncio.Queue[str]:
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=4)
        self._subscribers.setdefault(program_id, set()).add(queue)
        if self._task is None or self._task.done():
            self._ready.clear()
            self._task = asyncio.create_task(self._listen())
        try:
            await self._ready.wait()
        except BaseException:
            await self.unsubscribe(program_id, queue)
            raise
        return queue

    async def unsubscribe(self, program_id: str, queue: asyncio.Queue[str]) -> None:
        subscribers = self._subscribers.get(program_id)
        if subscribers is not None:
            subscribers.discard(queue)
            if not subscribers:
                self._subscribers.pop(program_id, None)
        if not self._subscribers and self._task is not None:
            task = self._task
            self._task = None
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    @staticmethod
    def _offer(queue: asyncio.Queue[str], message: str) -> None:
        if queue.full():
            queue.get_nowait()
        queue.put_nowait(message)

    def _broadcast(self, message: str, program_id: str | None = None) -> None:
        groups = (
            [self._subscribers.get(program_id, set())]
            if program_id is not None
            else list(self._subscribers.values())
        )
        for subscribers in groups:
            for queue in tuple(subscribers):
                self._offer(queue, message)

    async def _listen(self) -> None:
        needs_replay = False
        while self._subscribers:
            connection: psycopg.AsyncConnection[Any] | None = None
            try:
                connection = await _connect_program_listener()
                if needs_replay:
                    self._broadcast("replay")
                    needs_replay = False
                self._ready.set()
                while self._subscribers:
                    notification = await _wait_for_notification(connection)
                    if notification is None:
                        await asyncio.sleep(0)
                        continue
                    self._broadcast("replay", _notification_program_id(notification))
                    await asyncio.sleep(0)
            except asyncio.CancelledError:
                raise
            except Exception:
                needs_replay = True
                self._ready.set()
                self._broadcast("reconnecting")
                logger.warning("experiment program LISTEN reconnecting", exc_info=True)
                await asyncio.sleep(1)
            finally:
                if connection is not None:
                    try:
                        await connection.close()
                    except Exception:
                        logger.warning("experiment program LISTEN close failed", exc_info=True)


_PROGRAM_NOTIFICATION_HUBS: WeakKeyDictionary[
    asyncio.AbstractEventLoop,
    _ProgramNotificationHub,
] = WeakKeyDictionary()


def _program_notification_hub() -> _ProgramNotificationHub:
    loop = asyncio.get_running_loop()
    hub = _PROGRAM_NOTIFICATION_HUBS.get(loop)
    if hub is None:
        hub = _ProgramNotificationHub()
        _PROGRAM_NOTIFICATION_HUBS[loop] = hub
    return hub


async def _program_event_stream(
    program_id: str,
    *,
    after_id: int = 0,
) -> AsyncIterator[str]:
    cursor = max(0, int(after_id))
    hub = _program_notification_hub()
    notifications = await hub.subscribe(program_id)
    replay_required = True
    try:
        while True:
            if replay_required:
                while True:
                    events = await _load_event_page(program_id, cursor)
                    for event in events:
                        cursor = int(event["id"])
                        yield _format_sse_event(event)
                    if len(events) < _SSE_REPLAY_PAGE_SIZE:
                        break
                replay_required = False
            try:
                message = await asyncio.wait_for(
                    notifications.get(),
                    timeout=_SSE_HEARTBEAT_SECONDS,
                )
            except asyncio.TimeoutError:
                yield ": heartbeat\n\n"
                continue
            if message == "reconnecting":
                yield ": reconnecting\n\n"
            replay_required = True
    finally:
        await hub.unsubscribe(program_id, notifications)


__all__ = ["router"]
