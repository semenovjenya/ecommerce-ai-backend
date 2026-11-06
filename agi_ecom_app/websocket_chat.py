from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from fastapi import WebSocket, WebSocketDisconnect, status
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from agi_ecom_app.agent_run_notifications import get_agent_run_notification_hub
from agi_ecom_app.api_models import (
    ChatSocketConnectionReadyEvent,
    ChatSocketHeartbeatEvent,
    ChatSocketMessageAcceptedEvent,
    ChatSocketMessageCompletedEvent,
    ChatSocketMessageErrorEvent,
    ChatSocketMessageQueuedEvent,
    ChatSocketRunStatusEvent,
    ChatSocketSendMessage,
)
from agi_ecom_app.backend_service import get_agent_run_status_payload, process_chat_message
from agi_ecom_app.config import get_settings
from agi_ecom_app.llm_runtime import get_active_llm_runtime_spec
from agi_ecom_app.runner import AgentRuntimeError, DuplicateClientMessageError
from agi_ecom_app.services.memory import ensure_chat_session, list_agent_run_traces


logger = logging.getLogger(__name__)


class ChatSendLockRegistry:
    def __init__(self):
        self._locks: dict[str, asyncio.Lock] = {}
        self._guard = asyncio.Lock()

    async def acquire(self, chat_id: str) -> asyncio.Lock | None:
        async with self._guard:
            lock = self._locks.setdefault(chat_id, asyncio.Lock())
            if lock.locked():
                return None
            await lock.acquire()
            return lock

    def release(self, lock: asyncio.Lock | None) -> None:
        if lock and lock.locked():
            lock.release()


class ChatSocketConnection:
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self._send_lock = asyncio.Lock()
        self.disconnected = False

    async def send_event(self, event) -> None:
        if self.disconnected:
            raise WebSocketDisconnect(code=1006)
        async with self._send_lock:
            try:
                payload = event.model_dump(mode="json", by_alias=True)
                if payload.get("managerDecision") is None:
                    payload.pop("managerDecision", None)
                await self.websocket.send_json(payload)
            except WebSocketDisconnect:
                self.disconnected = True
                raise
            except RuntimeError as exc:
                self.disconnected = True
                raise WebSocketDisconnect(code=1006) from exc


CHAT_SEND_LOCKS = ChatSendLockRegistry()
HEARTBEAT_INTERVAL_SECONDS = 15


async def handle_chat_websocket(websocket: WebSocket) -> None:
    chat_id = (websocket.query_params.get("chat_id") or "").strip()
    if not chat_id:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="chat_id is required")
        return

    await websocket.accept()
    connection = ChatSocketConnection(websocket)
    try:
        model_name = get_active_llm_runtime_spec(require_configured=False).display_name
    except Exception:
        settings = get_settings()
        model_name = f"openai:{settings.openai_model}"
    ensure_chat_session(chat_id=chat_id, model_name=model_name)
    heartbeat_task = asyncio.create_task(_heartbeat_loop(connection))
    resume_task: asyncio.Task | None = None

    try:
        await connection.send_event(ChatSocketConnectionReadyEvent(chat_id=chat_id))
        resume_task = asyncio.create_task(_resume_latest_run(connection, chat_id))
        while True:
            try:
                raw_payload = await websocket.receive_json()
                payload = ChatSocketSendMessage.model_validate(raw_payload)
            except WebSocketDisconnect:
                raise
            except ValidationError as exc:
                await connection.send_event(
                    ChatSocketMessageErrorEvent(
                        code="validation_error",
                        detail=str(exc),
                        retryable=False,
                    )
                )
                continue

            if payload.chat_id != chat_id:
                await connection.send_event(
                    ChatSocketMessageErrorEvent(
                        client_message_id=payload.client_message_id,
                        code="chat_mismatch",
                        detail="chat_id in payload does not match websocket session.",
                        retryable=False,
                    )
                )
                continue

            message_text = payload.text.strip()
            if not message_text:
                await connection.send_event(
                    ChatSocketMessageErrorEvent(
                        client_message_id=payload.client_message_id,
                        code="validation_error",
                        detail="message must not be empty",
                        retryable=False,
                    )
                )
                continue

            lock = await CHAT_SEND_LOCKS.acquire(chat_id)
            if lock is None:
                await connection.send_event(
                    ChatSocketMessageErrorEvent(
                        client_message_id=payload.client_message_id,
                        code="chat_busy",
                        detail="Предыдущее сообщение еще обрабатывается.",
                        retryable=True,
                    )
                )
                continue

            try:
                await connection.send_event(
                    ChatSocketMessageAcceptedEvent(client_message_id=payload.client_message_id)
                )
                response = await run_in_threadpool(
                    process_chat_message,
                    chat_id,
                    message_text,
                    payload.client_message_id,
                )
                response_status = getattr(response, "status", "completed")
                response_run_id = getattr(response, "run_id", None)
                if response_status in {"queued", "running"} and response_run_id is not None:
                    await connection.send_event(
                        ChatSocketMessageQueuedEvent(
                            client_message_id=payload.client_message_id,
                            run_id=response_run_id,
                            status=response_status,
                            intent=response.intent,
                        )
                    )
                    response = await _wait_run_until_terminal(
                        connection=connection,
                        client_message_id=payload.client_message_id,
                        run_id=response_run_id,
                        chat_id=chat_id,
                    )
                if getattr(response, "status", "completed") == "failed":
                    await connection.send_event(
                        ChatSocketMessageErrorEvent(
                            client_message_id=payload.client_message_id,
                            code="agent_run_failed",
                            detail=response.answer or "Агентский run завершился ошибкой.",
                            retryable=True,
                        )
                    )
                    continue
                await connection.send_event(
                    ChatSocketMessageCompletedEvent(
                        client_message_id=payload.client_message_id,
                        server_message_id=response.server_message_id,
                        answer=response.answer or "",
                        intent=response.intent,
                        managerDecision=getattr(response, "managerDecision", None),
                    )
                )
            except WebSocketDisconnect:
                raise
            except DuplicateClientMessageError as exc:
                if exc.answer is not None and exc.intent is not None:
                    await connection.send_event(
                        ChatSocketMessageCompletedEvent(
                            client_message_id=payload.client_message_id,
                            server_message_id=exc.assistant_message_seq,
                            answer=exc.answer,
                            intent=exc.intent.value,
                        )
                    )
                else:
                    await connection.send_event(
                        ChatSocketMessageErrorEvent(
                            client_message_id=payload.client_message_id,
                            code="duplicate_message",
                            detail=str(exc),
                            retryable=False,
                        )
                    )
            except AgentRuntimeError as exc:
                await connection.send_event(
                    ChatSocketMessageErrorEvent(
                        client_message_id=payload.client_message_id,
                        code="agent_runtime_error",
                        detail=str(exc),
                        retryable=True,
                    )
                )
            except Exception as exc:
                await connection.send_event(
                    ChatSocketMessageErrorEvent(
                        client_message_id=payload.client_message_id,
                        code="internal_error",
                        detail=str(exc),
                        retryable=True,
                    )
                )
            finally:
                CHAT_SEND_LOCKS.release(lock)
    except WebSocketDisconnect:
        return
    finally:
        if resume_task:
            resume_task.cancel()
        heartbeat_task.cancel()
        for task in (resume_task, heartbeat_task):
            if task is None:
                continue
            try:
                await task
            except (asyncio.CancelledError, WebSocketDisconnect):
                pass


async def _resume_latest_run(connection: ChatSocketConnection, chat_id: str) -> None:
    try:
        trace = await run_in_threadpool(_latest_replayable_trace, chat_id)
        if trace is None or not trace.client_message_id:
            return

        status_payload = await run_in_threadpool(
            get_agent_run_status_payload,
            trace.id,
            chat_id,
        )
        if status_payload.status in {"queued", "running"}:
            await connection.send_event(
                ChatSocketMessageQueuedEvent(
                    client_message_id=trace.client_message_id,
                    run_id=trace.id,
                    status=status_payload.status,
                    intent=status_payload.intent,
                )
            )
            response = await _wait_run_until_terminal(
                connection=connection,
                client_message_id=trace.client_message_id,
                run_id=trace.id,
                chat_id=chat_id,
            )
            await _send_terminal_response(
                connection=connection,
                client_message_id=trace.client_message_id,
                response=response,
            )
            return

        if status_payload.status in {"completed", "failed", "cancelled"}:
            from agi_ecom_app.api_models import ChatMessageResponse

            await _send_terminal_response(
                connection=connection,
                client_message_id=trace.client_message_id,
                response=ChatMessageResponse(
                    chat_id=chat_id,
                    answer=status_payload.answer or status_payload.error,
                    intent=status_payload.intent,
                    run_id=trace.id,
                    server_message_id=status_payload.server_message_id,
                    status=status_payload.status,
                    managerDecision=status_payload.managerDecision,
                ),
            )
    except WebSocketDisconnect:
        raise
    except Exception as exc:
        logger.warning("chat websocket resume failed chat_id=%s error=%s", chat_id, exc)


def _latest_replayable_trace(chat_id: str):
    traces = list_agent_run_traces(chat_id=chat_id, limit=1)
    if not traces:
        return None
    return traces[0]


async def _send_terminal_response(
    *,
    connection: ChatSocketConnection,
    client_message_id: str,
    response,
) -> None:
    if getattr(response, "status", "completed") == "completed":
        await connection.send_event(
            ChatSocketMessageCompletedEvent(
                client_message_id=client_message_id,
                server_message_id=response.server_message_id,
                answer=response.answer or "",
                intent=response.intent,
                managerDecision=getattr(response, "managerDecision", None),
            )
        )
        return

    await connection.send_event(
        ChatSocketMessageErrorEvent(
            client_message_id=client_message_id,
            code="agent_run_failed",
            detail=response.answer or "Агентский run завершился ошибкой.",
            retryable=True,
        )
    )


async def _heartbeat_loop(connection: ChatSocketConnection) -> None:
    while True:
        await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)
        try:
            await connection.send_event(
                ChatSocketHeartbeatEvent(ts=datetime.now(timezone.utc))
            )
        except WebSocketDisconnect:
            return


async def _wait_run_until_terminal(
    *,
    connection: ChatSocketConnection,
    client_message_id: str,
    run_id: int,
    chat_id: str,
):
    hub = get_agent_run_notification_hub()
    notifications = await hub.subscribe(run_id)
    last_status_signature: tuple[str, int | None] | None = None
    try:
        # LISTEN is active before this read, closing the SELECT/subscription gap.
        # Every wakeup and reconnect rereads Postgres, which remains source of truth.
        while True:
            status_payload = await run_in_threadpool(
                get_agent_run_status_payload,
                run_id,
                chat_id,
            )
            status_signature = (
                status_payload.status,
                status_payload.server_message_id,
            )
            if status_signature != last_status_signature:
                await connection.send_event(
                    ChatSocketRunStatusEvent(
                        client_message_id=client_message_id,
                        run_id=run_id,
                        status=status_payload.status,
                        server_message_id=status_payload.server_message_id,
                    )
                )
                last_status_signature = status_signature

            if status_payload.status in {"completed", "failed", "cancelled"}:
                from agi_ecom_app.api_models import ChatMessageResponse

                return ChatMessageResponse(
                    chat_id=chat_id,
                    answer=status_payload.answer or status_payload.error,
                    intent=status_payload.intent,
                    run_id=run_id,
                    server_message_id=status_payload.server_message_id,
                    status=status_payload.status,
                    managerDecision=status_payload.managerDecision,
                )

            # Values are only wake-up hints. We intentionally ignore their
            # contents and rebuild the public status from the durable row.
            await notifications.get()
    finally:
        await hub.unsubscribe(run_id, notifications)
