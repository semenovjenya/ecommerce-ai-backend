from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any
from weakref import WeakKeyDictionary
from zoneinfo import ZoneInfo

import psycopg
from fastapi import APIRouter, Depends, Header, HTTPException, Query
from fastapi.responses import StreamingResponse

from agi_ecom_app.config import get_settings


logger = logging.getLogger(__name__)

from agi_ecom_app.manager_lifecycle.auth import (
    AuthConfigurationError,
    AuthenticationError,
    AuthorizationError,
    ManagerActor,
    authenticate,
    decode_access_token,
    require_role,
)
from agi_ecom_app.manager_lifecycle.service import (
    ManagerLifecycleConflict,
    ManagerFeatureDisabled,
    ManagerLifecycleNotFound,
    approve_manager_action,
    complete_manual_manager_action,
    get_decision_bundle,
    get_manager_action,
    get_today_decision_bundles,
    reject_manager_action,
)
from agi_ecom_app.services.memory import (
    AgentActionExperimentRecord,
    AgentActionMeasurementRecord,
    DecisionSimulationCandidateRecord,
    DecisionSimulationEventRecord,
    DecisionSimulationRunRecord,
    ManagerDecisionRecord,
    get_decision_simulation_run,
    get_latest_decision_simulation_run_for_decision,
    list_decision_simulation_candidates,
    list_decision_simulation_events,
    list_agent_action_measurements,
)

from agi_ecom_app.action_board.models import (
    AgentActionExecutionBatchResponse,
    AgentActionExecutionItem,
    AgentActionExecutionRequest,
    AgentActionExecutionsResponse,
    AgentActionSelectionRequest,
    AgentActionTaskItem,
    AgentActionTasksResponse,
    AuthLoginRequest,
    AuthLoginResponse,
    AuthUserView,
    DecisionSimulationCandidateView,
    DecisionSimulationCandidatesResponse,
    DecisionSimulationEventView,
    DecisionSimulationEventsResponse,
    DecisionSimulationRunView,
    DecisionSimulationSummaryView,
    ManagerActionExperimentView,
    ManagerActionCompleteRequest,
    ManagerActionMeasurementView,
    ManagerActionRejectRequest,
    ManagerDecisionView,
    ManagerTodayDecisionsResponse,
    PricingSimulationStartRequest,
)
from agi_ecom_app.action_board.service import (
    complete_agent_action_execution,
    get_agent_action_execution_payload,
    get_agent_action_tasks_payload,
    list_agent_action_executions_payload,
    select_agent_action_task,
    start_agent_action_execution,
    unselect_agent_action_task,
)

router = APIRouter()

_SIMULATION_EVENTS_CHANNEL = "decision_simulation_events"
_SSE_HEARTBEAT_SECONDS = 15.0
_SSE_REPLAY_PAGE_SIZE = 5000


def _current_manager_actor(
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


@router.post("/v1/auth/login", response_model=AuthLoginResponse)
def login(payload: AuthLoginRequest) -> AuthLoginResponse:
    try:
        result = authenticate(payload.username, payload.password)
    except AuthConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
    except AuthenticationError as exc:
        raise HTTPException(
            status_code=401,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from None
    return AuthLoginResponse(
        accessToken=result.access_token,
        tokenType=result.token_type,
        expiresIn=result.expires_in,
        user=AuthUserView(
            id=result.actor.user_id,
            username=result.actor.username,
            role=result.actor.role,
        ),
    )


@router.get(
    "/v1/manager/decisions/today",
    response_model=ManagerTodayDecisionsResponse,
)
def get_today_manager_decisions(
    _: ManagerActor = Depends(_current_manager_actor),
) -> ManagerTodayDecisionsResponse:
    today = datetime.now(ZoneInfo("Europe/Moscow")).date()
    bundles = get_today_decision_bundles(today=today)
    return ManagerTodayDecisionsResponse(
        date=today.isoformat(),
        items=[_decision_view(decision, actions) for decision, actions in bundles],
    )


@router.get(
    "/v1/manager/decisions/{decision_id}",
    response_model=ManagerDecisionView,
)
def get_manager_decision_by_id(
    decision_id: str,
    _: ManagerActor = Depends(_current_manager_actor),
) -> ManagerDecisionView:
    try:
        decision, actions = get_decision_bundle(decision_id)
    except ManagerLifecycleNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    return _decision_view(decision, actions)


@router.get(
    "/v1/manager/simulations/{simulation_run_id}",
    response_model=DecisionSimulationRunView,
)
def get_manager_decision_simulation(
    simulation_run_id: str,
    _: ManagerActor = Depends(_current_manager_actor),
) -> DecisionSimulationRunView:
    run = _require_decision_simulation_run(simulation_run_id)
    return _simulation_run_view(
        run,
        list_decision_simulation_candidates(simulation_run_id),
    )


@router.post(
    "/v1/manager/pricing/simulations",
    response_model=DecisionSimulationRunView,
    status_code=202,
)
def create_manager_pricing_simulation(
    payload: PricingSimulationStartRequest,
    idempotency_key: str = Header(..., alias="Idempotency-Key"),
    actor: ManagerActor = Depends(_current_manager_actor),
) -> DecisionSimulationRunView:
    try:
        require_role(actor, "manager")
    except AuthorizationError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from None

    try:
        # Keep NumPy and the analytics source out of the read-only manager API
        # import path. They are loaded only when this opt-in endpoint is used.
        from agi_ecom_app.decision_simulation.pricing_service import (
            PricingSimulationDeliveryError,
            PricingSimulationIdempotencyConflict,
            PricingSimulationSourceError,
            start_pricing_simulation,
        )

        run = start_pricing_simulation(
            nm_id=payload.nmId,
            candidate_prices=payload.candidatePrices,
            horizon_days=payload.horizonDays,
            idempotency_key=idempotency_key,
            actor=actor,
        )
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail="pricing simulation service is unavailable",
        ) from exc
    except AuthorizationError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from None
    except PricingSimulationSourceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except PricingSimulationIdempotencyConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    except PricingSimulationDeliveryError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "message": str(exc),
                "simulationRunId": exc.simulation_run_id,
            },
        ) from None
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None

    return _simulation_run_view(
        run,
        list_decision_simulation_candidates(run.id),
    )


@router.get(
    "/v1/manager/simulations/{simulation_run_id}/candidates",
    response_model=DecisionSimulationCandidatesResponse,
)
def get_manager_decision_simulation_candidates(
    simulation_run_id: str,
    _: ManagerActor = Depends(_current_manager_actor),
) -> DecisionSimulationCandidatesResponse:
    _require_decision_simulation_run(simulation_run_id)
    return DecisionSimulationCandidatesResponse(
        runId=simulation_run_id,
        items=[
            _simulation_candidate_view(candidate)
            for candidate in list_decision_simulation_candidates(simulation_run_id)
        ],
    )


@router.get(
    "/v1/manager/simulations/{simulation_run_id}/events",
    response_model=DecisionSimulationEventsResponse,
)
def get_manager_decision_simulation_events(
    simulation_run_id: str,
    after_id: int = Query(default=0, alias="afterId", ge=0),
    limit: int = Query(default=500, ge=1, le=5000),
    _: ManagerActor = Depends(_current_manager_actor),
) -> DecisionSimulationEventsResponse:
    _require_decision_simulation_run(simulation_run_id)
    events = list_decision_simulation_events(
        simulation_run_id,
        after_id=after_id,
        limit=limit,
    )
    return DecisionSimulationEventsResponse(
        runId=simulation_run_id,
        items=[_simulation_event_view(event) for event in events],
    )


@router.get("/v1/manager/simulations/{simulation_run_id}/events/stream")
async def stream_manager_decision_simulation_events(
    simulation_run_id: str,
    last_event_id: int = Header(default=0, alias="Last-Event-ID", ge=0),
    _: ManagerActor = Depends(_current_manager_actor),
) -> StreamingResponse:
    _require_decision_simulation_run(simulation_run_id)
    return StreamingResponse(
        _decision_simulation_event_stream(
            simulation_run_id,
            after_id=last_event_id,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post(
    "/v1/manager/simulations/{simulation_run_id}/retry",
    response_model=DecisionSimulationRunView,
    status_code=202,
)
def retry_manager_decision_simulation(
    simulation_run_id: str,
    idempotency_key: str = Header(..., alias="Idempotency-Key"),
    actor: ManagerActor = Depends(_current_manager_actor),
) -> DecisionSimulationRunView:
    try:
        require_role(actor, "admin")
    except AuthorizationError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from None

    try:
        # Lazy import avoids coupling read-only manager endpoints to the RQ worker
        # bootstrap path and keeps feature mode `off` fully backwards compatible.
        from agi_ecom_app.decision_simulation.service import (
            retry_decision_simulation,
        )

        run = retry_decision_simulation(
            simulation_run_id=simulation_run_id,
            idempotency_key=idempotency_key,
            actor=actor,
        )
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail="decision simulation retry service is unavailable",
        ) from exc
    except AuthorizationError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from None
    except (KeyError, LookupError, ManagerLifecycleNotFound) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    except ManagerLifecycleConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None

    return _simulation_run_view(
        run,
        list_decision_simulation_candidates(run.id),
    )


@router.get(
    "/v1/manager/actions/{action_id}",
    response_model=ManagerActionExperimentView,
)
def get_manager_action_by_id(
    action_id: str,
    _: ManagerActor = Depends(_current_manager_actor),
) -> ManagerActionExperimentView:
    try:
        return _action_view(get_manager_action(action_id), include_measurements=True)
    except ManagerLifecycleNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None


@router.post(
    "/v1/manager/actions/{action_id}/approve",
    response_model=ManagerActionExperimentView,
)
def approve_manager_action_by_id(
    action_id: str,
    idempotency_key: str = Header(..., alias="Idempotency-Key"),
    actor: ManagerActor = Depends(_current_manager_actor),
) -> ManagerActionExperimentView:
    try:
        action = approve_manager_action(
            experiment_id=action_id,
            actor=actor,
            idempotency_key=idempotency_key,
        )
    except ManagerFeatureDisabled as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
    except AuthorizationError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from None
    except ManagerLifecycleNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    except ManagerLifecycleConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    return _action_view(action, include_measurements=True)


@router.post(
    "/v1/manager/actions/{action_id}/reject",
    response_model=ManagerActionExperimentView,
)
def reject_manager_action_by_id(
    action_id: str,
    payload: ManagerActionRejectRequest,
    idempotency_key: str = Header(..., alias="Idempotency-Key"),
    actor: ManagerActor = Depends(_current_manager_actor),
) -> ManagerActionExperimentView:
    try:
        action = reject_manager_action(
            experiment_id=action_id,
            actor=actor,
            idempotency_key=idempotency_key,
            reason=payload.reason,
            manager_label=payload.managerLabel,
        )
    except ManagerFeatureDisabled as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
    except AuthorizationError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from None
    except ManagerLifecycleNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    except ManagerLifecycleConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    return _action_view(action, include_measurements=True)


@router.post(
    "/v1/manager/actions/{action_id}/complete",
    response_model=ManagerActionExperimentView,
)
def complete_manual_manager_action_by_id(
    action_id: str,
    payload: ManagerActionCompleteRequest,
    idempotency_key: str = Header(..., alias="Idempotency-Key"),
    actor: ManagerActor = Depends(_current_manager_actor),
) -> ManagerActionExperimentView:
    try:
        action = complete_manual_manager_action(
            experiment_id=action_id,
            actor=actor,
            idempotency_key=idempotency_key,
            actual_at=payload.actualAt,
        )
    except ManagerFeatureDisabled as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
    except AuthorizationError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from None
    except ManagerLifecycleNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    except ManagerLifecycleConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    return _action_view(action, include_measurements=True)


@router.get("/v1/agent-actions", response_model=AgentActionTasksResponse)
def get_agent_actions(
    chat_id: str,
    server_message_id: int | None = None,
) -> AgentActionTasksResponse:
    try:
        return get_agent_action_tasks_payload(
            chat_id=chat_id,
            server_message_id=server_message_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None


@router.post("/v1/agent-actions/selected", response_model=AgentActionTaskItem)
def select_agent_action(payload: AgentActionSelectionRequest) -> AgentActionTaskItem:
    try:
        return select_agent_action_task(chat_id=payload.chat_id, task_id=payload.task_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None


@router.delete("/v1/agent-actions/selected", response_model=AgentActionTaskItem)
def unselect_agent_action(chat_id: str, task_id: str) -> AgentActionTaskItem:
    try:
        return unselect_agent_action_task(chat_id=chat_id, task_id=task_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None


@router.post(
    "/v1/agent-actions/executions",
    response_model=AgentActionExecutionBatchResponse,
    status_code=202,
)
def create_agent_action_execution(
    payload: AgentActionExecutionRequest,
) -> AgentActionExecutionBatchResponse:
    try:
        return start_agent_action_execution(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None


@router.get(
    "/v1/agent-actions/executions",
    response_model=AgentActionExecutionsResponse,
)
def get_agent_action_executions(chat_id: str) -> AgentActionExecutionsResponse:
    return list_agent_action_executions_payload(chat_id)


@router.get(
    "/v1/agent-actions/executions/{execution_id}",
    response_model=AgentActionExecutionItem,
)
def get_agent_action_execution(execution_id: int) -> AgentActionExecutionItem:
    try:
        return get_agent_action_execution_payload(execution_id)
    except LookupError:
        raise HTTPException(status_code=404, detail="agent action execution not found") from None


@router.post(
    "/v1/agent-actions/executions/{execution_id}/done",
    response_model=AgentActionExecutionItem,
)
def mark_agent_action_execution_done(execution_id: int) -> AgentActionExecutionItem:
    try:
        return complete_agent_action_execution(execution_id)
    except LookupError:
        raise HTTPException(status_code=404, detail="agent action execution not found") from None


def _decision_view(
    decision: ManagerDecisionRecord,
    actions: list[AgentActionExperimentRecord],
) -> ManagerDecisionView:
    simulation_run = None
    if decision.simulation_run_id:
        simulation_run = get_decision_simulation_run(decision.simulation_run_id)
    if simulation_run is None:
        simulation_run = get_latest_decision_simulation_run_for_decision(decision.id)
    return ManagerDecisionView(
        id=decision.id,
        decisionDate=decision.decision_date.isoformat(),
        objective=decision.objective,
        experimentProgramId=decision.experiment_program_id,
        status=decision.status,
        requestCoverage=decision.request_coverage,
        evidenceAsOf=decision.evidence_as_of,
        rankedActions=decision.ranked_actions,
        rejectedCandidates=decision.rejected_candidates,
        blockers=decision.blockers,
        publicDecision=decision.public_json,
        simulationEligibility=_decision_simulation_eligibility(decision),
        simulationSummary=(
            _simulation_summary_view(simulation_run) if simulation_run else None
        ),
        actions=[_action_view(action) for action in actions],
        createdAt=decision.created_at,
        updatedAt=decision.updated_at,
    )


def _decision_simulation_eligibility(
    decision: ManagerDecisionRecord,
) -> dict[str, Any] | None:
    for payload in (decision.public_json, decision.decision_json):
        raw = payload.get("simulationEligibility", payload.get("simulation_eligibility"))
        if isinstance(raw, dict):
            return dict(raw)
    return None


def _require_decision_simulation_run(
    simulation_run_id: str,
) -> DecisionSimulationRunRecord:
    run = get_decision_simulation_run(simulation_run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="decision simulation not found")
    return run


def _simulation_summary_view(
    run: DecisionSimulationRunRecord,
) -> DecisionSimulationSummaryView:
    if run.total_samples > 0:
        progress_pct = min(100.0, max(0.0, run.completed_samples / run.total_samples * 100))
    else:
        progress_pct = min(100.0, max(0.0, run.progress * 100))
    return DecisionSimulationSummaryView(
        runId=run.id,
        decisionId=run.decision_id,
        experimentProgramId=run.experiment_program_id,
        experimentCandidateId=run.experiment_candidate_id,
        mode=run.mode,
        status=run.status,
        progressCompleted=run.completed_samples,
        progressTotal=run.total_samples,
        progressPct=progress_pct,
        candidateCount=run.candidate_count,
        provisionalLeaderId=run.provisional_winner_candidate_id,
        winnerCandidateId=run.winner_candidate_id,
        modelVersion=run.model_version,
        criticVerdict=_critic_verdict(run.critic_result),
        error=_simulation_error_text(run.error),
        createdAt=run.created_at,
        startedAt=run.started_at,
        completedAt=run.completed_at,
        heartbeatAt=run.heartbeat_at,
    )


def _simulation_run_view(
    run: DecisionSimulationRunRecord,
    candidates: list[DecisionSimulationCandidateRecord],
) -> DecisionSimulationRunView:
    summary = _simulation_summary_view(run)
    return DecisionSimulationRunView(
        **summary.model_dump(),
        objective=run.objective,
        snapshotHash=run.evidence_hash,
        evidenceSnapshot=run.evidence_snapshot,
        criticResult=run.critic_result,
        candidates=[_simulation_candidate_view(item) for item in candidates],
    )


def _simulation_candidate_view(
    candidate: DecisionSimulationCandidateRecord,
) -> DecisionSimulationCandidateView:
    return DecisionSimulationCandidateView(
        candidateId=candidate.candidate_id,
        experimentProgramId=candidate.experiment_program_id,
        experimentCandidateId=candidate.experiment_candidate_id,
        status=candidate.status,
        candidate=candidate.candidate_payload,
        constraintViolations=candidate.constraint_violations,
        distribution=candidate.distribution_summary,
        scoreBreakdown=candidate.score_breakdown,
        rank=candidate.rank,
        winner=candidate.is_winner,
        completedSamples=candidate.completed_samples,
        createdAt=candidate.created_at,
        updatedAt=candidate.updated_at,
    )


def _simulation_event_view(
    event: DecisionSimulationEventRecord,
) -> DecisionSimulationEventView:
    return DecisionSimulationEventView(
        id=event.id,
        runId=event.simulation_run_id,
        eventType=event.event_type,
        candidateId=event.candidate_id,
        batchIndex=event.batch_index,
        progressCompleted=_payload_int(
            event.payload,
            "completed_samples",
            "completedSamples",
            "progressCompleted",
        ),
        progressTotal=_payload_int(
            event.payload,
            "total_samples",
            "totalSamples",
            "progressTotal",
        ),
        payload=event.payload,
        createdAt=event.created_at or datetime.now(timezone.utc),
    )


def _payload_int(payload: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = payload.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
    return None


def _critic_verdict(result: dict[str, Any]) -> str | None:
    nested = result.get("result")
    verdict = result.get("verdict")
    if not verdict and isinstance(nested, dict):
        verdict = nested.get("verdict")
    return str(verdict) if verdict else None


def _simulation_error_text(error: dict[str, Any]) -> str | None:
    if not error:
        return None
    for key in ("message", "detail", "code"):
        value = error.get(key)
        if value:
            return str(value)
    return json.dumps(error, ensure_ascii=False, sort_keys=True, default=str)


def _format_simulation_sse_event(event: DecisionSimulationEventRecord) -> str:
    view = _simulation_event_view(event)
    return (
        f"id: {event.id}\n"
        f"event: {event.event_type}\n"
        f"data: {view.model_dump_json()}\n\n"
    )


async def _load_simulation_event_page(
    simulation_run_id: str,
    after_id: int,
) -> list[DecisionSimulationEventRecord]:
    return await asyncio.to_thread(
        list_decision_simulation_events,
        simulation_run_id,
        after_id=after_id,
        limit=_SSE_REPLAY_PAGE_SIZE,
    )


async def _connect_simulation_event_listener() -> psycopg.AsyncConnection[Any]:
    connection = await psycopg.AsyncConnection.connect(
        get_settings().memory_postgres_dsn,
        autocommit=True,
    )
    await connection.execute(f"LISTEN {_SIMULATION_EVENTS_CHANNEL}")
    return connection


async def _wait_for_simulation_notification(
    connection: psycopg.AsyncConnection[Any],
) -> Any | None:
    notification = None
    async for notification in connection.notifies(
        timeout=_SSE_HEARTBEAT_SECONDS,
        stop_after=1,
    ):
        # Consume the generator to its natural stop_after termination. Returning
        # from inside the loop can leave psycopg's connection lock owned by the
        # async generator and deadlock the next wait.
        pass
    return notification


def _simulation_notification_run_id(notification: Any) -> str | None:
    raw_payload = getattr(notification, "payload", None)
    if not isinstance(raw_payload, str) or not raw_payload:
        return None
    try:
        payload = json.loads(raw_payload)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    value = payload.get("simulation_run_id", payload.get("simulationRunId"))
    return str(value) if value is not None else None


class _SimulationNotificationHub:
    """One Postgres LISTEN connection per API event loop, fanned out by run."""

    def __init__(self) -> None:
        self._subscribers: dict[str, set[asyncio.Queue[str]]] = {}
        self._task: asyncio.Task[None] | None = None
        self._ready = asyncio.Event()

    async def subscribe(self, simulation_run_id: str) -> asyncio.Queue[str]:
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=4)
        self._subscribers.setdefault(simulation_run_id, set()).add(queue)
        if self._task is None or self._task.done():
            self._ready.clear()
            self._task = asyncio.create_task(self._listen())
        try:
            await self._ready.wait()
        except BaseException:
            await self.unsubscribe(simulation_run_id, queue)
            raise
        return queue

    async def unsubscribe(
        self,
        simulation_run_id: str,
        queue: asyncio.Queue[str],
    ) -> None:
        subscribers = self._subscribers.get(simulation_run_id)
        if subscribers is not None:
            subscribers.discard(queue)
            if not subscribers:
                self._subscribers.pop(simulation_run_id, None)
        if not self._subscribers and self._task is not None:
            task = self._task
            self._task = None
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    @classmethod
    def _offer(cls, queue: asyncio.Queue[str], message: str) -> None:
        if queue.full():
            queue.get_nowait()
        queue.put_nowait(message)

    @staticmethod
    async def _cooperate() -> None:
        """Yield even when tests replace asyncio.sleep with an immediate mock."""

        loop = asyncio.get_running_loop()
        resumed = loop.create_future()
        loop.call_soon(resumed.set_result, None)
        await resumed

    def _broadcast(self, message: str, simulation_run_id: str | None = None) -> None:
        groups = (
            [self._subscribers.get(simulation_run_id, set())]
            if simulation_run_id is not None
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
                connection = await _connect_simulation_event_listener()
                if needs_replay:
                    self._broadcast("replay")
                    needs_replay = False
                self._ready.set()
                while self._subscribers:
                    notification = await _wait_for_simulation_notification(connection)
                    if notification is None:
                        await self._cooperate()
                        continue
                    run_id = _simulation_notification_run_id(notification)
                    self._broadcast("replay", run_id)
                    await self._cooperate()
            except asyncio.CancelledError:
                raise
            except Exception:
                needs_replay = True
                self._ready.set()
                self._broadcast("reconnecting")
                await asyncio.sleep(1)
                await self._cooperate()
            finally:
                if connection is not None:
                    try:
                        await connection.close()
                    except Exception:
                        logger.warning(
                            "decision simulation LISTEN close failed",
                            exc_info=True,
                        )


_SIMULATION_NOTIFICATION_HUBS: WeakKeyDictionary[
    asyncio.AbstractEventLoop,
    _SimulationNotificationHub,
] = WeakKeyDictionary()


def _simulation_notification_hub() -> _SimulationNotificationHub:
    loop = asyncio.get_running_loop()
    hub = _SIMULATION_NOTIFICATION_HUBS.get(loop)
    if hub is None:
        hub = _SimulationNotificationHub()
        _SIMULATION_NOTIFICATION_HUBS[loop] = hub
    return hub


async def _decision_simulation_event_stream(
    simulation_run_id: str,
    *,
    after_id: int = 0,
) -> AsyncIterator[str]:
    cursor = max(0, int(after_id))
    hub = _simulation_notification_hub()
    notifications = await hub.subscribe(simulation_run_id)
    replay_required = True
    try:
        # The shared LISTEN is installed before replay. This closes the classic
        # gap where an event can be committed between SELECT and subscription.
        while True:
            if replay_required:
                while True:
                    events = await _load_simulation_event_page(simulation_run_id, cursor)
                    for event in events:
                        cursor = event.id
                        yield _format_simulation_sse_event(event)
                    if len(events) < _SSE_REPLAY_PAGE_SIZE:
                        break
                replay_required = False
            try:
                message = await asyncio.wait_for(
                    notifications.get(),
                    timeout=_SSE_HEARTBEAT_SECONDS,
                )
            except asyncio.TimeoutError:
                # Heartbeats are per client. Activity for another run on the
                # shared LISTEN connection must not keep this stream silent.
                yield ": heartbeat\n\n"
                continue
            if message == "reconnecting":
                yield ": reconnecting\n\n"
            # Notifications are only wake-up hints. Replay from the durable
            # journal; heartbeats deliberately do not query Postgres.
            replay_required = True
    finally:
        await hub.unsubscribe(simulation_run_id, notifications)


def _action_view(
    action: AgentActionExperimentRecord,
    *,
    include_measurements: bool = False,
) -> ManagerActionExperimentView:
    measurements = (
        list_agent_action_measurements(action.id) if include_measurements else []
    )
    return ManagerActionExperimentView(
        id=action.id,
        decisionId=action.decision_id,
        actionId=action.action_id,
        experimentProgramId=action.experiment_program_id,
        experimentCandidateId=action.experiment_candidate_id,
        family=action.family,
        target=action.target,
        executionMode=action.execution_mode,
        state=action.state,
        fromValue=action.from_value,
        toValue=action.to_value,
        expectedEffect=action.expected_effect,
        horizon=action.horizon,
        evidence=action.evidence,
        prerequisites=action.prerequisites,
        guardrails=action.guardrails,
        stopCondition=action.stop_condition,
        rollback=action.rollback,
        measurementSpec=action.measurement_spec,
        expiresAt=action.expires_at,
        ownerUserId=action.owner_user_id,
        approvedBy=action.approved_by,
        approvedAt=action.approved_at,
        rejectedBy=action.rejected_by,
        rejectedAt=action.rejected_at,
        rejectionReason=action.rejection_reason,
        managerLabel=action.manager_label,
        preflight=action.preflight_json,
        actuatorResult=action.actuator_result,
        externalOperationId=action.external_operation_id,
        appliedAt=action.applied_at,
        measurementDueAt=action.measurement_due_at,
        closeOutcome=action.close_outcome,
        lastError=action.last_error,
        version=action.version,
        createdAt=action.created_at,
        updatedAt=action.updated_at,
        measurements=[_measurement_view(item) for item in measurements],
    )


def _measurement_view(
    measurement: AgentActionMeasurementRecord,
) -> ManagerActionMeasurementView:
    return ManagerActionMeasurementView(
        id=measurement.id,
        phase=measurement.phase,
        status=measurement.status,
        periodStart=measurement.period_start,
        periodEnd=measurement.period_end,
        dataAsOf=measurement.data_as_of,
        metrics=measurement.metrics,
        causalValid=measurement.causal_valid,
        observedDelta=measurement.observed_delta,
        didEffect=measurement.did_effect,
        forecastLow=measurement.forecast_low,
        forecastBase=measurement.forecast_base,
        forecastHigh=measurement.forecast_high,
        signedBias=measurement.signed_bias,
        mae=measurement.mae,
        insideForecastRange=measurement.inside_forecast_range,
        createdAt=measurement.created_at,
    )
