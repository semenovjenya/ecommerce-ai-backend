from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import JSONResponse

from agi_ecom_app.api_models import (
    ActivePromosResponse,
    AgentRunDetailResponse,
    AgentRunGapsResponse,
    AgentRunRoundsResponse,
    AgentRunStatusResponse,
    AgentRunTrace,
    AgentRunTraceListResponse,
    ChatHistoryResponse,
    ChatMessageRequest,
    ChatMessageResponse,
    ChatSessionListResponse,
    ChatSessionResponse,
    DashboardResponse,
    FlashSaleWinterResponse,
    GrowthOpportunityForecastRequest,
    GrowthOpportunityForecastResponse,
    GrowthOpportunityListResponse,
    HealthResponse,
    ModelProfileResponse,
    ModelProfileUpdateRequest,
)
from agi_ecom_app.backend_service import (
    create_backend_chat_session,
    get_active_promos_payload,
    get_agent_run_detail_payload,
    get_agent_run_gaps_payload,
    get_agent_run_rounds_payload,
    get_agent_run_status_payload,
    get_agent_run_trace_payload,
    get_agent_run_traces_payload,
    get_chat_history_payload,
    get_chat_sessions_payload,
    get_daily_dashboard_payload,
    get_flash_sale_winter_payload,
    get_health_payload,
    get_model_profile_payload,
    process_chat_message,
    update_model_profile_payload,
)
from agi_ecom_app.agent_actions_view_api import router as agent_actions_view_router
from agi_ecom_app.agent_v2_read_api import router as agent_v2_read_router
from agi_ecom_app.runner import AgentRuntimeError
from agi_ecom_app.forecast.opportunity_finder import find_growth_opportunities
from agi_ecom_app.forecast.workflow import run_forecast_workflow
from agi_ecom_app.experiment_program.router import router as experiment_program_router
from agi_ecom_app.websocket_chat import handle_chat_websocket
from agi_ecom_app.sku_previews.router import router as feature1_router
from agi_ecom_app.document_import.router import router as feature2_router
from agi_ecom_app.action_board.router import router as feature3_router
from agi_ecom_app.unit_economics.router import router as feature4_router
from agi_ecom_app.commission_calculator.router import router as feature5_router


@asynccontextmanager
async def _lifespan(_: FastAPI):
    from agi_ecom_app.observability import init_langfuse, shutdown_langfuse

    init_langfuse(service_name="backend", worker_role="api")
    try:
        yield
    finally:
        shutdown_langfuse()


app = FastAPI(title="AI-ECOM Backend API", version="1.0.0", lifespan=_lifespan)
app.include_router(feature1_router)
app.include_router(feature2_router)
app.include_router(feature3_router)
app.include_router(experiment_program_router)
app.include_router(agent_actions_view_router)
app.include_router(agent_v2_read_router)
app.include_router(feature4_router)
app.include_router(feature5_router)


@app.exception_handler(AgentRuntimeError)
async def handle_agent_runtime_error(_, exc: AgentRuntimeError):
    return JSONResponse(
        status_code=503,
        content={"status": "error", "detail": str(exc)},
    )


@app.exception_handler(Exception)
async def handle_unexpected_error(_, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"status": "error", "detail": str(exc)},
    )


@app.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    return get_health_payload()


@app.get("/v1/model-profile", response_model=ModelProfileResponse)
def get_model_profile() -> ModelProfileResponse:
    return get_model_profile_payload()


@app.put("/v1/model-profile", response_model=ModelProfileResponse)
def update_model_profile(payload: ModelProfileUpdateRequest) -> ModelProfileResponse:
    try:
        return update_model_profile_payload(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None


@app.post("/v1/chat/sessions", response_model=ChatSessionResponse)
def create_chat_session() -> ChatSessionResponse:
    return create_backend_chat_session()


@app.get("/v1/chat/sessions", response_model=ChatSessionListResponse)
def get_chat_sessions(limit: int = 50) -> ChatSessionListResponse:
    return get_chat_sessions_payload(limit=limit)


@app.post("/v1/chat/messages", response_model=ChatMessageResponse, status_code=202)
def create_chat_message(payload: ChatMessageRequest) -> ChatMessageResponse:
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="message must not be empty")
    return process_chat_message(
        chat_id=payload.chat_id,
        message=message,
        client_message_id=payload.client_message_id,
    )


@app.get("/v1/chat/messages", response_model=ChatHistoryResponse)
def get_chat_history(chat_id: str) -> ChatHistoryResponse:
    try:
        return get_chat_history_payload(chat_id)
    except LookupError:
        raise HTTPException(status_code=404, detail="chat session not found") from None


@app.get("/v1/chat/agent-traces", response_model=AgentRunTraceListResponse)
def get_agent_run_traces(chat_id: str, limit: int = 50) -> AgentRunTraceListResponse:
    try:
        return get_agent_run_traces_payload(chat_id=chat_id, limit=limit)
    except LookupError:
        raise HTTPException(status_code=404, detail="chat session not found") from None


@app.get("/v1/chat/agent-traces/{server_message_id}", response_model=AgentRunTrace)
def get_agent_run_trace(server_message_id: int, chat_id: str) -> AgentRunTrace:
    try:
        return get_agent_run_trace_payload(
            chat_id=chat_id,
            server_message_id=server_message_id,
        )
    except LookupError:
        raise HTTPException(status_code=404, detail="agent trace not found") from None


@app.get("/v1/chat/agent-runs/{run_id}", response_model=AgentRunDetailResponse)
def get_agent_run_detail(run_id: int, chat_id: str | None = None) -> AgentRunDetailResponse:
    try:
        return get_agent_run_detail_payload(run_id=run_id, chat_id=chat_id)
    except LookupError:
        raise HTTPException(status_code=404, detail="agent run not found") from None


@app.get("/v1/chat/agent-runs/{run_id}/status", response_model=AgentRunStatusResponse)
def get_agent_run_status(run_id: int, chat_id: str | None = None) -> AgentRunStatusResponse:
    try:
        return get_agent_run_status_payload(run_id=run_id, chat_id=chat_id)
    except LookupError:
        raise HTTPException(status_code=404, detail="agent run not found") from None


@app.get("/v1/chat/agent-runs/{run_id}/rounds", response_model=AgentRunRoundsResponse)
def get_agent_run_rounds(run_id: int, chat_id: str | None = None) -> AgentRunRoundsResponse:
    try:
        return get_agent_run_rounds_payload(run_id=run_id, chat_id=chat_id)
    except LookupError:
        raise HTTPException(status_code=404, detail="agent run not found") from None


@app.get("/v1/chat/agent-runs/{run_id}/gaps", response_model=AgentRunGapsResponse)
def get_agent_run_gaps(
    run_id: int,
    chat_id: str | None = None,
    status: str | None = None,
    blocking: bool | None = None,
    sku: str | None = None,
) -> AgentRunGapsResponse:
    try:
        return get_agent_run_gaps_payload(
            run_id=run_id,
            chat_id=chat_id,
            status=status,
            blocking=blocking,
            sku=sku,
        )
    except LookupError:
        raise HTTPException(status_code=404, detail="agent run not found") from None
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None


@app.websocket("/v1/chat/ws")
async def chat_websocket(websocket: WebSocket):
    await handle_chat_websocket(websocket)


@app.get("/v1/dashboard/daily", response_model=DashboardResponse)
def get_daily_dashboard() -> DashboardResponse:
    return get_daily_dashboard_payload()


@app.get("/v1/forecast/growth-opportunities", response_model=GrowthOpportunityListResponse)
def get_growth_opportunities(
    period_days: int = 7,
    horizon_days: int = 14,
    limit: int = 20,
    focus: str = "margin_revenue",
) -> GrowthOpportunityListResponse:
    try:
        items = find_growth_opportunities(
            period_days=period_days,
            horizon_days=horizon_days,
            limit=limit,
            focus=focus,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    return GrowthOpportunityListResponse(
        period_days=period_days,
        horizon_days=horizon_days,
        focus=focus,
        items=items,
    )


@app.post(
    "/v1/forecast/growth-opportunities/forecast",
    response_model=GrowthOpportunityForecastResponse,
)
def forecast_growth_opportunity(
    payload: GrowthOpportunityForecastRequest,
) -> GrowthOpportunityForecastResponse:
    forecast = run_forecast_workflow(
        task_snapshot=None,
        opportunity=payload.opportunity,
    )
    if forecast is None:
        raise HTTPException(status_code=422, detail="growth opportunity cannot be forecasted")
    return GrowthOpportunityForecastResponse(forecast=forecast)


@app.get("/v1/promos/active", response_model=ActivePromosResponse)
def get_active_promos() -> ActivePromosResponse:
    return get_active_promos_payload()


@app.get("/v1/promos/flash-sale-winter", response_model=FlashSaleWinterResponse)
def get_flash_sale_winter() -> FlashSaleWinterResponse:
    return get_flash_sale_winter_payload()
