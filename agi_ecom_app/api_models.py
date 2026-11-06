from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field

from agi_ecom_app.forecast.models import ForecastDecisionResult, GrowthOpportunity
from agi_ecom_app.manager_decision.models import ManagerDecisionV1


class DependencyStatus(BaseModel):
    status: str
    detail: str | None = None


class HealthResponse(BaseModel):
    status: str
    memory_db: DependencyStatus
    analytics_db: DependencyStatus
    openai_config: DependencyStatus
    llm_config: DependencyStatus | None = None


class AvailableModelProfile(BaseModel):
    profile: str
    provider: str
    model: str
    api_style: str
    base_url: str | None = None
    required_env_vars: list[str] = Field(default_factory=list)
    missing_env_vars: list[str] = Field(default_factory=list)
    configured: bool
    experimental: bool = False
    description: str = ""


class ModelProfileResponse(BaseModel):
    profile: str
    provider: str
    model: str
    model_label: str
    api_style: str
    status: str
    detail: str | None = None
    memory_provider: str = "openai"
    memory_model: str
    available_profiles: list[AvailableModelProfile] = Field(default_factory=list)


class ModelProfileUpdateRequest(BaseModel):
    profile: str = Field(..., min_length=1)


class ChatSessionResponse(BaseModel):
    chat_id: str
    status: str = "created"


class ChatSessionListItem(BaseModel):
    chat_id: str
    status: str
    model_name: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    message_count: int = 0
    run_count: int = 0
    latest_run_id: int | None = None
    latest_run_status: str | None = None
    latest_run_intent: str | None = None
    latest_run_created_at: datetime | None = None
    latest_server_message_id: int | None = None
    last_user_message: str | None = None
    last_user_message_at: datetime | None = None
    last_assistant_message: str | None = None
    last_assistant_message_at: datetime | None = None


class ChatSessionListResponse(BaseModel):
    sessions: list[ChatSessionListItem]


class ChatMessageRequest(BaseModel):
    chat_id: str
    message: str = Field(..., min_length=1)
    client_message_id: str | None = Field(default=None, min_length=1)


class ChatMessageResponse(BaseModel):
    chat_id: str
    answer: str | None = None
    intent: str
    run_id: int | None = None
    server_message_id: int | None = None
    status: Literal["queued", "running", "completed", "failed", "cancelled"] = "completed"
    managerDecision: ManagerDecisionV1 | None = None


class GrowthOpportunityListResponse(BaseModel):
    status: str = "completed"
    period_days: int
    horizon_days: int
    focus: str
    items: list[GrowthOpportunity]


class GrowthOpportunityForecastRequest(BaseModel):
    opportunity: GrowthOpportunity
    period_days: int = 7
    horizon_days: int = 14


class GrowthOpportunityForecastResponse(BaseModel):
    status: str = "completed"
    forecast: ForecastDecisionResult


class AgentRunStatusResponse(BaseModel):
    run_id: int
    chat_id: str
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    intent: str
    server_message_id: int | None = None
    answer: str | None = None
    error: str | None = None
    managerDecision: ManagerDecisionV1 | None = None


class ChatHistoryMessage(BaseModel):
    seq: int
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime | None = None
    client_message_id: str | None = None
    reply_to_client_message_id: str | None = None
    server_message_id: int | None = None
    intent: str | None = None
    managerDecision: ManagerDecisionV1 | None = None


class ChatHistoryResponse(BaseModel):
    chat_id: str
    messages: list[ChatHistoryMessage]


class AgentTraceEvent(BaseModel):
    ts: datetime
    event: str
    agent: str | None = None
    tool: str | None = None
    round: str | None = None
    target_agent: str | None = None
    question: str | None = None
    status: str
    duration_ms: int | None = None
    details: dict = Field(default_factory=dict)


class AgentRunTrace(BaseModel):
    id: int
    chat_id: str
    server_message_id: int | None = None
    client_message_id: str | None = None
    intent: str
    primary_agent: str | None = None
    status: str
    started_at: datetime | None = None
    completed_at: datetime | None = None
    events: list[AgentTraceEvent] = Field(default_factory=list)
    summary: dict = Field(default_factory=dict)
    created_at: datetime | None = None


class AgentRunTraceListResponse(BaseModel):
    chat_id: str
    traces: list[AgentRunTrace]


class AgentRunGap(BaseModel):
    id: int
    run_id: int
    response_id: int | None = None
    sku: str | None = None
    source_agent: str | None = None
    target_agent: str | None = None
    metric: str
    reason: str
    impact: str | None = None
    blocking: bool
    status: str
    evidence_status: str
    created_at: datetime | None = None
    updated_at: datetime | None = None


class AgentRunResponseSummary(BaseModel):
    id: int
    question_id: int
    agent_name: str
    status: str
    summary: str | None = None
    has_business_evidence: bool
    runtime_failure: dict | None = None
    tool_report: dict | None = None
    created_at: datetime | None = None
    completed_at: datetime | None = None


class AgentRunQuestion(BaseModel):
    id: int
    run_id: int
    round_id: int
    source_agent: str | None = None
    target_agent: str
    question: str
    reason: str | None = None
    verification_goal: dict = Field(default_factory=dict)
    priority: int | None = None
    status: str
    job_id: str | None = None
    queued_at: datetime | None = None
    attempt_count: int = 0
    last_error: str | None = None
    timeout_seconds: int | None = None
    parent_question_id: int | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    responses: list[AgentRunResponseSummary] = Field(default_factory=list)
    gaps: list[AgentRunGap] = Field(default_factory=list)


class AgentRunRound(BaseModel):
    id: int
    run_id: int
    round_number: int
    round_type: str
    status: str
    created_at: datetime | None = None
    updated_at: datetime | None = None
    completed_at: datetime | None = None
    questions: list[AgentRunQuestion] = Field(default_factory=list)


class AgentRunDetailResponse(BaseModel):
    run: AgentRunTrace
    rounds: list[AgentRunRound] = Field(default_factory=list)


class AgentRunRoundsResponse(BaseModel):
    run_id: int
    rounds: list[AgentRunRound] = Field(default_factory=list)


class AgentRunGapsResponse(BaseModel):
    run_id: int
    gaps: list[AgentRunGap] = Field(default_factory=list)


class ChatSocketSendMessage(BaseModel):
    type: Literal["message.send"]
    client_message_id: str = Field(..., min_length=1)
    chat_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)


class ChatSocketConnectionReadyEvent(BaseModel):
    type: Literal["connection.ready"] = "connection.ready"
    chat_id: str


class ChatSocketMessageAcceptedEvent(BaseModel):
    type: Literal["message.accepted"] = "message.accepted"
    client_message_id: str


class ChatSocketMessageQueuedEvent(BaseModel):
    type: Literal["message.queued"] = "message.queued"
    client_message_id: str
    run_id: int
    status: str = "queued"
    intent: str


class ChatSocketRunStatusEvent(BaseModel):
    type: Literal["run.status"] = "run.status"
    client_message_id: str
    run_id: int
    status: str
    server_message_id: int | None = None


class ChatSocketMessageCompletedEvent(BaseModel):
    type: Literal["message.completed"] = "message.completed"
    client_message_id: str
    server_message_id: int | None = None
    answer: str
    intent: str
    managerDecision: ManagerDecisionV1 | None = None


class ChatSocketMessageErrorEvent(BaseModel):
    type: Literal["message.error"] = "message.error"
    client_message_id: str | None = None
    code: str
    detail: str
    retryable: bool = False


class ChatSocketHeartbeatEvent(BaseModel):
    type: Literal["heartbeat"] = "heartbeat"
    ts: datetime


class DashboardResponse(BaseModel):
    sales_units: int | None
    sales_rub: float | None
    orders_today: int | None
    sales_units_delta_pct: float | None
    sales_rub_delta_pct: float | None
    in_way_to_client_units: int | None
    sales_date: date | None
    prev_sales_date: date | None
    orders_date: date | None
    stocks_snapshot_hour: datetime | None
    is_partial: bool = False
    warning: str | None = None


class PromoItem(BaseModel):
    name: str
    eff: str
    sku: int
    alerts: str
    status: str


class ActivePromosResponse(BaseModel):
    items: list[PromoItem]


class FlashSaleSkuDetail(BaseModel):
    sku_label: str
    badge: str
    badge_color: str
    action_text: str
    action_color: str
    sales: str
    sales_status: str
    position: str
    position_status: str = "neutral"
    margin: str
    margin_status: str = "neutral"
    stock: str
    stock_status: str = "neutral"
    note: str | None = None
    note_color: str | None = None


class FlashSaleWinterResponse(BaseModel):
    analyzed_items_text: str
    impact_on_plan: str
    impact_on_plan_status: str
    revenue_text: str
    revenue_status: str
    profit_text: str
    profit_status: str
    plan_status_text: str
    plan_status_kind: str
    margin_text: str
    margin_status: str
    drr_text: str
    drr_status: str
    turnover_text: str
    turnover_status: str
    storage_savings_text: str
    storage_savings_status: str
    sku_details: list[FlashSaleSkuDetail]
    recommendation_text: str
