from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, FiniteFloat

from agi_ecom_app.forecast.models import ForecastDecisionResult
from agi_ecom_app.manager_decision.models import SimulationEligibilityV1
from agi_ecom_app.schemas import ActionExecutionResult


class AgentActionTaskItem(BaseModel):
    taskId: str
    actionId: str
    chatId: str
    serverMessageId: int
    source: str = "agent_finding"
    factor: str
    observation: str | None = None
    impact: str | None = None
    cause: str | None = None
    action: str
    title: str | None = None
    rawObjectText: str | None = None
    executionReadiness: str | None = None
    executionReason: str | None = None
    actionType: (
        Literal["check", "create", "update", "calculate", "monitor", "request_info"] | None
    ) = None
    priority: Literal["low", "medium", "high", "critical"] | None = None
    targets: list[str] = Field(default_factory=list)
    successCriteria: str | None = None
    agentPrompt: str | None = None
    actionKind: str | None = None
    businessGoal: str | None = None
    decisionType: str | None = None
    requiredMetrics: list[str] = Field(default_factory=list)
    forecastSpec: dict = Field(default_factory=dict)
    executionPlan: str | None = None
    objects: list["AgentActionTaskObject"] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    confidence: str | None = None
    driverClass: str | None = None
    intent: str | None = None
    createdAt: datetime | None = None
    selected: bool = False


class AgentActionRecommendationItem(BaseModel):
    recommendationId: str
    chatId: str
    serverMessageId: int
    source: str = "markdown_fallback"
    factor: str
    observation: str | None = None
    impact: str | None = None
    recommendation: str
    intent: str | None = None
    createdAt: datetime | None = None


class AgentActionTasksResponse(BaseModel):
    status: str = "completed"
    chatId: str
    sourceMessageId: int | None = None
    items: list[AgentActionTaskItem]
    recommendations: list[AgentActionRecommendationItem] = Field(default_factory=list)


class AgentActionSelectionRequest(BaseModel):
    chat_id: str = Field(..., min_length=1)
    task_id: str = Field(..., min_length=1)


class AgentActionTaskSnapshot(BaseModel):
    taskId: str = Field(..., min_length=1)
    actionId: str | None = None
    chatId: str | None = None
    serverMessageId: int | None = None
    source: str | None = None
    factor: str | None = None
    observation: str | None = None
    impact: str | None = None
    cause: str | None = None
    action: str | None = None
    title: str | None = None
    rawObjectText: str | None = None
    executionReadiness: str | None = None
    executionReason: str | None = None
    actionType: str | None = None
    priority: str | None = None
    targets: list[str] = Field(default_factory=list)
    successCriteria: str | None = None
    agentPrompt: str | None = None
    actionKind: str | None = None
    businessGoal: str | None = None
    decisionType: str | None = None
    requiredMetrics: list[str] = Field(default_factory=list)
    forecastSpec: dict = Field(default_factory=dict)
    executionPlan: str | None = None
    objects: list["AgentActionTaskObject"] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    confidence: str | None = None
    driverClass: str | None = None
    intent: str | None = None
    createdAt: datetime | None = None
    selected: bool | None = None


class AgentActionTaskObject(BaseModel):
    objectId: str = Field(..., min_length=1)
    label: str | None = None
    type: str | None = None
    externalId: str | None = None
    payload: dict = Field(default_factory=dict)


class AgentActionExecutionRequest(BaseModel):
    chat_id: str = Field(..., min_length=1)
    task_id: str = Field(..., min_length=1)
    task: AgentActionTaskSnapshot | None = None


class AgentActionExecutionItem(BaseModel):
    id: int
    sourceChatId: str
    taskId: str
    sourceTaskId: str
    executionChatId: str
    runId: int | None = None
    clientMessageId: str
    status: Literal["queued", "running", "completed", "failed", "cancelled", "done"]
    executionMode: str = "terminal_action"
    prompt: str
    task: AgentActionTaskSnapshot
    objectId: str
    objectLabel: str | None = None
    objectType: str | None = None
    objectExternalId: str | None = None
    objectPayload: dict = Field(default_factory=dict)
    resultKind: Literal["action_execution", "forecast_decision"] | None = None
    result: ActionExecutionResult | None = None
    forecastResult: ForecastDecisionResult | None = None
    decisionStatus: str | None = None
    reviewSummary: str | None = None
    answer: str | None = None
    serverMessageId: int | None = None
    error: str | None = None
    createdAt: datetime | None = None
    updatedAt: datetime | None = None
    completedAt: datetime | None = None


class AgentActionExecutionViewPlanStep(BaseModel):
    operation: str
    instruction: str


class AgentActionExecutionForecastView(BaseModel):
    planFamily: str | None = None
    decisionStatus: str | None = None
    confidencePercent: int | None = None
    profitForecastRub: float | None = None
    economicEffectRub: float | None = None
    economicEffectPctOfCurrentSpend: float | None = None
    blockedBy: list[str] = Field(default_factory=list)
    reviewBlocked: bool = False
    publishedWithTextFallback: bool = False
    summary: str | None = None
    finalRecommendation: str | None = None
    resolutionPlan: list[AgentActionExecutionViewPlanStep] = Field(default_factory=list)


class AgentActionExecutionResultView(BaseModel):
    decisionStatus: str | None = None
    confidence: str | None = None
    blockedBy: list[str] = Field(default_factory=list)
    summary: str | None = None
    finalRecommendation: str | None = None


class AgentActionExecutionViewItem(BaseModel):
    id: int
    status: Literal["queued", "running", "completed", "failed", "cancelled", "done"]
    resultKind: Literal["action_execution", "forecast_decision"] | None = None
    decisionStatus: str | None = None
    reviewSummary: str | None = None
    objectExternalId: str | None = None
    objectPayload: dict = Field(default_factory=dict)
    result: AgentActionExecutionResultView | None = None
    forecastResult: AgentActionExecutionForecastView | None = None
    error: str | None = None


class AgentActionExecutionsResponse(BaseModel):
    chatId: str
    items: list[AgentActionExecutionItem]


class AgentActionExecutionBatchResponse(BaseModel):
    chatId: str
    sourceTaskId: str
    items: list[AgentActionExecutionItem]


AgentActionTaskSnapshot.model_rebuild()


class AuthLoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=200)
    password: str = Field(..., min_length=1, max_length=1000)


class AuthUserView(BaseModel):
    id: int
    username: str
    role: Literal["viewer", "manager", "admin"]


class AuthLoginResponse(BaseModel):
    accessToken: str
    tokenType: str = "bearer"
    expiresIn: int
    user: AuthUserView


class ManagerActionExperimentView(BaseModel):
    id: str
    decisionId: str
    actionId: str
    experimentProgramId: str | None = None
    experimentCandidateId: str | None = None
    family: str
    target: dict[str, Any] = Field(default_factory=dict)
    executionMode: Literal["human_task", "wb_ads", "wb_price"]
    state: Literal[
        "proposed",
        "approved",
        "applying",
        "applied",
        "measuring",
        "measured",
        "closed",
        "blocked",
        "rejected",
        "failed",
    ]
    fromValue: Any = None
    toValue: Any = None
    expectedEffect: dict[str, Any] = Field(default_factory=dict)
    horizon: dict[str, Any] = Field(default_factory=dict)
    evidence: list[Any] = Field(default_factory=list)
    prerequisites: list[Any] = Field(default_factory=list)
    guardrails: dict[str, Any] = Field(default_factory=dict)
    stopCondition: dict[str, Any] = Field(default_factory=dict)
    rollback: dict[str, Any] = Field(default_factory=dict)
    measurementSpec: dict[str, Any] = Field(default_factory=dict)
    expiresAt: datetime | None = None
    ownerUserId: int | None = None
    approvedBy: int | None = None
    approvedAt: datetime | None = None
    rejectedBy: int | None = None
    rejectedAt: datetime | None = None
    rejectionReason: str | None = None
    managerLabel: str | None = None
    preflight: dict[str, Any] = Field(default_factory=dict)
    actuatorResult: dict[str, Any] = Field(default_factory=dict)
    externalOperationId: str | None = None
    appliedAt: datetime | None = None
    measurementDueAt: datetime | None = None
    closeOutcome: str | None = None
    lastError: str | None = None
    version: int
    createdAt: datetime | None = None
    updatedAt: datetime | None = None
    measurements: list["ManagerActionMeasurementView"] = Field(default_factory=list)


class ManagerActionMeasurementView(BaseModel):
    id: int
    phase: Literal["baseline", "post", "control", "readout"]
    status: Literal["waiting_for_data", "observed", "computed", "inconclusive"]
    periodStart: datetime | None = None
    periodEnd: datetime | None = None
    dataAsOf: datetime | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    causalValid: bool = False
    observedDelta: float | None = None
    didEffect: float | None = None
    forecastLow: float | None = None
    forecastBase: float | None = None
    forecastHigh: float | None = None
    signedBias: float | None = None
    mae: float | None = None
    insideForecastRange: bool | None = None
    createdAt: datetime | None = None


class DecisionSimulationSummaryView(BaseModel):
    runId: str
    decisionId: str
    experimentProgramId: str | None = None
    experimentCandidateId: str | None = None
    mode: Literal["shadow", "enforced"]
    status: Literal[
        "initializing",
        "queued",
        "running",
        "critic_running",
        "completed",
        "blocked",
        "failed",
        "cancelled",
        "superseded",
    ]
    progressCompleted: int = 0
    progressTotal: int = 0
    progressPct: float = 0
    candidateCount: int = 0
    provisionalLeaderId: str | None = None
    winnerCandidateId: str | None = None
    modelVersion: str
    criticVerdict: str | None = None
    error: str | None = None
    createdAt: datetime | None = None
    startedAt: datetime | None = None
    completedAt: datetime | None = None
    heartbeatAt: datetime | None = None


class DecisionSimulationCandidateView(BaseModel):
    candidateId: str
    experimentProgramId: str | None = None
    experimentCandidateId: str | None = None
    status: str
    candidate: dict[str, Any] = Field(default_factory=dict)
    constraintViolations: list[Any] = Field(default_factory=list)
    distribution: dict[str, Any] = Field(default_factory=dict)
    scoreBreakdown: dict[str, Any] = Field(default_factory=dict)
    rank: int | None = None
    winner: bool = False
    completedSamples: int = 0
    createdAt: datetime | None = None
    updatedAt: datetime | None = None


class DecisionSimulationRunView(DecisionSimulationSummaryView):
    objective: str
    snapshotHash: str
    evidenceSnapshot: dict[str, Any] = Field(default_factory=dict)
    criticResult: dict[str, Any] = Field(default_factory=dict)
    candidates: list[DecisionSimulationCandidateView] = Field(default_factory=list)


class DecisionSimulationCandidatesResponse(BaseModel):
    runId: str
    items: list[DecisionSimulationCandidateView] = Field(default_factory=list)


class DecisionSimulationEventView(BaseModel):
    id: int
    runId: str
    eventType: str
    candidateId: str | None = None
    batchIndex: int | None = None
    progressCompleted: int | None = None
    progressTotal: int | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    createdAt: datetime


class DecisionSimulationEventsResponse(BaseModel):
    runId: str
    items: list[DecisionSimulationEventView] = Field(default_factory=list)


class PricingSimulationStartRequest(BaseModel):
    """One SKU and an optional explicit shadow price grid.

    The durable source loader always adds the current price as the hold
    baseline, therefore callers provide at most six alternative prices.
    """

    nmId: int = Field(..., gt=0)
    candidatePrices: list[Annotated[FiniteFloat, Field(gt=0)]] | None = Field(
        default=None,
        min_length=2,
        max_length=6,
    )
    horizonDays: Literal[7] = 7


class ManagerDecisionView(BaseModel):
    id: str
    decisionDate: str
    objective: str
    experimentProgramId: str | None = None
    status: Literal["evaluating", "ready_for_approval", "needs_data", "blocked"]
    requestCoverage: dict[str, Any] = Field(default_factory=dict)
    evidenceAsOf: datetime | None = None
    rankedActions: list[dict[str, Any]] = Field(default_factory=list)
    rejectedCandidates: list[dict[str, Any]] = Field(default_factory=list)
    blockers: list[Any] = Field(default_factory=list)
    publicDecision: dict[str, Any] = Field(default_factory=dict)
    simulationEligibility: SimulationEligibilityV1 | None = None
    simulationSummary: DecisionSimulationSummaryView | None = None
    actions: list[ManagerActionExperimentView] = Field(default_factory=list)
    createdAt: datetime | None = None
    updatedAt: datetime | None = None


class ManagerTodayDecisionsResponse(BaseModel):
    date: str
    items: list[ManagerDecisionView] = Field(default_factory=list)


class ManagerActionRejectRequest(BaseModel):
    reason: str = Field(..., min_length=1, max_length=2000)
    managerLabel: Literal[
        "wrong_data",
        "wrong_priority",
        "unsafe",
        "not_executable",
        "missing_context",
    ] | None = None


class ManagerActionCompleteRequest(BaseModel):
    actualAt: datetime


ManagerActionExperimentView.model_rebuild()
