from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


ForecastDomain = Literal["marketing", "supply", "sales", "price", "finance", "catalog_demand"]
ForecastConfidence = Literal["low", "medium", "high"]
ForecastDecisionStatus = Literal["completed", "needs_review", "blocked"]
ForecastEntityType = Literal["sku", "campaign", "query", "mixed", "unknown"]
ForecastActionFamily = Literal[
    "stop_or_reduce_campaigns",
    "availability_before_scale_guardrail",
    "weak_sku_demand_recovery",
    "sales_trend_guardrail",
    "price_positioning_guardrail",
    "scale_profitable_sku_ads",
    "reprice_with_margin_guardrail",
    "reorder_for_growth",
    "overstock_liquidation",
    "normquery_bid_optimization",
    "recover_conversion_drop",
]
ForecastTargetMode = Literal["sku", "campaign", "query", "mixed", "unknown"]
ForecastCriticVerdict = Literal["passed", "revise_reasoning", "replan", "review_blocked"]
ForecastRetryScope = Literal["reasoning", "full_replan", "none"]
ForecastPlannerHintCode = Literal[
    "target_binding_missing",
    "wrong_target_mode",
    "wrong_action_family",
    "campaign_fact_missing",
    "query_fact_missing",
    "resolution_plan_not_executable",
    "technical_user_facing_text",
]


class ForecastHypothesis(BaseModel):
    domain: ForecastDomain
    action: str
    targets: list[str] = Field(default_factory=list)
    horizon_days: int = 14
    period_days: int = 7
    goal: str | None = None
    effect_metrics: list[str] = Field(default_factory=list)
    source_action_kind: str | None = None


class ResolvedTarget(BaseModel):
    id: str
    entity_type: ForecastEntityType = "unknown"
    source_type: str = "task_snapshot"
    confidence: ForecastConfidence = "medium"
    label: str | None = None


class ForecastPlan(BaseModel):
    primary_domain: ForecastDomain
    action_family: ForecastActionFamily
    target_mode: ForecastTargetMode = "unknown"
    secondary_domains: list[ForecastDomain] = Field(default_factory=list)
    horizon_days: int = 14
    period_days: int = 7
    effect_metrics: list[str] = Field(default_factory=list)
    goal: str | None = None
    source_action_kind: str | None = None


class ForecastRouteProposal(BaseModel):
    route: bool = False
    action_family: ForecastActionFamily | None = None
    primary_domain: ForecastDomain | None = None
    target_mode: ForecastTargetMode | None = None
    confidence: ForecastConfidence = "medium"
    rationale: str = ""


class ForecastScenario(BaseModel):
    name: Literal["baseline", "action"]
    metrics: dict[str, float | int | str | None] = Field(default_factory=dict)
    assumptions: list[str] = Field(default_factory=list)


class ForecastDelta(BaseModel):
    metrics: dict[str, float | int | str | None] = Field(default_factory=dict)


class ForecastUiMetric(BaseModel):
    label: str
    value: str
    tone: str | None = None


class ForecastResolutionAction(BaseModel):
    area: str
    action: str
    reason: str | None = None
    expectedMetric: str | None = None


class ForecastResolutionPlanStep(BaseModel):
    stepId: str
    area: str
    operation: str
    instruction: str
    ownerAgent: str | None = None
    targetEntityType: str | None = None
    targetEntityId: str | None = None
    advertId: str | None = None
    normQuery: str | None = None
    currentMetric: str | None = None
    targetMetric: str | None = None
    stopCondition: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class ForecastUiCard(BaseModel):
    title: str = "Прогноз эффекта"
    whatChecking: str = ""
    whatFound: str = ""
    ifNoAction: str = ""
    ifAction: str = ""
    expectedEffect: str = ""
    recommendation: str = ""
    confidenceReason: str | None = None
    blockers: list[str] = Field(default_factory=list)
    highlightMetrics: list[ForecastUiMetric] = Field(default_factory=list)
    nextChecks: list[str] = Field(default_factory=list)


class ForecastReasoningInput(BaseModel):
    hypothesis: str
    plan: dict[str, Any] = Field(default_factory=dict)
    resolved_targets: list[dict[str, Any]] = Field(default_factory=list)
    per_target_evidence: list[dict[str, Any]] = Field(default_factory=list)
    baseline_metrics: dict[str, Any] = Field(default_factory=dict)
    action_metrics: dict[str, Any] = Field(default_factory=dict)
    delta_metrics: dict[str, Any] = Field(default_factory=dict)
    hard_blockers: list[str] = Field(default_factory=list)
    soft_warnings: list[str] = Field(default_factory=list)
    data_freshness: dict[str, Any] = Field(default_factory=dict)
    source_status: dict[str, Any] = Field(default_factory=dict)
    margin_bridge: dict[str, Any] = Field(default_factory=dict)
    opportunity: dict[str, Any] | None = None
    evidence_summary: dict[str, Any] = Field(default_factory=dict)
    calculator_output: dict[str, Any] = Field(default_factory=dict)


class ForecastReasoningUiCardFields(BaseModel):
    title: str | None = None
    whatChecking: str | None = None
    whatFound: str | None = None
    ifNoAction: str | None = None
    ifAction: str | None = None
    expectedEffect: str | None = None
    recommendation: str | None = None
    confidenceReason: str | None = None
    blockers: list[str] = Field(default_factory=list)
    nextChecks: list[str] = Field(default_factory=list)


class ForecastReasoningResult(BaseModel):
    decision_status: ForecastDecisionStatus = "needs_review"
    confidence: ForecastConfidence = "medium"
    decision: str
    final_recommendation: str
    summary: str
    decision_why: str
    primary_action: str
    rejected_actions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    evidence_used: list[str] = Field(default_factory=list)
    ui_card_fields: ForecastReasoningUiCardFields | None = None
    resolution_plan_patch: list["ForecastResolutionPlanStep"] = Field(default_factory=list)
    forecast_details_patch: dict[str, Any] = Field(default_factory=dict)


class ForecastPlannerHint(BaseModel):
    code: ForecastPlannerHintCode
    message: str | None = None
    forcedActionFamily: ForecastActionFamily | None = None
    forcedTargetMode: ForecastTargetMode | None = None
    forcedEntityType: ForecastEntityType | None = None
    excludedActionFamilies: list[ForecastActionFamily] = Field(default_factory=list)
    preserveActionFamily: bool = False
    preferDeterministicPlan: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class ForecastRevisionInput(ForecastReasoningInput):
    previous_decision: dict[str, Any] = Field(default_factory=dict)
    critic_issues: list[dict[str, Any]] = Field(default_factory=list)
    revision_brief: list[str] = Field(default_factory=list)
    retry_attempt: int = 1


class ForecastIterationRecord(BaseModel):
    attempt: int
    stage: Literal["initial", "revise_reasoning", "replan"]
    verdict: ForecastCriticVerdict
    retryScope: ForecastRetryScope
    decisionStatus: ForecastDecisionStatus
    reviewBlocked: bool = False
    plannerHints: list["ForecastPlannerHint"] = Field(default_factory=list)
    criticIssues: list["ForecastCriticIssue"] = Field(default_factory=list)


class ForecastCriticIssue(BaseModel):
    path: str
    code: str
    severity: Literal["low", "medium", "high"] = "medium"
    message: str


class ForecastCriticResult(BaseModel):
    passed: bool = True
    verdict: ForecastCriticVerdict = "passed"
    retryScope: ForecastRetryScope = "none"
    issues: list[ForecastCriticIssue] = Field(default_factory=list)
    blockingIssues: list[str] = Field(default_factory=list)
    plannerHints: list[ForecastPlannerHint] = Field(default_factory=list)
    criticSummary: str = ""
    rewriteBrief: list[str] = Field(default_factory=list)
    revisionBrief: list[str] = Field(default_factory=list)
    planExecutable: bool = False
    uiSafe: bool = True


class ForecastDecisionResult(BaseModel):
    hypothesis: str
    domain: ForecastDomain
    baseline: ForecastScenario
    actionScenario: ForecastScenario
    delta: ForecastDelta
    decision: str
    decisionStatus: ForecastDecisionStatus = "completed"
    confidence: ForecastConfidence
    confidencePercent: int | None = None
    profitForecastRub: float | None = None
    economicEffectRub: float | None = None
    economicEffectPctOfCurrentSpend: float | None = None
    blockedBy: list[str] = Field(default_factory=list)
    finalRecommendation: str
    summary: str
    resolutionActions: list[ForecastResolutionAction] = Field(default_factory=list)
    resolutionPlan: list[ForecastResolutionPlanStep] = Field(default_factory=list)
    plan: ForecastPlan | None = None
    resolvedTargets: list[ResolvedTarget] = Field(default_factory=list)
    baselineNarrative: str | None = None
    actionNarrative: str | None = None
    effectNarrative: str | None = None
    decisionWhy: str | None = None
    reviewBlocked: bool = False
    reviewBlockReason: str | None = None
    criticIssues: list[ForecastCriticIssue] = Field(default_factory=list)
    iterationCount: int | None = None
    orchestrationTrace: dict[str, Any] | None = None
    uiCard: ForecastUiCard | None = None
    forecastDetails: dict[str, Any] = Field(default_factory=dict)
    debugData: dict[str, Any] = Field(default_factory=dict)
    evidence: dict[str, Any] = Field(default_factory=dict)


class GrowthOpportunity(BaseModel):
    opportunity_id: str
    action_family: ForecastActionFamily
    score: float = 0
    evidence_summary: str = ""
    blockers: list[str] = Field(default_factory=list)
    affected_skus: list[str] = Field(default_factory=list)
    affected_campaigns: list[str] = Field(default_factory=list)
    affected_queries: list[str] = Field(default_factory=list)
    expected_metric_focus: list[str] = Field(default_factory=list)
    target_mode: ForecastTargetMode = "unknown"
    primary_domain: ForecastDomain = "sales"
    growth_reason: str = ""
    source_facts: dict[str, Any] = Field(default_factory=dict)

    def target_ids(self) -> list[str]:
        if self.target_mode == "campaign":
            return list(self.affected_campaigns)
        if self.target_mode == "query":
            return list(self.affected_queries)
        if self.target_mode == "mixed":
            return [*self.affected_skus, *self.affected_campaigns, *self.affected_queries]
        return list(self.affected_skus)

    def to_task_snapshot(self, *, period_days: int = 7, horizon_days: int = 14) -> dict[str, Any]:
        targets = self.target_ids()
        return {
            "actionKind": "growth_opportunity_forecast",
            "decisionType": "growth_opportunity",
            "title": self.evidence_summary or self.growth_reason or self.action_family,
            "action": self.growth_reason or self.evidence_summary or self.action_family,
            "targets": targets,
            "businessGoal": "Рост маржи и выручки через безопасное действие по найденной возможности.",
            "forecastSpec": {
                "horizon_days": horizon_days,
                "period_days": period_days,
                "forecast_capable": True,
                "forecast_domain": self.primary_domain,
                "action_family": self.action_family,
                "target_mode": self.target_mode,
                "effect_metrics": list(self.expected_metric_focus),
            },
            "opportunity": self.model_dump(mode="json"),
        }
