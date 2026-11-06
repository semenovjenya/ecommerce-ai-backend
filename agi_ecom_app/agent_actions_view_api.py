from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from agi_ecom_app.forecast.models import ForecastDecisionResult
from agi_ecom_app.schemas import ActionExecutionResult
from agi_ecom_app.action_board.service import get_agent_action_execution_payload


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


router = APIRouter()


@router.get(
    "/v1/agent-actions/executions/{execution_id}/view",
    response_model=AgentActionExecutionViewItem,
)
def get_agent_action_execution_view(execution_id: int) -> AgentActionExecutionViewItem:
    try:
        item = get_agent_action_execution_payload(execution_id)
    except LookupError:
        raise HTTPException(status_code=404, detail="agent action execution not found") from None
    return _execution_item_to_view(item)


def _execution_item_to_view(item) -> AgentActionExecutionViewItem:
    return AgentActionExecutionViewItem(
        id=item.id,
        status=item.status,
        resultKind=item.resultKind,
        decisionStatus=item.decisionStatus,
        reviewSummary=item.reviewSummary,
        objectExternalId=item.objectExternalId,
        objectPayload=dict(item.objectPayload or {}),
        result=_compact_action_execution_result(item.result),
        forecastResult=_compact_forecast_result(item.forecastResult),
        error=item.error,
    )


def _compact_action_execution_result(
    result: ActionExecutionResult | None,
) -> AgentActionExecutionResultView | None:
    if result is None:
        return None
    return AgentActionExecutionResultView(
        decisionStatus=result.decisionStatus,
        confidence=result.confidence,
        blockedBy=list(result.blockedBy or []),
        summary=result.summary,
        finalRecommendation=result.finalRecommendation,
    )


def _compact_forecast_result(
    forecast_result: ForecastDecisionResult | None,
) -> AgentActionExecutionForecastView | None:
    if forecast_result is None:
        return None
    plan_family = None
    if forecast_result.plan is not None:
        plan_family = forecast_result.plan.action_family
    if plan_family is None:
        plan_family = _optional_string(
            (forecast_result.forecastDetails or {}).get("recommendedActionFamily")
        )
    return AgentActionExecutionForecastView(
        planFamily=plan_family,
        decisionStatus=forecast_result.decisionStatus,
        confidencePercent=forecast_result.confidencePercent,
        profitForecastRub=forecast_result.profitForecastRub,
        economicEffectRub=forecast_result.economicEffectRub,
        economicEffectPctOfCurrentSpend=forecast_result.economicEffectPctOfCurrentSpend,
        blockedBy=list(forecast_result.blockedBy or []),
        reviewBlocked=forecast_result.reviewBlocked,
        publishedWithTextFallback=bool(
            (forecast_result.forecastDetails or {}).get("publishedWithTextFallback")
        ),
        summary=forecast_result.summary,
        finalRecommendation=forecast_result.finalRecommendation,
        resolutionPlan=[
            AgentActionExecutionViewPlanStep(
                operation=step.operation,
                instruction=step.instruction,
            )
            for step in forecast_result.resolutionPlan
        ],
    )


def _optional_string(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
