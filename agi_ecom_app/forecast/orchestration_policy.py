from __future__ import annotations

from typing import Iterable

from agi_ecom_app.forecast.models import ForecastCriticIssue, ForecastPlan, ForecastResolutionPlanStep

_STRUCTURAL_ISSUE_CODES = {
    "missing_plan",
    "missing_target_binding",
    "missing_stop_condition",
    "missing_target_metric",
    "campaign_fact_missing",
    "query_fact_missing",
    "wrong_target_mode",
    "wrong_action_family",
    "resolution_plan_not_executable",
    "contradicting_price_execution",
    "completed_with_blockers",
    "unsafe_price_action_with_blocker",
    "stale_or_future_evidence_for_action",
}


def is_resolution_plan_executable(plan_steps: Iterable[ForecastResolutionPlanStep]) -> bool:
    steps = list(plan_steps)
    if not steps:
        return False
    for step in steps:
        if not step.operation or not step.instruction:
            return False
        if _requires_binding(step) and not (step.advertId or step.normQuery or step.targetEntityId):
            return False
    return True


def is_canonical_deterministic_plan(
    plan: ForecastPlan | None,
    plan_steps: Iterable[ForecastResolutionPlanStep],
) -> bool:
    if plan is None:
        return False
    steps = list(plan_steps)
    if not is_resolution_plan_executable(steps):
        return False
    if plan.action_family == "normquery_bid_optimization":
        return True
    if plan.primary_domain == "marketing" and plan.target_mode in {"sku", "mixed"}:
        return any(step.advertId or step.normQuery or step.targetEntityId for step in steps)
    return False


def has_bound_marketing_steps(plan_steps: Iterable[ForecastResolutionPlanStep]) -> bool:
    return any(step.advertId or step.normQuery or step.targetEntityId for step in plan_steps)


def has_only_textual_issues(
    issues: Iterable[ForecastCriticIssue],
    *,
    plan_executable: bool,
) -> bool:
    issue_list = list(issues)
    if not plan_executable or not issue_list:
        return False
    return all(issue.code not in _STRUCTURAL_ISSUE_CODES for issue in issue_list)


def build_safe_summary_from_plan(plan_steps: Iterable[ForecastResolutionPlanStep]) -> str:
    instructions = [_normalize_instruction(step.instruction) for step in list(plan_steps)[:2] if step.instruction]
    if not instructions:
        return "План действий собран, но требует дополнительной верификации формулировок."
    return " ".join(instructions)


def build_safe_recommendation_from_plan(plan_steps: Iterable[ForecastResolutionPlanStep]) -> str:
    return build_safe_summary_from_plan(plan_steps)


def _requires_binding(step: ForecastResolutionPlanStep) -> bool:
    text = f"{step.operation} {step.instruction}".lower()
    if step.advertId or step.normQuery:
        return True
    return any(marker in text for marker in ("кампан", "advert", "запрос", "query", "ставк", "ключ"))


def _normalize_instruction(value: str) -> str:
    text = value.strip()
    if text.endswith("."):
        return text
    return f"{text}."
