from __future__ import annotations

from datetime import date
from typing import Any

from agi_ecom_app.forecast.adapters.diagnostic_input import build_forecast_evidence_bundle
from agi_ecom_app.forecast.adapters.task_normalizer import (
    build_forecast_plan,
    resolve_forecast_targets,
)
from agi_ecom_app.forecast.calculators.action_family import calculate_forecast_scenario
from agi_ecom_app.forecast.critic import run_forecast_critic
from agi_ecom_app.forecast.models import (
    ForecastCriticResult,
    ForecastDecisionResult,
    ForecastHypothesis,
    ForecastIterationRecord,
    ForecastPlannerHint,
    GrowthOpportunity,
)
from agi_ecom_app.forecast.orchestration_policy import (
    build_safe_recommendation_from_plan,
    build_safe_summary_from_plan,
    has_only_textual_issues,
    is_canonical_deterministic_plan,
)
from agi_ecom_app.forecast.reasoning import (
    build_forecast_reasoning_input,
    build_forecast_revision_input,
    run_forecast_reasoning,
)
from agi_ecom_app.forecast.router import select_calculator
from agi_ecom_app.forecast.synthesis import build_forecast_decision

MAX_FORECAST_RETRIES = 2


def run_forecast_workflow(
    *,
    task_snapshot: dict[str, Any] | None,
    opportunity: GrowthOpportunity | dict[str, Any] | None = None,
    diagnostic_package: dict[str, Any] | None = None,
    today: date | None = None,
) -> ForecastDecisionResult | None:
    normalized_opportunity = _normalize_opportunity(opportunity)
    if task_snapshot is None and normalized_opportunity is not None:
        task_snapshot = normalized_opportunity.to_task_snapshot()
    diagnostic_package = _with_opportunity(diagnostic_package, normalized_opportunity)
    planner_hints: list[ForecastPlannerHint] = []
    iterations: list[ForecastIterationRecord] = []
    attempt = 0
    pass_state = _run_full_pass(
        task_snapshot=task_snapshot,
        diagnostic_package=diagnostic_package,
        opportunity=normalized_opportunity,
        planner_hints=planner_hints,
        today=today,
    )
    if pass_state is None:
        return None

    stage = "initial"
    while True:
        decision_result = pass_state["decision_result"]
        critic_result, critic_debug = run_forecast_critic(
            decision_result=decision_result,
            today=today,
            retry_attempt=attempt,
            max_retries=MAX_FORECAST_RETRIES,
        )
        decision_result = _apply_critic_result(
            decision_result,
            critic_result,
            critic_debug,
            attempt=attempt + 1,
            stage=stage,
            planner_hints=planner_hints,
        )
        iteration = ForecastIterationRecord(
            attempt=attempt + 1,
            stage=stage,
            verdict=critic_result.verdict,
            retryScope=critic_result.retryScope,
            decisionStatus=decision_result.decisionStatus,
            reviewBlocked=decision_result.reviewBlocked,
            plannerHints=list(critic_result.plannerHints),
            criticIssues=list(critic_result.issues),
        )
        iterations.append(iteration)
        _attach_iterations(decision_result, iterations)

        if critic_result.passed:
            return decision_result

        if critic_result.verdict == "review_blocked" or attempt >= MAX_FORECAST_RETRIES:
            if _should_publish_text_safe_result(decision_result, critic_result):
                return _build_text_safe_result(decision_result, critic_result, iterations)
            return _build_review_blocked_result(decision_result, critic_result, iterations)

        attempt += 1
        if critic_result.retryScope == "reasoning":
            pass_state = _run_revision_pass(
                pass_state=pass_state,
                decision_result=decision_result,
                critic_result=critic_result,
                retry_attempt=attempt,
                today=today,
            )
            stage = "revise_reasoning"
            continue

        planner_hints = _merge_planner_hints(planner_hints, critic_result.plannerHints)
        if pass_state.get("canonical_deterministic_plan"):
            planner_hints = _merge_planner_hints(planner_hints, [_preserve_action_family_hint(pass_state["plan"])])
        pass_state = _run_full_pass(
            task_snapshot=task_snapshot,
            diagnostic_package=diagnostic_package,
            opportunity=normalized_opportunity,
            planner_hints=planner_hints,
            today=today,
        )
        if pass_state is None:
            return _build_review_blocked_result(decision_result, critic_result, iterations)
        stage = "replan"


def _normalize_opportunity(value: GrowthOpportunity | dict[str, Any] | None) -> GrowthOpportunity | None:
    if isinstance(value, GrowthOpportunity):
        return value
    if isinstance(value, dict):
        return GrowthOpportunity.model_validate(value)
    return None


def _with_opportunity(
    diagnostic_package: dict[str, Any] | None,
    opportunity: GrowthOpportunity | None,
) -> dict[str, Any] | None:
    if opportunity is None:
        return diagnostic_package
    result = dict(diagnostic_package or {})
    result["opportunity"] = opportunity.model_dump(mode="json")
    blockers = list(result.get("blocking_gaps") or [])
    for blocker in opportunity.blockers:
        if blocker not in blockers:
            blockers.append(blocker)
    result["blocking_gaps"] = blockers
    return result


def _run_full_pass(
    *,
    task_snapshot: dict[str, Any] | None,
    diagnostic_package: dict[str, Any] | None,
    opportunity: GrowthOpportunity | None,
    planner_hints: list[ForecastPlannerHint],
    today: date | None,
) -> dict[str, Any] | None:
    plan = build_forecast_plan(task_snapshot, planner_hints=planner_hints)
    if plan is None:
        return None

    resolved_targets = resolve_forecast_targets(task_snapshot, plan, planner_hints=planner_hints)
    hypothesis = ForecastHypothesis(
        domain=plan.primary_domain,
        action=plan.action_family,
        targets=[target.id for target in resolved_targets],
        horizon_days=plan.horizon_days,
        period_days=plan.period_days,
        goal=plan.goal,
        effect_metrics=list(plan.effect_metrics),
        source_action_kind=plan.source_action_kind,
    )
    evidence_bundle = build_forecast_evidence_bundle(
        diagnostic_package,
        plan=plan,
        resolved_targets=resolved_targets,
    )
    calculator = select_calculator(plan)
    if calculator is None:
        return None

    scenario_output = calculate_forecast_scenario(
        plan=plan,
        resolved_targets=resolved_targets,
        evidence_bundle=evidence_bundle,
        today=today,
    )
    reasoning_input = build_forecast_reasoning_input(
        hypothesis=hypothesis.action,
        plan=plan,
        resolved_targets=resolved_targets,
        evidence_bundle=evidence_bundle,
        scenario_output=scenario_output,
        opportunity=opportunity,
    )
    reasoning_result, reasoning_debug = run_forecast_reasoning(
        reasoning_input=reasoning_input,
        plan=plan,
        today=today,
    )
    decision_result = build_forecast_decision(
        hypothesis=hypothesis,
        plan=plan,
        resolved_targets=resolved_targets,
        evidence_bundle=evidence_bundle,
        scenario_output=scenario_output,
        reasoning_result=reasoning_result,
        reasoning_debug=reasoning_debug,
    )
    return {
        "hypothesis": hypothesis,
        "plan": plan,
        "resolved_targets": resolved_targets,
        "evidence_bundle": evidence_bundle,
        "scenario_output": scenario_output,
        "reasoning_input": reasoning_input,
        "reasoning_result": reasoning_result,
        "reasoning_debug": reasoning_debug,
        "decision_result": decision_result,
        "canonical_deterministic_plan": is_canonical_deterministic_plan(
            plan,
            decision_result.resolutionPlan,
        ),
    }


def _run_revision_pass(
    *,
    pass_state: dict[str, Any],
    decision_result: ForecastDecisionResult,
    critic_result: ForecastCriticResult,
    retry_attempt: int,
    today: date | None,
) -> dict[str, Any]:
    revision_input = build_forecast_revision_input(
        reasoning_input=pass_state["reasoning_input"],
        previous_decision=decision_result,
        critic_result=critic_result,
        retry_attempt=retry_attempt,
    )
    reasoning_result, reasoning_debug = run_forecast_reasoning(
        reasoning_input=revision_input,
        plan=pass_state["plan"],
        today=today,
    )
    revised_decision = build_forecast_decision(
        hypothesis=pass_state["hypothesis"],
        plan=pass_state["plan"],
        resolved_targets=pass_state["resolved_targets"],
        evidence_bundle=pass_state["evidence_bundle"],
        scenario_output=pass_state["scenario_output"],
        reasoning_result=reasoning_result,
        reasoning_debug=reasoning_debug,
    )
    return {
        **pass_state,
        "reasoning_input": revision_input,
        "reasoning_result": reasoning_result,
        "reasoning_debug": reasoning_debug,
        "decision_result": revised_decision,
    }


def _apply_critic_result(
    decision_result: ForecastDecisionResult,
    critic_result: ForecastCriticResult,
    critic_debug: dict[str, Any],
    *,
    attempt: int,
    stage: str,
    planner_hints: list[ForecastPlannerHint],
) -> ForecastDecisionResult:
    decision_result.debugData["critic"] = {
        "result": critic_result.model_dump(mode="json"),
        "debug": critic_debug,
    }
    decision_result.forecastDetails["critic"] = critic_result.model_dump(mode="json")
    decision_result.criticIssues = list(critic_result.issues)
    decision_result.iterationCount = attempt
    decision_result.reviewBlocked = False
    decision_result.reviewBlockReason = None
    decision_result.orchestrationTrace = {
        "stage": stage,
        "planner_hints": [hint.model_dump(mode="json") for hint in planner_hints],
        "critic_verdict": critic_result.verdict,
        "retry_scope": critic_result.retryScope,
    }
    if decision_result.uiCard is not None:
        decision_result.uiCard.nextChecks = list(
            dict.fromkeys([*decision_result.uiCard.nextChecks, *critic_result.revisionBrief])
        )
    if critic_result.passed:
        return decision_result

    if "critic_rejected_action_plan" not in decision_result.blockedBy:
        decision_result.blockedBy.append("critic_rejected_action_plan")
    if decision_result.uiCard is not None and "critic_rejected_action_plan" not in decision_result.uiCard.blockers:
        decision_result.uiCard.blockers.append("critic_rejected_action_plan")
        if critic_result.criticSummary:
            decision_result.uiCard.confidenceReason = critic_result.criticSummary
    if decision_result.decisionStatus != "blocked":
        decision_result.decisionStatus = "needs_review"
    if decision_result.confidencePercent is not None:
        decision_result.confidencePercent = max(10, decision_result.confidencePercent - 18)
    return decision_result


def _should_publish_text_safe_result(
    decision_result: ForecastDecisionResult,
    critic_result: ForecastCriticResult,
) -> bool:
    if not is_canonical_deterministic_plan(decision_result.plan, decision_result.resolutionPlan):
        return False
    if not bool(decision_result.resolutionPlan):
        return False
    return has_only_textual_issues(
        critic_result.issues,
        plan_executable=critic_result.planExecutable,
    )


def _build_text_safe_result(
    decision_result: ForecastDecisionResult,
    critic_result: ForecastCriticResult,
    iterations: list[ForecastIterationRecord],
) -> ForecastDecisionResult:
    decision_result.reviewBlocked = False
    decision_result.reviewBlockReason = None
    decision_result.summary = build_safe_summary_from_plan(decision_result.resolutionPlan)
    decision_result.finalRecommendation = build_safe_recommendation_from_plan(decision_result.resolutionPlan)
    decision_result.blockedBy = [item for item in decision_result.blockedBy if item != "critic_rejected_action_plan"]
    decision_result.decisionStatus = "needs_review" if decision_result.blockedBy else "completed"
    if decision_result.uiCard is not None:
        decision_result.uiCard.recommendation = decision_result.finalRecommendation
        decision_result.uiCard.confidenceReason = critic_result.criticSummary
        decision_result.uiCard.blockers = [
            item for item in decision_result.uiCard.blockers if item != "critic_rejected_action_plan"
        ]
    decision_result.forecastDetails["reviewBlocked"] = False
    decision_result.forecastDetails["publishedWithTextFallback"] = True
    decision_result.forecastDetails["criticIssues"] = [issue.model_dump(mode="json") for issue in critic_result.issues]
    _attach_iterations(decision_result, iterations)
    return decision_result


def _build_review_blocked_result(
    decision_result: ForecastDecisionResult,
    critic_result: ForecastCriticResult,
    iterations: list[ForecastIterationRecord],
) -> ForecastDecisionResult:
    decision_result.reviewBlocked = True
    decision_result.reviewBlockReason = critic_result.criticSummary
    decision_result.decisionStatus = "blocked"
    decision_result.summary = "Решение заблокировано критиком; автоматический action-plan небезопасен."
    decision_result.finalRecommendation = "Нужен повторный сбор фактов или ручная верификация перед публикацией карточки."
    decision_result.resolutionPlan = []
    decision_result.resolutionActions = []
    if decision_result.uiCard is not None:
        decision_result.uiCard.recommendation = decision_result.finalRecommendation
        decision_result.uiCard.expectedEffect = ""
        decision_result.uiCard.highlightMetrics = []
        decision_result.uiCard.blockers = list(
            dict.fromkeys([*decision_result.uiCard.blockers, "critic_rejected_action_plan"])
        )
        decision_result.uiCard.nextChecks = list(
            dict.fromkeys([*decision_result.uiCard.nextChecks, *critic_result.revisionBrief])
        )
        decision_result.uiCard.confidenceReason = critic_result.criticSummary
    decision_result.forecastDetails["reviewBlocked"] = True
    decision_result.forecastDetails["criticIssues"] = [issue.model_dump(mode="json") for issue in critic_result.issues]
    _attach_iterations(decision_result, iterations)
    return decision_result


def _attach_iterations(
    decision_result: ForecastDecisionResult,
    iterations: list[ForecastIterationRecord],
) -> None:
    trace = {"iterations": [iteration.model_dump(mode="json") for iteration in iterations]}
    decision_result.iterationCount = len(iterations)
    decision_result.orchestrationTrace = trace
    decision_result.debugData["iterations"] = trace["iterations"]


def _merge_planner_hints(
    current: list[ForecastPlannerHint],
    new: list[ForecastPlannerHint],
) -> list[ForecastPlannerHint]:
    merged: list[ForecastPlannerHint] = []
    seen: set[tuple[str, str | None, str | None]] = set()
    for hint in [*current, *new]:
        key = (
            hint.code,
            hint.forcedActionFamily,
            hint.forcedTargetMode,
            hint.preserveActionFamily,
            hint.preferDeterministicPlan,
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(hint)
    return merged


def _preserve_action_family_hint(plan) -> ForecastPlannerHint:
    excluded_families = ["stop_or_reduce_campaigns"] if plan.action_family == "normquery_bid_optimization" else []
    return ForecastPlannerHint(
        code="wrong_action_family",
        message="Сохранить canonical deterministic family для следующего прохода.",
        forcedActionFamily=plan.action_family,
        forcedTargetMode=plan.target_mode,
        preserveActionFamily=True,
        preferDeterministicPlan=True,
        excludedActionFamilies=excluded_families,
    )
