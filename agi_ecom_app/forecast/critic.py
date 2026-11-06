from __future__ import annotations

import json
import re
from datetime import date
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

from agents import Agent, GuardrailFunctionOutput, ModelSettings, OutputGuardrailTripwireTriggered, Runner, output_guardrail

from agi_ecom_app.config import get_settings
from agi_ecom_app.forecast.models import (
    ForecastCriticIssue,
    ForecastCriticResult,
    ForecastDecisionResult,
    ForecastPlannerHint,
)
from agi_ecom_app.forecast.orchestration_policy import (
    has_bound_marketing_steps,
    has_only_textual_issues,
    is_canonical_deterministic_plan,
    is_resolution_plan_executable,
)
from agi_ecom_app.llm_runtime import (
    LLMRuntimeConfigurationError,
    LLMRuntimeSpec,
    build_agent_model,
    get_active_llm_runtime_spec,
)
from agi_ecom_app.llm_output_parsing import extract_model_payload
from agi_ecom_app.main_agent import build_run_hooks
from agi_ecom_app.schemas import EcomRunContext, QueryIntent

PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "forecast_critic_agent.md"

_VAGUE_PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "vague_test_scale",
        re.compile(r"\bтестов\w*\s+расшир\w*\s+реклам", re.IGNORECASE),
        "Фраза `тестово расширять рекламу` не задаёт конкретную операцию, кампанию и пределы действия.",
    ),
    (
        "vague_careful_scale",
        re.compile(r"\b(?:осторожн\w*|аккуратн\w*)\s+расшир\w*", re.IGNORECASE),
        "Фраза про осторожное расширение не описывает конкретный объём, объект и лимиты.",
    ),
    (
        "vague_return_positions",
        re.compile(r"\bвернут?\w*\s+(?:sku\s+)?(?:в\s+top-?\d+|позици\w*)", re.IGNORECASE),
        "Фраза про возврат позиций описывает цель, а не действие.",
    ),
    (
        "vague_strengthen",
        re.compile(r"\bусилить\b", re.IGNORECASE),
        "Глагол `усилить` без конкретного объекта и шага слишком абстрактен для action-plan.",
    ),
    (
        "vague_under_control",
        re.compile(r"\bпод\s+контролем\b", re.IGNORECASE),
        "Фраза `под контролем` не задаёт исполнимого действия.",
    ),
)

_TECHNICAL_ENGLISH = re.compile(r"\b(?:advert|headline|subheadline|title|scale|waste|winner|hold)\b", re.IGNORECASE)
_TECHNICAL_TERMS = re.compile(r"\b(?:supply-risk|auto-apply|mart|repricing|price decision)\b", re.IGNORECASE)
_QUERY_OR_CAMPAIGN_OPERATION = re.compile(r"(запрос|кампан)", re.IGNORECASE)
_SCALE_OPERATION = re.compile(r"(расшир|масштаб)", re.IGNORECASE)
_MUTATING_OPERATIONS = {
    "дозаказать_остаток",
    "запустить_скидку",
    "заменить_первый_экран",
    "изменить_цену",
    "обновить_оффер",
    "отключить_запрос",
    "перепаковать_карточку",
    "переписать_заголовок",
    "поднять_лимит_рекламы",
    "снизить_цену",
    "сократить_кампанию",
}
_REORDER_POSITIVE_OPERATIONS = {"дозаказать_остаток", "оформить_дозаказ"}
_REORDER_HOLD_OPERATIONS = {
    "удержать_остаток_без_дозаказа",
    "отменить_дозаказ",
    "приостановить_дозаказ",
}


def _load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


def _dynamic_instructions(context_wrapper, agent) -> str:
    del context_wrapper, agent
    return _load_prompt()


@output_guardrail
def forecast_critic_output_guardrail(context, agent, output):
    violations: list[str] = []
    if not isinstance(output, ForecastCriticResult):
        violations.append("Final output is not ForecastCriticResult.")
    return GuardrailFunctionOutput(
        output_info={"violations": violations},
        tripwire_triggered=bool(violations),
    )


def build_forecast_critic_agent() -> Agent[EcomRunContext]:
    return _build_forecast_critic_agent(get_active_llm_runtime_spec())


@lru_cache(maxsize=8)
def _build_forecast_critic_agent(runtime_spec: LLMRuntimeSpec) -> Agent[EcomRunContext]:
    return Agent(
        name="Forecast Critic Agent",
        instructions=_dynamic_instructions,
        model=build_agent_model(runtime_spec),
        model_settings=ModelSettings(tool_choice="none"),
        # The critic is advisory and always merged with deterministic rules.
        # Keeping its JSON as plain text avoids Agents SDK turn failures when a
        # provider emits an empty/partial structured payload; validation below
        # remains strict and fail-open to the rule result.
        output_type=None,
        output_guardrails=[],
    )


build_forecast_critic_agent.cache_clear = _build_forecast_critic_agent.cache_clear  # type: ignore[attr-defined]


def run_forecast_critic(
    *,
    decision_result: ForecastDecisionResult,
    today: date | None = None,
    retry_attempt: int = 0,
    max_retries: int = 2,
) -> tuple[ForecastCriticResult, dict[str, object]]:
    rule_result = _rule_based_critic(
        decision_result,
        retry_attempt=retry_attempt,
        max_retries=max_retries,
    )
    settings = get_settings()
    runtime_spec = get_active_llm_runtime_spec(require_configured=False, settings=settings)
    if runtime_spec.missing_env_vars:
        if runtime_spec.provider != "openai":
            missing = ", ".join(runtime_spec.missing_env_vars)
            raise LLMRuntimeConfigurationError(
                f"Model profile `{runtime_spec.profile}` requires missing environment variable(s): {missing}."
            )
        return rule_result, {
            "mode": "rules_only",
            "reason": "missing_llm_api_key",
            "provider": runtime_spec.provider,
            "profile": runtime_spec.profile,
            "missing_env_vars": list(runtime_spec.missing_env_vars),
        }

    context = EcomRunContext(
        user_query="Проверь, достаточно ли конкретен action-plan для UI.",
        intent=QueryIntent.SALES,
        today=today or date.today(),
        db_target="analytics",
        chat_id=f"forecast-critic-{uuid4()}",
        evidence_budget=0,
    )
    hooks = build_run_hooks()
    payload = json.dumps(
        build_forecast_critic_payload(decision_result),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    try:
        raw_output = Runner.run_sync(
            build_forecast_critic_agent(),
            payload,
            context=context,
            hooks=hooks,
            max_turns=3,
        ).final_output
        llm_payload = extract_model_payload(raw_output, ForecastCriticResult)
        if not llm_payload:
            raise ValueError("Forecast critic returned an empty JSON object.")
        llm_result = ForecastCriticResult.model_validate(llm_payload)
        merged = _merge_critic_results(rule_result, llm_result)
        return merged, {
            "mode": "agent+rules",
            "tool_trace": list(context.tool_trace),
            "agent_trace": list(context.agent_trace),
            "decision_trace": list(context.decision_trace),
        }
    except (OutputGuardrailTripwireTriggered, Exception) as exc:
        if runtime_spec.provider != "openai":
            raise
        return rule_result, {
            "mode": "rules_only",
            "reason": type(exc).__name__,
            "error": str(exc),
            "tool_trace": list(context.tool_trace),
            "agent_trace": list(context.agent_trace),
            "decision_trace": list(context.decision_trace),
        }


def build_forecast_critic_payload(
    decision_result: ForecastDecisionResult,
) -> dict[str, object]:
    """Return only fields needed to review the user-facing action contract."""

    payload = decision_result.model_dump(mode="json", exclude_none=True)
    fields = (
        "hypothesis",
        "domain",
        "baseline",
        "actionScenario",
        "delta",
        "decision",
        "decisionStatus",
        "confidence",
        "confidencePercent",
        "blockedBy",
        "finalRecommendation",
        "summary",
        "resolutionActions",
        "resolutionPlan",
        "plan",
        "resolvedTargets",
        "baselineNarrative",
        "actionNarrative",
        "effectNarrative",
        "decisionWhy",
        "reviewBlocked",
        "reviewBlockReason",
        "uiCard",
    )
    return {key: payload[key] for key in fields if key in payload}


def _rule_based_critic(
    decision_result: ForecastDecisionResult,
    *,
    retry_attempt: int,
    max_retries: int,
) -> ForecastCriticResult:
    issues: list[ForecastCriticIssue] = []
    plan_executable = is_resolution_plan_executable(decision_result.resolutionPlan)
    canonical_deterministic_plan = is_canonical_deterministic_plan(
        decision_result.plan,
        decision_result.resolutionPlan,
    )
    if not decision_result.resolutionPlan:
        issues.append(
            ForecastCriticIssue(
                path="resolutionPlan",
                code="missing_plan",
                severity="high",
                message="Финальный ответ не содержит стандартизированный список действий resolutionPlan.",
            )
        )
    issues.extend(_text_issues("summary", decision_result.summary))
    issues.extend(_text_issues("finalRecommendation", decision_result.finalRecommendation))
    for index, step in enumerate(decision_result.resolutionPlan):
        path = f"resolutionPlan[{index}]"
        instruction = step.instruction or ""
        issues.extend(_text_issues(f"{path}.instruction", instruction))
        for code, pattern, message in _VAGUE_PATTERNS:
            if pattern.search(instruction):
                issues.append(
                    ForecastCriticIssue(
                        path=f"{path}.instruction",
                        code=code,
                        severity="high",
                        message=message,
                    )
                )
        if _QUERY_OR_CAMPAIGN_OPERATION.search(instruction):
            if not (step.advertId or step.normQuery or step.targetEntityId):
                issues.append(
                    ForecastCriticIssue(
                        path=path,
                        code="missing_target_binding",
                        severity="high",
                        message="Шаг по кампании или запросу не привязан к конкретному объекту.",
                    )
                )
        if _SCALE_OPERATION.search(instruction):
            if not step.stopCondition:
                issues.append(
                    ForecastCriticIssue(
                        path=path,
                        code="missing_stop_condition",
                        severity="high",
                        message="Шаг по масштабированию должен содержать stop-condition.",
                    )
                )
            if not step.targetMetric:
                issues.append(
                    ForecastCriticIssue(
                        path=path,
                        code="missing_target_metric",
                        severity="medium",
                        message="Шаг по масштабированию должен содержать измеримый targetMetric.",
                    )
                )
        if step.operation in _MUTATING_OPERATIONS:
            issues.extend(_action_contract_issues(step, path=path))
    issues.extend(_contradiction_issues(decision_result))
    issues.extend(_target_coverage_issues(decision_result))
    issues.extend(_freshness_issues(decision_result))
    issues.extend(
        _data_gap_issues(
            decision_result,
            plan_executable=plan_executable,
            canonical_deterministic_plan=canonical_deterministic_plan,
        )
    )

    rewrite_brief = _rewrite_brief(issues)
    planner_hints = _planner_hints(decision_result, issues, plan_executable=plan_executable)
    blocking_issues = sorted({issue.code for issue in issues if issue.severity == "high"})
    verdict, retry_scope, ui_safe = _critic_policy(
        issues,
        plan_executable=plan_executable,
        canonical_deterministic_plan=canonical_deterministic_plan,
        retry_attempt=retry_attempt,
        max_retries=max_retries,
    )
    return ForecastCriticResult(
        passed=verdict == "passed",
        verdict=verdict,
        retryScope=retry_scope,
        issues=issues,
        blockingIssues=blocking_issues,
        plannerHints=planner_hints,
        criticSummary=_critic_summary(issues, verdict=verdict),
        rewriteBrief=rewrite_brief,
        revisionBrief=rewrite_brief,
        planExecutable=plan_executable,
        uiSafe=ui_safe,
    )


def _merge_critic_results(
    rule_result: ForecastCriticResult,
    llm_result: ForecastCriticResult,
) -> ForecastCriticResult:
    issues = [*rule_result.issues]
    existing = {(issue.path, issue.code) for issue in issues}
    for issue in llm_result.issues:
        key = (issue.path, issue.code)
        if key not in existing:
            issues.append(issue)
            existing.add(key)
    rewrite_brief = list(dict.fromkeys([*rule_result.rewriteBrief, *llm_result.rewriteBrief]))
    revision_brief = list(dict.fromkeys([*rule_result.revisionBrief, *llm_result.revisionBrief]))
    planner_hints = _merge_planner_hints(rule_result.plannerHints, llm_result.plannerHints)
    summaries = [summary for summary in (rule_result.criticSummary, llm_result.criticSummary) if summary]
    blocking_issues = sorted({*rule_result.blockingIssues, *llm_result.blockingIssues})
    verdict, retry_scope = _merge_verdicts(
        (rule_result.verdict, rule_result.retryScope),
        (llm_result.verdict, llm_result.retryScope),
    )
    return ForecastCriticResult(
        passed=verdict == "passed",
        verdict=verdict,
        retryScope=retry_scope,
        issues=issues,
        blockingIssues=blocking_issues,
        plannerHints=planner_hints,
        criticSummary=" ".join(summaries).strip(),
        rewriteBrief=rewrite_brief,
        revisionBrief=revision_brief,
        planExecutable=rule_result.planExecutable or llm_result.planExecutable,
        uiSafe=rule_result.uiSafe and llm_result.uiSafe,
    )


def _critic_summary(issues: list[ForecastCriticIssue], *, verdict: str) -> str:
    if not issues:
        return "Критик не нашёл расплывчатых или недоопределённых действий."
    if verdict == "review_blocked":
        return "Критик не смог довести ответ до UI-safe состояния; карточка должна уйти в review-blocked."
    high = [issue for issue in issues if issue.severity == "high"]
    if high:
        return "Критик нашёл недоопределённые действия; такой план нельзя отдавать в UI без доработки."
    return "Критик нашёл стилистические или структурные проблемы, но критичных блокеров нет."


def _text_issues(path: str, text: str | None) -> list[ForecastCriticIssue]:
    if not text:
        return []
    issues: list[ForecastCriticIssue] = []
    if _TECHNICAL_ENGLISH.search(text) or _TECHNICAL_TERMS.search(text):
        issues.append(
            ForecastCriticIssue(
                path=path,
                code="technical_user_facing_text",
                severity="medium",
                message="В пользовательском тексте остались технические или англоязычные термины.",
            )
        )
    for code, pattern, message in _VAGUE_PATTERNS:
        if pattern.search(text):
            issues.append(
                ForecastCriticIssue(
                    path=path,
                    code=code,
                    severity="high",
                    message=message,
                )
            )
    return issues


def _contradiction_issues(decision_result: ForecastDecisionResult) -> list[ForecastCriticIssue]:
    issues: list[ForecastCriticIssue] = []
    mutating_steps = [step for step in decision_result.resolutionPlan if step.operation in _MUTATING_OPERATIONS]
    if decision_result.decisionStatus == "completed" and decision_result.blockedBy:
        issues.append(
            ForecastCriticIssue(
                path="decisionStatus",
                code="completed_with_blockers",
                severity="high",
                message="Решение не может иметь статус completed при наличии блокеров.",
            )
        )
    price_mutations = [
        step for step in mutating_steps if step.operation in {"запустить_скидку", "изменить_цену", "снизить_цену"}
    ]
    blocker_text = " ".join([*decision_result.blockedBy, *_raw_scenario_blockers(decision_result)]).lower()
    unsafe_price_markers = (
        "thin_price",
        "stale_price",
        "future_price",
        "invalid_price",
        "low_price_decision",
        "missing_finance",
        "missing_cogs",
        "тонк",
        "неполн",
        "устарев",
        "будущ",
        "нет finance",
        "себестоим",
    )
    if price_mutations and any(marker in blocker_text for marker in unsafe_price_markers):
        issues.append(
            ForecastCriticIssue(
                path="resolutionPlan",
                code="unsafe_price_action_with_blocker",
                severity="high",
                message="Изменение цены или скидка запрещены при неполных, ненадёжных или устаревших ценовых данных.",
            )
        )
    final_text = (decision_result.finalRecommendation or "").lower()
    summary_text = (decision_result.summary or "").lower()
    texts = [final_text, summary_text]
    has_hold_language = any(
        phrase in text
        for text in texts
        for phrase in ("не применять автоматически", "не менять цену", "после снятия блокеров", "после снятия блокера")
    )
    if has_hold_language and any(step.operation == "изменить_цену" for step in decision_result.resolutionPlan):
        issues.append(
            ForecastCriticIssue(
                path="resolutionPlan",
                code="contradicting_price_execution",
                severity="high",
                message="Верхний текст запрещает немедленное изменение цены, но resolutionPlan всё равно требует `изменить_цену`.",
            )
        )
    operations_by_target: dict[str, set[str]] = {}
    for step in decision_result.resolutionPlan:
        target = step.targetEntityId or step.advertId or step.normQuery
        if target:
            operations_by_target.setdefault(str(target), set()).add(step.operation)
    for target, operations in operations_by_target.items():
        if operations & _REORDER_POSITIVE_OPERATIONS and operations & _REORDER_HOLD_OPERATIONS:
            issues.append(
                ForecastCriticIssue(
                    path="resolutionPlan",
                    code="contradicting_reorder_actions",
                    severity="high",
                    message=f"По target {target} одновременно разрешён и запрещён дозаказ.",
                )
            )
    return issues


def _raw_scenario_blockers(decision_result: ForecastDecisionResult) -> list[str]:
    scenario = decision_result.debugData.get("scenario_outputs") if isinstance(decision_result.debugData, dict) else None
    if not isinstance(scenario, dict):
        return []
    return [str(item) for item in scenario.get("blocked_by") or [] if item]


def _action_contract_issues(step, *, path: str) -> list[ForecastCriticIssue]:
    issues: list[ForecastCriticIssue] = []
    required = {
        "target_binding": step.targetEntityId or step.advertId or step.normQuery,
        "current_metric": step.currentMetric,
        "target_metric": step.targetMetric,
        "stop_condition": step.stopCondition,
        "payload": step.payload,
    }
    for field, value in required.items():
        if value:
            continue
        issues.append(
            ForecastCriticIssue(
                path=f"{path}.{field}",
                code=f"incomplete_action_contract:{field}",
                severity="high",
                message=(
                    f"Исполняемое действие `{step.operation}` не содержит обязательное поле {field}; "
                    "его нельзя передавать на подтверждение или выполнение."
                ),
            )
        )
    return issues


def _target_coverage_issues(decision_result: ForecastDecisionResult) -> list[ForecastCriticIssue]:
    if decision_result.decisionStatus != "completed" or not decision_result.resolvedTargets:
        return []
    per_target = decision_result.forecastDetails.get("perTargetEvidence")
    if not isinstance(per_target, list):
        return []
    covered = {
        str(item.get("target"))
        for item in per_target
        if isinstance(item, dict) and item.get("target") is not None
    }
    requested = {target.id for target in decision_result.resolvedTargets}
    missing = sorted(requested - covered)
    if not missing:
        return []
    return [
        ForecastCriticIssue(
            path="forecastDetails.perTargetEvidence",
            code="incomplete_target_coverage",
            severity="high",
            message=f"Нет evidence по target: {', '.join(missing)}.",
        )
    ]


def _freshness_issues(decision_result: ForecastDecisionResult) -> list[ForecastCriticIssue]:
    blocker_values = [*decision_result.blockedBy, *_raw_scenario_blockers(decision_result)]
    invalid = [
        blocker
        for blocker in blocker_values
        if any(token in blocker.lower() for token in ("stale", "future", "invalid_freshness", "устарев", "будущ"))
    ]
    mutating = [
        step.operation
        for step in decision_result.resolutionPlan
        if step.operation in _MUTATING_OPERATIONS
    ]
    if not invalid or not mutating:
        return []
    return [
        ForecastCriticIssue(
            path="resolutionPlan",
            code="stale_or_future_evidence_for_action",
            severity="high",
            message=(
                "Исполняемое действие построено на stale/future evidence; разрешён только needs-data/hold результат. "
                f"Блокеры: {', '.join(invalid)}."
            ),
        )
    ]


def _data_gap_issues(
    decision_result: ForecastDecisionResult,
    *,
    plan_executable: bool,
    canonical_deterministic_plan: bool,
) -> list[ForecastCriticIssue]:
    issues: list[ForecastCriticIssue] = []
    ignore_campaign_gap = (
        canonical_deterministic_plan
        and has_bound_marketing_steps(decision_result.resolutionPlan)
    )
    for blocker in decision_result.blockedBy:
        normalized = blocker.strip()
        lower = normalized.lower()
        if lower.startswith("marketing_campaign_not_found:") or lower.startswith("marketing campaign not found:"):
            if ignore_campaign_gap:
                continue
            issues.append(
                ForecastCriticIssue(
                    path="blockedBy",
                    code="campaign_fact_missing",
                    severity="high",
                    message=f"Campaign-level факт не найден для target: {normalized.split(':', 1)[1]}.",
                )
            )
        elif lower.startswith("marketing_query_not_found:") or lower.startswith("marketing query not found:"):
            issues.append(
                ForecastCriticIssue(
                    path="blockedBy",
                    code="query_fact_missing",
                    severity="high",
                    message=f"Query-level факт не найден для target: {normalized.split(':', 1)[1]}.",
                )
            )
    if (
        any(issue.code == "campaign_fact_missing" for issue in issues)
        and decision_result.plan is not None
        and decision_result.plan.target_mode == "campaign"
        and any(target.entity_type == "campaign" and target.id.isdigit() and len(target.id) <= 6 for target in decision_result.resolvedTargets)
    ):
        issues.append(
            ForecastCriticIssue(
                path="resolvedTargets",
                code="wrong_target_mode",
                severity="high",
                message="Короткий numeric target выглядит как SKU, а не как campaign target.",
            )
        )
        issues.append(
            ForecastCriticIssue(
                path="plan.action_family",
                code="wrong_action_family",
                severity="high",
                message="Scenario routed into campaign stop/reduce, хотя target выглядит как SKU review task.",
            )
        )
    return issues


def _planner_hints(
    decision_result: ForecastDecisionResult,
    issues: list[ForecastCriticIssue],
    *,
    plan_executable: bool,
) -> list[ForecastPlannerHint]:
    hints: list[ForecastPlannerHint] = []
    codes = {issue.code for issue in issues}
    if "missing_target_binding" in codes:
        hints.append(
            ForecastPlannerHint(
                code="target_binding_missing",
                message="Нужен шаг, привязанный к advertId, normQuery или targetEntityId.",
            )
        )
    if "missing_plan" in codes or ("technical_user_facing_text" in codes and not plan_executable):
        hints.append(
            ForecastPlannerHint(
                code="resolution_plan_not_executable",
                message="Нужно пересобрать UI-safe и исполнимый resolutionPlan.",
            )
        )
    if "campaign_fact_missing" in codes:
        hint = ForecastPlannerHint(
            code="campaign_fact_missing",
            message="Campaign-level факт отсутствует; перепроверь target binding и route.",
        )
        if any(target.entity_type == "campaign" and target.id.isdigit() and len(target.id) <= 6 for target in decision_result.resolvedTargets):
            hint.forcedTargetMode = "sku"
            hint.forcedEntityType = "sku"
            hint.excludedActionFamilies = ["stop_or_reduce_campaigns"]
        hints.append(hint)
    if "query_fact_missing" in codes:
        hints.append(
            ForecastPlannerHint(
                code="query_fact_missing",
                message="Нужен query-level route или явная привязка к query facts.",
                forcedTargetMode="query",
                forcedEntityType="query",
            )
        )
    if "wrong_target_mode" in codes:
        hints.append(
            ForecastPlannerHint(
                code="wrong_target_mode",
                message="Смени target mode на SKU-oriented route.",
                forcedTargetMode="sku",
                forcedEntityType="sku",
                excludedActionFamilies=["stop_or_reduce_campaigns"],
            )
        )
    if "wrong_action_family" in codes:
        hints.append(
            ForecastPlannerHint(
                code="wrong_action_family",
                message="Не использовать прежний action family для следующего прогона.",
                excludedActionFamilies=[decision_result.plan.action_family] if decision_result.plan is not None else [],
            )
        )
    if "technical_user_facing_text" in codes:
        hints.append(
            ForecastPlannerHint(
                code="technical_user_facing_text",
                message="Убрать technical/english terms из user-facing слоя.",
                preferDeterministicPlan=plan_executable,
            )
        )
    return hints


def _critic_policy(
    issues: list[ForecastCriticIssue],
    *,
    plan_executable: bool,
    canonical_deterministic_plan: bool,
    retry_attempt: int,
    max_retries: int,
) -> tuple[str, str, bool]:
    if not issues:
        return "passed", "none", True
    if canonical_deterministic_plan and has_only_textual_issues(issues, plan_executable=plan_executable):
        if retry_attempt >= max_retries:
            return "revise_reasoning", "none", False
        return "revise_reasoning", "reasoning", False
    high_codes = {issue.code for issue in issues if issue.severity == "high"}
    if not high_codes:
        return "passed", "none", True
    if retry_attempt >= max_retries:
        return "review_blocked", "none", False
    replan_codes = {
        "missing_target_binding",
        "campaign_fact_missing",
        "query_fact_missing",
        "wrong_target_mode",
        "wrong_action_family",
    }
    if high_codes & replan_codes:
        return "replan", "full_replan", False
    return "revise_reasoning", "reasoning", False


def _merge_planner_hints(
    left: list[ForecastPlannerHint],
    right: list[ForecastPlannerHint],
) -> list[ForecastPlannerHint]:
    merged: list[ForecastPlannerHint] = []
    seen: set[tuple[str, str | None, str | None]] = set()
    for hint in [*left, *right]:
        key = (hint.code, hint.forcedActionFamily, hint.forcedTargetMode)
        if key in seen:
            continue
        seen.add(key)
        merged.append(hint)
    return merged


def _merge_verdicts(
    left: tuple[str, str],
    right: tuple[str, str],
) -> tuple[str, str]:
    severity = {
        "passed": 0,
        "revise_reasoning": 1,
        "replan": 2,
        "review_blocked": 3,
    }
    return left if severity[left[0]] >= severity[right[0]] else right


def _rewrite_brief(issues: list[ForecastCriticIssue]) -> list[str]:
    instructions: list[str] = []
    codes = {issue.code for issue in issues}
    if "missing_plan" in codes:
        instructions.append("Собери resolutionPlan как список стандартизированных шагов, а не narrative-рассуждение.")
    if any(code in codes for code in {"vague_test_scale", "vague_careful_scale"}):
        instructions.append("Замени фразы про тестовое или осторожное расширение на конкретную кампанию, лимит и stop-condition.")
    if "vague_return_positions" in codes:
        instructions.append("Заменяй цель `вернуть позиции` на конкретные операции по рабочему ядру запросов.")
    if "english_user_facing_text" in codes:
        instructions.append("Убери английские термины из user-facing текста шага.")
    if "technical_user_facing_text" in codes:
        instructions.append("Убери технические и англоязычные термины из summary, finalRecommendation и шагов.")
    if "missing_target_binding" in codes:
        instructions.append("Для шагов по запросам и кампаниям указывай advertId, normQuery или targetEntityId.")
    if "missing_stop_condition" in codes:
        instructions.append("Для шагов по масштабированию указывай stopCondition.")
    if "missing_target_metric" in codes:
        instructions.append("Для шагов по масштабированию указывай измеримый targetMetric.")
    if "contradicting_price_execution" in codes:
        instructions.append("Не совмещай запрет на немедленное изменение цены с шагом `изменить_цену`; используй шаг подготовки теста или удержания.")
    return instructions
