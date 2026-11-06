from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, timedelta
from math import ceil
from typing import Any, Callable

from agi_ecom_app.schemas import (
    ActionExecutionCheck,
    ActionExecutionForecast,
    ActionExecutionResult,
)
from agi_ecom_app.services import strategic_db
from agi_ecom_app.services.db import get_finance_sku_report

ACTION_EXECUTION_DATA_VERSION = "v1"
PRESERVE_AVAILABILITY_AND_SALES_HANDLER = "preserve_availability_and_sales"
_NUMERIC_TARGET_PATTERN = re.compile(r"\b\d{4,}\b")
_SUPPLIER_INBOUND_KEYS = (
    "supplier_inbound_qty",
    "supplier_inbound_stock",
    "confirmed_supplier_inbound_qty",
)
_OPERATIONAL_MAX_LAG_DAYS = 3


@dataclass(frozen=True)
class ActionHandlerOutcome:
    handler: str
    data_version: str
    draft_result: ActionExecutionResult
    evidence_package: dict[str, Any]


ActionExecutionHandler = Callable[[dict[str, Any], date], ActionHandlerOutcome]


def execute_action_handler(
    task_snapshot: dict[str, Any] | None,
    *,
    today: date,
) -> ActionHandlerOutcome | None:
    if not isinstance(task_snapshot, dict):
        return None
    action_kind = _string(task_snapshot.get("actionKind"))
    handler = _ACTION_EXECUTION_HANDLERS.get(action_kind or "")
    if handler is None:
        return None
    return handler(task_snapshot, today)


def build_action_result_formatter_input(
    task_snapshot: dict[str, Any] | None,
    outcome: ActionHandlerOutcome,
) -> str:
    task_payload = dict(task_snapshot or {})
    return (
        "Сформируй финальный ActionExecutionResult по deterministic evidence package ниже.\n\n"
        "Правила:\n"
        "1. Не добавляй новые задачи, новый pipeline или раздел `Что делать`.\n"
        "2. Не меняй числа, decisionStatus, confidence, blockedBy, forecast, статусы checks и evidence.\n"
        "3. Можно улучшить только формулировки businessMeaning, decision, finalRecommendation, summary "
        "и текст summary у checks.\n"
        "4. Верни строго ActionExecutionResult.\n\n"
        f"Task snapshot:\n{json.dumps(task_payload, ensure_ascii=False, indent=2, default=str)}\n\n"
        "Evidence package:\n"
        f"{json.dumps(outcome.evidence_package, ensure_ascii=False, indent=2, default=str)}\n\n"
        "Draft result:\n"
        f"{outcome.draft_result.model_dump_json(indent=2)}"
    )


def merge_formatted_action_result(
    draft: ActionExecutionResult,
    formatted: ActionExecutionResult | None,
) -> ActionExecutionResult:
    if formatted is None:
        return draft

    merged_checks: list[ActionExecutionCheck] = []
    for index, draft_check in enumerate(draft.checks):
        if index < len(formatted.checks):
            formatted_check = formatted.checks[index]
            merged_checks.append(
                draft_check.model_copy(
                    update={
                        "summary": _non_empty(formatted_check.summary, draft_check.summary),
                    }
                )
            )
        else:
            merged_checks.append(draft_check)

    return draft.model_copy(
        update={
            "businessMeaning": _non_empty(formatted.businessMeaning, draft.businessMeaning),
            "checks": merged_checks,
            "decision": _non_empty(formatted.decision, draft.decision),
            "finalRecommendation": _non_empty(
                formatted.finalRecommendation,
                draft.finalRecommendation,
            ),
            "summary": _non_empty(formatted.summary, draft.summary),
        }
    )


def resolve_numeric_targets(task_snapshot: dict[str, Any] | None) -> list[str]:
    if not isinstance(task_snapshot, dict):
        return []

    targets: list[str] = []
    for raw_target in task_snapshot.get("targets") or []:
        candidate = _string(raw_target)
        if candidate and candidate.isdigit() and candidate not in targets:
            targets.append(candidate)
    if targets:
        return targets

    for raw_object in task_snapshot.get("objects") or []:
        if not isinstance(raw_object, dict):
            continue
        payload = raw_object.get("payload") if isinstance(raw_object.get("payload"), dict) else {}
        for raw_candidate in (
            raw_object.get("externalId"),
            payload.get("nm_id"),
            raw_object.get("objectId"),
            raw_object.get("label"),
        ):
            candidate = _string(raw_candidate)
            if not candidate:
                continue
            numeric_candidate = _extract_numeric_identifier(candidate)
            if numeric_candidate and numeric_candidate not in targets:
                targets.append(numeric_candidate)
                break
    if targets:
        return targets

    for value in _task_text_fragments(task_snapshot):
        for match in _NUMERIC_TARGET_PATTERN.findall(value):
            if match not in targets:
                targets.append(match)
    return targets


def _extract_numeric_identifier(value: str) -> str | None:
    stripped = value.strip()
    if stripped.isdigit():
        return stripped
    match = _NUMERIC_TARGET_PATTERN.search(stripped)
    if match:
        return match.group(0)
    return None


def _preserve_availability_and_sales_handler(
    task_snapshot: dict[str, Any],
    today: date,
) -> ActionHandlerOutcome:
    targets = resolve_numeric_targets(task_snapshot)
    if not targets:
        return _resolution_failure_outcome(
            task_snapshot=task_snapshot,
            blocker="target_resolution_failed",
            summary="Не удалось однозначно определить SKU для выполнения.",
            detail="В карточке нет одного numeric SKU target, поэтому terminal execution нельзя посчитать надежно.",
        )
    if len(targets) > 1:
        return _resolution_failure_outcome(
            task_snapshot=task_snapshot,
            blocker="multiple_targets_not_supported",
            summary=f"Найдено несколько SKU: {', '.join(targets[:4])}.",
            detail="V1 deep logic поддерживает только один SKU target на карточку.",
        )

    sku = targets[0]
    finance_end = today.isoformat()
    finance_start = (today - timedelta(days=6)).isoformat()

    supply_report = strategic_db.get_supply_overview(product_keyword=sku, limit=1)
    sales_report = strategic_db.get_sales_sku_revenue_change(
        product_keyword=sku,
        period_days=7,
        limit=20,
    )
    finance_report = get_finance_sku_report(
        date_from=finance_start,
        date_to=finance_end,
        product_keyword=sku,
        limit=1,
    )
    marketing_report = strategic_db.get_marketing_sku_period_comparison_report(
        product_keyword=sku,
        period_days=7,
        limit=20,
    )
    marketing_actions = strategic_db.get_marketing_action_matrix(
        product_keyword=sku,
        days=7,
        limit=5,
    )

    supply_row = _find_matching_row(_rows(supply_report), sku)
    sales_row = _find_matching_row(_rows(sales_report), sku)
    finance_row = _find_matching_row(_rows(finance_report), sku)
    marketing_row = _find_matching_row(_rows(marketing_report), sku)

    supply_as_of = _report_as_of(supply_report, supply_row, "report_date", "snapshot_date")
    sales_as_of = _report_as_of(
        sales_report,
        sales_row,
        "current_period_end",
        "date_to",
        "report_date",
    )
    finance_as_of = _report_as_of(finance_report, finance_row, "date_to", "report_date")
    source_freshness = {
        "supply": _freshness_status(supply_as_of, today=today),
        "sales": _freshness_status(sales_as_of, today=today),
        "finance": _freshness_status(finance_as_of, today=today),
    }

    # `stock_qty` is the sellable on-hand balance. `stock_qty_full` also contains
    # unavailable units and must never be used for availability decisions.
    available_stock = _number(_value(supply_row, "stock_qty"))
    # WB's `in_way_to_client` / `in_way_from_client` are customer fulfilment and
    # returns, not an incoming supplier shipment. Only an explicitly named,
    # supplier-level quantity may reduce the reorder preview.
    inbound_stock = _supplier_inbound_stock(supply_row)
    supplier_inbound_confirmed = inbound_stock is not None
    supply_sales_units = _number(_value(supply_row, "sales_7d_units"))
    current_sales_units = _number(_value(sales_row, "current_sales", "current_sales_units"))
    sales_units_recent = supply_sales_units if supply_sales_units and supply_sales_units > 0 else current_sales_units
    sales_velocity = (
        round(sales_units_recent / 7, 2)
        if sales_units_recent is not None and sales_units_recent > 0
        else None
    )
    coverage_days = (
        round(available_stock / sales_velocity, 1)
        if available_stock is not None and sales_velocity is not None and sales_velocity > 0
        else None
    )
    safety_stock_units = ceil(sales_velocity * 7) if sales_velocity is not None and sales_velocity > 0 else None
    conservative_reorder_qty = (
        max(
            0,
            ceil(
                sales_velocity * 14
                + safety_stock_units
                - available_stock
                - (inbound_stock if inbound_stock is not None else 0)
            ),
        )
        if sales_velocity is not None
        and sales_velocity > 0
        and safety_stock_units is not None
        and available_stock is not None
        else None
    )
    reorder_flag = _string(_value(supply_row, "reorder_flag"))
    stockout_risk = classify_stockout_risk(
        coverage_days=coverage_days,
        reorder_flag=reorder_flag,
        sales_velocity=sales_velocity,
    )

    margin_rub = _number(_value(finance_row, "margin_rub", "Маржа, руб"))
    profitability_pct = _number(
        _value(finance_row, "profitability_pct", "Рентабельность, %")
    )
    unit_cost_rub = _number(_value(finance_row, "unit_cost_rub", "Себестоимость"))

    finance_margin_confirmed = unit_cost_rub is not None and profitability_pct is not None
    has_supply_signal = supply_row is not None and available_stock is not None
    has_sales_signal = sales_velocity is not None and sales_velocity > 0

    marketing_rows = _rows(marketing_report)
    marketing_action_rows = _marketing_action_rows(marketing_actions)
    marketing_actions_fresh = _marketing_actions_are_fresh(marketing_actions)
    has_reduce_signal = any(
        row.get("status") in {"остановить", "сократить"} for row in marketing_action_rows
    ) and marketing_actions_fresh
    has_scale_signal = any(
        row.get("status") == "масштабировать" for row in marketing_action_rows
    ) and marketing_actions_fresh
    marketing_guardrail = classify_marketing_guardrail(
        profitability_pct=profitability_pct,
        coverage_days=coverage_days,
        stockout_risk=stockout_risk,
        has_reduce_signal=has_reduce_signal,
        has_scale_signal=has_scale_signal,
    )

    assumptions = [
        "velocity baseline = sales_7d_units / 7",
        "safety stock = 7 дней спроса",
        "reorder horizon = 14 дней спроса + safety stock",
        "available stock = sellable stock_qty; stock_qty_full is audit-only",
        "customer transit in_way_to_client/in_way_from_client is excluded from supplier inbound",
    ]
    if supplier_inbound_confirmed:
        assumptions.append("supplier inbound is taken only from an explicit supplier_inbound_qty field")
    else:
        assumptions.append("supplier inbound is unknown; reorder_qty is a conservative preview with inbound=0")
    if marketing_rows and marketing_rows[0].get("attribution_quality") == "estimated_allocated_from_campaign":
        assumptions.append("маркетинговые SKU-метрики распределены из campaign-day статистики и могут быть оценочными")

    blocked_by: list[str] = []
    decision_status = "completed"
    confidence = "high"
    if not has_supply_signal:
        decision_status = "blocked"
        confidence = "low"
        blocked_by.append("missing_supply_snapshot")
    if not has_sales_signal:
        decision_status = "blocked"
        confidence = "low"
        blocked_by.append("missing_sales_velocity")
    for source_name in ("supply", "sales"):
        freshness = source_freshness[source_name]
        if freshness["status"] != "fresh" and (
            (source_name == "supply" and has_supply_signal)
            or (source_name == "sales" and has_sales_signal)
        ):
            decision_status = "blocked"
            confidence = "low"
            blocked_by.append(f"{freshness['status']}_{source_name}_snapshot")
    if finance_margin_confirmed and source_freshness["finance"]["status"] != "fresh":
        decision_status = "blocked"
        confidence = "low"
        blocked_by.append(f"{source_freshness['finance']['status']}_finance_snapshot")
    if marketing_action_rows and not marketing_actions_fresh:
        blocked_by.append("invalid_marketing_action_freshness")
    if decision_status != "blocked" and not finance_margin_confirmed:
        decision_status = "needs_review"
        confidence = "medium"
        blocked_by.append("missing_confirmed_margin")

    reorder_decision = classify_reorder_decision(
        coverage_days=coverage_days,
        reorder_qty=conservative_reorder_qty,
    )
    if (
        decision_status != "blocked"
        and reorder_decision == "reorder_now"
        and not supplier_inbound_confirmed
    ):
        decision_status = "needs_review"
        confidence = "medium"
        blocked_by.append("missing_supplier_inbound_snapshot")
    business_meaning = (
        f"SKU {sku} нужно удержать в наличии, чтобы не потерять продажи и вклад в прибыль. "
        "Terminal execution проверяет запас, темп спроса, покрытие, необходимость дозаказа и "
        "можно ли безопасно поддерживать маркетинг."
    )
    decision = _compose_decision_text(
        sku=sku,
        decision_status=decision_status,
        reorder_decision=reorder_decision,
        reorder_qty=conservative_reorder_qty,
        stockout_risk=stockout_risk,
        marketing_guardrail=marketing_guardrail,
        profitability_pct=profitability_pct,
        blocked_by=blocked_by,
    )
    final_recommendation = _compose_final_recommendation(
        sku=sku,
        decision_status=decision_status,
        reorder_decision=reorder_decision,
        reorder_qty=conservative_reorder_qty,
        marketing_guardrail=marketing_guardrail,
        blocked_by=blocked_by,
    )
    summary = _compose_summary(
        decision_status=decision_status,
        reorder_decision=reorder_decision,
        reorder_qty=conservative_reorder_qty,
        marketing_guardrail=marketing_guardrail,
    )

    checks = [
        ActionExecutionCheck(
            name="остаток",
            status=_supply_check_status(has_supply_signal, stockout_risk),
            summary=_supply_check_summary(
                available_stock=available_stock,
                inbound_stock=inbound_stock,
                coverage_days=coverage_days,
                stockout_risk=stockout_risk,
                reorder_flag=reorder_flag,
            ),
            evidence=_compact_evidence(
                [
                    f"stock_qty={available_stock}" if available_stock is not None else None,
                    f"supplier_inbound_qty={inbound_stock}" if inbound_stock is not None else None,
                    f"customer_in_way_to_client={_number(_value(supply_row, 'in_way_to_client'))}"
                    if _number(_value(supply_row, "in_way_to_client")) is not None
                    else None,
                    f"customer_in_way_from_client={_number(_value(supply_row, 'in_way_from_client'))}"
                    if _number(_value(supply_row, "in_way_from_client")) is not None
                    else None,
                    f"reorder_flag={reorder_flag}" if reorder_flag else None,
                ]
            ),
        ),
        ActionExecutionCheck(
            name="спрос",
            status="ok" if has_sales_signal else "blocked",
            summary=_sales_check_summary(
                sales_units_recent=sales_units_recent,
                sales_velocity=sales_velocity,
                sales_row=sales_row,
            ),
            evidence=_compact_evidence(
                [
                    f"sales_7d_units={supply_sales_units}" if supply_sales_units is not None else None,
                    f"current_sales_units={current_sales_units}" if current_sales_units is not None else None,
                    _delta_evidence(
                        sales_row,
                        current_key="current_sales",
                        previous_key="previous_sales",
                        delta_key="delta_sales",
                    ),
                    _delta_evidence(
                        sales_row,
                        current_key="current_revenue",
                        previous_key="previous_revenue",
                        delta_key="delta_revenue",
                    ),
                ]
            ),
        ),
        ActionExecutionCheck(
            name="маржа",
            status="ok" if finance_margin_confirmed else "warning",
            summary=_finance_check_summary(
                unit_cost_rub=unit_cost_rub,
                margin_rub=margin_rub,
                profitability_pct=profitability_pct,
            ),
            evidence=_compact_evidence(
                [
                    f"unit_cost_rub={unit_cost_rub}" if unit_cost_rub is not None else None,
                    f"margin_rub={margin_rub}" if margin_rub is not None else None,
                    f"profitability_pct={profitability_pct}" if profitability_pct is not None else None,
                ]
            ),
        ),
        ActionExecutionCheck(
            name="маркетинговый guardrail",
            status=_marketing_check_status(marketing_guardrail),
            summary=_marketing_check_summary(
                marketing_guardrail=marketing_guardrail,
                marketing_row=marketing_row,
                has_reduce_signal=has_reduce_signal,
                has_scale_signal=has_scale_signal,
            ),
            evidence=_compact_evidence(
                [
                    f"current_spend_rub={_number(_value(marketing_row, 'current_spend_rub'))}"
                    if marketing_row
                    else None,
                    f"current_ad_orders={_number(_value(marketing_row, 'current_ad_orders'))}"
                    if marketing_row
                    else None,
                    f"current_estimated_drr_pct={_number(_value(marketing_row, 'current_estimated_drr_pct'))}"
                    if marketing_row
                    else None,
                    f"action_signal={'reduce' if has_reduce_signal else 'scale' if has_scale_signal else 'neutral'}",
                ]
            ),
        ),
        ActionExecutionCheck(
            name="дозаказ",
            status=_reorder_check_status(reorder_decision, decision_status),
            summary=_reorder_check_summary(
                reorder_decision=reorder_decision,
                reorder_qty=conservative_reorder_qty,
                coverage_days=coverage_days,
                inbound_stock=inbound_stock,
            ),
            evidence=_compact_evidence(
                [
                    f"conservative_reorder_qty={conservative_reorder_qty}"
                    if conservative_reorder_qty is not None
                    else None,
                    f"coverage_days={coverage_days}" if coverage_days is not None else None,
                    f"inbound_stock={inbound_stock}" if inbound_stock is not None else None,
                ]
            ),
        ),
    ]

    draft_result = ActionExecutionResult(
        businessMeaning=business_meaning,
        checks=checks,
        forecast=ActionExecutionForecast(
            salesVelocity=sales_velocity,
            activeDays=7,
            availableStock=available_stock,
            inboundStock=inbound_stock,
            coverageDays=coverage_days,
            leadTimeDays=None,
            safetyStockUnits=float(safety_stock_units) if safety_stock_units is not None else None,
            reorderQty=(
                float(conservative_reorder_qty)
                if conservative_reorder_qty is not None
                else None
            ),
            stockoutRisk=stockout_risk,
            assumptions=assumptions,
        ),
        decision=decision,
        decisionStatus=decision_status,
        confidence=confidence,
        blockedBy=blocked_by,
        finalRecommendation=final_recommendation,
        summary=summary,
    )
    evidence_package = {
        "handler": PRESERVE_AVAILABILITY_AND_SALES_HANDLER,
        "data_version": ACTION_EXECUTION_DATA_VERSION,
        "action_kind": _string(task_snapshot.get("actionKind")),
        "target": {"sku": sku, "resolution_status": "resolved"},
        "task": {
            "title": _string(task_snapshot.get("title")),
            "action": _string(task_snapshot.get("action")),
            "business_goal": _string(task_snapshot.get("businessGoal")),
            "decision_type": _string(task_snapshot.get("decisionType")),
        },
        "inputs": {
            "supply": supply_row or {},
            "sales": sales_row or {},
            "finance": finance_row or {},
            "marketing": marketing_row or {},
            "marketing_actions": marketing_action_rows[:5],
        },
        "source_freshness": source_freshness,
        "metrics": {
            "active_days": 7,
            "sales_velocity": sales_velocity,
            "available_stock": available_stock,
            "inbound_stock": inbound_stock,
            "coverage_days": coverage_days,
            "safety_stock_units": safety_stock_units,
            "reorder_qty": conservative_reorder_qty,
            "reorder_qty_is_conservative_preview": not supplier_inbound_confirmed,
            "supplier_inbound_confirmed": supplier_inbound_confirmed,
            "stockout_risk": stockout_risk,
            "profitability_pct": profitability_pct,
            "marketing_guardrail": marketing_guardrail,
            "reorder_decision": reorder_decision,
        },
        "decision": {
            "decision_status": decision_status,
            "confidence": confidence,
            "blocked_by": blocked_by,
            "decision": decision,
            "final_recommendation": final_recommendation,
            "summary": summary,
        },
        "assumptions": assumptions,
    }
    return ActionHandlerOutcome(
        handler=PRESERVE_AVAILABILITY_AND_SALES_HANDLER,
        data_version=ACTION_EXECUTION_DATA_VERSION,
        draft_result=draft_result,
        evidence_package=evidence_package,
    )


def classify_stockout_risk(
    *,
    coverage_days: float | None,
    reorder_flag: str | None,
    sales_velocity: float | None,
) -> str:
    normalized_flag = (reorder_flag or "").strip().lower()
    if sales_velocity is None or sales_velocity <= 0:
        return "unknown"
    if normalized_flag == "critical":
        return "high"
    if coverage_days is None:
        return "unknown"
    if coverage_days < 7:
        return "high"
    if normalized_flag == "warning" or coverage_days < 21:
        return "medium"
    return "low"


def classify_marketing_guardrail(
    *,
    profitability_pct: float | None,
    coverage_days: float | None,
    stockout_risk: str,
    has_reduce_signal: bool,
    has_scale_signal: bool,
) -> str:
    if profitability_pct is None or profitability_pct <= 0:
        return "block_scale_missing_or_negative_margin"
    if coverage_days is None:
        return "block_scale_missing_coverage"
    if stockout_risk == "high" or coverage_days < 14:
        return "block_scale_low_coverage"
    if has_reduce_signal:
        return "block_scale_marketing_reduce_signal"
    if coverage_days < 21:
        return "support_without_expansion"
    if has_scale_signal:
        return "allow_scale"
    return "support_without_expansion"


def classify_reorder_decision(
    *,
    coverage_days: float | None,
    reorder_qty: float | None,
) -> str:
    if coverage_days is None:
        return "blocked"
    if coverage_days < 14 and reorder_qty is not None and reorder_qty > 0:
        return "reorder_now"
    if coverage_days < 21:
        return "monitor"
    return "not_required"


def _resolution_failure_outcome(
    *,
    task_snapshot: dict[str, Any],
    blocker: str,
    summary: str,
    detail: str,
) -> ActionHandlerOutcome:
    draft_result = ActionExecutionResult(
        businessMeaning=detail,
        checks=[
            ActionExecutionCheck(
                name="target resolution",
                status="blocked" if blocker == "target_resolution_failed" else "warning",
                summary=summary,
                evidence=resolve_numeric_targets(task_snapshot),
            )
        ],
        forecast=ActionExecutionForecast(stockoutRisk="unknown"),
        decision="Нужна ручная проверка target перед выполнением terminal action.",
        decisionStatus="needs_review",
        confidence="low",
        blockedBy=[blocker],
        finalRecommendation="Оставить один SKU target в карточке и запустить выполнение повторно.",
        summary=summary,
    )
    evidence_package = {
        "handler": PRESERVE_AVAILABILITY_AND_SALES_HANDLER,
        "data_version": ACTION_EXECUTION_DATA_VERSION,
        "action_kind": _string(task_snapshot.get("actionKind")),
        "target": {
            "resolution_status": "failed",
            "candidates": resolve_numeric_targets(task_snapshot),
        },
        "decision": {
            "decision_status": "needs_review",
            "confidence": "low",
            "blocked_by": [blocker],
            "summary": summary,
        },
    }
    return ActionHandlerOutcome(
        handler=PRESERVE_AVAILABILITY_AND_SALES_HANDLER,
        data_version=ACTION_EXECUTION_DATA_VERSION,
        draft_result=draft_result,
        evidence_package=evidence_package,
    )


def _task_text_fragments(task_snapshot: dict[str, Any]) -> list[str]:
    fragments: list[str] = []
    for key in (
        "action",
        "title",
        "factor",
        "observation",
        "impact",
        "cause",
        "agentPrompt",
        "businessGoal",
        "executionPlan",
    ):
        value = _string(task_snapshot.get(key))
        if value:
            fragments.append(value)
    for item in task_snapshot.get("evidence") or []:
        value = _string(item)
        if value:
            fragments.append(value)
    return fragments


def _rows(report: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(report, dict):
        return []
    rows = report.get("rows")
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def _find_matching_row(rows: list[dict[str, Any]], sku: str) -> dict[str, Any] | None:
    if not rows:
        return None
    for row in rows:
        if _matches_sku(row, sku):
            return row
    return rows[0]


def _matches_sku(row: dict[str, Any], sku: str) -> bool:
    for key in ("nm_id", "vendor_code", "Артикул"):
        value = _string(row.get(key))
        if value == sku:
            return True
    return False


def _value(row: dict[str, Any] | None, *keys: str) -> Any:
    if not isinstance(row, dict):
        return None
    for key in keys:
        if key in row:
            return row.get(key)
    return None


def _string(value: Any) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().replace(" ", "").replace(",", ".")
        if not cleaned:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _supplier_inbound_stock(row: dict[str, Any] | None) -> float | None:
    if not isinstance(row, dict):
        return None
    for key in _SUPPLIER_INBOUND_KEYS:
        if key not in row:
            continue
        value = _number(row.get(key))
        if value is not None and value >= 0:
            return value
    return None


def _as_date(value: Any) -> date | None:
    if isinstance(value, date):
        return value
    if value is None:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def _report_as_of(
    report: dict[str, Any] | None,
    row: dict[str, Any] | None,
    *keys: str,
) -> date | None:
    for payload in (row, report):
        if not isinstance(payload, dict):
            continue
        for key in keys:
            parsed = _as_date(payload.get(key))
            if parsed is not None:
                return parsed
    return None


def _freshness_status(source_date: date | None, *, today: date) -> dict[str, Any]:
    if source_date is None:
        return {"status": "missing_date", "as_of": None, "lag_days": None}
    lag_days = (today - source_date).days
    if lag_days < 0:
        status = "future"
    elif lag_days > _OPERATIONAL_MAX_LAG_DAYS:
        status = "stale"
    else:
        status = "fresh"
    return {"status": status, "as_of": source_date.isoformat(), "lag_days": lag_days}


def _marketing_actions_are_fresh(report: dict[str, Any] | None) -> bool:
    if not isinstance(report, dict):
        return False
    freshness = report.get("source_freshness")
    if not isinstance(freshness, dict):
        return False
    sources = freshness.get("sources")
    if not isinstance(sources, dict) or not sources:
        return False
    legacy_campaign = freshness.get("legacy_campaign")
    legacy_query = freshness.get("legacy_query")
    requirements: list[bool] = []
    if report.get("campaign_actions"):
        requirements.append(
            (isinstance(sources.get("campaign_mart"), dict) and sources["campaign_mart"].get("status") == "fresh")
            or (isinstance(legacy_campaign, dict) and legacy_campaign.get("status") == "fresh")
        )
    if report.get("query_actions"):
        requirements.append(
            (isinstance(sources.get("query_bands_mart"), dict) and sources["query_bands_mart"].get("status") == "fresh")
            or (isinstance(legacy_query, dict) and legacy_query.get("status") == "fresh")
        )
    return bool(requirements) and all(requirements)


def _sum_numbers(*values: float | None) -> float | None:
    numeric_values = [value for value in values if value is not None]
    if not numeric_values and any(value is None for value in values):
        return None
    return round(sum(numeric_values), 2)


def _compact_evidence(items: list[str | None]) -> list[str]:
    result: list[str] = []
    for item in items:
        value = _string(item)
        if value:
            result.append(value)
    return result


def _delta_evidence(
    row: dict[str, Any] | None,
    *,
    current_key: str,
    previous_key: str,
    delta_key: str,
) -> str | None:
    if row is None:
        return None
    current = _number(_value(row, current_key))
    previous = _number(_value(row, previous_key))
    delta = _number(_value(row, delta_key))
    if current is None and previous is None and delta is None:
        return None
    return f"{current_key}={current}, {previous_key}={previous}, {delta_key}={delta}"


def _supply_check_status(has_supply_signal: bool, stockout_risk: str) -> str:
    if not has_supply_signal:
        return "blocked"
    if stockout_risk == "high":
        return "warning"
    return "ok"


def _supply_check_summary(
    *,
    available_stock: float | None,
    inbound_stock: float | None,
    coverage_days: float | None,
    stockout_risk: str,
    reorder_flag: str | None,
) -> str:
    if available_stock is None:
        return "Нет supply snapshot с остатком, поэтому покрытие не подтверждено."
    parts = [f"Доступный остаток {available_stock:g} шт."]
    if inbound_stock is not None:
        parts.append(f"в пути {inbound_stock:g} шт.")
    if coverage_days is not None:
        parts.append(f"покрытие {coverage_days:g} дн.")
    parts.append(f"риск {stockout_risk}.")
    if reorder_flag:
        parts.append(f"Supply flag: {reorder_flag}.")
    return " ".join(parts)


def _sales_check_summary(
    *,
    sales_units_recent: float | None,
    sales_velocity: float | None,
    sales_row: dict[str, Any] | None,
) -> str:
    if sales_velocity is None:
        return "Скорость продаж не подтверждена, coverageDays рассчитать нельзя."
    current_revenue = _number(_value(sales_row, "current_revenue"))
    delta_revenue = _number(_value(sales_row, "delta_revenue"))
    parts = [
        f"За последние 7 дней подтверждено {sales_units_recent:g} продаж, скорость {sales_velocity:g} шт./день."
    ]
    if current_revenue is not None:
        parts.append(f"Текущая выручка {current_revenue:g} руб.")
    if delta_revenue is not None:
        parts.append(f"Delta выручки {delta_revenue:g} руб.")
    return " ".join(parts)


def _finance_check_summary(
    *,
    unit_cost_rub: float | None,
    margin_rub: float | None,
    profitability_pct: float | None,
) -> str:
    if unit_cost_rub is None or profitability_pct is None:
        return "Нет подтвержденной finance margin: себестоимость или profitability_pct отсутствуют."
    parts = [f"Себестоимость {unit_cost_rub:g} руб."]
    if margin_rub is not None:
        parts.append(f"маржа {margin_rub:g} руб.")
    parts.append(f"рентабельность {profitability_pct:g}%.")
    return " ".join(parts)


def _marketing_check_status(marketing_guardrail: str) -> str:
    if marketing_guardrail == "allow_scale":
        return "ok"
    if marketing_guardrail == "support_without_expansion":
        return "warning"
    return "blocked"


def _marketing_check_summary(
    *,
    marketing_guardrail: str,
    marketing_row: dict[str, Any] | None,
    has_reduce_signal: bool,
    has_scale_signal: bool,
) -> str:
    spend = _number(_value(marketing_row, "current_spend_rub"))
    orders = _number(_value(marketing_row, "current_ad_orders"))
    drr = _number(_value(marketing_row, "current_estimated_drr_pct"))
    parts: list[str] = []
    if spend is not None:
        parts.append(f"Spend {spend:g} руб.")
    if orders is not None:
        parts.append(f"ad orders {orders:g}.")
    if drr is not None:
        parts.append(f"estimated DRR {drr:g}%.")
    parts.append(
        {
            "allow_scale": "Масштабирование рекламы допустимо.",
            "support_without_expansion": "Рекламу можно только поддерживать без расширения.",
            "block_scale_marketing_reduce_signal": "Маркетинговые сигналы рекомендуют сокращение или остановку.",
            "block_scale_low_coverage": "Масштабирование запрещено из-за низкого покрытия или высокого stockout risk.",
            "block_scale_missing_or_negative_margin": "Масштабирование запрещено, пока маржа не подтверждена или <= 0.",
            "block_scale_missing_coverage": "Масштабирование запрещено, пока не подтверждено покрытие.",
        }.get(marketing_guardrail, "Рекламный режим требует ручной проверки.")
    )
    if has_reduce_signal:
        parts.append("Есть reduce/stop signal из action matrix.")
    elif has_scale_signal:
        parts.append("Есть scale signal из action matrix.")
    return " ".join(parts)


def _reorder_check_status(reorder_decision: str, decision_status: str) -> str:
    if decision_status == "blocked":
        return "blocked"
    if reorder_decision == "reorder_now":
        return "warning"
    return "ok"


def _reorder_check_summary(
    *,
    reorder_decision: str,
    reorder_qty: float | None,
    coverage_days: float | None,
    inbound_stock: float | None,
) -> str:
    if reorder_decision == "blocked":
        return "Без coverageDays нельзя надежно посчитать дозаказ."
    if reorder_decision == "reorder_now":
        inbound_text = (
            f"подтверждено входящих поставок {inbound_stock:g} шт."
            if inbound_stock is not None
            else "входящие поставки поставщика не подтверждены; объём является консервативным черновиком"
        )
        return (
            f"Нужен дозаказ примерно {reorder_qty:g} шт.; текущее покрытие {coverage_days:g} дн.; "
            f"{inbound_text}."
        )
    if reorder_decision == "monitor":
        return (
            f"Срочный дозаказ не обязателен, но покрытие {coverage_days:g} дн. требует контроля и "
            "проверки входящих поставок."
        )
    return f"Покрытие {coverage_days:g} дн.; дозаказ в ближайшем цикле не требуется."


def _compose_decision_text(
    *,
    sku: str,
    decision_status: str,
    reorder_decision: str,
    reorder_qty: float | None,
    stockout_risk: str,
    marketing_guardrail: str,
    profitability_pct: float | None,
    blocked_by: list[str],
) -> str:
    if decision_status == "blocked":
        return (
            f"Terminal action по SKU {sku} нельзя закрыть автоматически: не хватает supply/sales сигнала "
            f"для покрытия. Блокеры: {', '.join(blocked_by)}."
        )
    if decision_status == "needs_review":
        if "missing_supplier_inbound_snapshot" in blocked_by:
            return (
                f"По SKU {sku} подтверждён риск дефицита, но объём дозаказа нельзя утверждать: "
                "нет SKU-level снимка входящей поставки поставщика. "
                f"Консервативный черновик без учёта поставок: {reorder_qty:g} шт."
                if reorder_qty is not None
                else f"По SKU {sku} сначала нужно подтвердить входящие поставки поставщика."
            )
        return (
            f"SKU {sku} можно оценить по запасу и спросу, но финальное решение требует ручной проверки "
            "маржи перед подтверждением масштаба действий."
        )
    reorder_part = {
        "reorder_now": f"рекомендован дозаказ {reorder_qty:g} шт." if reorder_qty is not None else "нужен дозаказ",
        "monitor": "срочный дозаказ не обязателен, но нужен контроль покрытия",
        "not_required": "дозаказ не требуется",
    }.get(reorder_decision, "решение по дозаказу требует проверки")
    marketing_part = {
        "allow_scale": "рекламу можно расширять",
        "support_without_expansion": "рекламу можно только поддерживать без расширения",
        "block_scale_marketing_reduce_signal": "масштабирование рекламы запрещено по маркетинговым сигналам",
        "block_scale_low_coverage": "масштабирование рекламы запрещено из-за низкого покрытия",
        "block_scale_missing_or_negative_margin": "масштабирование рекламы запрещено до подтверждения положительной маржи",
        "block_scale_missing_coverage": "масштабирование рекламы запрещено до подтверждения покрытия",
    }.get(marketing_guardrail, "маркетинг требует ручной проверки")
    margin_part = (
        f"подтвержденная рентабельность {profitability_pct:g}%."
        if profitability_pct is not None
        else "маржа не подтверждена."
    )
    return (
        f"По SKU {sku}: {reorder_part}; риск out-of-stock {stockout_risk}; "
        f"{marketing_part}; {margin_part}"
    )


def _compose_final_recommendation(
    *,
    sku: str,
    decision_status: str,
    reorder_decision: str,
    reorder_qty: float | None,
    marketing_guardrail: str,
    blocked_by: list[str],
) -> str:
    if decision_status == "blocked":
        return (
            f"Сначала восстановить данные по запасу/скорости продаж для SKU {sku}, затем повторить execution. "
            f"Блокеры: {', '.join(blocked_by)}."
        )
    if decision_status == "needs_review":
        if "missing_supplier_inbound_snapshot" in blocked_by:
            return (
                f"Сначала сверить SKU-level входящие поставки поставщика по SKU {sku}; "
                f"до сверки дозаказ {reorder_qty:g} шт. остаётся только консервативным черновиком."
                if reorder_qty is not None
                else f"Сначала сверить SKU-level входящие поставки поставщика по SKU {sku}."
            )
        reorder_hint = (
            f"Дозаказ {reorder_qty:g} шт. можно подготовить черновиком."
            if reorder_decision == "reorder_now" and reorder_qty is not None
            else "Пока держать задачу в ручной проверке."
        )
        return (
            f"{reorder_hint} Рекламу не расширять, пока finance не подтвердит себестоимость и маржу."
        )
    if reorder_decision == "reorder_now" and reorder_qty is not None:
        base = f"Подтвердить дозаказ {reorder_qty:g} шт. по SKU {sku}."
    elif reorder_decision == "monitor":
        base = f"Оставить SKU {sku} под ежедневным контролем покрытия и входящих поставок."
    else:
        base = f"Держать SKU {sku} в наличии без срочного дозаказа."
    marketing_note = {
        "allow_scale": " Рекламу можно расширять в пределах рабочего бюджета.",
        "support_without_expansion": " Рекламу держать только в режиме поддержки без расширения.",
        "block_scale_marketing_reduce_signal": " Рекламу не расширять; сначала отработать сигнал на сокращение.",
        "block_scale_low_coverage": " Рекламу не расширять до стабилизации покрытия.",
        "block_scale_missing_or_negative_margin": " Рекламу не расширять до подтверждения положительной маржи.",
        "block_scale_missing_coverage": " Рекламу не расширять до подтверждения покрытия.",
    }.get(marketing_guardrail, "")
    return (base + marketing_note).strip()


def _compose_summary(
    *,
    decision_status: str,
    reorder_decision: str,
    reorder_qty: float | None,
    marketing_guardrail: str,
) -> str:
    if decision_status == "blocked":
        return "Нет данных для покрытия; задача требует ручной проверки."
    if decision_status == "needs_review":
        if reorder_decision == "reorder_now" and reorder_qty is not None:
            return f"Черновик дозаказа {reorder_qty:g} шт.; рекламу не расширять до подтверждения маржи."
        return "Запас оценен, но маржа не подтверждена; рекламу не расширять."
    reorder_part = {
        "reorder_now": f"Дозаказ {reorder_qty:g} шт." if reorder_qty is not None else "Нужен дозаказ",
        "monitor": "Контроль покрытия без срочного дозаказа",
        "not_required": "Дозаказ не требуется",
    }.get(reorder_decision, "Решение требует проверки")
    marketing_part = {
        "allow_scale": "рекламу можно расширять",
        "support_without_expansion": "рекламу только поддерживать",
        "block_scale_marketing_reduce_signal": "рекламу не расширять",
        "block_scale_low_coverage": "рекламу не расширять",
        "block_scale_missing_or_negative_margin": "рекламу не расширять",
        "block_scale_missing_coverage": "рекламу не расширять",
    }.get(marketing_guardrail, "маркетинг под контролем")
    return f"{reorder_part}; {marketing_part}."


def _marketing_action_rows(report: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(report, dict):
        return []
    rows: list[dict[str, Any]] = []
    for key in ("campaign_actions", "query_actions", "sku_actions"):
        value = report.get(key)
        if isinstance(value, list):
            rows.extend([row for row in value if isinstance(row, dict)])
    return rows


def _non_empty(value: str | None, fallback: str | None) -> str | None:
    cleaned = _string(value)
    if cleaned:
        return cleaned
    return fallback


_ACTION_EXECUTION_HANDLERS: dict[str, ActionExecutionHandler] = {
    PRESERVE_AVAILABILITY_AND_SALES_HANDLER: _preserve_availability_and_sales_handler,
}
