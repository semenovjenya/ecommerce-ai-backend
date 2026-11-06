from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Iterable, Mapping, Sequence

from pydantic import BaseModel, ValidationError

from agi_ecom_app.manager_decision.models import (
    ActionEconomicsV1,
    DecisionBlockerV1,
    ExactValueV1,
    ManagerActionV1,
    ManagerDecisionV1,
    ManagerDecisionWorkflowResult,
    ManagerEvidenceFactV1,
    RejectedCandidateV1,
    RequestCoverageV1,
)


_FAMILY_ALIASES = {
    "advertising": "advertising",
    "advert": "advertising",
    "ads": "advertising",
    "ad": "advertising",
    "marketing": "advertising",
    "campaign": "advertising",
    "promotion": "advertising",
    "реклама": "advertising",
    "pricing": "pricing",
    "price": "pricing",
    "discount": "pricing",
    "цена": "pricing",
    "supply": "supply",
    "stock": "supply",
    "inventory": "supply",
    "reorder": "supply",
    "остатки": "supply",
    "finance": "finance",
    "financial": "finance",
    "финансы": "finance",
    "sales": "sales",
    "sale": "sales",
    "продажи": "sales",
}

_METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    # Generic ``status`` is deliberately excluded: campaign efficiency marts
    # use values such as ``negative_outlier`` and ``strong_leader`` which are
    # not live WB campaign states.
    "campaign_status": ("campaign_status", "advert_status", "wb_campaign_status", "статус_кампании"),
    "bid": ("bid", "current_bid", "cpm", "ставка"),
    "ad_spend": ("ad_spend", "spend", "advertising_cost", "расходы_на_рекламу"),
    "clicks": ("clicks", "клики"),
    "ad_orders": ("ad_orders", "orders_from_ads", "advert_orders", "заказы_из_рекламы"),
    "ad_units": ("ad_units", "units_from_ads", "advert_units", "единицы_из_рекламы"),
    "revenue": ("revenue", "sales_revenue", "выручка"),
    "contribution_profit": (
        "contribution_profit",
        "contribution_profit_rub",
        "contribution_margin",
        "contribution_margin_rub",
        "маржинальная_прибыль",
    ),
    "budget_remaining": ("budget_remaining", "remaining_budget", "остаток_бюджета"),
    "stock_on_hand": (
        "stock_on_hand",
        "available_stock",
        "available_stock_qty",
        "stock_qty",
        "on_hand",
        "доступный_остаток",
    ),
    "current_price": ("current_price", "price", "sale_price", "текущая_цена"),
    "current_discount": ("current_discount", "discount", "discount_pct", "текущая_скидка"),
    "cogs": ("cogs", "unit_cost", "cost_price", "себестоимость"),
    "commission": ("commission", "marketplace_commission", "комиссия"),
    "logistics": ("logistics", "logistics_cost", "логистика"),
    "storage": ("storage", "storage_cost", "хранение"),
    "price_volume_history": (
        "price_volume_history",
        "price_sales_history",
        "price_elasticity_observations",
        "история_цена_объем",
    ),
    "competitor_price": ("competitor_price", "competitor_median_price", "цена_конкурента"),
    "supplier_inbound": (
        "supplier_inbound",
        "confirmed_supplier_inbound",
        "purchase_order_inbound",
        "поставка_поставщика",
    ),
    "sales_velocity": ("sales_velocity", "daily_sales_rate", "скорость_продаж"),
    "lead_time_days": ("lead_time_days", "supplier_lead_time_days", "срок_поставки"),
    "units": ("units", "sales_units", "orders", "quantity", "штуки"),
    "conversion": ("conversion", "conversion_rate", "конверсия"),
}

_REQUIRED_METRICS: dict[str, tuple[str, ...]] = {
    "advertising": (
        "campaign_status",
        "bid",
        "ad_spend",
        "clicks",
        "ad_orders",
        "revenue",
        "contribution_profit",
        "budget_remaining",
        "stock_on_hand",
    ),
    "pricing": (
        "current_price",
        "current_discount",
        "cogs",
        "commission",
        "price_volume_history",
        "competitor_price",
        "stock_on_hand",
    ),
    "supply": ("stock_on_hand", "supplier_inbound", "sales_velocity", "lead_time_days"),
    "finance": (
        "revenue",
        "commission",
        "logistics",
        "storage",
        "ad_spend",
        "cogs",
        "contribution_profit",
    ),
    "sales": ("units", "revenue", "conversion"),
}

# Every published action is ranked by contribution profit. A COGS fact from the
# finance domain is therefore mandatory even when the operational family is ads,
# pricing, supply or sales.
_GLOBAL_PROFIT_METRICS = ("cogs",)

_FLOW_PERIOD_METRICS = {
    "advertising": {"ad_spend", "clicks", "ad_orders", "revenue", "contribution_profit"},
    "finance": {"revenue", "commission", "logistics", "storage", "ad_spend", "contribution_profit"},
    "sales": {"units", "revenue", "conversion"},
}

_WB_API_OPERATIONS = {
    "advertising": {"set_bid", "pause_campaign", "resume_campaign"},
    "pricing": {"set_price", "set_discount"},
}

_WB_PAYLOAD_KEYS = {
    "set_bid": {"advert_id", "nm_id", "bid", "placement"},
    "pause_campaign": {"advert_id"},
    "resume_campaign": {"advert_id"},
    "set_price": {"nm_id", "price"},
    "set_discount": {"nm_id", "discount"},
}

_WB_CHANGE_PAYLOAD_KEY = {
    "set_bid": "bid",
    "set_price": "price",
    "set_discount": "discount",
}

_WB_OPERATION_UNITS = {
    "set_bid": {"kopecks/1000_impressions", "kopecks"},
    "set_price": {"RUB"},
    "set_discount": {"%", "percent"},
}

_REQUIRED_LIVE_GUARDRAILS = {
    "advertising": {"stock_on_hand", "contribution_profit", "budget_remaining"},
    "pricing": {"stock_on_hand", "cogs", "contribution_profit"},
}

_OPERATION_ALIASES = {
    "change_bid": "set_bid",
    "update_bid": "set_bid",
    "decrease_bid": "set_bid",
    "increase_bid": "set_bid",
    "stop_campaign": "pause_campaign",
    "restart_campaign": "resume_campaign",
    "start_campaign": "resume_campaign",
    "change_price": "set_price",
    "update_price": "set_price",
    "change_discount": "set_discount",
    "update_discount": "set_discount",
}

_EFFECT_METHOD_DENYLIST = (
    "heuristic",
    "fixed_percent",
    "fixed percentage",
    "llm",
    "language model",
    "assumption",
    "предполож",
    "эврист",
)

_URGENCY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}
_CONFIDENCE_ORDER = {"high": 0, "medium": 1, "low": 2}

_OUTPUT_LABELS_RU = {
    "advertising": "реклама",
    "pricing": "цены",
    "stock": "остатки",
    "unit_economics": "юнит-экономика",
    "sales": "продажи",
    "actions": "готовые действия",
}

_OPERATION_LABELS_RU = {
    "set_bid": "изменить ставку рекламы",
    "pause_campaign": "приостановить рекламную кампанию",
    "resume_campaign": "запустить рекламную кампанию",
    "set_price": "изменить цену",
    "set_discount": "изменить скидку",
    "create_replenishment_task": "создать задачу на поставку",
}

_METRIC_LABELS_RU = {
    "campaign_status": "статус кампании",
    "bid": "ставка",
    "ad_spend": "расход на рекламу",
    "clicks": "клики",
    "ad_orders": "заказы из рекламы",
    "revenue": "выручка",
    "contribution_profit": "маржинальная прибыль",
    "budget_remaining": "остаток бюджета",
    "stock_on_hand": "доступный остаток",
    "supplier_inbound": "поставка поставщика",
    "cogs": "себестоимость единицы",
    "daily_ad_spend_distribution": "дневной расход: минимум / медиана / максимум",
    "sales_velocity": "скорость продаж",
    "legacy_reorder_qty_preview": "предварительное количество к заказу",
}

_TARGET_ALIAS_KEYS = (
    "target",
    "target_id",
    "targetId",
    "object",
    "sku",
    "vendor_code",
    "vendorCode",
    "nm_id",
    "nmId",
    "campaign_id",
    "campaignId",
)


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", by_alias=True)
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _first(source: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in source and source[key] is not None:
            return source[key]
    return default


def _list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _text(value: Any) -> str | None:
    if value is None:
        return None
    result = str(value).strip()
    return result or None


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _datetime(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return _utc(value)
    if isinstance(value, date):
        return datetime.combine(value, time.min, tzinfo=timezone.utc)
    text = str(value).strip()
    if not text:
        return None
    normalized = text.replace("Z", "+00:00")
    try:
        return _utc(datetime.fromisoformat(normalized))
    except ValueError:
        match = re.search(r"\d{4}-\d{2}-\d{2}", text)
        if match:
            try:
                return datetime.combine(date.fromisoformat(match.group(0)), time.min, tzinfo=timezone.utc)
            except ValueError:
                return None
    return None


def _period_bounds(source: Mapping[str, Any]) -> tuple[datetime | None, datetime | None]:
    start = _datetime(_first(source, "period_start", "periodStart", "date_from", "dateFrom"))
    end = _datetime(_first(source, "period_end", "periodEnd", "date_to", "dateTo"))
    period = _text(source.get("period"))
    if period and (start is None or end is None):
        dates = re.findall(r"\d{4}-\d{2}-\d{2}", period)
        if dates:
            start = start or _datetime(dates[0])
            end = end or _datetime(dates[-1])
    return start, end


def _canonical_family(value: Any) -> str | None:
    normalized = str(value or "").strip().lower().replace("ё", "е")
    if normalized in _FAMILY_ALIASES:
        return _FAMILY_ALIASES[normalized]
    for token, family in _FAMILY_ALIASES.items():
        if token and token in normalized:
            return family
    return None


def _metric_token(value: Any) -> str:
    return re.sub(r"[^a-zа-я0-9]+", "_", str(value or "").strip().lower().replace("ё", "е")).strip("_")


def _canonical_metric(value: Any) -> str:
    token = _metric_token(value)
    for canonical, aliases in _METRIC_ALIASES.items():
        if token == canonical or token in aliases:
            return canonical
    return token


def _freshness(
    source: Mapping[str, Any],
    *,
    metric: str,
    snapshot_at: datetime | None,
    now: datetime,
) -> str:
    declared = str(_first(source, "freshness_status", "freshnessStatus", default="unknown")).lower()
    if declared in {"future", "stale"}:
        return declared
    if snapshot_at is None:
        return "unknown"
    age_seconds = (_utc(now) - snapshot_at).total_seconds()
    if age_seconds < 0:
        return "future"
    max_days = 7 if _canonical_metric(metric) == "competitor_price" else 3
    computed = "fresh" if age_seconds <= max_days * 86400 else "stale"
    if declared == "unknown" or declared not in {"fresh", "stale", "future"}:
        return computed
    return declared if declared != "fresh" else computed


def _looks_like_evidence(value: Mapping[str, Any]) -> bool:
    return "value" in value and bool(value.get("metric")) and bool(
        _first(value, "evidence_ref", "evidenceRef", "fact_id", "factId")
    )


def _walk_evidence(value: Any, *, depth: int = 0) -> Iterable[dict[str, Any]]:
    if depth > 7:
        return
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", by_alias=True)
    if isinstance(value, Mapping):
        source = dict(value)
        if _looks_like_evidence(source):
            yield source
            return
        for key, child in source.items():
            if key in {"blocking_gaps", "non_blocking_gaps", "resolved_gaps", "data_gaps"}:
                continue
            yield from _walk_evidence(child, depth=depth + 1)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk_evidence(child, depth=depth + 1)


def _normalize_evidence_fact(raw: Mapping[str, Any], *, now: datetime) -> ManagerEvidenceFactV1 | None:
    metric = _text(raw.get("metric"))
    evidence_ref = _text(_first(raw, "evidence_ref", "evidenceRef", "fact_id", "factId"))
    if not metric or not evidence_ref:
        return None
    period_start, period_end = _period_bounds(raw)
    snapshot_at = _datetime(
        _first(
            raw,
            "source_snapshot_at",
            "sourceSnapshotAt",
            "snapshot_at",
            "snapshotAt",
            "snapshot_date",
            "snapshotDate",
            "data_as_of",
            "as_of",
        )
    )
    period = _text(raw.get("period"))
    if snapshot_at is None and period and period.lower().startswith("as_of"):
        snapshot_at = period_end
    retrieved_at = _datetime(_first(raw, "retrieved_at", "retrievedAt", "collected_at", "collectedAt"))
    confidence = str(raw.get("confidence") or "medium").lower()
    if confidence not in {"low", "medium", "high"}:
        confidence = "medium"
    freshness_status = _freshness(raw, metric=metric, snapshot_at=snapshot_at, now=now)
    if period_end is not None and _utc(period_end) > _utc(now):
        freshness_status = "future"
    return ManagerEvidenceFactV1(
        metric=metric,
        value=raw.get("value"),
        unit=_text(raw.get("unit")),
        target=_text(
            _first(
                raw,
                "target",
                "target_id",
                "targetId",
                "object",
                "sku",
                "vendor_code",
                "vendorCode",
                "campaign_id",
                "campaignId",
            )
        ),
        period=period,
        period_start=period_start,
        period_end=period_end,
        source_table=_text(_first(raw, "source_table", "sourceTable", "table", "table_name", "tableName")),
        snapshot_at=snapshot_at,
        retrieved_at=retrieved_at,
        definition=_text(_first(raw, "definition", "metric_definition", "metricDefinition")),
        freshness_status=freshness_status,
        confidence=confidence,
        evidence_ref=evidence_ref,
        source_agent=_text(_first(raw, "source_agent", "sourceAgent", "agent")),
    )


def extract_evidence_snapshot(
    synthesis_package: Mapping[str, Any],
    *,
    now: datetime,
    action_candidates: Sequence[Any] | None = None,
) -> list[ManagerEvidenceFactV1]:
    roots = [
        synthesis_package.get("evidence_snapshot"),
        synthesis_package.get("evidence_facts"),
        synthesis_package.get("confirmed_facts"),
        synthesis_package.get("primary_evidence"),
        synthesis_package.get("supporting_evidence"),
        synthesis_package.get("manager_candidate_evidence"),
        action_candidates,
    ]
    unique: dict[str, ManagerEvidenceFactV1] = {}
    for root in roots:
        for raw in _walk_evidence(root):
            fact = _normalize_evidence_fact(raw, now=now)
            if fact is not None:
                unique[fact.evidence_ref] = fact
    return sorted(unique.values(), key=lambda fact: (fact.evidence_ref, fact.metric))


def _candidate_sequences(package: Mapping[str, Any]) -> Iterable[Any]:
    for key in (
        "manager_action_candidates",
        "managerActionCandidates",
        "action_candidates",
        "actionCandidates",
        "proposed_actions",
        "proposedActions",
        "ranked_actions",
        "rankedActions",
    ):
        value = package.get(key)
        if value is not None:
            yield from _list(value)
    for container_key in ("manager_decision", "managerDecision", "forecast_decision", "forecastDecision"):
        container = _as_dict(package.get(container_key))
        for key in ("action_candidates", "actionCandidates", "ranked_actions", "rankedActions"):
            if container.get(key) is not None:
                yield from _list(container[key])
        if "forecast" in container_key.lower() and container:
            yield from manager_action_candidates_from_forecast(container)


def extract_action_candidates(
    synthesis_package: Mapping[str, Any],
    action_candidates: Sequence[Any] | None = None,
) -> list[Any]:
    values = [*_list(action_candidates), *_candidate_sequences(synthesis_package)]
    result: list[Any] = []
    seen: set[str] = set()
    for value in values:
        payload = _as_dict(value)
        if not payload:
            payload = {"unstructured_action": str(value)}
        fingerprint = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        if fingerprint not in seen:
            seen.add(fingerprint)
            result.append(payload)
    return result


def manager_action_candidates_from_forecast(forecast_result: Any) -> list[dict[str, Any]]:
    """Adapt forecast output without upgrading point estimates or prose to facts.

    The adapter copies an already complete manager-action contract when one is
    present. Legacy resolution-plan steps become deliberately incomplete
    candidates unless they contain exact from/to, an effect interval, evidence,
    rollback, measurement and expiry. In particular, ``economicEffectRub`` is a
    point estimate and is never expanded into a low/base/high interval.
    """

    source = _as_dict(forecast_result)
    if not source:
        return []
    explicit = _first(
        source,
        "manager_action_candidates",
        "managerActionCandidates",
        "action_candidates",
        "actionCandidates",
    )
    if explicit is not None:
        return [_as_dict(item) for item in _list(explicit) if _as_dict(item)]

    plan = _as_dict(source.get("plan"))
    family = _forecast_family(
        _first(plan, "action_family", "actionFamily", default=_first(source, "action_family", "actionFamily", "domain"))
    )
    candidates: list[dict[str, Any]] = []
    steps = _list(_first(source, "resolutionPlan", "resolution_plan"))
    for index, step_value in enumerate(steps):
        step = _as_dict(step_value)
        if not step:
            continue
        payload = _as_dict(step.get("payload"))
        target = _first(
            step,
            "targetEntityId",
            "target_entity_id",
            "advertId",
            "advert_id",
            default=_first(payload, "target", "sku", "nm_id", "nmId", "advert_id", "advertId"),
        )
        operation = _first(step, "operation", default=payload.get("operation"))
        normalized_operation = _OPERATION_ALIASES.get(str(operation or "").lower(), str(operation or "").lower())
        execution_mode = _first(payload, "executionMode", "execution_mode")
        if execution_mode is None:
            execution_mode = (
                "wb_api"
                if family in _WB_API_OPERATIONS and normalized_operation in _WB_API_OPERATIONS[family]
                else "manual_task"
            )
        exact_from = _first(payload, "from", "fromValue", "from_value")
        exact_to = _first(payload, "to", "toValue", "to_value")
        if exact_from is None and isinstance(step.get("currentMetric"), Mapping):
            exact_from = step.get("currentMetric")
        if exact_to is None and isinstance(step.get("targetMetric"), Mapping):
            exact_to = step.get("targetMetric")
        effect = _first(payload, "expectedEffect", "expected_effect")
        if effect is None:
            details = _as_dict(_first(source, "forecastDetails", "forecast_details"))
            effect = _first(details, "expectedEffectRange", "expected_effect_range")
        # Do not fall back to source.economicEffectRub/profitForecastRub here.
        candidates.append(
            {
                "actionId": _first(step, "stepId", "step_id", default=f"forecast-step-{index + 1}"),
                "family": family,
                "target": target,
                "targetType": _first(step, "targetEntityType", "target_entity_type", default="task"),
                "targetRefs": _list(_first(payload, "targetRefs", "target_refs")),
                "operation": normalized_operation,
                "executionMode": execution_mode,
                "executionPayload": _as_dict(_first(payload, "executionPayload", "execution_payload")),
                "from": exact_from,
                "to": exact_to,
                "economics": _first(
                    payload, "economics", "actionEconomics", "action_economics"
                ),
                "expectedEffect": effect,
                "evidence": _list(_first(payload, "evidence", "evidenceFacts", "evidence_facts")),
                "evidenceRefs": _list(_first(payload, "evidenceRefs", "evidence_refs")),
                "requiredMetrics": _list(_first(payload, "requiredMetrics", "required_metrics")),
                "prerequisites": _list(payload.get("prerequisites")),
                "stopCondition": _first(step, "stopCondition", "stop_condition", default=_first(payload, "stopCondition", "stop_condition")),
                "rollback": payload.get("rollback"),
                "measurementSpec": _first(payload, "measurementSpec", "measurement_spec"),
                "guardrails": _list(payload.get("guardrails")),
                "expiresAt": _first(payload, "expiresAt", "expires_at"),
                "urgency": _first(payload, "urgency", default="medium"),
                "confidence": _first(source, "confidence", default="medium"),
            }
        )
    return candidates


def _forecast_family(value: Any) -> str | None:
    normalized = str(value or "").strip().lower()
    direct = _canonical_family(normalized)
    if direct:
        return direct
    if any(token in normalized for token in ("campaign", "advert", "bid", "profitable_sku_ads")):
        return "advertising"
    if any(token in normalized for token in ("reprice", "price", "overstock_liquidation")):
        return "pricing"
    if any(token in normalized for token in ("reorder", "availability")):
        return "supply"
    if "sales" in normalized or "demand" in normalized or "conversion" in normalized:
        return "sales"
    return None


def _normalize_exact(value: Any, *, default_unit: Any = None) -> dict[str, Any] | None:
    source = _as_dict(value)
    if source:
        exact_value = _first(source, "value", "exact_value", "exactValue")
        unit = _first(source, "unit", default=default_unit)
    else:
        exact_value = value
        unit = default_unit
    if exact_value is None or not _text(unit):
        return None
    return {"value": exact_value, "unit": str(unit).strip()}


def _normalize_effect(value: Any) -> dict[str, Any] | None:
    source = _as_dict(value)
    if not source:
        return None
    low = _first(
        source,
        "contribution_profit_low_per_day",
        "contributionProfitLowPerDay",
        "low_per_day",
        "lowPerDay",
        "low",
    )
    base = _first(
        source,
        "contribution_profit_base_per_day",
        "contributionProfitBasePerDay",
        "base_per_day",
        "basePerDay",
        "base",
    )
    high = _first(
        source,
        "contribution_profit_high_per_day",
        "contributionProfitHighPerDay",
        "high_per_day",
        "highPerDay",
        "high",
    )
    return {
        "contribution_profit_low_per_day": low,
        "contribution_profit_base_per_day": base,
        "contribution_profit_high_per_day": high,
        "currency": _first(source, "currency", default="RUB"),
        "horizon_days": _first(source, "horizon_days", "horizonDays"),
        "calculation_method": _first(source, "calculation_method", "calculationMethod"),
        "evidence_refs": _list(_first(source, "evidence_refs", "evidenceRefs")),
        "causal_valid": bool(_first(source, "causal_valid", "causalValid", default=False)),
    }


def _normalize_economic_range(value: Any) -> dict[str, Any] | None:
    source = _as_dict(value)
    if not source:
        return None
    return {
        "low": _first(source, "low"),
        "base": _first(source, "base"),
        "high": _first(source, "high"),
    }


def _normalize_economics(value: Any) -> dict[str, Any] | None:
    source = _as_dict(value)
    if not source:
        return None
    return {
        "deltaRevenuePerDay": _normalize_economic_range(
            _first(source, "delta_revenue_per_day", "deltaRevenuePerDay")
        ),
        "deltaCogsPerDay": _normalize_economic_range(
            _first(source, "delta_cogs_per_day", "deltaCogsPerDay")
        ),
        "deltaCommissionPerDay": _normalize_economic_range(
            _first(source, "delta_commission_per_day", "deltaCommissionPerDay")
        ),
        "deltaLogisticsPerDay": _normalize_economic_range(
            _first(source, "delta_logistics_per_day", "deltaLogisticsPerDay")
        ),
        "deltaStoragePerDay": _normalize_economic_range(
            _first(source, "delta_storage_per_day", "deltaStoragePerDay")
        ),
        "deltaAdvertisingPerDay": _normalize_economic_range(
            _first(source, "delta_advertising_per_day", "deltaAdvertisingPerDay")
        ),
        "actionCostPerDay": _normalize_economic_range(
            _first(source, "action_cost_per_day", "actionCostPerDay")
        ),
        "currency": _first(source, "currency", default="RUB"),
        "horizonDays": _first(source, "horizon_days", "horizonDays"),
        "calculationMethod": _first(source, "calculation_method", "calculationMethod"),
        "evidenceRefs": _list(_first(source, "evidence_refs", "evidenceRefs")),
        "causalValid": bool(_first(source, "causal_valid", "causalValid", default=False)),
    }


def _effect_from_economics(economics: ActionEconomicsV1) -> dict[str, Any]:
    """Calculate the profit interval from source-addressable components only."""

    costs = (
        economics.delta_cogs_per_day,
        economics.delta_commission_per_day,
        economics.delta_logistics_per_day,
        economics.delta_storage_per_day,
        economics.delta_advertising_per_day,
        economics.action_cost_per_day,
    )
    low = economics.delta_revenue_per_day.low - sum(item.high for item in costs)
    base = economics.delta_revenue_per_day.base - sum(item.base for item in costs)
    high = economics.delta_revenue_per_day.high - sum(item.low for item in costs)
    return {
        "contributionProfitLowPerDay": low,
        "contributionProfitBasePerDay": base,
        "contributionProfitHighPerDay": high,
        "currency": economics.currency,
        "horizonDays": economics.horizon_days,
        "calculationMethod": economics.calculation_method,
        "evidenceRefs": economics.evidence_refs,
        "causalValid": economics.causal_valid,
    }


def _normalize_measurement(value: Any) -> dict[str, Any] | None:
    source = _as_dict(value)
    if not source:
        return None
    return {
        "primary_metric": _first(source, "primary_metric", "primaryMetric", default="contribution_profit_rub"),
        "baseline_days": _first(source, "baseline_days", "baselineDays"),
        "evaluation_days": _first(source, "evaluation_days", "evaluationDays"),
        "attribution_method": _first(source, "attribution_method", "attributionMethod"),
        "guardrail_metrics": _list(_first(source, "guardrail_metrics", "guardrailMetrics")),
        "control_target": _first(source, "control_target", "controlTarget"),
    }


def _normalize_rollback(value: Any, *, unit: Any = None) -> dict[str, Any] | None:
    source = _as_dict(value)
    if not source:
        return None
    return {
        "operation": _first(source, "operation", "action"),
        "from": _normalize_exact(_first(source, "from", "from_value", "fromValue"), default_unit=unit),
        "to": _normalize_exact(_first(source, "to", "to_value", "toValue"), default_unit=unit),
        "instruction": _first(source, "instruction", "description"),
    }


def _candidate_identity(raw: Mapping[str, Any], index: int) -> str:
    explicit = _text(_first(raw, "action_id", "actionId", "candidate_id", "candidateId", "stepId"))
    if explicit:
        return explicit
    fingerprint = json.dumps(raw, ensure_ascii=False, sort_keys=True, default=str)
    return f"candidate-{index + 1}-{hashlib.sha256(fingerprint.encode('utf-8')).hexdigest()[:10]}"


def _validation_blockers(error: ValidationError, *, candidate_id: str, target: str | None) -> list[DecisionBlockerV1]:
    blockers: list[DecisionBlockerV1] = []
    for item in error.errors(include_url=False):
        location = ".".join(str(part) for part in item.get("loc") or [])
        code = "incomplete_action_contract"
        if location.startswith("economics"):
            code = "missing_or_invalid_economics"
        elif location.startswith("expectedEffect") or location.startswith("expected_effect"):
            code = "missing_or_invalid_expected_effect"
        elif location in {"from", "to", "from_value", "to_value"} or location.startswith(("from.", "to.")):
            code = "missing_or_invalid_exact_change"
        elif location.startswith("measurement"):
            code = "missing_or_invalid_measurement_spec"
        elif location.startswith("guardrails"):
            code = "missing_or_invalid_guardrails"
        elif location.startswith("rollback"):
            code = "missing_or_invalid_rollback"
        blockers.append(
            DecisionBlockerV1(
                code=code,
                message=f"Некорректное поле {location or 'action'}: {item.get('msg')}",
                action_id=candidate_id,
                target=target,
                severity="needs_data",
            )
        )
    return blockers


def _preview_exact(value: Any, *, default_unit: Any = None) -> ExactValueV1 | None:
    normalized = _normalize_exact(value, default_unit=default_unit)
    if normalized is None:
        return None
    try:
        return ExactValueV1.model_validate(normalized)
    except ValidationError:
        return None


def _rejected_preview(
    raw: Mapping[str, Any],
    *,
    candidate_id: str,
    family: str | None,
    target: str | None,
    reasons: Sequence[DecisionBlockerV1],
) -> RejectedCandidateV1:
    change = _as_dict(_first(raw, "change", "exact_change", "exactChange"))
    unit = _first(raw, "unit", "change_unit", "changeUnit", default=change.get("unit"))
    from_raw = _first(
        raw,
        "from",
        "from_value",
        "fromValue",
        "current_value",
        "currentValue",
        default=change.get("from"),
    )
    to_raw = _first(
        raw,
        "to",
        "to_value",
        "toValue",
        "proposed_value",
        "proposedValue",
        default=change.get("to"),
    )
    operation = str(
        _first(raw, "operation", "action", "action_type", "actionType", default="")
        or ""
    ).strip().lower()
    operation = _OPERATION_ALIASES.get(operation, operation) or None
    execution_mode_value = _first(raw, "execution_mode", "executionMode")
    execution_mode = (
        execution_mode_value
        if execution_mode_value in {"wb_api", "manual_task"}
        else None
    )
    evidence_refs = [
        str(item)
        for item in _list(_first(raw, "evidence_refs", "evidenceRefs"))
        if _text(item)
    ]
    for item in _list(_first(raw, "evidence", "evidence_facts", "evidenceFacts")):
        if isinstance(item, str) and _text(item):
            evidence_refs.append(item)
            continue
        item_dict = _as_dict(item)
        ref = _text(_first(item_dict, "evidence_ref", "evidenceRef", "fact_id", "factId"))
        if ref:
            evidence_refs.append(ref)
    return RejectedCandidateV1(
        candidate_id=candidate_id,
        family=family,
        target=target,
        target_refs=[
            str(item)
            for item in _list(_first(raw, "target_refs", "targetRefs"))
            if _text(item)
        ],
        operation=operation,
        execution_mode=execution_mode,
        from_value=_preview_exact(from_raw, default_unit=unit),
        to_value=_preview_exact(to_raw, default_unit=unit),
        evidence_refs=list(dict.fromkeys(evidence_refs)),
        reasons=list(reasons),
    )


def _declared_candidate_blockers(
    raw: Mapping[str, Any],
    *,
    candidate_id: str,
    target: str | None,
) -> list[DecisionBlockerV1]:
    blockers: list[DecisionBlockerV1] = []
    for index, value in enumerate(
        _list(_first(raw, "candidate_blockers", "candidateBlockers", "generation_blockers", "generationBlockers"))
    ):
        item = _as_dict(value)
        if item:
            severity = str(item.get("severity") or "needs_data").lower()
            if severity not in {"needs_data", "blocked"}:
                severity = "needs_data"
            code = _text(item.get("code")) or f"candidate_data_gap_{index + 1}"
            message = _text(_first(item, "message", "reason")) or "Кандидат требует дополнительных данных."
        else:
            severity = "needs_data"
            code = f"candidate_data_gap_{index + 1}"
            message = _text(value) or "Кандидат требует дополнительных данных."
        blockers.append(
            DecisionBlockerV1(
                code=code,
                message=message,
                action_id=candidate_id,
                target=target,
                severity=severity,
            )
        )
    return _unique_blockers(blockers)


def _build_action(
    raw_value: Any,
    *,
    index: int,
    evidence_by_ref: Mapping[str, ManagerEvidenceFactV1],
) -> tuple[ManagerActionV1 | None, RejectedCandidateV1 | None]:
    raw = _as_dict(raw_value)
    candidate_id = _candidate_identity(raw, index)
    family = _canonical_family(_first(raw, "family", "action_family", "actionFamily", "domain"))
    target = _text(
        _first(raw, "target", "target_id", "targetId", "sku", "vendor_code", "vendorCode", "campaign_id", "campaignId")
    )
    if not raw or raw.get("unstructured_action"):
        reason = DecisionBlockerV1(
            code="unstructured_action_candidate",
            message="Свободный текст не является исполняемым контрактом действия.",
            action_id=candidate_id,
            target=target,
            severity="needs_data",
        )
        return None, _rejected_preview(
            raw,
            candidate_id=candidate_id,
            family=family,
            target=target,
            reasons=[reason],
        )

    declared_blockers = _declared_candidate_blockers(
        raw,
        candidate_id=candidate_id,
        target=target,
    )
    if declared_blockers:
        return None, _rejected_preview(
            raw,
            candidate_id=candidate_id,
            family=family,
            target=target,
            reasons=declared_blockers,
        )

    change = _as_dict(_first(raw, "change", "exact_change", "exactChange"))
    unit = _first(raw, "unit", "change_unit", "changeUnit", default=change.get("unit"))
    from_raw = _first(raw, "from", "from_value", "fromValue", "current_value", "currentValue", default=change.get("from"))
    to_raw = _first(raw, "to", "to_value", "toValue", "proposed_value", "proposedValue", default=change.get("to"))
    economics_payload = _normalize_economics(
        _first(raw, "economics", "action_economics", "actionEconomics")
    )
    economics: ActionEconomicsV1 | None = None
    if economics_payload is not None:
        try:
            economics = ActionEconomicsV1.model_validate(economics_payload)
        except ValidationError:
            pass
    # Ignore candidate-provided expectedEffect. The public interval is always
    # derived by deterministic interval arithmetic from the economics breakdown.
    effect = _effect_from_economics(economics) if economics is not None else None
    effect_refs = list(economics.evidence_refs) if economics is not None else []
    raw_evidence = _list(_first(raw, "evidence", "evidence_facts", "evidenceFacts"))
    evidence_refs = [str(item) for item in _list(_first(raw, "evidence_refs", "evidenceRefs")) if _text(item)]
    facts: dict[str, ManagerEvidenceFactV1] = {}
    for item in raw_evidence:
        if isinstance(item, str):
            evidence_refs.append(item)
            continue
        item_dict = _as_dict(item)
        ref = _text(_first(item_dict, "evidence_ref", "evidenceRef", "fact_id", "factId"))
        if ref and ref in evidence_by_ref:
            facts[ref] = evidence_by_ref[ref]
    for ref in [*evidence_refs, *effect_refs]:
        if str(ref) in evidence_by_ref:
            facts[str(ref)] = evidence_by_ref[str(ref)]
    target_refs = [
        str(item)
        for item in _list(_first(raw, "target_refs", "targetRefs"))
        if _text(item)
    ]
    if not facts and target:
        for ref, fact in evidence_by_ref.items():
            if fact.target is None or any(
                _same_target(fact.target, candidate_target)
                for candidate_target in [target, *target_refs]
            ):
                facts[ref] = fact

    operation = str(_first(raw, "operation", "action", "action_type", "actionType", default="")).strip().lower()
    operation = _OPERATION_ALIASES.get(operation, operation)
    execution_mode = _first(raw, "execution_mode", "executionMode")
    target_type = _first(raw, "target_type", "targetType", "entity_type", "entityType")
    if target_type is None:
        target_type = "campaign" if family == "advertising" else "sku"
    payload = {
        "action_id": candidate_id,
        "family": family,
        "target": target,
        "target_type": target_type,
        "target_refs": target_refs,
        "operation": operation,
        "executionMode": execution_mode,
        "executionPayload": _as_dict(_first(raw, "execution_payload", "executionPayload", "payload")),
        "from": _normalize_exact(from_raw, default_unit=unit),
        "to": _normalize_exact(to_raw, default_unit=unit),
        "economics": economics_payload,
        "expectedEffect": effect,
        "evidence": list(facts.values()),
        "required_metrics": [str(item) for item in _list(_first(raw, "required_metrics", "requiredMetrics"))],
        "prerequisites": [str(item) for item in _list(raw.get("prerequisites")) if _text(item)],
        "stopCondition": _first(raw, "stop_condition", "stopCondition"),
        "rollback": _normalize_rollback(raw.get("rollback"), unit=unit),
        "measurementSpec": _normalize_measurement(_first(raw, "measurement_spec", "measurementSpec")),
        "guardrails": _list(raw.get("guardrails")),
        "expiresAt": _first(raw, "expires_at", "expiresAt", "expiry"),
        "urgency": str(raw.get("urgency") or "medium").lower(),
        "confidence": str(raw.get("confidence") or "medium").lower(),
    }
    try:
        return ManagerActionV1.model_validate(payload), None
    except ValidationError as exc:
        reasons = _validation_blockers(exc, candidate_id=candidate_id, target=target)
        return None, _rejected_preview(
            raw,
            candidate_id=candidate_id,
            family=family,
            target=target,
            reasons=reasons,
        )


def _same_target(left: Any, right: Any) -> bool:
    left_token = re.sub(r"\W+", "", str(left or "").lower())
    right_token = re.sub(r"\W+", "", str(right or "").lower())
    return bool(left_token and right_token and left_token == right_token)


def _target_token(value: Any) -> str:
    return re.sub(r"\W+", "", str(value or "").casefold())


def _target_alias_groups(
    package: Mapping[str, Any],
    actions: Sequence[ManagerActionV1],
) -> list[set[str]]:
    """Return identifiers proven to name the same target in one source row.

    WB facts commonly carry both a seller ``vendor_code`` and the marketplace
    ``nm_id``.  The normalized public fact keeps one display target, so request
    coverage must retain the co-occurring identifiers before normalization.
    """

    raw_groups: list[set[str]] = []
    for raw in _walk_evidence(package):
        group = {
            token
            for key in _TARGET_ALIAS_KEYS
            for value in _list(raw.get(key))
            if (token := _target_token(value))
        }
        if len(group) > 1:
            raw_groups.append(group)
    for action in actions:
        group = {
            token
            for value in _action_target_tokens(action)
            if (token := _target_token(value))
        }
        if len(group) > 1:
            raw_groups.append(group)

    merged: list[set[str]] = []
    for group in raw_groups:
        overlapping = [known for known in merged if known & group]
        if not overlapping:
            merged.append(set(group))
            continue
        combined = set(group)
        for known in overlapping:
            combined.update(known)
            merged.remove(known)
        merged.append(combined)
    return merged


def _same_target_or_alias(left: Any, right: Any, alias_groups: Sequence[set[str]]) -> bool:
    if _same_target(left, right):
        return True
    left_token = _target_token(left)
    right_token = _target_token(right)
    return bool(
        left_token
        and right_token
        and any(left_token in group and right_token in group for group in alias_groups)
    )


def _fact_complete(fact: ManagerEvidenceFactV1) -> bool:
    return bool(
        fact.value is not None
        and _text(fact.unit)
        and _text(fact.period)
        and fact.period_end is not None
        and _text(fact.source_table)
        and fact.snapshot_at is not None
        and fact.retrieved_at is not None
        and _text(fact.definition)
        and fact.freshness_status == "fresh"
        and _text(fact.evidence_ref)
    )


def _complete_zero_fact(facts: Sequence[ManagerEvidenceFactV1]) -> bool:
    for fact in facts:
        value = fact.value
        if (
            _fact_complete(fact)
            and not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
            and float(value) == 0
        ):
            return True
    return False


def _complete_seven_day_spend_fact(
    facts: Sequence[ManagerEvidenceFactV1],
) -> bool:
    for fact in facts:
        if not _fact_complete(fact) or not isinstance(fact.value, Mapping):
            continue
        values = fact.value.get("valuesRub")
        if not isinstance(values, list) or len(values) != 7:
            continue
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in values
        ):
            continue
        if (
            fact.period_start is not None
            and fact.period_end is not None
            and (fact.period_end.date() - fact.period_start.date()).days == 6
        ):
            return True
    return False


def _zero_opportunity_cost_pause_proven(
    action: ManagerActionV1,
    facts_by_metric: Mapping[str, Sequence[ManagerEvidenceFactV1]],
) -> bool:
    """Allow the narrow zero-sales pause model to work without an unused COGS fact."""

    if (
        action.family != "advertising"
        or action.operation != "pause_campaign"
        or action.economics.calculation_method != "zero_stock_daily_spend_distribution_v1"
    ):
        return False
    if not all(
        _complete_zero_fact(facts_by_metric.get(metric, ()))
        for metric in ("ad_orders", "ad_units", "revenue")
    ):
        return False
    if not _complete_seven_day_spend_fact(
        facts_by_metric.get("daily_ad_spend_distribution", ())
    ):
        return False
    zero_ranges = (
        action.economics.delta_revenue_per_day,
        action.economics.delta_cogs_per_day,
        action.economics.delta_commission_per_day,
        action.economics.delta_logistics_per_day,
        action.economics.delta_storage_per_day,
        action.economics.action_cost_per_day,
    )
    return all(
        component.low == component.base == component.high == 0
        for component in zero_ranges
    )


def _blocker(
    code: str,
    message: str,
    *,
    action: ManagerActionV1 | None = None,
    severity: str = "needs_data",
) -> DecisionBlockerV1:
    return DecisionBlockerV1(
        code=code,
        message=message,
        action_id=action.action_id if action else None,
        target=action.target if action else None,
        severity=severity,
    )


def validate_manager_action(
    action: ManagerActionV1,
    *,
    now: datetime,
    allowed_families: set[str] | None = None,
    safety_checks: Sequence[Any] = (),
) -> list[DecisionBlockerV1]:
    """Run deterministic, fail-closed validation for one typed action."""

    blockers: list[DecisionBlockerV1] = []
    if allowed_families and action.family not in allowed_families:
        blockers.append(
            _blocker(
                "action_family_mismatch",
                f"Действие {action.family} не соответствует запрошенному типу решения.",
                action=action,
                severity="blocked",
            )
        )
    if action.execution_mode == "wb_api":
        allowed_operations = _WB_API_OPERATIONS.get(action.family, set())
        if action.operation not in allowed_operations:
            blockers.append(
                _blocker(
                    "invalid_execution_mode",
                    "WB API разрешён только для поддерживаемых действий рекламы и цены.",
                    action=action,
                    severity="blocked",
                )
            )
        if not action.execution_payload:
            blockers.append(
                _blocker(
                    "missing_execution_payload",
                    "Для WB API нужен точный payload внешней операции.",
                    action=action,
                    severity="blocked",
                )
            )
        else:
            required_payload_keys = _WB_PAYLOAD_KEYS.get(action.operation, set())
            missing_payload_keys = sorted(required_payload_keys - set(action.execution_payload))
            if missing_payload_keys:
                blockers.append(
                    _blocker(
                        "incomplete_execution_payload",
                        "В payload WB API отсутствуют поля: " + ", ".join(missing_payload_keys) + ".",
                        action=action,
                        severity="blocked",
                    )
                )
            change_key = _WB_CHANGE_PAYLOAD_KEY.get(action.operation)
            if (
                change_key
                and change_key in action.execution_payload
                and action.execution_payload[change_key] != action.to_value.value
            ):
                blockers.append(
                    _blocker(
                        "execution_payload_change_mismatch",
                        f"Поле {change_key} в payload не совпадает с точным значением to.",
                        action=action,
                        severity="blocked",
                    )
                )
            allowed_units = _WB_OPERATION_UNITS.get(action.operation)
            if allowed_units and action.to_value.unit not in allowed_units:
                blockers.append(
                    _blocker(
                        "invalid_execution_unit",
                        f"Единица {action.to_value.unit} не поддерживается для {action.operation}.",
                        action=action,
                        severity="blocked",
                    )
                )
    elif action.family in {"sales", "finance", "supply"} and action.execution_mode != "manual_task":
        blockers.append(
            _blocker(
                "invalid_execution_mode",
                "Продажи, финансы и поставки исполняются только как ручная задача.",
                action=action,
                severity="blocked",
            )
        )
    if _utc(action.expires_at) <= _utc(now):
        blockers.append(_blocker("expired_action", "Срок действия рекомендации истёк.", action=action))

    effect = action.expected_effect
    computed_effect = _effect_from_economics(action.economics)
    computed_values = (
        computed_effect["contributionProfitLowPerDay"],
        computed_effect["contributionProfitBasePerDay"],
        computed_effect["contributionProfitHighPerDay"],
    )
    effect_values = (
        effect.contribution_profit_low_per_day,
        effect.contribution_profit_base_per_day,
        effect.contribution_profit_high_per_day,
    )
    if any(abs(actual - expected) > 1e-9 for actual, expected in zip(effect_values, computed_values)):
        blockers.append(
            _blocker(
                "effect_economics_mismatch",
                "Диапазон эффекта не совпадает с детерминированным расчётом экономики.",
                action=action,
                severity="blocked",
            )
        )
    if not all(math.isfinite(value) for value in effect_values) or effect.contribution_profit_low_per_day <= 0:
        blockers.append(
            _blocker(
                "non_positive_or_invalid_effect",
                "Нижняя граница добавочной маржинальной прибыли должна быть положительной.",
                action=action,
            )
        )
    method = action.economics.calculation_method.lower().replace("ё", "е")
    if any(token in method for token in _EFFECT_METHOD_DENYLIST):
        blockers.append(
            _blocker(
                "unsupported_effect_method",
                "Числовой эффект основан на эвристике или тексте модели, а не на проверяемом расчёте.",
                action=action,
            )
        )
    action_refs = {fact.evidence_ref for fact in action.evidence}
    missing_effect_refs = sorted(set(effect.evidence_refs) - action_refs)
    if missing_effect_refs:
        blockers.append(
            _blocker(
                "effect_evidence_missing",
                f"Нет фактов расчёта эффекта: {', '.join(missing_effect_refs)}.",
                action=action,
            )
        )

    guardrail_metrics = {_canonical_metric(rule.metric) for rule in action.guardrails}
    required_guardrails = _REQUIRED_LIVE_GUARDRAILS.get(action.family, set())
    missing_guardrails = sorted(required_guardrails - guardrail_metrics)
    if action.execution_mode == "wb_api" and missing_guardrails:
        blockers.append(
            _blocker(
                "missing_live_guardrails",
                "Нет обязательных execution-preflight guardrails: "
                + ", ".join(missing_guardrails)
                + ".",
                action=action,
                severity="blocked",
            )
        )
    for rule in action.guardrails:
        if rule.evidence_ref not in action_refs:
            blockers.append(
                _blocker(
                    "guardrail_evidence_missing",
                    f"Guardrail {rule.metric} не связан с evidence {rule.evidence_ref}.",
                    action=action,
                )
            )

    facts_by_metric: dict[str, list[ManagerEvidenceFactV1]] = defaultdict(list)
    action_targets = [action.target, *action.target_refs]
    for fact in action.evidence:
        if fact.target is None or any(
            _same_target(fact.target, target) for target in action_targets
        ):
            facts_by_metric[_canonical_metric(fact.metric)].append(fact)
    profit_metrics = (
        ()
        if _zero_opportunity_cost_pause_proven(action, facts_by_metric)
        else _GLOBAL_PROFIT_METRICS
    )
    required = {
        *_REQUIRED_METRICS[action.family],
        *profit_metrics,
        *(_canonical_metric(item) for item in action.required_metrics),
    }
    missing_metrics = sorted(metric for metric in required if not facts_by_metric.get(metric))
    if missing_metrics:
        blockers.append(
            _blocker(
                "missing_required_metrics",
                f"Нет обязательных метрик {action.family}: {', '.join(missing_metrics)}.",
                action=action,
            )
        )
    incomplete_metrics = sorted(
        metric
        for metric in required
        if facts_by_metric.get(metric) and not any(_fact_complete(fact) for fact in facts_by_metric[metric])
    )
    if incomplete_metrics:
        blockers.append(
            _blocker(
                "incomplete_or_stale_evidence",
                "Факты не имеют полного источника, периода или свежести: " + ", ".join(incomplete_metrics) + ".",
                action=action,
            )
        )
    relevant_facts = [
        fact
        for metric in required
        for fact in facts_by_metric.get(metric, [])
        if _fact_complete(fact)
    ]
    period_ends = [fact.period_end for fact in relevant_facts if fact.period_end is not None]
    if period_ends and (max(period_ends) - min(period_ends)).total_seconds() > 3 * 86400:
        blockers.append(
            _blocker(
                "incoherent_evidence_periods",
                "Обязательные метрики относятся к несовместимым периодам.",
                action=action,
            )
        )
    flow_facts: list[ManagerEvidenceFactV1] = []
    for metric in _FLOW_PERIOD_METRICS.get(action.family, set()):
        complete = [fact for fact in facts_by_metric.get(metric, []) if _fact_complete(fact)]
        if complete:
            flow_facts.append(complete[0])
    if len(flow_facts) > 1:
        starts = [fact.period_start for fact in flow_facts if fact.period_start is not None]
        ends = [fact.period_end for fact in flow_facts if fact.period_end is not None]
        durations = [
            (fact.period_end - fact.period_start).total_seconds()
            for fact in flow_facts
            if fact.period_start is not None and fact.period_end is not None
        ]
        if (
            len(starts) != len(flow_facts)
            or len(ends) != len(flow_facts)
            or not durations
            or max(ends) - min(ends) > timedelta(days=1)
            or max(durations) - min(durations) > 86400
        ):
            blockers.append(
                _blocker(
                    "incoherent_flow_periods",
                    "Потоковые метрики действия относятся к несовместимым окнам.",
                    action=action,
                )
            )
    rollback = action.rollback
    if not (
        rollback.from_value.unit == action.to_value.unit
        and rollback.to_value.unit == action.from_value.unit
        and rollback.from_value.value == action.to_value.value
        and rollback.to_value.value == action.from_value.value
    ):
        blockers.append(
            _blocker(
                "invalid_rollback",
                "Rollback должен точно отменять изменение from/to.",
                action=action,
                severity="blocked",
            )
        )
    for item in safety_checks:
        check = _as_dict(item)
        check_target = _first(check, "target", "sku", "campaign_id", "campaignId")
        check_family = _canonical_family(_first(check, "family", "action_family", "actionFamily"))
        status = str(check.get("status") or "").lower()
        if check_target and not _same_target(check_target, action.target):
            continue
        if check_family and check_family != action.family:
            continue
        if status in {"unsafe", "blocked", "rejected", "failed"}:
            blockers.append(
                _blocker(
                    "unsafe_action",
                    _text(_first(check, "reason", "message")) or "Safety check запретил действие.",
                    action=action,
                    severity="blocked",
                )
            )
    return _unique_blockers(blockers)


def _unique_blockers(blockers: Iterable[DecisionBlockerV1]) -> list[DecisionBlockerV1]:
    result: list[DecisionBlockerV1] = []
    seen: set[tuple[Any, ...]] = set()
    for blocker in blockers:
        key = (blocker.code, blocker.action_id, blocker.target, blocker.message)
        if key not in seen:
            seen.add(key)
            result.append(blocker)
    return result


def rank_manager_actions(actions: Sequence[ManagerActionV1]) -> list[ManagerActionV1]:
    """Rank by the guaranteed daily contribution, then urgency and confidence."""

    return sorted(
        actions,
        key=lambda action: (
            -action.expected_effect.contribution_profit_low_per_day,
            _URGENCY_ORDER[action.urgency],
            _CONFIDENCE_ORDER[action.confidence],
            action.action_id,
        ),
    )[:7]


def _explicit_targets(query: str, package: Mapping[str, Any]) -> list[str]:
    del query
    query_plan = _first(package, "query_plan", "queryPlan")
    if isinstance(query_plan, Mapping):
        targets: list[str] = []
        for target in _list(query_plan.get("targets")):
            if not isinstance(target, Mapping):
                continue
            value = _text(target.get("value"))
            if value and value not in targets:
                targets.append(value)
        return targets
    configured = [str(item).strip() for item in _list(_first(package, "requested_targets", "requestedTargets")) if _text(item)]
    if configured:
        return list(dict.fromkeys(configured))
    return []


def extract_manager_request_targets(
    query: str,
    package: Mapping[str, Any] | None = None,
) -> list[str]:
    """Return the same explicit target set used by request-coverage checks."""

    return _explicit_targets(query, package or {})


def _requested_outputs(query: str, package: Mapping[str, Any]) -> list[str]:
    del query
    configured = [str(item).strip().lower() for item in _list(_first(package, "requested_outputs", "requestedOutputs")) if _text(item)]
    if configured:
        return list(dict.fromkeys(configured))
    query_plan = _first(package, "query_plan", "queryPlan")
    if not isinstance(query_plan, Mapping):
        return ["actions"]
    analyses = {str(item) for item in _list(query_plan.get("requested_analyses"))}
    outputs: list[str] = []
    capability_groups = {
        "advertising": {"advertising"},
        "pricing": {"pricing", "price_scenarios"},
        "stock": {"inventory_matrix", "inventory_turnover", "current_stock", "inventory_value", "replenishment"},
        "unit_economics": {"contribution_economics"},
        "sales": {"sales_velocity", "conversions"},
    }
    for output, capabilities in capability_groups.items():
        if analyses.intersection(capabilities):
            outputs.append(output)
    if str(query_plan.get("decision_kind") or "") != "none":
        outputs.append("actions")
    return outputs or ["actions"]


def _allowed_families(requested_outputs: Sequence[str], package: Mapping[str, Any]) -> set[str] | None:
    configured = {
        family
        for value in _list(_first(package, "requested_action_families", "requestedActionFamilies"))
        if (family := _canonical_family(value))
    }
    if configured:
        return configured
    operational = {
        {
            "advertising": "advertising",
            "pricing": "pricing",
            "stock": "supply",
            "unit_economics": "finance",
            "sales": "sales",
        }[output]
        for output in requested_outputs
        if output in {"advertising", "pricing", "stock", "unit_economics", "sales"}
    }
    return operational or None


def _action_target_tokens(action: ManagerActionV1) -> list[str]:
    return [action.target, *action.target_refs]


def _coverage(
    *,
    query: str,
    package: Mapping[str, Any],
    actions: Sequence[ManagerActionV1],
    evidence_snapshot: Sequence[ManagerEvidenceFactV1],
) -> RequestCoverageV1:
    explicit_targets = _explicit_targets(query, package)
    alias_groups = _target_alias_groups(package, actions)
    coverage_target_refs = [
        *(
            ref
            for action in actions
            for ref in _action_target_tokens(action)
        ),
        *(fact.target for fact in evidence_snapshot if fact.target),
    ]
    covered_targets = [
        target
        for target in explicit_targets
        if any(
            _same_target_or_alias(target, ref, alias_groups)
            for ref in coverage_target_refs
        )
    ]
    requested_outputs = _requested_outputs(query, package)
    metrics = {
        _canonical_metric(fact.metric)
        for fact in [
            *(fact for action in actions for fact in action.evidence),
            *evidence_snapshot,
        ]
        if _fact_complete(fact)
    }
    families = {action.family for action in actions}
    covered_outputs: list[str] = []
    for output in requested_outputs:
        covered = False
        if output == "actions":
            covered = bool(actions)
        elif output == "advertising":
            covered = "advertising" in families
        elif output == "pricing":
            covered = "pricing" in families
        elif output == "stock":
            covered = "stock_on_hand" in metrics
        elif output == "unit_economics":
            covered = {"cogs", "contribution_profit"}.issubset(metrics)
        elif output == "sales":
            covered = {"units", "revenue"}.issubset(metrics)
        if covered:
            covered_outputs.append(output)
    missing_targets = [target for target in explicit_targets if target not in covered_targets]
    missing_outputs = [output for output in requested_outputs if output not in covered_outputs]
    return RequestCoverageV1(
        explicit_targets=explicit_targets,
        covered_targets=covered_targets,
        missing_targets=missing_targets,
        requested_outputs=requested_outputs,
        covered_outputs=covered_outputs,
        missing_outputs=missing_outputs,
        complete=not missing_targets and not missing_outputs,
    )


def _conflicts(actions: Sequence[ManagerActionV1]) -> list[DecisionBlockerV1]:
    groups: dict[tuple[str, str], list[ManagerActionV1]] = defaultdict(list)
    for action in actions:
        groups[(action.family, re.sub(r"\W+", "", action.target.lower()))].append(action)
    blockers: list[DecisionBlockerV1] = []
    for items in groups.values():
        if len(items) < 2:
            continue
        changes = {
            (
                item.operation,
                json.dumps(item.to_value.model_dump(mode="json"), ensure_ascii=False, sort_keys=True),
            )
            for item in items
        }
        if len(changes) == 1:
            for duplicate in items[1:]:
                blockers.append(
                    _blocker(
                        "duplicate_action_candidate",
                        "Повторный кандидат дублирует то же действие.",
                        action=duplicate,
                    )
                )
            continue
        for item in items:
            blockers.append(
                _blocker(
                    "conflicting_target_actions",
                    "Для одного объекта и семейства предложены несовместимые действия.",
                    action=item,
                    severity="blocked",
                )
            )
    return blockers


def _decision_id(query: str, query_mode: str, package: Mapping[str, Any], candidates: Sequence[Any]) -> str:
    stable = {
        "query": query,
        "query_mode": query_mode,
        "run_id": _first(package, "run_id", "runId"),
        "candidates": candidates,
        "evidence_refs": sorted(
            str(_first(raw, "evidence_ref", "evidenceRef", "fact_id", "factId"))
            for raw in _walk_evidence(package)
        ),
    }
    payload = json.dumps(stable, ensure_ascii=False, sort_keys=True, default=str)
    return "md1-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _action_as_rejected(
    action: ManagerActionV1,
    reasons: Sequence[DecisionBlockerV1],
) -> RejectedCandidateV1:
    action_reasons = [
        DecisionBlockerV1(
            code=reason.code,
            message=reason.message,
            action_id=action.action_id,
            target=action.target,
            severity=reason.severity,
        )
        for reason in reasons
    ]
    if not action_reasons:
        action_reasons = [
            DecisionBlockerV1(
                code="portfolio_hard_gate_failed",
                message="Действие проверено, но весь портфель не прошёл hard gate.",
                action_id=action.action_id,
                target=action.target,
                severity="needs_data",
            )
        ]
    return RejectedCandidateV1(
        candidate_id=action.action_id,
        family=action.family,
        target=action.target,
        target_refs=list(action.target_refs),
        operation=action.operation,
        execution_mode=action.execution_mode,
        from_value=action.from_value,
        to_value=action.to_value,
        evidence_refs=[fact.evidence_ref for fact in action.evidence],
        reasons=_unique_blockers(action_reasons),
    )


def apply_manager_hard_gate(
    *,
    decision_id: str,
    query_mode: str,
    now: datetime,
    actions: Sequence[ManagerActionV1],
    rejected_candidates: Sequence[RejectedCandidateV1],
    evidence_snapshot: Sequence[ManagerEvidenceFactV1],
    coverage: RequestCoverageV1,
    package_blockers: Sequence[Any] = (),
    conflict_blockers: Sequence[DecisionBlockerV1] = (),
    candidate_blockers: Sequence[DecisionBlockerV1] = (),
) -> ManagerDecisionV1:
    """Create a public decision after portfolio and per-action hard gates pass."""

    blockers: list[DecisionBlockerV1] = []
    if query_mode != "decision":
        blockers.append(
            _blocker(
                "non_decision_mode",
                "Исполняемые рекомендации формируются только в query_mode=decision.",
                severity="blocked",
            )
        )
    for gap_value in package_blockers:
        gap = _as_dict(gap_value)
        if str(gap.get("status") or "").lower() in {"resolved", "closed"}:
            continue
        gap_severity = str(gap.get("severity") or "needs_data").lower()
        blockers.append(
            _blocker(
                "open_blocking_gap",
                _text(_first(gap, "reason", "impact", "metric")) or "Не закрыт обязательный пробел данных.",
                severity="blocked" if gap_severity == "blocked" else "needs_data",
            )
        )
    future_facts = [fact for fact in evidence_snapshot if fact.freshness_status == "future"]
    if future_facts:
        blockers.append(
            _blocker(
                "future_evidence",
                "Источник содержит дату из будущего: " + ", ".join(fact.evidence_ref for fact in future_facts[:5]) + ".",
                severity="blocked",
            )
        )
    # Action-scoped failures stay attached to their rejected candidate. They do
    # not invalidate three or more independent actions that passed every gate.
    blockers.extend(blocker for blocker in conflict_blockers if blocker.action_id is None)
    blockers.extend(
        blocker
        for blocker in candidate_blockers
        if blocker.action_id is None and blocker.severity == "blocked"
    )
    if len(actions) < 3:
        # With no viable portfolio, hard safety failures still determine the
        # decision status. Once 3-7 independent actions pass, these same
        # failures remain local diagnostics and cannot poison the portfolio.
        blockers.extend(
            blocker
            for blocker in [*conflict_blockers, *candidate_blockers]
            if blocker.action_id is not None and blocker.severity == "blocked"
        )
    if coverage.missing_targets:
        blockers.append(
            _blocker(
                "incomplete_target_coverage",
                "Не покрыты объекты запроса: " + ", ".join(coverage.missing_targets) + ".",
            )
        )
    if coverage.missing_outputs:
        blockers.append(
            _blocker(
                "incomplete_requested_output_coverage",
                "Не покрыты части запроса: "
                + ", ".join(
                    _OUTPUT_LABELS_RU.get(output, output)
                    for output in coverage.missing_outputs
                )
                + ".",
            )
        )
    if len(actions) < 3:
        blockers.append(
            _blocker(
                "insufficient_action_portfolio",
                f"Для решения нужно 3–7 проверенных действий; готово {len(actions)}.",
            )
        )
    blockers = _unique_blockers(blockers)
    exposed_rejections = list(rejected_candidates)
    if blockers:
        status = "blocked" if any(item.severity == "blocked" for item in blockers) else "needs_data"
        exposed_actions: list[ManagerActionV1] = []
        rejected_ids = {candidate.candidate_id for candidate in exposed_rejections}
        portfolio_reasons = [item for item in blockers if item.action_id is None]
        for action in actions:
            if action.action_id in rejected_ids:
                continue
            exposed_rejections.append(_action_as_rejected(action, portfolio_reasons))
            rejected_ids.add(action.action_id)
    else:
        status = "ready_for_approval"
        exposed_actions = list(actions)
    as_of_values = [
        fact.snapshot_at
        for fact in evidence_snapshot
        if fact.snapshot_at is not None
    ]
    evidence_as_of = min(as_of_values) if as_of_values else None
    return ManagerDecisionV1(
        decision_id=decision_id,
        query_mode=query_mode,
        status=status,
        created_at=now,
        evidence_as_of=evidence_as_of,
        request_coverage=coverage,
        ranked_actions=exposed_actions,
        rejected_candidates=exposed_rejections,
        blockers=blockers,
        evidence_snapshot=list(evidence_snapshot),
    )


def build_manager_decision(
    synthesis_package: Mapping[str, Any],
    user_query: str | None = None,
    query_mode: str | Any | None = None,
    *,
    now: datetime | None = None,
    action_candidates: Sequence[Any] | None = None,
) -> ManagerDecisionV1:
    package = dict(synthesis_package or {})
    current_time = _utc(now or datetime.now(timezone.utc))
    query = str(user_query if user_query is not None else package.get("user_query") or "").strip()
    mode_value = getattr(query_mode, "value", query_mode)
    mode = str(mode_value or package.get("query_mode") or "decision").strip().lower()
    if mode not in {"decision", "diagnostic", "report", "lookup"}:
        mode = "decision"
    raw_candidates = extract_action_candidates(package, action_candidates)
    evidence_snapshot = extract_evidence_snapshot(package, now=current_time, action_candidates=raw_candidates)
    evidence_by_ref = {fact.evidence_ref: fact for fact in evidence_snapshot}
    requested_outputs = _requested_outputs(query, package)
    allowed_families = _allowed_families(requested_outputs, package)
    safety_checks = _list(_first(package, "action_safety_checks", "actionSafetyChecks"))

    parsed_actions: list[ManagerActionV1] = []
    rejected: list[RejectedCandidateV1] = []
    for index, raw in enumerate(raw_candidates):
        action, rejected_candidate = _build_action(raw, index=index, evidence_by_ref=evidence_by_ref)
        if rejected_candidate is not None:
            rejected.append(rejected_candidate)
        elif action is not None:
            parsed_actions.append(action)

    conflict_blockers = _conflicts(parsed_actions)
    conflict_ids = {
        blocker.action_id
        for blocker in conflict_blockers
        if blocker.code in {"conflicting_target_actions", "duplicate_action_candidate"}
    }
    valid_actions: list[ManagerActionV1] = []
    for action in parsed_actions:
        reasons = validate_manager_action(
            action,
            now=current_time,
            allowed_families=allowed_families,
            safety_checks=safety_checks,
        )
        reasons.extend(blocker for blocker in conflict_blockers if blocker.action_id == action.action_id)
        reasons = _unique_blockers(reasons)
        if reasons:
            rejected.append(
                RejectedCandidateV1(
                    candidate_id=action.action_id,
                    family=action.family,
                    target=action.target,
                    reasons=reasons,
                )
            )
        elif action.action_id not in conflict_ids:
            valid_actions.append(action)
    ranked = rank_manager_actions(valid_actions)
    coverage = _coverage(
        query=query,
        package=package,
        actions=ranked,
        evidence_snapshot=evidence_snapshot,
    )
    package_gaps = _list(_first(package, "blocking_gaps", "blockingGaps"))
    decision = apply_manager_hard_gate(
        decision_id=_decision_id(query, mode, package, raw_candidates),
        query_mode=mode,
        now=current_time,
        actions=ranked,
        rejected_candidates=rejected,
        evidence_snapshot=evidence_snapshot,
        coverage=coverage,
        package_blockers=package_gaps,
        conflict_blockers=[
            blocker for blocker in conflict_blockers if blocker.code == "conflicting_target_actions"
        ],
        candidate_blockers=[reason for candidate in rejected for reason in candidate.reasons],
    )
    return decision


def _format_value(value: ExactValueV1) -> str:
    raw = value.value
    if isinstance(raw, float):
        formatted = f"{raw:,.2f}".replace(",", " ")
    else:
        formatted = str(raw)
    unit = {
        "wb_status_code": "(статус WB)",
        "units_ordered": "шт. к заказу",
    }.get(value.unit, value.unit)
    return f"{formatted} {unit}".strip()


def _operation_label(operation: str | None) -> str:
    return _OPERATION_LABELS_RU.get(str(operation or ""), str(operation or "действие"))


def _target_label(family: str | None, target: str) -> str:
    prefix = {
        "advertising": "Кампания",
        "pricing": "SKU",
        "supply": "SKU",
        "sales": "SKU",
        "finance": "SKU",
    }.get(str(family or ""), "Объект")
    return f"{prefix} {target}"


def render_manager_decision(decision: ManagerDecisionV1) -> str:
    """Render only typed values; non-ready decisions are always facts-only."""

    if decision.status == "evaluating":
        return "\n".join(
            [
                "# Решение рассчитывается",
                "",
                "Допустимые рекламные действия проходят 500 воспроизводимых симуляций на кандидата.",
                "Исполняемые рекомендации появятся только после численного ranking и проверки LLM Critic.",
                f"Decision ID: `{decision.decision_id}`.",
            ]
        )

    if decision.status == "ready_for_approval":
        lines = ["# План менеджера", ""]
        for index, action in enumerate(decision.ranked_actions, start=1):
            effect = action.expected_effect
            lines.extend(
                [
                    f"{index}. **{_target_label(action.family, action.target)}: {_operation_label(action.operation)}**",
                    f"   Изменение: {_format_value(action.from_value)} → {_format_value(action.to_value)}.",
                    (
                        "   Ожидаемая добавочная маржинальная прибыль: "
                        f"{effect.contribution_profit_low_per_day:,.2f}–"
                        f"{effect.contribution_profit_high_per_day:,.2f} ₽/день "
                        f"(база {effect.contribution_profit_base_per_day:,.2f} ₽/день, "
                        f"горизонт {effect.horizon_days} дн.)."
                    ).replace(",", " "),
                    f"   Стоп-условие: {action.stop_condition}",
                    f"   Исполнение: {action.execution_mode}; подтверждение обязательно.",
                    "",
                ]
            )
        return "\n".join(lines).rstrip()

    title = "# Решение заблокировано" if decision.status == "blocked" else "# Недостаточно данных для решения"
    lines = [title, "", "Исполняемые рекомендации не сформированы."]
    if decision.blockers:
        lines.extend(["", "## Что не подтверждено"])
        for blocker in decision.blockers:
            target = f"{blocker.target}: " if blocker.target else ""
            lines.append(f"- {target}{blocker.message}")
    if decision.rejected_candidates:
        lines.extend(["", "## Возможные действия — не исполнять"])
        for candidate in decision.rejected_candidates[:7]:
            operation = f": {_operation_label(candidate.operation)}" if candidate.operation else ""
            change = ""
            if candidate.from_value is not None and candidate.to_value is not None:
                change = (
                    f" ({_format_value(candidate.from_value)} → "
                    f"{_format_value(candidate.to_value)})"
                )
            target = _target_label(
                candidate.family,
                candidate.target or candidate.candidate_id,
            )
            reason_text = "; ".join(
                dict.fromkeys(reason.message for reason in candidate.reasons[:3])
            )
            lines.append(f"- **{target}{operation}{change}.** Заблокировано: {reason_text}")
    facts = [fact for fact in decision.evidence_snapshot if fact.value is not None]
    if facts:
        lines.extend(["", "## Доступные факты"])
        for fact in facts[:12]:
            target = f"{fact.target}: " if fact.target else ""
            metric = _METRIC_LABELS_RU.get(_canonical_metric(fact.metric), fact.metric)
            unit_value = {
                "count": "шт.",
                "units": "шт.",
                "RUB": "₽",
                "RUB/day": "₽/день",
                "RUB/unit": "₽/шт.",
                "kopecks/1000_impressions": "коп./1000 показов",
                "wb_status_code": "код WB",
            }.get(str(fact.unit or ""), str(fact.unit or ""))
            unit = f" {unit_value}" if unit_value else ""
            if str(fact.period or "").startswith("as_of "):
                period = f" на {str(fact.period)[6:]}"
            else:
                period = f" за период {fact.period}" if fact.period else ""
            lines.append(f"- {target}{metric}: {fact.value}{unit}{period}.")
    return "\n".join(lines)


def manager_actions_to_action_board(decision: ManagerDecisionV1) -> list[dict[str, Any]]:
    """Convert an approved-ready portfolio to the current Action Board ingestion shape."""

    if decision.status != "ready_for_approval":
        return []
    result: list[dict[str, Any]] = []
    for index, action in enumerate(decision.ranked_actions, start=1):
        result.append(
            {
                "actionId": action.action_id,
                "priority": index,
                "actionFamily": action.family,
                "executionMode": action.execution_mode,
                "task": f"{action.operation}: {action.target}",
                "object": action.target,
                "objectType": action.target_type,
                "change": {
                    "from": action.from_value.model_dump(mode="json"),
                    "to": action.to_value.model_dump(mode="json"),
                },
                "executionPayload": action.execution_payload,
                "economics": action.economics.model_dump(mode="json", by_alias=True),
                "guardrails": [
                    rule.model_dump(mode="json", by_alias=True) for rule in action.guardrails
                ],
                "why": (
                    "Нижняя граница эффекта "
                    f"{action.expected_effect.contribution_profit_low_per_day:.2f} ₽/день; "
                    f"evidence: {', '.join(action.expected_effect.evidence_refs)}"
                ),
                "successCriteria": (
                    f"Измерить {action.measurement_spec.primary_metric} через "
                    f"{action.measurement_spec.evaluation_days} дн.; "
                    f"stop: {action.stop_condition}"
                ),
                "rollback": action.rollback.model_dump(mode="json", by_alias=True),
                "expiresAt": action.expires_at.isoformat(),
                "managerDecisionId": decision.decision_id,
            }
        )
    return result


def run_manager_decision_workflow(
    synthesis_package: Mapping[str, Any],
    user_query: str | None = None,
    query_mode: str | Any | None = None,
    *,
    now: datetime | None = None,
    action_candidates: Sequence[Any] | None = None,
) -> ManagerDecisionWorkflowResult:
    decision = build_manager_decision(
        synthesis_package,
        user_query,
        query_mode,
        now=now,
        action_candidates=action_candidates,
    )
    return ManagerDecisionWorkflowResult(
        decision=decision,
        rendered_markdown=render_manager_decision(decision),
        facts_only=decision.status != "ready_for_approval",
    )


__all__ = [
    "apply_manager_hard_gate",
    "build_manager_decision",
    "extract_action_candidates",
    "extract_evidence_snapshot",
    "manager_action_candidates_from_forecast",
    "manager_actions_to_action_board",
    "rank_manager_actions",
    "render_manager_decision",
    "run_manager_decision_workflow",
    "validate_manager_action",
]
