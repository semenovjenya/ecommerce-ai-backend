from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
import math
import os
from typing import Any, Iterable, Mapping, Sequence


_OPERATIONAL_FRESHNESS_DAYS = 3
_AD_BASELINE_DAYS = 7
_AD_EVALUATION_DAYS = 7
_AD_ACTION_TTL = timedelta(hours=2)
_SUPPLY_PREVIEW_TTL = timedelta(hours=24)


def _simulation_evidence_enabled() -> bool:
    return os.getenv("DECISION_SIMULATION_MODE", "off").strip().lower() in {
        "shadow",
        "enforced",
    }


@dataclass(frozen=True)
class CandidateBuildResultV1:
    """Pure output of the deterministic action-candidate builder."""

    candidates: list[dict[str, Any]]
    evidence_facts: list[dict[str, Any]]
    diagnostics: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": "1.0",
            "candidates": self.candidates,
            "evidenceFacts": self.evidence_facts,
            "diagnostics": self.diagnostics,
        }


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
    try:
        return _utc(datetime.fromisoformat(str(value).strip().replace("Z", "+00:00")))
    except (TypeError, ValueError):
        try:
            return datetime.combine(date.fromisoformat(str(value).strip()), time.min, tzinfo=timezone.utc)
        except (TypeError, ValueError):
            return None


def _number(value: Any) -> int | float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, Decimal):
        value = float(value)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    if result.is_integer():
        return int(result)
    return result


def _integer(value: Any) -> int | None:
    number = _number(value)
    if number is None or int(number) != number:
        return None
    return int(number)


def _number_list(value: Any) -> list[int | float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    numbers = [_number(item) for item in value]
    if any(item is None for item in numbers):
        return None
    return [item for item in numbers if item is not None]


def _text(value: Any) -> str | None:
    if value is None:
        return None
    result = str(value).strip()
    return result or None


def _period(snapshot_at: Any) -> str | None:
    parsed = _datetime(snapshot_at)
    return f"as_of {parsed.date().isoformat()}" if parsed else None


def _range_period(start: Any, end: Any) -> str | None:
    parsed_start = _datetime(start)
    parsed_end = _datetime(end)
    if parsed_start is None or parsed_end is None:
        return None
    return f"{parsed_start.date().isoformat()}/{parsed_end.date().isoformat()}"


def _evidence_ref(domain: str, target_type: str, target: Any, metric: str) -> str:
    return f"manager:{domain}:{target_type}:{target}:{metric}"


def _fact(
    *,
    metric: str,
    value: Any,
    unit: str,
    target: Any,
    evidence_ref: str,
    source_table: str,
    definition: str,
    retrieved_at: Any,
    snapshot_at: Any = None,
    period: str | None = None,
    period_start: Any = None,
    period_end: Any = None,
    source_agent: str,
) -> dict[str, Any] | None:
    if value is None:
        return None
    payload: dict[str, Any] = {
        "metric": metric,
        "value": value,
        "unit": unit,
        "target": str(target),
        "period": period,
        "sourceTable": source_table,
        "definition": definition,
        "freshnessStatus": "unknown",
        "confidence": "high",
        "evidenceRef": evidence_ref,
        "sourceAgent": source_agent,
    }
    parsed_snapshot = _datetime(snapshot_at)
    parsed_retrieved = _datetime(retrieved_at)
    parsed_start = _datetime(period_start)
    parsed_end = _datetime(period_end)
    if parsed_snapshot is not None:
        payload["snapshotAt"] = parsed_snapshot.isoformat()
    if parsed_retrieved is not None:
        payload["retrievedAt"] = parsed_retrieved.isoformat()
    if parsed_start is not None:
        payload["periodStart"] = parsed_start.isoformat()
    if parsed_end is not None:
        payload["periodEnd"] = parsed_end.isoformat()
    return payload


def _blocker(code: str, message: str, *, severity: str = "needs_data") -> dict[str, str]:
    return {"code": code, "message": message, "severity": severity}


def _unique_blockers(values: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for value in values:
        key = (value["code"], value["message"])
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _date_blockers(
    named_values: Sequence[tuple[str, Any]],
    *,
    now: datetime,
) -> list[dict[str, str]]:
    parsed = [(name, value) for name, raw in named_values if (value := _datetime(raw)) is not None]
    future = sorted(name for name, value in parsed if value.date() > now.date())
    stale = sorted(
        name
        for name, value in parsed
        if (now.date() - value.date()).days > _OPERATIONAL_FRESHNESS_DAYS
    )
    blockers: list[dict[str, str]] = []
    if future:
        blockers.append(
            _blocker(
                "future_source_data",
                "Дата источника находится в будущем: " + ", ".join(future) + ".",
                severity="blocked",
            )
        )
    if stale:
        blockers.append(
            _blocker(
                "stale_source_data",
                "Данные старше трёх завершённых дней: " + ", ".join(stale) + ".",
            )
        )
    non_future = [value for _name, value in parsed if value.date() <= now.date()]
    if non_future and (max(non_future).date() - min(non_future).date()).days > _OPERATIONAL_FRESHNESS_DAYS:
        blockers.append(
            _blocker(
                "incoherent_source_dates",
                "Обязательные источники относятся к несовместимым датам.",
            )
        )
    return blockers


def _append_fact(target: list[dict[str, Any]], fact: dict[str, Any] | None) -> None:
    if fact is not None:
        target.append(fact)


def _advertising_groups(rows: Sequence[Mapping[str, Any]]) -> list[list[Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        advert_id = _text(row.get("advert_id"))
        if advert_id is not None:
            groups.setdefault(advert_id, []).append(row)
    return sorted(
        groups.values(),
        key=lambda group: (
            -float(_number(group[0].get("ad_spend_7d_rub")) or 0),
            str(group[0].get("advert_id")),
        ),
    )


def _select_bid(row: Mapping[str, Any]) -> tuple[Any, str | None, list[dict[str, str]]]:
    explicit_bid = _number(row.get("current_bid_kopecks"))
    explicit_placement = _text(row.get("bid_placement"))
    if explicit_bid is not None and explicit_bid > 0 and explicit_placement:
        return explicit_bid, explicit_placement, []

    placements = {
        "search": _number(row.get("bid_search_kopecks")),
        "recommendations": _number(row.get("bid_recommendations_kopecks")),
    }
    # WB snapshots use zero for an inactive/unset placement. It is not a
    # competing current bid and must not make one positive placement ambiguous.
    known = {
        placement: value
        for placement, value in placements.items()
        if value is not None and value > 0
    }
    if not known:
        return None, None, [
            _blocker("missing_current_bid", "Нет актуальной ставки рекламной кампании.")
        ]
    distinct_values = set(known.values())
    if len(known) > 1 and len(distinct_values) == 1:
        return next(iter(distinct_values)), "+".join(sorted(known)), []
    if len(known) > 1:
        return known, None, [
            _blocker(
                "ambiguous_current_bid_placement",
                "Есть ставки для нескольких плейсментов, но не выбран фактический текущий плейсмент.",
            )
        ]
    placement, bid = next(iter(known.items()))
    return bid, placement, []


def _ad_evidence(
    row: Mapping[str, Any],
    *,
    advert_id: str,
    nm_id: str,
    retrieved_at: Any,
    bid_value: Any,
    bid_placement: str | None,
) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    simulation_enabled = _simulation_evidence_enabled()
    stats_source_table = (
        "agent_marts.v_agent_ad_campaign_sku_daily"
        if simulation_enabled
        else "marketing.wb_marketing_campaign_stats_daily"
    )
    stats_period = _range_period(row.get("stats_period_start"), row.get("stats_period_end"))
    campaign_ref = lambda metric: _evidence_ref("advertising", "campaign", advert_id, metric)
    sku_ref = lambda metric: _evidence_ref("advertising", "sku", nm_id, metric)

    _append_fact(
        facts,
        _fact(
            metric="campaign_status",
            value=_integer(row.get("campaign_status")),
            unit="wb_status_code",
            target=advert_id,
            evidence_ref=campaign_ref("campaign_status"),
            source_table="marketing.wb_marketing_campaigns",
            snapshot_at=row.get("campaign_snapshot_date"),
            retrieved_at=row.get("campaign_retrieved_at") or retrieved_at,
            period=_period(row.get("campaign_snapshot_date")),
            period_end=row.get("campaign_snapshot_date"),
            definition="WB campaign status code from the latest campaign snapshot; 9 is active and 11 is paused.",
            source_agent="marketing",
        ),
    )
    bid_definition = (
        f"Current campaign bid for the explicitly selected {bid_placement} placement."
        if bid_placement
        else "Placement-level campaign bids; no single current placement is proven."
    )
    _append_fact(
        facts,
        _fact(
            metric="bid",
            value=bid_value,
            unit=(
                "kopecks/1000_impressions"
                if bid_placement
                else "kopecks/1000_impressions_by_placement"
            ),
            target=advert_id,
            evidence_ref=campaign_ref("bid"),
            source_table="marketing.wb_marketing_campaign_nms",
            snapshot_at=row.get("bid_snapshot_date"),
            retrieved_at=row.get("bid_retrieved_at") or retrieved_at,
            period=_period(row.get("bid_snapshot_date")),
            period_end=row.get("bid_snapshot_date"),
            definition=bid_definition,
            source_agent="marketing",
        ),
    )
    flow_metrics = [
        (
            "ad_spend",
            "ad_spend_7d_rub",
            "RUB",
            stats_source_table,
            "marketing",
            (
                "Advertising spend attributed to this SKU inside the campaign in the exact completed-day window."
                if simulation_enabled
                else "Advertising spend attributed to the campaign in the exact completed-day window."
            ),
        ),
        (
            "clicks",
            "clicks_7d",
            "count",
            stats_source_table,
            "marketing",
            (
                "Clicks attributed to this SKU inside the campaign in the exact completed-day window."
                if simulation_enabled
                else "Campaign clicks in the exact completed-day window."
            ),
        ),
        (
            "ad_orders",
            "ad_orders_7d",
            "count",
            stats_source_table,
            "marketing",
            (
                "Orders attributed to this SKU inside the campaign in the exact completed-day window."
                if simulation_enabled
                else "Orders attributed to the campaign in the exact completed-day window."
            ),
        ),
    ]
    if simulation_enabled:
        flow_metrics.append(
            (
                "ad_units",
                "ad_units_7d",
                "units",
                stats_source_table,
                "marketing",
                "Ordered units attributed to this SKU inside the campaign in the exact completed-day window.",
            )
        )
    flow_metrics.extend(
        [
            (
                "revenue",
                "attributed_revenue_7d_rub",
                "RUB",
                _text(row.get("attributed_revenue_source_table")),
                "marketing",
                "Revenue attributed to the campaign in the exact completed-day window.",
            ),
            (
                "contribution_profit",
                "campaign_contribution_profit_7d_rub",
                "RUB",
                _text(row.get("campaign_contribution_source_table")),
                "finance",
                "Campaign-attributed contribution profit after COGS, commissions, logistics, storage and advertising.",
            ),
        ]
    )
    for metric, field, unit, source_table, source_agent, definition in flow_metrics:
        _append_fact(
            facts,
            _fact(
                metric=metric,
                value=_number(row.get(field)),
                unit=unit,
                target=advert_id,
                evidence_ref=campaign_ref(metric),
                source_table=source_table,
                snapshot_at=row.get("stats_period_end"),
                retrieved_at=retrieved_at,
                period=stats_period,
                period_start=row.get("stats_period_start"),
                period_end=row.get("stats_period_end"),
                definition=definition,
                source_agent=source_agent,
            ),
        )
    _append_fact(
        facts,
        _fact(
            metric="budget_remaining",
            value=_number(row.get("budget_remaining_rub")),
            unit="RUB",
            target=advert_id,
            evidence_ref=campaign_ref("budget_remaining"),
            source_table="marketing.wb_marketing_budget",
            snapshot_at=row.get("budget_snapshot_date"),
            retrieved_at=row.get("budget_retrieved_at") or retrieved_at,
            period=_period(row.get("budget_snapshot_date")),
            period_end=row.get("budget_snapshot_date"),
            definition="Current remaining campaign budget from the latest budget snapshot.",
            source_agent="marketing",
        ),
    )
    _append_fact(
        facts,
        _fact(
            metric="stock_on_hand",
            value=_number(row.get("stock_on_hand")),
            unit="units",
            target=nm_id,
            evidence_ref=sku_ref("stock_on_hand"),
            source_table="supply.wb_supplier_stocks",
            snapshot_at=row.get("stock_snapshot_date"),
            retrieved_at=retrieved_at,
            period=_period(row.get("stock_snapshot_date")),
            period_end=row.get("stock_snapshot_date"),
            definition="Sellable on-hand quantity only; quantity_full and customer transit are excluded.",
            source_agent="supply",
        ),
    )
    _append_fact(
        facts,
        _fact(
            metric="supplier_inbound",
            value=_number(row.get("supplier_inbound_qty")),
            unit="units",
            target=nm_id,
            evidence_ref=sku_ref("supplier_inbound"),
            source_table="supply.wb_supplies + supply.wb_supplies_goods",
            snapshot_at=row.get("supplier_snapshot_date"),
            retrieved_at=retrieved_at,
            period=_period(row.get("supplier_snapshot_date")),
            period_end=row.get("supplier_snapshot_date"),
            definition=(
                "Confirmed outstanding supplier inbound by SKU in the complete latest "
                "supply snapshot; absence of an outstanding row is an audited zero, and "
                "customer in_way_to/from_client is excluded."
            ),
            source_agent="supply",
        ),
    )
    _append_fact(
        facts,
        _fact(
            metric="cogs",
            value=_number(row.get("unit_cogs_rub")),
            unit="RUB/unit",
            target=nm_id,
            evidence_ref=sku_ref("cogs"),
            source_table="finance.wb_product_coast",
            snapshot_at=row.get("cogs_snapshot_date"),
            retrieved_at=row.get("cogs_retrieved_at") or retrieved_at,
            period=_period(row.get("cogs_snapshot_date")),
            period_end=row.get("cogs_snapshot_date"),
            definition="Latest source unit cost for the SKU.",
            source_agent="finance",
        ),
    )
    daily_values: dict[str, Any] = {
        "min": _number(row.get("daily_spend_min_rub")),
        "median": _number(row.get("daily_spend_median_rub")),
        "max": _number(row.get("daily_spend_max_rub")),
    }
    if simulation_enabled:
        daily_values = {
            "valuesRub": _number_list(row.get("daily_spend_values_rub")),
            **daily_values,
        }
    complete_daily_values = all(
        daily_values[key] is not None for key in ("min", "median", "max")
    ) and (
        not simulation_enabled
        or (
            isinstance(daily_values.get("valuesRub"), list)
            and len(daily_values["valuesRub"]) == _AD_BASELINE_DAYS
        )
    )
    if complete_daily_values:
        _append_fact(
            facts,
            _fact(
                metric="daily_ad_spend_distribution",
                value=daily_values,
                unit="RUB/day",
                target=advert_id,
                evidence_ref=campaign_ref("daily_ad_spend_distribution"),
                source_table=stats_source_table,
                snapshot_at=row.get("stats_period_end"),
                retrieved_at=retrieved_at,
                period=stats_period,
                period_start=row.get("stats_period_start"),
                period_end=row.get("stats_period_end"),
                definition=(
                    "Observed campaign spend for each of seven completed days, ordered by day, "
                    "with the minimum, median and maximum of the same values."
                    if simulation_enabled
                    else "Observed minimum, median and maximum daily campaign spend over seven completed days."
                ),
                source_agent="marketing",
            ),
        )
    return facts


def _build_ad_candidate(
    group: Sequence[Mapping[str, Any]],
    *,
    now: datetime,
    retrieved_at: Any,
) -> dict[str, Any] | None:
    row = group[0]
    simulation_enabled = _simulation_evidence_enabled()
    advert_id = _text(row.get("advert_id"))
    if advert_id is None or _integer(row.get("campaign_status")) != 9:
        return None
    spend = _number(row.get("ad_spend_7d_rub"))
    diagnostic_spend = (
        _number(row.get("observed_ad_spend_7d_rub"))
        if simulation_enabled
        else None
    )
    if (spend if spend is not None else diagnostic_spend) in (None, 0):
        return None
    if (spend or diagnostic_spend or 0) < 0:
        return None

    nms = list(
        dict.fromkeys(
            str(value)
            for source in group
            if (value := _integer(source.get("nm_id"))) is not None
        )
    )
    zero_stock_nms = [
        str(nm_id)
        for source in group
        if (nm_id := _integer(source.get("nm_id"))) is not None
        and _number(source.get("stock_on_hand")) == 0
    ]
    if zero_stock_nms:
        nm_id = zero_stock_nms[0]
    elif simulation_enabled and nms:
        # Preserve the active/spending campaign as a rejected diagnostic when
        # the stock snapshot is missing or does not prove zero. Economics stay
        # unavailable and the normal hard-gate blockers explain why.
        nm_id = nms[0]
    else:
        return None
    sku_row = dict(next(
        (source for source in group if _text(source.get("nm_id")) == nm_id),
        row,
    ))
    if simulation_enabled and "supplier_audit_snapshot_date" in sku_row:
        sku_row.update(
            {
                "supplier_snapshot_date": sku_row.get("supplier_audit_snapshot_date"),
                "supplier_inbound_qty": sku_row.get("supplier_inbound_audited_qty"),
                "supplier_goods_rows": sku_row.get("supplier_goods_rows_audited"),
            }
        )
    declared_sku_counts = [
        value
        for source in group
        if (value := _integer(source.get("active_sku_count"))) is not None
    ]
    active_sku_count = max([len(nms), *declared_sku_counts]) if (nms or declared_sku_counts) else None
    bid_value, bid_placement, blockers = _select_bid(sku_row)

    if active_sku_count is None:
        blockers.append(
            _blocker("missing_campaign_sku_count", "Неизвестно число SKU в кампании.")
        )
    elif active_sku_count != 1:
        blockers.append(
            _blocker(
                "multi_sku_campaign",
                "Кампания содержит несколько SKU; пауза всей кампании небезопасна для одного товара.",
                severity="blocked",
            )
        )

    stats_start = _datetime(sku_row.get("stats_period_start"))
    stats_end = _datetime(sku_row.get("stats_period_end"))
    stats_days = _integer(sku_row.get("stats_day_count"))
    exact_completed_window = bool(
        stats_start is not None
        and stats_end is not None
        and stats_days == _AD_BASELINE_DAYS
        and (stats_end.date() - stats_start.date()).days == _AD_BASELINE_DAYS - 1
        and stats_end.date() < now.date()
    )
    stats_window_complete = (
        sku_row.get("stats_window_complete") is True if simulation_enabled else True
    )
    campaign_sku_attribution_reconciled = (
        sku_row.get("campaign_sku_attribution_reconciled") is True
        if simulation_enabled
        else True
    )

    ad_orders = _number(sku_row.get("ad_orders_7d"))
    ad_units = _number(sku_row.get("ad_units_7d"))
    attributed_revenue = _number(sku_row.get("attributed_revenue_7d_rub"))
    contribution = _number(sku_row.get("campaign_contribution_profit_7d_rub"))
    supplier_inbound = _number(sku_row.get("supplier_inbound_qty"))
    cogs = _number(sku_row.get("unit_cogs_rub"))
    budget = _number(sku_row.get("budget_remaining_rub"))
    zero_commercial_effect_proven = bool(
        simulation_enabled
        and exact_completed_window
        and stats_window_complete
        and campaign_sku_attribution_reconciled
        and ad_orders == 0
        and ad_units == 0
        and attributed_revenue == 0
        and _text(sku_row.get("attributed_revenue_source_table"))
    )

    required_values = [
        (
            "budget_remaining_rub",
            "missing_campaign_budget",
            "Нет актуального остатка бюджета кампании.",
        ),
        (
            "stock_on_hand",
            "missing_stock_on_hand",
            "Нет доступного on-hand остатка SKU.",
        ),
        (
            "supplier_inbound_qty",
            "missing_confirmed_supplier_inbound",
            "Нет подтверждённой поставки поставщика по этому SKU; клиентский транзит не учитывается.",
        ),
        (
            "attributed_revenue_7d_rub",
            "missing_attributed_revenue",
            "Нет выручки, атрибутированной этой кампании за тот же период.",
        ),
        (
            "campaign_contribution_profit_7d_rub",
            "missing_campaign_contribution",
            "Нет полной маржинальной прибыли по кампании за тот же период.",
        ),
        (
            "clicks_7d",
            "missing_campaign_clicks",
            "Нет кликов кампании за полный период.",
        ),
        (
            "ad_orders_7d",
            "missing_campaign_orders",
            "Нет заказов кампании за полный период.",
        ),
    ]
    if simulation_enabled:
        required_values.append(
            (
                "ad_units_7d",
                "missing_campaign_units",
                "Нет количества заказанных единиц SKU за полный период.",
            )
        )
    if not zero_commercial_effect_proven:
        required_values.append(
            (
                "unit_cogs_rub",
                "missing_cogs",
                "Нет себестоимости SKU; прибыль заявлять нельзя.",
            )
        )
    for field, code, message in required_values:
        if _number(sku_row.get(field)) is None:
            blockers.append(_blocker(code, message))
    if not _text(sku_row.get("attributed_revenue_source_table")):
        blockers.append(
            _blocker(
                "missing_attributed_revenue_source",
                "Не указан проверяемый источник атрибутированной выручки кампании.",
            )
        )
    if not _text(sku_row.get("campaign_contribution_source_table")):
        blockers.append(
            _blocker(
                "missing_campaign_contribution_source",
                "Не указан проверяемый источник маржинальной прибыли кампании.",
            )
        )
    if simulation_enabled and sku_row.get("stock_snapshot_complete") is not True:
        blockers.append(
            _blocker(
                "incomplete_stock_snapshot",
                "Наличие строк остатков не доказывает полноту снимка; нужен ETL watermark.",
            )
        )
    if simulation_enabled and sku_row.get("supplier_snapshot_complete") is not True:
        blockers.append(
            _blocker(
                "incomplete_supplier_snapshot",
                "Наличие строк поставок не доказывает полноту снимка; нужен ETL watermark.",
            )
        )

    if (
        stats_start is None
        or stats_end is None
        or stats_days != _AD_BASELINE_DAYS
        or (stats_end.date() - stats_start.date()).days != _AD_BASELINE_DAYS - 1
    ):
        blockers.append(
            _blocker(
                "incomplete_campaign_stats_window",
                (
                    "Нужны ровно семь календарных дней статистики campaign × SKU."
                    if simulation_enabled
                    else "Нужны ровно семь завершённых календарных дней статистики кампании."
                ),
            )
        )
    elif stats_end.date() >= now.date():
        blockers.append(
            _blocker(
                "campaign_stats_window_not_completed",
                "Окно рекламы включает текущий или будущий незавершённый день.",
            )
        )
    if simulation_enabled and not stats_window_complete:
        blockers.append(
            _blocker(
                "incomplete_campaign_sku_stats",
                "Источник не подтвердил полноту spend/clicks/orders/units для каждого из семи дней campaign × SKU.",
            )
        )
    if simulation_enabled and not campaign_sku_attribution_reconciled:
        blockers.append(
            _blocker(
                "unreconciled_campaign_sku_attribution",
                "Сумма campaign × SKU не согласована с campaign-level атрибутированной выручкой за каждый день окна.",
            )
        )

    minimum = _number(sku_row.get("daily_spend_min_rub"))
    median = _number(sku_row.get("daily_spend_median_rub"))
    maximum = _number(sku_row.get("daily_spend_max_rub"))
    daily_spend_values = _number_list(sku_row.get("daily_spend_values_rub"))
    if simulation_enabled and (
        daily_spend_values is None or len(daily_spend_values) != _AD_BASELINE_DAYS
    ):
        blockers.append(
            _blocker(
                "incomplete_daily_spend_history",
                "Нужны фактические расходы за каждый из семи завершённых дней.",
            )
        )
    if minimum is None or median is None or maximum is None:
        blockers.append(
            _blocker(
                "missing_daily_spend_distribution",
                "Нет min/median/max дневного расхода за полный период.",
            )
        )
    elif not minimum <= median <= maximum:
        blockers.append(
            _blocker(
                "invalid_daily_spend_distribution",
                "Диапазон дневного расхода противоречив: требуется min <= median <= max.",
                severity="blocked",
            )
        )
    elif minimum <= 0 and not simulation_enabled:
        blockers.append(
            _blocker(
                "non_positive_daily_spend_floor",
                "Нижняя граница наблюдаемой экономии равна нулю; положительный эффект не доказан.",
            )
        )

    if ad_orders is not None and ad_orders != 0:
        blockers.append(
            _blocker(
                "unsafe_pause_with_attributed_orders",
                "В периоде есть заказы из рекламы; нулевой упущенный эффект от паузы не доказан.",
                severity="blocked",
            )
        )
    if simulation_enabled and ad_units is not None and ad_units != 0:
        blockers.append(
            _blocker(
                "unsafe_pause_with_attributed_units",
                "В периоде есть заказанные единицы SKU из рекламы; нулевой упущенный эффект от паузы не доказан.",
                severity="blocked",
            )
        )
    if attributed_revenue is not None and attributed_revenue != 0:
        blockers.append(
            _blocker(
                "unsafe_pause_with_attributed_revenue",
                "У кампании есть атрибутированная выручка; пауза требует отдельной causal-модели.",
                severity="blocked",
            )
        )
    if contribution is not None and contribution > 0:
        blockers.append(
            _blocker(
                "positive_campaign_contribution",
                "Кампания показывает положительную маржинальную прибыль; пауза не обоснована.",
                severity="blocked",
            )
        )
    if supplier_inbound is not None and supplier_inbound != 0:
        blockers.append(
            _blocker(
                "confirmed_supplier_inbound_exists",
                "По SKU есть подтверждённая поставка поставщика; действие нужно пересчитать к ETA.",
                severity="blocked",
            )
        )
    if cogs is not None and cogs <= 0:
        blockers.append(_blocker("invalid_cogs", "Себестоимость SKU должна быть больше нуля."))
    if budget is not None and budget < 0:
        blockers.append(_blocker("invalid_campaign_budget", "Остаток бюджета не может быть отрицательным."))

    date_fields: list[tuple[str, Any]] = [
        ("campaign", sku_row.get("campaign_snapshot_date")),
        ("bid", sku_row.get("bid_snapshot_date")),
        ("budget", sku_row.get("budget_snapshot_date")),
        ("campaign_stats", sku_row.get("stats_period_end")),
        ("stock_on_hand", sku_row.get("stock_snapshot_date")),
    ]
    if cogs is not None:
        date_fields.append(("cogs", sku_row.get("cogs_snapshot_date")))
        if _datetime(sku_row.get("cogs_snapshot_date")) is None:
            blockers.append(_blocker("missing_cogs_snapshot_date", "Нет даты снимка себестоимости."))
    if supplier_inbound is not None:
        date_fields.append(("supplier_inbound", sku_row.get("supplier_snapshot_date")))
        if _datetime(sku_row.get("supplier_snapshot_date")) is None:
            blockers.append(
                _blocker("missing_supplier_snapshot_date", "Нет даты поставки поставщика по SKU.")
            )
    blockers.extend(_date_blockers(date_fields, now=now))

    facts = _ad_evidence(
        sku_row,
        advert_id=advert_id,
        nm_id=nm_id,
        retrieved_at=retrieved_at,
        bid_value=bid_value,
        bid_placement=bid_placement,
    )
    candidate_id = f"det-ad-pause-{advert_id}"
    candidate: dict[str, Any] = {
        "actionId": candidate_id,
        "family": "advertising",
        "target": advert_id,
        "targetType": "campaign",
        "targetRefs": nms or [nm_id],
        "operation": "pause_campaign",
        "executionMode": "wb_api",
        "executionPayload": {"advert_id": int(advert_id) if advert_id.isdigit() else advert_id},
        "from": {"value": 9, "unit": "wb_status_code"},
        "to": {"value": 11, "unit": "wb_status_code"},
        "evidence": facts,
        "evidenceRefs": [fact["evidenceRef"] for fact in facts],
        "requiredMetrics": (
            [
                "campaign_status",
                "stock_on_hand",
                "supplier_inbound",
                "ad_orders",
                "ad_units",
                "revenue",
                "contribution_profit",
                "budget_remaining",
            ]
            if simulation_enabled
            else ["supplier_inbound"]
        ),
        "prerequisites": [
            (
                "За пять минут до применения перечитать WB status, бюджет, on-hand, поставку, "
                "рекламные заказы/единицы, атрибутированную выручку и contribution profit; from и "
                "все structured guardrails должны совпасть."
                if simulation_enabled
                else "За пять минут до применения перечитать WB status, бюджет, on-hand и поставку; from и guardrails должны совпасть."
            )
        ],
        "stopCondition": (
            (
                "Не применять или откатить паузу, если появился on-hand, подтверждённая поставка, "
                "рекламный заказ/единица, атрибутированная выручка либо campaign contribution стал положительным."
                if simulation_enabled
                else "Не применять или откатить паузу, если появился on-hand, подтверждённая поставка либо campaign contribution стал положительным."
            )
        ),
        "rollback": {
            "operation": "resume_campaign",
            "from": {"value": 11, "unit": "wb_status_code"},
            "to": {"value": 9, "unit": "wb_status_code"},
            "instruction": "Повторно запустить ту же кампанию после успешного WB preflight.",
        },
        "measurementSpec": {
            "primaryMetric": "contribution_profit_rub",
            "baselineDays": _AD_BASELINE_DAYS,
            "evaluationDays": _AD_EVALUATION_DAYS,
            "attributionMethod": "observed_change",
            "guardrailMetrics": (
                [
                    "stock_on_hand",
                    "supplier_inbound",
                    "contribution_profit",
                    "budget_remaining",
                    "ad_orders",
                    "ad_units",
                    "attributed_revenue",
                ]
                if simulation_enabled
                else [
                    "stock_on_hand",
                    "supplier_inbound",
                    "contribution_profit",
                    "budget_remaining",
                    "ad_orders",
                ]
            ),
        },
        "expiresAt": (now + _AD_ACTION_TTL).isoformat(),
        "urgency": "critical",
        "confidence": "high",
    }

    blockers = _unique_blockers(blockers)
    if blockers:
        blockers.append(
            _blocker(
                "missing_deterministic_economics",
                "Полный диапазон добавочной маржинальной прибыли не рассчитан из совместимых фактов.",
            )
        )
        candidate["candidateBlockers"] = _unique_blockers(blockers)
        return candidate

    assert minimum is not None and median is not None and maximum is not None
    daily_ref = _evidence_ref(
        "advertising", "campaign", advert_id, "daily_ad_spend_distribution"
    )
    economics_refs = [
        daily_ref,
        _evidence_ref("advertising", "campaign", advert_id, "revenue"),
        _evidence_ref("advertising", "campaign", advert_id, "contribution_profit"),
        _evidence_ref("advertising", "sku", nm_id, "stock_on_hand"),
        _evidence_ref("advertising", "sku", nm_id, "supplier_inbound"),
    ]
    if not zero_commercial_effect_proven:
        economics_refs.append(_evidence_ref("advertising", "sku", nm_id, "cogs"))
    zero = {"low": 0, "base": 0, "high": 0}
    candidate["economics"] = {
        "deltaRevenuePerDay": dict(zero),
        "deltaCogsPerDay": dict(zero),
        "deltaCommissionPerDay": dict(zero),
        "deltaLogisticsPerDay": dict(zero),
        "deltaStoragePerDay": dict(zero),
        "deltaAdvertisingPerDay": {
            "low": -float(maximum),
            "base": -float(median),
            "high": -float(minimum),
        },
        "actionCostPerDay": dict(zero),
        "currency": "RUB",
        "horizonDays": _AD_EVALUATION_DAYS,
        "calculationMethod": "zero_stock_daily_spend_distribution_v1",
        "evidenceRefs": economics_refs,
        "causalValid": False,
    }
    candidate["expectedEffect"] = {
        "contributionProfitLowPerDay": float(minimum),
        "contributionProfitBasePerDay": float(median),
        "contributionProfitHighPerDay": float(maximum),
        "currency": "RUB",
        "horizonDays": _AD_EVALUATION_DAYS,
        "calculationMethod": "zero_stock_daily_spend_distribution_v1",
        "evidenceRefs": economics_refs,
        "causalValid": False,
    }
    candidate["guardrails"] = [
        {
            "metric": "stock_on_hand",
            "operator": "eq",
            "threshold": {"value": 0, "unit": "units"},
            "onBreach": "rollback",
            "evidenceRef": _evidence_ref("advertising", "sku", nm_id, "stock_on_hand"),
        },
        {
            "metric": "supplier_inbound",
            "operator": "eq",
            "threshold": {"value": 0, "unit": "units"},
            "onBreach": "hold",
            "evidenceRef": _evidence_ref("advertising", "sku", nm_id, "supplier_inbound"),
        },
        {
            "metric": "contribution_profit",
            "operator": "lte",
            "threshold": {"value": 0, "unit": "RUB"},
            "onBreach": "block",
            "evidenceRef": _evidence_ref("advertising", "campaign", advert_id, "contribution_profit"),
        },
        {
            "metric": "budget_remaining",
            "operator": "gte",
            "threshold": {"value": 0, "unit": "RUB"},
            "onBreach": "block",
            "evidenceRef": _evidence_ref("advertising", "campaign", advert_id, "budget_remaining"),
        },
    ]
    if simulation_enabled:
        candidate["guardrails"][2:2] = [
            {
                "metric": "ad_orders",
                "operator": "eq",
                "threshold": {"value": 0, "unit": "count"},
                "onBreach": "block",
                "evidenceRef": _evidence_ref(
                    "advertising", "campaign", advert_id, "ad_orders"
                ),
            },
            {
                "metric": "ad_units",
                "operator": "eq",
                "threshold": {"value": 0, "unit": "units"},
                "onBreach": "block",
                "evidenceRef": _evidence_ref(
                    "advertising", "campaign", advert_id, "ad_units"
                ),
            },
            {
                "metric": "attributed_revenue",
                "operator": "eq",
                "threshold": {"value": 0, "unit": "RUB"},
                "onBreach": "block",
                "evidenceRef": _evidence_ref(
                    "advertising", "campaign", advert_id, "revenue"
                ),
            },
        ]
    return candidate


def _supply_evidence(
    row: Mapping[str, Any],
    *,
    nm_id: str,
    retrieved_at: Any,
) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    ref = lambda metric: _evidence_ref("supply", "sku", nm_id, metric)
    report_date = row.get("report_date")
    _append_fact(
        facts,
        _fact(
            metric="stock_on_hand",
            value=_number(row.get("stock_qty")),
            unit="units",
            target=nm_id,
            evidence_ref=ref("stock_on_hand"),
            source_table="supply.wb_supplier_stocks",
            snapshot_at=report_date,
            retrieved_at=retrieved_at,
            period=_period(report_date),
            period_end=report_date,
            definition="Sellable on-hand quantity only; quantity_full and customer transit are excluded.",
            source_agent="supply",
        ),
    )
    _append_fact(
        facts,
        _fact(
            metric="sales_velocity",
            value=_number(row.get("avg_daily_sales_7d")),
            unit="units/day",
            target=nm_id,
            evidence_ref=ref("sales_velocity"),
            source_table="sales.wb_supplier_sales",
            retrieved_at=retrieved_at,
            period="latest_7_completed_sales_days; exact bounds unavailable",
            definition="Average daily realized units over the latest seven full sales days; exact dates are not exposed by this snapshot.",
            source_agent="sales",
        ),
    )
    _append_fact(
        facts,
        _fact(
            metric="supplier_inbound",
            value=_number(row.get("supplier_inbound_qty")),
            unit="units",
            target=nm_id,
            evidence_ref=ref("supplier_inbound"),
            source_table="supply.wb_supplies + supply.wb_supplies_goods",
            snapshot_at=row.get("supplier_snapshot_date"),
            retrieved_at=retrieved_at,
            period=_period(row.get("supplier_snapshot_date")),
            period_end=row.get("supplier_snapshot_date"),
            definition="Confirmed supplier inbound by SKU; customer in_way_to/from_client is excluded.",
            source_agent="supply",
        ),
    )
    _append_fact(
        facts,
        _fact(
            metric="cogs",
            value=_number(row.get("unit_cogs_rub")),
            unit="RUB/unit",
            target=nm_id,
            evidence_ref=ref("cogs"),
            source_table="finance.wb_product_coast",
            snapshot_at=row.get("cogs_snapshot_date"),
            retrieved_at=row.get("cogs_retrieved_at") or retrieved_at,
            period=_period(row.get("cogs_snapshot_date")),
            period_end=row.get("cogs_snapshot_date"),
            definition="Latest source unit cost for the SKU.",
            source_agent="finance",
        ),
    )
    _append_fact(
        facts,
        _fact(
            metric="legacy_reorder_qty_preview",
            value=_number(row.get("reorder_qty_recommended")),
            unit="units",
            target=nm_id,
            evidence_ref=ref("legacy_reorder_qty_preview"),
            source_table="derived:get_supply_overview",
            snapshot_at=report_date,
            retrieved_at=retrieved_at,
            period=_period(report_date),
            period_end=report_date,
            definition="Legacy preview based on fixed 3/8-week cover; it is not an executable replenishment policy.",
            source_agent="supply",
        ),
    )
    return facts


def _build_supply_candidate(
    row: Mapping[str, Any],
    *,
    now: datetime,
    retrieved_at: Any,
) -> dict[str, Any] | None:
    nm_value = _integer(row.get("nm_id"))
    preview_qty = _number(row.get("reorder_qty_recommended"))
    if nm_value is None or preview_qty is None or preview_qty <= 0:
        return None
    nm_id = str(nm_value)
    blockers: list[dict[str, str]] = [
        _blocker(
            "legacy_reorder_formula_not_action_safe",
            "Количество рассчитано по старому правилу покрытия, без срока поставки и утверждённой политики.",
            severity="blocked",
        ),
        _blocker(
            "missing_sales_period_bounds",
            "В снимке нет точных дат начала и конца семидневного окна продаж.",
        ),
    ]
    if _number(row.get("stock_qty")) is None:
        blockers.append(_blocker("missing_stock_on_hand", "Нет доступного on-hand остатка SKU."))
    if _number(row.get("avg_daily_sales_7d")) is None:
        blockers.append(_blocker("missing_sales_velocity", "Нет скорости продаж SKU."))
    if _number(row.get("supplier_inbound_qty")) is None:
        blockers.append(
            _blocker(
                "missing_confirmed_supplier_inbound",
                "Нет подтверждённой поставки поставщика по SKU; клиентский транзит поставкой не считается.",
            )
        )
    if _number(row.get("lead_time_days")) is None:
        blockers.append(_blocker("missing_supplier_lead_time", "Нет подтверждённого срока поставки поставщика."))
    if _number(row.get("target_cover_days")) is None:
        blockers.append(_blocker("missing_target_cover_policy", "Нет утверждённого целевого покрытия в днях."))
    if _number(row.get("minimum_order_qty")) is None:
        blockers.append(_blocker("missing_minimum_order_quantity", "Нет MOQ/кратности заказа поставщика."))
    if not _text(row.get("replenishment_policy_version")):
        blockers.append(_blocker("missing_replenishment_policy", "Нет версии утверждённой политики пополнения."))
    if _number(row.get("unit_cogs_rub")) is None:
        blockers.append(_blocker("missing_cogs", "Нет себестоимости SKU; прибыль заявлять нельзя."))

    date_fields: list[tuple[str, Any]] = [("stock_on_hand", row.get("report_date"))]
    if _number(row.get("supplier_inbound_qty")) is not None:
        date_fields.append(("supplier_inbound", row.get("supplier_snapshot_date")))
        if _datetime(row.get("supplier_snapshot_date")) is None:
            blockers.append(_blocker("missing_supplier_snapshot_date", "Нет даты поставки поставщика."))
    if _number(row.get("unit_cogs_rub")) is not None:
        date_fields.append(("cogs", row.get("cogs_snapshot_date")))
        if _datetime(row.get("cogs_snapshot_date")) is None:
            blockers.append(_blocker("missing_cogs_snapshot_date", "Нет даты снимка себестоимости."))
    blockers.extend(_date_blockers(date_fields, now=now))
    blockers.append(
        _blocker(
            "missing_deterministic_economics",
            "Нет unit contribution и проверяемого диапазона эффекта поставки.",
        )
    )

    facts = _supply_evidence(row, nm_id=nm_id, retrieved_at=retrieved_at)
    candidate_id = f"det-supply-task-{nm_id}"
    return {
        "actionId": candidate_id,
        "family": "supply",
        "target": nm_id,
        "targetType": "sku",
        "targetRefs": [value for value in [_text(row.get("vendor_code"))] if value],
        "operation": "create_replenishment_task",
        "executionMode": "manual_task",
        "executionPayload": {
            "nm_id": nm_value,
            "vendor_code": _text(row.get("vendor_code")),
            "quantity": preview_qty,
            "quantity_source": "legacy_supply_overview_preview",
            "requires_human_confirmation": True,
        },
        "from": {"value": 0, "unit": "units_ordered"},
        "to": {"value": preview_qty, "unit": "units_ordered"},
        "evidence": facts,
        "evidenceRefs": [fact["evidenceRef"] for fact in facts],
        "candidateBlockers": _unique_blockers(blockers),
        "expiresAt": (now + _SUPPLY_PREVIEW_TTL).isoformat(),
        "urgency": "critical" if row.get("reorder_flag") == "critical" else "high",
        "confidence": "low",
    }


def _supply_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    flag_order = {"critical": 0, "warning": 1}
    days = _number(row.get("days_of_stock"))
    quantity = _number(row.get("reorder_qty_recommended"))
    return (
        flag_order.get(str(row.get("reorder_flag") or "").lower(), 2),
        float(days) if days is not None else math.inf,
        -float(quantity or 0),
        str(row.get("nm_id") or ""),
    )


def _candidate_aliases(
    candidate: Mapping[str, Any],
    *,
    identity_aliases: Mapping[str, set[str]],
) -> set[str]:
    aliases = {
        str(value).strip()
        for value in [
            candidate.get("target"),
            *(candidate.get("targetRefs") or []),
            (candidate.get("executionPayload") or {}).get("nm_id"),
            (candidate.get("executionPayload") or {}).get("vendor_code"),
        ]
        if value is not None and str(value).strip()
    }
    expanded = set(aliases)
    for alias in aliases:
        expanded.update(identity_aliases.get(alias, set()))
    return expanded


def _identity_aliases(
    advertising_rows: Sequence[Mapping[str, Any]],
    supply_rows: Sequence[Mapping[str, Any]],
) -> dict[str, set[str]]:
    groups: list[set[str]] = []
    for row in [*advertising_rows, *supply_rows]:
        aliases = {
            str(value).strip()
            for value in (row.get("nm_id"), row.get("vendor_code"))
            if value is not None and str(value).strip()
        }
        if aliases:
            groups.append(aliases)
    result: dict[str, set[str]] = {}
    for group in groups:
        merged = set(group)
        for alias in list(group):
            merged.update(result.get(alias, set()))
        for alias in merged:
            result.setdefault(alias, set()).update(merged)
    return result


def _target_candidates(
    candidates: Sequence[dict[str, Any]],
    *,
    requested_targets: Sequence[str],
    identity_aliases: Mapping[str, set[str]],
) -> list[dict[str, Any]]:
    normalized = {
        str(target).strip()
        for target in requested_targets
        if str(target).strip()
    }
    expanded = set(normalized)
    for target in normalized:
        expanded.update(identity_aliases.get(target, set()))
    if not expanded:
        return list(candidates)
    return [
        candidate
        for candidate in candidates
        if _candidate_aliases(candidate, identity_aliases=identity_aliases) & expanded
    ]


def _balanced_domain_slice(
    candidates: Sequence[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    family_order = list(
        dict.fromkeys(str(candidate.get("family") or "unknown") for candidate in candidates)
    )
    if len(family_order) <= 1:
        return list(candidates[:limit])
    queues = {
        family: [
            candidate
            for candidate in candidates
            if str(candidate.get("family") or "unknown") == family
        ]
        for family in family_order
    }
    selected: list[dict[str, Any]] = []
    while len(selected) < limit and any(queues.values()):
        for family in family_order:
            queue = queues[family]
            if queue and len(selected) < limit:
                selected.append(queue.pop(0))
    return selected


def build_manager_action_candidates(
    snapshot: Mapping[str, Any],
    *,
    now: datetime | None = None,
    max_candidates: int = 7,
    requested_targets: Sequence[str] | None = None,
) -> CandidateBuildResultV1:
    """Build exact candidates from source rows without LLM output or assumptions.

    ``now`` is an explicit input. When omitted it is derived from the snapshot's
    ``retrievedAt`` so the same snapshot produces the same result.
    """

    effective_now = _datetime(now) or _datetime(snapshot.get("retrievedAt"))
    if effective_now is None:
        raise ValueError("now or snapshot.retrievedAt is required")
    effective_now = _utc(effective_now)
    bounded_limit = max(1, min(int(max_candidates), 7))
    retrieved_at = snapshot.get("retrievedAt") or effective_now.isoformat()

    raw_ad_rows = snapshot.get("advertisingRows") or []
    ad_rows = [dict(row) for row in raw_ad_rows if isinstance(row, Mapping)]
    raw_supply_rows = snapshot.get("supplyRows") or []
    supply_rows = [dict(row) for row in raw_supply_rows if isinstance(row, Mapping)]

    ad_candidates = [
        candidate
        for group in _advertising_groups(ad_rows)
        if (candidate := _build_ad_candidate(
            group,
            now=effective_now,
            retrieved_at=retrieved_at,
        ))
        is not None
    ]
    supply_candidates = [
        candidate
        for row in sorted(supply_rows, key=_supply_sort_key)
        if (candidate := _build_supply_candidate(
            row,
            now=effective_now,
            retrieved_at=retrieved_at,
        ))
        is not None
    ]
    identity_aliases = _identity_aliases(ad_rows, supply_rows)
    all_candidates = _target_candidates(
        [*ad_candidates, *supply_candidates],
        requested_targets=list(requested_targets or []),
        identity_aliases=identity_aliases,
    )
    if not _simulation_evidence_enabled():
        candidates = _balanced_domain_slice(all_candidates, limit=bounded_limit)
    else:
        ready_candidates = [
            candidate for candidate in all_candidates if not candidate.get("candidateBlockers")
        ]
        rejected_candidates = [
            candidate for candidate in all_candidates if candidate.get("candidateBlockers")
        ]
        # A rejected high-spend preview is useful for diagnostics, but it must never
        # consume one of the seven slots before an independently executable action.
        selected_ready = _balanced_domain_slice(
            ready_candidates,
            limit=bounded_limit,
        )
        selected_rejected = _balanced_domain_slice(
            rejected_candidates,
            limit=bounded_limit - len(selected_ready),
        )
        candidates = [*selected_ready, *selected_rejected]

    facts_by_ref: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        for fact in candidate.get("evidence") or []:
            evidence_ref = _text(fact.get("evidenceRef"))
            if evidence_ref:
                facts_by_ref[evidence_ref] = fact

    ready_count = sum(1 for candidate in candidates if not candidate.get("candidateBlockers"))
    diagnostics = [
        {
            "domain": "advertising",
            "sourceRowCount": len(ad_rows),
            "candidateCount": sum(1 for item in candidates if item.get("family") == "advertising"),
        },
        {
            "domain": "supply",
            "sourceRowCount": len(supply_rows),
            "candidateCount": sum(1 for item in candidates if item.get("family") == "supply"),
        },
        {
            "candidateCount": len(candidates),
            "readyCandidateCount": ready_count,
            "blockedCandidateCount": len(candidates) - ready_count,
            "maxCandidates": bounded_limit,
            "requestedTargets": [
                str(target).strip()
                for target in (requested_targets or [])
                if str(target).strip()
            ],
        },
    ]
    return CandidateBuildResultV1(
        candidates=candidates,
        evidence_facts=list(facts_by_ref.values()),
        diagnostics=diagnostics,
    )


__all__ = ["CandidateBuildResultV1", "build_manager_action_candidates"]
