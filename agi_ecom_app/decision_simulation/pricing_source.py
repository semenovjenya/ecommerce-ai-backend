from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from hashlib import sha256
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

from agi_ecom_app.decision_simulation.pricing_models import (
    PricingAvailableStockEvidenceV1,
    PricingCandidateV1,
    PricingCompletedDayObservationV1,
    PricingContextFeatureEvidenceV1,
    PricingDailyObservationsEvidenceV1,
    PricingElasticityEvidenceV1,
    PricingForecastContextEvidenceV1,
    PricingFutureInboundEventV1,
    PricingRealizationEvidenceV1,
    PricingSimulationEvidenceV1,
    PricingSimulationRequestV1,
    PricingUnitEconomicsEvidenceV1,
)
from agi_ecom_app.decision_simulation.pricing_engine import MAX_PRICE_MOVE_FRACTION
from agi_ecom_app.services.strategic_db import get_connection, serialize_rows


PRICE_DAILY_MART = "agent_marts.v_agent_price_sku_daily"
PRICING_SOURCE_SCHEMA_VERSION = "pricing-source-snapshot-v1"
MIN_SHADOW_OBSERVATION_DAYS = 28
MIN_PRICE_LEVELS = 3
MAX_OBSERVATION_DAYS = 56
# The mart contains completed calendar days that can be unusable for demand
# estimation (missing exact-day price, observed OOS, or incomplete orders).
# Read a wider bounded window, then keep the latest 56 eligible observations.
MAX_SOURCE_ROWS = 180
ELASTICITY_BOOTSTRAP_DRAWS = 200
REALIZATION_BOOTSTRAP_DRAWS = 200
MIN_REALIZATION_OBSERVATION_DAYS = 28
_PLACEHOLDER_SNAPSHOT_HASH = "0" * 64
MIN_CONTEXT_OBSERVATION_DAYS = 14
MIN_CONTEXT_COVERAGE_RATIO = 0.25
MAX_AD_FEATURE_CORRELATION = 0.88
CONTEXT_SOURCE_SCHEMA_VERSION = "pricing-context-controls-v1"


@dataclass(frozen=True)
class _ContextFeatureSpec:
    name: str
    transform: str
    minimum_days: int = MIN_CONTEXT_OBSERVATION_DAYS


@dataclass(frozen=True)
class _ContextualDemandBootstrap:
    elasticity_values: list[float]
    coefficient_values: dict[str, list[float]]
    centers: dict[str, float]
    scales: dict[str, float]
    coverage_days: dict[str, int]
    coverage_ratios: dict[str, float]
    active_features: tuple[str, ...]
    excluded_reasons: dict[str, str]


_CONTEXT_FEATURE_SPECS: tuple[_ContextFeatureSpec, ...] = (
    _ContextFeatureSpec("ad_spend_rub", "log1p"),
    _ContextFeatureSpec("ad_views", "log1p"),
    _ContextFeatureSpec("ad_clicks", "log1p"),
    _ContextFeatureSpec("lagged_organic_position", "log1p"),
    _ContextFeatureSpec("lagged_open_to_cart_conversion", "identity"),
    _ContextFeatureSpec("lagged_cart_to_order_conversion", "identity"),
    _ContextFeatureSpec("lagged_competitor_median_price_rub", "log1p", 28),
    _ContextFeatureSpec("promotion_active", "identity", 14),
    _ContextFeatureSpec("promotion_discount_pct", "identity", 14),
    _ContextFeatureSpec("annual_season_sin", "identity", 365),
    _ContextFeatureSpec("annual_season_cos", "identity", 365),
)


class PricingSimulationSourceError(RuntimeError):
    """The live analytics source cannot produce a reproducible price snapshot."""


def load_pricing_simulation_source(
    *,
    nm_id: int,
    candidate_prices: Sequence[float] | None = None,
    horizon_days: int = 7,
) -> tuple[PricingSimulationRequestV1, dict[str, Any]]:
    """Load current price plus completed daily evidence without changing state."""

    if int(nm_id) <= 0:
        raise ValueError("nm_id must be positive")
    if not 1 <= int(horizon_days) <= 90:
        raise ValueError("pricing simulation horizon must be between 1 and 90 days")

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            select distinct on (nm_id)
                nm_id,
                snapshot_date,
                discounted_price as seller_effective_price_rub,
                price as base_price_rub,
                discount as seller_discount_pct,
                last_synced_at
            from pricing.wb_supplier_prices
            where nm_id = %s
              and discounted_price > 0
            order by nm_id, snapshot_date desc, last_synced_at desc, id desc
            """,
            [int(nm_id)],
        )
        current_rows = serialize_rows(cur.fetchall())
        if not current_rows:
            raise PricingSimulationSourceError(
                f"Current seller-controlled price is missing for nm_id={nm_id}."
            )
        current = current_rows[0]

        try:
            cur.execute(
                f"""
                select
                    price_day.*,
                    case when price_day.ad_spend_complete
                        then price_day.ad_spend_rub
                    end as context_ad_spend_rub,
                    case when price_day.marketing_day_complete
                        then price_day.ad_views
                    end as context_ad_views,
                    case when price_day.marketing_day_complete
                        then price_day.ad_clicks
                    end as context_ad_clicks,
                    marketing_day.avg_position_current
                        as context_organic_position,
                    sales_day.open_to_cart_conversion
                        as context_open_to_cart_conversion,
                    sales_day.cart_to_order_conversion
                        as context_cart_to_order_conversion,
                    case
                        when price_context.query_count >= 3
                         and price_context.source_found_query_share >= 0.5
                         and price_context.organic_competitors_before_source >= 5
                         and price_context.source_to_before_median_price_ratio
                             between 0.5 and 2.0
                        then price_context.price_before_median
                    end as context_competitor_median_price_rub,
                    promotion_day.in_promotion as context_promotion_active,
                    promotion_day.promotion_discount_pct
                        as context_promotion_discount_pct
                from {PRICE_DAILY_MART} price_day
                left join agent_marts.v_agent_sales_sku_daily sales_day
                  on sales_day.report_date = price_day.report_date - 1
                 and sales_day.nm_id = price_day.nm_id
                left join agent_marts.v_agent_marketing_sku_daily marketing_day
                  on marketing_day.report_date = price_day.report_date - 1
                 and marketing_day.nm_id = price_day.nm_id
                left join analytics_ai.mart_price_context_daily price_context
                  on price_context.snapshot_date = price_day.report_date - 1
                 and price_context.source_nm_id = price_day.nm_id
                left join lateral (
                    select
                        bool_or(coalesce(promo.in_action, false)) as in_promotion,
                        max(coalesce(promo.plan_discount, promo.discount))
                            filter (where promo.in_action is true)
                            as promotion_discount_pct
                    from marketing.v_wb_calendar_promotions_nomenclatures_latest_daily promo
                    where promo.snapshot_date = price_day.report_date
                      and promo.nm_id = price_day.nm_id
                ) promotion_day on true
                where price_day.nm_id = %s
                  and price_day.report_date < %s
                order by price_day.report_date desc
                limit %s
                """,
                [
                    int(nm_id),
                    _as_date(current.get("snapshot_date")),
                    MAX_SOURCE_ROWS,
                ],
            )
        except Exception as exc:
            if "v_agent_price_sku_daily" in str(exc):
                raise PricingSimulationSourceError(
                    "Price daily mart is unavailable; apply agent_marts before starting pricing simulation."
                ) from exc
            raise
        history = serialize_rows(cur.fetchall())

        cur.execute(
            f"""
            with latest_supply_day as (
                select max(snapshot_date) as snapshot_date
                from supply.wb_supplies
                where snapshot_date <= least(
                    (%s::date - interval '1 day')::date,
                    (current_date - interval '1 day')::date
                )
            ),
            inbound_events as (
                select jsonb_agg(
                    jsonb_build_object(
                        'supply_id', supply.supply_id,
                        'eta_date', supply.supply_date,
                        'status_id', supply.status_id,
                        'fact_date', supply.fact_date,
                        'units', goods.remaining_units,
                        'snapshot_date', supply.snapshot_date,
                        'last_synced_at', greatest(
                            supply.last_synced_at,
                            goods.last_synced_at
                        )
                    )
                    order by supply.supply_date, supply.supply_id
                ) as events
                from supply.wb_supplies supply
                join lateral (
                    select
                        sum(greatest(
                            coalesce(item.quantity, 0)
                            - coalesce(item.ready_for_sale_quantity, 0),
                            0
                        ))::numeric as remaining_units,
                        max(item.last_synced_at) as last_synced_at
                    from supply.wb_supplies_goods item
                    where item.snapshot_date = supply.snapshot_date
                      and item.supply_id = supply.supply_id
                      and item.nm_id = %s
                ) goods on goods.remaining_units > 0
                where supply.snapshot_date = (select snapshot_date from latest_supply_day)
                  and supply.status_id in (2, 3)
                  and supply.fact_date is null
            ),
            latest_promotion_snapshot as (
                select max(snapshot_date) as snapshot_date
                from marketing.v_wb_calendar_promotions_nomenclatures_latest_snapshot
                where snapshot_date <= %s::date
            ),
            future_promotions as (
                select jsonb_agg(
                    jsonb_build_object(
                        'promotion_id', sku_promo.promotion_id,
                        'name', promo.name,
                        'start_date', promo.start_date_time::date,
                        'end_date', promo.end_date_time::date,
                        'in_action', sku_promo.in_action,
                        'discount_pct', coalesce(
                            sku_promo.plan_discount,
                            sku_promo.discount
                        ),
                        'plan_price_rub', sku_promo.plan_price,
                        'snapshot_date', sku_promo.snapshot_date,
                        'last_synced_at', greatest(
                            sku_promo.last_synced_at,
                            promo.last_synced_at
                        )
                    )
                    order by promo.start_date_time, sku_promo.promotion_id
                ) as events
                from marketing.v_wb_calendar_promotions_nomenclatures_latest_snapshot sku_promo
                join lateral (
                    select promo.*
                    from marketing.v_wb_calendar_promotions_latest_snapshot promo
                    where promo.promotion_id = sku_promo.promotion_id
                      and promo.snapshot_date <= %s::date
                    order by promo.snapshot_date desc
                    limit 1
                ) promo on true
                where sku_promo.snapshot_date = (
                    select snapshot_date from latest_promotion_snapshot
                )
                  and sku_promo.nm_id = %s
                  and sku_promo.in_action is true
                  and sku_promo.snapshot_date >= (
                      %s::date - interval '7 days'
                  )::date
                  and promo.end_date_time::date >= %s::date
            ),
            current_market_context as (
                select (
                    select to_jsonb(context_day)
                    from analytics_ai.mart_price_context_daily context_day
                    where context_day.source_nm_id = %s
                      and context_day.snapshot_date <= %s::date
                    order by context_day.snapshot_date desc
                    limit 1
                ) as context
            ),
            current_market_history as (
                select coalesce(
                    jsonb_agg(to_jsonb(history_day) order by history_day.snapshot_date desc),
                    '[]'::jsonb
                ) as history
                from (
                    select context_day.*
                    from analytics_ai.mart_price_context_daily context_day
                    where context_day.source_nm_id = %s
                      and context_day.snapshot_date <= %s::date
                    order by context_day.snapshot_date desc
                    limit 7
                ) history_day
            )
            select
                price_day.report_date as snapshot_date,
                price_day.stock_on_hand as available_stock,
                price_day.stock_last_synced_at as last_synced_at,
                price_day.stock_source_row_count,
                price_day.stock_revisions_dropped,
                inbound_events.events as future_inbound_events,
                future_promotions.events as future_promotion_events,
                current_market_context.context as current_market_context,
                current_market_history.history as market_context_history
            from {PRICE_DAILY_MART} price_day
            cross join inbound_events
            cross join future_promotions
            cross join current_market_context
            cross join current_market_history
            where price_day.nm_id = %s
              and price_day.stock_snapshot_present
              and price_day.report_date <= %s::date
            order by price_day.report_date desc
            limit 1
            """,
            [
                _as_date(current.get("snapshot_date")),
                int(nm_id),
                _as_date(current.get("snapshot_date")),
                _as_date(current.get("snapshot_date")),
                int(nm_id),
                _as_date(current.get("snapshot_date")),
                _as_date(current.get("snapshot_date")),
                int(nm_id),
                _as_date(current.get("snapshot_date")),
                int(nm_id),
                _as_date(current.get("snapshot_date")),
                int(nm_id),
                _as_date(current.get("snapshot_date")),
            ],
        )
        stock_rows = serialize_rows(cur.fetchall())

    request = build_pricing_simulation_request(
        nm_id=int(nm_id),
        current_price_row=current,
        history_rows=history,
        stock_row=stock_rows[0] if stock_rows else None,
        candidate_prices=candidate_prices,
        horizon_days=horizon_days,
    )
    metadata = {
        "schemaVersion": PRICING_SOURCE_SCHEMA_VERSION,
        "sourceTable": PRICE_DAILY_MART,
        "nmId": int(nm_id),
        "currentPriceSnapshot": current,
        "stockSnapshot": stock_rows[0] if stock_rows else None,
        "currentMarketContext": (
            (stock_rows[0] if stock_rows else {}).get("current_market_context")
        ),
        "marketContextHistory": _json_list(
            (stock_rows[0] if stock_rows else {}).get("market_context_history")
        ),
        "futurePromotionEvents": _json_list(
            (stock_rows[0] if stock_rows else {}).get("future_promotion_events")
        ),
        "historyRowCount": len(history),
        "historyPeriodStart": min(
            (_iso_date(row.get("report_date")) for row in history),
            default=None,
        ),
        "historyPeriodEnd": max(
            (_iso_date(row.get("report_date")) for row in history),
            default=None,
        ),
        "horizonDays": int(horizon_days),
        "shadowOnly": True,
        "causalValid": False,
        "profitEvidenceTier": "observed_unwatermarked_shadow",
        **_context_source_metadata(request),
    }
    return request, metadata


def build_pricing_simulation_request(
    *,
    nm_id: int,
    current_price_row: Mapping[str, Any],
    history_rows: Sequence[Mapping[str, Any]],
    stock_row: Mapping[str, Any] | None,
    candidate_prices: Sequence[float] | None = None,
    horizon_days: int = 7,
    snapshot_hash: str = _PLACEHOLDER_SNAPSHOT_HASH,
) -> PricingSimulationRequestV1:
    current_price = _positive_number(
        current_price_row.get("seller_effective_price_rub")
    )
    if current_price is None:
        raise PricingSimulationSourceError("Current seller effective price is missing.")
    try:
        snapshot_date = _as_date(current_price_row.get("snapshot_date"))
    except (TypeError, ValueError):
        raise PricingSimulationSourceError(
            "Current seller price snapshot date is missing."
        ) from None

    eligible_rows = _eligible_demand_rows(history_rows)
    observations = [
        PricingCompletedDayObservationV1(
            observedDate=_as_date(row.get("report_date")),
            priceRub=float(row["seller_effective_price_rub"]),
            unitsSold=float(row["non_cancelled_order_units"]),
            completed=True,
            notObservedOutOfStock=True,
            contextValues=_context_values_from_row(row),
            evidenceRef=(
                f"{PRICE_DAILY_MART}:{nm_id}:{_iso_date(row.get('report_date'))}"
            ),
        )
        for row in eligible_rows
    ]
    contextual_fit = estimate_contextual_demand_bootstrap(observations)
    elasticity_values = contextual_fit.elasticity_values
    elasticity_ref = _period_evidence_ref(
        "pricing-elasticity-bootstrap",
        nm_id,
        eligible_rows,
    )
    realization_rows = _eligible_realization_rows(history_rows)
    realization_values = estimate_empirical_realization_bootstrap(
        realization_rows
    )
    realization_ref = _period_evidence_ref(
        "pricing-order-realization-bootstrap",
        nm_id,
        realization_rows,
    )
    economics = _estimate_shadow_unit_economics(
        history_rows,
        snapshot_date=snapshot_date,
    )
    economics_ref = _period_evidence_ref(
        "pricing-unit-economics-shadow",
        nm_id,
        economics["rows"],
    )
    stock_units = _nonnegative_number(
        (stock_row or {}).get("available_stock")
    )
    stock_snapshot_date = None
    stock_age_days = None
    stock_ref = None
    if stock_row:
        try:
            stock_snapshot_date = _as_date(stock_row.get("snapshot_date"))
            stock_age_days = _age_days(snapshot_date, stock_snapshot_date)
        except (TypeError, ValueError):
            stock_snapshot_date = None
        stock_ref = (
            f"{PRICE_DAILY_MART}:stock:"
            f"{nm_id}:{_iso_date(stock_row.get('snapshot_date'))}"
        )
    current_ref = (
        "pricing.wb_supplier_prices:"
        f"{nm_id}:{_iso_date(current_price_row.get('snapshot_date'))}"
    )
    future_inbound = _future_inbound_events(
        nm_id=nm_id,
        snapshot_date=snapshot_date,
        horizon_days=horizon_days,
        stock_row=stock_row or {},
    )
    forecast_context = _build_forecast_context(
        nm_id=nm_id,
        snapshot_date=snapshot_date,
        horizon_days=horizon_days,
        observations=observations,
        contextual_fit=contextual_fit,
        stock_row=stock_row or {},
        future_inbound=future_inbound,
    )
    candidates = build_pricing_candidates(
        nm_id=nm_id,
        current_price_rub=current_price,
        candidate_prices=candidate_prices,
    )
    return PricingSimulationRequestV1(
        snapshotHash=snapshot_hash,
        decisionGroupId=f"pricing:sku:{nm_id}",
        skuId=str(nm_id),
        currentPriceRub=current_price,
        currentPriceEvidenceRef=current_ref,
        horizonDays=horizon_days,
        candidates=candidates,
        evidence=PricingSimulationEvidenceV1(
            dailyObservations=PricingDailyObservationsEvidenceV1(
                values=observations,
                latestObservationAgeDays=(
                    _age_days(snapshot_date, observations[-1].observed_date)
                    if observations
                    else None
                ),
                evidenceRef=_period_evidence_ref(
                    "pricing-price-volume-history",
                    nm_id,
                    eligible_rows,
                ),
            ),
            elasticity=PricingElasticityEvidenceV1(
                bootstrapValues=elasticity_values or None,
                source=(
                    "historical_price_volume_residual_bootstrap_v1"
                    if elasticity_values
                    else None
                ),
                evidenceRef=elasticity_ref,
            ),
            realization=PricingRealizationEvidenceV1(
                bootstrapValues=realization_values or None,
                source=(
                    "cohort_order_to_realized_sale_day_bootstrap_v1"
                    if realization_values
                    else None
                ),
                evidenceRef=realization_ref,
            ),
            unitEconomics=PricingUnitEconomicsEvidenceV1(
                netRevenueRate=economics["net_revenue_rate"],
                variableCostPerUnitRub=economics["variable_cost_per_unit_rub"],
                minimumUnitContributionRub=0,
                observationCount=len(economics["rows"]),
                latestObservationAgeDays=economics["latest_observation_age_days"],
                basis="wb_net_payout_before_cogs",
                evidenceRef=economics_ref,
            ),
            availableStock=PricingAvailableStockEvidenceV1(
                units=stock_units,
                snapshotDate=stock_snapshot_date,
                ageDaysAtSnapshot=stock_age_days,
                evidenceRef=stock_ref,
                futureInbound=future_inbound,
            ),
            forecastContext=forecast_context,
        ),
    )


def build_pricing_candidates(
    *,
    nm_id: int,
    current_price_rub: float,
    candidate_prices: Sequence[float] | None = None,
) -> list[PricingCandidateV1]:
    current = float(current_price_rub)
    if not math.isfinite(current) or current <= 0:
        raise ValueError("current_price_rub must be a positive finite number")
    if candidate_prices is None:
        # Keep small-price grids from collapsing after rounding while using the
        # operational five-ruble step for the usual WB price range.
        step = 5.0 if current >= 200 else 1.0
        generated = [
            _round_price(current * factor, step)
            for factor in (0.95, 0.975, 1.025, 1.05)
        ]
    else:
        generated = [float(value) for value in candidate_prices]
    prices: list[float] = [current]
    for price in generated:
        if not math.isfinite(price) or price <= 0:
            raise ValueError("candidate prices must be positive finite numbers")
        if abs(price / current - 1) > MAX_PRICE_MOVE_FRACTION + 1e-12:
            raise ValueError(
                "pricing v1 candidates must stay within "
                f"+/-{MAX_PRICE_MOVE_FRACTION:.0%} of current price"
            )
        if not any(math.isclose(price, item, abs_tol=1e-9) for item in prices):
            prices.append(float(price))
    if not 3 <= len(prices) <= 7:
        raise ValueError("pricing simulation requires 3-7 unique prices including hold")

    def stable_price(value: float) -> str:
        return (f"{value:.4f}").rstrip("0").rstrip(".").replace(".", "_")

    return [
        PricingCandidateV1(
            candidateId=f"pricing:{nm_id}:{stable_price(price)}",
            actionId=(
                f"pricing:{nm_id}:hold"
                if index == 0
                else f"pricing:{nm_id}:set:{stable_price(price)}"
            ),
            operation="hold_price" if index == 0 else "set_price",
            priceRub=price,
        )
        for index, price in enumerate(prices)
    ]


def estimate_empirical_elasticity_bootstrap(
    observations: Sequence[PricingCompletedDayObservationV1],
    *,
    draws: int = ELASTICITY_BOOTSTRAP_DRAWS,
) -> list[float]:
    """Residual bootstrap of a transparent observational log-demand model.

    Positive draws are deliberately retained. The pricing hard gate rejects a
    distribution that crosses zero, so a noisy or confounded history cannot be
    made executable by filtering inconvenient bootstrap outcomes.
    """

    return estimate_contextual_demand_bootstrap(
        observations,
        draws=draws,
    ).elasticity_values


def estimate_contextual_demand_bootstrap(
    observations: Sequence[PricingCompletedDayObservationV1],
    *,
    draws: int = ELASTICITY_BOOTSTRAP_DRAWS,
) -> _ContextualDemandBootstrap:
    """Fit price elasticity with only sufficiently covered point-in-time controls.

    Missing contextual values are median-imputed for fitting, but their missingness
    remains visible in feature coverage. Features with thin history, zero variance,
    or a numerically dependent design column stay out of the model.
    """

    empty = _ContextualDemandBootstrap([], {}, {}, {}, {}, {}, (), {})
    values = sorted(observations, key=lambda item: item.observed_date)
    if len(values) < MIN_SHADOW_OBSERVATION_DAYS:
        return empty
    if len({round(item.price_rub, 6) for item in values}) < MIN_PRICE_LEVELS:
        return empty
    if draws < 3:
        raise ValueError("elasticity bootstrap requires at least three draws")

    prices = np.asarray([item.price_rub for item in values], dtype=np.float64)
    units = np.asarray([item.units_sold for item in values], dtype=np.float64)
    if bool(np.any(prices <= 0)) or not bool(np.any(units > 0)):
        return empty
    trend = np.linspace(-1.0, 1.0, num=len(values), dtype=np.float64)
    weekdays = np.asarray(
        [item.observed_date.weekday() for item in values], dtype=np.float64
    )
    design = np.column_stack(
        (
            np.ones(len(values), dtype=np.float64),
            np.log(prices),
            trend,
            np.sin(2 * np.pi * weekdays / 7),
            np.cos(2 * np.pi * weekdays / 7),
        )
    )
    if np.linalg.matrix_rank(design) < design.shape[1]:
        return empty

    coverage_days: dict[str, int] = {}
    coverage_ratios: dict[str, float] = {}
    centers: dict[str, float] = {}
    scales: dict[str, float] = {}
    active_features: list[str] = []
    raw_by_name: dict[str, list[float | None]] = {}
    excluded_reasons: dict[str, str] = {}
    for spec in _CONTEXT_FEATURE_SPECS:
        raw = [
            _context_value(item.context_values, spec.name)
            for item in values
        ]
        raw_by_name[spec.name] = raw
        observed = [item for item in raw if item is not None]
        coverage_days[spec.name] = len(observed)
        coverage_ratios[spec.name] = len(observed) / len(values)
        if (
            len(observed) < spec.minimum_days
            or coverage_ratios[spec.name] < MIN_CONTEXT_COVERAGE_RATIO
        ):
            excluded_reasons[spec.name] = "insufficient_history"
            continue
        transformed_observed = np.asarray(
            [_transform_context_value(item, spec.transform) for item in observed],
            dtype=np.float64,
        )
        center = float(np.median(transformed_observed))
        scale = float(np.std(transformed_observed))
        if not math.isfinite(center) or not math.isfinite(scale) or scale <= 1e-9:
            excluded_reasons[spec.name] = "zero_variance"
            continue
        if spec.name in {"ad_spend_rub", "ad_views", "ad_clicks"}:
            correlated_with = next(
                (
                    active_name
                    for active_name in active_features
                    if active_name in {"ad_spend_rub", "ad_views", "ad_clicks"}
                    and _paired_feature_correlation(
                        raw,
                        raw_by_name[active_name],
                        transform=spec.transform,
                    ) >= MAX_AD_FEATURE_CORRELATION
                ),
                None,
            )
            if correlated_with is not None:
                excluded_reasons[spec.name] = f"collinear_with_{correlated_with}"
                continue
        transformed = np.asarray(
            [
                center
                if item is None
                else _transform_context_value(item, spec.transform)
                for item in raw
            ],
            dtype=np.float64,
        )
        column = ((transformed - center) / scale)[:, np.newaxis]
        candidate_design = np.column_stack((design, column))
        if np.linalg.matrix_rank(candidate_design) < candidate_design.shape[1]:
            excluded_reasons[spec.name] = "rank_dependent"
            continue
        if not math.isfinite(float(np.linalg.cond(candidate_design))):
            excluded_reasons[spec.name] = "ill_conditioned"
            continue
        if float(np.linalg.cond(candidate_design)) > 1e8:
            excluded_reasons[spec.name] = "ill_conditioned"
            continue
        design = candidate_design
        active_features.append(spec.name)
        centers[spec.name] = center
        scales[spec.name] = scale

    target = np.log(units + 0.5)
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    fitted = design @ coefficients
    residuals = target - fitted
    residuals -= float(np.mean(residuals))
    if not bool(np.all(np.isfinite(residuals))):
        return empty
    has_context = bool(active_features)
    canonical = json.dumps(
        [
            (
                [
                    item.observed_date.isoformat(),
                    item.price_rub,
                    item.units_sold,
                    {
                        name: _context_value(item.context_values, name)
                        for name in active_features
                    },
                ]
                if has_context
                else [item.observed_date.isoformat(), item.price_rub, item.units_sold]
            )
            for item in values
        ],
        separators=(",", ":"),
    )
    seed = int.from_bytes(
        sha256(canonical.encode("utf-8")).digest()[:8],
        byteorder="big",
        signed=False,
    )
    generator = np.random.Generator(np.random.PCG64(seed))
    result: list[float] = []
    coefficient_values: dict[str, list[float]] = {
        name: [] for name in active_features
    }
    for _ in range(draws):
        sampled = residuals[generator.integers(0, len(residuals), size=len(residuals))]
        bootstrap_target = fitted + sampled
        bootstrap_coefficients, *_ = np.linalg.lstsq(
            design,
            bootstrap_target,
            rcond=None,
        )
        elasticity = float(bootstrap_coefficients[1])
        if math.isfinite(elasticity):
            result.append(elasticity)
            for index, name in enumerate(active_features, start=5):
                coefficient_values[name].append(float(bootstrap_coefficients[index]))
    if len(result) < 3 or any(
        len(items) != len(result) for items in coefficient_values.values()
    ):
        return empty
    return _ContextualDemandBootstrap(
        elasticity_values=result,
        coefficient_values=coefficient_values,
        centers=centers,
        scales=scales,
        coverage_days=coverage_days,
        coverage_ratios=coverage_ratios,
        active_features=tuple(active_features),
        excluded_reasons=excluded_reasons,
    )


def _paired_feature_correlation(
    left: Sequence[float | None],
    right: Sequence[float | None],
    *,
    transform: str,
) -> float:
    paired = [
        (
            _transform_context_value(float(left_value), transform),
            _transform_context_value(float(right_value), transform),
        )
        for left_value, right_value in zip(left, right)
        if left_value is not None and right_value is not None
    ]
    if len(paired) < MIN_CONTEXT_OBSERVATION_DAYS:
        return 0.0
    values = np.asarray(paired, dtype=np.float64)
    if float(np.std(values[:, 0])) <= 1e-9 or float(np.std(values[:, 1])) <= 1e-9:
        return 0.0
    correlation = float(np.corrcoef(values[:, 0], values[:, 1])[0, 1])
    return abs(correlation) if math.isfinite(correlation) else 0.0


def _context_values_from_row(row: Mapping[str, Any]) -> dict[str, float] | None:
    values: dict[str, float] = {}
    source_fields = {
        "ad_spend_rub": "context_ad_spend_rub",
        "ad_views": "context_ad_views",
        "ad_clicks": "context_ad_clicks",
        "lagged_organic_position": "context_organic_position",
        "lagged_open_to_cart_conversion": "context_open_to_cart_conversion",
        "lagged_cart_to_order_conversion": "context_cart_to_order_conversion",
        "lagged_competitor_median_price_rub": "context_competitor_median_price_rub",
        "promotion_discount_pct": "context_promotion_discount_pct",
    }
    for name, field in source_fields.items():
        value = _number(row.get(field))
        if value is None or value < 0:
            continue
        if name in {"lagged_organic_position", "lagged_competitor_median_price_rub"} and value <= 0:
            continue
        values[name] = value
    promotion = row.get("context_promotion_active")
    if isinstance(promotion, bool):
        values["promotion_active"] = 1.0 if promotion else 0.0
    try:
        observed_date = _as_date(row.get("report_date"))
    except (TypeError, ValueError):
        observed_date = None
    if observed_date is not None:
        angle = 2 * math.pi * observed_date.timetuple().tm_yday / 365.2425
        values["annual_season_sin"] = math.sin(angle)
        values["annual_season_cos"] = math.cos(angle)
    return values or None


def _context_value(
    values: Mapping[str, float] | None,
    name: str,
) -> float | None:
    if not values:
        return None
    return _number(values.get(name))


def _transform_context_value(value: float, transform: str) -> float:
    if transform == "log1p":
        if value < 0:
            raise ValueError("log1p context feature cannot be negative")
        return math.log1p(value)
    if transform == "identity":
        return value
    raise ValueError(f"unsupported context transform: {transform}")


def _build_forecast_context(
    *,
    nm_id: int,
    snapshot_date: date,
    horizon_days: int,
    observations: Sequence[PricingCompletedDayObservationV1],
    contextual_fit: _ContextualDemandBootstrap,
    stock_row: Mapping[str, Any],
    future_inbound: Sequence[PricingFutureInboundEventV1],
) -> PricingForecastContextEvidenceV1:
    specs = {item.name: item for item in _CONTEXT_FEATURE_SPECS}
    promotion_active, promotion_discount = _future_promotion_paths(
        snapshot_date=snapshot_date,
        horizon_days=horizon_days,
        raw_events=stock_row.get("future_promotion_events"),
    )
    features: list[PricingContextFeatureEvidenceV1] = []
    data_gaps: list[str] = []
    for name, spec in specs.items():
        raw_values = [
            _context_value(item.context_values, name)
            for item in observations
        ]
        observed_values = [item for item in raw_values if item is not None]
        observed_days = len(observed_values)
        current_value = _current_market_context_value(name, stock_row)
        coverage_ratio = (
            observed_days / len(observations) if observations else 0.0
        )
        if name in contextual_fit.active_features:
            status = "modeled"
            gap = None
        elif observed_days or current_value is not None:
            status = "observed_only"
            reason = contextual_fit.excluded_reasons.get(
                name,
                "not_modeled",
            )
            gap = f"context_{name}_{reason}"
            data_gaps.append(gap)
        else:
            status = "missing"
            gap = f"context_{name}_missing"
            data_gaps.append(gap)

        if name == "promotion_active":
            forward = promotion_active
        elif name == "promotion_discount_pct":
            forward = promotion_discount
        elif name in {"annual_season_sin", "annual_season_cos"}:
            forward = []
            for offset in range(1, horizon_days + 1):
                forecast_date = snapshot_date + timedelta(days=offset)
                angle = 2 * math.pi * forecast_date.timetuple().tm_yday / 365.2425
                forward.append(
                    math.sin(angle)
                    if name.endswith("sin")
                    else math.cos(angle)
                )
        else:
            forward = (
                [current_value] * horizon_days
                if current_value is not None
                else [
                    _weekday_context_baseline(
                        observations,
                        name=name,
                        forecast_date=snapshot_date + timedelta(days=offset),
                    )
                    for offset in range(1, horizon_days + 1)
                ]
            )

        if status == "modeled" and (
            len(forward) != horizon_days
            or any(value is None for value in forward)
        ):
            status = "observed_only"
            gap = f"context_{name}_forecast_path_missing"
            data_gaps.append(gap)

        latest = current_value
        if latest is None:
            latest = next(
                (
                    value
                    for value in reversed(raw_values)
                    if value is not None
                ),
                None,
            )
        features.append(
            PricingContextFeatureEvidenceV1(
                name=name,
                transform=spec.transform,
                status=status,
                observedDays=observed_days,
                coverageRatio=coverage_ratio,
                center=(
                    contextual_fit.centers.get(name)
                    if status == "modeled"
                    else None
                ),
                scale=(
                    contextual_fit.scales.get(name)
                    if status == "modeled"
                    else None
                ),
                bootstrapCoefficients=(
                    contextual_fit.coefficient_values.get(name)
                    if status == "modeled"
                    else None
                ),
                forwardValues=forward,
                latestValue=latest,
                evidenceRef=(
                    _context_evidence_ref(name, nm_id, observations)
                    or _current_market_context_evidence_ref(
                        name,
                        nm_id=nm_id,
                        stock_row=stock_row,
                    )
                ),
                dataGap=gap,
            )
        )

    if not any(
        event.include_in_capacity and event.status == "scheduled"
        for event in future_inbound
    ):
        data_gaps.append("future_supplier_inbound_not_confirmed")
    return PricingForecastContextEvidenceV1(
        snapshotDate=snapshot_date,
        features=features,
        dataGaps=list(dict.fromkeys(data_gaps)),
        shadowOnly=True,
    )


def _current_market_context_value(
    name: str,
    stock_row: Mapping[str, Any],
) -> float | None:
    raw_context = stock_row.get("current_market_context")
    if isinstance(raw_context, str):
        try:
            raw_context = json.loads(raw_context)
        except json.JSONDecodeError:
            return None
    if not isinstance(raw_context, Mapping):
        return None
    if name == "lagged_organic_position":
        value = _positive_number(raw_context.get("weighted_organic_position"))
        found_share = _number(raw_context.get("source_found_query_share"))
        query_count = _number(raw_context.get("query_count"))
        if value is not None and (found_share or 0) >= 0.5 and (query_count or 0) >= 2:
            return value
    if name == "lagged_competitor_median_price_rub":
        value = _positive_number(raw_context.get("price_before_median"))
        found_share = _number(raw_context.get("source_found_query_share"))
        query_count = _number(raw_context.get("query_count"))
        ratio = _number(raw_context.get("source_to_before_median_price_ratio"))
        competitor_count = _number(
            raw_context.get("organic_competitors_before_source")
        )
        if (
            value is not None
            and (found_share or 0) >= 0.5
            and (query_count or 0) >= 3
            and (competitor_count or 0) >= 5
            and ratio is not None
            and 0.5 <= ratio <= 2.0
        ):
            return value
    return None


def _current_market_context_evidence_ref(
    name: str,
    *,
    nm_id: int,
    stock_row: Mapping[str, Any],
) -> str | None:
    if name not in {
        "lagged_organic_position",
        "lagged_competitor_median_price_rub",
    }:
        return None
    if _current_market_context_value(name, stock_row) is None:
        return None
    raw_context = stock_row.get("current_market_context")
    if isinstance(raw_context, str):
        try:
            raw_context = json.loads(raw_context)
        except json.JSONDecodeError:
            return None
    if not isinstance(raw_context, Mapping):
        return None
    return (
        "analytics_ai.mart_price_context_daily:"
        f"{nm_id}:{_iso_date(raw_context.get('snapshot_date')) or 'unknown'}"
    )


def _weekday_context_baseline(
    observations: Sequence[PricingCompletedDayObservationV1],
    *,
    name: str,
    forecast_date: date,
) -> float | None:
    recent = sorted(observations, key=lambda item: item.observed_date)[-28:]
    same_weekday = [
        value
        for item in recent
        if item.observed_date.weekday() == forecast_date.weekday()
        for value in [_context_value(item.context_values, name)]
        if value is not None
    ]
    all_recent = [
        value
        for item in recent
        for value in [_context_value(item.context_values, name)]
        if value is not None
    ]
    pool = same_weekday if len(same_weekday) >= 2 else all_recent
    return float(np.median(np.asarray(pool, dtype=np.float64))) if pool else None


def _future_promotion_paths(
    *,
    snapshot_date: date,
    horizon_days: int,
    raw_events: Any,
) -> tuple[list[float | None], list[float | None]]:
    valid_events: list[tuple[date, date, float | None]] = []
    for raw in _json_list(raw_events):
        if not isinstance(raw, Mapping) or raw.get("in_action") is not True:
            continue
        try:
            start = _as_date(raw.get("start_date"))
            end = _as_date(raw.get("end_date"))
        except (TypeError, ValueError):
            continue
        if end < start:
            continue
        valid_events.append(
            (start, end, _nonnegative_number(raw.get("discount_pct")))
        )
    if not valid_events:
        return [None] * horizon_days, [None] * horizon_days
    active: list[float | None] = [0.0] * horizon_days
    discount: list[float | None] = [0.0] * horizon_days
    for start, end, event_discount in valid_events:
        for index in range(horizon_days):
            forecast_date = snapshot_date + timedelta(days=index + 1)
            if start <= forecast_date <= end:
                active[index] = 1.0
                if event_discount is None:
                    discount[index] = None
                elif discount[index] is not None:
                    discount[index] = max(float(discount[index]), event_discount)
    return active, discount


def _future_inbound_events(
    *,
    nm_id: int,
    snapshot_date: date,
    horizon_days: int,
    stock_row: Mapping[str, Any],
) -> list[PricingFutureInboundEventV1]:
    result: list[PricingFutureInboundEventV1] = []
    seen: set[tuple[str | None, date]] = set()
    for raw in _json_list(stock_row.get("future_inbound_events")):
        if not isinstance(raw, Mapping):
            continue
        units = _positive_number(raw.get("units"))
        try:
            eta = _as_date(raw.get("eta_date"))
        except (TypeError, ValueError):
            continue
        if units is None:
            continue
        supply_id = str(raw.get("supply_id") or "").strip() or None
        identity = (supply_id, eta)
        if identity in seen:
            continue
        seen.add(identity)
        status_id = _number(raw.get("status_id"))
        confirmed_stage = (
            status_id is not None
            and int(status_id) in {2, 3}
            and raw.get("fact_date") is None
        )
        overdue = confirmed_stage and eta <= snapshot_date
        status = (
            "unconfirmed"
            if not confirmed_stage
            else ("overdue" if overdue else "scheduled")
        )
        include = (
            status == "scheduled"
            and eta <= snapshot_date + timedelta(days=horizon_days)
        )
        result.append(
            PricingFutureInboundEventV1(
                etaDate=eta,
                units=units,
                includeInCapacity=include,
                status=status,
                supplyId=supply_id,
                evidenceRef=(
                    "supply.wb_supplies_goods:"
                    f"{nm_id}:{supply_id or 'unknown'}:"
                    f"{_iso_date(raw.get('snapshot_date')) or snapshot_date.isoformat()}"
                ),
            )
        )
    return result


def _context_evidence_ref(
    name: str,
    nm_id: int,
    observations: Sequence[PricingCompletedDayObservationV1],
) -> str | None:
    dated = [
        item.observed_date
        for item in observations
        if _context_value(item.context_values, name) is not None
    ]
    if not dated:
        return None
    source = {
        "ad_spend_rub": PRICE_DAILY_MART,
        "ad_views": PRICE_DAILY_MART,
        "ad_clicks": PRICE_DAILY_MART,
        "lagged_organic_position": "agent_marts.v_agent_marketing_sku_daily:lag_1d",
        "lagged_open_to_cart_conversion": "agent_marts.v_agent_sales_sku_daily:lag_1d",
        "lagged_cart_to_order_conversion": "agent_marts.v_agent_sales_sku_daily:lag_1d",
        "lagged_competitor_median_price_rub": "analytics_ai.mart_price_context_daily:lag_1d",
        "promotion_active": "marketing.wb_calendar_promotions_nomenclatures",
        "promotion_discount_pct": "marketing.wb_calendar_promotions_nomenclatures",
        "annual_season_sin": "calendar:Europe/Moscow",
        "annual_season_cos": "calendar:Europe/Moscow",
    }.get(name, "pricing-context")
    return f"{source}:{nm_id}:{min(dated).isoformat()}:{max(dated).isoformat()}"


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _context_source_metadata(request: PricingSimulationRequestV1) -> dict[str, Any]:
    evidence = getattr(request, "evidence", None)
    context = getattr(evidence, "forecast_context", None)
    if context is None:
        return {
            "contextSchemaVersion": None,
            "activeContextFeatures": [],
            "featureCoverage": {},
            "contextDataGaps": ["pricing_context_not_loaded"],
        }
    by_name = {item.name: item for item in context.features}
    groups = {
        "advertising": ("ad_spend_rub", "ad_views", "ad_clicks"),
        "organicVisibility": ("lagged_organic_position",),
        "competitors": ("lagged_competitor_median_price_rub",),
        "cardConversion": (
            "lagged_open_to_cart_conversion",
            "lagged_cart_to_order_conversion",
        ),
        "seasonalityPromotions": (
            "annual_season_sin",
            "annual_season_cos",
            "promotion_active",
            "promotion_discount_pct",
        ),
    }
    coverage: dict[str, Any] = {}
    for group, names in groups.items():
        features = [by_name[name] for name in names if name in by_name]
        modeled = [item.name for item in features if item.status == "modeled"]
        observed = [item.name for item in features if item.status == "observed_only"]
        if len(modeled) == len(features) and features:
            status = "modeled"
        elif modeled:
            status = "modeled_partial"
        elif observed:
            status = "observed_only"
        else:
            status = "missing"
        coverage[group] = {
            "status": status,
            "modeledFeatures": modeled,
            "observedOnlyFeatures": observed,
            "features": {
                item.name: {
                    "status": item.status,
                    "observedDays": item.observed_days,
                    "coverageRatio": round(item.coverage_ratio, 6),
                }
                for item in features
            },
        }
    inbound = evidence.available_stock.future_inbound
    included = [item for item in inbound if item.include_in_capacity]
    coverage["futureSupply"] = {
        "status": "scheduled" if included else ("observed_only" if inbound else "missing"),
        "eventCount": len(inbound),
        "includedEventCount": len(included),
        "includedUnits": round(sum(item.units for item in included), 4),
    }
    complete = all(
        item.get("status") in {"modeled", "scheduled"}
        for item in coverage.values()
    )
    return {
        "contextSchemaVersion": context.schema_version,
        "activeContextFeatures": [
            item.name for item in context.features if item.status == "modeled"
        ],
        "featureCoverage": {**coverage, "complete": complete},
        "contextDataGaps": list(context.data_gaps),
        "futureInboundEvents": [
            item.model_dump(mode="json", by_alias=True) for item in inbound
        ],
    }


def _eligible_demand_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    deduplicated: dict[date, Mapping[str, Any]] = {}
    for row in rows:
        try:
            report_date = _as_date(row.get("report_date"))
        except (TypeError, ValueError):
            continue
        price = _positive_number(row.get("seller_effective_price_rub"))
        units = _nonnegative_number(row.get("non_cancelled_order_units"))
        if (
            row.get("shadow_demand_observation_usable") is True
            and row.get("observed_out_of_stock") is False
            and price is not None
            and units is not None
        ):
            deduplicated[report_date] = row
    return [deduplicated[key] for key in sorted(deduplicated)][-MAX_OBSERVATION_DAYS:]


def _eligible_realization_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    deduplicated: dict[date, Mapping[str, Any]] = {}
    for row in rows:
        try:
            report_date = _as_date(row.get("report_date"))
        except (TypeError, ValueError):
            continue
        orders = _positive_number(row.get("non_cancelled_order_units"))
        realized = _nonnegative_number(row.get("realized_sales_units"))
        if (
            row.get("orders_day_complete") is True
            and row.get("sales_day_complete") is True
            and orders is not None
            and realized is not None
            and realized <= orders
        ):
            deduplicated[report_date] = row
    return [deduplicated[key] for key in sorted(deduplicated)]


def estimate_empirical_realization_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    draws: int = REALIZATION_BOOTSTRAP_DRAWS,
) -> list[float]:
    values = list(rows)
    if len(values) < MIN_REALIZATION_OBSERVATION_DAYS or draws < 3:
        return []
    orders = np.asarray(
        [float(row["non_cancelled_order_units"]) for row in values],
        dtype=np.float64,
    )
    realized = np.asarray(
        [float(row["realized_sales_units"]) for row in values],
        dtype=np.float64,
    )
    canonical = json.dumps(
        [
            [
                _iso_date(row.get("report_date")),
                float(row["non_cancelled_order_units"]),
                float(row["realized_sales_units"]),
            ]
            for row in values
        ],
        separators=(",", ":"),
    )
    seed = int.from_bytes(
        sha256(canonical.encode("utf-8")).digest()[:8],
        byteorder="big",
        signed=False,
    )
    generator = np.random.Generator(np.random.PCG64(seed))
    sampled_indices = generator.integers(
        0,
        len(values),
        size=(draws, len(values)),
    )
    sampled_orders = np.sum(orders[sampled_indices], axis=1)
    sampled_realized = np.sum(realized[sampled_indices], axis=1)
    ratios = np.divide(
        sampled_realized,
        sampled_orders,
        out=np.zeros_like(sampled_realized),
        where=sampled_orders > 0,
    )
    return [float(value) for value in ratios if math.isfinite(float(value))]


def _estimate_shadow_unit_economics(
    rows: Sequence[Mapping[str, Any]],
    *,
    snapshot_date: date,
) -> dict[str, Any]:
    usable: list[Mapping[str, Any]] = []
    gross = payout = cogs = units = 0.0
    for row in rows:
        row_units = _positive_number(row.get("realized_sales_units"))
        row_gross = _positive_number(row.get("gross_revenue_rub"))
        row_payout = _number(row.get("payout_total_rub"))
        row_cogs = _nonnegative_number(row.get("cogs_total_rub"))
        if (
            row.get("finance_values_complete") is True
            and row_units is not None
            and row_gross is not None
            and row_payout is not None
            and row_cogs is not None
        ):
            usable.append(row)
            gross += row_gross
            payout += row_payout
            cogs += row_cogs
            units += row_units
    rate = payout / gross if gross > 0 else None
    variable_cost = cogs / units if units > 0 else None
    if rate is not None and (not math.isfinite(rate) or not 0 < rate <= 1):
        rate = None
    if variable_cost is not None and (
        not math.isfinite(variable_cost) or variable_cost < 0
    ):
        variable_cost = None
    return {
        "net_revenue_rate": rate,
        "variable_cost_per_unit_rub": variable_cost,
        "rows": usable,
        "latest_observation_age_days": (
            _age_days(
                snapshot_date,
                max(_as_date(row.get("report_date")) for row in usable),
            )
            if usable
            else None
        ),
    }


def pricing_request_snapshot_payload(
    request: PricingSimulationRequestV1,
) -> dict[str, Any]:
    payload = request.model_dump(mode="json", by_alias=True, exclude_none=True)
    payload.pop("snapshotHash", None)
    return payload


def _period_evidence_ref(
    prefix: str,
    nm_id: int,
    rows: Sequence[Mapping[str, Any]],
) -> str | None:
    dates = [
        _iso_date(row.get("report_date"))
        for row in rows
        if row.get("report_date") is not None
    ]
    if not dates:
        return None
    return f"{prefix}:{nm_id}:{min(dates)}:{max(dates)}"


def _round_price(value: float, step: float) -> float:
    return float(round(value / step) * step)


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _positive_number(value: Any) -> float | None:
    number = _number(value)
    return number if number is not None and number > 0 else None


def _nonnegative_number(value: Any) -> float | None:
    number = _number(value)
    return number if number is not None and number >= 0 else None


def _as_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value[:10])
    raise ValueError(f"invalid date: {value!r}")


def _iso_date(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return _as_date(value).isoformat()
    except (TypeError, ValueError):
        return None


def _age_days(as_of: date, observed: date) -> int:
    age = (as_of - observed).days
    if age < 0:
        raise ValueError("evidence date cannot be after the pinned price snapshot")
    return age


__all__ = [
    "ELASTICITY_BOOTSTRAP_DRAWS",
    "REALIZATION_BOOTSTRAP_DRAWS",
    "MAX_OBSERVATION_DAYS",
    "MIN_PRICE_LEVELS",
    "MIN_SHADOW_OBSERVATION_DAYS",
    "PRICE_DAILY_MART",
    "PRICING_SOURCE_SCHEMA_VERSION",
    "PricingSimulationSourceError",
    "build_pricing_candidates",
    "build_pricing_simulation_request",
    "estimate_empirical_elasticity_bootstrap",
    "estimate_contextual_demand_bootstrap",
    "estimate_empirical_realization_bootstrap",
    "load_pricing_simulation_source",
    "pricing_request_snapshot_payload",
]
