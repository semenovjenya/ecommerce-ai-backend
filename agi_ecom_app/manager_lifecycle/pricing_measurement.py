from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal, InvalidOperation
import logging
import math
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

from agi_ecom_app.services.db import get_connection
from agi_ecom_app.services.memory import AgentActionExperimentRecord

if TYPE_CHECKING:
    from agi_ecom_app.manager_lifecycle.jobs import ManagerMeasurementReadout


logger = logging.getLogger(__name__)

PRICING_MEASUREMENT_SOURCE = "agent_marts.v_agent_price_sku_daily"
PRICING_CONTRIBUTION_BASIS = "finance.wb_report_detail_daily.ppvz_for_pay"
_MOSCOW = ZoneInfo("Europe/Moscow")

_PRICING_MEASUREMENT_SQL = f"""
select
    report_date,
    nm_id,
    base_price_rub,
    seller_discount_pct,
    seller_effective_price_rub,
    currency_iso_code,
    pricing_contribution_rub,
    pricing_payout_before_cogs_rub,
    pricing_cogs_charged_rub,
    pricing_contribution_basis,
    payout_missing_row_count,
    missing_cost_row_count,
    finance_missing_business_key_count,
    partial_ad_spend_subtracted_from_pricing_contribution,
    business_day_complete,
    price_snapshot_present,
    price_snapshot_exact_day,
    price_values_complete,
    orders_day_complete,
    stock_snapshot_present,
    finance_values_complete,
    ad_spend_complete,
    shadow_profit_observation_usable,
    observed_out_of_stock,
    price_last_synced_at,
    stock_last_synced_at,
    orders_change_watermark_at,
    marketing_source_completed_at,
    marketing_last_synced_at,
    finance_last_synced_at
from {PRICING_MEASUREMENT_SOURCE}
where nm_id = %s::bigint
  and report_date >= %s::date
  and report_date < %s::date
order by report_date, nm_id
"""


class PricingMeasurementProvider:
    """Measure an observed pricing effect only from complete, fixed day windows.

    The provider intentionally publishes no control series. Consequently the
    lifecycle records the pre/post change as non-causal unless a future pricing
    provider supplies an explicit control SKU.
    """

    def __init__(self, connection_factory: Callable[[], Any] | None = None) -> None:
        self._connection_factory = connection_factory

    def collect(
        self,
        experiment: AgentActionExperimentRecord,
        *,
        baseline_start: datetime,
        baseline_end: datetime,
        evaluation_start: datetime,
        evaluation_end: datetime,
    ) -> ManagerMeasurementReadout:
        contract = _pricing_contract(experiment)
        if isinstance(contract, str):
            return _incomplete_readout(reason=contract, disposition="blocked")
        nm_id, baseline_price, evaluation_price = contract

        windows = _measurement_windows(
            baseline_start=baseline_start,
            baseline_end=baseline_end,
            evaluation_start=evaluation_start,
            evaluation_end=evaluation_end,
        )
        if windows is None:
            return _incomplete_readout(
                reason="invalid_measurement_window",
                disposition="blocked",
            )
        baseline_dates, evaluation_dates = windows
        first_date = baseline_dates[0]
        end_date = evaluation_dates[-1] + timedelta(days=1)

        try:
            rows = self._load_rows(nm_id, first_date, end_date)
        except Exception as exc:
            logger.exception(
                "pricing_measurement_query_failed experiment_id=%s",
                experiment.id,
            )
            return _incomplete_readout(
                reason="measurement_source_unavailable",
                disposition="waiting_for_data",
                extra={"errorType": type(exc).__name__},
            )

        return _build_readout(
            rows,
            nm_id=nm_id,
            baseline_price=baseline_price,
            evaluation_price=evaluation_price,
            baseline_dates=baseline_dates,
            evaluation_dates=evaluation_dates,
            evaluation_end=evaluation_end,
        )

    def _load_rows(
        self,
        nm_id: int,
        first_date: date,
        end_date: date,
    ) -> list[dict[str, Any]]:
        connection_factory = self._connection_factory or get_connection
        with connection_factory() as connection, connection.cursor() as cursor:
            cursor.execute(
                _PRICING_MEASUREMENT_SQL,
                [nm_id, first_date, end_date],
            )
            return [dict(row) for row in cursor.fetchall()]


def _pricing_contract(
    experiment: AgentActionExperimentRecord,
) -> tuple[int, Decimal, Decimal] | str:
    if experiment.family.strip().lower() != "pricing":
        return "unsupported_measurement_family"
    target = experiment.target
    if str(target.get("operation") or "") != "set_price":
        return "unsupported_pricing_operation"
    if str(target.get("target_type", target.get("targetType")) or "") != "sku":
        return "unsupported_pricing_target"

    nm_id = _payload_nm_id(target)
    if nm_id is None:
        return "missing_target_sku"
    if any(candidate != nm_id for candidate in _target_sku_ids(target)):
        return "target_sku_conflict"
    direct_nm_id = _integer(target.get("nm_id", target.get("nmId")))
    if direct_nm_id is not None and direct_nm_id != nm_id:
        return "target_sku_conflict"
    label_nm_id = _integer(target.get("label"))
    if label_nm_id is not None and label_nm_id != nm_id:
        return "target_sku_conflict"

    baseline_price = _price(experiment.from_value)
    evaluation_price = _price(experiment.to_value)
    if baseline_price is None:
        return "missing_baseline_price"
    if evaluation_price is None:
        return "missing_evaluation_price"
    if baseline_price == evaluation_price:
        return "unchanged_price_contract"
    if _price_unit(experiment.from_value) != "RUB" or _price_unit(
        experiment.to_value
    ) != "RUB":
        return "unsupported_price_currency"

    primary_metric = str(
        experiment.measurement_spec.get(
            "primaryMetric",
            experiment.measurement_spec.get("primary_metric", ""),
        )
        or ""
    )
    if primary_metric != "contribution_profit_rub":
        return "unsupported_measurement_metric"
    attribution = str(
        experiment.measurement_spec.get(
            "attributionMethod",
            experiment.measurement_spec.get("attribution_method", ""),
        )
        or ""
    )
    if attribution != "observed_change":
        return "unsupported_attribution_method"
    return nm_id, baseline_price, evaluation_price


def _measurement_windows(
    *,
    baseline_start: datetime,
    baseline_end: datetime,
    evaluation_start: datetime,
    evaluation_end: datetime,
) -> tuple[list[date], list[date]] | None:
    local = tuple(
        _utc(value).astimezone(_MOSCOW)
        for value in (
            baseline_start,
            baseline_end,
            evaluation_start,
            evaluation_end,
        )
    )
    if any(
        value != value.replace(hour=0, minute=0, second=0, microsecond=0)
        for value in local
    ):
        return None
    baseline_first, baseline_last, evaluation_first, evaluation_last = (
        value.date() for value in local
    )
    if (
        baseline_first >= baseline_last
        or evaluation_first >= evaluation_last
        or baseline_last + timedelta(days=1) != evaluation_first
    ):
        return None
    return (
        _date_range(baseline_first, baseline_last),
        _date_range(evaluation_first, evaluation_last),
    )


def _build_readout(
    rows: Sequence[Mapping[str, Any]],
    *,
    nm_id: int,
    baseline_price: Decimal,
    evaluation_price: Decimal,
    baseline_dates: Sequence[date],
    evaluation_dates: Sequence[date],
    evaluation_end: datetime,
) -> ManagerMeasurementReadout:
    expected_dates = set([*baseline_dates, *evaluation_dates])
    grouped: dict[date, list[Mapping[str, Any]]] = defaultdict(list)
    invalid_rows = 0
    integrity_reasons: set[str] = set()
    for row in rows:
        report_date = _date(row.get("report_date"))
        row_nm_id = _integer(row.get("nm_id"))
        if report_date is None or row_nm_id is None:
            invalid_rows += 1
            continue
        if row_nm_id != nm_id:
            integrity_reasons.add("unexpected_sku_id")
        if report_date not in expected_dates:
            integrity_reasons.add("unexpected_report_date")
        grouped[report_date].append(row)

    duplicate_dates = sorted(
        report_date for report_date, day_rows in grouped.items() if len(day_rows) != 1
    )
    if duplicate_dates:
        integrity_reasons.add("duplicate_mart_grain")
    if invalid_rows:
        integrity_reasons.add("invalid_mart_grain")
    if integrity_reasons:
        return _incomplete_readout(
            reason="measurement_source_integrity_error",
            disposition="blocked",
            extra={
                **_common_metrics(nm_id, expected_dates, grouped),
                "integrityReasons": sorted(integrity_reasons),
                "duplicateDates": [value.isoformat() for value in duplicate_dates],
                "invalidRowCount": invalid_rows,
            },
        )

    missing_dates = sorted(expected_dates - set(grouped))
    incomplete_days: list[dict[str, Any]] = []
    contributions: dict[date, Decimal] = {}
    effective_prices: dict[date, Decimal] = {}
    discounts: dict[date, Decimal] = {}
    final_source_timestamps: list[datetime] = []
    for report_date in sorted(expected_dates & set(grouped)):
        row = grouped[report_date][0]
        reasons = _incomplete_reasons(row, report_date=report_date)
        contribution = _decimal(row.get("pricing_contribution_rub"))
        effective_price = _positive_decimal(row.get("seller_effective_price_rub"))
        discount = _decimal(row.get("seller_discount_pct"))
        if contribution is None:
            reasons.append("contribution_missing")
        if effective_price is None:
            reasons.append("seller_effective_price_missing")
        if discount is None:
            reasons.append("seller_discount_missing")
        if reasons:
            incomplete_days.append(
                {
                    "reportDate": report_date.isoformat(),
                    "reasons": sorted(set(reasons)),
                }
            )
            continue
        assert contribution is not None and effective_price is not None and discount is not None
        contributions[report_date] = contribution
        effective_prices[report_date] = effective_price
        discounts[report_date] = discount
        if report_date == evaluation_dates[-1]:
            final_source_timestamps.extend(_completion_timestamps(row))

    common = _common_metrics(nm_id, expected_dates, grouped)
    terminal_integrity_reasons = sorted(
        {
            reason
            for item in incomplete_days
            for reason in item["reasons"]
            if reason
            in {
                "pricing_contribution_basis_mismatch",
                "pricing_contribution_semantics_mismatch",
                "pricing_contribution_arithmetic_mismatch",
                "pricing_currency_mismatch",
                "finance_business_key_integrity_error",
                "observed_out_of_stock",
            }
        }
    )
    if terminal_integrity_reasons:
        return _incomplete_readout(
            reason="measurement_source_integrity_error",
            disposition="blocked",
            extra={
                **common,
                "integrityReasons": terminal_integrity_reasons,
                "incompleteSkuDays": incomplete_days,
            },
        )
    if missing_dates or incomplete_days:
        return _incomplete_readout(
            reason=("missing_sku_days" if missing_dates else "incomplete_sku_days"),
            disposition="waiting_for_data",
            extra={
                **common,
                "missingSkuDays": [
                    {"reportDate": report_date.isoformat(), "nmId": nm_id}
                    for report_date in missing_dates
                ],
                "incompleteSkuDays": incomplete_days,
            },
        )

    exposure_mismatches = [
        {
            "reportDate": report_date.isoformat(),
            "expectedSellerEffectivePriceRub": float(expected_price),
            "observedSellerEffectivePriceRub": float(effective_prices[report_date]),
        }
        for report_date, expected_price in [
            *((item, baseline_price) for item in baseline_dates),
            *((item, evaluation_price) for item in evaluation_dates),
        ]
        if not _money_equal(effective_prices[report_date], expected_price)
    ]
    if exposure_mismatches:
        return _incomplete_readout(
            reason="price_exposure_mismatch",
            disposition="blocked",
            extra={
                **common,
                "baselineSellerEffectivePriceRub": float(baseline_price),
                "evaluationSellerEffectivePriceRub": float(evaluation_price),
                "priceExposureMismatches": exposure_mismatches,
            },
        )

    distinct_discounts = sorted(set(discounts.values()))
    if len(distinct_discounts) != 1:
        return _incomplete_readout(
            reason="seller_discount_changed",
            disposition="blocked",
            extra={
                **common,
                "observedSellerDiscountPct": [
                    float(value) for value in distinct_discounts
                ],
            },
        )

    baseline_values = [contributions[item] for item in baseline_dates]
    evaluation_values = [contributions[item] for item in evaluation_dates]
    baseline_total = sum(baseline_values, Decimal(0))
    evaluation_total = sum(evaluation_values, Decimal(0))
    baseline_mean = baseline_total / Decimal(len(baseline_values))
    evaluation_mean = evaluation_total / Decimal(len(evaluation_values))
    observed_per_day = evaluation_mean - baseline_mean
    expected_baseline_total = baseline_mean * Decimal(len(evaluation_values))
    observed_total = evaluation_total - expected_baseline_total

    source_data_as_of = min(final_source_timestamps) if final_source_timestamps else None
    if source_data_as_of is None or source_data_as_of < _utc(evaluation_end):
        return _incomplete_readout(
            reason=(
                "source_timestamp_missing"
                if source_data_as_of is None
                else "source_freshness_lag"
            ),
            disposition="waiting_for_data",
            extra={
                **common,
                "sourceDataAsOf": (
                    source_data_as_of.isoformat() if source_data_as_of else None
                ),
            },
        )

    return _readout(
        data_complete=True,
        data_as_of=source_data_as_of,
        baseline_treated=float(baseline_mean),
        post_treated=float(evaluation_mean),
        baseline_control=None,
        post_control=None,
        disposition="ready",
        metrics={
            **common,
            "disposition": "ready",
            "attributionMethod": "observed_change",
            "causalValid": False,
            "baselineSellerEffectivePriceRub": float(baseline_price),
            "evaluationSellerEffectivePriceRub": float(evaluation_price),
            "sellerDiscountPct": float(distinct_discounts[0]),
            "baseline": {
                "start": baseline_dates[0].isoformat(),
                "endExclusive": (baseline_dates[-1] + timedelta(days=1)).isoformat(),
                "dayCount": len(baseline_dates),
                "totalRub": float(baseline_total),
                "dailyMeanRub": float(baseline_mean),
            },
            "evaluation": {
                "start": evaluation_dates[0].isoformat(),
                "endExclusive": (evaluation_dates[-1] + timedelta(days=1)).isoformat(),
                "dayCount": len(evaluation_dates),
                "totalRub": float(evaluation_total),
                "dailyMeanRub": float(evaluation_mean),
            },
            "observedPricingContributionChangePerDayRub": float(observed_per_day),
            "observedPricingContributionChangeTotalRub": float(observed_total),
            "baselineExpectedTotalForEvaluationRub": float(expected_baseline_total),
            "sourceDataAsOf": source_data_as_of.isoformat(),
        },
    )


def _incomplete_reasons(
    row: Mapping[str, Any],
    *,
    report_date: date,
) -> list[str]:
    checks = {
        "business_day_incomplete": row.get("business_day_complete") is True,
        "price_snapshot_missing": row.get("price_snapshot_present") is True,
        "price_snapshot_not_exact_day": row.get("price_snapshot_exact_day") is True,
        "price_values_incomplete": row.get("price_values_complete") is True,
        "orders_day_incomplete": row.get("orders_day_complete") is True,
        "stock_snapshot_missing": row.get("stock_snapshot_present") is True,
        "finance_values_incomplete": row.get("finance_values_complete") is True,
        "ad_spend_incomplete": row.get("ad_spend_complete") is True,
        "shadow_profit_observation_incomplete": (
            row.get("shadow_profit_observation_usable") is True
        ),
    }
    reasons = [reason for reason, complete in checks.items() if not complete]
    if row.get("observed_out_of_stock") is True:
        reasons.append("observed_out_of_stock")
    if row.get("pricing_contribution_basis") != PRICING_CONTRIBUTION_BASIS:
        reasons.append("pricing_contribution_basis_mismatch")
    if row.get("partial_ad_spend_subtracted_from_pricing_contribution") is not False:
        reasons.append("pricing_contribution_semantics_mismatch")
    if str(row.get("currency_iso_code") or "").upper() != "RUB":
        reasons.append("pricing_currency_mismatch")
    payout = _decimal(row.get("pricing_payout_before_cogs_rub"))
    cogs = _decimal(row.get("pricing_cogs_charged_rub"))
    contribution = _decimal(row.get("pricing_contribution_rub"))
    if payout is None:
        reasons.append("pricing_payout_missing")
    if cogs is None or cogs < 0:
        reasons.append("pricing_cogs_missing")
    if (
        payout is not None
        and cogs is not None
        and cogs >= 0
        and contribution is not None
        and not _money_equal(contribution, payout - cogs)
    ):
        reasons.append("pricing_contribution_arithmetic_mismatch")
    payout_missing_rows = _nonnegative_integer(row.get("payout_missing_row_count"))
    missing_cost_rows = _nonnegative_integer(row.get("missing_cost_row_count"))
    missing_business_keys = _nonnegative_integer(
        row.get("finance_missing_business_key_count")
    )
    if payout_missing_rows is None:
        reasons.append("payout_completeness_audit_missing")
    elif payout_missing_rows > 0:
        reasons.append("pricing_payout_incomplete")
    if missing_cost_rows is None:
        reasons.append("cogs_completeness_audit_missing")
    elif missing_cost_rows > 0:
        reasons.append("pricing_cogs_incomplete")
    if missing_business_keys is None:
        reasons.append("finance_business_key_audit_missing")
    elif missing_business_keys > 0:
        reasons.append("finance_business_key_integrity_error")
    price_timestamp = _datetime(row.get("price_last_synced_at"))
    stock_timestamp = _datetime(row.get("stock_last_synced_at"))
    if price_timestamp is None:
        reasons.append("price_source_timestamp_missing")
    elif price_timestamp.astimezone(_MOSCOW).date() < report_date:
        reasons.append("price_source_timestamp_stale")
    if stock_timestamp is None:
        reasons.append("stock_source_timestamp_missing")
    elif stock_timestamp.astimezone(_MOSCOW).date() < report_date:
        reasons.append("stock_source_timestamp_stale")
    completed_day_cutoff = datetime.combine(
        report_date + timedelta(days=1),
        time.min,
        tzinfo=_MOSCOW,
    ).astimezone(timezone.utc)
    completion_sources = {
        "orders": _datetime(row.get("orders_change_watermark_at")),
        "marketing": _datetime(row.get("marketing_source_completed_at"))
        or _datetime(row.get("marketing_last_synced_at")),
        "finance": _datetime(row.get("finance_last_synced_at")),
    }
    for source, timestamp in completion_sources.items():
        if timestamp is None:
            reasons.append(f"{source}_source_timestamp_missing")
        elif timestamp < completed_day_cutoff:
            reasons.append(f"{source}_source_freshness_lag")
    return reasons


def _completion_timestamps(row: Mapping[str, Any]) -> list[datetime]:
    values = [
        _datetime(row.get("orders_change_watermark_at")),
        _datetime(row.get("marketing_source_completed_at"))
        or _datetime(row.get("marketing_last_synced_at")),
        _datetime(row.get("finance_last_synced_at")),
    ]
    return [value for value in values if value is not None]


def _common_metrics(
    nm_id: int,
    expected_dates: set[date],
    grouped: Mapping[date, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    return {
        "source": PRICING_MEASUREMENT_SOURCE,
        "primaryMetric": "pricing_contribution_rub",
        "contributionBasis": PRICING_CONTRIBUTION_BASIS,
        "evidenceTier": "observed_unwatermarked_shadow",
        "outcomeIncludesAdvertisingSpend": False,
        "aggregation": "sku_day_contribution_then_daily_mean",
        "targetSkuId": nm_id,
        "expectedSkuDayCount": len(expected_dates),
        "observedSkuDayCount": len(grouped),
    }


def _target_sku_ids(target: Mapping[str, Any]) -> list[int]:
    refs = target.get("target_refs", target.get("targetRefs", ()))
    values = (
        refs
        if isinstance(refs, Sequence) and not isinstance(refs, (str, bytes))
        else ()
    )
    result: list[int] = []
    for candidate in values:
        if isinstance(candidate, Mapping):
            candidate = candidate.get("nm_id", candidate.get("nmId"))
        parsed = _integer(candidate)
        if parsed is not None and parsed not in result:
            result.append(parsed)
    return result


def _payload_nm_id(target: Mapping[str, Any]) -> int | None:
    payload = target.get("payload")
    if not isinstance(payload, Mapping):
        return None
    return _integer(payload.get("nm_id", payload.get("nmId")))


def _price(value: Any) -> Decimal | None:
    if isinstance(value, Mapping):
        if "value" in value:
            value = value["value"]
        elif "price" in value:
            value = value["price"]
    return _positive_decimal(value)


def _price_unit(value: Any) -> str | None:
    if not isinstance(value, Mapping):
        return None
    unit = value.get("unit")
    return str(unit).strip().upper() if unit not in (None, "") else None


def _incomplete_readout(
    *,
    reason: str,
    disposition: str,
    extra: Mapping[str, Any] | None = None,
) -> ManagerMeasurementReadout:
    return _readout(
        data_complete=False,
        data_as_of=None,
        baseline_treated=None,
        post_treated=None,
        baseline_control=None,
        post_control=None,
        disposition=disposition,
        metrics={
            **dict(extra or {}),
            "reason": reason,
            "disposition": disposition,
        },
    )


def _readout(**kwargs: Any) -> ManagerMeasurementReadout:
    from agi_ecom_app.manager_lifecycle.jobs import ManagerMeasurementReadout

    return ManagerMeasurementReadout(**kwargs)


def _date_range(start: date, end: date) -> list[date]:
    return [start + timedelta(days=offset) for offset in range((end - start).days)]


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return _utc(value).astimezone(_MOSCOW).date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None
    return None


def _datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return _utc(value)
    if isinstance(value, str):
        try:
            return _utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
        except ValueError:
            return None
    return None


def _integer(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def _nonnegative_integer(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def _decimal(value: Any) -> Decimal | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return result if math.isfinite(float(result)) else None


def _positive_decimal(value: Any) -> Decimal | None:
    result = _decimal(value)
    return result if result is not None and result > 0 else None


def _money_equal(left: Decimal, right: Decimal) -> bool:
    return abs(left - right) <= Decimal("0.01")


def register_pricing_measurement_provider() -> PricingMeasurementProvider:
    from agi_ecom_app.manager_lifecycle.jobs import (
        register_manager_measurement_provider,
    )

    provider = PricingMeasurementProvider()
    register_manager_measurement_provider("pricing", provider)
    return provider


__all__ = [
    "PRICING_CONTRIBUTION_BASIS",
    "PRICING_MEASUREMENT_SOURCE",
    "PricingMeasurementProvider",
    "register_pricing_measurement_provider",
]
