from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import logging
import math
from typing import TYPE_CHECKING, Any

from agi_ecom_app.services.db import get_connection
from agi_ecom_app.services.memory import AgentActionExperimentRecord

if TYPE_CHECKING:
    from agi_ecom_app.manager_lifecycle.jobs import ManagerMeasurementReadout


logger = logging.getLogger(__name__)

ADVERTISING_MEASUREMENT_SOURCE = "agent_marts.v_agent_ad_campaign_sku_daily"

_ADVERTISING_MEASUREMENT_SQL = f"""
select
    report_date,
    advert_id,
    nm_id,
    sku_contribution_after_ads_rub,
    sku_contribution_after_ads_complete,
    finance_complete,
    finance_snapshot_complete,
    sku_ad_spend_complete,
    stats_day_watermark_complete,
    finance_last_synced_at,
    stats_source_completed_at,
    stats_last_synced_at
from {ADVERTISING_MEASUREMENT_SOURCE}
where nm_id = any(%s::bigint[])
  and advert_id = %s::bigint
  and report_date >= %s::date
  and report_date < %s::date
order by report_date, nm_id, advert_id
"""


class AdvertisingMeasurementProvider:
    """Read complete SKU contribution windows without fabricating missing zeroes."""

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
        contract_error = _contract_error(experiment)
        if contract_error is not None:
            return _incomplete_readout(
                reason=contract_error,
                disposition="blocked",
            )
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

        sku_ids = _target_sku_ids(experiment.target)
        if not sku_ids:
            return _incomplete_readout(
                reason="missing_target_sku",
                disposition="blocked",
            )
        advert_id = _target_advert_id(experiment.target)
        if advert_id is None:
            return _incomplete_readout(
                reason="missing_target_campaign",
                disposition="blocked",
            )

        baseline_dates, evaluation_dates = windows
        first_date = baseline_dates[0]
        end_date = evaluation_dates[-1] + timedelta(days=1)
        try:
            rows = self._load_rows(sku_ids, advert_id, first_date, end_date)
        except Exception as exc:
            logger.exception(
                "advertising_measurement_query_failed experiment_id=%s",
                experiment.id,
            )
            return _incomplete_readout(
                reason="measurement_source_unavailable",
                disposition="waiting_for_data",
                extra={"errorType": type(exc).__name__},
            )

        return _build_readout(
            rows,
            sku_ids=sku_ids,
            target_advert_id=advert_id,
            baseline_dates=baseline_dates,
            evaluation_dates=evaluation_dates,
            evaluation_end=evaluation_end,
        )

    def _load_rows(
        self,
        sku_ids: Sequence[int],
        advert_id: int,
        first_date: date,
        end_date: date,
    ) -> list[dict[str, Any]]:
        connection_factory = self._connection_factory or get_connection
        with connection_factory() as connection, connection.cursor() as cursor:
            cursor.execute(
                _ADVERTISING_MEASUREMENT_SQL,
                [list(sku_ids), advert_id, first_date, end_date],
            )
            return [dict(row) for row in cursor.fetchall()]


def _measurement_windows(
    *,
    baseline_start: datetime,
    baseline_end: datetime,
    evaluation_start: datetime,
    evaluation_end: datetime,
) -> tuple[list[date], list[date]] | None:
    normalized = tuple(
        _utc(value)
        for value in (
            baseline_start,
            baseline_end,
            evaluation_start,
            evaluation_end,
        )
    )
    if any(
        value != value.replace(hour=0, minute=0, second=0, microsecond=0)
        for value in normalized
    ):
        return None
    baseline_first, baseline_last, evaluation_first, evaluation_last = (
        value.date() for value in normalized
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


def _target_sku_ids(target: Mapping[str, Any]) -> list[int]:
    candidates: list[Any] = []
    refs = target.get("target_refs", target.get("targetRefs", ()))
    if isinstance(refs, Sequence) and not isinstance(refs, (str, bytes)):
        candidates.extend(refs)

    result: list[int] = []
    for candidate in candidates:
        if isinstance(candidate, Mapping):
            candidate = candidate.get("nm_id", candidate.get("nmId"))
        if isinstance(candidate, bool) or candidate is None:
            continue
        try:
            value = int(str(candidate).strip())
        except (TypeError, ValueError):
            continue
        if value > 0 and value not in result:
            result.append(value)
    return result


def _target_advert_id(target: Mapping[str, Any]) -> int | None:
    payload = target.get("payload")
    if not isinstance(payload, Mapping):
        return None
    return _integer(payload.get("advert_id", payload.get("advertId")))


def _contract_error(experiment: AgentActionExperimentRecord) -> str | None:
    target = experiment.target
    if str(target.get("operation") or "") != "pause_campaign":
        return "unsupported_advertising_operation"
    if str(target.get("target_type", target.get("targetType")) or "") != "campaign":
        return "unsupported_advertising_target"
    payload = target.get("payload")
    advert_id = (
        payload.get("advert_id", payload.get("advertId"))
        if isinstance(payload, Mapping)
        else None
    )
    if _integer(advert_id) is None:
        return "missing_target_campaign"
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
    return None


def _build_readout(
    rows: Sequence[Mapping[str, Any]],
    *,
    sku_ids: Sequence[int],
    target_advert_id: int,
    baseline_dates: Sequence[date],
    evaluation_dates: Sequence[date],
    evaluation_end: datetime,
) -> ManagerMeasurementReadout:
    grouped: dict[tuple[date, int], list[Mapping[str, Any]]] = defaultdict(list)
    invalid_rows = 0
    for row in rows:
        report_date = _date(row.get("report_date"))
        nm_id = _integer(row.get("nm_id"))
        if report_date is None or nm_id is None:
            invalid_rows += 1
            continue
        grouped[(report_date, nm_id)].append(row)

    all_dates = [*baseline_dates, *evaluation_dates]
    expected = {(report_date, nm_id) for report_date in all_dates for nm_id in sku_ids}
    missing = sorted(expected - set(grouped))
    incomplete: list[dict[str, Any]] = []
    sku_day_values: dict[tuple[date, int], Decimal] = {}
    duplicate_campaign_rows = 0
    coverage_source_timestamps: list[datetime] = []

    for key in sorted(expected & set(grouped)):
        day_rows = grouped[key]
        reasons: list[str] = []
        campaign_ids = [_integer(row.get("advert_id")) for row in day_rows]
        if any(advert_id is None for advert_id in campaign_ids):
            reasons.append("campaign_id_missing")
        if any(
            advert_id is not None and advert_id != target_advert_id
            for advert_id in campaign_ids
        ):
            reasons.append("unexpected_campaign_id")
        distinct_campaign_ids = {
            advert_id for advert_id in campaign_ids if advert_id is not None
        }
        if len(distinct_campaign_ids) != len(day_rows):
            reasons.append("duplicate_mart_grain")
        duplicate_campaign_rows += max(0, len(distinct_campaign_ids) - 1)
        if any(row.get("stats_day_watermark_complete") is not True for row in day_rows):
            reasons.append("ad_stats_day_incomplete")
        if any(row.get("finance_complete") is not True for row in day_rows):
            reasons.append("finance_incomplete")
        if any(
            row.get("finance_snapshot_complete") is not True
            for row in day_rows
        ):
            reasons.append("finance_snapshot_incomplete")
        if any(row.get("sku_ad_spend_complete") is not True for row in day_rows):
            reasons.append("ad_spend_incomplete")
        if any(
            row.get("sku_contribution_after_ads_complete") is not True
            for row in day_rows
        ):
            reasons.append("contribution_incomplete")
        finance_timestamps = [
            _datetime(row.get("finance_last_synced_at")) for row in day_rows
        ]
        stats_timestamps = [
            _datetime(row.get("stats_source_completed_at"))
            or _datetime(row.get("stats_last_synced_at"))
            for row in day_rows
        ]
        if any(value is None for value in finance_timestamps):
            reasons.append("finance_source_timestamp_missing")
        if any(value is None for value in stats_timestamps):
            reasons.append("ad_stats_source_timestamp_missing")

        values = [_decimal(row.get("sku_contribution_after_ads_rub")) for row in day_rows]
        if any(value is None for value in values):
            reasons.append("contribution_missing")
        distinct_values = {value for value in values if value is not None}
        if len(distinct_values) > 1:
            reasons.append("duplicate_campaign_values_disagree")
        if reasons:
            incomplete.append(
                {
                    "reportDate": key[0].isoformat(),
                    "nmId": key[1],
                    "reasons": sorted(set(reasons)),
                }
            )
            continue
        if key[0] == evaluation_dates[-1]:
            coverage_source_timestamps.extend(
                value
                for value in [*finance_timestamps, *stats_timestamps]
                if value is not None
            )
        sku_day_values[key] = next(iter(distinct_values))

    common_metrics: dict[str, Any] = {
        "source": ADVERTISING_MEASUREMENT_SOURCE,
        "primaryMetric": "sku_contribution_after_ads_rub",
        "aggregation": "campaign_deduplicated_sku_day_then_daily_mean",
        "targetSkuIds": list(sku_ids),
        "expectedSkuDayCount": len(expected),
        "observedSkuDayCount": len(grouped),
        "duplicateCampaignRowsDeduplicated": duplicate_campaign_rows,
        "invalidRowCount": invalid_rows,
    }
    terminal_integrity_reasons = {
        reason
        for item in incomplete
        for reason in item["reasons"]
        if reason
        in {
            "campaign_id_missing",
            "unexpected_campaign_id",
            "duplicate_mart_grain",
            "duplicate_campaign_values_disagree",
        }
    }
    if terminal_integrity_reasons:
        return _incomplete_readout(
            reason="measurement_source_integrity_error",
            disposition="blocked",
            extra={
                **common_metrics,
                "integrityReasons": sorted(terminal_integrity_reasons),
                "missingSkuDays": [
                    {"reportDate": report_date.isoformat(), "nmId": nm_id}
                    for report_date, nm_id in missing
                ],
                "incompleteSkuDays": incomplete,
            },
        )
    if missing or incomplete or invalid_rows:
        return _incomplete_readout(
            reason=("missing_sku_days" if missing else "incomplete_sku_days"),
            disposition="waiting_for_data",
            extra={
                **common_metrics,
                "missingSkuDays": [
                    {"reportDate": report_date.isoformat(), "nmId": nm_id}
                    for report_date, nm_id in missing
                ],
                "incompleteSkuDays": incomplete,
            },
        )

    baseline_daily = _daily_totals(baseline_dates, sku_ids, sku_day_values)
    evaluation_daily = _daily_totals(evaluation_dates, sku_ids, sku_day_values)
    if baseline_daily is None or evaluation_daily is None:
        return _incomplete_readout(
            reason="incomplete_sku_days",
            disposition="waiting_for_data",
            extra=common_metrics,
        )

    baseline_total = sum(baseline_daily, Decimal(0))
    evaluation_total = sum(evaluation_daily, Decimal(0))
    baseline_mean = baseline_total / Decimal(len(baseline_daily))
    evaluation_mean = evaluation_total / Decimal(len(evaluation_daily))
    # data_as_of is the latest instant covered by *all* required sources for
    # every target SKU on the final evaluation day, hence the conservative min.
    source_data_as_of = (
        min(coverage_source_timestamps)
        if coverage_source_timestamps
        else None
    )
    if source_data_as_of is None or source_data_as_of < _utc(evaluation_end):
        return _incomplete_readout(
            reason=(
                "source_timestamp_missing"
                if source_data_as_of is None
                else "source_freshness_lag"
            ),
            disposition="waiting_for_data",
            extra={
                **common_metrics,
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
        disposition="ready",
        metrics={
            **common_metrics,
            "disposition": "ready",
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
            "sourceDataAsOf": (
                source_data_as_of.isoformat() if source_data_as_of else None
            ),
        },
    )


def _daily_totals(
    dates: Sequence[date],
    sku_ids: Sequence[int],
    values: Mapping[tuple[date, int], Decimal],
) -> list[Decimal] | None:
    totals: list[Decimal] = []
    for report_date in dates:
        day_values = [values.get((report_date, nm_id)) for nm_id in sku_ids]
        if any(value is None for value in day_values):
            return None
        totals.append(sum((value for value in day_values if value is not None), Decimal(0)))
    return totals


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
        return _utc(value).date()
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
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def _decimal(value: Any) -> Decimal | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return result if math.isfinite(float(result)) else None


def register_advertising_measurement_provider() -> AdvertisingMeasurementProvider:
    from agi_ecom_app.manager_lifecycle.jobs import (
        register_manager_measurement_provider,
    )

    provider = AdvertisingMeasurementProvider()
    register_manager_measurement_provider("advertising", provider)
    return provider


__all__ = [
    "ADVERTISING_MEASUREMENT_SOURCE",
    "AdvertisingMeasurementProvider",
    "register_advertising_measurement_provider",
]
