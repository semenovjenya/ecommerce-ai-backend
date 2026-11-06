from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

from agi_ecom_app.manager_lifecycle import jobs
from agi_ecom_app.manager_lifecycle.jobs import manager_measurement_windows
from agi_ecom_app.manager_lifecycle.pricing_measurement import (
    PRICING_CONTRIBUTION_BASIS,
    PRICING_MEASUREMENT_SOURCE,
    PricingMeasurementProvider,
    register_pricing_measurement_provider,
)
from tests.test_manager_lifecycle import _experiment


MOSCOW = ZoneInfo("Europe/Moscow")
APPLIED_AT = datetime(2026, 7, 15, 9, 0, tzinfo=timezone.utc)
BASELINE_START = datetime(2026, 6, 30, 21, 0, tzinfo=timezone.utc)
BASELINE_END = datetime(2026, 7, 14, 21, 0, tzinfo=timezone.utc)
EVALUATION_START = datetime(2026, 7, 15, 21, 0, tzinfo=timezone.utc)
EVALUATION_END = datetime(2026, 7, 29, 21, 0, tzinfo=timezone.utc)


class FakeCursor:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows
        self.executed: tuple[str, list] | None = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def execute(self, query: str, params: list) -> None:
        self.executed = (query, params)

    def fetchall(self) -> list[dict]:
        return list(self.rows)


class FakeConnection:
    def __init__(self, rows: list[dict]) -> None:
        self.cursor_instance = FakeCursor(rows)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def cursor(self) -> FakeCursor:
        return self.cursor_instance


def _local_timestamp(report_date: date, *, next_day: bool = False) -> datetime:
    source_date = report_date + timedelta(days=1 if next_day else 0)
    source_time = time(10 if next_day else 23, 0)
    return datetime.combine(source_date, source_time, tzinfo=MOSCOW).astimezone(
        timezone.utc
    )


def _rows() -> list[dict]:
    baseline_first = date(2026, 7, 1)
    evaluation_first = date(2026, 7, 16)
    report_dates = [
        *(baseline_first + timedelta(days=offset) for offset in range(14)),
        *(evaluation_first + timedelta(days=offset) for offset in range(14)),
    ]
    rows: list[dict] = []
    for index, report_date in enumerate(report_dates):
        baseline = index < 14
        rows.append(
            {
                "report_date": report_date,
                "nm_id": 55,
                "base_price_rub": Decimal("1100" if baseline else "1000"),
                "seller_discount_pct": Decimal("10"),
                "seller_effective_price_rub": Decimal("1000" if baseline else "900"),
                "currency_iso_code": "RUB",
                "pricing_contribution_rub": Decimal("100" if baseline else "130"),
                "pricing_payout_before_cogs_rub": Decimal(
                    "160" if baseline else "190"
                ),
                "pricing_cogs_charged_rub": Decimal("60"),
                "pricing_contribution_basis": PRICING_CONTRIBUTION_BASIS,
                "payout_missing_row_count": 0,
                "missing_cost_row_count": 0,
                "finance_missing_business_key_count": 0,
                "partial_ad_spend_subtracted_from_pricing_contribution": False,
                "business_day_complete": True,
                "price_snapshot_present": True,
                "price_snapshot_exact_day": True,
                "price_values_complete": True,
                "orders_day_complete": True,
                "stock_snapshot_present": True,
                "finance_values_complete": True,
                "ad_spend_complete": True,
                "shadow_profit_observation_usable": True,
                "observed_out_of_stock": False,
                "price_last_synced_at": _local_timestamp(report_date),
                "stock_last_synced_at": _local_timestamp(report_date),
                "orders_change_watermark_at": _local_timestamp(
                    report_date, next_day=True
                ),
                "marketing_source_completed_at": _local_timestamp(
                    report_date, next_day=True
                ),
                "marketing_last_synced_at": None,
                "finance_last_synced_at": _local_timestamp(
                    report_date, next_day=True
                ),
            }
        )
    return rows


def _pricing_experiment(*, target: dict | None = None):
    return replace(
        _experiment(mode="wb_price", state="measuring"),
        family="pricing",
        target=(
            target
            if target is not None
            else {
                "target_type": "sku",
                "target_refs": ["55"],
                "operation": "set_price",
                "payload": {"nm_id": 55},
            }
        ),
        from_value={"value": 1000, "unit": "RUB"},
        to_value={"value": 900, "unit": "RUB"},
        measurement_spec={
            "primaryMetric": "contribution_profit_rub",
            "baselineDays": 14,
            "evaluationDays": 14,
            "attributionMethod": "observed_change",
        },
    )


def _collect(rows: list[dict], *, experiment=None):
    connection = FakeConnection(rows)
    provider = PricingMeasurementProvider(lambda: connection)
    readout = provider.collect(
        experiment or _pricing_experiment(),
        baseline_start=BASELINE_START,
        baseline_end=BASELINE_END,
        evaluation_start=EVALUATION_START,
        evaluation_end=EVALUATION_END,
    )
    return readout, connection.cursor_instance.executed


def test_complete_fixed_windows_return_observed_pricing_contribution_change() -> None:
    readout, executed = _collect(_rows())

    assert readout.data_complete is True
    assert readout.disposition == "ready"
    assert readout.baseline_treated == 100
    assert readout.post_treated == 130
    assert readout.baseline_control is None
    assert readout.post_control is None
    assert readout.data_as_of == datetime(
        2026, 7, 30, 7, 0, tzinfo=timezone.utc
    )
    assert readout.metrics["causalValid"] is False
    assert readout.metrics["observedPricingContributionChangePerDayRub"] == 30
    assert readout.metrics["observedPricingContributionChangeTotalRub"] == 420
    assert readout.metrics["baseline"]["totalRub"] == 1400
    assert readout.metrics["evaluation"]["totalRub"] == 1820
    assert readout.metrics["source"] == PRICING_MEASUREMENT_SOURCE
    assert readout.metrics["evidenceTier"] == "observed_unwatermarked_shadow"
    assert readout.metrics["outcomeIncludesAdvertisingSpend"] is False
    assert PRICING_MEASUREMENT_SOURCE in executed[0]
    assert executed[1] == [55, date(2026, 7, 1), date(2026, 7, 30)]


def test_pricing_windows_use_completed_moscow_days_and_exclude_action_day() -> None:
    assert manager_measurement_windows(
        family="pricing",
        applied_at=APPLIED_AT,
        baseline_days=14,
        evaluation_days=14,
    ) == (
        BASELINE_START,
        BASELINE_END,
        EVALUATION_START,
        EVALUATION_END,
    )


def test_missing_day_waits_without_synthetic_zero() -> None:
    rows = [row for row in _rows() if row["report_date"] != date(2026, 7, 20)]

    readout, _ = _collect(rows)

    assert readout.data_complete is False
    assert readout.disposition == "waiting_for_data"
    assert readout.baseline_treated is None
    assert readout.post_treated is None
    assert readout.metrics["reason"] == "missing_sku_days"
    assert readout.metrics["missingSkuDays"] == [
        {"reportDate": "2026-07-20", "nmId": 55}
    ]


@pytest.mark.parametrize(
    ("field", "reason"),
    [
        ("price_values_complete", "price_values_incomplete"),
        ("finance_values_complete", "finance_values_incomplete"),
        ("shadow_profit_observation_usable", "shadow_profit_observation_incomplete"),
    ],
)
def test_incomplete_price_or_finance_data_waits_without_values(
    field: str,
    reason: str,
) -> None:
    rows = _rows()
    rows[20][field] = False

    readout, _ = _collect(rows)

    assert readout.data_complete is False
    assert readout.disposition == "waiting_for_data"
    assert readout.baseline_treated is None
    assert readout.post_treated is None
    assert reason in readout.metrics["incompleteSkuDays"][0]["reasons"]


def test_stale_final_day_finance_never_claims_freshness() -> None:
    rows = _rows()
    stale_at = EVALUATION_END - timedelta(minutes=1)
    rows[-1]["finance_last_synced_at"] = stale_at

    readout, _ = _collect(rows)

    assert readout.data_complete is False
    assert readout.disposition == "waiting_for_data"
    assert readout.data_as_of is None
    assert readout.baseline_treated is None
    assert readout.post_treated is None
    assert readout.metrics["reason"] == "incomplete_sku_days"
    assert "finance_source_freshness_lag" in readout.metrics[
        "incompleteSkuDays"
    ][0]["reasons"]


def test_stale_middle_day_watermark_cannot_be_hidden_by_fresh_final_day() -> None:
    rows = _rows()
    stale_row = rows[18]
    stale_row["orders_change_watermark_at"] = datetime.combine(
        stale_row["report_date"] + timedelta(days=1),
        time.min,
        tzinfo=MOSCOW,
    ).astimezone(timezone.utc) - timedelta(seconds=1)

    readout, _ = _collect(rows)

    assert readout.data_complete is False
    assert readout.disposition == "waiting_for_data"
    assert readout.baseline_treated is None
    assert readout.post_treated is None
    assert readout.metrics["reason"] == "incomplete_sku_days"
    assert readout.metrics["incompleteSkuDays"] == [
        {
            "reportDate": stale_row["report_date"].isoformat(),
            "reasons": ["orders_source_freshness_lag"],
        }
    ]


def test_evaluation_price_mismatch_is_terminally_blocked() -> None:
    rows = _rows()
    rows[20]["seller_effective_price_rub"] = Decimal("950")

    readout, _ = _collect(rows)

    assert readout.data_complete is False
    assert readout.disposition == "blocked"
    assert readout.baseline_treated is None
    assert readout.post_treated is None
    assert readout.metrics["reason"] == "price_exposure_mismatch"
    assert readout.metrics["priceExposureMismatches"] == [
        {
            "reportDate": "2026-07-22",
            "expectedSellerEffectivePriceRub": 900.0,
            "observedSellerEffectivePriceRub": 950.0,
        }
    ]


def test_changed_discount_is_terminally_blocked_as_a_confounded_treatment() -> None:
    rows = _rows()
    rows[20]["seller_discount_pct"] = Decimal("15")

    readout, _ = _collect(rows)

    assert readout.data_complete is False
    assert readout.disposition == "blocked"
    assert readout.metrics["reason"] == "seller_discount_changed"


def test_completed_out_of_stock_day_is_terminally_blocked() -> None:
    rows = _rows()
    rows[20]["observed_out_of_stock"] = True

    readout, _ = _collect(rows)

    assert readout.data_complete is False
    assert readout.disposition == "blocked"
    assert readout.metrics["reason"] == "measurement_source_integrity_error"
    assert "observed_out_of_stock" in readout.metrics["integrityReasons"]


def test_duplicate_mart_grain_is_terminally_blocked() -> None:
    rows = _rows()
    rows.append(dict(rows[0]))

    readout, _ = _collect(rows)

    assert readout.data_complete is False
    assert readout.disposition == "blocked"
    assert readout.metrics["reason"] == "measurement_source_integrity_error"
    assert "duplicate_mart_grain" in readout.metrics["integrityReasons"]


def test_changed_contribution_semantics_are_terminally_blocked() -> None:
    rows = _rows()
    rows[0]["pricing_contribution_basis"] = "unknown_basis"

    readout, _ = _collect(rows)

    assert readout.data_complete is False
    assert readout.disposition == "blocked"
    assert readout.metrics["reason"] == "measurement_source_integrity_error"
    assert readout.metrics["integrityReasons"] == [
        "pricing_contribution_basis_mismatch"
    ]


def test_missing_target_sku_is_blocked_without_query() -> None:
    experiment = _pricing_experiment(
        target={
            "target_type": "sku",
            "target_refs": [],
            "operation": "set_price",
            "payload": {},
        }
    )

    readout, executed = _collect(_rows(), experiment=experiment)

    assert executed is None
    assert readout.data_complete is False
    assert readout.disposition == "blocked"
    assert readout.metrics["reason"] == "missing_target_sku"


def test_payload_nm_id_is_authoritative_when_target_ref_is_vendor_code() -> None:
    experiment = _pricing_experiment(
        target={
            "label": "vendor-code-55",
            "target_type": "sku",
            "target_refs": ["vendor-code-55"],
            "operation": "set_price",
            "payload": {"nm_id": 55},
        }
    )

    readout, executed = _collect(_rows(), experiment=experiment)

    assert readout.data_complete is True
    assert executed is not None
    assert executed[1][0] == 55


def test_pricing_provider_registration_populates_production_registry() -> None:
    with patch.dict(jobs._MEASUREMENT_PROVIDERS, {}, clear=True):
        provider = register_pricing_measurement_provider()

        assert jobs._MEASUREMENT_PROVIDERS["pricing"] is provider
        assert isinstance(provider, PricingMeasurementProvider)
