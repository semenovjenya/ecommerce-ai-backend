from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import pytest
from unittest.mock import patch

from agi_ecom_app.manager_lifecycle import jobs
from agi_ecom_app.manager_lifecycle.advertising_measurement import (
    ADVERTISING_MEASUREMENT_SOURCE,
    AdvertisingMeasurementProvider,
    register_advertising_measurement_provider,
)
from tests.test_manager_lifecycle import _experiment


BASELINE_START = datetime(2026, 7, 16, tzinfo=timezone.utc)
BASELINE_END = datetime(2026, 7, 23, tzinfo=timezone.utc)
EVALUATION_START = datetime(2026, 7, 24, tzinfo=timezone.utc)
EVALUATION_END = datetime(2026, 7, 31, tzinfo=timezone.utc)


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


def _rows(*, campaigns: tuple[int, ...] = (10,)) -> list[dict]:
    result: list[dict] = []
    report_dates = [
        *(BASELINE_START.date() + timedelta(days=offset) for offset in range(7)),
        *(EVALUATION_START.date() + timedelta(days=offset) for offset in range(7)),
    ]
    for index, report_date in enumerate(report_dates):
        contribution = Decimal("100") if index < 7 else Decimal("130")
        for advert_id in campaigns:
            result.append(
                {
                    "report_date": report_date,
                    "advert_id": advert_id,
                    "nm_id": 20,
                    "sku_contribution_after_ads_rub": contribution,
                    "sku_contribution_after_ads_complete": True,
                    "finance_complete": True,
                    "finance_snapshot_complete": True,
                    "sku_ad_spend_complete": True,
                    "stats_day_watermark_complete": True,
                    "finance_last_synced_at": datetime.combine(
                        report_date + timedelta(days=1),
                        datetime.min.time(),
                        tzinfo=timezone.utc,
                    )
                    + timedelta(hours=10),
                    "stats_source_completed_at": datetime.combine(
                        report_date + timedelta(days=1),
                        datetime.min.time(),
                        tzinfo=timezone.utc,
                    )
                    + timedelta(hours=8),
                    "stats_last_synced_at": None,
                }
            )
    return result


def _collect(rows: list[dict], *, target: dict | None = None):
    connection = FakeConnection(rows)
    provider = AdvertisingMeasurementProvider(lambda: connection)
    experiment = replace(
        _experiment(state="measuring"),
        target=(
            target
            if target is not None
            else {
                "target_type": "campaign",
                "target_refs": ["20"],
                "operation": "pause_campaign",
                "payload": {"advert_id": 10},
            }
        ),
        measurement_spec={
            "primaryMetric": "contribution_profit_rub",
            "baselineDays": 7,
            "evaluationDays": 7,
            "attributionMethod": "observed_change",
        },
    )
    readout = provider.collect(
        experiment,
        baseline_start=BASELINE_START,
        baseline_end=BASELINE_END,
        evaluation_start=EVALUATION_START,
        evaluation_end=EVALUATION_END,
    )
    return readout, connection.cursor_instance.executed


def test_complete_windows_return_daily_mean_contribution_from_production_mart() -> None:
    readout, executed = _collect(_rows())

    assert readout.data_complete is True
    assert readout.disposition == "ready"
    assert readout.baseline_treated == 100
    assert readout.post_treated == 130
    assert readout.baseline_control is None
    assert readout.post_control is None
    assert readout.data_as_of == datetime(2026, 7, 31, 8, 0, tzinfo=timezone.utc)
    assert readout.metrics["baseline"]["totalRub"] == 700
    assert readout.metrics["evaluation"]["totalRub"] == 910
    assert readout.metrics["source"] == ADVERTISING_MEASUREMENT_SOURCE
    assert ADVERTISING_MEASUREMENT_SOURCE in executed[0]
    assert executed[1] == [[20], 10, date(2026, 7, 16), date(2026, 7, 31)]


def test_rows_from_a_different_campaign_are_rejected_fail_closed() -> None:
    readout, _ = _collect(_rows(campaigns=(10, 11)))

    assert readout.data_complete is False
    assert readout.disposition == "blocked"
    assert "unexpected_campaign_id" in readout.metrics["integrityReasons"]


def test_sql_binds_readout_to_the_campaign_that_was_paused() -> None:
    readout, executed = _collect(_rows())

    assert readout.data_complete is True
    assert "advert_id = %s::bigint" in executed[0]
    assert executed[1][1] == 10


def test_missing_day_waits_without_synthetic_zero() -> None:
    rows = [
        row
        for row in _rows()
        if row["report_date"] != date(2026, 7, 25)
    ]

    readout, _ = _collect(rows)

    assert readout.data_complete is False
    assert readout.disposition == "waiting_for_data"
    assert readout.baseline_treated is None
    assert readout.post_treated is None
    assert readout.metrics["reason"] == "missing_sku_days"
    assert readout.metrics["missingSkuDays"] == [
        {"reportDate": "2026-07-25", "nmId": 20}
    ]


@pytest.mark.parametrize(
    ("field", "reason"),
    [
        ("finance_complete", "finance_incomplete"),
        ("finance_snapshot_complete", "finance_snapshot_incomplete"),
        ("sku_ad_spend_complete", "ad_spend_incomplete"),
        ("stats_day_watermark_complete", "ad_stats_day_incomplete"),
    ],
)
def test_incomplete_finance_or_ad_spend_waits_without_values(
    field: str,
    reason: str,
) -> None:
    rows = _rows()
    rows[9][field] = False

    readout, _ = _collect(rows)

    assert readout.data_complete is False
    assert readout.disposition == "waiting_for_data"
    assert readout.baseline_treated is None
    assert readout.post_treated is None
    assert reason in readout.metrics["incompleteSkuDays"][0]["reasons"]


def test_missing_source_timestamp_never_claims_data_freshness() -> None:
    rows = _rows()
    rows[-1]["finance_last_synced_at"] = None

    readout, _ = _collect(rows)

    assert readout.data_complete is False
    assert readout.disposition == "waiting_for_data"
    assert readout.data_as_of is None
    assert readout.baseline_treated is None
    assert readout.post_treated is None
    assert "finance_source_timestamp_missing" in readout.metrics[
        "incompleteSkuDays"
    ][0]["reasons"]


def test_source_timestamp_before_completed_window_waits_without_values() -> None:
    rows = _rows()
    stale_at = EVALUATION_END - timedelta(minutes=1)
    for row in rows:
        row["finance_last_synced_at"] = stale_at
        row["stats_source_completed_at"] = stale_at

    readout, _ = _collect(rows)

    assert readout.data_complete is False
    assert readout.disposition == "waiting_for_data"
    assert readout.data_as_of is None
    assert readout.baseline_treated is None
    assert readout.post_treated is None
    assert readout.metrics["reason"] == "source_freshness_lag"
    assert readout.metrics["sourceDataAsOf"] == stale_at.isoformat()


def test_one_stale_final_day_source_cannot_be_hidden_by_a_fresher_source() -> None:
    rows = _rows()
    stale_at = EVALUATION_END - timedelta(minutes=1)
    final_rows = [row for row in rows if row["report_date"] == date(2026, 7, 30)]
    final_rows[0]["stats_source_completed_at"] = stale_at
    final_rows[0]["finance_last_synced_at"] = EVALUATION_END + timedelta(hours=2)

    readout, _ = _collect(rows)

    assert readout.data_complete is False
    assert readout.disposition == "waiting_for_data"
    assert readout.metrics["reason"] == "source_freshness_lag"
    assert readout.metrics["sourceDataAsOf"] == stale_at.isoformat()


def test_missing_target_sku_is_terminally_blocked_without_query() -> None:
    readout, executed = _collect(
        _rows(),
        target={
            "target_type": "campaign",
            "target_refs": [],
            "operation": "pause_campaign",
            "payload": {"advert_id": 10},
        },
    )

    assert executed is None
    assert readout.data_complete is False
    assert readout.disposition == "blocked"
    assert readout.baseline_treated is None
    assert readout.post_treated is None
    assert readout.metrics["reason"] == "missing_target_sku"


def test_duplicate_mart_grain_is_terminally_blocked() -> None:
    rows = _rows()
    rows.append(dict(rows[0]))

    readout, _ = _collect(rows)

    assert readout.data_complete is False
    assert readout.disposition == "blocked"
    assert readout.baseline_treated is None
    assert readout.post_treated is None
    assert readout.metrics["reason"] == "measurement_source_integrity_error"
    assert "duplicate_mart_grain" in readout.metrics["integrityReasons"]


def test_advertising_provider_registration_populates_production_registry() -> None:
    with patch.dict(jobs._MEASUREMENT_PROVIDERS, {}, clear=True):
        provider = register_advertising_measurement_provider()

        assert jobs._MEASUREMENT_PROVIDERS["advertising"] is provider
        assert isinstance(provider, AdvertisingMeasurementProvider)
