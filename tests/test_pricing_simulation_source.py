from __future__ import annotations

from datetime import date, timedelta
import math

import numpy as np
import pytest

from agi_ecom_app.decision_simulation import pricing_source
from agi_ecom_app.decision_simulation.pricing_engine import PricingSimulationEngine
from agi_ecom_app.decision_simulation.pricing_models import (
    PricingCompletedDayObservationV1,
)
from agi_ecom_app.decision_simulation.pricing_source import (
    build_pricing_candidates,
    build_pricing_simulation_request,
    estimate_contextual_demand_bootstrap,
    estimate_empirical_elasticity_bootstrap,
    pricing_request_snapshot_payload,
)


def _history(*, inverse: bool = True, count: int = 42) -> list[dict[str, object]]:
    start = date(2026, 4, 1)
    prices = [90.0, 100.0, 110.0]
    rows: list[dict[str, object]] = []
    for index in range(count):
        price = prices[(index // 7) % len(prices)]
        units = (26 - price / 10) if inverse else (price / 10 + 2)
        rows.append(
            {
                "report_date": start + timedelta(days=index),
                "seller_effective_price_rub": price,
                "non_cancelled_order_units": units,
                "shadow_demand_observation_usable": True,
                "observed_out_of_stock": False,
                "orders_day_complete": True,
                "sales_day_complete": True,
                "finance_values_complete": True,
                "marketing_source_row_present": True,
                "realized_sales_units": units,
                "gross_revenue_rub": units * price,
                "net_revenue_rub": units * price * 0.9,
                "payout_total_rub": units * price * 0.75,
                "marketplace_costs_rub": units * price * 0.1,
                "cogs_total_rub": units * 25,
                "ad_spend_observed_partial_rub": units * price * 0.05,
            }
        )
    return rows


def _observations(*, inverse: bool = True, count: int = 42):
    return [
        PricingCompletedDayObservationV1(
            observedDate=row["report_date"],
            priceRub=row["seller_effective_price_rub"],
            unitsSold=row["non_cancelled_order_units"],
            evidenceRef=f"day:{index}",
        )
        for index, row in enumerate(_history(inverse=inverse, count=count))
    ]


def test_empirical_elasticity_bootstrap_is_deterministic_and_not_a_fallback() -> None:
    observations = _observations()

    first = estimate_empirical_elasticity_bootstrap(observations)
    second = estimate_empirical_elasticity_bootstrap(observations)

    assert first == second
    assert len(first) == 200
    assert all(value < 0 for value in first)


def test_contextual_bootstrap_separates_ad_traffic_from_price_elasticity() -> None:
    start = date(2026, 1, 1)
    observations: list[PricingCompletedDayObservationV1] = []
    prices = [90.0, 100.0, 110.0]
    ad_views = [0.0, 20.0, 80.0, 200.0]
    for index in range(56):
        price = prices[index % len(prices)]
        views = ad_views[(index // len(prices)) % len(ad_views)]
        log_units = 7.0 - math.log(price) + 0.45 * math.log1p(views)
        observations.append(
            PricingCompletedDayObservationV1(
                observedDate=start + timedelta(days=index),
                priceRub=price,
                unitsSold=math.exp(log_units) - 0.5,
                contextValues={"ad_views": views},
                evidenceRef=f"context-day:{index}",
            )
        )

    fit = estimate_contextual_demand_bootstrap(observations)

    assert "ad_views" in fit.active_features
    assert len(fit.elasticity_values) == 200
    assert float(np.median(fit.elasticity_values)) == pytest.approx(-1.0, abs=0.05)
    assert float(np.median(fit.coefficient_values["ad_views"])) > 0


def test_positive_historical_relationship_is_retained_and_fails_hard_gate() -> None:
    request = build_pricing_simulation_request(
        nm_id=42,
        current_price_row={
            "snapshot_date": "2026-07-16",
            "seller_effective_price_rub": 100,
        },
        history_rows=_history(inverse=False),
        stock_row={"snapshot_date": "2026-07-16", "available_stock": 500},
    )

    values = request.evidence.elasticity.bootstrap_values
    assert values is not None
    assert any(value >= 0 for value in values)
    violations = PricingSimulationEngine().validate_candidate(
        request,
        request.candidates[0],
    )
    assert "invalid_price_elasticity" in {item.code for item in violations}


def test_request_builder_creates_hold_and_four_safe_shadow_prices() -> None:
    request = build_pricing_simulation_request(
        nm_id=42,
        current_price_row={
            "snapshot_date": "2026-07-16",
            "seller_effective_price_rub": 100,
        },
        history_rows=_history(),
        stock_row={"snapshot_date": "2026-07-16", "available_stock": 500},
    )

    assert len(request.candidates) == 5
    assert request.candidates[0].operation == "hold_price"
    assert request.candidates[0].price_rub == 100
    assert {item.price_rub for item in request.candidates} == {95, 98, 100, 102, 105}
    assert request.evidence.unit_economics.net_revenue_rate == pytest.approx(0.75)
    assert request.evidence.unit_economics.variable_cost_per_unit_rub == pytest.approx(25)
    assert request.evidence.unit_economics.observation_count == 42
    assert request.evidence.realization.bootstrap_values is not None
    assert len(request.evidence.realization.bootstrap_values) == 200
    assert request.evidence.realization.bootstrap_values == pytest.approx([1] * 200)
    assert request.evidence.available_stock.units == 500


def test_insufficient_history_remains_missing_instead_of_using_default_elasticity() -> None:
    request = build_pricing_simulation_request(
        nm_id=42,
        current_price_row={
            "snapshot_date": "2026-07-16",
            "seller_effective_price_rub": 100,
        },
        history_rows=_history(count=14),
        stock_row={"snapshot_date": "2026-07-16", "available_stock": 500},
    )

    assert request.evidence.elasticity.bootstrap_values is None
    assert request.evidence.elasticity.source is None
    violations = PricingSimulationEngine().validate_candidate(
        request,
        request.candidates[0],
    )
    assert "missing_elasticity_source" in {item.code for item in violations}


def test_explicit_candidates_are_unique_and_snapshot_payload_omits_hash() -> None:
    candidates = build_pricing_candidates(
        nm_id=42,
        current_price_rub=100,
        candidate_prices=[90, 90, 110],
    )
    assert [item.price_rub for item in candidates] == [100, 90, 110]
    request = build_pricing_simulation_request(
        nm_id=42,
        current_price_row={
            "snapshot_date": "2026-07-16",
            "seller_effective_price_rub": 100,
        },
        history_rows=_history(),
        stock_row={"snapshot_date": "2026-07-16", "available_stock": 500},
        candidate_prices=[90, 110],
    )
    payload = pricing_request_snapshot_payload(request)
    assert "snapshotHash" not in payload
    assert payload["decisionGroupId"] == "pricing:sku:42"


def test_live_loader_reads_a_wider_calendar_window_before_eligibility_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executed: list[tuple[str, list[object]]] = []
    result_sets = iter(
        [
            [
                {
                    "nm_id": 42,
                    "snapshot_date": "2026-07-16",
                    "seller_effective_price_rub": 100,
                    "last_synced_at": "2026-07-17T00:00:00Z",
                }
            ],
            [
                {
                    "report_date": "2026-07-15",
                    "shadow_demand_observation_usable": False,
                }
            ],
            [
                {
                    "snapshot_date": "2026-07-16",
                    "available_stock": 10,
                    "market_context_history": [
                        {
                            "snapshot_date": "2026-07-16",
                            "market_price_median": 771,
                        },
                        {
                            "snapshot_date": "2026-07-15",
                            "market_price_median": 767,
                        },
                    ],
                }
            ],
        ]
    )

    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def execute(self, sql, params):  # type: ignore[no-untyped-def]
            executed.append((sql, list(params)))

        def fetchall(self):
            return next(result_sets)

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def cursor(self):
            return Cursor()

    sentinel = object()
    monkeypatch.setattr(pricing_source, "get_connection", lambda: Connection())
    monkeypatch.setattr(
        pricing_source,
        "build_pricing_simulation_request",
        lambda **kwargs: sentinel,
    )

    request, metadata = pricing_source.load_pricing_simulation_source(nm_id=42)

    assert request is sentinel
    assert metadata["historyRowCount"] == 1
    assert executed[1][1] == [
        42,
        date(2026, 7, 16),
        pricing_source.MAX_SOURCE_ROWS,
    ]
    assert "price_day.report_date < %s" in executed[1][0]
    assert "sales_day.report_date = price_day.report_date - 1" in executed[1][0]
    assert "marketing_day.report_date = price_day.report_date - 1" in executed[1][0]
    assert "price_context.snapshot_date = price_day.report_date - 1" in executed[1][0]
    assert "supply.status_id in (2, 3)" in executed[2][0]
    assert "current_market_history" in executed[2][0]
    assert metadata["marketContextHistory"][0]["market_price_median"] == 771
    assert pricing_source.MAX_SOURCE_ROWS > pricing_source.MAX_OBSERVATION_DAYS


def test_future_promotion_changes_only_confirmed_scheduled_days() -> None:
    history = _history()
    for index, row in enumerate(history):
        active = index % 5 == 0
        row["context_promotion_active"] = active
        row["context_promotion_discount_pct"] = 15 if active else 0

    request = build_pricing_simulation_request(
        nm_id=42,
        current_price_row={
            "snapshot_date": "2026-07-16",
            "seller_effective_price_rub": 100,
        },
        history_rows=history,
        stock_row={
            "snapshot_date": "2026-07-16",
            "available_stock": 500,
            "future_promotion_events": [
                {
                    "promotion_id": 7,
                    "start_date": "2026-07-18",
                    "end_date": "2026-07-19",
                    "in_action": True,
                    "discount_pct": 20,
                }
            ],
        },
        horizon_days=4,
    )

    context = request.evidence.forecast_context
    assert context is not None
    features = {item.name: item for item in context.features}
    assert features["promotion_active"].status == "modeled"
    assert features["promotion_active"].forward_values == [0, 1, 1, 0]
    assert features["promotion_discount_pct"].forward_values == [0, 20, 20, 0]


def test_missing_or_invalid_future_promotion_stays_unknown_not_zero() -> None:
    active, discount = pricing_source._future_promotion_paths(
        snapshot_date=date(2026, 7, 16),
        horizon_days=3,
        raw_events=[
            {
                "start_date": "2026-07-18",
                "end_date": "2026-07-17",
                "in_action": True,
                "discount_pct": 20,
            },
            {
                "start_date": "2026-07-18",
                "end_date": "2026-07-19",
                "in_action": False,
                "discount_pct": 30,
            },
        ],
    )

    assert active == [None, None, None]
    assert discount == [None, None, None]


def test_only_confirmed_future_supply_inside_horizon_enters_capacity() -> None:
    request = build_pricing_simulation_request(
        nm_id=42,
        current_price_row={
            "snapshot_date": "2026-07-16",
            "seller_effective_price_rub": 100,
        },
        history_rows=_history(),
        stock_row={
            "snapshot_date": "2026-07-16",
            "available_stock": 500,
            "future_inbound_events": [
                {
                    "supply_id": "scheduled",
                    "eta_date": "2026-07-18",
                    "status_id": 2,
                    "fact_date": None,
                    "units": 30,
                    "snapshot_date": "2026-07-15",
                },
                {
                    "supply_id": "same-day",
                    "eta_date": "2026-07-16",
                    "status_id": 3,
                    "fact_date": None,
                    "units": 20,
                    "snapshot_date": "2026-07-15",
                },
                {
                    "supply_id": "completed",
                    "eta_date": "2026-07-19",
                    "status_id": 5,
                    "fact_date": "2026-07-19T10:00:00Z",
                    "units": 40,
                    "snapshot_date": "2026-07-15",
                },
                {
                    "supply_id": "outside",
                    "eta_date": "2026-08-01",
                    "status_id": 2,
                    "fact_date": None,
                    "units": 50,
                    "snapshot_date": "2026-07-15",
                },
            ],
        },
        horizon_days=7,
    )

    events = {
        item.supply_id: item for item in request.evidence.available_stock.future_inbound
    }
    assert events["scheduled"].status == "scheduled"
    assert events["scheduled"].include_in_capacity is True
    assert events["same-day"].status == "overdue"
    assert events["same-day"].include_in_capacity is False
    assert events["completed"].status == "unconfirmed"
    assert events["completed"].include_in_capacity is False
    assert events["outside"].status == "scheduled"
    assert events["outside"].include_in_capacity is False
