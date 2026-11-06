from __future__ import annotations

from datetime import date, timedelta
import math
import time

import numpy as np
import pytest
from pydantic import ValidationError

from agi_ecom_app.decision_simulation.models import (
    DistributionPairV1,
    DistributionSummaryV1,
    HistogramSummaryV1,
)
from agi_ecom_app.decision_simulation.pricing_engine import (
    MAX_DEMAND_OBSERVATION_AGE_DAYS,
    MAX_STOCK_AGE_DAYS,
    MAX_UNIT_ECONOMICS_AGE_DAYS,
    PricingSimulationEngine,
    derive_pricing_batch_seed,
    run_pricing_simulation,
    simulate_pricing_batch_values,
)
from agi_ecom_app.decision_simulation.pricing_models import (
    PricingAvailableStockEvidenceV1,
    PricingCandidateSimulationResultV1,
    PricingCandidateV1,
    PricingCompletedDayObservationV1,
    PricingContextFeatureEvidenceV1,
    PricingDailyObservationsEvidenceV1,
    PricingElasticityEvidenceV1,
    PricingRealizationEvidenceV1,
    PricingForecastContextEvidenceV1,
    PricingFutureInboundEventV1,
    PricingRankingScoreV1,
    PricingSimulationBatchResultV1,
    PricingSimulationEvidenceV1,
    PricingSimulationRequestV1,
    PricingUnitEconomicsEvidenceV1,
)


SNAPSHOT_HASH = "b" * 64


def _candidate(
    suffix: str,
    price_rub: float,
    *,
    operation: str = "set_price",
    action_id: str | None = None,
) -> PricingCandidateV1:
    return PricingCandidateV1(
        candidate_id=f"price-{suffix}",
        action_id=action_id or f"action-{suffix}",
        operation=operation,
        price_rub=price_rub,
    )


def _observations(
    *,
    count: int = 28,
    units: list[float] | None = None,
    completed_count: int | None = None,
) -> list[PricingCompletedDayObservationV1]:
    values = units or [8, 10, 12, 9, 11, 14, 7]
    completed_count = count if completed_count is None else completed_count
    start = date(2026, 1, 1)
    return [
        PricingCompletedDayObservationV1(
            observed_date=start + timedelta(days=index),
            price_rub=100 + (index % 3 - 1) * 2,
            units_sold=values[index % len(values)],
            completed=index < completed_count,
            not_observed_out_of_stock=index < completed_count,
            evidence_ref=f"price-day:{index}",
        )
        for index in range(count)
    ]


def _request(
    *candidates: PricingCandidateV1,
    observations: list[PricingCompletedDayObservationV1] | None = None,
    elasticity_source: str = "historical_regression",
    elasticity_values: list[float] | None = None,
    realization_values: list[float] | None = None,
    available_stock: float = 1_000,
    net_revenue_rate: float = 0.8,
    variable_cost: float = 20,
) -> PricingSimulationRequestV1:
    return PricingSimulationRequestV1(
        snapshot_hash=SNAPSHOT_HASH,
        decision_group_id="price:sku-42",
        sku_id="sku-42",
        current_price_rub=100,
        current_price_evidence_ref="wb-price:sku-42:2026-01-15",
        candidates=list(candidates),
        evidence=PricingSimulationEvidenceV1(
            daily_observations=PricingDailyObservationsEvidenceV1(
                values=observations if observations is not None else _observations(),
                evidence_ref="mart:price-sku-daily:sku-42",
                latest_observation_age_days=1,
            ),
            elasticity=PricingElasticityEvidenceV1(
                bootstrap_values=(
                    elasticity_values
                    if elasticity_values is not None
                    else [-1.2, -1.0, -0.8, -1.1, -0.9] * 20
                ),
                source=elasticity_source,
                evidence_ref="price-experiment:sku-42:v3",
            ),
            realization=PricingRealizationEvidenceV1(
                bootstrap_values=(
                    realization_values
                    if realization_values is not None
                    else [1.0] * 100
                ),
                source="cohort_order_to_realized_sale_day_bootstrap_v1",
                evidence_ref="sales-realization:sku-42:v1",
            ),
            unit_economics=PricingUnitEconomicsEvidenceV1(
                net_revenue_rate=net_revenue_rate,
                variable_cost_per_unit_rub=variable_cost,
                observation_count=28,
                latest_observation_age_days=1,
                basis="wb_net_payout_before_cogs",
                evidence_ref="finance-margin:sku-42:2026-01-15",
            ),
            available_stock=PricingAvailableStockEvidenceV1(
                units=available_stock,
                snapshot_date=date(2026, 1, 15),
                age_days_at_snapshot=0,
                evidence_ref="stock:sku-42:2026-01-15",
            ),
        ),
    )


def _default_request(**overrides: object) -> PricingSimulationRequestV1:
    return _request(
        _candidate("hold", 100, operation="hold_price"),
        _candidate("down", 90),
        _candidate("up", 110),
        **overrides,
    )


def test_one_sku_run_has_500_samples_in_ten_batches_and_a_hold_baseline() -> None:
    result = run_pricing_simulation(_default_request())

    assert result.sample_count == 500
    assert result.batch_size == 50
    assert result.candidate_count == 3
    assert result.completed_candidate_count == 3
    assert result.winner_candidate_id == "price-up"
    hold = result.candidates[0]
    assert hold.operation == "hold_price"
    assert hold.distribution is not None
    assert hold.distribution.total.mean == 0
    assert hold.distribution.per_day.cvar_10 == 0
    assert hold.distribution.per_day.probability_positive == 0
    for candidate in result.candidates:
        assert len(candidate.batch_results) == 10
        assert candidate.distribution is not None
        assert candidate.distribution.total.count == 500
        assert candidate.units_sold_distribution is not None
        assert candidate.units_sold_distribution.total.count == 500


def test_pricing_simulation_supports_sixty_day_portfolio_horizon() -> None:
    request = PricingSimulationRequestV1.model_validate(
        {
            **_default_request().model_dump(mode="json", by_alias=True),
            "horizonDays": 60,
        }
    )

    result = run_pricing_simulation(request)

    assert request.horizon_days == 60
    assert result.sample_count == 500
    assert all(candidate.distribution is not None for candidate in result.candidates)


def test_batch_seed_is_shared_by_all_candidates_and_whole_run_is_reproducible() -> None:
    request = _default_request()

    first = run_pricing_simulation(request)
    second = run_pricing_simulation(request)

    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    expected = [
        derive_pricing_batch_seed(SNAPSHOT_HASH, "price:sku-42", index)
        for index in range(10)
    ]
    assert len(set(expected)) == 10
    for candidate in first.candidates:
        assert [batch.seed for batch in candidate.batch_results] == expected


def test_aggregate_replays_all_ten_group_seed_batches_exactly() -> None:
    request = _default_request()
    result = run_pricing_simulation(request)
    candidate = request.candidates[2]
    aggregate = result.candidates[2].distribution
    assert aggregate is not None

    replayed = np.concatenate(
        [
            simulate_pricing_batch_values(request, candidate, index)
            .incremental_contribution_profit
            for index in range(10)
        ]
    )
    sorted_values = np.sort(replayed / request.horizon_days)
    assert aggregate.total.mean == pytest.approx(float(np.mean(replayed)))
    assert aggregate.per_day.p10 == pytest.approx(
        float(np.quantile(replayed / request.horizon_days, 0.1, method="linear"))
    )
    assert aggregate.per_day.cvar_10 == pytest.approx(
        float(np.mean(sorted_values[: math.ceil(500 * 0.1)]))
    )


def test_stock_cap_is_applied_before_incremental_contribution() -> None:
    observations = _observations(count=28, units=[100])
    request = _default_request(
        observations=observations,
        elasticity_values=[-1] * 100,
        available_stock=10,
        net_revenue_rate=1,
        variable_cost=20,
    )

    result = run_pricing_simulation(request)
    hold, down, up = result.candidates

    assert hold.units_sold_distribution is not None
    assert hold.units_sold_distribution.total.mean == 10
    assert down.units_sold_distribution is not None
    assert down.units_sold_distribution.total.mean == 10
    assert up.units_sold_distribution is not None
    assert up.units_sold_distribution.total.mean == 10
    assert down.distribution is not None
    assert down.distribution.total.mean == -100
    assert up.distribution is not None
    assert up.distribution.total.mean == 100
    assert up.stockout_probability == 1
    assert result.winner_candidate_id == "price-up"


def test_context_feature_changes_forward_demand_without_changing_price_grid() -> None:
    observations = _observations(count=28, units=[10])
    observations = [
        item.model_copy(update={"context_values": {"ad_views": 0.0}})
        for item in observations
    ]
    baseline = _default_request(
        observations=observations,
        elasticity_values=[-1.0] * 100,
        realization_values=[1.0] * 100,
        available_stock=10_000,
    )
    feature = PricingContextFeatureEvidenceV1(
        name="ad_views",
        transform="identity",
        status="modeled",
        observedDays=28,
        coverageRatio=1.0,
        center=0.0,
        scale=1.0,
        bootstrapCoefficients=[math.log(2.0)] * 100,
        forwardValues=[1.0] * baseline.horizon_days,
        latestValue=0.0,
        evidenceRef="ads:sku-42",
    )
    context = PricingForecastContextEvidenceV1(
        snapshotDate=date(2026, 1, 15),
        features=[feature],
    )
    contextual = baseline.model_copy(
        update={
            "evidence": baseline.evidence.model_copy(
                update={"forecast_context": context}
            )
        }
    )

    baseline_result = run_pricing_simulation(baseline)
    contextual_result = run_pricing_simulation(contextual)

    baseline_hold = baseline_result.candidates[0].units_sold_distribution
    contextual_hold = contextual_result.candidates[0].units_sold_distribution
    assert baseline_hold is not None and contextual_hold is not None
    assert contextual_hold.total.mean == pytest.approx(
        baseline_hold.total.mean * 2,
        rel=1e-10,
    )
    assert [item.price_rub for item in contextual.candidates] == [100, 90, 110]


def test_supplier_inbound_becomes_sellable_only_on_eta() -> None:
    baseline = _default_request(
        observations=_observations(count=28, units=[8]),
        elasticity_values=[-1.0] * 100,
        realization_values=[1.0] * 100,
        available_stock=10,
    )
    baseline = PricingSimulationRequestV1.model_validate(
        {
            **baseline.model_dump(mode="json", by_alias=True),
            "horizonDays": 3,
        }
    )
    stock = baseline.evidence.available_stock.model_copy(
        update={
            "future_inbound": [
                PricingFutureInboundEventV1(
                    etaDate=date(2026, 1, 18),
                    units=20,
                    includeInCapacity=True,
                    status="scheduled",
                    supplyId="supply-1",
                    evidenceRef="supply:1",
                )
            ]
        }
    )
    with_inbound = baseline.model_copy(
        update={
            "evidence": baseline.evidence.model_copy(
                update={"available_stock": stock}
            )
        }
    )

    baseline_result = run_pricing_simulation(baseline)
    inbound_result = run_pricing_simulation(with_inbound)

    baseline_hold = baseline_result.candidates[0].units_sold_distribution
    inbound_hold = inbound_result.candidates[0].units_sold_distribution
    assert baseline_hold is not None and inbound_hold is not None
    assert baseline_hold.total.mean == 10
    # Jan 16 sells 8, Jan 17 sells the remaining 2, and only Jan 18 may use
    # the arriving units. Treating the inbound as day-one stock would sell 24.
    assert inbound_hold.total.mean == pytest.approx(18, abs=0.01)
    assert inbound_hold.total.mean < 20


def test_legacy_request_without_context_or_inbound_replays_original_v1_math() -> None:
    request = _default_request(
        realization_values=[0.73] * 100,
        available_stock=10_000,
    )
    candidate = request.candidates[1]
    batch = simulate_pricing_batch_values(request, candidate, 0)

    observations = request.evidence.daily_observations.values or []
    elasticity_values = request.evidence.elasticity.bootstrap_values or []
    seed = derive_pricing_batch_seed(
        request.snapshot_hash,
        request.decision_group_id,
        0,
    )
    generator = np.random.Generator(np.random.PCG64(seed))
    observation_indices = generator.integers(
        0,
        len(observations),
        size=(request.batch_size, request.horizon_days),
    )
    elasticity_indices = generator.integers(
        0,
        len(elasticity_values),
        size=request.batch_size,
    )
    realization_indices = generator.integers(0, 100, size=request.batch_size)
    observed_prices = np.asarray([item.price_rub for item in observations])[
        observation_indices
    ]
    observed_units = np.asarray([item.units_sold for item in observations])[
        observation_indices
    ]
    sampled_elasticity = np.asarray(elasticity_values)[elasticity_indices][
        :, np.newaxis
    ]
    sampled_realization = np.asarray([0.73] * 100)[realization_indices]
    expected_orders = np.sum(
        observed_units
        * np.power(candidate.price_rub / observed_prices, sampled_elasticity),
        axis=1,
        dtype=np.float64,
    )
    expected_units = np.minimum(expected_orders * sampled_realization, 10_000)

    assert np.array_equal(batch.units_sold, expected_units)


def test_invalid_context_values_fail_closed_before_sampling() -> None:
    request = _default_request()
    observations = [
        item.model_copy(update={"context_values": {"ad_views": -1.0}})
        for item in (request.evidence.daily_observations.values or [])
    ]
    feature = PricingContextFeatureEvidenceV1(
        name="ad_views",
        transform="log1p",
        status="modeled",
        observedDays=28,
        coverageRatio=1.0,
        center=0,
        scale=1,
        bootstrapCoefficients=[0.1] * 100,
        forwardValues=[10.0] * request.horizon_days,
        evidenceRef="ads:sku-42",
    )
    context = PricingForecastContextEvidenceV1(
        snapshotDate=date(2026, 1, 15),
        features=[feature],
    )
    contextual = request.model_copy(
        update={
            "evidence": request.evidence.model_copy(
                update={
                    "daily_observations": request.evidence.daily_observations.model_copy(
                        update={"values": observations}
                    ),
                    "forecast_context": context,
                }
            )
        }
    )

    violations = PricingSimulationEngine().validate_candidate(
        contextual,
        contextual.candidates[0],
    )

    assert "invalid_context_observation_value" in {
        item.code for item in violations
    }


def test_past_or_duplicate_inbound_fails_closed_and_is_not_double_counted() -> None:
    request = _default_request()
    stock = request.evidence.available_stock.model_copy(
        update={
            "future_inbound": [
                PricingFutureInboundEventV1(
                    etaDate=date(2026, 1, 14),
                    units=20,
                    includeInCapacity=True,
                    status="scheduled",
                    supplyId="supply-1",
                    evidenceRef="supply:1",
                ),
                PricingFutureInboundEventV1(
                    etaDate=date(2026, 1, 14),
                    units=20,
                    includeInCapacity=True,
                    status="scheduled",
                    supplyId="supply-1",
                    evidenceRef="supply:1:duplicate",
                ),
            ]
        }
    )
    invalid = request.model_copy(
        update={
            "evidence": request.evidence.model_copy(
                update={"available_stock": stock}
            )
        }
    )

    violations = PricingSimulationEngine().validate_candidate(
        invalid,
        invalid.candidates[0],
    )
    codes = {item.code for item in violations}

    assert "past_future_inbound_eta" in codes
    assert "duplicate_future_inbound_event" in codes


def test_missing_context_is_neutral_and_not_serialized_as_observed_zero() -> None:
    baseline = _default_request()
    missing_feature = PricingContextFeatureEvidenceV1(
        name="lagged_competitor_median_price_rub",
        status="missing",
        observedDays=0,
        coverageRatio=0.0,
        forwardValues=[None] * baseline.horizon_days,
        dataGap="context_lagged_competitor_median_price_rub_missing",
    )
    context = PricingForecastContextEvidenceV1(
        snapshotDate=date(2026, 1, 15),
        features=[missing_feature],
        dataGaps=["context_lagged_competitor_median_price_rub_missing"],
    )
    contextual = baseline.model_copy(
        update={
            "evidence": baseline.evidence.model_copy(
                update={"forecast_context": context}
            )
        }
    )

    baseline_result = run_pricing_simulation(baseline)
    contextual_result = run_pricing_simulation(contextual)

    assert baseline_result.model_dump(mode="json") == contextual_result.model_dump(
        mode="json"
    )
    payload = contextual.model_dump(mode="json", by_alias=True)
    feature_payload = payload["evidence"]["forecastContext"]["features"][0]
    assert feature_payload["latestValue"] is None
    assert feature_payload["forwardValues"] == [None] * baseline.horizon_days


def test_realization_distribution_converts_orders_before_profit_and_stock_cap() -> None:
    request = _default_request(
        observations=_observations(count=28, units=[100]),
        elasticity_values=[-1] * 100,
        realization_values=[0.5] * 100,
        available_stock=1_000,
        net_revenue_rate=1,
        variable_cost=20,
    )

    result = run_pricing_simulation(request)
    hold, _, up = result.candidates

    assert hold.units_sold_distribution is not None
    hold_units = hold.units_sold_distribution.total.mean
    assert hold_units == pytest.approx(350, rel=0.01)
    assert up.units_sold_distribution is not None
    assert up.units_sold_distribution.total.mean == pytest.approx(hold_units / 1.1)
    assert up.distribution is not None
    assert up.distribution.total.mean == pytest.approx(
        (hold_units / 1.1) * 90 - hold_units * 80
    )


def test_missing_realization_evidence_fails_closed_without_sampling() -> None:
    request = _default_request()
    request.evidence.realization = PricingRealizationEvidenceV1()

    result = run_pricing_simulation(request)

    expected = {
        "missing_realization_evidence_ref",
        "missing_realization_source",
        "missing_realization_bootstrap_values",
    }
    assert result.completed_candidate_count == 0
    assert result.winner_candidate_id is None
    for candidate in result.candidates:
        assert candidate.status == "needs_data"
        assert candidate.batch_results == []
        assert expected <= {
            violation.code for violation in candidate.constraint_violations
        }


@pytest.mark.parametrize(
    ("evidence_path", "stale_age", "expected_code"),
    [
        (
            "demand",
            MAX_DEMAND_OBSERVATION_AGE_DAYS + 1,
            "stale_demand_observations",
        ),
        (
            "unit_economics",
            MAX_UNIT_ECONOMICS_AGE_DAYS + 1,
            "stale_unit_economics",
        ),
        (
            "stock",
            MAX_STOCK_AGE_DAYS + 1,
            "stale_available_stock",
        ),
    ],
)
def test_stale_pricing_evidence_fails_closed_without_sampling(
    evidence_path: str,
    stale_age: int,
    expected_code: str,
) -> None:
    request = _default_request()
    if evidence_path == "demand":
        request.evidence.daily_observations.latest_observation_age_days = stale_age
    elif evidence_path == "unit_economics":
        request.evidence.unit_economics.latest_observation_age_days = stale_age
    else:
        request.evidence.available_stock.age_days_at_snapshot = stale_age

    result = run_pricing_simulation(request)

    assert result.completed_candidate_count == 0
    assert result.winner_candidate_id is None
    for candidate in result.candidates:
        assert candidate.status == "needs_data"
        assert candidate.batch_results == []
        assert expected_code in {
            violation.code for violation in candidate.constraint_violations
        }


def test_price_move_bound_is_inclusive_at_ten_percent_and_blocks_beyond_it() -> None:
    request = _request(
        _candidate("below-bound", 89),
        _candidate("down-boundary", 90),
        _candidate("hold", 100, operation="hold_price"),
        _candidate("up-boundary", 110),
        _candidate("above-bound", 111),
    )

    result = run_pricing_simulation(request)
    by_id = {candidate.candidate_id: candidate for candidate in result.candidates}

    assert by_id["price-down-boundary"].status == "completed"
    assert by_id["price-up-boundary"].status == "completed"
    for candidate_id in ("price-below-bound", "price-above-bound"):
        candidate = by_id[candidate_id]
        assert candidate.status == "blocked"
        assert candidate.batch_results == []
        assert "price_move_out_of_v1_bounds" in {
            violation.code for violation in candidate.constraint_violations
        }


def test_fallback_elasticity_blocks_every_candidate_without_sampling() -> None:
    result = run_pricing_simulation(
        _default_request(elasticity_source="fallback_default")
    )

    assert result.completed_candidate_count == 0
    assert result.winner_candidate_id is None
    for candidate in result.candidates:
        assert candidate.status == "blocked"
        assert candidate.batch_results == []
        assert "fallback_elasticity_not_allowed" in {
            item.code for item in candidate.constraint_violations
        }


def test_extreme_observational_elasticity_requires_controlled_experiment() -> None:
    result = run_pricing_simulation(
        _default_request(elasticity_values=[-12.0] * 100)
    )

    assert result.completed_candidate_count == 0
    assert result.winner_candidate_id is None
    for candidate in result.candidates:
        assert candidate.status == "blocked"
        assert "implausible_price_elasticity_magnitude" in {
            item.code for item in candidate.constraint_violations
        }


def test_fewer_than_seven_completed_in_stock_days_is_needs_data() -> None:
    observations = _observations(count=10, completed_count=6)

    result = run_pricing_simulation(
        _default_request(observations=observations)
    )

    assert result.completed_candidate_count == 0
    assert all(item.status == "needs_data" for item in result.candidates)
    assert all(
        "insufficient_completed_day_observations"
        in {violation.code for violation in item.constraint_violations}
        for item in result.candidates
    )


def test_candidate_below_unit_contribution_floor_is_rejected_only_for_that_price() -> None:
    request = _default_request(net_revenue_rate=0.5, variable_cost=48)
    request.evidence.unit_economics.minimum_unit_contribution_rub = 0

    result = run_pricing_simulation(request)

    assert result.candidates[1].status == "blocked"
    assert "minimum_unit_contribution_violated" in {
        item.code for item in result.candidates[1].constraint_violations
    }
    assert result.candidates[0].status == "completed"
    assert result.candidates[2].status == "completed"


def _fixed_distribution() -> DistributionPairV1:
    histogram = HistogramSummaryV1(bin_edges=[0, 1], counts=[50])
    summary = DistributionSummaryV1(
        count=50,
        mean=1,
        p05=1,
        p10=1,
        p25=1,
        p50=1,
        p75=1,
        p90=1,
        p95=1,
        cvar_10=1,
        probability_positive=1,
        histogram=histogram,
    )
    return DistributionPairV1(total=summary, per_day=summary)


def test_ranking_uses_cvar_then_p10_mean_smaller_move_and_action_id() -> None:
    candidates = [
        _candidate("hold", 100, operation="hold_price", action_id="action-hold"),
        _candidate("low", 90, action_id="action-z"),
        _candidate("high", 110, action_id="action-a"),
        _candidate("deep-low", 80, action_id="action-deep-low"),
        _candidate("deep-high", 120, action_id="action-deep-high"),
    ]
    request = _request(*candidates)
    scores = {
        "price-hold": (10, 20, 30),
        "price-low": (10, 20, 30),
        "price-high": (10, 20, 30),
        "price-deep-low": (10, 21, 0),
        "price-deep-high": (11, 0, 0),
    }

    class FixedScoreEngine(PricingSimulationEngine):
        def simulate_candidate(self, request, candidate):  # type: ignore[no-untyped-def]
            cvar, p10, mean = scores[candidate.candidate_id]
            distribution = _fixed_distribution()
            return PricingCandidateSimulationResultV1(
                candidate_id=candidate.candidate_id,
                action_id=candidate.action_id,
                operation=candidate.operation,
                price_rub=candidate.price_rub,
                status="completed",
                evidence_refs=["evidence:test"],
                batch_results=[
                    PricingSimulationBatchResultV1(
                        candidate_id=candidate.candidate_id,
                        batch_index=0,
                        seed=1,
                        sample_count=50,
                        distribution=distribution,
                        units_sold_distribution=distribution,
                        stockout_probability=0,
                    )
                ],
                distribution=distribution,
                units_sold_distribution=distribution,
                stockout_probability=0,
                unit_contribution_rub=1,
                score_breakdown=PricingRankingScoreV1(
                    cvar_10_per_day=cvar,
                    p10_per_day=p10,
                    mean_per_day=mean,
                    absolute_price_move_rub=abs(candidate.price_rub - 100),
                ),
            )

    result = FixedScoreEngine().run(request)

    assert result.ranked_candidate_ids == [
        "price-deep-high",
        "price-deep-low",
        "price-hold",
        "price-high",
        "price-low",
    ]


def test_request_requires_exactly_one_hold_candidate_at_current_price() -> None:
    with pytest.raises(ValidationError, match="exactly one hold candidate"):
        _request(
            _candidate("one", 90),
            _candidate("two", 100),
            _candidate("three", 110),
        )

    with pytest.raises(ValidationError, match="must equal current_price_rub"):
        _request(
            _candidate("hold", 99, operation="hold_price"),
            _candidate("two", 90),
            _candidate("three", 110),
        )


def test_seven_prices_times_500_samples_finish_under_five_seconds() -> None:
    request = _request(
        _candidate("hold", 100, operation="hold_price"),
        _candidate("p1", 90),
        _candidate("p2", 94),
        _candidate("p3", 97),
        _candidate("p4", 103),
        _candidate("p5", 106),
        _candidate("p6", 110),
    )

    started = time.perf_counter()
    result = run_pricing_simulation(request)
    duration = time.perf_counter() - started

    assert result.completed_candidate_count == 7
    assert sum(
        item.distribution.total.count
        for item in result.candidates
        if item.distribution is not None
    ) == 3_500
    assert duration < 5
