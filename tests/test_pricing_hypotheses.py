from __future__ import annotations

from datetime import date, timedelta

import pytest

from agi_ecom_app.decision_simulation.pricing_engine import PricingSimulationEngine
from agi_ecom_app.decision_simulation.pricing_hypotheses import (
    build_pricing_hypotheses,
)
from agi_ecom_app.decision_simulation.pricing_models import (
    PricingAvailableStockEvidenceV1,
    PricingCandidateV1,
    PricingCompletedDayObservationV1,
    PricingContextFeatureEvidenceV1,
    PricingDailyObservationsEvidenceV1,
    PricingElasticityEvidenceV1,
    PricingForecastContextEvidenceV1,
    PricingRealizationEvidenceV1,
    PricingSimulationEvidenceV1,
    PricingSimulationRequestV1,
    PricingUnitEconomicsEvidenceV1,
)


def _request(
    elasticity_values: list[float],
    *,
    missing_basic_data: bool = False,
) -> PricingSimulationRequestV1:
    horizon_days = 7
    observations = None
    realization = None
    stock = None
    net_revenue_rate = None
    variable_cost = None
    if not missing_basic_data:
        observations = [
            PricingCompletedDayObservationV1(
                observedDate=date(2026, 6, 20) + timedelta(days=index),
                priceRub=100 + (index % 5 - 2) * 2,
                unitsSold=8 + index % 7,
                contextValues={"ad_clicks": float(10 + index % 4)},
                evidenceRef=f"demand:{index}",
            )
            for index in range(28)
        ]
        realization = [0.75, 0.80, 0.85, 0.90] * 25
        stock = 1_000.0
        net_revenue_rate = 0.75
        variable_cost = 30.0

    return PricingSimulationRequestV1(
        snapshotHash="a" * 64,
        decisionGroupId="pricing:sku:42",
        skuId="42",
        currentPriceRub=100,
        currentPriceEvidenceRef="price:42:2026-07-20",
        horizonDays=horizon_days,
        candidates=[
            PricingCandidateV1(
                candidateId="price:42:hold",
                actionId="action:42:hold",
                operation="hold_price",
                priceRub=100,
            ),
            PricingCandidateV1(
                candidateId="price:42:down-5",
                actionId="action:42:down-5",
                operation="set_price",
                priceRub=95,
            ),
            PricingCandidateV1(
                candidateId="price:42:down-2.5",
                actionId="action:42:down-2.5",
                operation="set_price",
                priceRub=97.5,
            ),
            PricingCandidateV1(
                candidateId="price:42:up-2.5",
                actionId="action:42:up-2.5",
                operation="set_price",
                priceRub=102.5,
            ),
            PricingCandidateV1(
                candidateId="price:42:up-5",
                actionId="action:42:up-5",
                operation="set_price",
                priceRub=105,
            ),
        ],
        evidence=PricingSimulationEvidenceV1(
            dailyObservations=PricingDailyObservationsEvidenceV1(
                values=observations,
                evidenceRef="demand:42",
                latestObservationAgeDays=1,
            ),
            elasticity=PricingElasticityEvidenceV1(
                bootstrapValues=elasticity_values,
                source="historical_price_volume_residual_bootstrap_v1",
                evidenceRef="elasticity:42",
            ),
            realization=PricingRealizationEvidenceV1(
                bootstrapValues=realization,
                source="cohort_order_to_realized_sale_day_bootstrap_v1",
                evidenceRef="realization:42",
            ),
            unitEconomics=PricingUnitEconomicsEvidenceV1(
                netRevenueRate=net_revenue_rate,
                variableCostPerUnitRub=variable_cost,
                observationCount=28 if not missing_basic_data else None,
                latestObservationAgeDays=1,
                basis="wb_net_payout_before_cogs",
                evidenceRef="economics:42",
            ),
            availableStock=PricingAvailableStockEvidenceV1(
                units=stock,
                snapshotDate=date(2026, 7, 20),
                ageDaysAtSnapshot=0,
                evidenceRef="stock:42",
            ),
            forecastContext=(
                PricingForecastContextEvidenceV1(
                    snapshotDate=date(2026, 7, 20),
                    features=[
                        PricingContextFeatureEvidenceV1(
                            name="ad_clicks",
                            status="modeled",
                            observedDays=28,
                            coverageRatio=1.0,
                            center=11.5,
                            scale=1.0,
                            bootstrapCoefficients=[0.1] * len(elasticity_values),
                            forwardValues=[12.0] * horizon_days,
                            evidenceRef="ads:42",
                        )
                    ],
                )
                if not missing_basic_data
                else None
            ),
        ),
    )


@pytest.mark.parametrize(
    "elasticity_values, expected_gap",
    [
        ([0.8] * 200, "invalid_price_elasticity"),
        ([-12.0] * 200, "implausible_price_elasticity_magnitude"),
    ],
)
def test_unreliable_elasticity_yields_five_numeric_non_executable_hypotheses(
    elasticity_values: list[float],
    expected_gap: str,
) -> None:
    request = _request(elasticity_values)
    strict = PricingSimulationEngine().run(request)
    strict_before = strict.model_dump(mode="json", by_alias=True)
    request_before = request.model_dump(mode="json", by_alias=True)

    first = build_pricing_hypotheses(request, strict)
    second = build_pricing_hypotheses(
        request,
        strict.model_dump(mode="json", by_alias=True),
    )

    assert first == second
    assert first["status"] == "hypothesis"
    assert first["executionEligible"] is False
    assert first["calculationMode"] == "bounded_elasticity_sensitivity_grid"
    assert expected_gap in first["evidenceGaps"]
    assert len(first["candidates"]) == 5
    assert sorted(row["rank"] for row in first["candidates"]) == [1, 2, 3, 4, 5]
    assert sum(row["isWinner"] for row in first["candidates"]) == 1
    for row in first["candidates"]:
        assert row["status"] == "hypothesis"
        assert row["executionEligible"] is False
        for key in (
            "mean",
            "p10",
            "cvar10",
            "probPositive",
            "expectedUnits",
            "stockout",
            "unitContribution",
        ):
            assert isinstance(row[key], float)

    assert all(candidate.status == "blocked" for candidate in strict.candidates)
    assert strict.model_dump(mode="json", by_alias=True) == strict_before
    assert request.model_dump(mode="json", by_alias=True) == request_before


def test_plausible_share_of_own_draws_is_preferred_and_stays_deterministic() -> None:
    request = _request([-1.3] * 120 + [-0.8] * 40 + [0.5] * 40)
    strict = PricingSimulationEngine().run(request)

    first = build_pricing_hypotheses(request, strict)
    second = build_pricing_hypotheses(request, strict)

    assert first == second
    assert first["calculationMode"] == "observed_plausible_elasticity_sensitivity"
    assert first["confidenceScore"] > 0.5
    assert len(first["candidates"]) == 5
    assert all(row["mean"] is not None for row in first["candidates"])
    assert all(candidate.status == "blocked" for candidate in strict.candidates)


def test_stale_low_confidence_prior_is_only_a_sensitivity_center() -> None:
    request = _request([0.8] * 200)
    strict = PricingSimulationEngine().run(request)

    result = build_pricing_hypotheses(
        request,
        strict,
        price_prior={
            "elasticity_to_apply": -1.7,
            "decision_confidence": "low",
            "decision_status": "needs_review",
            "price_ai_run_date": "2026-04-15",
            "recommended_price": 80,
        },
    )

    assert result["calculationMode"] == "price_prior_elasticity_sensitivity"
    assert result["confidence"] == "low"
    assert result["confidenceScore"] == 0.1
    assert "low_confidence_price_prior" in result["evidenceGaps"]
    assert "non_actionable_price_prior" in result["evidenceGaps"]
    assert "stale_price_prior" in result["evidenceGaps"]
    assert [row["priceRub"] for row in result["candidates"]] == [
        candidate.price_rub for candidate in request.candidates
    ]
    assert all(row["priceRub"] != 80 for row in result["candidates"])


def test_missing_non_elastic_basics_keeps_all_five_variants_qualitative() -> None:
    request = _request([0.8] * 200, missing_basic_data=True)
    strict = PricingSimulationEngine().run(request)
    strict_before = strict.model_dump(mode="json", by_alias=True)

    result = build_pricing_hypotheses(request, strict)

    assert result["status"] == "qualitative"
    assert result["calculationMode"] == "qualitative_only"
    assert result["executionEligible"] is False
    assert result["winnerCandidateId"] is None
    assert len(result["candidates"]) == 5
    assert {
        "missing_completed_day_observations",
        "missing_realization_bootstrap_values",
        "missing_available_stock",
        "missing_net_revenue_rate",
        "missing_variable_cost_per_unit",
    }.issubset(set(result["evidenceGaps"]))
    for row in result["candidates"]:
        assert row["status"] == "hypothesis"
        assert row["executionEligible"] is False
        assert row["rank"] is None
        assert row["isWinner"] is False
        for key in (
            "mean",
            "p10",
            "cvar10",
            "probPositive",
            "expectedUnits",
            "stockout",
            "unitContribution",
        ):
            assert row[key] is None

    assert all(candidate.status == "blocked" for candidate in strict.candidates)
    assert strict.model_dump(mode="json", by_alias=True) == strict_before
