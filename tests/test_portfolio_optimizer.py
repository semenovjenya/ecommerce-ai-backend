from __future__ import annotations

from agi_ecom_app.manager_decision.portfolio_optimizer import (
    _goal_attainment,
    optimize_portfolio,
)


def _price_simulation(target: str, *, effect: float, stockout: float = 0.0) -> dict:
    return {
        "vendorCode": target,
        "horizonDays": 30,
        "currentPriceRub": 1000,
        "availableStockUnits": 90,
        "currentMarketContext": {
            "query_count": 5,
            "source_found_query_share": 0.8,
            "organic_competitors_before_source": 12,
            "source_price_effective": 1000,
            "price_before_median": 950,
        },
        "candidates": [
            {
                "operation": "hold_price",
                "priceRub": 1000,
                "status": "completed",
                "expectedUnits": 30,
                "meanEffectRub": 0,
                "p10EffectRub": 0,
                "stockoutProbability": 0,
            },
            {
                "operation": "set_price",
                "priceRub": 950,
                "priceMovePct": -5,
                "status": "completed",
                "expectedUnits": 45,
                "meanEffectRub": effect,
                "p10EffectRub": effect * 0.6,
                "stockoutProbability": stockout,
            },
        ],
    }


def _ad_row(advert_id: str, nm_id: int, loss: float) -> dict:
    return {
        "advert_id": advert_id,
        "nm_id": nm_id,
        "campaign_contribution_profit_7d_rub": -loss,
        "ad_orders_7d": 0,
        "ad_units_7d": 0,
        "stats_window_complete": True,
    }


def test_optimizer_returns_five_numeric_business_decisions_without_fillers() -> None:
    inventory = [
        {"vendor_code": "A", "nm_id": 1, "days_on_hand": 90},
        {"vendor_code": "B", "nm_id": 2, "days_on_hand": 90},
        {"vendor_code": "C", "nm_id": 3, "days_on_hand": 90},
    ]
    result = optimize_portfolio(
        query_plan={
            "constraints": {
                "horizon_days": 30,
                "target_cover_days": 30,
                "avoid_stockout": True,
                "preserve_contribution_profit": True,
            }
        },
        price_simulations=[
            _price_simulation("A", effect=3000),
            _price_simulation("B", effect=2500),
            _price_simulation("C", effect=2000),
        ],
        inventory_rows=inventory,
        candidate_snapshot={
            "advertisingRows": [
                _ad_row("ad-1", 1, 700),
                _ad_row("ad-2", 2, 600),
                _ad_row("ad-3", 3, 500),
            ],
            "supplyRows": [
                {
                    "vendor_code": "A",
                    "stock_qty": 3,
                    "avg_daily_sales_7d": 2,
                    "supplier_inbound_audited_qty": 0,
                    "lead_time_days": None,
                    "minimum_order_qty": None,
                    "replenishment_policy_version": None,
                }
            ],
        },
    )

    assert result["complete"] is True
    assert result["violations"] == []
    assert len(result["topRecommendations"]) == 5
    assert {row["family"] for row in result["topRecommendations"]} == {
        "price",
        "marketing",
    }
    assert all(row["eligible"] is True for row in result["topRecommendations"])
    assert all(
        isinstance(row[field], (int, float))
        for row in result["topRecommendations"]
        for field in (
            "inventoryDaysReduction",
            "expectedContributionProfitRub",
            "p10ContributionProfitRub",
            "stockoutProbability",
            "businessScore",
        )
    )
    assert all("SKU " in row["action"] for row in result["topRecommendations"] if row["family"] == "price")
    assert all(
        "приостановить" in row["action"]
        for row in result["topRecommendations"]
        if row["family"] == "marketing"
    )
    supply = next(row for row in result["scenarios"] if row["family"] == "supply")
    assert supply["eligible"] is False
    assert "lead time" in supply["constraintReasons"][0]


def test_action_portfolio_returns_exactly_five_blocked_checks_without_fake_effects() -> None:
    blocked_prices = [
        _price_simulation("A", effect=3000, stockout=0.2),
        _price_simulation("B", effect=2500, stockout=0.2),
    ]
    for simulation in blocked_prices:
        for candidate in simulation["candidates"]:
            candidate["status"] = "blocked"
    result = optimize_portfolio(
        query_plan={
            "decision_kind": "action_portfolio",
            "targets": [
                {"target_type": "vendor_code", "value": "A"},
                {"target_type": "vendor_code", "value": "B"},
            ],
            "required_domains": ["price", "marketing", "supply"],
            "requested_analyses": [
                "pricing",
                "price_scenarios",
                "advertising",
                "conversions",
                "replenishment",
                "inventory_matrix",
            ],
            "constraints": {
                "horizon_days": 30,
                "target_cover_days": 30,
                "avoid_stockout": True,
                "preserve_contribution_profit": True,
            },
        },
        price_simulations=blocked_prices,
        inventory_rows=[
            {
                "vendor_code": "A",
                "nm_id": 1,
                "current_stock_qty": 90,
                "sales_7d_units": 7,
                "days_on_hand": 90,
            },
            {
                "vendor_code": "B",
                "nm_id": 2,
                "current_stock_qty": 60,
                "sales_7d_units": 7,
                "days_on_hand": 60,
            },
        ],
        candidate_snapshot={
            "advertisingRows": [_ad_row("ad-1", 1, 700)],
            "supplyRows": [
                {
                    "vendor_code": "A",
                    "stock_qty": 3,
                    "avg_daily_sales_7d": 2,
                    "supplier_inbound_audited_qty": 0,
                    "lead_time_days": None,
                    "minimum_order_qty": None,
                    "replenishment_policy_version": None,
                }
            ],
        },
    )

    assert result["topRecommendations"] == []
    assert len(result["priorityRecommendations"]) == 5
    assert result["priorityComplete"] is True
    assert result["priorityViolations"] == []
    assert {row["family"] for row in result["priorityRecommendations"]} == {
        "price",
        "marketing",
        "supply",
    }
    assert all(row["eligible"] is False for row in result["priorityRecommendations"])
    assert all(
        -50 <= row["businessScore"] <= 100
        for row in result["priorityRecommendations"]
    )
    verification_rows = [
        row
        for row in result["priorityRecommendations"]
        if str(row["scenarioId"]).startswith("verification:")
    ]
    assert verification_rows
    assert all(row["expectedContributionProfitRub"] is None for row in verification_rows)
    assert all(row["inventoryDaysAfter"] is None for row in verification_rows)
    assert {
        row["target"]
        for row in verification_rows
        if row["family"] == "price"
    } == {"A", "B"}


def test_blocked_price_simulation_keeps_numeric_hypotheses_visible() -> None:
    simulation = _price_simulation("A", effect=3000)
    for candidate in simulation["candidates"]:
        candidate["status"] = "blocked"
    simulation.pop("currentMarketContext")
    simulation["hypothesisCandidates"] = {
        "status": "numeric",
        "executionEligible": False,
        "calculationMode": "observational_hypothesis",
        "confidence": "high",
        "confidenceScore": 0.8,
        "assumptions": [
            "спрос реагирует на изменение цены в тестовом диапазоне"
        ],
        "evidenceGaps": ["нет контролируемого ценового эксперимента"],
        "candidates": [
            {
                "operation": "hold_price",
                "priceRub": 1000,
                "expectedUnits": 30,
                "mean": 0,
                "p10": 0,
                "stockout": 0,
                "status": "hypothesis",
            }
        ]
        + [
            {
                "operation": "set_price",
                "priceRub": price,
                "expectedUnits": units,
                "mean": effect,
                "p10": effect * 0.6,
                "stockout": 0,
                "status": "hypothesis",
            }
            for price, units, effect in (
                (950, 45, 3000),
                (980, 40, 2200),
                (1020, 36, 1600),
                (1050, 33, 900),
            )
        ],
    }

    result = optimize_portfolio(
        query_plan={
            "decision_kind": "action_portfolio",
            "targets": [{"target_type": "vendor_code", "value": "A"}],
            "required_domains": ["price", "marketing", "supply"],
            "requested_analyses": [
                "pricing",
                "price_scenarios",
                "advertising",
                "replenishment",
            ],
            "constraints": {
                "horizon_days": 30,
                "target_cover_days": 30,
                "avoid_stockout": True,
                "preserve_contribution_profit": True,
            },
        },
        price_simulations=[simulation],
        inventory_rows=[
            {
                "vendor_code": "A",
                "nm_id": 1,
                "current_stock_qty": 90,
                "sales_7d_units": 7,
                "days_on_hand": 90,
            }
        ],
        candidate_snapshot={},
    )

    price_scenarios = [
        row for row in result["scenarios"] if row["family"] == "price"
    ]
    assert len(price_scenarios) == 4
    assert all(row["calculationStatus"] == "hypothesis" for row in price_scenarios)
    assert all(row["executionStatus"] == "shadow_test" for row in price_scenarios)
    assert all(row["eligible"] is False for row in price_scenarios)
    assert all(row["hypothesisEligible"] is True for row in price_scenarios)
    assert all(row["confidence"] == "low" for row in price_scenarios)
    assert all(row["assumptions"] for row in price_scenarios)
    assert all(row["evidenceGaps"] for row in price_scenarios)
    assert all(row["limitations"] for row in price_scenarios)
    assert all(
        isinstance(row[field], (int, float))
        for row in price_scenarios
        for field in (
            "inventoryDaysAfter",
            "inventoryDaysReduction",
            "expectedContributionProfitRub",
            "p10ContributionProfitRub",
            "stockoutProbability",
            "businessScore",
        )
    )
    assert result["topRecommendations"] == []
    assert len(result["topHypotheses"]) == 1
    top_hypothesis = result["topHypotheses"][0]
    assert top_hypothesis["family"] == "price"
    assert top_hypothesis["target"] == "A"
    assert top_hypothesis["eligible"] is False
    assert top_hypothesis["hypothesisEligible"] is True
    assert any(
        row["scenarioId"] == top_hypothesis["scenarioId"]
        for row in result["priorityRecommendations"]
    )
    assert len(result["priorityRecommendations"]) == 5


def test_list_hypothesis_contract_preserves_sibling_metadata() -> None:
    simulation = _price_simulation("A", effect=3000)
    for candidate in simulation["candidates"]:
        candidate["status"] = "blocked"
    simulation.update(
        {
            "hypothesisStatus": "hypothesis",
            "hypothesisCalculationMode": "bounded_elasticity_sensitivity_grid",
            "hypothesisConfidence": "low",
            "hypothesisConfidenceScore": 0.31,
            "hypothesisAssumptions": ["проверяемое допущение"],
            "hypothesisEvidenceGaps": ["нет причинного эксперимента"],
            "hypothesisCandidates": [
                {
                    "operation": "hold_price",
                    "priceRub": 1000,
                    "expectedUnits": 30,
                    "mean": 0,
                    "p10": 0,
                    "stockout": 0,
                    "status": "hypothesis",
                },
                {
                    "operation": "set_price",
                    "priceRub": 950,
                    "expectedUnits": 45,
                    "mean": 3000,
                    "p10": 1800,
                    "stockout": 0,
                    "status": "hypothesis",
                },
            ],
        }
    )

    result = optimize_portfolio(
        query_plan={
            "requested_analyses": ["price_scenarios"],
            "constraints": {"horizon_days": 30, "target_cover_days": 30},
        },
        price_simulations=[simulation],
        inventory_rows=[
            {
                "vendor_code": "A",
                "current_stock_qty": 90,
                "sales_7d_units": 7,
                "days_on_hand": 90,
            }
        ],
        candidate_snapshot={},
    )

    scenario = next(row for row in result["scenarios"] if row["family"] == "price")
    assert scenario["calculationStatus"] == "hypothesis"
    assert scenario["confidenceScore"] == 0.31
    assert (
        scenario["hypothesisCalculationMode"]
        == "bounded_elasticity_sensitivity_grid"
    )
    assert "проверяемое допущение" in scenario["assumptions"]
    assert "нет причинного эксперимента" in scenario["evidenceGaps"]


def test_mixed_price_result_merges_four_moves_with_strict_priority() -> None:
    prices = (1000, 950, 975, 1025, 1050)
    simulation = _price_simulation("A", effect=3000)
    simulation["candidates"] = [
        {
            "operation": "hold_price" if price == 1000 else "set_price",
            "priceRub": price,
            "status": "completed" if price in {1000, 975} else "blocked",
            "expectedUnits": 30 if price == 1000 else 40,
            "meanEffectRub": 0 if price == 1000 else 2000,
            "p10EffectRub": 0 if price == 1000 else 1000,
            "stockoutProbability": 0,
        }
        for price in prices
    ]
    simulation.update(
        {
            "hypothesisCalculationMode": "bounded_elasticity_sensitivity_grid",
            "hypothesisConfidence": "low",
            "hypothesisConfidenceScore": 0.2,
            "hypothesisCandidates": [
                {
                    "operation": "hold_price" if price == 1000 else "set_price",
                    "priceRub": price,
                    "status": "hypothesis",
                    "expectedUnits": 30 if price == 1000 else 42,
                    "mean": 0 if price == 1000 else 1800,
                    "p10": 0 if price == 1000 else 900,
                    "stockout": 0,
                }
                for price in prices
            ],
        }
    )

    result = optimize_portfolio(
        query_plan={
            "requested_analyses": ["price_scenarios"],
            "constraints": {
                "horizon_days": 30,
                "target_cover_days": 30,
                "preserve_contribution_profit": True,
            },
        },
        price_simulations=[simulation],
        inventory_rows=[
            {
                "vendor_code": "A",
                "current_stock_qty": 90,
                "sales_7d_units": 7,
                "days_on_hand": 90,
            }
        ],
        candidate_snapshot={},
    )

    scenarios = [row for row in result["scenarios"] if row["family"] == "price"]
    assert {row["scenarioId"] for row in scenarios} == {
        "price:A:950",
        "price:A:975",
        "price:A:1025",
        "price:A:1050",
    }
    strict = next(row for row in scenarios if row["scenarioId"] == "price:A:975")
    assert strict["calculationStatus"] == "completed"
    assert strict["executionStatus"] == "shadow"
    hypotheses = [row for row in scenarios if row is not strict]
    assert all(row["calculationStatus"] == "hypothesis" for row in hypotheses)
    assert all(row["executionStatus"] == "shadow_test" for row in hypotheses)
    assert all(row["eligible"] is False for row in hypotheses)


def test_priority_recommendations_preserve_required_families_with_five_price_hypotheses() -> None:
    simulations = []
    targets = ["A", "B", "C", "D", "E"]
    for index, target in enumerate(targets):
        simulation = _price_simulation(target, effect=3000 - index * 100)
        for candidate in simulation["candidates"]:
            candidate["status"] = "blocked"
        simulation["hypothesisCandidates"] = {
            "status": "numeric",
            "executionEligible": False,
            "calculationMode": "observational_hypothesis",
            "confidence": "high",
            "assumptions": ["ценовой отклик проверяется в теневом режиме"],
            "evidenceGaps": ["нет контролируемого ценового эксперимента"],
            "candidates": [
                {
                    "operation": "hold_price",
                    "priceRub": 1000,
                    "expectedUnits": 30,
                    "mean": 0,
                    "p10": 0,
                    "stockout": 0,
                    "status": "hypothesis",
                },
                {
                    "operation": "set_price",
                    "priceRub": 950,
                    "expectedUnits": 45,
                    "mean": 3000 - index * 100,
                    "p10": 1800 - index * 60,
                    "stockout": 0,
                    "status": "hypothesis",
                },
            ],
        }
        simulations.append(simulation)

    result = optimize_portfolio(
        query_plan={
            "decision_kind": "action_portfolio",
            "targets": [
                {"target_type": "vendor_code", "value": target}
                for target in targets
            ],
            "required_domains": ["price", "marketing", "supply"],
            "requested_analyses": [
                "pricing",
                "price_scenarios",
                "advertising",
                "replenishment",
            ],
            "constraints": {
                "horizon_days": 30,
                "target_cover_days": 30,
                "avoid_stockout": True,
                "preserve_contribution_profit": True,
            },
        },
        price_simulations=simulations,
        inventory_rows=[],
        candidate_snapshot={},
    )

    assert result["topRecommendations"] == []
    assert len(result["topHypotheses"]) == 5
    assert {row["family"] for row in result["topHypotheses"]} == {"price"}
    assert len(result["priorityRecommendations"]) == 5
    assert {row["family"] for row in result["priorityRecommendations"]} == {
        "price",
        "marketing",
        "supply",
    }
    assert result["priorityViolations"] == []


def test_explicit_price_action_scope_keeps_supporting_domains_out_of_scenarios() -> None:
    simulations = []
    for index, target in enumerate(["A", "B", "C", "D", "E"]):
        simulation = _price_simulation(target, effect=3000 - index * 100)
        for candidate in simulation["candidates"]:
            candidate["status"] = "blocked"
        simulation["hypothesisCandidates"] = {
            "status": "numeric",
            "executionEligible": False,
            "calculationMode": "observational_hypothesis",
            "confidence": "low",
            "candidates": [
                {
                    "operation": "hold_price",
                    "priceRub": 1000,
                    "expectedUnits": 30,
                    "mean": 0,
                    "p10": 0,
                    "stockout": 0,
                    "status": "hypothesis",
                },
                {
                    "operation": "set_price",
                    "priceRub": 950,
                    "expectedUnits": 45,
                    "mean": 3000 - index * 100,
                    "p10": 1800 - index * 60,
                    "stockout": 0,
                    "status": "hypothesis",
                },
            ],
        }
        simulations.append(simulation)

    result = optimize_portfolio(
        query_plan={
            "decision_kind": "action_portfolio",
            "required_domains": ["price", "finance", "sales", "marketing", "supply"],
            "allowed_action_families": ["price"],
            "requested_analyses": [
                "pricing",
                "price_scenarios",
                "advertising",
                "current_stock",
                "replenishment",
            ],
            "constraints": {
                "horizon_days": 30,
                "target_cover_days": 30,
            },
        },
        price_simulations=simulations,
        inventory_rows=[],
        candidate_snapshot={
            "advertisingRows": [{"campaign_id": "ad-1"}],
            "supplyRows": [{"vendor_code": "A"}],
        },
    )

    assert result["scenarioCount"] == 5
    assert {row["family"] for row in result["scenarios"]} == {"price"}
    assert len(result["topHypotheses"]) == 5
    assert len(result["priorityRecommendations"]) == 5
    assert {row["family"] for row in result["priorityRecommendations"]} == {"price"}
    assert result["priorityRecommendations"][0]["currentPriceRub"] == 1000
    assert result["priorityRecommendations"][0]["candidatePriceRub"] == 950
    assert result["priorityRecommendations"][0]["availableStockUnits"] == 90
    assert result["priorityRecommendations"][0]["expectedUnitsDelta"] == 15
    assert result["priorityRecommendations"][0]["gapToTargetDays"] == 30
    assert result["priorityViolations"] == []


def test_optimizer_rejects_price_scenario_with_excess_stockout_risk() -> None:
    result = optimize_portfolio(
        query_plan={
            "constraints": {
                "horizon_days": 30,
                "target_cover_days": 30,
                "avoid_stockout": True,
                "preserve_contribution_profit": True,
            }
        },
        price_simulations=[_price_simulation("A", effect=3000, stockout=0.2)],
        inventory_rows=[{"vendor_code": "A", "nm_id": 1, "days_on_hand": 90}],
        candidate_snapshot={},
    )

    price = next(row for row in result["scenarios"] if row["family"] == "price")
    assert price["eligible"] is False
    assert "риск дефицита выше 5%" in price["constraintReasons"]
    assert result["topRecommendations"] == []
    assert result["violations"] == ["expected 5 optimized recommendations, got 0"]


def test_competitor_guardrail_applies_only_to_positive_comparable_prices() -> None:
    comparable = _price_simulation("A", effect=3000)
    comparable["currentMarketContext"] = {
        "query_count": 4,
        "source_found_query_share": 0.75,
        "organic_competitors_before_source": 8,
        "source_price_effective": 1000,
        "price_before_median": 900,
    }
    accepted = optimize_portfolio(
        query_plan={"constraints": {"horizon_days": 30, "target_cover_days": 30}},
        price_simulations=[comparable],
        inventory_rows=[{"vendor_code": "A", "nm_id": 1, "days_on_hand": 90}],
        candidate_snapshot={},
    )
    accepted_price = next(
        row for row in accepted["scenarios"] if row["family"] == "price"
    )
    assert accepted_price["competitorGuardrail"]["status"] == "applied"
    assert accepted_price["eligible"] is True

    too_expensive = _price_simulation("A", effect=3000)
    too_expensive["currentMarketContext"] = {
        **comparable["currentMarketContext"],
        "price_before_median": 700,
    }
    rejected = optimize_portfolio(
        query_plan={"constraints": {"horizon_days": 30, "target_cover_days": 30}},
        price_simulations=[too_expensive],
        inventory_rows=[{"vendor_code": "A", "nm_id": 1, "days_on_hand": 90}],
        candidate_snapshot={},
    )
    rejected_price = next(
        row for row in rejected["scenarios"] if row["family"] == "price"
    )
    assert rejected_price["eligible"] is False
    assert "более чем на 20%" in rejected_price["constraintReasons"][0]

    invalid = _price_simulation("A", effect=3000)
    invalid["currentMarketContext"] = {
        **comparable["currentMarketContext"],
        "price_before_median": 0,
    }
    neutral = optimize_portfolio(
        query_plan={"constraints": {"horizon_days": 30, "target_cover_days": 30}},
        price_simulations=[invalid],
        inventory_rows=[{"vendor_code": "A", "nm_id": 1, "days_on_hand": 90}],
        candidate_snapshot={},
    )
    neutral_price = next(
        row for row in neutral["scenarios"] if row["family"] == "price"
    )
    assert neutral_price["competitorGuardrail"]["status"] == "applied_current_anchor_fallback"
    assert neutral_price["eligible"] is False
    assert "временной ценовой коридор" in neutral_price["constraintReasons"][0]


def test_supply_scenario_uses_only_complete_inbound_with_eta_inside_horizon() -> None:
    base_supply = {
        "vendor_code": "A",
        "stock_qty": 0,
        "avg_daily_sales_7d": 10,
        "supplier_inbound_audited_qty": 100,
        "supplier_audit_snapshot_date": "2026-07-20",
        "supplier_eta_date": "2026-07-25",
        "lead_time_days": 5,
        "target_cover_days": 30,
        "minimum_order_qty": 10,
        "replenishment_policy_version": "v1",
    }
    incomplete = optimize_portfolio(
        query_plan={"constraints": {"horizon_days": 30, "target_cover_days": 30}},
        price_simulations=[],
        inventory_rows=[],
        candidate_snapshot={
            "supplyRows": [
                {**base_supply, "supplier_snapshot_complete": False}
            ]
        },
    )
    incomplete_supply = next(
        row for row in incomplete["scenarios"] if row["family"] == "supply"
    )
    assert incomplete_supply["supplierInboundStatus"] == "unconfirmed"
    assert incomplete_supply["includedSupplierInboundUnits"] == 0
    assert "дозаказ 300 шт." in incomplete_supply["action"]

    complete = optimize_portfolio(
        query_plan={"constraints": {"horizon_days": 30, "target_cover_days": 30}},
        price_simulations=[],
        inventory_rows=[],
        candidate_snapshot={
            "supplyRows": [
                {**base_supply, "supplier_snapshot_complete": True}
            ]
        },
    )
    complete_supply = next(
        row for row in complete["scenarios"] if row["family"] == "supply"
    )
    assert complete_supply["supplierInboundStatus"] == "scheduled"
    assert complete_supply["includedSupplierInboundUnits"] == 100
    assert "дозаказ 200 шт." in complete_supply["action"]


def test_optimizer_uses_yesterday_today_corridor_when_competitors_are_low_quality() -> None:
    simulation = _price_simulation("A", effect=3000)
    simulation["currentMarketContext"] = {
        "query_count": 1,
        "source_found_query_share": 0.2,
        "organic_competitors_before_source": 2,
        "source_price_effective": 1000,
        "price_before_median": 950,
    }
    simulation["marketContextHistory"] = [
        {
            "snapshot_date": "2026-07-20",
            "market_price_median": 771,
            "market_price_min": 125,
            "market_price_max": 4243,
        },
        {
            "snapshot_date": "2026-07-19",
            "market_price_median": 767,
            "market_price_min": 125,
            "market_price_max": 9377,
        },
    ]
    simulation["candidates"][1]["priceRub"] = 975
    result = optimize_portfolio(
        query_plan={"constraints": {"horizon_days": 30, "target_cover_days": 30}},
        price_simulations=[simulation],
        inventory_rows=[{"vendor_code": "A", "nm_id": 1, "days_on_hand": 90}],
        candidate_snapshot={},
    )

    price = next(row for row in result["scenarios"] if row["family"] == "price")
    assert price["eligible"] is True
    assert price["constraintReasons"] == []
    assert price["competitorGuardrail"]["status"] == "applied_competitor_temporal_fallback"
    assert price["competitorGuardrail"]["basis"] == "competitor_market_median_yesterday_today"
    assert price["competitorGuardrail"]["competitorYesterdayToTodayChangePct"] == 0.52
    assert price["competitorGuardrail"]["corridorLowerPriceRub"] == 969.78
    assert price["competitorGuardrail"]["corridorUpperPriceRub"] == 1030.22
    assert len(result["topRecommendations"]) == 1


def test_real_competitor_history_allows_sku_46303_candidate() -> None:
    simulation = _price_simulation("46303", effect=77600)
    simulation["currentPriceRub"] = 5615.85
    simulation["currentMarketContext"] = {
        "query_count": 2,
        "source_found_query_share": 0.5,
        "organic_competitors_before_source": 5,
        "source_price_effective": 3280,
        "price_before_median": 408,
    }
    simulation["marketContextHistory"] = [
        {
            "snapshot_date": "2026-07-21",
            "market_price_median": 771,
            "market_price_min": 125,
            "market_price_max": 4243,
        },
        {
            "snapshot_date": "2026-07-20",
            "market_price_median": 767,
            "market_price_min": 125,
            "market_price_max": 9377,
        },
    ]
    simulation["candidates"][0]["priceRub"] = 5615.85
    simulation["candidates"][1]["priceRub"] = 5475

    result = optimize_portfolio(
        query_plan={"constraints": {"horizon_days": 30, "target_cover_days": 30}},
        price_simulations=[simulation],
        inventory_rows=[{"vendor_code": "46303", "nm_id": 171742855, "days_on_hand": 90}],
        candidate_snapshot={},
    )

    price = next(row for row in result["scenarios"] if row["family"] == "price")
    guardrail = price["competitorGuardrail"]
    assert guardrail["status"] == "applied_competitor_temporal_fallback"
    assert guardrail["competitorTodayMedianRub"] == 771
    assert guardrail["competitorYesterdayMedianRub"] == 767
    assert guardrail["allowedMovePct"] == 3.02
    assert guardrail["corridorLowerPriceRub"] == 5446.17
    assert guardrail["corridorUpperPriceRub"] == 5785.53
    assert price["eligible"] is True


def test_temporal_corridor_allows_candidate_rounded_to_whole_rubles() -> None:
    simulation = _price_simulation("A", effect=3000)
    simulation["currentMarketContext"] = {"query_count": 0}
    simulation["dailyPriceRows"] = [
        {"date": "2026-07-20", "sellerEffectivePriceRub": 1000},
        {"date": "2026-07-19", "sellerEffectivePriceRub": 1000},
    ]
    simulation["candidates"][1]["priceRub"] = 974.5
    result = optimize_portfolio(
        query_plan={"constraints": {"horizon_days": 30, "target_cover_days": 30}},
        price_simulations=[simulation],
        inventory_rows=[{"vendor_code": "A", "nm_id": 1, "days_on_hand": 90}],
        candidate_snapshot={},
    )

    price = next(row for row in result["scenarios"] if row["family"] == "price")
    assert price["competitorGuardrail"]["corridorLowerPriceRub"] == 975
    assert price["competitorGuardrail"]["roundingToleranceRub"] == 1
    assert price["eligible"] is True


def test_single_competitor_day_still_produces_conservative_corridor() -> None:
    simulation = _price_simulation("A", effect=3000)
    simulation["currentMarketContext"] = {"query_count": 0}
    simulation["marketContextHistory"] = [
        {"snapshot_date": "2026-07-20", "market_price_median": 771}
    ]
    simulation["candidates"][1]["priceRub"] = 975

    result = optimize_portfolio(
        query_plan={"constraints": {"horizon_days": 30, "target_cover_days": 30}},
        price_simulations=[simulation],
        inventory_rows=[{"vendor_code": "A", "nm_id": 1, "days_on_hand": 90}],
        candidate_snapshot={},
    )

    price = next(row for row in result["scenarios"] if row["family"] == "price")
    assert price["competitorGuardrail"]["status"] == "applied_competitor_single_day_fallback"
    assert price["competitorGuardrail"]["corridorLowerPriceRub"] == 975
    assert price["eligible"] is True


def test_optimizer_does_not_fill_top_five_with_zero_coverage_progress() -> None:
    result = optimize_portfolio(
        query_plan={"constraints": {"horizon_days": 60, "target_cover_days": 30}},
        price_simulations=[],
        inventory_rows=[{"vendor_code": "A", "nm_id": 1, "days_on_hand": 90}],
        candidate_snapshot={"advertisingRows": [_ad_row("ad-1", 1, 700)]},
    )

    marketing = next(row for row in result["scenarios"] if row["family"] == "marketing")
    assert marketing["inventoryDaysReduction"] == 0
    assert marketing["eligible"] is False
    assert "не снижает складское покрытие" in marketing["constraintReasons"][0]
    assert result["topRecommendations"] == []


def test_optimizer_reports_goal_attainment_with_unresolved_skus_and_metric_names() -> None:
    result = optimize_portfolio(
        query_plan={
            "constraints": {
                "horizon_days": 30,
                "target_cover_days": 30,
                "avoid_stockout": True,
                "preserve_contribution_profit": True,
            }
        },
        price_simulations=[_price_simulation("A", effect=3000)],
        inventory_rows=[
            {
                "vendor_code": "A",
                "nm_id": 1,
                "current_stock_qty": 90,
                "sales_7d_units": 7,
                "days_on_hand": 90,
                "latest_snapshot_date": "2026-07-20",
            },
            {
                "vendor_code": "B",
                "nm_id": 2,
                "current_stock_qty": 10,
                "sales_7d_units": 7,
                "days_on_hand": 15,
                "latest_snapshot_date": "2026-07-20",
            },
        ],
        candidate_snapshot={},
    )

    goal = result["goalAttainment"]
    assert goal["status"] == "not_achieved"
    assert goal["goalAchieved"] is False
    assert goal["aboveTargetSkuCount"] == 1
    assert goal["remainingAboveTargetSkuCount"] == 1
    assert goal["remainingAboveTargetSkus"][0] == {
        "target": "A",
        "currentCoverDays7d": 90.0,
        "currentStockUnits": 90.0,
        "salesUnits7d": 7.0,
        "latestSnapshotDate": "2026-07-20",
        "bestScenarioCoverDays": 60.0,
        "bestScenarioMetric": "simulated_forward_cover_days",
        "bestScenarioId": "price:A:950",
    }
    price = next(row for row in result["scenarios"] if row["family"] == "price")
    assert price["inventoryMetric"] == "simulated_forward_cover_days"


def test_zero_inventory_progress_scenario_is_not_presented_as_goal_solution() -> None:
    goal = _goal_attainment(
        [{"vendor_code": "A", "current_stock_qty": 90, "sales_7d_units": 7}],
        recommendations=[
            {
                "scenarioId": "marketing:pause:1",
                "family": "marketing",
                "affectedSkus": ["A"],
                "inventoryMetric": "turnover_days_30d_average_stock",
                "inventoryDaysBefore": 90,
                "inventoryDaysAfter": 90,
                "inventoryDaysReduction": 0,
            }
        ],
        target_cover_days=30,
    )

    remaining = goal["remainingAboveTargetSkus"][0]
    assert remaining["bestScenarioId"] is None
    assert remaining["bestScenarioCoverDays"] is None
    assert remaining["bestScenarioMetric"] is None


def test_goal_attainment_is_not_proven_without_evaluated_inventory() -> None:
    goal = _goal_attainment(
        [],
        recommendations=[],
        target_cover_days=30,
    )

    assert goal["status"] == "not_proven"
    assert goal["goalAchieved"] is False
    assert goal["evaluatedSkuCount"] == 0


def test_zero_stock_is_stockout_not_successful_turnover_goal() -> None:
    goal = _goal_attainment(
        [
            {
                "vendor_code": "A",
                "current_stock_qty": 0,
                "sales_7d_units": 7,
                "latest_snapshot_date": "2026-07-22",
            }
        ],
        recommendations=[],
        target_cover_days=30,
    )

    assert goal["status"] == "not_achieved"
    assert goal["goalAchieved"] is False
    assert goal["stockoutSkuCount"] == 1
    assert goal["stockoutSkus"][0]["target"] == "A"
