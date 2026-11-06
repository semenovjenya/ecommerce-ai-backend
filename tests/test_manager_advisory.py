from __future__ import annotations

import re
from types import SimpleNamespace
from unittest.mock import patch

from agi_ecom_app.forecast.models import GrowthOpportunity
from agi_ecom_app.manager_decision.advisory import (
    _optimized_recommendations,
    _portfolio_targets,
    _targets_near_markers,
    build_manager_advisory,
    render_manager_advisory,
    should_build_manager_advisory,
    validate_scenario_completeness,
)


def _opportunity() -> GrowthOpportunity:
    return GrowthOpportunity(
        opportunity_id="reorder_for_growth:46303",
        action_family="reorder_for_growth",
        score=82,
        evidence_summary="Проверить запас",
        blockers=["thin_price_data"],
        affected_skus=["46303"],
        affected_campaigns=[],
        affected_queries=[],
        expected_metric_focus=["coverage_days"],
        target_mode="sku",
        primary_domain="supply",
        growth_reason="Проверить запас",
        source_facts={},
    )


def test_advisory_requires_explicit_target_and_scenario_marker():
    assert should_build_manager_advisory(
        "Для SKU 46303 сравни два варианта цены и ожидаемый эффект",
        query_plan={
            "scope": "single_sku",
            "targets": [{"target_type": "vendor_code", "value": "46303"}],
            "requested_analyses": ["price_scenarios", "forecast"],
        },
    )
    assert not should_build_manager_advisory(
        "Составь единый план действий",
        query_plan={"scope": "portfolio", "requested_analyses": []},
    )
    assert should_build_manager_advisory(
        "Сравни варианты цены по всей матрице",
        query_plan={
            "scope": "portfolio",
            "targets": [],
            "requested_analyses": ["price_scenarios"],
        },
    )


def test_price_analysis_is_limited_to_target_in_matching_query_segment():
    assert _targets_near_markers(
        "SKU 46229 и SKU 46313: проверить остатки; по SKU 46303 сопоставить цену и продажи.",
        ["46229", "46313", "46303"],
        markers=("цен", "сопостав"),
    ) == {"46303"}


def test_portfolio_targets_rank_urgent_rows_but_keep_the_full_matrix():
    with patch(
        "agi_ecom_app.manager_decision.advisory.get_inventory_turnover",
        return_value={
            "totals": {"period_days": 30},
            "rows": [
                {"vendor_code": "46000", "sales_units": 0, "current_stock_qty": 12},
                {"vendor_code": "46230", "sales_units": 37, "current_stock_qty": 223},
                {"vendor_code": "46303", "sales_units": 94, "current_stock_qty": 228},
                {"vendor_code": "46313", "sales_units": 81, "current_stock_qty": 1},
            ],
        },
    ):
        targets = _portfolio_targets(
            {"scope": "portfolio", "constraints": {"target_cover_days": 30}},
            max_targets=4,
        )

    assert targets == ["46000", "46230", "46303", "46313"]


def test_portfolio_targets_preserve_all_17_eligible_skus() -> None:
    rows = [
        {
            "vendor_code": str(46220 + index),
            "sales_units": 30,
            "current_stock_qty": 100 + index,
        }
        for index in range(17)
    ]
    rows[0]["sales_units"] = 0
    rows[1]["current_stock_qty"] = 0
    with patch(
        "agi_ecom_app.manager_decision.advisory.get_inventory_turnover",
        return_value={"totals": {"period_days": 30}, "rows": rows},
    ):
        targets = _portfolio_targets(
            {"scope": "portfolio", "constraints": {"target_cover_days": 30}},
            max_targets=100,
        )

    assert len(targets) == 17
    assert set(targets) == {str(46220 + index) for index in range(17)}


def test_portfolio_focus_targets_are_deep_dives_over_full_inventory() -> None:
    turnover = {
        "totals": {"period_days": 30},
        "rows": [
            {"vendor_code": "46227", "sales_units": 30, "current_stock_qty": 30},
            {"vendor_code": "46230", "sales_units": 30, "current_stock_qty": 130},
            {"vendor_code": "46303", "sales_units": 30, "current_stock_qty": 90},
        ],
    }
    identities = {
        target: {"nm_id": index, "vendor_code": target}
        for index, target in enumerate(("46227", "46230", "46303"), start=1)
    }
    with (
        patch(
            "agi_ecom_app.manager_decision.advisory._portfolio_turnover_snapshot",
            return_value=turnover,
        ),
        patch(
            "agi_ecom_app.manager_decision.advisory.get_sku_identity_map",
            return_value=identities,
        ),
        patch(
            "agi_ecom_app.manager_decision.advisory.find_growth_opportunities",
            return_value=[],
        ),
        patch(
            "agi_ecom_app.manager_decision.advisory.optimize_portfolio",
            return_value={"schemaVersion": "portfolio-optimizer-v1"},
        ) as optimize,
    ):
        result = build_manager_advisory(
            "portfolio with focus",
            query_plan={
                "scope": "portfolio",
                "targets": [{"target_type": "vendor_code", "value": "46227"}],
                "requested_analyses": ["forecast"],
                "constraints": {"target_cover_days": 30},
            },
            max_targets=3,
        )

    assert result["focusTargets"] == ["46227"]
    assert result["portfolioTargets"] == ["46230", "46303", "46227"]
    assert result["requestedTargets"] == ["46227", "46230", "46303"]
    assert optimize.call_args.kwargs["inventory_rows"] == turnover["rows"]


def test_build_manager_advisory_compacts_forecast_and_pricing_simulation():
    forecast = SimpleNamespace(
        model_dump=lambda **_kwargs: {
            "summary": "Не пополнять",
            "finalRecommendation": "Дозаказ не нужен.",
            "decisionWhy": "Покрытие высокое.",
            "economicEffectRub": None,
            "profitForecastRub": None,
            "blockedBy": ["thin_price_data"],
            "resolutionPlan": [],
            "iterationCount": 1,
        }
    )
    request = SimpleNamespace(model_copy=lambda update: ("request", update))
    pricing_result = SimpleNamespace(
        model_dump=lambda **_kwargs: {
            "modelVersion": "pricing-elasticity-bootstrap-v1",
            "sampleCount": 500,
            "candidateCount": 3,
            "winnerCandidateId": "price-5335",
            "candidates": [
                {
                    "priceRub": 5335,
                    "operation": "set_price",
                    "status": "completed",
                    "rank": 1,
                    "isWinner": True,
                    "distribution": {
                        "total": {
                            "mean": 8231,
                            "p10": 4976,
                            "cvar10": 4072,
                            "probabilityPositive": 1,
                        }
                    },
                    "unitsSoldDistribution": {"total": {"mean": 8.88}},
                    "stockoutProbability": 0,
                    "constraintViolations": [],
                }
            ],
        }
    )
    source = {
        "currentPriceSnapshot": {"seller_effective_price_rub": 5615.85},
        "marketContextHistory": [
            {"snapshot_date": "2026-07-20", "market_price_median": 771},
            {"snapshot_date": "2026-07-19", "market_price_median": 767},
        ],
        "historyPeriodStart": "2026-01-21",
        "historyPeriodEnd": "2026-07-19",
        "historyRowCount": 180,
        "causalValid": False,
    }

    with (
        patch(
            "agi_ecom_app.manager_decision.advisory.get_sku_identity_map",
            return_value={
                "46303": {"nm_id": 171742855, "vendor_code": "46303"}
            },
        ),
        patch(
            "agi_ecom_app.manager_decision.advisory.find_growth_opportunities",
            return_value=[_opportunity()],
        ),
        patch(
            "agi_ecom_app.manager_decision.advisory.run_forecast_workflow",
            return_value=forecast,
        ),
        patch(
            "agi_ecom_app.manager_decision.advisory.get_sku_daily_price_sales_comparison",
            return_value={
                "causal_valid": False,
                "rows": [
                    {
                        "report_date": "2026-07-19",
                        "seller_effective_price_rub": 5615.85,
                        "seller_discount_pct": 13,
                        "add_to_cart_conversion": 4,
                        "cart_to_order_conversion": 25,
                        "non_cancelled_order_units": 0,
                        "realized_sales_units": 5,
                    }
                ],
            },
        ),
        patch(
            "agi_ecom_app.manager_decision.advisory.get_price_decision_overview",
            return_value={
                "rows": [
                    {
                        "vendor_code": "46303",
                        "nm_id": 171742855,
                        "elasticity_to_apply": -1.1,
                    }
                ]
            },
        ) as price_overview,
        patch(
            "agi_ecom_app.manager_decision.advisory.load_pricing_simulation_source",
            return_value=(request, source),
        ) as load_source,
        patch(
            "agi_ecom_app.manager_decision.advisory.run_pricing_simulation",
            return_value=pricing_result,
        ) as run_pricing,
    ):
            result = build_manager_advisory(
                "Для SKU 46303 сравни варианты цены и ожидаемый эффект",
                query_plan={
                    "scope": "single_sku",
                    "targets": [
                        {"target_type": "vendor_code", "value": "46303"}
                    ],
                    "requested_analyses": ["price_scenarios", "forecast"],
                },
            )

    assert result["requestedTargets"] == ["46303"]
    assert result["forecasts"][0]["finalRecommendation"] == "Дозаказ не нужен."
    assert result["dailyComparisons"][0]["rows"][0]["realizedSales"] == 5
    assert result["priceSimulations"][0]["candidates"][0] == {
        "priceRub": 5335,
        "operation": "set_price",
        "status": "completed",
        "rank": 1,
        "isWinner": True,
        "meanEffectRub": 8231,
        "p10EffectRub": 4976,
        "cvar10EffectRub": 4072,
        "probabilityPositive": 1,
        "expectedUnits": 8.88,
        "stockoutProbability": 0,
        "unitContributionRub": None,
        "priceMovePct": -5.0,
        "violations": [],
    }
    assert result["portfolioOptimization"]["schemaVersion"] == "portfolio-optimizer-v1"
    assert result["portfolioOptimization"]["complete"] is False
    assert result["priceSimulations"][0]["horizonDays"] == 7
    assert result["priceSimulations"][0]["currentPriceRub"] == 5615.85
    assert result["priceSimulations"][0]["availableStockUnits"] is None
    assert result["priceSimulations"][0]["dailyPriceRows"] == result["dailyComparisons"][0]["rows"]
    assert result["priceSimulations"][0]["marketContextHistory"] == source["marketContextHistory"]
    load_source.assert_called_once_with(nm_id=171742855, horizon_days=7)
    price_overview.assert_called_once_with(None, limit=100)
    assert run_pricing.call_args.args[0][0] == "request"
    assert len(run_pricing.call_args.args[0][1]["snapshot_hash"]) == 64


def test_build_manager_advisory_attaches_hypotheses_for_mixed_strict_result() -> None:
    request = SimpleNamespace()
    request.model_copy = lambda update: request
    pricing_result = SimpleNamespace(
        model_dump=lambda **_kwargs: {
            "modelVersion": "pricing-elasticity-bootstrap-v1",
            "sampleCount": 500,
            "candidateCount": 1,
            "winnerCandidateId": None,
            "candidates": [
                {
                    "priceRub": 1000,
                    "operation": "hold_price",
                    "status": "completed",
                    "constraintViolations": [],
                },
                {
                    "priceRub": 975,
                    "operation": "set_price",
                    "status": "completed",
                    "constraintViolations": [],
                },
                {
                    "priceRub": 950,
                    "operation": "set_price",
                    "status": "blocked",
                    "constraintViolations": [
                        {"code": "invalid_price_elasticity"}
                    ],
                }
            ],
        }
    )
    prior = {
        "vendor_code": "46303",
        "nm_id": 171742855,
        "elasticity_to_apply": -1.1,
    }
    hypothesis = {
        "status": "hypothesis",
        "calculationMode": "price_prior_elasticity_sensitivity",
        "confidence": "low",
        "confidenceScore": 0.3,
        "executionEligible": False,
        "assumptions": ["Эластичность используется как prior."],
        "evidenceGaps": ["invalid_price_elasticity"],
        "candidates": [
            {
                "priceRub": 950,
                "status": "hypothesis",
                "rank": 1,
                "mean": 1500,
                "p10": -200,
                "probPositive": 0.7,
                "expectedUnits": 12,
                "stockout": 0.1,
            }
        ],
    }
    source = {
        "currentPriceSnapshot": {"seller_effective_price_rub": 1000},
        "stockSnapshot": {"available_stock": 100},
    }

    with (
        patch(
            "agi_ecom_app.manager_decision.advisory.get_sku_identity_map",
            return_value={
                "46303": {"nm_id": 171742855, "vendor_code": "46303"}
            },
        ),
        patch(
            "agi_ecom_app.manager_decision.advisory.find_growth_opportunities",
            return_value=[],
        ),
        patch(
            "agi_ecom_app.manager_decision.advisory.get_price_decision_overview",
            return_value={"rows": [prior]},
        ) as overview,
        patch(
            "agi_ecom_app.manager_decision.advisory.get_sku_daily_price_sales_comparison",
            return_value={"rows": []},
        ),
        patch(
            "agi_ecom_app.manager_decision.advisory.load_pricing_simulation_source",
            return_value=(request, source),
        ),
        patch(
            "agi_ecom_app.manager_decision.advisory.run_pricing_simulation",
            return_value=pricing_result,
        ),
        patch(
            "agi_ecom_app.manager_decision.advisory.build_pricing_hypotheses",
            return_value=hypothesis,
        ) as build_hypotheses,
        patch(
            "agi_ecom_app.manager_decision.advisory.optimize_portfolio",
            return_value={"schemaVersion": "portfolio-optimizer-v1"},
        ),
    ):
        result = build_manager_advisory(
            "Покажи варианты цены для SKU 46303",
            query_plan={
                "scope": "single_sku",
                "targets": [{"target_type": "vendor_code", "value": "46303"}],
                "requested_analyses": ["price_scenarios"],
            },
        )

    simulation = result["priceSimulations"][0]
    assert simulation["hypothesisCandidates"] == hypothesis["candidates"]
    assert simulation["hypothesisConfidence"] == "low"
    assert simulation["hypothesisExecutionEligible"] is False
    overview.assert_called_once_with(None, limit=100)
    build_hypotheses.assert_called_once_with(
        request,
        pricing_result,
        price_prior=prior,
    )


def test_render_manager_advisory_keeps_blocked_execution_but_shows_options():
    markdown = render_manager_advisory(
        {
            "forecasts": [
                {
                    "target": "46303",
                    "vendorCode": "46303",
                    "finalRecommendation": "Дозаказ не нужен.",
                    "decisionWhy": "Покрытие 99.8 дня.",
                    "blockedBy": ["thin_price_data"],
                    "resolutionPlan": [],
                }
            ],
            "dailyComparisons": [
                {
                    "target": "46303",
                    "vendorCode": "46303",
                    "rows": [
                        {
                            "date": "2026-07-19",
                            "sellerEffectivePriceRub": 5615.85,
                            "sellerDiscountPct": 13,
                            "addToCartConversionPct": 4,
                            "cartToOrderConversionPct": 25,
                            "orders": 0,
                            "realizedSales": 5,
                        }
                    ],
                }
            ],
            "priceSimulations": [
                {
                    "target": "46303",
                    "vendorCode": "46303",
                    "sampleCount": 500,
                    "candidates": [
                        {
                            "priceRub": 5335,
                            "status": "completed",
                            "rank": 1,
                            "isWinner": True,
                            "meanEffectRub": 8231,
                            "p10EffectRub": 4976,
                            "probabilityPositive": 1,
                            "stockoutProbability": 0,
                        }
                    ],
                }
            ],
        }
    )

    assert "Автоматическое исполнение может оставаться заблокированным" in markdown
    assert "Дозаказ не нужен" in markdown
    assert "Цена, скидка, конверсия и продажи по дням" in markdown
    assert "| 2026-07-19 | 5 616 ₽ | 13.0% | 4.0% | 25.0% | 0 | 5 |" in markdown
    assert "| 1 | 5 335 ₽ | — | 8 231 ₽ | 4 976 ₽ | 100.0% | 0.0% |" in markdown
    assert "не проверен оптимизатором" in markdown
    assert "Допустимого изменения цены нет" in markdown


def test_render_manager_advisory_enforces_cross_domain_scenarios_and_five_recommendations():
    query_plan = {
        "scope": "portfolio",
        "decision_kind": "action_portfolio",
        "required_domains": ["price", "marketing", "supply"],
        "requested_analyses": ["price_scenarios", "advertising", "replenishment"],
        "scenario_requests": ["ценовые", "рекламные", "поставочные"],
    }
    manager_decision = {
        "status": "blocked",
        "rankedActions": [],
        "rejectedCandidates": [
            {
                "candidateId": "ad-1",
                "family": "advertising",
                "target": "35839929",
                "operation": "pause_campaign",
                "from": {"value": 9, "unit": "wb_status_code"},
                "to": {"value": 11, "unit": "wb_status_code"},
                "reasons": [{"message": "Нет campaign contribution profit."}],
            },
            {
                "candidateId": "supply-1",
                "family": "supply",
                "target": "155863996",
                "operation": "create_replenishment_task",
                "from": {"value": 0, "unit": "units_ordered"},
                "to": {"value": 56, "unit": "units_ordered"},
                "reasons": [{"message": "Нет lead time и MOQ."}],
            },
        ],
    }
    markdown = render_manager_advisory(
        {
            "forecasts": [],
            "dailyComparisons": [],
            "priceSimulations": [
                {
                    "target": "46303",
                    "vendorCode": "46303",
                    "sampleCount": 500,
                    "candidates": [
                        {
                            "priceRub": 5335,
                            "status": "completed",
                            "rank": 1,
                            "isWinner": True,
                            "meanEffectRub": 8231,
                            "p10EffectRub": 4976,
                            "probabilityPositive": 1,
                            "stockoutProbability": 0,
                        }
                    ],
                }
            ],
            "portfolioOptimization": {
                "complete": True,
                "violations": [],
                "scenarios": [
                    {
                        "scenarioId": "price:46303:5335",
                        "family": "price",
                        "target": "46303",
                        "inventoryDaysReduction": 11.4,
                    },
                    {
                        "scenarioId": "marketing:pause:ad-1",
                        "family": "marketing",
                        "target": "ad-1",
                        "action": "Кампания ad-1: приостановить расход без заказов",
                        "inventoryDaysReduction": 0,
                        "expectedContributionProfitRub": 1000,
                        "p10ContributionProfitRub": 1000,
                        "stockoutProbability": 0,
                        "calculationStatus": "completed",
                        "constraintReasons": [],
                    },
                    {
                        "scenarioId": "supply:replenish:46229",
                        "family": "supply",
                        "target": "46229",
                        "action": "SKU 46229: рассмотреть дозаказ 40 шт.",
                        "inventoryDaysReduction": -20,
                        "expectedContributionProfitRub": None,
                        "p10ContributionProfitRub": None,
                        "stockoutProbability": 1,
                        "calculationStatus": "needs_data",
                        "constraintReasons": ["нет lead time и MOQ"],
                    },
                ],
                "topRecommendations": [
                    {
                        "family": "price" if index < 3 else "marketing",
                        "target": f"target-{index}",
                        "action": f"Числовое решение {index}",
                        "inventoryDaysBefore": 90,
                        "inventoryDaysAfter": 60 if index < 3 else 90,
                        "inventoryDaysReduction": 30 if index < 3 else 0,
                        "expectedContributionProfitRub": 5000 - index * 100,
                        "p10ContributionProfitRub": 3000 - index * 100,
                        "stockoutProbability": 0.01,
                        "businessScore": 90 - index,
                        "eligible": True,
                        "limitations": ["теневой расчёт"],
                    }
                    for index in range(5)
                ],
            },
        },
        manager_decision=manager_decision,
        query_plan=query_plan,
    )

    assert "## Проверка вариантов цены" in markdown
    assert "## Рекламные сценарии" in markdown
    assert "## Поставочные сценарии" in markdown
    assert "## 5 приоритетных рекомендаций" in markdown
    priority_section = markdown.split("## 5 приоритетных рекомендаций", 1)[1]
    priority_section = priority_section.split("\n## ", 1)[0]
    priority_rows = [
        line for line in priority_section.splitlines() if line.startswith("| ") and line[2:3].isdigit()
    ]
    assert len(priority_rows) == 5
    assert all("| blocked |" in row for row in priority_rows)
    assert all("Числовое решение" in row for row in priority_rows)
    assert "Рассчитано допустимых решений" not in markdown
    assert markdown.index("## 5 приоритетных рекомендаций") < markdown.index(
        "## Проверка вариантов цены"
    )
    assert validate_scenario_completeness(markdown, query_plan) == []


def test_price_only_business_answer_uses_supporting_domains_only_as_evidence() -> None:
    targets = ["A", "B", "C", "D", "E"]
    hypotheses = [
        {
            "scenarioId": f"price:{target}:{950 - index * 5}",
            "family": "price",
            "target": target,
            "action": f"SKU {target}: проверить снижение цены",
            "calculationStatus": "hypothesis",
            "executionStatus": "shadow_test",
            "currentPriceRub": 1000,
            "candidatePriceRub": 950 - index * 5,
            "priceMovePct": -5 - index * 0.5,
            "baselineExpectedUnits": 30,
            "expectedUnits": 40 + index,
            "inventoryDaysBefore": 90,
            "inventoryDaysAfter": 70 - index,
            "inventoryDaysReduction": 20 + index,
            "expectedContributionProfitRub": 3000 - index * 100,
            "p10ContributionProfitRub": 1800 - index * 60,
            "probabilityPositive": 0.91,
            "stockoutProbability": 0.01,
            "confidence": "low",
            "eligible": False,
            "hypothesisEligible": True,
            "evidenceGaps": ["нет контролируемого ценового эксперимента"],
        }
        for index, target in enumerate(targets)
    ]
    polluted_priority = [
        *hypotheses[:3],
        {
            "scenarioId": "verification:marketing:A",
            "family": "marketing",
            "target": "A",
            "action": "проверить рекламу",
            "eligible": False,
            "constraintReasons": ["нет причинной оценки"],
        },
        {
            "scenarioId": "verification:supply:A",
            "family": "supply",
            "target": "A",
            "action": "проверить поставку",
            "eligible": False,
            "constraintReasons": ["нет ETA"],
        },
    ]
    simulations = [
        {
            "target": target,
            "vendorCode": target,
            "currentPriceRub": 1000,
            "currentPriceSnapshot": {
                "snapshot_date": "2026-07-20" if index == 0 else "2026-07-21"
            },
            "availableStockUnits": 90,
            "historyPeriodStart": "2026-01-01",
            "historyPeriodEnd": "2026-07-20",
            "causalValid": False,
            "candidates": [
                {
                    "operation": "hold_price",
                    "priceRub": 1000,
                    "status": "completed",
                    "expectedUnits": 30,
                    "meanEffectRub": 0,
                    "p10EffectRub": 0,
                    "stockoutProbability": 0,
                }
            ],
            "hypothesisCandidates": [
                {
                    "operation": "set_price",
                    "priceRub": 950 - index * 5,
                    "status": "hypothesis",
                    "expectedUnits": 40 + index,
                    "mean": 3000 - index * 100,
                    "p10": 1800 - index * 60,
                    "stockout": 0.01,
                }
            ],
        }
        for index, target in enumerate(targets)
    ]
    query_plan = {
        "scope": "portfolio",
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
        "scenario_requests": ["-5%", "-2.5%", "hold", "+2.5%", "+5%"],
        "constraints": {"horizon_days": 30, "target_cover_days": 30},
    }

    markdown = render_manager_advisory(
        {
            "focusTargets": ["A"],
            "requestedTargets": targets,
            "portfolioTargets": targets,
            "coverage": {
                "targetSkuCount": 5,
                "coveredSkuCount": 5,
                "artifactTargets": targets,
                "errorTargets": [],
                "missingArtifactTargets": [],
                "numericPriceScenarioSkuCount": 5,
            },
            "priceSimulations": simulations,
            "portfolioOptimization": {
                "scenarioCount": 25,
                "horizonDays": 30,
                "priorityRecommendations": polluted_priority,
                "topHypotheses": hypotheses,
                "topRecommendations": [],
                "scenarios": hypotheses,
                "goalAttainment": {
                    "status": "not_achieved",
                    "targetCoverDays": 30,
                    "remainingAboveTargetSkus": [
                        {"target": target} for target in targets
                    ],
                    "unmeasuredSkus": [],
                },
            },
        },
        manager_decision={"status": "blocked"},
        query_plan=query_plan,
    )

    assert markdown.index("## Итог для бизнеса") < markdown.index(
        "## 5 приоритетных рекомендаций"
    )
    assert "Цена сейчас → тест" in markdown
    assert "1 000 ₽ → 950 ₽" in markdown
    assert "91.0%" in markdown
    assert "Сумма средних ΔCP выбранных вариантов" in markdown
    assert "рассчитано 5/5 SKU; ошибки 0; без результата 0" in markdown
    assert "срезы 2026-07-20–2026-07-21" in markdown
    assert "## Полная матрица ценовых вариантов" in markdown
    assert "## Рекламные сценарии" not in markdown
    assert "## Поставочные сценарии" not in markdown
    assert "проверить рекламу" not in markdown
    assert "проверить поставку" not in markdown
    priority_section = markdown.split("## 5 приоритетных рекомендаций", 1)[1]
    priority_section = priority_section.split("\n## ", 1)[0]
    priority_rows = [
        line
        for line in priority_section.splitlines()
        if re.match(r"^\|\s*\d+\s*\|", line.strip())
    ]
    assert len(priority_rows) == 5
    assert validate_scenario_completeness(markdown, query_plan) == []


def test_render_manager_advisory_replaces_empty_price_scenario_table_with_status():
    query_plan = {
        "scope": "portfolio",
        "decision_kind": "none",
        "required_domains": ["price"],
        "requested_analyses": ["price_scenarios"],
        "scenario_requests": ["ценовые"],
    }
    markdown = render_manager_advisory(
        {
            "priceSimulations": [
                {
                    "target": "46230",
                    "vendorCode": "46230",
                    "sampleCount": 500,
                    "candidates": [],
                }
            ]
        },
        query_plan=query_plan,
    )

    assert "### SKU 46230" in markdown
    assert "пустая таблица не публикуется" in markdown
    assert "| Ранг | Цена |" not in markdown
    assert validate_scenario_completeness(markdown, query_plan) == []


def test_render_manager_advisory_shows_hypotheses_when_strict_variants_blocked():
    markdown = render_manager_advisory(
        {
            "priceSimulations": [
                {
                    "target": "46303",
                    "vendorCode": "46303",
                    "currentPriceRub": 1000,
                    "candidates": [
                        {"priceRub": 950, "status": "blocked", "rank": None}
                    ],
                    "hypothesisConfidence": "low",
                    "hypothesisConfidenceScore": 0.3,
                    "hypothesisAssumptions": [
                        "Эластичность используется только как чувствительность"
                    ],
                    "hypothesisEvidenceGaps": [
                        "Нужно подтвердить собственную эластичность"
                    ],
                    "hypothesisCandidates": [
                        {
                            "priceRub": 950,
                            "status": "hypothesis",
                            "rank": 1,
                            "expectedUnits": 12,
                            "mean": 1500,
                            "p10": -200,
                            "probPositive": 0.7,
                            "stockout": 0.1,
                        }
                    ],
                }
            ]
        },
        query_plan={
            "scope": "single_sku",
            "required_domains": ["price"],
            "requested_analyses": ["price_scenarios"],
        },
    )

    assert "### SKU 46303: гипотезы чувствительности" in markdown
    assert "автоматическое исполнение остаётся отдельным и заблокировано" in markdown
    assert "Уверенность гипотезы: low (30.0%)" in markdown
    assert "Эластичность используется только как чувствительность" in markdown
    assert "| гипотеза / sensitivity | 1 | 950 ₽ | -5.0% | 12 | 1 500 ₽ | -200 ₽ | 70.0% | 10.0% |" in markdown
    assert "пустая таблица не публикуется" not in markdown


def test_render_manager_advisory_shows_hypotheses_when_only_hold_completed() -> None:
    markdown = render_manager_advisory(
        {
            "priceSimulations": [
                {
                    "target": "46303",
                    "vendorCode": "46303",
                    "currentPriceRub": 1000,
                    "candidates": [
                        {
                            "operation": "hold_price",
                            "priceRub": 1000,
                            "status": "completed",
                        },
                        {
                            "operation": "set_price",
                            "priceRub": 950,
                            "status": "blocked",
                        },
                    ],
                    "hypothesisConfidence": "low",
                    "hypothesisCandidates": [
                        {
                            "operation": "set_price",
                            "priceRub": 950,
                            "status": "hypothesis",
                            "rank": 1,
                            "expectedUnits": 12,
                            "mean": 1500,
                            "p10": -200,
                            "probPositive": 0.7,
                            "stockout": 0.1,
                        }
                    ],
                }
            ]
        },
        query_plan={
            "scope": "single_sku",
            "required_domains": ["price"],
            "requested_analyses": ["price_scenarios"],
        },
    )

    assert "### SKU 46303: гипотезы чувствительности" in markdown
    assert "| гипотеза / sensitivity | 1 | 950 ₽ |" in markdown


def test_render_manager_advisory_merges_mixed_strict_and_hypothesis_grid() -> None:
    hypothesis_candidates = [
        {
            "operation": "hold_price" if price == 1000 else "set_price",
            "priceRub": price,
            "status": "hypothesis",
            "rank": rank,
            "expectedUnits": 30 + rank,
            "mean": 1000 - rank * 10,
            "p10": 500 - rank * 10,
            "probPositive": 0.7,
            "stockout": 0.01,
        }
        for rank, price in enumerate((1000, 950, 975, 1025, 1050), start=1)
    ]
    markdown = render_manager_advisory(
        {
            "priceSimulations": [
                {
                    "target": "46303",
                    "vendorCode": "46303",
                    "currentPriceRub": 1000,
                    "sampleCount": 500,
                    "candidates": [
                        {
                            "operation": "hold_price",
                            "priceRub": 1000,
                            "status": "completed",
                            "expectedUnits": 30,
                            "meanEffectRub": 0,
                            "p10EffectRub": 0,
                            "probabilityPositive": 0,
                            "stockoutProbability": 0,
                        },
                        {
                            "operation": "set_price",
                            "priceRub": 975,
                            "status": "completed",
                            "expectedUnits": 35,
                            "meanEffectRub": 900,
                            "p10EffectRub": 400,
                            "probabilityPositive": 0.8,
                            "stockoutProbability": 0.01,
                        },
                        *[
                            {
                                "operation": "set_price",
                                "priceRub": price,
                                "status": "blocked",
                            }
                            for price in (950, 1025, 1050)
                        ],
                    ],
                    "hypothesisConfidence": "low",
                    "hypothesisCandidates": hypothesis_candidates,
                }
            ]
        },
        query_plan={
            "scope": "single_sku",
            "required_domains": ["price"],
            "requested_analyses": ["price_scenarios"],
        },
    )

    price_rows = [
        line
        for line in markdown.splitlines()
        if line.startswith("| ")
        and any(f"| {price:,} ₽ |".replace(",", " ") in line for price in (950, 975, 1000, 1025, 1050))
    ]
    assert len(price_rows) == 5
    assert {line.split("|")[2].strip() for line in price_rows} == {
        f"{price:,} ₽".replace(",", " ")
        for price in (950, 975, 1000, 1025, 1050)
    }
    assert "гипотеза / только тест" in markdown
    strict_row = next(line for line in price_rows if "| 975 ₽ |" in line)
    assert "гипотеза / только тест" not in strict_row


def test_render_manager_advisory_reports_actual_portfolio_coverage() -> None:
    targets = [str(46220 + index) for index in range(17)]
    markdown = render_manager_advisory(
        {
            "requestedTargets": targets,
            "portfolioTargets": targets,
            "priceSimulations": [
                {"target": targets[0], "vendorCode": targets[0], "candidates": []}
            ],
            "errors": [
                {
                    "target": targets[1],
                    "stage": "pricing_simulation",
                    "error_type": "ValueError",
                }
            ],
        },
        query_plan={
            "scope": "portfolio",
            "required_domains": ["price"],
            "requested_analyses": ["price_scenarios"],
        },
    )

    assert "Охвачено SKU: 2/17" in markdown
    assert "Всего SKU в целевой матрице: 17." in markdown
    assert "Численные ценовые сценарии: 0/17." in markdown
    assert f"SKU с явно зафиксированной ошибкой/пробелом: {targets[1]}." in markdown
    assert "Проанализировано SKU: 17/17." not in markdown


def test_hypothesis_recommendation_is_labeled_without_execution_approval() -> None:
    rows = _optimized_recommendations(
        {
            "portfolioOptimization": {
                "priorityRecommendations": [
                    {
                        "target": "46303",
                        "eligible": False,
                        "calculationStatus": "hypothesis",
                    }
                ]
            }
        },
        {"status": "blocked"},
    )

    assert rows[0]["executionStatus"] == "hypothesis"


def test_render_manager_advisory_shows_five_blocked_priority_checks() -> None:
    priority_rows = [
        {
            "scenarioId": f"verification:{index}",
            "family": ("price", "marketing", "supply", "price", "marketing")[index],
            "target": f"SKU-{index}",
            "action": f"Проверить данные для варианта {index + 1}",
            "inventoryMetric": "current_cover_days_7d",
            "inventoryDaysBefore": 90,
            "inventoryDaysAfter": None,
            "inventoryDaysReduction": None,
            "expectedContributionProfitRub": None,
            "p10ContributionProfitRub": None,
            "stockoutProbability": None,
            "businessScore": 0,
            "eligible": False,
            "constraintReasons": ["эффект не подтверждён"],
            "limitations": [],
        }
        for index in range(5)
    ]
    query_plan = {
        "scope": "portfolio",
        "decision_kind": "action_portfolio",
    }

    markdown = render_manager_advisory(
        {
            "portfolioOptimization": {
                "topRecommendations": [],
                "priorityRecommendations": priority_rows,
            }
        },
        manager_decision={"status": "blocked"},
        query_plan=query_plan,
    )

    section = markdown.split("## 5 приоритетных рекомендаций", 1)[1]
    rows = [
        line
        for line in section.splitlines()
        if line.strip().startswith(("| 1 |", "| 2 |", "| 3 |", "| 4 |", "| 5 |"))
    ]
    assert len(rows) == 5
    assert all("| blocked |" in row for row in rows)
    assert "эффект не подтверждён" in markdown
    assert "Заполнители не добавляются" not in markdown
    assert validate_scenario_completeness(markdown, query_plan) == []


def test_goal_attainment_is_explicit_and_coverage_metrics_are_not_mixed() -> None:
    query_plan = {
        "scope": "portfolio",
        "decision_kind": "action_portfolio",
        "constraints": {"target_cover_days": 30, "horizon_days": 60},
    }
    advisory = {
        "portfolioOptimization": {
            "goalAttainment": {
                "status": "not_achieved",
                "targetCoverDays": 30,
                "remainingAboveTargetSkus": [
                    {
                        "target": "46227",
                        "currentCoverDays7d": 78.9,
                        "bestScenarioCoverDays": 60.4,
                        "bestScenarioMetric": "simulated_forward_cover_days",
                        "bestScenarioId": "price:46227:5450",
                    },
                    {
                        "target": "46230",
                        "currentCoverDays7d": 195.1,
                        "bestScenarioCoverDays": None,
                        "bestScenarioMetric": None,
                        "bestScenarioId": None,
                    },
                ],
                "unmeasuredSkus": [],
            },
            "topRecommendations": [],
        }
    }

    markdown = render_manager_advisory(
        advisory,
        manager_decision={"status": "blocked"},
        query_plan=query_plan,
    )

    assert "## Достижение цели 30 дней" in markdown
    assert "Итог: цель не достигнута" in markdown
    assert "Текущее покрытие считается только" in markdown
    assert "| 46227 | 78.9 | 60.4 | прогноз покрытия по 500 симуляциям" in markdown
    assert "| 46230 | 195.1 | — | сценарий в top-5 отсутствует" in markdown
    violations = validate_scenario_completeness(markdown, query_plan)
    assert "missing goal attainment section" not in violations
    assert "missing explicit goal attainment verdict" not in violations


def test_goal_attainment_validator_rejects_missing_verdict() -> None:
    query_plan = {
        "decision_kind": "action_portfolio",
        "constraints": {"target_cover_days": 30},
    }
    violations = validate_scenario_completeness(
        "## Достижение цели 30 дней\n\n## 5 приоритетных рекомендаций",
        query_plan,
    )

    assert "missing explicit goal attainment verdict" in violations
