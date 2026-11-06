import json
import unittest
from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

from agi_ecom_app.forecast import find_growth_opportunities
from agi_ecom_app.forecast.critic import (
    build_forecast_critic_agent,
    build_forecast_critic_payload,
    run_forecast_critic,
)
from agi_ecom_app.forecast.calculators.action_family import _append_price_safety_blockers
from agi_ecom_app.forecast.opportunity_finder import _sku_opportunities
from agi_ecom_app.forecast.adapters.diagnostic_input import ForecastEvidenceBundle
from agi_ecom_app.forecast.adapters.task_normalizer import (
    build_forecast_hypothesis,
    build_forecast_plan,
    resolve_forecast_targets,
)
from agi_ecom_app.forecast.reasoning import (
    build_forecast_reasoning_agent,
    build_forecast_reasoning_input,
    build_forecast_reasoning_payload,
    build_forecast_revision_input,
)
from agi_ecom_app.forecast.models import (
    ForecastCriticIssue,
    ForecastCriticResult,
    ForecastDecisionResult,
    ForecastDelta,
    ForecastReasoningInput,
    ForecastRevisionInput,
    ForecastPlan,
    ForecastReasoningResult,
    ForecastRouteProposal,
    ForecastResolutionPlanStep,
    ForecastScenario,
    GrowthOpportunity,
    ResolvedTarget,
)
from agi_ecom_app.forecast.rendering import render_forecast_decision_result
from agi_ecom_app.forecast.workflow import run_forecast_workflow


def _marketing_task(**updates):
    payload = {
        "actionKind": "optimize_marketing_visibility",
        "decisionType": "marketing_visibility_guardrail",
        "title": "Сократить или остановить слабые кампании: 295603220, 309891193",
        "action": "Сократить или остановить слабые кампании: 295603220, 309891193",
        "targets": ["295603220", "309891193"],
        "businessGoal": "Понять, какие рекламные действия можно менять без потери эффективности.",
        "forecastSpec": {
            "horizon_days": 14,
            "period_days": 7,
            "forecast_capable": True,
            "forecast_domain": "marketing",
        },
    }
    payload.update(updates)
    return payload


def _weak_sku_task(**updates):
    payload = {
        "actionKind": "diagnose_card_funnel",
        "decisionType": "card_funnel_diagnostic",
        "title": "Проверить карточки и спрос по слабым SKU: 46303, 46313, 46229",
        "action": "Проверить карточки и спрос по слабым SKU: 46303, 46313, 46229",
        "targets": ["46303", "46313", "46229"],
        "businessGoal": "Понять причину просадки спроса и выбрать следующий шаг по SKU.",
        "forecastSpec": {
            "horizon_days": 14,
            "period_days": 7,
            "forecast_capable": True,
            "forecast_domain": "sales",
        },
    }
    payload.update(updates)
    return payload


def _marketing_sku_decline_task(**updates):
    payload = {
        "actionKind": "optimize_marketing_visibility",
        "decisionType": "marketing_visibility_guardrail",
        "title": "Разобрать SKU с падением продаж: 46227, 46313, 46303, 46311",
        "action": "Разобрать SKU с падением продаж: 46227, 46313, 46303, 46311",
        "targets": ["46311"],
        "businessGoal": "Понять, какие рекламные или поисковые действия можно менять без потери эффективности.",
        "observation": "падение выручки подтверждено, первопричина не закрыта",
        "executionPlan": (
            "Проверить видимость, позиции, спрос, рекламные расходы и заказы; затем выбрать: "
            "масштабировать, сократить, остановить или оставить с контролем."
        ),
        "forecastSpec": {
            "horizon_days": 14,
            "period_days": 7,
            "forecast_capable": True,
            "forecast_domain": "marketing",
        },
    }
    payload.update(updates)
    return payload


def _availability_before_scale_task(**updates):
    payload = {
        "actionKind": "preserve_availability_and_sales",
        "decisionType": "availability_replenishment_marketing_guardrail",
        "title": "Сверить доступность перед расширением рекламы: 46310, 46313, 46227",
        "action": "Сверить доступность перед расширением рекламы: 46310, 46313, 46227",
        "targets": ["46310", "46313", "46227"],
        "businessGoal": "Не допустить out-of-stock перед расширением рекламы.",
        "forecastSpec": {
            "horizon_days": 14,
            "period_days": 7,
            "forecast_capable": True,
            "forecast_domain": "supply",
        },
    }
    payload.update(updates)
    return payload


def _price_task(**updates):
    payload = {
        "actionKind": "validate_price_positioning",
        "decisionType": "price_margin_competitor_guardrail",
        "title": "Проверить цену SKU: 46227",
        "action": "Проверить цену SKU: 46227",
        "targets": ["46227"],
        "businessGoal": "Понять безопасное ценовое решение.",
        "forecastSpec": {"horizon_days": 14, "period_days": 7, "forecast_domain": "price"},
    }
    payload.update(updates)
    return payload


def _mock_reasoning_result(*, reasoning_input, plan, today=None):
    calculator_output = reasoning_input.calculator_output
    details_patch = {
        "reasoningPrimaryAction": plan.action_family,
        "reasoningRejectedActions": [],
        "reasoningEvidenceUsed": ["unit_test_reasoning_mock"],
        "reasoningWarnings": list(reasoning_input.soft_warnings),
        "reasoningMode": "mock",
    }
    result = ForecastReasoningResult(
        decision_status=str(calculator_output.get("decision_status") or "needs_review"),
        confidence=str(calculator_output.get("confidence") or "medium"),
        decision=str(calculator_output.get("decision") or calculator_output.get("final_recommendation") or plan.action_family),
        final_recommendation=str(
            calculator_output.get("final_recommendation")
            or calculator_output.get("decision")
            or plan.action_family
        ),
        summary=str(
            calculator_output.get("summary")
            or calculator_output.get("effect_narrative")
            or calculator_output.get("baseline_narrative")
            or plan.action_family
        ),
        decision_why=str(
            calculator_output.get("decision_why")
            or "Mock reasoning layer reused deterministic math output for test stability."
        ),
        primary_action=plan.action_family,
        rejected_actions=[],
        warnings=list(reasoning_input.soft_warnings),
        evidence_used=["unit_test_reasoning_mock"],
        forecast_details_patch=details_patch,
    )
    debug = {"mode": "mock", "used_tools": [], "decision_trace": []}
    return result, debug


class ForecastWorkflowTests(unittest.TestCase):
    def setUp(self):
        # New mart lookups are optional evidence. Tests with concrete evidence
        # override these defaults with their own per-test patches.
        for name in (
            "get_marketing_campaign_daily_mart",
            "get_marketing_forecast_ready_mart",
            "get_marketing_query_bands_mart",
            "get_marketing_action_matrix",
            "get_marketing_visibility_snapshot",
            "get_marketing_search_demand_snapshot",
            "get_sales_sku_revenue_change",
            "get_sales_overview",
        ):
            source = patch(
                "agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db." + name,
                return_value={"rows": [], "source_status": "missing"},
            )
            source.start()
            self.addCleanup(source.stop)
        self.reasoning_patcher = patch(
            "agi_ecom_app.forecast.workflow.run_forecast_reasoning",
            side_effect=_mock_reasoning_result,
        )
        self.critic_settings_patcher = patch(
            "agi_ecom_app.forecast.critic.get_settings",
            return_value=SimpleNamespace(openai_api_key="", openai_model="test-model"),
        )
        self.reasoning_patcher.start()
        self.critic_settings_patcher.start()

    def tearDown(self):
        self.reasoning_patcher.stop()
        self.critic_settings_patcher.stop()

    def test_task_normalizer_builds_marketing_stop_reduce_plan(self):
        plan = build_forecast_plan(_marketing_task())
        hypothesis = build_forecast_hypothesis(_marketing_task())

        self.assertIsNotNone(plan)
        self.assertIsNotNone(hypothesis)
        self.assertEqual(plan.primary_domain, "marketing")
        self.assertEqual(plan.action_family, "stop_or_reduce_campaigns")
        self.assertEqual(hypothesis.action, "stop_or_reduce_campaigns")

    def test_task_normalizer_routes_weak_sku_to_recovery_instead_of_campaign_stop(self):
        plan = build_forecast_plan(_weak_sku_task())
        hypothesis = build_forecast_hypothesis(_weak_sku_task())

        self.assertIsNotNone(plan)
        self.assertIsNotNone(hypothesis)
        self.assertEqual(plan.primary_domain, "sales")
        self.assertEqual(plan.action_family, "weak_sku_demand_recovery")
        self.assertIn("catalog_demand", plan.secondary_domains)
        self.assertEqual(hypothesis.action, "weak_sku_demand_recovery")

    def test_task_normalizer_routes_marketing_sku_decline_to_recovery(self):
        plan = build_forecast_plan(_marketing_sku_decline_task())
        hypothesis = build_forecast_hypothesis(_marketing_sku_decline_task())

        self.assertIsNotNone(plan)
        self.assertIsNotNone(hypothesis)
        self.assertEqual(plan.primary_domain, "sales")
        self.assertEqual(plan.action_family, "weak_sku_demand_recovery")
        self.assertEqual(hypothesis.action, "weak_sku_demand_recovery")

    def test_task_normalizer_supports_stored_task_snake_case_fields(self):
        stored_like = {
            "action_kind": "optimize_marketing_visibility",
            "decision_type": "marketing_visibility_guardrail",
            "title": "Остановить неэффективную кампанию: 309891193",
            "action": "Остановить неэффективную кампанию: 309891193",
            "targets": ["309891193", "кампанию"],
            "business_goal": "Понять, какие рекламные или поисковые действия можно менять без потери эффективности.",
            "forecast_spec": {
                "horizon_days": 14,
                "period_days": 7,
                "forecast_capable": True,
                "forecast_domain": "marketing",
            },
            "agent_prompt": (
                "Проверить видимость, позиции, спрос, рекламные расходы и заказы; затем выбрать: "
                "масштабировать, сократить, остановить или оставить с контролем."
            ),
        }

        plan = build_forecast_plan(stored_like)

        self.assertIsNotNone(plan)
        self.assertEqual(plan.action_family, "stop_or_reduce_campaigns")

    def test_task_normalizer_prefers_normquery_for_single_sku_marketing_review_even_if_title_mentions_campaigns(self):
        payload = {
            "actionKind": "optimize_marketing_visibility",
            "decisionType": "marketing_visibility_guardrail",
            "title": "Проверить кампании с ростом расхода: 46227",
            "action": "Проверить кампании с ростом расхода: 46227",
            "targets": ["46227"],
            "objects": [
                {
                    "objectId": "sku-46227",
                    "label": "SKU 46227",
                    "type": "sku",
                    "externalId": "46227",
                    "payload": {"nm_id": 46227},
                }
            ],
            "businessGoal": "Понять, какие рекламные или поисковые действия можно менять без потери эффективности.",
            "forecastSpec": {
                "horizon_days": 14,
                "period_days": 7,
                "forecast_capable": True,
                "forecast_domain": "marketing",
            },
        }

        plan = build_forecast_plan(payload)

        self.assertIsNotNone(plan)
        self.assertEqual(plan.action_family, "normquery_bid_optimization")
        self.assertEqual(plan.target_mode, "sku")

    @patch("agi_ecom_app.forecast.planner.route_task_with_llm")
    def test_task_normalizer_uses_llm_router_for_ambiguous_marketing_visibility_tasks(self, mock_route):
        mock_route.return_value = (
            ForecastRouteProposal(
                route=True,
                action_family="normquery_bid_optimization",
                primary_domain="marketing",
                target_mode="mixed",
                confidence="high",
                rationale="Задача про запросы, ставки и видимость.",
            ),
            {"mode": "mock"},
        )

        plan = build_forecast_plan(
            {
                "actionKind": "optimize_marketing_visibility",
                "decisionType": "marketing_visibility_guardrail",
                "title": "Перенести бюджет из слабой кампании в лидеров: 309891193, 156051054, 70194686",
                "action": "Перенести бюджет из слабой кампании в лидеров: 309891193, 156051054, 70194686",
                "targets": ["309891193", "156051054", "70194686", "лидеров", "кампании"],
                "businessGoal": "Понять, какие рекламные или поисковые действия можно менять без потери эффективности.",
                "forecastSpec": {
                    "horizon_days": 14,
                    "period_days": 7,
                    "forecast_capable": True,
                    "forecast_domain": "marketing",
                },
            }
        )

        self.assertIsNotNone(plan)
        self.assertEqual(plan.action_family, "normquery_bid_optimization")
        self.assertEqual(plan.target_mode, "mixed")
        mock_route.assert_called_once()

    @patch("agi_ecom_app.forecast.planner.route_task_with_llm")
    def test_task_normalizer_uses_llm_router_for_unit_economics_tasks(self, mock_route):
        mock_route.return_value = (
            ForecastRouteProposal(
                route=True,
                action_family="reprice_with_margin_guardrail",
                primary_domain="price",
                target_mode="sku",
                confidence="high",
                rationale="Задача про маржу перед scale.",
            ),
            {"mode": "mock"},
        )

        plan = build_forecast_plan(
            {
                "actionKind": "validate_unit_economics",
                "decisionType": "unit_economics_guardrail",
                "title": "Проверить маржу перед усилением рекламы: 46302",
                "action": "Проверить маржу перед усилением рекламы: 46302",
                "targets": ["46302"],
                "businessGoal": "Понять, можно ли выполнять действие без ухудшения экономики.",
                "forecastSpec": {
                    "horizon_days": 14,
                    "forecast_capable": True,
                    "forecast_domain": "marketing",
                },
            }
        )

        self.assertIsNotNone(plan)
        self.assertEqual(plan.action_family, "reprice_with_margin_guardrail")
        self.assertEqual(plan.primary_domain, "price")

    def test_execution_object_scope_wins_over_numbers_in_task_title(self):
        task = _marketing_sku_decline_task(
            objects=[
                {
                    "objectId": "sku-46311",
                    "externalId": "46311",
                    "payload": {"nm_id": 46311},
                }
            ]
        )
        plan = build_forecast_plan(task)

        self.assertIsNotNone(plan)
        targets = resolve_forecast_targets(task, plan)
        self.assertEqual([target.id for target in targets], ["46311"])

    def test_forecast_critic_rejects_vague_scale_language(self):
        decision = ForecastDecisionResult(
            hypothesis="test",
            domain="marketing",
            baseline=ForecastScenario(name="baseline"),
            actionScenario=ForecastScenario(name="action"),
            delta=ForecastDelta(),
            decision="test",
            decisionStatus="completed",
            confidence="high",
            confidencePercent=82,
            finalRecommendation="Тестово расширять рекламу.",
            summary="Тестово расширять рекламу.",
            resolutionPlan=[
                ForecastResolutionPlanStep(
                    stepId="шаг-1",
                    area="guardrail",
                    operation="удержать_масштабирование",
                    instruction="Тестово расширять рекламу только если покрытие остаётся не ниже 38.2 дня и скорость продаж не выходит из плана.",
                )
            ],
        )
        critic_result, _ = run_forecast_critic(decision_result=decision)
        self.assertFalse(critic_result.passed)
        self.assertEqual(critic_result.verdict, "revise_reasoning")
        self.assertEqual(critic_result.retryScope, "reasoning")
        self.assertFalse(critic_result.uiSafe)
        self.assertTrue(any(issue.code == "vague_test_scale" for issue in critic_result.issues))

    def test_forecast_critic_accepts_concrete_resolution_plan(self):
        decision = ForecastDecisionResult(
            hypothesis="test",
            domain="marketing",
            baseline=ForecastScenario(name="baseline"),
            actionScenario=ForecastScenario(name="action"),
            delta=ForecastDelta(),
            decision="test",
            decisionStatus="completed",
            confidence="high",
            confidencePercent=82,
            finalRecommendation="В кампании 16255557 отключить нецелевые запросы и не расширять кампанию до повторной сверки покрытия.",
            summary="В кампании 16255557 отключить нецелевые запросы и не расширять кампанию до повторной сверки покрытия.",
            resolutionPlan=[
                ForecastResolutionPlanStep(
                    stepId="отключить-нецелевой-запрос-1",
                    area="traffic",
                    operation="отключить_запрос",
                    instruction="В кампании 16255557 отключить запрос `сыворотка с лифтинг эффектом`.",
                    advertId="16255557",
                    normQuery="сыворотка с лифтинг эффектом",
                    currentMetric="расход за 7 дней=120 ₽; заказы=0",
                    targetMetric="расход за 7 дней -> 0",
                    stopCondition="не возвращать запрос в кампанию, пока заказов за 7 дней = 0",
                    payload={"advertId": "16255557", "normQuery": "сыворотка с лифтинг эффектом"},
                ),
                ForecastResolutionPlanStep(
                    stepId="удержать-масштабирование",
                    area="guardrail",
                    operation="удержать_масштабирование",
                    instruction="Не расширять рекламу, пока покрытие не будет повторно подтверждено выше 30 дней.",
                    targetMetric="покрытие > 30 дней / риск OOS = low",
                    stopCondition="масштабирование открывать только после повторной сверки покрытия",
                ),
            ],
        )
        critic_result, _ = run_forecast_critic(decision_result=decision)
        self.assertTrue(critic_result.passed)
        self.assertEqual(critic_result.verdict, "passed")
        self.assertTrue(critic_result.uiSafe)

    def test_forecast_critic_rejects_technical_text_and_price_contradiction(self):
        decision = ForecastDecisionResult(
            hypothesis="test",
            domain="price",
            baseline=ForecastScenario(name="baseline"),
            actionScenario=ForecastScenario(name="action"),
            delta=ForecastDelta(),
            decision="test",
            decisionStatus="needs_review",
            confidence="medium",
            confidencePercent=64,
            finalRecommendation="Снизить цену как тестовый кандидат, но не применять автоматически до снятия supply-risk и обновления price decision mart.",
            summary="Сделать repricing после обновления mart.",
            resolutionPlan=[
                ForecastResolutionPlanStep(
                    stepId="изменить-цену",
                    area="price",
                    operation="изменить_цену",
                    instruction="По SKU 46310 изменить цену с 1656 ₽ до 1410 ₽.",
                )
            ],
        )
        critic_result, _ = run_forecast_critic(decision_result=decision)
        self.assertFalse(critic_result.passed)
        self.assertTrue(any(issue.code == "technical_user_facing_text" for issue in critic_result.issues))
        self.assertTrue(any(issue.code == "contradicting_price_execution" for issue in critic_result.issues))

    def test_forecast_critic_blocks_mutation_on_stale_evidence_and_partial_target_coverage(self):
        decision = ForecastDecisionResult(
            hypothesis="test",
            domain="price",
            baseline=ForecastScenario(name="baseline", metrics={"current_price": 1600}),
            actionScenario=ForecastScenario(name="action", metrics={"recommended_price": 1400}),
            delta=ForecastDelta(metrics={"price_delta": -200}),
            decision="изменить цену",
            decisionStatus="completed",
            confidence="high",
            blockedBy=["stale_price_decision"],
            finalRecommendation="Изменить цену по SKU 46227.",
            summary="Изменить цену по SKU 46227.",
            resolvedTargets=[
                ResolvedTarget(id="46227", entity_type="sku"),
                ResolvedTarget(id="46230", entity_type="sku"),
            ],
            forecastDetails={"perTargetEvidence": [{"target": "46227"}]},
            resolutionPlan=[
                ForecastResolutionPlanStep(
                    stepId="изменить-цену",
                    area="price",
                    operation="изменить_цену",
                    instruction="По SKU 46227 изменить цену с 1600 ₽ до 1400 ₽.",
                    targetEntityType="sku",
                    targetEntityId="46227",
                    currentMetric="цена=1600 ₽",
                    targetMetric="цена=1400 ₽; contribution margin > 0",
                    stopCondition="откатить, если contribution margin становится неположительной",
                    payload={"from": 1600, "to": 1400},
                )
            ],
        )

        critic_result, _ = run_forecast_critic(decision_result=decision)

        codes = {issue.code for issue in critic_result.issues}
        self.assertIn("stale_or_future_evidence_for_action", codes)
        self.assertIn("incomplete_target_coverage", codes)
        self.assertFalse(critic_result.passed)

    def test_growth_finder_does_not_treat_missing_stock_as_zero_stock(self):
        opportunities = _sku_opportunities(
            key="46225",
            sales={"current_sales": 7},
            stock={"sales_7d_units": 7, "days_of_stock": 102.4, "reorder_qty_recommended": 0},
            price={"price_status": "thin_data", "competitor_count": 2},
            finance={},
            conversion={},
            period_days=7,
            horizon_days=14,
            focus="profit",
        )

        self.assertNotIn("reorder_for_growth", {item.action_family for item in opportunities})

    def test_stale_thin_price_data_blocks_discount_mutation(self):
        row = {
            "price_status": "thin_data",
            "competitor_count": 2,
            "price_decision_confidence": "low",
            "price_decision_reason_code": "stale_price_ai",
            "price_ai_run_date": "2026-04-14",
            "safety_blockers": [],
        }
        _append_price_safety_blockers(row, today=date(2026, 7, 21))
        self.assertEqual(
            set(row["safety_blockers"]),
            {"thin_price_data", "low_price_decision_confidence", "stale_price_decision"},
        )

        decision = ForecastDecisionResult(
            hypothesis="test",
            domain="price",
            baseline=ForecastScenario(name="baseline"),
            actionScenario=ForecastScenario(name="action"),
            delta=ForecastDelta(),
            decision="запустить скидку",
            decisionStatus="completed",
            confidence="medium",
            blockedBy=["Конкурентный price-срез тонкий или неполный.", "Price decision опирается на устаревший price AI слой."],
            finalRecommendation="Запустить скидку 15%.",
            summary="Запустить скидку 15%.",
            resolutionPlan=[
                ForecastResolutionPlanStep(
                    stepId="запустить-скидку",
                    area="price",
                    operation="запустить_скидку",
                    instruction="По SKU 46303 запустить скидку 15% на 7 дней.",
                    targetEntityType="sku",
                    targetEntityId="46303",
                    currentMetric="покрытие=94 дня",
                    targetMetric="оборачиваемость растёт; маржа положительная",
                    stopCondition="остановить при неположительной марже",
                    payload={"discountPct": 15},
                )
            ],
        )
        critic_result, _ = run_forecast_critic(decision_result=decision)
        codes = {issue.code for issue in critic_result.issues}
        self.assertIn("completed_with_blockers", codes)
        self.assertIn("unsafe_price_action_with_blocker", codes)
        self.assertIn("stale_or_future_evidence_for_action", codes)
        self.assertFalse(critic_result.passed)

    def test_forecast_critic_blocks_incomplete_mutating_action_contract(self):
        decision = ForecastDecisionResult(
            hypothesis="test",
            domain="marketing",
            baseline=ForecastScenario(name="baseline"),
            actionScenario=ForecastScenario(name="action"),
            delta=ForecastDelta(),
            decision="сократить",
            decisionStatus="completed",
            confidence="high",
            finalRecommendation="Сократить кампанию 12345678.",
            summary="Сократить кампанию 12345678.",
            resolutionPlan=[
                ForecastResolutionPlanStep(
                    stepId="сократить-кампанию",
                    area="marketing",
                    operation="сократить_кампанию",
                    instruction="Сократить кампанию 12345678.",
                    targetEntityType="campaign",
                    targetEntityId="12345678",
                )
            ],
        )

        critic_result, _ = run_forecast_critic(decision_result=decision)

        self.assertTrue(
            any(issue.code.startswith("incomplete_action_contract:") for issue in critic_result.issues)
        )
        self.assertFalse(critic_result.passed)

    def test_forecast_critic_returns_replan_for_missing_target_binding(self):
        decision = ForecastDecisionResult(
            hypothesis="test",
            domain="marketing",
            baseline=ForecastScenario(name="baseline"),
            actionScenario=ForecastScenario(name="action"),
            delta=ForecastDelta(),
            decision="test",
            decisionStatus="needs_review",
            confidence="medium",
            confidencePercent=64,
            blockedBy=["marketing_campaign_not_found:46227"],
            finalRecommendation="Не останавливать кампании автоматически; сначала подтвердить слабость по spend и orders.",
            summary="Не останавливать кампании автоматически; сначала подтвердить слабость по spend и orders.",
            plan=ForecastPlan(
                primary_domain="marketing",
                action_family="stop_or_reduce_campaigns",
                target_mode="campaign",
            ),
            resolvedTargets=[],
            resolutionPlan=[
                ForecastResolutionPlanStep(
                    stepId="шаг-1",
                    area="decision",
                    operation="выполнить_рекомендацию",
                    instruction="Не останавливать кампании автоматически; сначала подтвердить слабость по spend и orders.",
                )
            ],
        )
        critic_result, _ = run_forecast_critic(decision_result=decision)
        self.assertFalse(critic_result.passed)
        self.assertEqual(critic_result.verdict, "replan")
        self.assertEqual(critic_result.retryScope, "full_replan")
        self.assertTrue(any(issue.code == "campaign_fact_missing" for issue in critic_result.issues))

    def test_forecast_critic_returns_review_blocked_after_final_retry(self):
        decision = ForecastDecisionResult(
            hypothesis="test",
            domain="marketing",
            baseline=ForecastScenario(name="baseline"),
            actionScenario=ForecastScenario(name="action"),
            delta=ForecastDelta(),
            decision="test",
            decisionStatus="needs_review",
            confidence="medium",
            confidencePercent=64,
            finalRecommendation="Тестово расширять рекламу.",
            summary="Тестово расширять рекламу.",
            resolutionPlan=[
                ForecastResolutionPlanStep(
                    stepId="шаг-1",
                    area="guardrail",
                    operation="удержать_масштабирование",
                    instruction="Тестово расширять рекламу только если покрытие остаётся не ниже 38.2 дня.",
                )
            ],
        )
        critic_result, _ = run_forecast_critic(
            decision_result=decision,
            retry_attempt=2,
            max_retries=2,
        )
        self.assertEqual(critic_result.verdict, "review_blocked")
        self.assertEqual(critic_result.retryScope, "none")
        self.assertFalse(critic_result.uiSafe)

    def test_forecast_critic_does_not_review_block_executable_plan_for_text_only_issues(self):
        decision = ForecastDecisionResult(
            hypothesis="test",
            domain="marketing",
            baseline=ForecastScenario(name="baseline"),
            actionScenario=ForecastScenario(name="action"),
            delta=ForecastDelta(),
            decision="test",
            decisionStatus="completed",
            confidence="high",
            confidencePercent=82,
            finalRecommendation="Нужно почистить advert waste по этой кампании.",
            summary="Нужно почистить advert waste по этой кампании.",
            plan=ForecastPlan(
                primary_domain="marketing",
                action_family="normquery_bid_optimization",
                target_mode="sku",
            ),
            resolutionPlan=[
                ForecastResolutionPlanStep(
                    stepId="отключить-1",
                    area="traffic",
                    operation="отключить_запрос",
                    instruction="В кампании 16255557 отключить запрос `сыворотка с лифтинг эффектом`.",
                    advertId="16255557",
                    normQuery="сыворотка с лифтинг эффектом",
                    targetEntityId="46227",
                )
            ],
        )
        critic_result, _ = run_forecast_critic(
            decision_result=decision,
            retry_attempt=2,
            max_retries=2,
        )
        self.assertEqual(critic_result.verdict, "revise_reasoning")
        self.assertEqual(critic_result.retryScope, "none")
        self.assertTrue(critic_result.planExecutable)
        self.assertFalse(critic_result.uiSafe)

    def test_forecast_critic_ignores_campaign_gap_when_normquery_plan_is_executable(self):
        decision = ForecastDecisionResult(
            hypothesis="test",
            domain="marketing",
            baseline=ForecastScenario(name="baseline"),
            actionScenario=ForecastScenario(name="action"),
            delta=ForecastDelta(),
            decision="test",
            decisionStatus="completed",
            confidence="high",
            confidencePercent=82,
            blockedBy=["marketing_campaign_not_found:46227"],
            finalRecommendation="Отключить waste-запросы по SKU 46227.",
            summary="Отключить waste-запросы по SKU 46227.",
            plan=ForecastPlan(
                primary_domain="marketing",
                action_family="normquery_bid_optimization",
                target_mode="sku",
            ),
            resolutionPlan=[
                ForecastResolutionPlanStep(
                    stepId="отключить-1",
                    area="traffic",
                    operation="отключить_запрос",
                    instruction="В кампании 16255557 отключить запрос `сыворотка с лифтинг эффектом`.",
                    advertId="16255557",
                    normQuery="сыворотка с лифтинг эффектом",
                    targetEntityId="46227",
                )
            ],
        )
        critic_result, _ = run_forecast_critic(decision_result=decision)
        self.assertFalse(any(issue.code == "campaign_fact_missing" for issue in critic_result.issues))
        self.assertFalse(any(issue.code == "wrong_target_mode" for issue in critic_result.issues))
        self.assertFalse(any(issue.code == "wrong_action_family" for issue in critic_result.issues))

    def test_build_forecast_revision_input_is_stable_for_second_retry(self):
        base = ForecastReasoningInput(
            hypothesis="test",
            plan={"action_family": "weak_sku_demand_recovery"},
        )
        decision = ForecastDecisionResult(
            hypothesis="test",
            domain="marketing",
            baseline=ForecastScenario(name="baseline"),
            actionScenario=ForecastScenario(name="action"),
            delta=ForecastDelta(),
            decision="test",
            decisionStatus="needs_review",
            confidence="medium",
            finalRecommendation="Тестово расширять рекламу.",
            summary="Тестово расширять рекламу.",
        )
        critic_result, _ = run_forecast_critic(decision_result=decision)
        first = build_forecast_revision_input(
            reasoning_input=base,
            previous_decision=decision,
            critic_result=critic_result,
            retry_attempt=1,
        )
        second = build_forecast_revision_input(
            reasoning_input=first,
            previous_decision=decision,
            critic_result=critic_result,
            retry_attempt=2,
        )
        self.assertEqual(second.retry_attempt, 2)
        self.assertTrue(second.previous_decision)

    def test_revision_reasoning_payload_drops_repeated_evidence_and_debug(self):
        source_facts = {"price": {"history": ["x" * 500] * 8}}
        repeated_opportunity = {
            "opportunity_id": "price:test",
            "action_family": "price_positioning_guardrail",
            "source_facts": source_facts,
        }
        base = ForecastReasoningInput(
            hypothesis="test",
            plan={"action_family": "price_positioning_guardrail"},
            opportunity=repeated_opportunity,
            per_target_evidence=[
                {
                    "target": "46227",
                    "facts": {
                        "opportunity": repeated_opportunity,
                        "opportunity_source_facts": source_facts,
                        "entity_matches": [{"blob": "x" * 5_000}],
                        "query_rows": [{"blob": "x" * 5_000}],
                        "price_row": {"current_price": 1500},
                    },
                    "scenario": {
                        "opportunity": repeated_opportunity,
                        "source_facts": source_facts,
                        "query_rows": [{"blob": "x" * 5_000}],
                        "recommended_price": 1400,
                    },
                }
            ],
            evidence_summary={"diagnostic": {"opportunity": repeated_opportunity}},
            calculator_output={
                "per_target_evidence": [
                    {"target": "46227", "opportunity": repeated_opportunity}
                ],
                "decision": "Снизить цену до 1400 ₽.",
            },
        )
        decision = ForecastDecisionResult(
            hypothesis="test",
            domain="price",
            baseline=ForecastScenario(name="baseline"),
            actionScenario=ForecastScenario(name="action"),
            delta=ForecastDelta(),
            decision="Снизить цену до 1400 ₽.",
            confidence="medium",
            finalRecommendation="Снизить цену до 1400 ₽.",
            summary="Снизить цену до 1400 ₽.",
            debugData={"evidence_bundle": {"blob": "x" * 20_000}},
            evidence={"blob": "x" * 20_000},
            forecastDetails={"perTargetEvidence": [{"blob": "x" * 20_000}]},
        )
        critic_result = ForecastCriticResult(
            passed=False,
            verdict="revise_reasoning",
            retryScope="reasoning",
            revisionBrief=["Уточнить порог остановки."],
        )

        revision = build_forecast_revision_input(
            reasoning_input=base,
            previous_decision=decision,
            critic_result=critic_result,
            retry_attempt=1,
        )
        payload = build_forecast_reasoning_payload(revision)

        self.assertNotIn("debugData", revision.previous_decision)
        self.assertNotIn("evidence", revision.previous_decision)
        self.assertNotIn("forecastDetails", revision.previous_decision)
        self.assertEqual(payload["opportunity"]["source_facts"], source_facts)
        self.assertNotIn("opportunity", payload["per_target_evidence"][0]["facts"])
        self.assertNotIn(
            "opportunity_source_facts",
            payload["per_target_evidence"][0]["facts"],
        )
        self.assertNotIn("entity_matches", payload["per_target_evidence"][0]["facts"])
        self.assertNotIn("query_rows", payload["per_target_evidence"][0]["facts"])
        self.assertNotIn("query_rows", payload["per_target_evidence"][0]["scenario"])
        self.assertNotIn("per_target_evidence", payload["calculator_output"])
        self.assertLess(
            len(json.dumps(payload, ensure_ascii=False)),
            len(base.model_dump_json()) // 2,
        )

    def test_forecast_critic_payload_excludes_internal_trace_trees(self):
        decision = ForecastDecisionResult(
            hypothesis="test",
            domain="marketing",
            baseline=ForecastScenario(name="baseline"),
            actionScenario=ForecastScenario(name="action"),
            delta=ForecastDelta(),
            decision="Отключить запрос.",
            confidence="high",
            finalRecommendation="Отключить запрос в кампании 123.",
            summary="Отключить запрос в кампании 123.",
            resolutionPlan=[
                ForecastResolutionPlanStep(
                    stepId="stop-query",
                    area="traffic",
                    operation="отключить_запрос",
                    instruction="В кампании 123 отключить запрос `пример`.",
                )
            ],
            debugData={"blob": "x" * 20_000},
            evidence={"blob": "x" * 20_000},
            forecastDetails={"blob": "x" * 20_000},
        )

        payload = build_forecast_critic_payload(decision)

        self.assertNotIn("debugData", payload)
        self.assertNotIn("evidence", payload)
        self.assertNotIn("forecastDetails", payload)
        self.assertIn("resolutionPlan", payload)
        self.assertIn("finalRecommendation", payload)
        self.assertLess(len(json.dumps(payload, ensure_ascii=False)), 5_000)

    def test_forecast_critic_agent_instructions_accept_context_and_agent(self):
        agent = build_forecast_critic_agent()
        rendered = agent.instructions(SimpleNamespace(context=None), None)
        self.assertIsInstance(rendered, str)
        self.assertTrue(rendered.strip())
        self.assertIsNone(agent.output_type)
        self.assertEqual(agent.output_guardrails, [])

    def test_forecast_reasoning_uses_plain_json_with_local_validation(self):
        agent = build_forecast_reasoning_agent()
        rendered = agent.instructions(SimpleNamespace(context=None), None)
        self.assertIn("валидный JSON-объект", rendered)
        self.assertIsNone(agent.output_type)
        self.assertEqual(agent.output_guardrails, [])

        revision_agent = build_forecast_reasoning_agent(allow_tools=False)
        self.assertEqual(revision_agent.tools, [])
        self.assertEqual(revision_agent.model_settings.tool_choice, "none")

    def test_task_normalizer_routes_availability_before_scale_to_supply(self):
        plan = build_forecast_plan(_availability_before_scale_task())

        self.assertIsNotNone(plan)
        self.assertEqual(plan.primary_domain, "supply")
        self.assertEqual(plan.action_family, "availability_before_scale_guardrail")
        self.assertIn("marketing", plan.secondary_domains)

    @patch("agi_ecom_app.forecast.calculators.action_family.strategic_db.get_marketing_drr_report")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_marketing_action_matrix")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_marketing_campaign_efficiency_snapshot")
    def test_marketing_workflow_returns_human_ui_card(
        self,
        mock_efficiency,
        mock_action_matrix,
        mock_drr,
    ):
        mock_efficiency.return_value = {
            "campaign_rows": [
                {"advert_id": 309891193, "status": "inefficient", "sum_spend": 2013.75, "orders": 0},
                {"advert_id": 295603220, "status": "working", "sum_spend": 3043.78, "orders": 1},
            ]
        }
        mock_action_matrix.return_value = {
            "campaign_actions": [
                {"entity_id": 309891193, "status": "остановить"},
                {"entity_id": 295603220, "status": "оставить"},
            ]
        }
        mock_drr.return_value = {"totals": {"estimated_drr_promoted_sku_pct": 12.4}}

        result = run_forecast_workflow(task_snapshot=_marketing_task(), today=date(2026, 4, 24))

        self.assertIsInstance(result, ForecastDecisionResult)
        self.assertEqual(result.plan.action_family, "stop_or_reduce_campaigns")
        self.assertIsNotNone(result.uiCard)
        self.assertIn("рекламные кампании", result.uiCard.whatChecking.lower())
        self.assertIn("309891193", result.uiCard.recommendation)
        self.assertNotIn("orders_at_risk", render_forecast_decision_result(result))
        self.assertTrue(any(step.operation in {"сократить_кампанию", "заморозить_расширение_кампании"} for step in result.resolutionPlan))

    @patch("agi_ecom_app.forecast.workflow.build_forecast_evidence_bundle", return_value=ForecastEvidenceBundle())
    @patch("agi_ecom_app.forecast.workflow.calculate_forecast_scenario")
    def test_workflow_revises_reasoning_after_vague_critic_reject(
        self,
        mock_calculate,
        _mock_bundle,
    ):
        mock_calculate.return_value = {
            "decision": "Тестово расширять рекламу.",
            "decision_status": "completed",
            "confidence": "high",
            "summary": "Тестово расширять рекламу.",
            "final_recommendation": "Тестово расширять рекламу.",
            "per_target_evidence": [
                {"target": "295603220"},
                {"target": "309891193"},
            ],
        }

        def _revision_side_effect(*, reasoning_input, plan, today=None):
            if isinstance(reasoning_input, ForecastRevisionInput):
                result = ForecastReasoningResult(
                    decision_status="completed",
                    confidence="high",
                    decision="В кампании 16255557 отключить нецелевые запросы и не расширять её до повторной сверки покрытия.",
                    final_recommendation="В кампании 16255557 отключить нецелевые запросы и не расширять её до повторной сверки покрытия.",
                    summary="В кампании 16255557 отключить нецелевые запросы и не расширять её до повторной сверки покрытия.",
                    decision_why="Критик потребовал убрать vague wording и привязать шаг к объекту.",
                    primary_action=plan.action_family,
                    evidence_used=["revision"],
                    resolution_plan_patch=[
                        ForecastResolutionPlanStep(
                            stepId="отключить-1",
                            area="traffic",
                            operation="отключить_запрос",
                            instruction="В кампании 16255557 отключить запрос `сыворотка с лифтинг эффектом`.",
                            advertId="16255557",
                            normQuery="сыворотка с лифтинг эффектом",
                            currentMetric="расход за 7 дней=120 ₽; заказы=0",
                            targetMetric="расход за 7 дней -> 0",
                            stopCondition="не возвращать запрос, пока заказов за 7 дней = 0",
                            payload={"advertId": "16255557", "normQuery": "сыворотка с лифтинг эффектом"},
                        )
                    ],
                )
                return result, {"mode": "revision-mock"}
            result = ForecastReasoningResult(
                decision_status="completed",
                confidence="high",
                decision="Тестово расширять рекламу.",
                final_recommendation="Тестово расширять рекламу.",
                summary="Тестово расширять рекламу.",
                decision_why="Первичный ответ ещё абстрактный.",
                primary_action=plan.action_family,
                evidence_used=["initial"],
                resolution_plan_patch=[
                    ForecastResolutionPlanStep(
                        stepId="шаг-1",
                        area="guardrail",
                        operation="удержать_масштабирование",
                        instruction="Тестово расширять рекламу только если покрытие остаётся не ниже 38.2 дня.",
                    )
                ],
            )
            return result, {"mode": "initial-mock"}

        with patch("agi_ecom_app.forecast.workflow.run_forecast_reasoning", side_effect=_revision_side_effect):
            result = run_forecast_workflow(task_snapshot=_marketing_task(), today=date(2026, 4, 24))

        self.assertEqual(result.decisionStatus, "completed")
        self.assertFalse(result.reviewBlocked)
        self.assertEqual(result.iterationCount, 2)
        self.assertTrue(any(step.operation == "отключить_запрос" for step in result.resolutionPlan))

    @patch("agi_ecom_app.forecast.planner.route_task_with_llm")
    @patch("agi_ecom_app.forecast.workflow.build_forecast_evidence_bundle", return_value=ForecastEvidenceBundle())
    @patch("agi_ecom_app.forecast.workflow.calculate_forecast_scenario")
    def test_workflow_replans_after_campaign_fact_missing(
        self,
        mock_calculate,
        _mock_bundle,
        mock_route,
    ):
        mock_route.return_value = (
            ForecastRouteProposal(
                route=True,
                action_family="stop_or_reduce_campaigns",
                primary_domain="marketing",
                target_mode="campaign",
                confidence="high",
                rationale="Initial ambiguous spend-growth route starts from campaign reduction.",
            ),
            {"mode": "mock"},
        )

        def _scenario(*, plan, resolved_targets, evidence_bundle, today=None):
            if plan.action_family == "stop_or_reduce_campaigns":
                return {
                    "decision": "Не останавливать кампании автоматически; сначала подтвердить слабость по spend и orders.",
                    "decision_status": "needs_review",
                    "confidence": "medium",
                    "summary": "Не останавливать кампании автоматически; сначала подтвердить слабость по spend и orders.",
                    "final_recommendation": "Не останавливать кампании автоматически; сначала подтвердить слабость по spend и orders.",
                    "blocked_by": ["marketing_campaign_not_found:46227"],
                    "per_target_evidence": [{"target": "46227"}],
                    "resolution_plan": [
                        {
                            "stepId": "шаг-1",
                            "area": "decision",
                            "operation": "выполнить_рекомендацию",
                            "instruction": "Не останавливать кампании автоматически; сначала подтвердить слабость по spend и orders.",
                        }
                    ],
                }
            return {
                "decision": "В кампании 16255557 отключить нецелевые запросы по SKU 46227 и не расширять ядро до повторной сверки метрик.",
                "decision_status": "completed",
                "confidence": "high",
                "summary": "В кампании 16255557 отключить нецелевые запросы по SKU 46227 и не расширять ядро до повторной сверки метрик.",
                "final_recommendation": "В кампании 16255557 отключить нецелевые запросы по SKU 46227 и не расширять ядро до повторной сверки метрик.",
                "per_target_evidence": [{"target": "46227"}],
                "resolution_plan": [
                    {
                        "stepId": "отключить-1",
                        "area": "traffic",
                        "operation": "отключить_запрос",
                        "instruction": "В кампании 16255557 отключить запрос `сыворотка с лифтинг эффектом`.",
                        "advertId": "16255557",
                        "normQuery": "сыворотка с лифтинг эффектом",
                        "targetEntityId": "46227",
                        "currentMetric": "расход за 7 дней=120 ₽; заказы=0",
                        "targetMetric": "расход за 7 дней -> 0",
                        "stopCondition": "не возвращать запрос, пока заказов за 7 дней = 0",
                        "payload": {"advertId": "16255557", "normQuery": "сыворотка с лифтинг эффектом"},
                    }
                ],
            }

        mock_calculate.side_effect = _scenario
        task = _marketing_task(
            title="Проверить кампании с ростом расхода: 46227",
            action="Проверить кампании с ростом расхода: 46227",
            targets=["46227"],
        )

        result = run_forecast_workflow(task_snapshot=task, today=date(2026, 4, 24))

        self.assertEqual(result.decisionStatus, "completed")
        self.assertFalse(result.reviewBlocked)
        self.assertEqual(result.plan.action_family, "normquery_bid_optimization")
        self.assertEqual(result.iterationCount, 2)

    @patch("agi_ecom_app.forecast.planner.route_task_with_llm")
    @patch("agi_ecom_app.forecast.workflow.build_forecast_evidence_bundle", return_value=ForecastEvidenceBundle())
    @patch("agi_ecom_app.forecast.workflow.calculate_forecast_scenario")
    def test_workflow_returns_review_blocked_after_two_failed_retries(
        self,
        mock_calculate,
        _mock_bundle,
        mock_route,
    ):
        mock_route.return_value = (
            ForecastRouteProposal(
                route=True,
                action_family="stop_or_reduce_campaigns",
                primary_domain="marketing",
                target_mode="campaign",
                confidence="high",
                rationale="Initial ambiguous spend-growth route starts from campaign reduction.",
            ),
            {"mode": "mock"},
        )
        mock_calculate.return_value = {
            "decision": "Не останавливать кампании автоматически; сначала подтвердить слабость по spend и orders.",
            "decision_status": "needs_review",
            "confidence": "medium",
            "summary": "Не останавливать кампании автоматически; сначала подтвердить слабость по spend и orders.",
            "final_recommendation": "Не останавливать кампании автоматически; сначала подтвердить слабость по spend и orders.",
            "blocked_by": ["marketing_campaign_not_found:46227"],
            "resolution_plan": [
                {
                    "stepId": "шаг-1",
                    "area": "decision",
                    "operation": "выполнить_рекомендацию",
                    "instruction": "Не останавливать кампании автоматически; сначала подтвердить слабость по spend и orders.",
                }
            ],
        }
        task = _marketing_task(
            title="Проверить кампании с ростом расхода: 46227",
            action="Проверить кампании с ростом расхода: 46227",
            targets=["46227"],
        )

        result = run_forecast_workflow(task_snapshot=task, today=date(2026, 4, 24))

        self.assertTrue(result.reviewBlocked)
        self.assertEqual(result.decisionStatus, "blocked")
        self.assertEqual(result.resolutionPlan, [])
        self.assertGreaterEqual(result.iterationCount, 2)

    @patch("agi_ecom_app.forecast.workflow.build_forecast_evidence_bundle", return_value=ForecastEvidenceBundle())
    @patch("agi_ecom_app.forecast.workflow.calculate_forecast_scenario")
    def test_workflow_publishes_text_safe_fallback_for_executable_deterministic_plan(
        self,
        mock_calculate,
        _mock_bundle,
    ):
        mock_calculate.return_value = {
            "decision": "Нужно почистить advert waste по этой кампании.",
            "decision_status": "completed",
            "confidence": "high",
            "summary": "Нужно почистить advert waste по этой кампании.",
            "final_recommendation": "Нужно почистить advert waste по этой кампании.",
            "resolution_plan": [
                {
                    "stepId": "отключить-1",
                    "area": "traffic",
                    "operation": "отключить_запрос",
                    "instruction": "В кампании 16255557 отключить запрос `сыворотка с лифтинг эффектом`.",
                    "advertId": "16255557",
                    "normQuery": "сыворотка с лифтинг эффектом",
                    "targetEntityId": "46227",
                }
            ],
        }
        task = {
            "actionKind": "optimize_marketing_visibility",
            "decisionType": "marketing_visibility_guardrail",
            "title": "Проверить видимость и ставки: 46227",
            "action": "Проверить видимость и ставки: 46227",
            "targets": ["46227"],
            "businessGoal": "Понять, какие рекламные или поисковые действия можно менять без потери эффективности.",
            "forecastSpec": {
                "horizon_days": 14,
                "period_days": 7,
                "forecast_capable": True,
                "forecast_domain": "marketing",
                "action_family": "normquery_bid_optimization",
            },
        }

        result = run_forecast_workflow(task_snapshot=task, today=date(2026, 4, 24))

        self.assertEqual(result.decisionStatus, "completed")
        self.assertFalse(result.reviewBlocked)
        self.assertTrue(result.forecastDetails.get("publishedWithTextFallback"))
        self.assertTrue(result.resolutionPlan)
        self.assertIn("В кампании 16255557 отключить запрос", result.summary)

    @patch("agi_ecom_app.forecast.workflow.run_forecast_critic")
    @patch("agi_ecom_app.forecast.workflow.build_forecast_evidence_bundle", return_value=ForecastEvidenceBundle())
    @patch("agi_ecom_app.forecast.workflow.calculate_forecast_scenario")
    def test_workflow_keeps_canonical_normquery_plan_even_if_critic_returns_review_blocked(
        self,
        mock_calculate,
        _mock_bundle,
        mock_run_critic,
    ):
        mock_calculate.return_value = {
            "decision": "Отключить waste-запросы и зафиксировать параметры кампании.",
            "decision_status": "completed",
            "confidence": "high",
            "summary": "Отключить waste-запросы и зафиксировать параметры кампании.",
            "final_recommendation": "Отключить waste-запросы и зафиксировать параметры кампании.",
            "resolution_plan": [
                {
                    "stepId": "отключить-1",
                    "area": "traffic",
                    "operation": "отключить_запрос",
                    "instruction": "В кампании 16255557 отключить запрос `сыворотка с лифтинг эффектом`.",
                    "advertId": "16255557",
                    "normQuery": "сыворотка с лифтинг эффектом",
                    "targetEntityId": "сыворотка с лифтинг эффектом",
                    "targetEntityType": "query",
                },
                {
                    "stepId": "guardrail-1",
                    "area": "guardrail",
                    "operation": "зафиксировать_ставки_и_ключи",
                    "instruction": "В кампании 16255557 на 7 дней оставить текущие ставки без изменений и не добавлять новые ключи по SKU 46227; повторно открывать расширение только если средняя позиция не хуже 10, а видимость не ниже 85.",
                    "advertId": "16255557",
                    "targetEntityId": "16255557",
                    "targetEntityType": "campaign",
                    "targetMetric": "средняя позиция <= 10; видимость >= 85",
                    "stopCondition": "через 7 дней пересмотреть кампанию; расширение разрешать только если порог по видимости и позиции выполнен",
                },
            ],
        }
        mock_run_critic.return_value = (
            ForecastCriticResult(
                passed=False,
                verdict="review_blocked",
                retryScope="none",
                planExecutable=True,
                uiSafe=False,
                issues=[
                    ForecastCriticIssue(
                        path="summary",
                        code="technical_user_facing_text",
                        severity="medium",
                        message="Текст можно упростить.",
                    )
                ],
                criticSummary="Текст для UI нужно упростить.",
            ),
            {"mode": "forced-review-block"},
        )
        task = {
            "actionKind": "optimize_marketing_visibility",
            "decisionType": "marketing_visibility_guardrail",
            "title": "Проверить видимость и ставки: 46227",
            "action": "Проверить видимость и ставки: 46227",
            "targets": ["46227"],
            "businessGoal": "Понять, какие рекламные или поисковые действия можно менять без потери эффективности.",
            "forecastSpec": {
                "horizon_days": 14,
                "period_days": 7,
                "forecast_capable": True,
                "forecast_domain": "marketing",
                "action_family": "normquery_bid_optimization",
            },
        }

        result = run_forecast_workflow(task_snapshot=task, today=date(2026, 4, 24))

        self.assertEqual(result.decisionStatus, "completed")
        self.assertFalse(result.reviewBlocked)
        self.assertTrue(result.forecastDetails.get("publishedWithTextFallback"))
        self.assertEqual(len(result.resolutionPlan), 2)

    @patch("agi_ecom_app.forecast.workflow.build_forecast_evidence_bundle", return_value=ForecastEvidenceBundle())
    @patch("agi_ecom_app.forecast.workflow.calculate_forecast_scenario")
    def test_workflow_prefers_deterministic_summary_over_reasoning_for_canonical_normquery(
        self,
        mock_calculate,
        _mock_bundle,
    ):
        mock_calculate.return_value = {
            "decision": "В кампании 16255557 отключить waste-запросы.",
            "decision_status": "completed",
            "confidence": "high",
            "summary": "В кампании 16255557 отключить waste-запросы.",
            "final_recommendation": "В кампании 16255557 отключить waste-запросы.",
            "resolution_plan": [
                {
                    "stepId": "отключить-1",
                    "area": "traffic",
                    "operation": "отключить_запрос",
                    "instruction": "В кампании 16255557 отключить запрос `сыворотка с лифтинг эффектом`.",
                    "advertId": "16255557",
                    "normQuery": "сыворотка с лифтинг эффектом",
                    "targetEntityId": "46227",
                }
            ],
        }

        def _bad_reasoning(*, reasoning_input, plan, today=None):
            return (
                ForecastReasoningResult(
                    decision_status="completed",
                    confidence="high",
                    decision="Сократить или остановить слабые кампании.",
                    final_recommendation="Сократить или остановить слабые кампании.",
                    summary="Сократить или остановить слабые кампании.",
                    decision_why="Reasoning перепридумал route.",
                    primary_action="stop_or_reduce_campaigns",
                ),
                {"mode": "bad-reasoning"},
            )

        task = {
            "actionKind": "optimize_marketing_visibility",
            "decisionType": "marketing_visibility_guardrail",
            "title": "Проверить видимость и ставки: 46227",
            "action": "Проверить видимость и ставки: 46227",
            "targets": ["46227"],
            "businessGoal": "Понять, какие рекламные или поисковые действия можно менять без потери эффективности.",
            "forecastSpec": {
                "horizon_days": 14,
                "period_days": 7,
                "forecast_capable": True,
                "forecast_domain": "marketing",
                "action_family": "normquery_bid_optimization",
            },
        }

        with patch("agi_ecom_app.forecast.workflow.run_forecast_reasoning", side_effect=_bad_reasoning):
            result = run_forecast_workflow(task_snapshot=task, today=date(2026, 4, 24))

        self.assertEqual(result.plan.action_family, "normquery_bid_optimization")
        self.assertNotEqual(result.summary, "Сократить или остановить слабые кампании.")
        self.assertIn("В кампании 16255557 отключить", result.summary)

    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_price_decision_overview")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_price_competitor_overview")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_finance_overview")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_supply_overview")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_marketing_search_demand_snapshot")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_marketing_visibility_snapshot")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_sales_overview")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_sales_sku_revenue_change")
    def test_weak_sku_workflow_returns_recovery_card_instead_of_zero_campaign_metrics(
        self,
        mock_sales_change,
        mock_sales_overview,
        mock_visibility,
        mock_demand,
        mock_supply,
        mock_finance,
        mock_price,
        mock_price_decision,
    ):
        mock_price_decision.return_value = {"rows": []}
        mock_sales_change.return_value = {
            "rows": [
                {
                    "nm_id": 46303,
                    "vendor_code": "46303",
                    "current_sales": 3,
                    "previous_sales": 10,
                    "current_revenue": 9000,
                    "previous_revenue": 24000,
                }
            ]
        }
        mock_sales_overview.return_value = {
            "conversion_watchlist": [
                {
                    "nm_id": 46303,
                    "vendor_code": "46303",
                    "open_count": 300,
                    "cart_count": 15,
                    "order_count": 2,
                    "cart_to_order_conversion": 0.05,
                }
            ],
            "top_revenue_rows": [],
        }
        mock_visibility.return_value = {
            "visibility_rows": [
                {
                    "nm_id": 46303,
                    "vendor_code": "46303",
                    "avg_position_dynamics": -7,
                    "visibility_current": 22,
                    "open_card_dynamics": -12,
                }
            ],
            "position_loss_rows": [{"nm_id": 46303}],
            "open_card_gap_rows": [{"nm_id": 46303}],
        }
        mock_demand.return_value = {
            "high_frequency_low_order_rows": [
                {"nm_id": 46303, "frequency_sum": 120, "orders_count": 1}
            ],
            "order_driving_query_rows": [],
        }
        mock_supply.return_value = {
            "rows": [
                {
                    "nm_id": 46303,
                    "vendor_code": "46303",
                    "stock_qty_full": 40,
                    "in_way_to_client": 0,
                    "in_way_from_client": 0,
                    "sales_7d_units": 21,
                    "days_of_stock": 13,
                    "reorder_flag": "warning",
                    "reorder_qty_recommended": 40,
                }
            ]
        }
        mock_price.return_value = {
            "rows": [
                {
                    "nm_id": 46303,
                    "vendor_code": "46303",
                    "source_price_promo": 1600,
                    "competitor_price_median": 1300,
                    "source_vs_median_gap_rub": 300,
                    "source_vs_median_gap_pct": 23,
                    "competitor_count": 9,
                    "price_status": "above_market",
                }
            ]
        }
        mock_finance.return_value = {
            "unit_economics_watchlist": [
                {
                    "nm_id": 46303,
                    "vendor_code": "46303",
                    "payout_sum": 12000,
                    "retail_amount_sum": 30000,
                    "payout_margin_before_cogs_pct": 40,
                }
            ],
            "margin_watchlist": [],
            "fee_pressure_rows": [],
        }

        result = run_forecast_workflow(task_snapshot=_weak_sku_task())

        self.assertIsInstance(result, ForecastDecisionResult)
        self.assertEqual(result.plan.action_family, "weak_sku_demand_recovery")
        self.assertIsNotNone(result.uiCard)
        self.assertIn("слабые sku", result.uiCard.whatChecking.lower())
        self.assertNotIn("кампани", result.uiCard.recommendation.lower())
        self.assertNotIn("проверить", result.summary.lower())
        self.assertTrue(result.summary.startswith(("Сократить", "Дозаказать", "Снизить", "Перезапустить")))
        self.assertTrue(
            any(token in result.summary.lower() for token in ("закрыть", "восстановить", "снизить", "перезапустить", "дозаказать", "сократить"))
        )
        self.assertIn("численн", result.uiCard.expectedEffect.lower())
        self.assertGreaterEqual(len(result.resolutionActions), 2)
        self.assertTrue(any(item.area in {"waste_traffic", "supply", "price"} for item in result.resolutionActions))
        self.assertGreaterEqual(len(result.resolutionPlan), 1)
        self.assertNotIn("ad_spend_saved_rub", render_forecast_decision_result(result))
        self.assertGreater(len(result.forecastDetails.get("perTargetEvidence") or []), 0)
        self.assertIsNone(result.profitForecastRub)
        self.assertIsNone(result.economicEffectRub)

    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_marketing_action_matrix")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_marketing_campaign_daily_mart")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_marketing_forecast_ready_mart")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_marketing_query_bands_mart")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_marketing_search_demand_snapshot")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_marketing_visibility_snapshot")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_price_decision_overview")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_price_competitor_overview")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_finance_overview")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_supply_overview")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_sales_overview")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_sales_sku_revenue_change")
    def test_weak_sku_card_driver_builds_execution_plan(
        self,
        mock_sales_change,
        mock_sales_overview,
        mock_supply,
        mock_finance,
        mock_price,
        mock_price_decision,
        mock_visibility,
        mock_demand,
        mock_query_bands,
        mock_forecast_ready,
        mock_campaign_daily,
        mock_action_matrix,
    ):
        mock_sales_change.return_value = {
            "rows": [
                {
                    "nm_id": 46311,
                    "vendor_code": "46311",
                    "current_sales": 1,
                    "previous_sales": 7,
                    "current_revenue": 1875,
                    "previous_revenue": 13394,
                }
            ]
        }
        mock_sales_overview.return_value = {"conversion_watchlist": [], "top_revenue_rows": []}
        mock_supply.return_value = {
            "rows": [
                {
                    "nm_id": 46311,
                    "vendor_code": "46311",
                    "stock_qty_full": 100,
                    "sales_7d_units": 4,
                    "days_of_stock": 203,
                    "reorder_flag": "ok",
                    "reorder_qty_recommended": 0,
                }
            ]
        }
        mock_finance.return_value = {
            "unit_economics_watchlist": [
                {
                    "nm_id": 46311,
                    "vendor_code": "46311",
                    "payout_sum": 44194,
                    "retail_amount_sum": 40292,
                    "payout_margin_before_cogs_pct": 35,
                }
            ],
            "margin_watchlist": [],
            "fee_pressure_rows": [],
        }
        mock_price_decision.return_value = {"rows": []}
        mock_price.return_value = {"rows": []}
        mock_visibility.return_value = {
            "visibility_rows": [
                {
                    "nm_id": 46311,
                    "vendor_code": "46311",
                    "name": "Увлажняющая спрей-сыворотка для лица с витаминами",
                    "avg_position_current": 12,
                    "visibility_current": 79,
                    "open_card_current": 83,
                    "add_to_cart_current": 14,
                    "cart_to_order_current": 14,
                }
            ],
            "position_loss_rows": [{"nm_id": 46311}],
            "open_card_gap_rows": [{"nm_id": 46311}],
        }
        mock_demand.return_value = {
            "order_driving_query_rows": [],
            "high_frequency_low_order_rows": [],
            "growing_query_rows": [],
        }
        mock_query_bands.return_value = {
            "rows": [
                {
                    "advert_id": 26018961,
                    "campaign_name": "295603220 МПГ",
                    "payment_type": "cpm",
                    "vendor_code": "46311",
                    "norm_query": "спрей для лица увлажняющий с витаминами",
                    "query_band": "waste",
                    "spend_7d": 34.77,
                    "views_7d": 57,
                    "clicks_7d": 0,
                    "orders_7d": 0,
                    "avg_pos_7d": 12.17,
                    "effective_cpm_7d": 610.0,
                },
                {
                    "advert_id": 26018961,
                    "campaign_name": "295603220 МПГ",
                    "payment_type": "cpm",
                    "vendor_code": "46311",
                    "norm_query": "увлажняющий спрей для лица",
                    "query_band": "traffic_only",
                    "spend_7d": 158.6,
                    "views_7d": 260,
                    "clicks_7d": 7,
                    "orders_7d": 0,
                    "avg_pos_7d": 29.06,
                    "effective_cpm_7d": 610.0,
                    "effective_cpc_7d": 22.66,
                },
            ]
        }
        mock_forecast_ready.return_value = {"rows": []}
        mock_campaign_daily.return_value = {"rows": []}
        mock_action_matrix.return_value = {
            "marketing_source_status": "ready",
            "demand_source_status": "ready",
            "campaign_actions": [],
            "query_actions": [],
            "sku_actions": [],
        }

        result = run_forecast_workflow(task_snapshot=_weak_sku_task(objects=[{"externalId": "46311", "label": "SKU 46311"}]))

        self.assertNotIn("урезать", result.summary.lower())
        self.assertNotIn("вернуть sku", result.summary.lower())
        self.assertNotIn("advert", result.summary.lower())
        self.assertNotIn("title", result.summary.lower())
        operations = {step.operation for step in result.resolutionPlan}
        self.assertIn("отключить_запрос", operations)
        self.assertIn("удерживать_запрос", operations)
        self.assertIn("переписать_заголовок", operations)
        self.assertIn("заменить_первый_экран", operations)
        self.assertTrue(any("Спрей-сыворотка для увлажнения лица с витаминами" in step.instruction for step in result.resolutionPlan))

    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_marketing_action_matrix")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_marketing_campaign_daily_mart")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_marketing_forecast_ready_mart")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_marketing_query_bands_mart")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_marketing_search_demand_snapshot")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_marketing_visibility_snapshot")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_finance_overview")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_supply_overview")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_sales_overview")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_sales_sku_revenue_change")
    def test_normquery_workflow_builds_fact_based_plan_without_marketing_action(
        self,
        mock_sales_change,
        mock_sales_overview,
        mock_supply,
        mock_finance,
        mock_visibility,
        mock_demand,
        mock_query_bands,
        mock_forecast_ready,
        mock_campaign_daily,
        mock_action_matrix,
    ):
        task = {
            "actionKind": "optimize_marketing_visibility",
            "decisionType": "marketing_visibility_guardrail",
            "title": "Проверить видимость и ставки: 46225",
            "action": "Проверить видимость и ставки: 46225",
            "targets": ["46225", "видимость"],
            "businessGoal": "Понять, какие рекламные или поисковые действия можно менять без потери эффективности.",
            "forecastSpec": {
                "horizon_days": 14,
                "period_days": 7,
                "forecast_capable": True,
                "forecast_domain": "marketing",
            },
        }
        mock_action_matrix.return_value = {
            "marketing_source_status": "ready",
            "demand_source_status": "ready",
            "campaign_actions": [],
            "query_actions": [],
            "sku_actions": [],
        }
        mock_campaign_daily.return_value = {"rows": []}
        mock_forecast_ready.return_value = {"rows": []}
        mock_query_bands.return_value = {
            "rows": [
                {
                    "advert_id": 26018961,
                    "campaign_name": "295603220 МПГ",
                    "payment_type": "cpm",
                    "vendor_code": "46225",
                    "norm_query": "патчи гидрогелевые",
                    "query_band": "waste",
                    "spend_7d": 90.0,
                    "views_7d": 200,
                    "clicks_7d": 4,
                    "orders_7d": 0,
                    "avg_pos_7d": 17,
                    "effective_cpm_7d": 560.0,
                },
                {
                    "advert_id": 26018961,
                    "campaign_name": "295603220 МПГ",
                    "payment_type": "cpm",
                    "vendor_code": "46225",
                    "norm_query": "патчи от отеков",
                    "query_band": "traffic_only",
                    "spend_7d": 150.0,
                    "views_7d": 410,
                    "clicks_7d": 9,
                    "orders_7d": 0,
                    "avg_pos_7d": 12,
                    "effective_cpm_7d": 540.0,
                },
            ]
        }
        mock_visibility.return_value = {
            "visibility_rows": [
                {
                    "nm_id": 51931559,
                    "vendor_code": "46225",
                    "name": "Патчи гидрогелевые от отеков и морщин с ботокс эффектом",
                    "avg_position_current": 11,
                    "visibility_current": 79,
                    "open_card_current": 163,
                }
            ],
            "position_loss_rows": [],
            "open_card_gap_rows": [],
        }
        mock_demand.return_value = {"order_driving_query_rows": [], "high_frequency_low_order_rows": []}
        mock_sales_change.return_value = {"rows": [{"vendor_code": "46225", "current_sales": 6, "previous_sales": 5, "current_revenue": 9269.46, "previous_revenue": 7850.13}]}
        mock_sales_overview.return_value = {"conversion_watchlist": [], "top_revenue_rows": []}
        mock_supply.return_value = {"rows": [{"vendor_code": "46225", "days_of_stock": 115.5, "reorder_flag": "ok", "reorder_qty_recommended": 0}]}
        mock_finance.return_value = {"unit_economics_watchlist": [{"vendor_code": "46225", "payout_margin_before_cogs_pct": 97.68}], "margin_watchlist": [], "fee_pressure_rows": []}

        result = run_forecast_workflow(task_snapshot=task)

        self.assertEqual(result.plan.action_family, "normquery_bid_optimization")
        self.assertNotIn("Нет рекламных фактов", result.summary)
        operations = [step.operation for step in result.resolutionPlan]
        self.assertIn("отключить_запрос", operations)
        self.assertIn("удерживать_запрос", operations)
        self.assertIn("зафиксировать_ставки_и_ключи", operations)
        self.assertIn("Экономический эффект", result.summary)
        self.assertIn("пустого рекламного расхода", result.summary)
        self.assertNotIn("Экономический эффект", result.finalRecommendation)
        self.assertIsNone(result.profitForecastRub)
        self.assertEqual(result.economicEffectRub, 90.0)
        self.assertEqual(result.economicEffectPctOfCurrentSpend, 37.5)
        guardrail_step = next(step for step in result.resolutionPlan if step.operation == "зафиксировать_ставки_и_ключи")
        self.assertEqual(guardrail_step.targetEntityType, "campaign")
        self.assertEqual(guardrail_step.advertId, "26018961")
        self.assertIn("на 7 дней", guardrail_step.instruction)

    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_marketing_search_demand_snapshot")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_marketing_visibility_snapshot")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_supply_overview")
    def test_availability_before_scale_returns_supply_guardrail(
        self,
        mock_supply,
        mock_visibility,
        mock_demand,
    ):
        mock_supply.return_value = {
            "rows": [
                {
                    "nm_id": 46310,
                    "vendor_code": "46310",
                    "stock_qty": 10,
                    "stock_qty_full": 10,
                    "supplier_inbound_qty": 0,
                    "in_way_to_client": 2,
                    "in_way_from_client": 1,
                    "sales_7d_units": 35,
                    "days_of_stock": 2,
                    "reorder_flag": "critical",
                    "reorder_qty_recommended": 270,
                }
            ]
        }
        mock_visibility.return_value = {"visibility_rows": [], "position_loss_rows": [], "open_card_gap_rows": []}
        mock_demand.return_value = {"high_frequency_low_order_rows": [], "order_driving_query_rows": []}

        result = run_forecast_workflow(
            task_snapshot=_availability_before_scale_task(targets=["46310"])
        )

        self.assertIsInstance(result, ForecastDecisionResult)
        self.assertEqual(result.domain, "supply")
        self.assertIn("остатки", result.uiCard.whatChecking.lower())
        self.assertIn("oos", result.uiCard.expectedEffect.lower())
        self.assertEqual(result.delta.metrics["reorder_qty"], 270.0)
        self.assertGreaterEqual(len(result.resolutionPlan), 1)
        self.assertTrue(any(step.operation == "дозаказать_остаток" for step in result.resolutionPlan))

    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_marketing_action_matrix")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_marketing_campaign_daily_mart")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_marketing_forecast_ready_mart")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_marketing_query_bands_mart")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_marketing_search_demand_snapshot")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_marketing_visibility_snapshot")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_supply_overview")
    def test_availability_before_scale_builds_standard_resolution_plan_for_low_oos(
        self,
        mock_supply,
        mock_visibility,
        mock_demand,
        mock_query_bands,
        mock_forecast_ready,
        mock_campaign_daily,
        mock_action_matrix,
    ):
        mock_supply.return_value = {
            "rows": [
                {
                    "nm_id": 46227,
                    "vendor_code": "46227",
                    "stock_qty": 150,
                    "stock_qty_full": 191,
                    "in_way_to_client": 34,
                    "in_way_from_client": 7,
                    "sales_7d_units": 35,
                    "days_of_stock": 38.2,
                    "reorder_flag": "ok",
                    "reorder_qty_recommended": 0,
                }
            ]
        }
        mock_visibility.return_value = {"visibility_rows": [], "position_loss_rows": [], "open_card_gap_rows": []}
        mock_demand.return_value = {"high_frequency_low_order_rows": [], "order_driving_query_rows": []}
        mock_query_bands.return_value = {
            "rows": [
                {
                    "advert_id": 16255557,
                    "campaign_name": "70194686 сыворотка",
                    "payment_type": "cpm",
                    "vendor_code": "46227",
                    "norm_query": "сыворотка с лифтинг эффектом",
                    "query_band": "waste",
                    "spend_7d": 18.7,
                    "views_7d": 14,
                    "clicks_7d": 0,
                    "orders_7d": 0,
                    "avg_pos_7d": 6.5,
                    "effective_cpm_7d": 1335.7,
                }
            ]
        }
        mock_forecast_ready.return_value = {"rows": []}
        mock_campaign_daily.return_value = {"rows": []}
        mock_action_matrix.return_value = {
            "marketing_source_status": "ready",
            "demand_source_status": "ready",
            "campaign_actions": [],
            "query_actions": [],
            "sku_actions": [],
        }

        result = run_forecast_workflow(task_snapshot=_availability_before_scale_task(targets=["46227"]))

        self.assertGreaterEqual(len(result.resolutionPlan), 2)
        operations = {step.operation for step in result.resolutionPlan}
        self.assertIn("отключить_запрос", operations)
        self.assertIn("заморозить_расширение_кампании", operations)
        critic_result, _ = run_forecast_critic(decision_result=result)
        self.assertTrue(critic_result.passed)

    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_sales_overview")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_price_decision_overview")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_price_competitor_overview")
    def test_price_workflow_returns_guardrail_and_hides_irrelevant_metrics(self, mock_price, mock_price_decision, mock_sales_overview):
        mock_sales_overview.return_value = {"conversion_watchlist": [], "top_revenue_rows": []}
        mock_price_decision.return_value = {"rows": []}
        mock_price.return_value = {
            "rows": [
                {
                    "nm_id": 46227,
                    "vendor_code": "46227",
                    "source_price_promo": 1500,
                    "competitor_price_median": 1000,
                    "source_vs_median_gap_rub": 500,
                    "source_vs_median_gap_pct": 50,
                    "competitor_count": 8,
                    "price_status": "above_market",
                }
            ]
        }

        result = run_forecast_workflow(task_snapshot=_price_task())
        rendered = render_forecast_decision_result(result)

        self.assertEqual(result.plan.action_family, "price_positioning_guardrail")
        self.assertIn("цен", result.uiCard.whatChecking.lower())
        self.assertNotIn("coverage_days", rendered)
        self.assertNotIn("current_ad_orders", rendered)
        self.assertEqual(result.forecastDetails["reasoningPrimaryAction"], "price_positioning_guardrail")
        self.assertEqual(result.forecastDetails["reasoningEvidenceUsed"], ["unit_test_reasoning_mock"])
        self.assertTrue(any(step.operation in {"изменить_цену", "удержать_цену"} for step in result.resolutionPlan))

    def test_reasoning_input_captures_blockers_and_source_status(self):
        plan = build_forecast_plan(_price_task())
        hypothesis = build_forecast_hypothesis(_price_task())
        self.assertIsNotNone(plan)
        self.assertIsNotNone(hypothesis)
        evidence_bundle = ForecastEvidenceBundle(
            diagnostic={"primary_summary": "price evidence ready"},
            workers={
                "forecast_price_agent": {
                    "confidence": "medium",
                    "facts": {"matched_targets": 1},
                    "data_gaps": ["stale_price_decision"],
                    "recommended_interpretation": "Price mart is stale.",
                }
            },
            per_target={"46227": {"price_row": {"vendor_code": "46227"}}},
            blocking_gaps=["stale_price_decision"],
        )
        scenario_output = {
            "blocked_by": ["supply_risk"],
            "baseline_metrics": {"current_price": 1500},
            "action_metrics": {"candidate_price": 1450},
            "delta_metrics": {"recommended_price_delta_pct": -3.3},
        }
        reasoning_input = build_forecast_reasoning_input(
            hypothesis=hypothesis.action,
            plan=plan,
            resolved_targets=resolve_forecast_targets(_price_task(), plan),
            evidence_bundle=evidence_bundle,
            scenario_output=scenario_output,
        )

        self.assertIn("supply_risk", reasoning_input.hard_blockers)
        self.assertIn("stale_price_decision", reasoning_input.soft_warnings)
        self.assertEqual(
            reasoning_input.source_status["forecast_price_agent"]["confidence"],
            "medium",
        )

    @patch("agi_ecom_app.forecast.workflow.run_forecast_reasoning")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_sales_overview")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_price_decision_overview")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_price_competitor_overview")
    def test_workflow_prefers_reasoning_output_over_calculator_text(
        self,
        mock_price,
        mock_price_decision,
        mock_sales_overview,
        mock_reasoning,
    ):
        mock_sales_overview.return_value = {"conversion_watchlist": [], "top_revenue_rows": []}
        mock_price_decision.return_value = {"rows": []}
        mock_price.return_value = {
            "rows": [
                {
                    "nm_id": 46227,
                    "vendor_code": "46227",
                    "source_price_promo": 1500,
                    "competitor_price_median": 1000,
                    "source_vs_median_gap_rub": 500,
                    "source_vs_median_gap_pct": 50,
                    "competitor_count": 8,
                    "price_status": "above_market",
                }
            ]
        }
        mock_reasoning.return_value = (
            ForecastReasoningResult(
                decision_status="needs_review",
                confidence="medium",
                decision="AI decision",
                final_recommendation="Сделать тестовое снижение цены на 5% и не запускать auto-apply.",
                summary="Reasoning summary",
                decision_why="Reasoning layer selected controlled price test.",
                primary_action="Тестовое снижение цены на 5%",
                rejected_actions=["Не масштабировать рекламу"],
                warnings=["stale_price_decision"],
                evidence_used=["price mart", "sales trend"],
                forecast_details_patch={
                    "reasoningPrimaryAction": "Тестовое снижение цены на 5%",
                    "reasoningRejectedActions": ["Не масштабировать рекламу"],
                    "reasoningEvidenceUsed": ["price mart", "sales trend"],
                    "reasoningWarnings": ["stale_price_decision"],
                },
            ),
            {"mode": "mock_override"},
        )

        result = run_forecast_workflow(task_snapshot=_price_task())

        self.assertEqual(result.finalRecommendation, "Сделать тестовое снижение цены на 5% и не запускать auto-apply.")
        self.assertEqual(result.summary, "Reasoning summary")
        self.assertEqual(result.forecastDetails["reasoningPrimaryAction"], "Тестовое снижение цены на 5%")

    @patch("agi_ecom_app.forecast.opportunity_finder.strategic_db.get_marketing_action_matrix")
    @patch("agi_ecom_app.forecast.opportunity_finder.strategic_db.get_finance_overview")
    @patch("agi_ecom_app.forecast.opportunity_finder.strategic_db.get_price_decision_overview")
    @patch("agi_ecom_app.forecast.opportunity_finder.strategic_db.get_price_competitor_overview")
    @patch("agi_ecom_app.forecast.opportunity_finder.strategic_db.get_supply_overview")
    @patch("agi_ecom_app.forecast.opportunity_finder.strategic_db.get_sales_overview")
    @patch("agi_ecom_app.forecast.opportunity_finder.strategic_db.get_sales_sku_revenue_change")
    def test_growth_opportunity_finder_ranks_scale_candidate(
        self,
        mock_sales_change,
        mock_sales_overview,
        mock_supply,
        mock_price,
        mock_price_decision,
        mock_finance,
        mock_marketing,
    ):
        mock_sales_change.return_value = {
            "rows": [
                {
                    "vendor_code": "46300",
                    "nm_id": 156051054,
                    "current_revenue": 22000,
                    "previous_revenue": 10000,
                    "delta_revenue": 12000,
                    "current_sales": 13,
                    "previous_sales": 6,
                }
            ]
        }
        mock_sales_overview.return_value = {"conversion_watchlist": []}
        mock_supply.return_value = {
            "rows": [
                {
                    "vendor_code": "46300",
                    "nm_id": 156051054,
                    "stock_qty": 96,
                    "stock_qty_full": 123,
                    "sales_7d_units": 13,
                    "days_of_stock": 51,
                    "reorder_qty_recommended": 0,
                }
            ]
        }
        mock_price_decision.return_value = {
            "rows": [
                {
                    "vendor_code": "46300",
                    "nm_id": 156051054,
                    "competitor_count": 12,
                    "price_status": "market_aligned",
                    "expected_margin_pct": 38.5,
                    "recommended_price": 2100,
                }
            ]
        }
        mock_price.return_value = {"rows": []}
        mock_finance.return_value = {
            "unit_economics_watchlist": [
                {"vendor_code": "46300", "nm_id": 156051054, "contribution_margin": 17200}
            ],
            "margin_watchlist": [],
            "fee_pressure_rows": [],
        }
        mock_marketing.return_value = {"campaign_actions": [], "query_actions": []}

        opportunities = find_growth_opportunities(limit=5)

        self.assertGreater(len(opportunities), 0)
        self.assertEqual(opportunities[0].action_family, "scale_profitable_sku_ads")
        self.assertEqual(opportunities[0].affected_skus, ["46300"])
        self.assertGreater(opportunities[0].score, 80)

    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_finance_overview")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_price_decision_overview")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_price_competitor_overview")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_supply_overview")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_marketing_search_demand_snapshot")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_marketing_visibility_snapshot")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_sales_overview")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_sales_sku_revenue_change")
    def test_forecast_workflow_accepts_growth_opportunity(
        self,
        mock_sales_change,
        mock_sales_overview,
        mock_visibility,
        mock_demand,
        mock_supply,
        mock_price,
        mock_price_decision,
        mock_finance,
    ):
        mock_sales_change.return_value = {
            "rows": [
                {
                    "vendor_code": "46300",
                    "nm_id": 156051054,
                    "current_revenue": 22000,
                    "previous_revenue": 10000,
                    "delta_revenue": 12000,
                    "current_sales": 13,
                    "previous_sales": 6,
                }
            ]
        }
        mock_sales_overview.return_value = {"conversion_watchlist": [], "top_revenue_rows": []}
        mock_visibility.return_value = {"visibility_rows": [], "position_loss_rows": [], "open_card_gap_rows": []}
        mock_demand.return_value = {"high_frequency_low_order_rows": [], "order_driving_query_rows": []}
        mock_supply.return_value = {
            "rows": [
                {
                    "vendor_code": "46300",
                    "nm_id": 156051054,
                    "stock_qty": 96,
                    "stock_qty_full": 123,
                    "sales_7d_units": 13,
                    "days_of_stock": 51,
                    "reorder_qty_recommended": 0,
                }
            ]
        }
        mock_price_decision.return_value = {"rows": []}
        mock_price.return_value = {"rows": [{"vendor_code": "46300", "nm_id": 156051054, "competitor_count": 12, "price_status": "market_aligned"}]}
        mock_finance.return_value = {
            "unit_economics_watchlist": [
                {"vendor_code": "46300", "nm_id": 156051054, "contribution_margin": 17200}
            ],
            "margin_watchlist": [],
            "fee_pressure_rows": [],
        }
        opportunity = GrowthOpportunity(
            opportunity_id="scale_profitable_sku_ads:46300",
            action_family="scale_profitable_sku_ads",
            score=92,
            evidence_summary="SKU растет и может масштабироваться.",
            affected_skus=["46300"],
            expected_metric_focus=["revenue_uplift", "contribution_margin"],
            target_mode="sku",
            primary_domain="marketing",
            growth_reason="SKU растет при положительной марже и достаточном запасе.",
        )

        result = run_forecast_workflow(task_snapshot=None, opportunity=opportunity)

        self.assertIsInstance(result, ForecastDecisionResult)
        self.assertEqual(result.plan.action_family, "scale_profitable_sku_ads")
        self.assertIsInstance(result.confidencePercent, int)
        self.assertGreaterEqual(result.confidencePercent, 10)
        self.assertLessEqual(result.confidencePercent, 95)
        self.assertEqual(result.forecastDetails["opportunityScore"], 92)
        self.assertEqual(result.forecastDetails["recommendedActionFamily"], "scale_profitable_sku_ads")
        self.assertIn("46300", result.finalRecommendation)
        self.assertTrue(any(step.operation == "подтвердить_рекламный_лимит" for step in result.resolutionPlan))
        self.assertFalse(any(step.operation == "поднять_лимит_рекламы" for step in result.resolutionPlan))

    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_finance_overview")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_price_decision_overview")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_price_competitor_overview")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_supply_overview")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_marketing_search_demand_snapshot")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_marketing_visibility_snapshot")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_sales_overview")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_sales_sku_revenue_change")
    def test_reprice_forecast_uses_price_mart_recommendation(
        self,
        mock_sales_change,
        mock_sales_overview,
        mock_visibility,
        mock_demand,
        mock_supply,
        mock_price,
        mock_price_decision,
        mock_finance,
    ):
        mock_sales_change.return_value = {
            "rows": [
                {
                    "vendor_code": "46277",
                    "nm_id": 142745322,
                    "current_revenue": 1335.15,
                    "previous_revenue": 6820.65,
                    "delta_revenue": -5485.5,
                    "current_sales": 1,
                    "previous_sales": 5,
                }
            ]
        }
        mock_sales_overview.return_value = {"conversion_watchlist": [], "top_revenue_rows": []}
        mock_visibility.return_value = {"visibility_rows": [], "position_loss_rows": [], "open_card_gap_rows": []}
        mock_demand.return_value = {"high_frequency_low_order_rows": [], "order_driving_query_rows": []}
        mock_supply.return_value = {
            "rows": [
                {
                    "vendor_code": "46277",
                    "nm_id": 142745322,
                    "stock_qty": 14,
                    "stock_qty_full": 18,
                    "sales_7d_units": 1,
                    "days_of_stock": 126,
                    "reorder_qty_recommended": 0,
                }
            ]
        }
        mock_price_decision.return_value = {
            "rows": [
                {
                    "vendor_code": "46277",
                    "nm_id": 142745322,
                    "source_price_promo": 1977.29,
                    "competitor_price_median": 442.5,
                    "source_vs_median_gap_pct": 346.85,
                    "competitor_count": 192,
                    "price_status": "above_market",
                    "recommended_price": 1550,
                    "recommended_price_delta_pct": -21.61,
                    "expected_orders_7d": 3.4,
                    "expected_net_profit_7d": 4100,
                    "expected_margin_pct": 31.2,
                    "decision_status": "lower",
                    "decision_confidence": "high",
                    "decision_reason_code": "elasticity_profit_optimum",
                    "elasticity_to_apply": -1.7,
                    "elasticity_source": "mart",
                    "cost_price_rub": 428,
                    "baseline_orders_7d": 1,
                    "current_net_profit_7d": 1200,
                    "margin_pct_current": 22.4,
                    "candidate_price_min": 1450,
                    "candidate_price_max": 1800,
                }
            ]
        }
        mock_price.return_value = {"rows": []}
        mock_finance.return_value = {"unit_economics_watchlist": [], "margin_watchlist": [], "fee_pressure_rows": []}

        opportunity = GrowthOpportunity(
            opportunity_id="reprice_with_margin_guardrail:46277",
            action_family="reprice_with_margin_guardrail",
            score=94.08,
            evidence_summary="SKU проседает, а цена выше конкурентного коридора.",
            affected_skus=["46277"],
            expected_metric_focus=["price_gap_pct", "safe_price_floor", "sales_recovery", "margin_after_price"],
            target_mode="sku",
            primary_domain="price",
            growth_reason="Цена выше рынка, есть пространство для безопасного repricing.",
        )

        result = run_forecast_workflow(task_snapshot=None, opportunity=opportunity)

        self.assertEqual(result.plan.action_family, "reprice_with_margin_guardrail")
        # The price recommendation is available, but the optional marketing mart
        # is explicitly missing in this offline fixture; the guardrail must stay on.
        self.assertEqual(result.decisionStatus, "needs_review")
        self.assertTrue(any("Marketing mart" in blocker for blocker in result.blockedBy))
        self.assertIn("1550", result.finalRecommendation)
        self.assertEqual(result.forecastDetails["recommendedActionFamily"], "reprice_with_margin_guardrail")
        margin_bridge = result.forecastDetails["marginBridge"]
        self.assertEqual(margin_bridge["type"], "price_margin_bridge")
        self.assertIn("price_cut_effect_on_gross_margin", margin_bridge)
        self.assertIn("volume_compensation_effect_on_gross_margin", margin_bridge)
        self.assertIn("components", margin_bridge)
        self.assertIn("net_profit", margin_bridge["components"])
        self.assertTrue(any(step.operation == "изменить_цену" for step in result.resolutionPlan))

    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_finance_overview")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_price_decision_overview")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_price_competitor_overview")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_supply_overview")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_marketing_search_demand_snapshot")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_marketing_visibility_snapshot")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_sales_overview")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_sales_sku_revenue_change")
    def test_reprice_forecast_uses_prepare_test_step_when_blocked(
        self,
        mock_sales_change,
        mock_sales_overview,
        mock_visibility,
        mock_demand,
        mock_supply,
        mock_price,
        mock_price_decision,
        mock_finance,
    ):
        mock_sales_change.return_value = {
            "rows": [
                {
                    "vendor_code": "46310",
                    "nm_id": 142745399,
                    "current_revenue": 1500,
                    "previous_revenue": 6500,
                    "delta_revenue": -5000,
                    "current_sales": 1,
                    "previous_sales": 4,
                }
            ]
        }
        mock_sales_overview.return_value = {"conversion_watchlist": [], "top_revenue_rows": []}
        mock_visibility.return_value = {"visibility_rows": [], "position_loss_rows": [], "open_card_gap_rows": []}
        mock_demand.return_value = {"high_frequency_low_order_rows": [], "order_driving_query_rows": []}
        mock_supply.return_value = {
            "rows": [
                {
                    "vendor_code": "46310",
                    "nm_id": 142745399,
                    "stock_qty": 3,
                    "stock_qty_full": 5,
                    "sales_7d_units": 1,
                    "days_of_stock": 14,
                    "reorder_qty_recommended": 39,
                }
            ]
        }
        mock_price_decision.return_value = {
            "rows": [
                {
                    "vendor_code": "46310",
                    "nm_id": 142745399,
                    "source_price_promo": 1656,
                    "competitor_price_median": 1100,
                    "source_vs_median_gap_pct": 50.5,
                    "competitor_count": 45,
                    "price_status": "above_market",
                    "recommended_price": 1410,
                    "expected_net_profit_7d": 5779.25,
                    "expected_orders_7d": 4.0,
                    "decision_status": "lower",
                    "decision_confidence": "high",
                    "cost_price_rub": 480,
                    "baseline_orders_7d": 1,
                    "current_net_profit_7d": 1200,
                }
            ]
        }
        mock_price.return_value = {"rows": []}
        mock_finance.return_value = {"unit_economics_watchlist": [], "margin_watchlist": [], "fee_pressure_rows": []}

        opportunity = GrowthOpportunity(
            opportunity_id="reprice_with_margin_guardrail:46310",
            action_family="reprice_with_margin_guardrail",
            score=83.96,
            evidence_summary="SKU проседает, а цена выше конкурентного коридора; нужен безопасный ценовой тест.",
            affected_skus=["46310"],
            expected_metric_focus=["price_gap_pct", "safe_price_floor", "sales_recovery", "margin_after_price"],
            target_mode="sku",
            primary_domain="price",
            growth_reason="Цена выше рынка, но сначала нужно снять блокер по остатку.",
        )

        result = run_forecast_workflow(task_snapshot=None, opportunity=opportunity)

        self.assertTrue(any(step.operation == "подготовить_тест_цены" for step in result.resolutionPlan))
        self.assertFalse(any(step.operation == "изменить_цену" for step in result.resolutionPlan))

    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_finance_overview")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_price_decision_overview")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_price_competitor_overview")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_supply_overview")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_marketing_search_demand_snapshot")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_marketing_visibility_snapshot")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_sales_overview")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_sales_sku_revenue_change")
    def test_reorder_forecast_does_not_emit_zero_qty_reorder_step(
        self,
        mock_sales_change,
        mock_sales_overview,
        mock_visibility,
        mock_demand,
        mock_supply,
        mock_price,
        mock_price_decision,
        mock_finance,
    ):
        mock_sales_change.return_value = {
            "rows": [
                {
                    "vendor_code": "46311",
                    "nm_id": 150000001,
                    "current_revenue": 5000,
                    "previous_revenue": 4800,
                    "delta_revenue": 200,
                    "current_sales": 5,
                    "previous_sales": 5,
                }
            ]
        }
        mock_sales_overview.return_value = {"conversion_watchlist": [], "top_revenue_rows": []}
        mock_visibility.return_value = {"visibility_rows": [], "position_loss_rows": [], "open_card_gap_rows": []}
        mock_demand.return_value = {"high_frequency_low_order_rows": [], "order_driving_query_rows": []}
        mock_supply.return_value = {
            "rows": [
                {
                    "vendor_code": "46311",
                    "nm_id": 150000001,
                    "stock_qty": 40,
                    "stock_qty_full": 45,
                    "sales_7d_units": 5,
                    "days_of_stock": 56,
                    "reorder_qty_recommended": 0,
                }
            ]
        }
        mock_price_decision.return_value = {"rows": []}
        mock_price.return_value = {"rows": [{"vendor_code": "46311", "nm_id": 150000001, "competitor_count": 3, "price_status": "thin_data"}]}
        mock_finance.return_value = {"unit_economics_watchlist": [], "margin_watchlist": [], "fee_pressure_rows": []}

        opportunity = GrowthOpportunity(
            opportunity_id="reorder_for_growth:46311",
            action_family="reorder_for_growth",
            score=82,
            evidence_summary="Есть скорость продаж и риск дефицита; рост нужно защищать дозаказом.",
            affected_skus=["46311"],
            expected_metric_focus=["reorder_qty", "coverage_days", "lost_sales_risk"],
            target_mode="sku",
            primary_domain="supply",
            growth_reason="Есть скорость продаж и риск дефицита; рост нужно защищать дозаказом.",
        )

        result = run_forecast_workflow(task_snapshot=None, opportunity=opportunity)

        self.assertTrue(any(step.operation == "удержать_остаток_без_дозаказа" for step in result.resolutionPlan))
        self.assertFalse(any(step.operation == "дозаказать_остаток" and step.payload.get("reorderQty") == 0 for step in result.resolutionPlan))

    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_marketing_search_demand_snapshot")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_marketing_visibility_snapshot")
    @patch("agi_ecom_app.forecast.adapters.diagnostic_input.strategic_db.get_supply_overview")
    def test_supply_guardrail_prefers_sellable_stock_quantity(
        self,
        mock_supply,
        mock_visibility,
        mock_demand,
    ):
        mock_supply.return_value = {
            "rows": [
                {
                    "nm_id": 46310,
                    "vendor_code": "46310",
                    "stock_qty": 0,
                    "stock_qty_full": 100,
                    "in_way_to_client": 100,
                    "in_way_from_client": 0,
                    "sales_7d_units": 35,
                    "days_of_stock": 0,
                    "reorder_flag": "critical",
                    "reorder_qty_recommended": 270,
                }
            ]
        }
        mock_visibility.return_value = {"visibility_rows": [], "position_loss_rows": [], "open_card_gap_rows": []}
        mock_demand.return_value = {"high_frequency_low_order_rows": [], "order_driving_query_rows": []}

        result = run_forecast_workflow(task_snapshot=_availability_before_scale_task(targets=["46310"]))

        self.assertEqual(result.forecastDetails["perTargetEvidence"][0]["available_stock"], 0.0)
        self.assertEqual(result.delta.metrics["reorder_qty"], 270.0)


if __name__ == "__main__":
    unittest.main()
