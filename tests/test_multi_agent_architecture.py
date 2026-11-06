import unittest
from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

import agi_ecom_app.main_agent as main_agent_module
from agi_ecom_app.main_agent import (
    build_action_executor_agent,
    build_diagnostic_orchestrator_agent,
    build_domain_router_agent,
    build_finance_agent,
    build_marketing_agent,
    build_price_agent,
    build_sales_agent,
    build_supply_agent,
    ensure_supply_aggregated_stock_facts,
    ensure_supply_current_stock_lookup,
    ensure_sales_decomposition_contract,
    enforce_sales_user_facing_contract,
    ensure_marketing_report_snapshot,
    ensure_sales_report_snapshot,
    validate_agent_diagnostic_result,
    validate_marketing_diagnostic_result,
    validate_sales_diagnostic_result,
)
from agi_ecom_app.rendering import render_agent_diagnostic, render_sales_diagnostic
from agi_ecom_app.routing import determine_query_intent
from agi_ecom_app.schemas import (
    AgentDiagnosticResult,
    ChatMemoryMessage,
    ChatRole,
    CrossAgentRequest,
    CrossCheckResult,
    DiagnosticFinding,
    DriverClass,
    EcomRunContext,
    EvidenceFact,
    MarketingDiagnosticResult,
    QueryIntent,
    ReportTable,
    SalesDiagnosticResult,
)
from agi_ecom_app.tools.strategic_tools import (
    _get_formula_catalog,
    build_cross_domain_dependency_plan,
)


class RoutingTests(unittest.TestCase):
    def test_finance_query_detected(self):
        intent = determine_query_intent("Сделай финансовый отчет за 23.02-01.03")
        self.assertEqual(intent, QueryIntent.FINANCE)

    def test_marketing_query_detected(self):
        intent = determine_query_intent("Покажи кампании с CTR и CPC за 23.03-29.03")
        self.assertEqual(intent, QueryIntent.MARKETING)

    def test_drr_query_detected_as_marketing(self):
        self.assertEqual(determine_query_intent("Посчитай ДРР"), QueryIntent.MARKETING)
        self.assertEqual(determine_query_intent("ДРР по кампаниям"), QueryIntent.MARKETING)

    def test_price_query_detected(self):
        self.assertEqual(
            determine_query_intent("Покажи цены конкурентов по 46203"),
            QueryIntent.PRICE,
        )
        self.assertEqual(determine_query_intent("Какие товары дороже рынка?"), QueryIntent.PRICE)

    def test_click_price_query_stays_marketing(self):
        self.assertEqual(determine_query_intent("Покажи цену клика и CPC"), QueryIntent.MARKETING)

    def test_clicks_without_orders_with_price_signal_stays_marketing_primary(self):
        self.assertEqual(
            determine_query_intent("Клики есть, заказов нет, возможно цена высокая"),
            QueryIntent.MARKETING,
        )

    def test_profit_by_sku_query_detected_as_finance(self):
        self.assertEqual(
            determine_query_intent("Сравни прибыль по SKU: 1–7 апреля vs 8–14 апреля 2026"),
            QueryIntent.FINANCE,
        )

    def test_broad_sales_profit_diagnostic_detected_as_finance(self):
        query = (
            "Почему за последнюю неделю изменились продажи и прибыль по ключевым SKU? "
            "Проверь гипотезы: спрос, реклама, цена, остатки и экономика."
        )

        self.assertEqual(determine_query_intent(query), QueryIntent.FINANCE)

    def test_sales_query_detected(self):
        intent = determine_query_intent("Покажи продажи и заказы по артикулам за неделю")
        self.assertEqual(intent, QueryIntent.SALES)

    def test_sales_status_query_detected_as_sales(self):
        self.assertEqual(determine_query_intent("Что с продажами?"), QueryIntent.SALES)

    def test_sales_report_aliases_detected_as_sales(self):
        self.assertEqual(determine_query_intent("Сделай sales отчет"), QueryIntent.SALES)
        self.assertEqual(
            determine_query_intent("Отчет вчера к позавчера, неделя к неделе, месяц к месяцу"),
            QueryIntent.SALES,
        )

    def test_supply_query_detected(self):
        intent = determine_query_intent("Покажи остатки и поставки по артикулам")
        self.assertEqual(intent, QueryIntent.SUPPLY)

    def test_supply_for_marketing_query_detected_as_supply(self):
        self.assertEqual(
            determine_query_intent("Хватит ли остатков на маркетинг?"),
            QueryIntent.SUPPLY,
        )

    def test_inventory_turnover_query_detected_as_supply(self):
        self.assertEqual(
            determine_query_intent("Посчитай оборачиваемость склада"),
            QueryIntent.SUPPLY,
        )
        self.assertEqual(
            determine_query_intent("Мне нужно понять, какие товары держат деньги в складе"),
            QueryIntent.SUPPLY,
        )

    def test_default_route_falls_back_to_finance(self):
        intent = determine_query_intent("Что у нас по цифрам?")
        self.assertEqual(intent, QueryIntent.FINANCE)


class CrossDomainDependencyPlanTests(unittest.TestCase):
    def _targets(self, plan):
        return [item["target_agent"] for item in plan["recommended_secondary_agents"]]

    def test_drr_adds_sales_for_marketing_primary(self):
        plan = build_cross_domain_dependency_plan(
            user_query="Посчитай ДРР по кампаниям",
            primary_agent="marketing",
        )

        self.assertEqual(self._targets(plan), ["sales"])

    def test_sales_status_adds_supply_and_marketing(self):
        plan = build_cross_domain_dependency_plan(
            user_query="Что с продажами?",
            primary_agent="sales",
            max_secondary_agents=3,
        )

        self.assertEqual(self._targets(plan), ["supply", "marketing"])

    def test_sales_report_query_does_not_add_secondary_agents(self):
        plan = build_cross_domain_dependency_plan(
            user_query="Покажи продажи за неделю",
            primary_agent="sales",
            max_secondary_agents=3,
        )

        self.assertEqual(self._targets(plan), [])

    def test_supply_stock_coverage_adds_sales(self):
        plan = build_cross_domain_dependency_plan(
            user_query="Хватит ли остатков?",
            primary_agent="supply",
            max_secondary_agents=3,
        )

        self.assertEqual(self._targets(plan), ["sales"])

    def test_supply_for_marketing_adds_sales_and_marketing(self):
        plan = build_cross_domain_dependency_plan(
            user_query="Хватит ли остатков на маркетинг?",
            primary_agent="supply",
            max_secondary_agents=3,
        )

        self.assertEqual(self._targets(plan), ["sales", "marketing"])

    def test_overstock_adds_sales_and_finance_for_supply_primary(self):
        plan = build_cross_domain_dependency_plan(
            user_query="Есть overstock и зависшие остатки?",
            primary_agent="supply",
            max_secondary_agents=3,
        )

        self.assertEqual(self._targets(plan), ["sales", "finance"])

    def test_romi_limits_marketing_primary_to_sales_and_finance(self):
        plan = build_cross_domain_dependency_plan(
            user_query="Проверь ROMI и окупаемость рекламы",
            primary_agent="marketing",
        )

        self.assertEqual(self._targets(plan), ["sales", "finance"])
        self.assertEqual(plan["max_secondary_agents"], 2)

    def test_sales_traffic_signal_recommends_marketing(self):
        plan = build_cross_domain_dependency_plan(
            user_query="Почему изменились продажи?",
            primary_agent="sales",
            findings=[
                {
                    "factor": "Верх воронки",
                    "observation": "Просели открытия карточки и трафик.",
                }
            ],
        )

        self.assertIn("marketing", self._targets(plan))

    def test_sales_stock_signal_recommends_supply(self):
        plan = build_cross_domain_dependency_plan(
            user_query="Почему изменились продажи?",
            primary_agent="sales",
            findings=[
                {
                    "factor": "Доступность",
                    "observation": "Есть риск out-of-stock и просадка остатков по складам.",
                }
            ],
        )

        self.assertIn("supply", self._targets(plan))

    def test_finance_profit_query_recommends_sales_and_marketing_when_ads_matter(self):
        plan = build_cross_domain_dependency_plan(
            user_query="Почему выручка выросла, а прибыль нет из-за рекламных расходов?",
            primary_agent="finance",
        )

        self.assertEqual(self._targets(plan), ["sales", "marketing"])

    def test_finance_profit_week_to_week_does_not_trigger_false_sales_marketing(self):
        plan = build_cross_domain_dependency_plan(
            user_query="Сравни прибыль неделя к неделе",
            primary_agent="finance",
            max_secondary_agents=3,
        )

        self.assertEqual(self._targets(plan), [])

    def test_broad_sales_profit_diagnostic_adds_all_round_one_domains(self):
        plan = build_cross_domain_dependency_plan(
            user_query=(
                "Почему за последнюю неделю изменились продажи и прибыль по ключевым SKU? "
                "Проверь гипотезы: спрос, реклама, цена, остатки и экономика. "
                "Разложи вклад по SKU."
            ),
            primary_agent="finance",
            max_secondary_agents=5,
        )

        self.assertEqual(self._targets(plan), ["sales", "marketing", "price", "supply"])
        self.assertTrue(
            all(
                item["source"] == "broad_diagnostic"
                for item in plan["recommended_secondary_agents"]
            )
        )

    def test_broad_sales_profit_diagnostic_adds_ads_when_primary_is_sales(self):
        plan = build_cross_domain_dependency_plan(
            user_query=(
                "Почему за последнюю неделю изменились продажи и прибыль по ключевым SKU? "
                "Проверь гипотезы: спрос, реклама, цена, остатки и экономика."
            ),
            primary_agent="sales",
            max_secondary_agents=5,
        )

        self.assertEqual(self._targets(plan), ["marketing", "price", "supply", "finance"])
        self.assertTrue(
            all(
                item["source"] == "broad_diagnostic"
                for item in plan["recommended_secondary_agents"]
            )
        )

    def test_explicit_hypothesis_query_adds_named_domains_without_profit_scope(self):
        plan = build_cross_domain_dependency_plan(
            user_query="Почему просели продажи? Проверь гипотезы: спрос, реклама, цена, остатки.",
            primary_agent="sales",
            max_secondary_agents=5,
        )

        self.assertEqual(self._targets(plan), ["marketing", "price", "supply"])
        self.assertTrue(
            all(
                item["source"] == "explicit_hypothesis"
                for item in plan["recommended_secondary_agents"]
            )
        )

    def test_profit_leader_query_adds_hypothesis_domains(self):
        plan = build_cross_domain_dependency_plan(
            user_query="Какой товар принес больше прибыли в этом месяце?",
            primary_agent="finance",
            max_secondary_agents=5,
        )

        self.assertEqual(self._targets(plan), ["sales", "marketing", "price", "supply"])

    def test_price_query_uses_price_domain(self):
        plan = build_cross_domain_dependency_plan(
            user_query="Какие товары дороже рынка?",
            primary_agent="sales",
            max_secondary_agents=3,
        )

        self.assertEqual(self._targets(plan), ["price"])

    def test_sales_query_with_price_signal_recommends_price(self):
        plan = build_cross_domain_dependency_plan(
            user_query="Почему продажи упали, товар дороже конкурентов?",
            primary_agent="sales",
            max_secondary_agents=3,
        )

        self.assertIn("price", self._targets(plan))

    def test_clicks_without_orders_with_price_signal_recommends_price(self):
        plan = build_cross_domain_dependency_plan(
            user_query="Клики есть, заказов нет, возможно цена высокая",
            primary_agent="marketing",
            max_secondary_agents=3,
        )

        self.assertIn("price", self._targets(plan))

    def test_price_decision_adds_sales_and_finance_for_price_primary(self):
        plan = build_cross_domain_dependency_plan(
            user_query="Нужно ли снизить цену и скидку?",
            primary_agent="price",
            max_secondary_agents=3,
        )

        self.assertEqual(self._targets(plan), ["sales", "finance"])

    def test_price_sales_signal_recommends_price(self):
        plan = build_cross_domain_dependency_plan(
            user_query="Почему изменились продажи?",
            primary_agent="sales",
            findings=[
                {
                    "factor": "Цена",
                    "observation": "Артикул выглядит дороже конкурентов и вне ценового сегмента.",
                }
            ],
            max_secondary_agents=3,
        )

        self.assertIn("price", self._targets(plan))

    def test_cross_agent_requests_are_limited_to_two(self):
        plan = build_cross_domain_dependency_plan(
            user_query="Почему изменились продажи?",
            primary_agent="sales",
            cross_agent_requests=[
                {"target_agent": "marketing", "reason": "Нужен трафик", "question": "Проверь трафик."},
                {"target_agent": "supply", "reason": "Нужны остатки", "question": "Проверь остатки."},
                {"target_agent": "finance", "reason": "Нужна маржа", "question": "Проверь маржу."},
            ],
        )

        self.assertEqual(self._targets(plan), ["marketing", "supply"])
        self.assertEqual([item["target_agent"] for item in plan["skipped_agents"]], ["finance"])

    def test_cross_agent_requests_allow_three_or_five_when_requested(self):
        requests = [
            {"target_agent": "marketing", "reason": "Нужен трафик", "question": "Проверь трафик."},
            {"target_agent": "supply", "reason": "Нужны остатки", "question": "Проверь остатки."},
            {"target_agent": "finance", "reason": "Нужна маржа", "question": "Проверь маржу."},
        ]

        plan = build_cross_domain_dependency_plan(
            user_query="Почему изменились продажи?",
            primary_agent="sales",
            cross_agent_requests=requests,
            max_secondary_agents=3,
        )

        self.assertEqual(self._targets(plan), ["marketing", "supply", "finance"])
        self.assertEqual(plan["max_secondary_agents"], 3)

        capped_plan = build_cross_domain_dependency_plan(
            user_query="Почему изменились продажи?",
            primary_agent="sales",
            cross_agent_requests=requests,
            max_secondary_agents=10,
        )

        self.assertEqual(capped_plan["max_secondary_agents"], 5)
        self.assertLessEqual(len(capped_plan["recommended_secondary_agents"]), 5)

    def test_finance_is_not_added_without_financial_signal(self):
        plan = build_cross_domain_dependency_plan(
            user_query="Какие кампании масштабировать?",
            primary_agent="marketing",
            max_secondary_agents=3,
        )

        self.assertNotIn("finance", self._targets(plan))

    def test_repeated_same_question_is_skipped(self):
        plan = build_cross_domain_dependency_plan(
            user_query="Проверь смежные факторы.",
            primary_agent="sales",
            cross_agent_requests=[
                {"target_agent": "marketing", "reason": "Нужен трафик", "question": "Проверь трафик."},
                {"target_agent": "supply", "reason": "Нужны остатки", "question": "Проверь остатки."},
            ],
            already_asked=[
                {"target_agent": "marketing", "question": "Проверь трафик."},
            ],
            max_secondary_agents=3,
        )

        self.assertEqual(self._targets(plan), ["supply"])
        self.assertEqual([item["target_agent"] for item in plan["skipped_agents"]], ["marketing"])


class SupplyCurrentStockLookupTests(unittest.TestCase):
    @patch("agi_ecom_app.main_agent.strategic_db.get_supply_overview")
    def test_simulated_latest_known_stock_is_labeled_in_rewritten_summary(
        self,
        mock_overview,
    ):
        mock_overview.return_value = {
            "simulation": {
                "enabled": True,
                "effective_status": "simulated_current",
                "source_snapshot_date": "2026-07-08",
            },
            "sku_stock_lookup": {
                "status": "found",
                "snapshot_date": date(2026, 7, 8),
                "nm_id": 70195171,
                "vendor_code": "46229",
                "available_stock_qty": 0,
                "full_stock_qty": 3,
                "in_way_to_client": 2,
                "in_way_from_client": 1,
            },
        }
        output = CrossCheckResult(
            agent_name="supply",
            summary="Остаток 0 шт.",
            verdict="confirmed",
            facts=[
                EvidenceFact(
                    metric="available_stock_qty",
                    value=0,
                    unit="pcs",
                    period="2026-07-08",
                    sku="46229",
                    evidence_ref="get_supply_overview.sku_stock_lookup.available_stock_qty",
                )
            ],
        )

        result = ensure_supply_aggregated_stock_facts(output, replace_summary=True)

        self.assertIn("Эмуляция", result.summary)
        self.assertIn("2026-07-08", result.summary)
        self.assertIn("доступно 0 шт", result.summary)

    @patch("agi_ecom_app.main_agent.strategic_db.get_supply_overview")
    def test_single_sku_lookup_replaces_warehouse_row_with_aggregate(self, mock_overview):
        mock_overview.return_value = {
            "sku_stock_lookup": {
                "status": "found",
                "snapshot_date": date(2026, 7, 8),
                "nm_id": 70194686,
                "vendor_code": "46227",
                "available_stock_qty": 327,
                "full_stock_qty": 343,
                "unavailable_stock_qty": 16,
                "in_way_to_client": 13,
                "in_way_from_client": 3,
                "matches": [],
            }
        }
        output = AgentDiagnosticResult(
            agent_name="supply",
            summary="В одной складской строке найдено 210 шт.",
            tables=[
                ReportTable(
                    title="Складская строка",
                    columns=["SKU", "quantity"],
                    rows=[["46227", 210]],
                )
            ],
        )

        result = ensure_supply_current_stock_lookup(
            output,
            user_query="Какой текущий доступный остаток по SKU/артикулу 46227.",
            query_plan={
                "scope": "single_sku",
                "targets": [{"target_type": "vendor_code", "value": "46227"}],
            },
        )

        mock_overview.assert_called_once_with(product_keyword="46227", limit=10)
        self.assertIn("327 шт", result.summary)
        self.assertNotIn("210 шт", result.summary)
        self.assertEqual(result.tables[0].rows[0][3], 327)
        self.assertEqual(result.facts[0].value, 327)
        self.assertEqual(
            result.facts[0].evidence_ref,
            "get_supply_overview.sku_stock_lookup",
        )

    @patch("agi_ecom_app.main_agent.strategic_db.get_supply_overview")
    def test_missing_sku_row_is_not_converted_to_zero(self, mock_overview):
        mock_overview.return_value = {
            "sku_stock_lookup": {
                "status": "no_data",
                "available_stock_qty": None,
                "matches": [],
            }
        }
        output = AgentDiagnosticResult(
            agent_name="supply",
            summary="Нет строки, считаю остаток равным 0.",
            tables=[
                ReportTable(
                    title="Ошибочный результат",
                    columns=["SKU", "quantity"],
                    rows=[["UNKNOWN-1", 0]],
                )
            ],
        )

        result = ensure_supply_current_stock_lookup(
            output,
            user_query="Текущий доступный остаток по SKU UNKNOWN-1",
            query_plan={
                "scope": "single_sku",
                "targets": [{"target_type": "vendor_code", "value": "UNKNOWN-1"}],
            },
        )

        self.assertIn("не считается равным нулю", result.summary)
        self.assertNotIn("0 шт", result.summary)
        self.assertIsNone(result.tables[0].rows[0][3])
        self.assertEqual(result.facts, [])
        self.assertEqual(result.data_gaps[0].metric, "current_available_stock")

    @patch("agi_ecom_app.main_agent.strategic_db.get_supply_overview")
    def test_non_lookup_supply_question_is_left_unchanged(self, mock_overview):
        output = AgentDiagnosticResult(
            agent_name="supply",
            summary="Покрытие запасом требует расчёта.",
            tables=[ReportTable(title="Покрытие", columns=["SKU"], rows=[["46227"]])],
        )

        result = ensure_supply_current_stock_lookup(
            output,
            user_query="На сколько дней хватит запасов по SKU 46227?",
        )

        self.assertIs(result, output)
        self.assertEqual(result.summary, "Покрытие запасом требует расчёта.")
        mock_overview.assert_not_called()

    @patch("agi_ecom_app.main_agent.strategic_db.get_supply_overview")
    def test_portfolio_query_plan_never_rewrites_sku_matrix_as_single_sku(self, mock_overview):
        output = AgentDiagnosticResult(
            agent_name="supply",
            summary="Полная матрица рассчитана.",
            tables=[
                ReportTable(
                    title="SKU-матрица",
                    columns=["SKU", "days_of_cover"],
                    rows=[["46227", 75.5], ["46303", 72.8]],
                )
            ],
        )

        result = ensure_supply_current_stock_lookup(
            output,
            user_query="Построй SKU-матрицу по текущему доступному остатку",
            query_plan={"scope": "portfolio", "targets": []},
        )

        self.assertIs(result, output)
        self.assertEqual(len(result.tables[0].rows), 2)
        mock_overview.assert_not_called()

    @patch("agi_ecom_app.main_agent.strategic_db.get_supply_overview")
    def test_secondary_stock_facts_replace_single_warehouse_rows(self, mock_overview):
        mock_overview.return_value = {
            "sku_stock_lookup": {
                "status": "found",
                "snapshot_date": date(2026, 7, 8),
                "nm_id": 70194686,
                "vendor_code": "46227",
                "available_stock_qty": 327,
                "full_stock_qty": 343,
                "in_way_to_client": 13,
                "in_way_from_client": 3,
            }
        }
        output = CrossCheckResult(
            agent_name="supply",
            summary="Одна складская строка: 210 шт., в пути 2 шт.",
            verdict="confirmed",
            facts=[
                EvidenceFact(
                    metric="stock_quantity",
                    value=210,
                    unit="pcs",
                    period="2026-07-08",
                    sku="46227",
                    source_table="supply.wb_supplier_stocks",
                    evidence_ref="supply_run_sql.row[1]",
                ),
                EvidenceFact(
                    metric="in_transit_to_client",
                    value=2,
                    unit="pcs",
                    period="2026-07-08",
                    sku="46227",
                    source_table="supply.wb_supplier_stocks",
                    evidence_ref="supply_run_sql.row[1]",
                ),
            ],
            tables=[
                ReportTable(
                    title="Складская строка",
                    columns=["SKU", "stock_quantity", "in_transit_to_client"],
                    rows=[["46227", 210, 2]],
                )
            ],
        )

        result = ensure_supply_aggregated_stock_facts(output, replace_summary=True)

        self.assertEqual(
            [(fact.metric, fact.value) for fact in result.facts],
            [("available_stock_qty", 327), ("in_way_to_client", 13)],
        )
        self.assertNotIn("210", result.summary)
        self.assertIn("327", result.summary)
        self.assertEqual(result.tables[0].rows[0][1:5], [327, 343, 13, 3])
        self.assertTrue(
            all(
                fact.evidence_ref.startswith("get_supply_overview.sku_stock_lookup")
                for fact in result.facts
            )
        )

    @patch("agi_ecom_app.main_agent.strategic_db.get_supply_overview")
    def test_secondary_missing_stock_row_is_not_kept_as_zero(self, mock_overview):
        mock_overview.return_value = {
            "sku_stock_lookup": {"status": "no_data", "matches": []}
        }
        output = CrossCheckResult(
            agent_name="supply",
            summary="Нет строки, считаю 0.",
            verdict="no_data",
            facts=[
                EvidenceFact(
                    metric="stock_quantity",
                    value=0,
                    unit="pcs",
                    period="2026-07-08",
                    sku="UNKNOWN-1",
                    source_table="supply.wb_supplier_stocks",
                    evidence_ref="supply_run_sql.empty",
                )
            ],
        )

        result = ensure_supply_aggregated_stock_facts(output, replace_summary=True)

        self.assertEqual(result.facts, [])
        self.assertIn("не равен нулю", result.summary)
        self.assertEqual(result.tables[0].rows[0][1], None)
        self.assertEqual(result.data_gaps[0].metric, "current_available_stock:UNKNOWN-1")

    @patch("agi_ecom_app.main_agent.strategic_db.get_supply_overview")
    def test_claimed_lookup_reference_is_still_reverified(self, mock_overview):
        mock_overview.return_value = {
            "sku_stock_lookup": {
                "status": "found",
                "snapshot_date": date(2026, 7, 8),
                "nm_id": 70194686,
                "vendor_code": "46227",
                "available_stock_qty": 327,
                "full_stock_qty": 343,
                "in_way_to_client": 13,
                "in_way_from_client": 3,
            }
        }
        output = CrossCheckResult(
            agent_name="supply",
            summary="Подтверждено 210 шт.",
            verdict="confirmed",
            facts=[
                EvidenceFact(
                    metric="available_stock_qty",
                    value=210,
                    unit="pcs",
                    period="2026-07-08",
                    sku="46227",
                    evidence_ref=(
                        "get_supply_overview.sku_stock_lookup.available_stock_qty"
                    ),
                )
            ],
        )

        result = ensure_supply_aggregated_stock_facts(output, replace_summary=True)

        mock_overview.assert_called_once_with(product_keyword="46227", limit=10)
        self.assertEqual([(fact.metric, fact.value) for fact in result.facts], [
            ("available_stock_qty", 327),
        ])
        self.assertEqual(result.facts[0].source_table, "supply.wb_supplier_stocks")
        self.assertNotIn("210", result.model_dump_json())

    @patch("agi_ecom_app.main_agent.strategic_db.get_supply_overview")
    def test_lookup_exception_drops_raw_stock_evidence(self, mock_overview):
        mock_overview.side_effect = RuntimeError("database unavailable")
        output = CrossCheckResult(
            agent_name="supply",
            summary="Остаток 210 шт.",
            verdict="confirmed",
            facts=[
                EvidenceFact(
                    metric="stock_quantity",
                    value=210,
                    unit="pcs",
                    period="2026-07-08",
                    sku="46227",
                    evidence_ref="supply_run_sql.row[1]",
                )
            ],
            tables=[
                ReportTable(
                    title="Складская строка",
                    columns=["SKU", "quantity"],
                    rows=[["46227", 210]],
                )
            ],
        )

        result = ensure_supply_aggregated_stock_facts(output, replace_summary=True)

        self.assertEqual(result.facts, [])
        self.assertNotIn("210", result.model_dump_json())
        self.assertEqual(result.tables[0].columns[1], "available_stock_qty")
        self.assertIsNone(result.tables[0].rows[0][1])
        self.assertEqual(result.data_gaps[0].metric, "current_available_stock:46227")
        self.assertIn("ошибкой", result.data_gaps[0].reason)

    @patch("agi_ecom_app.main_agent.strategic_db.get_supply_overview")
    def test_raw_stock_table_without_facts_is_replaced(self, mock_overview):
        mock_overview.return_value = {
            "sku_stock_lookup": {
                "status": "found",
                "snapshot_date": date(2026, 7, 8),
                "nm_id": 70194686,
                "vendor_code": "46227",
                "available_stock_qty": 327,
                "full_stock_qty": 343,
                "in_way_to_client": 13,
                "in_way_from_client": 3,
            }
        }
        output = CrossCheckResult(
            agent_name="supply",
            summary="Одна строка склада.",
            verdict="confirmed",
            facts=[],
            tables=[
                ReportTable(
                    title="Складской остаток",
                    columns=["SKU", "quantity"],
                    rows=[["46227", 210]],
                )
            ],
        )

        result = ensure_supply_aggregated_stock_facts(output, replace_summary=True)

        mock_overview.assert_called_once_with(product_keyword="46227", limit=10)
        self.assertEqual([(fact.metric, fact.value) for fact in result.facts], [
            ("available_stock_qty", 327),
        ])
        self.assertEqual(result.tables[0].rows[0][1:5], [327, 343, 13, 3])
        self.assertNotIn("210", result.model_dump_json())

    @patch("agi_ecom_app.main_agent.strategic_db.get_supply_overview")
    def test_pure_turnover_evidence_does_not_trigger_stock_lookup(self, mock_overview):
        output = CrossCheckResult(
            agent_name="supply",
            summary="Оборачиваемость 45 дней.",
            verdict="confirmed",
            facts=[
                EvidenceFact(
                    metric="inventory_turnover_days",
                    value=45,
                    unit="days",
                    period="2026-06-09/2026-07-08",
                    sku="46227",
                    evidence_ref="get_inventory_turnover.sku[46227]",
                )
            ],
            tables=[
                ReportTable(
                    title="Оборачиваемость",
                    columns=["SKU", "turnover_days"],
                    rows=[["46227", 45]],
                )
            ],
        )

        result = ensure_supply_aggregated_stock_facts(output, replace_summary=True)

        self.assertIs(result, output)
        self.assertEqual(result.facts[0].value, 45)
        self.assertEqual(result.tables[0].rows, [["46227", 45]])
        mock_overview.assert_not_called()


class BuilderTests(unittest.TestCase):
    def setUp(self):
        patcher = patch(
            "agi_ecom_app.main_agent.get_settings",
            return_value=SimpleNamespace(openai_model="gpt-test"),
        )
        self.addCleanup(patcher.stop)
        patcher.start()
        main_agent_module._build_domain_agents.cache_clear()
        build_action_executor_agent.cache_clear()
        build_diagnostic_orchestrator_agent.cache_clear()
        build_domain_router_agent.cache_clear()
        build_finance_agent.cache_clear()
        build_marketing_agent.cache_clear()
        build_price_agent.cache_clear()
        build_sales_agent.cache_clear()
        build_supply_agent.cache_clear()

    def test_domain_agents_have_expected_tools(self):
        finance = build_finance_agent()
        marketing = build_marketing_agent()
        price = build_price_agent()
        sales = build_sales_agent()
        supply = build_supply_agent()

        self.assertEqual(
            [tool.name for tool in finance.tools],
            [
                "get_database_time",
                "get_finance_formula_catalog",
                "get_finance_overview",
                "get_finance_marketplace_report",
                "get_finance_marketplace_report_text",
                "get_finance_sku_report",
                "get_finance_sku_report_text",
                "get_finance_sku_profit_change_report",
                "get_finance_sku_profit_change_report_text",
                "finance_list_tables",
                "finance_describe_table",
                "finance_search_columns",
                "finance_sample_rows",
                "finance_run_sql",
            ],
        )
        self.assertEqual(
            [tool.name for tool in marketing.tools],
            [
                "get_database_time",
                "get_marketing_report_snapshot",
                "get_marketing_formula_catalog",
                "get_marketing_table_catalog",
                "get_marketing_deep_diagnostic",
                "get_marketing_overview",
                "get_marketing_campaign_period_report",
                "get_marketing_drr_report",
                "get_marketing_sku_period_comparison_report",
                "get_marketing_campaign_efficiency_snapshot",
                "get_marketing_query_performance_snapshot",
                "get_marketing_bid_pressure_snapshot",
                "get_marketing_visibility_snapshot",
                "get_marketing_search_demand_snapshot",
                "get_marketing_action_matrix",
                "marketing_list_tables",
                "marketing_describe_table",
                "marketing_search_columns",
                "marketing_sample_rows",
                "marketing_run_sql",
            ],
        )
        self.assertEqual(
            [tool.name for tool in sales.tools],
            [
                "get_database_time",
                "get_sales_formula_catalog",
                "get_sales_report_snapshot",
                "get_sales_table_catalog",
                "get_sales_overview",
                "get_sku_daily_price_sales_comparison",
                "get_sales_sku_revenue_change_report",
                "sales_list_tables",
                "sales_describe_table",
                "sales_search_columns",
                "sales_sample_rows",
                "sales_run_sql",
            ],
        )
        self.assertEqual(
            [tool.name for tool in price.tools],
            [
                "get_database_time",
                "get_price_formula_catalog",
                "get_price_table_catalog",
                "get_price_context",
                "get_price_decision_overview",
                "get_price_competitor_overview",
                "get_price_competitor_details",
                "get_sku_daily_price_sales_comparison",
                "price_list_tables",
                "price_describe_table",
                "price_search_columns",
                "price_sample_rows",
                "price_run_sql",
            ],
        )
        self.assertEqual(
            [tool.name for tool in supply.tools],
            [
                "get_database_time",
                "get_supply_formula_catalog",
                "get_supply_table_catalog",
                "get_supply_overview",
                "get_current_inventory_value",
                "get_inventory_turnover",
                "supply_list_tables",
                "supply_describe_table",
                "supply_search_columns",
                "supply_sample_rows",
                "supply_run_sql",
            ],
        )
        self.assertFalse(getattr(finance, "handoffs", []))
        self.assertFalse(getattr(marketing, "handoffs", []))
        self.assertFalse(getattr(price, "handoffs", []))
        self.assertFalse(getattr(sales, "handoffs", []))
        self.assertFalse(getattr(supply, "handoffs", []))

    def test_sales_report_snapshot_is_sales_only(self):
        finance = build_finance_agent()
        marketing = build_marketing_agent()
        price = build_price_agent()
        sales = build_sales_agent()
        supply = build_supply_agent()

        self.assertIn("get_sales_report_snapshot", [tool.name for tool in sales.tools])
        for agent in [finance, marketing, price, supply]:
            self.assertNotIn("get_sales_report_snapshot", [tool.name for tool in agent.tools])

    def test_marketing_report_snapshot_is_marketing_only(self):
        finance = build_finance_agent()
        marketing = build_marketing_agent()
        price = build_price_agent()
        sales = build_sales_agent()
        supply = build_supply_agent()

        self.assertIn("get_marketing_report_snapshot", [tool.name for tool in marketing.tools])
        for agent in [finance, price, sales, supply]:
            self.assertNotIn("get_marketing_report_snapshot", [tool.name for tool in agent.tools])

    def test_specialist_agents_can_choose_data_access_path(self):
        agents = [
            build_sales_agent(),
            build_marketing_agent(),
            build_finance_agent(),
            build_price_agent(),
            build_supply_agent(),
            build_action_executor_agent(),
        ]

        for agent in agents:
            with self.subTest(agent=agent.name):
                self.assertEqual(agent.model_settings.tool_choice, "auto")

    def test_domain_router_exposes_expected_specialists(self):
        router = build_domain_router_agent()

        self.assertEqual(
            [tool.name for tool in router.tools],
            ["sales_agent", "finance_agent", "marketing_agent", "price_agent", "supply_agent"],
        )
        self.assertIsNone(router.output_type)

    def test_diagnostic_orchestrator_exposes_only_domain_specialists(self):
        orchestrator = build_diagnostic_orchestrator_agent()

        self.assertEqual(orchestrator.name, "Diagnostic Orchestrator")
        self.assertEqual([tool.name for tool in orchestrator.tools], [])
        self.assertNotIn("price_agent", [tool.name for tool in orchestrator.tools])
        self.assertNotIn("product_agent", [tool.name for tool in orchestrator.tools])
        self.assertIsNone(orchestrator.output_type)

    def test_diagnostic_orchestrator_instructions_hide_service_protocol(self):
        orchestrator = build_diagnostic_orchestrator_agent()
        context = EcomRunContext(
            user_query="Почему изменились продажи?",
            intent=QueryIntent.SALES,
            today=date(2026, 4, 7),
            db_target="localhost:55434/wb_demo",
            chat_id="chat-1",
        )

        instructions = orchestrator.instructions(SimpleNamespace(context=context), orchestrator)

        self.assertIn("cross_agent_requests", instructions)
        self.assertIn("Python-controlled workflow", instructions)
        self.assertIn("не вызывая tools", instructions)
        self.assertIn("не выводи", instructions.lower())
        self.assertIn("Подтверждено", instructions)
        self.assertIn("Ограничения", instructions)
        self.assertIn("resolved_gaps", instructions)
        self.assertIn("Что изменилось механически", instructions)
        self.assertIn("primary_evidence.sales_report_snapshot", instructions)
        self.assertIn("перед заголовком", instructions)
        self.assertIn("матрица гипотез по SKU", instructions)
        self.assertIn("Проверка действий", instructions)
        self.assertIn("Проверка гипотез", instructions)
        self.assertIn("`Гипотеза`, `Признак`, `Результат`, `Статус`, `Смысл`", instructions)
        self.assertIn("запрещено писать `nm_id`", instructions)
        self.assertIn("Price Agent", instructions)
        self.assertIn("Не создавай 4P/CEO/BusinessReport", instructions)
        self.assertNotIn("orders_delta_pct", instructions)
        self.assertNotIn("price_gap_pct", instructions)

    def test_domain_prompts_include_price_cross_agent_rules(self):
        context = EcomRunContext(
            user_query="Почему изменились продажи?",
            intent=QueryIntent.SALES,
            today=date(2026, 4, 7),
            db_target="localhost:55434/wb_demo",
            chat_id="chat-1",
        )
        agents = [
            build_sales_agent(),
            build_marketing_agent(),
            build_finance_agent(),
            build_supply_agent(),
        ]

        for agent in agents:
            instructions = agent.instructions(SimpleNamespace(context=context), agent)
            self.assertIn("cross_agent_requests", instructions)
            self.assertIn("price", instructions)

    def test_domain_prompts_keep_nm_id_internal_only(self):
        context = EcomRunContext(
            user_query="Почему изменились продажи?",
            intent=QueryIntent.FINANCE,
            today=date(2026, 4, 7),
            db_target="localhost:55434/wb_demo",
            chat_id="chat-1",
        )
        agents = [
            build_sales_agent(),
            build_marketing_agent(),
            build_price_agent(),
            build_supply_agent(),
            build_finance_agent(),
        ]

        for agent in agents:
            instructions = agent.instructions(SimpleNamespace(context=context), agent)
            self.assertIn(
                "Пользователю показывай SKU/артикул; `nm_id` используй только во внутренних полях",
                instructions,
            )

        marketing_instructions = build_marketing_agent().instructions(
            SimpleNamespace(context=context),
            build_marketing_agent(),
        )
        self.assertIn(
            "без доказательной таблицы с расходом, ценой клика, заказами и ДРР",
            marketing_instructions,
        )

    def test_domain_prompts_include_common_workflow_rules(self):
        context = EcomRunContext(
            user_query="Почему изменились продажи?",
            intent=QueryIntent.SALES,
            today=date(2026, 4, 7),
            db_target="localhost:55434/wb_demo",
            chat_id="chat-1",
        )
        agents = [
            build_sales_agent(),
            build_marketing_agent(),
            build_price_agent(),
            build_supply_agent(),
            build_finance_agent(),
        ]

        for agent in agents:
            instructions = agent.instructions(SimpleNamespace(context=context), agent)
            self.assertIn("COMMON_DOMAIN_WORKFLOW_RULES", instructions)
            self.assertIn("runner сам запустит следующий раунд", instructions)
            self.assertIn("Не подтверждай факты чужого домена самостоятельно", instructions)
            self.assertIn("data_gaps", instructions)

    def test_domain_prompts_include_hypothesis_maps(self):
        context = EcomRunContext(
            user_query="Почему изменились продажи?",
            intent=QueryIntent.SALES,
            today=date(2026, 4, 7),
            db_target="localhost:55434/wb_demo",
            chat_id="chat-1",
        )
        expectations = [
            (build_sales_agent(), ("`marketing`", "`price`", "`supply`", "`finance`")),
            (build_marketing_agent(), ("`sales`", "`price`", "`supply`", "`finance`")),
            (build_price_agent(), ("`sales`", "`marketing`", "`supply`", "`finance`")),
            (build_supply_agent(), ("`sales`", "`marketing`", "`price`", "`finance`")),
            (build_finance_agent(), ("`sales`", "`marketing`", "`price`", "`supply`")),
        ]

        for agent, domains in expectations:
            instructions = agent.instructions(SimpleNamespace(context=context), agent)
            self.assertIn("смежные гипотезы", instructions)
            for domain in domains:
                self.assertIn(domain, instructions)

    def test_domain_prompt_budget_stays_compact(self):
        agents = [
            build_sales_agent(),
            build_marketing_agent(),
            build_price_agent(),
            build_supply_agent(),
            build_finance_agent(),
        ]

        for agent in agents:
            instructions = agent.instructions(SimpleNamespace(context=None), agent)
            self.assertLess(len(instructions), 8000, agent.name)

    def test_domain_prompts_include_derived_feature_protocol(self):
        context = EcomRunContext(
            user_query="Почему изменились продажи?",
            intent=QueryIntent.FINANCE,
            today=date(2026, 4, 7),
            db_target="localhost:55434/wb_demo",
            chat_id="chat-1",
        )
        agents = [
            build_sales_agent(),
            build_marketing_agent(),
            build_price_agent(),
            build_supply_agent(),
            build_finance_agent(),
        ]

        for agent in agents:
            instructions = agent.instructions(SimpleNamespace(context=context), agent)
            self.assertIn(
                "гипотеза -> производный признак -> расчет по данным -> статус -> рекомендация",
                instructions,
            )
            self.assertIn("read-only SQL", instructions)
            self.assertIn("Не сохраняй признаки в БД", instructions)
            self.assertIn("cross_agent_requests", instructions)

    def test_domain_prompts_reference_formula_catalog_tools_without_raw_formulas(self):
        context = EcomRunContext(
            user_query="Проверь гипотезы по SKU",
            intent=QueryIntent.FINANCE,
            today=date(2026, 4, 7),
            db_target="localhost:55434/wb_demo",
            chat_id="chat-1",
        )
        tool_expectations = [
            (build_sales_agent(), "get_sales_formula_catalog"),
            (build_marketing_agent(), "get_marketing_formula_catalog"),
            (build_price_agent(), "get_price_formula_catalog"),
            (build_supply_agent(), "get_supply_formula_catalog"),
            (build_finance_agent(), "get_finance_formula_catalog"),
        ]
        raw_formula_names = (
            "orders_delta_pct",
            "wasted_spend_flag",
            "price_gap_pct",
            "contribution_margin_pct",
        )

        for agent, tool_name in tool_expectations:
            instructions = agent.instructions(SimpleNamespace(context=context), agent)
            self.assertIn(tool_name, instructions)
            for feature_name in raw_formula_names:
                self.assertNotIn(feature_name, instructions)
            if agent.name == "Supply Agent":
                self.assertIn("days_of_stock", instructions)
                self.assertIn("total_stock_days_for_audit", instructions)
            else:
                self.assertNotIn("days_of_stock", instructions)

    def test_sales_prompt_requires_report_snapshot_for_sales_reports(self):
        context = EcomRunContext(
            user_query="Сделай отчет по продажам",
            intent=QueryIntent.SALES,
            today=date(2026, 4, 7),
            db_target="localhost:55434/wb_demo",
            chat_id="chat-1",
        )

        instructions = build_sales_agent().instructions(
            SimpleNamespace(context=context),
            build_sales_agent(),
        )

        self.assertIn("get_sales_report_snapshot", instructions)
        self.assertIn("report_table", instructions)
        self.assertIn("sales_report_snapshot", instructions)
        self.assertIn(
            "последний день в данных к предыдущему, последние 7 дней к предыдущим 7, "
            "последние 30 дней к предыдущим 30",
            instructions,
        )

    def test_formula_catalogs_expose_domain_feature_examples(self):
        expectations = [
            (QueryIntent.SALES, "orders_delta_pct"),
            (QueryIntent.MARKETING, "wasted_spend_flag"),
            (QueryIntent.PRICE, "price_gap_pct"),
            (QueryIntent.SUPPLY, "days_of_stock"),
            (QueryIntent.FINANCE, "contribution_margin_pct"),
        ]

        for domain, feature_name in expectations:
            catalog = _get_formula_catalog(domain)
            self.assertEqual(catalog["domain"], domain.value)
            self.assertIn("usage_rule", catalog)
            self.assertIn(
                feature_name,
                [formula["name"] for formula in catalog["formulas"]],
            )

    def test_router_does_not_include_domain_feature_catalog(self):
        router = build_domain_router_agent()
        context = EcomRunContext(
            user_query="Почему изменились продажи?",
            intent=QueryIntent.SALES,
            today=date(2026, 4, 7),
            db_target="localhost:55434/wb_demo",
            chat_id="chat-1",
        )

        instructions = router.instructions(SimpleNamespace(context=context), router)

        self.assertNotIn("Общий протокол производных признаков", instructions)
        self.assertNotIn("orders_delta_pct", instructions)
        self.assertNotIn("price_gap_pct", instructions)

    def test_domain_router_instructions_include_followup_memory_rules(self):
        router = build_domain_router_agent()
        context = EcomRunContext(
            user_query="Да",
            intent=QueryIntent.FINANCE,
            today=date(2026, 4, 7),
            db_target="localhost:55434/wb_demo",
            chat_id="chat-1",
            recent_messages=[
                ChatMemoryMessage(
                    seq=1,
                    role=ChatRole.ASSISTANT,
                    content=(
                        "Если хочешь, я следующим сообщением сделаю прицельный разбор "
                        "по 5 проблемным артикулам с таблицей."
                    ),
                ),
                ChatMemoryMessage(seq=2, role=ChatRole.USER, content="Да"),
            ],
        )

        instructions = router.instructions(SimpleNamespace(context=context), router)

        self.assertIn("Chat memory", instructions)
        self.assertIn("короткое подтверждение", instructions)
        self.assertIn("не отвечай `чем помочь?`", instructions)
        self.assertIn("прицельный разбор", instructions)


class ValidatorAndRenderingTests(unittest.TestCase):
    def _finding(self, factor="Finance"):
        return DiagnosticFinding(
            factor=factor,
            observation="Observed change",
            impact="Business impact",
            cause="Observed in database",
            action="Act on the signal",
            evidence=["metric moved"],
            confidence="high",
            driver_class=DriverClass.PRIMARY_DRIVER,
        )

    def test_agent_result_requires_tables(self):
        result = AgentDiagnosticResult(
            agent_name="Finance Agent",
            summary="Finance posture changed.",
            tables=[],
            findings=[self._finding()],
        )

        violations = validate_agent_diagnostic_result(result)

        self.assertTrue(any("returned no tables" in violation for violation in violations))

    def test_diagnostic_finding_normalizes_russian_confidence(self):
        result = AgentDiagnosticResult(
            agent_name="Finance Agent",
            summary="Finance posture changed.",
            tables=[ReportTable(title="Finance", columns=["Metric"], rows=[["Profit"]])],
            findings=[
                DiagnosticFinding(
                    factor="Finance",
                    observation="Observed change",
                    impact="Business impact",
                    cause="Observed in database",
                    action="Act on the signal",
                    evidence=["metric moved"],
                    confidence="высокая",
                    driver_class=DriverClass.PRIMARY_DRIVER,
                )
            ],
        )

        self.assertEqual(result.findings[0].confidence, "high")
        self.assertEqual(validate_agent_diagnostic_result(result), [])

    def test_validator_allows_unsupported_metric_named_only_as_missing_data(self):
        finding = self._finding("Прибыль не подтверждена")
        finding.evidence = [
            "known_data_gaps: no working capital / cash-flow layer",
            "COGS доступна только в finance layer",
        ]
        result = AgentDiagnosticResult(
            agent_name="Supply Agent",
            summary="Чистая прибыль не подтверждена.",
            tables=[ReportTable(title="Supply", columns=["Metric"], rows=[["Stock"]])],
            findings=[finding],
        )

        self.assertEqual(validate_agent_diagnostic_result(result), [])

    def test_validator_still_rejects_unsupported_metric_claim(self):
        finding = self._finding("Liquidity")
        finding.observation = "Working capital improved by 20%."
        result = AgentDiagnosticResult(
            agent_name="Supply Agent",
            summary="Supply posture.",
            tables=[ReportTable(title="Supply", columns=["Metric"], rows=[["Stock"]])],
            findings=[finding],
        )

        violations = validate_agent_diagnostic_result(result)

        self.assertTrue(any("working capital" in violation for violation in violations))

    def test_negative_evidence_does_not_mask_positive_unsupported_claim(self):
        finding = self._finding("Liquidity")
        finding.observation = "Working capital improved by 20%."
        finding.evidence = ["No working capital data is available."]
        result = AgentDiagnosticResult(
            agent_name="Supply Agent",
            summary="Supply posture.",
            tables=[ReportTable(title="Supply", columns=["Metric"], rows=[["Stock"]])],
            findings=[finding],
        )

        violations = validate_agent_diagnostic_result(result)

        self.assertTrue(any("working capital" in violation for violation in violations))

    def test_same_field_missing_source_does_not_mask_positive_metric_claim(self):
        finding = self._finding("Liquidity")
        finding.observation = (
            "Working capital improved by 20%, but no source is available for it."
        )
        result = AgentDiagnosticResult(
            agent_name="Supply Agent",
            summary="Supply posture.",
            tables=[ReportTable(title="Supply", columns=["Metric"], rows=[["Stock"]])],
            findings=[finding],
        )

        violations = validate_agent_diagnostic_result(result)

        self.assertTrue(any("working capital" in violation for violation in violations))

    def test_pure_negative_unsupported_metric_limitation_is_allowed(self):
        finding = self._finding("Ограничение данных")
        finding.observation = "Working capital data is not available."
        finding.evidence = ["No working capital / cash-flow layer is available."]
        result = AgentDiagnosticResult(
            agent_name="Supply Agent",
            summary="Supply posture.",
            tables=[ReportTable(title="Supply", columns=["Metric"], rows=[["Stock"]])],
            findings=[finding],
        )

        self.assertEqual(validate_agent_diagnostic_result(result), [])

    def test_render_agent_diagnostic_outputs_markdown_tables(self):
        result = AgentDiagnosticResult(
            agent_name="Marketing Agent",
            summary="Маркетинговая картина собрана.",
            tables=[ReportTable(title="Sample", columns=["Metric", "Value"], rows=[["Spend", "1200"]])],
            findings=[self._finding("Traffic")],
            recommendations=["Сфокусироваться на лучших кампаниях"],
            cross_agent_requests=[
                CrossAgentRequest(
                    target_agent=QueryIntent.FINANCE,
                    reason="Нужна проверка экономики.",
                    question="Проверь комиссии и выплаты.",
                )
            ],
        )

        rendered = render_agent_diagnostic(result)

        self.assertIn("# Marketing Agent", rendered)
        self.assertIn("**Sample**", rendered)
        self.assertIn("Выводы:", rendered)
        self.assertNotIn("Действие:", rendered)
        self.assertNotIn("Act on the signal", rendered)
        self.assertIn("Рекомендации:", rendered)
        self.assertNotIn("cross_agent_requests", rendered)
        self.assertNotIn("Проверь комиссии", rendered)

    def test_render_agent_diagnostic_adds_plus_for_delta_columns(self):
        result = AgentDiagnosticResult(
            agent_name="Sales Agent",
            summary="Продажи разобраны.",
            tables=[
                ReportTable(
                    title="Delta Sample",
                    columns=["Артикул", "Выручка", "Δ выручки", "Изменение"],
                    rows=[
                        ["46313", 31054.43, 28810.3, "-88"],
                        ["46227", "262893.69", "60576.19", 0],
                    ],
                )
            ],
            findings=[self._finding("Sales")],
        )

        rendered = render_agent_diagnostic(result)

        self.assertIn("| 46313 | 31054.43 | +28810.3 | -88 |", rendered)
        self.assertIn("| 46227 | 262893.69 | +60576.19 | 0 |", rendered)

    def test_sales_decomposition_rejects_empty_delta_cells(self):
        result = SalesDiagnosticResult(
            agent_name="Sales Agent",
            summary="Продажи разобраны.",
            sales_overview=ReportTable(
                title="Обзор",
                columns=["Метрика", "Значение"],
                rows=[["Выручка", 31054.43]],
            ),
            sales_dynamics=ReportTable(
                title="Динамика",
                columns=["Период", "Выручка"],
                rows=[["Текущий", 31054.43]],
            ),
            sales_decomposition=ReportTable(
                title="Вклад артикулов в изменение выручки",
                columns=[
                    "Артикул",
                    "Текущая выручка",
                    "Предыдущая выручка",
                    "Δ выручки",
                    "Текущие продажи",
                    "Предыдущие продажи",
                    "Δ продаж",
                ],
                rows=[
                    ["46313", 31054.43, None, "—", 12, 20, -8],
                ],
            ),
            findings=[self._finding("Sales")],
        )

        violations = validate_sales_diagnostic_result(result)

        self.assertTrue(any("previous_revenue" in violation for violation in violations))
        self.assertTrue(any("delta_revenue" in violation for violation in violations))

    @patch("agi_ecom_app.main_agent.strategic_db.get_sales_report_snapshot")
    def test_sales_report_request_backfills_snapshot(self, mock_snapshot):
        mock_snapshot.return_value = {
            "report_table": {
                "title": "Sales-сводка",
                "columns": [
                    "Сравнение",
                    "Периоды",
                    "Продажи тек./пред.",
                    "Δ продаж",
                    "Выручка тек./пред.",
                    "Δ выручки",
                ],
                "rows": [
                    ["Вчера к позавчера", "2026-04-06 к 2026-04-05", "10 / 8", 2, "1000 ₽ / 800 ₽", "200 ₽"],
                    ["Неделя к неделе", "2026-03-31..2026-04-06 к 2026-03-24..2026-03-30", "70 / 100", -30, "7000 ₽ / 10000 ₽", "-3000 ₽"],
                    ["Месяц к месяцу", "2026-03-08..2026-04-06 к 2026-02-06..2026-03-07", "300 / 200", 100, "30000 ₽ / 25000 ₽", "5000 ₽"],
                ],
            }
        }
        result = SalesDiagnosticResult(
            agent_name="Sales Agent",
            summary="Продажи разобраны.",
            sales_overview=ReportTable(
                title="Обзор",
                columns=["Метрика", "Значение"],
                rows=[["Выручка", 31054.43]],
            ),
            sales_dynamics=ReportTable(
                title="Динамика",
                columns=["Период", "Выручка"],
                rows=[["Текущий", 31054.43]],
            ),
            sales_decomposition=ReportTable(
                title="Вклад артикулов в изменение выручки",
                columns=[
                    "Артикул",
                    "Текущая выручка",
                    "Предыдущая выручка",
                    "Δ выручки",
                    "Текущие продажи",
                    "Предыдущие продажи",
                    "Δ продаж",
                ],
                rows=[["46313", 31054.43, 0, 31054.43, 12, 0, 12]],
            ),
            findings=[self._finding("Sales")],
        )

        updated = ensure_sales_report_snapshot(result, user_query="Сделай отчет по продажам")

        self.assertIsNotNone(updated.sales_report_snapshot)
        self.assertEqual(updated.sales_report_snapshot.rows[0][0], "Вчера к позавчера")
        self.assertEqual(validate_sales_diagnostic_result(updated, user_query="Сделай отчет по продажам"), [])

    @patch("agi_ecom_app.main_agent.strategic_db.get_marketing_report_snapshot")
    def test_marketing_report_request_backfills_snapshot(self, mock_snapshot):
        mock_snapshot.return_value = {
            "report_table": {
                "title": "Marketing-сводка",
                "columns": [
                    "Сравнение",
                    "Периоды",
                    "Расход тек./пред.",
                    "Выручка SKU тек./пред.",
                    "ДРР тек./пред.",
                ],
                "rows": [
                    ["Вчера к позавчера", "2026-04-06 к 2026-04-05", "1000 ₽ / 500 ₽", "10000 ₽ / 10000 ₽", "10.00% / 5.00%"],
                    ["Неделя к неделе", "2026-03-31 - 2026-04-06 к 2026-03-24 - 2026-03-30", "7000 ₽ / 5000 ₽", "35000 ₽ / 50000 ₽", "20.00% / 10.00%"],
                    ["Месяц к месяцу", "2026-03-08 - 2026-04-06 к 2026-02-06 - 2026-03-07", "30000 ₽ / 20000 ₽", "150000 ₽ / 200000 ₽", "20.00% / 10.00%"],
                ],
            }
        }
        result = MarketingDiagnosticResult(
            agent_name="Marketing Agent",
            summary="Реклама разобрана.",
            tables=[
                ReportTable(
                    title="Реклама",
                    columns=["Метрика", "Значение"],
                    rows=[["ДРР", "10%"]],
                )
            ],
            findings=[self._finding("Marketing")],
        )

        updated = ensure_marketing_report_snapshot(result, user_query="Сделай маркетинговый отчет")

        self.assertIsNotNone(updated.marketing_report_snapshot)
        self.assertEqual(updated.marketing_report_snapshot.rows[0][0], "Вчера к позавчера")
        self.assertEqual(
            validate_marketing_diagnostic_result(updated, user_query="Сделай маркетинговый отчет"),
            [],
        )

    def test_period_scoped_snapshot_does_not_require_legacy_three_period_report(self):
        weekly_snapshot = ReportTable(
            title="Последние 7 дней",
            columns=["Период", "Продажи, шт.", "Выручка, руб."],
            rows=[["2026-07-06..2026-07-12", 70, 7_000]],
        )
        sales = SalesDiagnosticResult(
            agent_name="Sales Agent",
            summary="Проверен запрошенный недельный период.",
            sales_report_snapshot=weekly_snapshot,
            sales_overview=ReportTable(
                title="Обзор",
                columns=["Метрика", "Значение"],
                rows=[["Продажи, шт.", 70]],
            ),
            sales_dynamics=ReportTable(
                title="Динамика",
                columns=["Период", "Продажи, шт."],
                rows=[["2026-07-06..2026-07-12", 70]],
            ),
            sales_decomposition=ReportTable(
                title="Вклад SKU",
                columns=[
                    "SKU",
                    "Текущая выручка",
                    "Предыдущая выручка",
                    "Изменение выручки",
                    "Текущие продажи",
                    "Предыдущие продажи",
                    "Изменение продаж",
                ],
                rows=[["46227", 7_000, 6_000, 1_000, 70, 60, 10]],
            ),
            findings=[self._finding("Sales")],
        )
        marketing = MarketingDiagnosticResult(
            agent_name="Marketing Agent",
            summary="Проверен запрошенный недельный период.",
            marketing_report_snapshot=ReportTable(
                title="Последние 7 дней",
                columns=["Период", "Расход, руб.", "ДРР, %"],
                rows=[["2026-07-06..2026-07-12", 1_000, 14.3]],
            ),
            tables=[
                ReportTable(
                    title="Реклама",
                    columns=["Метрика", "Значение"],
                    rows=[["ДРР, %", 14.3]],
                )
            ],
            findings=[self._finding("Marketing")],
        )

        sales_violations = validate_sales_diagnostic_result(
            sales,
            user_query="Покажи изменения продаж за последние 7 дней",
        )
        marketing_violations = validate_marketing_diagnostic_result(
            marketing,
            user_query="Покажи рекламные показатели за последние 7 дней",
        )

        self.assertEqual(sales_violations, [])
        self.assertEqual(marketing_violations, [])

    def test_marketing_malformed_rows_are_quarantined_with_snapshot_fallback(self):
        result = MarketingDiagnosticResult(
            agent_name="Marketing Agent",
            summary="Проверены рекламные метрики.",
            marketing_report_snapshot=ReportTable(
                title="Последние 7 дней",
                columns=["Период", "Расход, руб.", "ДРР, %"],
                rows=[["2026-07-06..2026-07-12", 1_000, 14.3]],
            ),
            tables=[
                ReportTable(
                    title="Повреждённая детализация",
                    columns=["SKU", "Расход", "Клики"],
                    rows=[["46227", 1_000, 50, "лишняя ячейка"]],
                )
            ],
            findings=[self._finding("Marketing")],
        )

        updated = ensure_marketing_report_snapshot(
            result,
            user_query="Почему изменился ДРР за последние 7 дней?",
        )

        self.assertEqual(len(updated.tables), 1)
        self.assertEqual(updated.tables[0].title, "Последние 7 дней")
        self.assertEqual(updated.data_gaps[0].metric, "marketing_table_shape")
        self.assertEqual(
            validate_marketing_diagnostic_result(
                updated,
                user_query="Почему изменился ДРР за последние 7 дней?",
            ),
            [],
        )

    def test_sales_report_snapshot_renders_first(self):
        result = SalesDiagnosticResult(
            agent_name="Sales Agent",
            summary="Продажи разобраны.",
            sales_report_snapshot=ReportTable(
                title="Sales-сводка",
                columns=[
                    "Сравнение",
                    "Периоды",
                    "Продажи тек./пред.",
                    "Δ продаж",
                    "Выручка тек./пред.",
                    "Δ выручки",
                ],
                rows=[
                    ["Вчера к позавчера", "2026-04-06 к 2026-04-05", "10 / 8", 2, "1000 ₽ / 800 ₽", "200 ₽"],
                    ["Неделя к неделе", "2026-03-31..2026-04-06 к 2026-03-24..2026-03-30", "70 / 100", -30, "7000 ₽ / 10000 ₽", "-3000 ₽"],
                    ["Месяц к месяцу", "2026-03-08..2026-04-06 к 2026-02-06..2026-03-07", "300 / 200", 100, "30000 ₽ / 25000 ₽", "5000 ₽"],
                ],
            ),
            sales_overview=ReportTable(
                title="Обзор",
                columns=["Метрика", "Значение"],
                rows=[["Выручка", 31054.43]],
            ),
            sales_dynamics=ReportTable(
                title="Динамика",
                columns=["Период", "Выручка"],
                rows=[["Текущий", 31054.43]],
            ),
            sales_decomposition=ReportTable(
                title="Вклад артикулов в изменение выручки",
                columns=[
                    "Артикул",
                    "Текущая выручка",
                    "Предыдущая выручка",
                    "Δ выручки",
                    "Текущие продажи",
                    "Предыдущие продажи",
                    "Δ продаж",
                ],
                rows=[["46313", 31054.43, 0, 31054.43, 12, 0, 12]],
            ),
            findings=[self._finding("Sales")],
        )

        rendered = render_sales_diagnostic(result)

        self.assertTrue(rendered.startswith("**Sales-сводка**"))
        self.assertLess(rendered.index("**Sales-сводка**"), rendered.index("# Sales Agent"))
        self.assertEqual(validate_sales_diagnostic_result(result, user_query="Сделай отчет по продажам"), [])

    @patch("agi_ecom_app.main_agent.strategic_db.get_sales_vendor_code_map")
    def test_sales_user_facing_contract_replaces_nm_ids_and_internal_terms(self, mock_map):
        mock_map.return_value = {"70194686": "46227", "768260062": "46313"}
        result = SalesDiagnosticResult(
            agent_name="Sales Agent",
            summary="Продажи разобраны.",
            sales_overview=ReportTable(
                title="Обзор",
                columns=["Метрика", "Значение"],
                rows=[["Выручка", "2 455 032 ₽"]],
            ),
            sales_dynamics=ReportTable(
                title="Лидеры",
                columns=["SKU/артикул", "Метрика", "Значение"],
                rows=[["70194686", "Выручка", "2 455 032 ₽"]],
            ),
            sales_decomposition=ReportTable(
                title="Вклад артикулов в изменение выручки",
                columns=[
                    "SKU/артикул",
                    "Текущая выручка",
                    "Предыдущая выручка",
                    "Δ выручки",
                    "Текущие продажи",
                    "Предыдущие продажи",
                    "Δ продаж",
                ],
                rows=[["768260062", 31054.43, 0, 31054.43, 12, 0, 12]],
            ),
            findings=[
                DiagnosticFinding(
                    factor="SKU 70194686",
                    observation="SKU 70194686 снизился.",
                    impact="Влияет на продажи.",
                    cause="Механическое снижение реализации.",
                    action="Проверить price, marketing и supply по SKU 70194686.",
                    evidence=["70194686: -113 230 ₽"],
                    confidence="high",
                    driver_class=DriverClass.PRIMARY_DRIVER,
                )
            ],
            recommendations=["Проверить на price, marketing и supply SKU 70194686."],
        )

        updated = enforce_sales_user_facing_contract(result)

        self.assertEqual(updated.sales_dynamics.rows[0][0], "46227")
        self.assertEqual(updated.sales_decomposition.rows[0][0], "46313")
        self.assertEqual(updated.sales_overview.rows[0][1], "2 455 032 ₽")
        self.assertIn("46227", updated.findings[0].observation)
        self.assertNotIn("70194686", updated.findings[0].observation)
        self.assertEqual(
            updated.findings[0].cause,
            "Sales-данные подтверждают изменение метрик; первопричина требует смежной проверки.",
        )
        self.assertIn("цену, рекламу и остатки", updated.recommendations[0])
        self.assertEqual(validate_sales_diagnostic_result(updated), [])

    @patch("agi_ecom_app.main_agent.strategic_db.get_sales_sku_revenue_change")
    def test_sales_decomposition_contract_backfills_required_columns(self, mock_change):
        mock_change.return_value = {
            "rows": [
                {
                    "vendor_code": "46227",
                    "brand_name": "LIFECODE",
                    "subject_name": "Сыворотки",
                    "current_revenue": 1000,
                    "previous_revenue": 800,
                    "delta_revenue": 200,
                    "current_sales": 10,
                    "previous_sales": 8,
                    "delta_sales": 2,
                    "driver_role": "growth_driver",
                }
            ]
        }
        result = SalesDiagnosticResult(
            agent_name="Sales Agent",
            summary="Продажи разобраны.",
            sales_overview=ReportTable(
                title="Обзор",
                columns=["Метрика", "Значение"],
                rows=[["Выручка", 1000]],
            ),
            sales_dynamics=ReportTable(
                title="Динамика",
                columns=["Период", "Выручка"],
                rows=[["Текущий", 1000]],
            ),
            sales_decomposition=ReportTable(
                title="Сжатая декомпозиция",
                columns=["SKU", "Δ выручки"],
                rows=[["46227", 200]],
            ),
            findings=[self._finding("Sales")],
        )

        updated = ensure_sales_decomposition_contract(result)

        self.assertIn("Текущая выручка", updated.sales_decomposition.columns)
        self.assertIn("Предыдущие продажи", updated.sales_decomposition.columns)
        self.assertEqual(updated.sales_decomposition.rows[0][0], "46227")
        self.assertEqual(validate_sales_diagnostic_result(updated), [])


if __name__ == "__main__":
    unittest.main()
