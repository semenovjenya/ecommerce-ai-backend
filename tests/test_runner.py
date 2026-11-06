from __future__ import annotations

import json
import unittest
from datetime import date
from types import SimpleNamespace
from unittest.mock import ANY, patch

from agents import OutputGuardrailTripwireTriggered

from agi_ecom_app.llm_routing import DomainRouteDecision
from agi_ecom_app.runner import (
    AgentRuntimeError,
    MAX_DIAGNOSTIC_ROUNDS,
    MAX_TOTAL_FOLLOWUP_DOMAIN_CALLS,
    OPTIMIZED_SECONDARY_QUESTION_BUDGET,
    ROUND_3_GAP_RESOLUTION_LIMIT,
    _attach_tool_report,
    _apply_supply_domain_authority,
    _build_synthesis_package,
    _build_secondary_task_prompt,
    _build_sku_evidence_matrix,
    _diagnostic_payload,
    _deduplicate_optimized_gap_records,
    _domain_tool_policy,
    _enforce_gap_resolution_fact_policy,
    _gap_entries_from_payload,
    _gap_entries_from_sku_matrix,
    _inventory_turnover_coverage_plan,
    _load_persisted_evidence_safely,
    _claim_verification_plan,
    _prepend_marketing_report_snapshot,
    _prepend_sales_report_snapshot,
    _persist_agent_run_response_safely,
    _reduce_evidence,
    _render_prompt_v2_facts_only_answer,
    remove_empty_markdown_tables,
    _run_agent,
    _sanitize_inventory_turnover_cause_claims,
    _select_confirmed_facts_for_query,
    _strip_unrequested_sku_cause_matrix,
    _sku_only_display_was_sanitized,
    _typed_fact_for_synthesis,
    _v2_required_tools_for_query,
    classify_gap_resolution_response,
    run_chat_turn,
    run_main_agent,
    sanitize_user_facing_markdown,
    should_persist_gap_from_gap_resolution,
    validate_final_diagnostic_markdown,
)
from agi_ecom_app.prompt_pipeline import compact_synthesis_package
from agi_ecom_app.tools.strategic_tools import build_cross_domain_dependency_plan
from agi_ecom_app.schemas import (
    AgentDiagnosticResult,
    ChatContextWindow,
    ChatMemoryMessage,
    ChatRole,
    CrossAgentRequest,
    DataGap,
    DiagnosticFinding,
    DriverClass,
    EcomRunContext,
    MarketingDiagnosticResult,
    QueryIntent,
    ReportTable,
    RoundType,
    SalesDiagnosticResult,
    WorkflowVersion,
)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        # Snapshot enrichment reads this source independently of the mocked LLM.
        # Keep the boundary deterministic and avoid connecting to a seller DB.
        source = patch(
            "agi_ecom_app.main_agent.strategic_db.get_sales_sku_revenue_change",
            return_value={"rows": [{
                "vendor_code": "DEMO-001", "brand_name": "Demo", "subject_name": "Demo category",
                "current_revenue": 1000, "previous_revenue": 800, "delta_revenue": 200,
                "current_sales": 10, "previous_sales": 8, "delta_sales": 2,
                "driver_role": "growth",
            }]},
        )
        source.start()
        self.addCleanup(source.stop)

    def _settings(self):
        return SimpleNamespace(
            openai_api_key="test-key",
            openai_model="gpt-test",
            openai_memory_model="gpt-test-mini",
            db_host="localhost",
            db_port=55434,
            db_name="wb_demo",
            memory_context_message_limit=12,
        )

    def _start_mocks(
        self,
        user_input: str,
        intent: QueryIntent,
        output="Ответ собран.",
        has_domain_signal: bool = True,
        route_intent: QueryIntent | None = None,
        tool_trace: list[str] | None = None,
    ):
        routed_intent = route_intent or intent
        patches = [
            patch("agi_ecom_app.runner.get_settings", return_value=self._settings()),
            patch("agi_ecom_app.runner.determine_query_intent", return_value=intent),
            patch("agi_ecom_app.runner.has_domain_routing_signal", return_value=has_domain_signal),
            patch(
                "agi_ecom_app.runner.build_diagnostic_orchestrator_agent",
                return_value="diagnostic-orchestrator",
            ),
            patch("agi_ecom_app.runner.build_sales_agent", return_value="sales-agent"),
            patch("agi_ecom_app.runner.build_marketing_agent", return_value="marketing-agent"),
            patch("agi_ecom_app.runner.build_supply_agent", return_value="supply-agent"),
            patch("agi_ecom_app.runner.build_finance_agent", return_value="finance-agent"),
            patch("agi_ecom_app.runner.build_domain_router_agent", return_value="domain-router-agent"),
            patch("agi_ecom_app.runner.build_run_hooks", return_value="hooks"),
            patch("agi_ecom_app.runner.Runner.run_sync"),
            patch("agi_ecom_app.runner.ensure_chat_session"),
            patch(
                "agi_ecom_app.runner.get_context_window",
                return_value=ChatContextWindow(
                    chat_id="chat-1",
                    summary_text=None,
                    recent_messages=[],
                ),
            ),
            patch(
                "agi_ecom_app.runner.append_user_message_idempotent",
                return_value=(
                    ChatMemoryMessage(seq=1, role=ChatRole.USER, content=user_input),
                    True,
                ),
            ),
            patch("agi_ecom_app.runner.append_message"),
            patch("agi_ecom_app.runner.refresh_chat_summary"),
            patch("agi_ecom_app.runner.build_price_agent", return_value="price-agent"),
            patch("agi_ecom_app.runner.create_agent_run_trace"),
            patch("agi_ecom_app.runner.complete_agent_run_trace"),
            patch("agi_ecom_app.runner.persist_agent_action_tasks_for_message"),
            patch("agi_ecom_app.runner.create_agent_run_round"),
            patch("agi_ecom_app.runner.update_agent_run_round_status"),
            patch("agi_ecom_app.runner.create_agent_run_question"),
            patch("agi_ecom_app.runner.update_agent_run_question_status"),
            patch("agi_ecom_app.runner.save_agent_run_response"),
            patch("agi_ecom_app.runner.save_agent_run_gaps"),
            patch("agi_ecom_app.runner.list_agent_run_gaps"),
            patch("agi_ecom_app.runner.list_agent_run_responses"),
            patch("agi_ecom_app.runner.update_agent_run_gap_status"),
            patch(
                "agi_ecom_app.runner.route_query_intent_with_llm",
                return_value=DomainRouteDecision(
                    route_to_domain=has_domain_signal,
                    intent=routed_intent,
                    confidence="high",
                    reason="test route",
                    supporting_agents=[],
                ),
            ),
        ]
        started = [mock.start() for mock in patches]

        mock_run_sync = started[10]
        started[14].return_value = ChatMemoryMessage(seq=2, role=ChatRole.ASSISTANT, content="Ответ")
        started[17].return_value = SimpleNamespace(id=101)
        started[20].side_effect = [
            SimpleNamespace(id=index, run_id=101, round_number=index, round_type="test", status="running")
            for index in range(1, 20)
        ]
        started[22].side_effect = [
            SimpleNamespace(id=index, run_id=101, round_id=1, target_agent="test", question="q")
            for index in range(1, 40)
        ]
        started[24].side_effect = [
            SimpleNamespace(id=index, question_id=index)
            for index in range(1, 40)
        ]
        started[25].return_value = []
        started[26].return_value = []
        started[27].return_value = []
        output_queue = list(output) if isinstance(output, list) else None

        def _run_sync(_, __, **kwargs):
            if tool_trace:
                kwargs["context"].tool_trace.extend(tool_trace)
            final_output = output_queue.pop(0) if output_queue is not None else output
            return SimpleNamespace(final_output=final_output)

        mock_run_sync.side_effect = _run_sync
        return patches, started

    def _finding(self, factor="Sales", observation="Metric moved."):
        return DiagnosticFinding(
            factor=factor,
            observation=observation,
            impact="Business impact",
            cause="Observed in tools",
            action="Check",
            evidence=["metric moved"],
            confidence="high",
            driver_class=DriverClass.PRIMARY_DRIVER,
        )

    def _sales_output(self):
        return SalesDiagnosticResult(
            agent_name="Sales Agent",
            summary="Продажи изменились, нужен supply-check.",
            sales_overview=ReportTable(title="Обзор", columns=["Метрика", "Значение"], rows=[["Продажи", 10]]),
            sales_dynamics=ReportTable(title="Динамика", columns=["Дата", "Продажи"], rows=[["2026-04-01", 10]]),
            sales_decomposition=ReportTable(
                title="Вклад артикулов в изменение выручки",
                columns=["Артикул", "Текущая выручка", "Предыдущая выручка", "Δ выручки"],
                rows=[["46313", 1000, 800, 200]],
            ),
            findings=[self._finding("Sales", "Есть риск out-of-stock по ключевому артикулу.")],
            cross_agent_requests=[
                CrossAgentRequest(
                    target_agent=QueryIntent.SUPPLY,
                    reason="Нужно проверить доступность.",
                    question="Проверь остатки.",
                )
            ],
        )

    def _neutral_sales_output(self):
        return SalesDiagnosticResult(
            agent_name="Sales Agent",
            summary="Продажи за период собраны.",
            sales_overview=ReportTable(title="Обзор", columns=["Метрика", "Значение"], rows=[["Продажи", 10]]),
            sales_dynamics=ReportTable(title="Динамика", columns=["Дата", "Продажи"], rows=[["2026-04-01", 10]]),
            sales_decomposition=ReportTable(
                title="Вклад артикулов в изменение выручки",
                columns=["Артикул", "Текущая выручка", "Предыдущая выручка", "Δ выручки"],
                rows=[["46313", 1000, 800, 200]],
            ),
        )

    def _sales_snapshot_table(self):
        return ReportTable(
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
                ["Вчера к позавчера", "2026-04-19 к 2026-04-18", "15 / 13", "+2 (+15.4%)", "38 827 ₽ / 35 615 ₽", "+3 212 ₽ (+9.0%)"],
                ["Неделя к неделе", "2026-04-13..2026-04-19 к 2026-04-06..2026-04-12", "90 / 84", "+6 (+7.1%)", "220 000 ₽ / 210 000 ₽", "+10 000 ₽ (+4.8%)"],
                ["Месяц к месяцу", "2026-03-21..2026-04-19 к 2026-02-19..2026-03-20", "300 / 280", "+20 (+7.1%)", "700 000 ₽ / 650 000 ₽", "+50 000 ₽ (+7.7%)"],
            ],
        )

    def _marketing_snapshot_table(self):
        return ReportTable(
            title="Marketing-сводка",
            columns=[
                "Сравнение",
                "Периоды",
                "Расход тек./пред.",
                "Выручка SKU тек./пред.",
                "ДРР тек./пред.",
            ],
            rows=[
                ["Вчера к позавчера", "2026-04-19 к 2026-04-18", "1 000 ₽ / 500 ₽", "10 000 ₽ / 10 000 ₽", "10.00% / 5.00%"],
                ["Неделя к неделе", "2026-04-13..2026-04-19 к 2026-04-06..2026-04-12", "7 000 ₽ / 5 000 ₽", "35 000 ₽ / 50 000 ₽", "20.00% / 10.00%"],
                ["Месяц к месяцу", "2026-03-21..2026-04-19 к 2026-02-19..2026-03-20", "30 000 ₽ / 20 000 ₽", "150 000 ₽ / 200 000 ₽", "20.00% / 10.00%"],
            ],
        )

    def _tool_agent(self, *tool_names):
        return SimpleNamespace(tools=[SimpleNamespace(name=name) for name in tool_names])

    def _context(self, user_query="почему упали продажи", intent=QueryIntent.SALES):
        return EcomRunContext(
            user_query=user_query,
            intent=intent,
            today=date(2026, 4, 21),
            db_target="localhost:55434/wb_demo",
            chat_id="chat-1",
        )

    def _append_tool_trace(self, context, tool_name, duration_ms=7):
        context.tool_trace.append(tool_name)
        context.add_decision_trace("tool_start", tool=tool_name, status="started")
        context.add_decision_trace("tool_end", tool=tool_name, status="completed", duration_ms=duration_ms)

    def test_domain_tool_policy_tracks_formula_catalog_without_requiring_it(self):
        expectations = {
            QueryIntent.SALES: "get_sales_formula_catalog",
            QueryIntent.MARKETING: "get_marketing_formula_catalog",
            QueryIntent.PRICE: "get_price_formula_catalog",
            QueryIntent.SUPPLY: "get_supply_formula_catalog",
            QueryIntent.FINANCE: "get_finance_formula_catalog",
        }

        for intent, formula_tool in expectations.items():
            policy = _domain_tool_policy(
                target=intent,
                user_query="почему изменились показатели",
                round_name="primary",
                question_id=11,
            )
            self.assertEqual(policy.required_tools, ())
            self.assertEqual(policy.formula_tool, formula_tool)

    def test_sales_report_policy_does_not_require_snapshot_or_formula_catalog(self):
        policy = _domain_tool_policy(
            target=QueryIntent.SALES,
            user_query="sales отчет вчера к позавчера неделя к неделе месяц к месяцу",
            round_name="primary",
            question_id=12,
        )

        self.assertEqual(policy.required_tools, ())
        self.assertEqual(policy.formula_tool, "get_sales_formula_catalog")

    def test_marketing_report_policy_does_not_require_snapshot_or_formula_catalog(self):
        policy = _domain_tool_policy(
            target=QueryIntent.MARKETING,
            user_query="маркетинговый отчет вчера к позавчера неделя к неделе месяц к месяцу с ДРР",
            round_name="primary",
            question_id=13,
        )

        self.assertEqual(policy.required_tools, ())
        self.assertEqual(policy.formula_tool, "get_marketing_formula_catalog")

    def test_price_decision_policy_requires_decision_overview(self):
        policy = _domain_tool_policy(
            target=QueryIntent.PRICE,
            user_query="Какую цену на товар с низкой оборачиваемостью мы можем поставить?",
            round_name="primary",
            question_id=14,
        )

        self.assertEqual(policy.required_tools, ("get_price_decision_overview",))
        self.assertEqual(policy.formula_tool, "get_price_formula_catalog")

    def test_price_decision_policy_ignores_marketing_bid_queries(self):
        policy = _domain_tool_policy(
            target=QueryIntent.PRICE,
            user_query="Покажи цену клика и ставки по рекламе",
            round_name="primary",
            question_id=15,
        )

        self.assertEqual(policy.required_tools, ())

    def test_v2_supply_derived_metrics_require_deterministic_calculators(self):
        required = _v2_required_tools_for_query(
            QueryIntent.SUPPLY,
            "Посчитай оборачиваемость склада и деньги в текущих остатках.",
        )

        self.assertEqual(
            required,
            (
                "get_inventory_turnover",
                "get_current_inventory_value",
                "get_supply_overview",
            ),
        )

    def test_v2_supply_lookup_requires_aggregated_stock_tool(self):
        self.assertEqual(
            _v2_required_tools_for_query(
                QueryIntent.SUPPLY,
                "Какой текущий доступный остаток по SKU 46227?",
            ),
            ("get_supply_overview",),
        )

    def test_price_floor_gap_rejects_a_single_profitable_scenario(self):
        payload = {
            "summary": "Текущая цена 4774 прибыльна, значит это минимум.",
            "_round_result": {
                "round_type": "gap_resolution",
                "gap_id": 7,
                "verdict": "resolved",
                "facts": [
                    {
                        "metric": "minimum profitable price",
                        "value": 4774,
                        "unit": "RUB",
                        "period": "2026-04-06",
                        "evidence_ref": "finance_run_sql:2",
                    }
                ],
            },
            "findings": [{"factor": "minimum profitable price"}],
            "tables": [{"title": "scenario", "columns": ["price"], "rows": [[4774]]}],
        }
        original_gap = SimpleNamespace(metric="minimum profitable price")

        result = _enforce_gap_resolution_fact_policy(payload, original_gap)

        self.assertEqual(result["_round_result"]["verdict"], "no_data")
        self.assertEqual(result["_round_result"]["facts"], [])
        self.assertEqual(result["gap_resolution_status"], "no_data")
        self.assertEqual(result["findings"], [])
        self.assertEqual(result["tables"], [])

    def test_price_floor_gap_accepts_dedicated_floor_evidence(self):
        payload = {
            "summary": "Порог подтверждён.",
            "_round_result": {
                "round_type": "gap_resolution",
                "gap_id": 7,
                "verdict": "resolved",
                "facts": [
                    {
                        "metric": "minimum profitable price",
                        "value": 4060,
                        "unit": "RUB",
                        "period": "2026-07-13",
                        "evidence_ref": "tool:get_price_floor.minimum_profitable_price",
                    }
                ],
            },
        }

        result = _enforce_gap_resolution_fact_policy(
            payload,
            SimpleNamespace(metric="minimum profitable price"),
        )

        self.assertIs(result, payload)

    def test_synthesis_fact_selection_keeps_query_critical_secondary_metrics(self):
        primary = {
            "facts": [
                {
                    "metric": f"generic_finance_metric_{index}",
                    "value": index,
                    "unit": "RUB",
                    "period": "2026-07-01/2026-07-13",
                    "status": "confirmed",
                }
                for index in range(12)
            ],
            "tables": [],
            "findings": [],
            "data_gaps": [],
        }
        secondary = {
            "round": "cross_check",
            "target_agent": "supply",
            "output": {
                "_round_result": {
                    "verdict": "confirmed",
                    "facts": [
                        {
                            "metric": "current_inventory_value_total",
                            "value": 438767,
                            "unit": "RUB",
                            "period": "2026-07-08",
                            "status": "confirmed",
                        },
                        {
                            "metric": "turnover_ratio",
                            "value": 0.1482,
                            "unit": "x",
                            "period": "2026-07-01/2026-07-13",
                            "status": "confirmed",
                        },
                    ],
                },
                "tables": [],
                "findings": [],
                "data_gaps": [],
            },
        }

        reduced = _reduce_evidence(
            primary_agent=QueryIntent.FINANCE,
            primary_output=primary,
            round_1_evidence=[secondary],
            followup_evidence=[],
            user_input="Посчитай оборачиваемость и деньги в текущих остатках.",
        )

        metrics = [fact["metric"] for fact in reduced["confirmed_facts"]]
        self.assertEqual(metrics[:2], ["current_inventory_value_total", "turnover_ratio"])

    def test_supply_calculators_shadow_conflicting_finance_stock_evidence(self):
        primary = {
            "facts": [
                {
                    "metric": "current_stock_value_confirmed",
                    "value": 17460,
                    "unit": "RUB",
                    "period": "2026-07-13 snapshot",
                    "status": "confirmed",
                    "source_table": "finance.wb_report_detail_daily",
                },
                {
                    "metric": "payout_30d",
                    "value": 1415172.24,
                    "unit": "RUB",
                    "period": "2026-06-14/2026-07-13",
                    "status": "confirmed",
                },
            ],
            "tables": [
                {
                    "title": "Текущие остатки и стоимость остатка по SKU",
                    "columns": ["SKU", "current_stock", "stock_value_rub", "contribution_margin_rub"],
                    "rows": [["46227", 8, 2752, 999]],
                },
                {
                    "title": "Финансовый срез",
                    "columns": ["metric", "value"],
                    "rows": [["payout", 1415172.24]],
                },
                {
                    "title": "Ключевые показатели WB",
                    "columns": ["Показатель", "Значение", "Ед."],
                    "rows": [
                        ["Подтвержденная стоимость текущих остатков", 17460, "RUB"],
                        ["Нетто-выручка", 31873.61, "RUB"],
                    ],
                },
            ],
            "findings": [
                {
                    "factor": "Стоимость текущих остатков",
                    "observation": "17 460 RUB при 43 единицах остатка.",
                },
                {
                    "factor": "Комиссии",
                    "observation": "Комиссии подтверждены.",
                },
                {
                    "factor": "Contribution margin",
                    "observation": "999 RUB сопоставлена со стоимостью текущих остатков.",
                },
            ],
            "data_gaps": [],
        }
        supply = {
            "round": "cross_check",
            "target_agent": "supply",
            "output": {
                "_round_result": {
                    "verdict": "confirmed",
                    "facts": [
                        {
                            "metric": "inventory_turnover_ratio",
                            "value": 0.4629,
                            "unit": "ratio",
                            "period": "2026-06-13/2026-07-13",
                            "status": "confirmed",
                            "evidence_ref": "get_inventory_turnover.totals",
                        },
                        {
                            "metric": "confirmed_current_inventory_value",
                            "value": 438767,
                            "unit": "RUB",
                            "period": "2026-07-08 snapshot",
                            "status": "confirmed",
                            "evidence_ref": "get_current_inventory_value.totals",
                        },
                    ],
                },
                "tables": [],
                "findings": [],
                "data_gaps": [],
            },
        }

        package = _build_synthesis_package(
            user_input="Посчитай оборачиваемость склада и деньги в текущих остатках.",
            primary_agent=QueryIntent.FINANCE,
            primary_output=primary,
            round_1_plan={},
            round_1_evidence=[supply],
            round_2_plan={},
            followup_evidence=[],
            persisted_blocking_gaps=[
                {
                    "source_agent": "finance",
                    "metric": "Полноценная оборачиваемость склада",
                    "reason": "Нет среднего запаса.",
                    "blocking": True,
                    "status": "open",
                },
                {
                    "source_agent": "finance",
                    "metric": "Полная прибыль после себестоимости",
                    "reason": "Нет COGS для части SKU.",
                    "blocking": True,
                    "status": "open",
                },
            ],
        )
        compact, _ = compact_synthesis_package(package)
        serialized = json.dumps(compact, ensure_ascii=False)

        self.assertIn("438767", serialized)
        self.assertNotIn("17460", serialized)
        self.assertNotIn("2752", serialized)
        self.assertIn("999", serialized)
        self.assertIn("1415172.24", serialized)
        self.assertIn("31873.61", serialized)
        for matrix in compact["evidence_tables"]:
            table = matrix.get("table") if isinstance(matrix, dict) else None
            if isinstance(table, dict):
                self.assertNotIn("current_stock", table.get("columns") or [])
                self.assertNotIn("stock_value_rub", table.get("columns") or [])
        self.assertTrue(
            any("прибыль" in str(gap.get("metric") or "").lower() for gap in package["blocking_gaps"])
        )

    def test_supply_authority_accepts_latest_stale_snapshot_but_not_future_or_other_scope(self):
        package = {
            "primary_agent": "finance",
            "confirmed_facts": [
                {
                    "source_agent": "finance",
                    "metric": "current_stock_cost_rub",
                    "value": 6427,
                    "unit": "RUB",
                    "sku": None,
                },
                {
                    "source_agent": "finance",
                    "metric": "current_stock_cost_rub",
                    "value": 2752,
                    "unit": "RUB",
                    "sku": "46227",
                },
                {
                    "source_agent": "supply",
                    "metric": "confirmed_stock_value_rub",
                    "value": 438767,
                    "unit": "RUB",
                    "sku": None,
                    "status": "confirmed",
                    "freshness_status": "future",
                    "evidence_ref": "get_current_inventory_value.totals",
                },
            ],
            "primary_evidence": {
                "facts": [],
                "findings": [],
                "data_gaps": [],
                "tables": [
                    {
                        "title": "Текущие доступные агрегаты",
                        "columns": ["Метрика", "Значение", "Ед."],
                        "rows": [
                            ["Стоимость текущих остатков", 6427, "RUB"],
                            ["Суммарная model net_profit", 23020.61, "RUB"],
                        ],
                    }
                ],
            },
            "supporting_evidence": [],
            "blocking_gaps": [],
            "non_blocking_gaps": [],
        }

        supply_fact = package["confirmed_facts"][-1]
        _apply_supply_domain_authority(package)
        self.assertEqual(
            [fact["value"] for fact in package["confirmed_facts"]],
            [6427, 2752],
        )

        supply_fact["freshness_status"] = "stale"
        package["confirmed_facts"].append(supply_fact)
        _apply_supply_domain_authority(package)
        self.assertEqual(
            [(fact["source_agent"], fact["value"], fact.get("sku")) for fact in package["confirmed_facts"]],
            [("finance", 2752, "46227"), ("supply", 438767, None)],
        )
        self.assertEqual(
            package["primary_evidence"]["tables"][0]["rows"],
            [["Суммарная model net_profit", 23020.61, "RUB"]],
        )

    def test_supply_authority_sanitizes_raw_sku_matrix_only_for_matching_scope(self):
        package = {
            "primary_agent": "finance",
            "confirmed_facts": [
                {
                    "source_agent": "supply",
                    "metric": "confirmed_stock_value_rub",
                    "value": 2752,
                    "unit": "RUB",
                    "sku": "46227",
                    "status": "confirmed",
                    "evidence_ref": "tool:get_current_inventory_value",
                }
            ],
            "primary_evidence": {"facts": [], "findings": [], "tables": []},
            "supporting_evidence": [],
            "contradictions": [],
            "sku_metric_matrix": [
                {
                    "title": "SKU-матрица",
                    "source": "sku_metric_matrix",
                    "rows": [
                        {
                            "sku": "46227",
                            "current_stock_cost_rub": 17460,
                            "contribution_margin_rub": 999,
                        },
                        {
                            "sku": "46230",
                            "current_stock_cost_rub": 8888,
                            "contribution_margin_rub": 777,
                        },
                    ],
                }
            ],
        }

        _apply_supply_domain_authority(package)
        rows = package["sku_metric_matrix"][0]["rows"]

        self.assertNotIn("current_stock_cost_rub", rows[0])
        self.assertEqual(rows[0]["contribution_margin_rub"], 999)
        self.assertEqual(rows[1]["current_stock_cost_rub"], 8888)

    def test_unconfirmed_primary_fact_is_not_promoted_to_confirmed_evidence(self):
        reduced = _reduce_evidence(
            primary_agent=QueryIntent.FINANCE,
            primary_output={
                "facts": [
                    {
                        "metric": "turnover_ratio_costed_90d",
                        "value": None,
                        "unit": "ratio",
                        "period": "last 90 days",
                        "status": "unconfirmed",
                    }
                ],
                "tables": [],
                "findings": [],
                "data_gaps": [],
            },
            round_1_evidence=[],
            followup_evidence=[],
            user_input="Посчитай оборачиваемость.",
        )

        self.assertEqual(reduced["confirmed_facts"], [])
        self.assertTrue(
            any(gap.get("metric") == "turnover_ratio_costed_90d" for gap in reduced["non_blocking_gaps"])
        )

    def test_no_data_secondary_table_is_not_synthesis_evidence(self):
        package = _build_synthesis_package(
            user_input="Покажи чистую прибыль.",
            primary_agent=QueryIntent.FINANCE,
            primary_output={"facts": [], "tables": [], "findings": [], "data_gaps": []},
            round_1_plan={},
            round_1_evidence=[
                {
                    "round": "cross_check",
                    "target_agent": "finance",
                    "output": {
                        "_round_result": {
                            "verdict": "no_data",
                            "facts": [
                                {
                                    "metric": "net_profit_rub",
                                    "value": 12345,
                                    "unit": "RUB",
                                    "period": "latest",
                                    "status": "unconfirmed",
                                }
                            ],
                        },
                        "tables": [
                            {
                                "title": "Неподтвержденный расчет",
                                "columns": ["metric", "value"],
                                "rows": [["net_profit_rub", 12345]],
                            }
                        ],
                        "findings": [],
                        "data_gaps": [],
                    },
                }
            ],
            round_2_plan={},
            followup_evidence=[],
        )
        compact, _ = compact_synthesis_package(package)

        self.assertNotIn("12345", json.dumps(compact, ensure_ascii=False))

    def test_positive_sales_turn_empty_finance_zero_aggregate_into_no_data(self):
        primary = {
            "facts": [
                {
                    "metric": "skus_with_sales_90d",
                    "value": 0,
                    "unit": "SKU",
                    "period": "last 90 days",
                    "status": "confirmed",
                    "source_table": "finance.wb_report_detail_daily",
                },
                {
                    "metric": "total_contribution_margin_90d",
                    "value": 0,
                    "unit": "RUB",
                    "period": "last 90 days",
                    "status": "confirmed",
                    "source_table": "finance.wb_report_detail_daily",
                },
            ],
            "tables": [
                {
                    "title": "Агрегаты",
                    "columns": ["Показатель", "Значение", "Ед."],
                    "rows": [["Contribution margin", 0, "RUB"]],
                },
                {
                    "title": "SKU-агрегаты",
                    "columns": ["SKU", "total_contribution_margin_90d"],
                    "rows": [["46227", 0]],
                },
            ],
            "sku_metric_matrix": [
                {"sku": "46227", "total_contribution_margin_90d": 0}
            ],
            "findings": [],
            "data_gaps": [],
        }
        sales = {
            "round": "cross_check",
            "target_agent": "sales",
            "output": {
                "_round_result": {
                    "verdict": "confirmed",
                    "facts": [
                        {
                            "metric": "sales_units",
                            "value": 127,
                            "unit": "units",
                            "period": "last 7 days",
                            "status": "confirmed",
                        }
                    ],
                },
                "tables": [],
                "findings": [],
                "data_gaps": [],
            },
        }

        package = _build_synthesis_package(
            user_input="Посчитай прибыль по SKU по доступным данным.",
            primary_agent=QueryIntent.FINANCE,
            primary_output=primary,
            round_1_plan={},
            round_1_evidence=[sales],
            round_2_plan={},
            followup_evidence=[],
        )
        compact, _ = compact_synthesis_package(package)
        serialized = json.dumps(compact, ensure_ascii=False)

        self.assertNotIn("total_contribution_margin_90d", serialized)
        self.assertNotIn('"Contribution margin", 0', serialized)
        self.assertTrue(
            any(gap.get("metric") == "finance aggregate coverage" for gap in package["blocking_gaps"])
        )

    def test_facts_only_fallback_keeps_composite_metrics_without_manager_boilerplate(self):
        rendered = _render_prompt_v2_facts_only_answer(
            "Посчитай оборачиваемость, деньги в остатках и прибыль.",
            {
                "confirmed_facts": [
                    {
                        "source_agent": "supply",
                        "metric": "confirmed_current_inventory_value",
                        "value": 438767,
                        "unit": "RUB",
                        "period": "2026-07-08",
                    },
                    {
                        "source_agent": "supply",
                        "metric": "turnover_ratio",
                        "value": 0.1482,
                        "unit": "x",
                        "period": "2026-07-01/2026-07-13",
                    },
                    {
                        "source_agent": "finance",
                        "metric": "contribution_margin_total",
                        "value": 312310.35,
                        "unit": "RUB",
                        "period": "2026-07-01/2026-07-13",
                    },
                    {
                        "source_agent": "finance",
                        "metric": "contribution_margin_note",
                        "value": "see SKU-level report",
                        "unit": "RUB",
                        "period": "2026-07-01/2026-07-13",
                    },
                    {
                        "source_agent": "finance",
                        "metric": "profit_note",
                        "value": "calculation is available in another report",
                        "unit": "RUB",
                        "period": "2026-07-01/2026-07-13",
                    },
                    {
                        "source_agent": "supply",
                        "metric": "days_on_hand",
                        "value": "64.8",
                        "unit": "days",
                        "period": "2026-07-01/2026-07-13",
                    },
                    {
                        "source_agent": "finance",
                        "metric": "turnover_ratio_costed_90d",
                        "value": None,
                        "unit": "ratio",
                        "period": "last 90 days",
                        "status": "unconfirmed",
                    },
                ],
                "blocking_gaps": [
                    {
                        "metric": "supply_confirmed_stock_snapshot",
                        "reason": "Нет supply-слоя для подтверждения остатков.",
                    },
                    {
                        "metric": "net_profit",
                        "reason": "Нет полного P&L для чистой прибыли.",
                    },
                ],
            },
        )

        self.assertIn("438767 RUB", rendered)
        self.assertIn("0.1482 x", rendered)
        self.assertIn("312 310.35 RUB", rendered)
        self.assertIn("Покрытие запасом: 64.8 days за 2026-07-01/2026-07-13", rendered)
        self.assertIn("Contribution margin (не чистая прибыль)", rendered)
        self.assertNotIn("Нет supply-слоя", rendered)
        self.assertIn("Нет полного P&L", rendered)
        self.assertNotIn("Для решения нужно 3–7", rendered)
        self.assertNotIn("Небезопасный модельный synthesis", rendered)
        self.assertNotIn("None", rendered)
        self.assertNotIn("see SKU-level report", rendered)
        self.assertNotIn("calculation is available in another report", rendered)

    def test_facts_only_fallback_separates_causal_facts_from_price_and_ad_gaps(self):
        rendered = _render_prompt_v2_facts_only_answer(
            (
                "Почему изменились продажи за последние 7 дней? Проверь спрос, "
                "рекламу, цену и остатки, отделяя факты от гипотез."
            ),
            {
                "confirmed_facts": [
                    {
                        "source_agent": "sales",
                        "metric": "sales_units_delta",
                        "value": -26,
                        "unit": "шт",
                        "period": "2026-07-04..2026-07-10 vs 2026-06-27..2026-07-03",
                    },
                    {
                        "source_agent": "price",
                        "metric": "latest price decision snapshot date",
                        "value": "2026-04-24",
                        "unit": "date",
                        "period": "2026-04-24",
                    },
                    {
                        "source_agent": "supply",
                        "metric": "available_stock_qty",
                        "value": 1,
                        "unit": "шт",
                        "period": "2026-07-08",
                        "sku": "768260062",
                    },
                ],
                "blocking_gaps": [
                    {
                        "metric": "ad attribution / campaign incrementality",
                        "reason": "Нет прямой привязки продаж к рекламным кампаниям.",
                    },
                    {
                        "metric": "daily sale price / discount / promo history",
                        "reason": "Нет дневной истории цены и скидки за нужный период.",
                    },
                    {
                        "metric": "stock history",
                        "reason": "Есть только последний срез остатков, без дневного ряда.",
                    },
                ],
            },
        )

        self.assertIn("Изменение продаж: -26 шт", rendered)
        self.assertNotIn("Дата доступного ценового снимка", rendered)
        self.assertIn("SKU 768260062: Текущий доступный остаток: 1 шт", rendered)
        self.assertIn("Сами по себе эти факты не доказывают причинную связь", rendered)
        self.assertIn("## Неподтверждённые гипотезы и ограничения", rendered)
        self.assertIn("Нет прямой привязки продаж к рекламным кампаниям", rendered)
        self.assertIn("Нет дневной истории цены и скидки", rendered)
        self.assertIn("Есть только последний срез остатков", rendered)

    def test_confirmed_fact_selector_reserves_each_requested_domain(self):
        facts = [
            {
                "source_agent": "sales",
                "metric": f"sales_units_sku_{index}",
                "value": index + 1,
            }
            for index in range(8)
        ]
        facts.extend(
            [
                {
                    "source_agent": "supply",
                    "metric": "available_stock_qty",
                    "value": 1,
                },
                {
                    "source_agent": "marketing",
                    "metric": "advertising_spend",
                    "value": 100,
                },
                {
                    "source_agent": "price",
                    "metric": "latest_price_snapshot",
                    "value": "2026-04-24",
                    "freshness_status": "stale",
                },
            ]
        )

        selected = _select_confirmed_facts_for_query(
            facts,
            user_input=(
                "Почему изменились продажи? Проверь рекламу, цену и остатки, "
                "отделяя факты от гипотез."
            ),
            primary_agent=QueryIntent.SALES,
            limit=4,
        )

        self.assertEqual(
            {fact["source_agent"] for fact in selected},
            {"sales", "marketing", "price", "supply"},
        )

    def test_typed_fact_does_not_trust_model_authored_provenance_prefixes(self):
        fact = _typed_fact_for_synthesis(
            "marketing",
            {
                "source_agent": "sales",
                "source_table": "sales.wb_supplier_sales",
                "evidence_ref": "get_sales_report_snapshot.rows[1]",
                "metric": "sales_units",
                "value": 127,
            },
            verification_verdict="no_data",
        )

        self.assertEqual(fact["source_agent"], "marketing")

    def test_typed_fact_does_not_trust_domain_word_in_model_provenance(self):
        fact = _typed_fact_for_synthesis(
            "marketing",
            {
                "source_agent": "sales",
                "source_table": "marketing.sales_attribution_estimate",
                "evidence_ref": "marketing_run_sql.sales_projection",
                "metric": "sales_units",
                "value": 127,
            },
            verification_verdict="no_data",
        )

        self.assertEqual(fact["source_agent"], "marketing")

    def test_confirmed_fact_selector_requires_matching_source_for_domain_slot(self):
        misleading_supply_facts = [
            {
                "source_agent": "supply",
                "metric": metric,
                "value": index,
            }
            for index, metric in enumerate(
                ("sales_units", "advertising_spend", "competitor_price", "available_stock")
            )
        ]
        genuine = [
            {"source_agent": "sales", "metric": "sales_units", "value": 127},
            {"source_agent": "marketing", "metric": "advertising_spend", "value": 100},
            {"source_agent": "price", "metric": "competitor_price", "value": 90},
            {"source_agent": "supply", "metric": "available_stock", "value": 10},
        ]

        selected = _select_confirmed_facts_for_query(
            [*misleading_supply_facts, *genuine],
            user_input="Проверь продажи, рекламу, цену и остатки.",
            primary_agent=QueryIntent.SALES,
            limit=4,
        )

        self.assertEqual(
            {fact["source_agent"] for fact in selected},
            {"sales", "marketing", "price", "supply"},
        )

    def test_confirmed_fact_selector_omits_wrong_authors_when_domains_are_absent(self):
        facts = [
            {"source_agent": "supply", "metric": "sales_units", "value": 127},
            {"source_agent": "supply", "metric": "advertising_spend", "value": 100},
            {"source_agent": "supply", "metric": "competitor_price", "value": 90},
            {"source_agent": "supply", "metric": "available_stock", "value": 10},
        ]

        selected = _select_confirmed_facts_for_query(
            facts,
            user_input="Проверь продажи, рекламу, цену и остатки.",
            primary_agent=QueryIntent.SALES,
            limit=4,
        )

        self.assertEqual(
            [(fact["source_agent"], fact["metric"]) for fact in selected],
            [("supply", "available_stock")],
        )

    @patch("agi_ecom_app.runner.list_agent_run_responses")
    def test_persisted_evidence_loader_ignores_old_synthesis_responses(self, mock_rows):
        mock_rows.return_value = [
            {
                "round_type": "primary",
                "round_number": 0,
                "target_agent": "supply",
                "response": SimpleNamespace(
                    raw_output={"summary": "primary"},
                    runtime_failure=None,
                ),
            },
            {
                "round_type": "synthesis",
                "round_number": 5,
                "target_agent": "diagnostic_orchestrator",
                "response": SimpleNamespace(
                    raw_output={"summary": "old synthesis"},
                    runtime_failure={"kind": "runtime"},
                ),
            },
        ]

        persisted = _load_persisted_evidence_safely(276)

        self.assertEqual(persisted["primary_output"], {"summary": "primary"})
        self.assertEqual(persisted["round_1_evidence"], [])
        self.assertEqual(persisted["followup_evidence"], [])

    @patch("agi_ecom_app.runner.Runner.run_sync")
    def test_run_agent_stores_tool_report_when_formula_catalog_is_used(self, mock_run_sync):
        context = self._context()

        def _run_sync(_, __, **kwargs):
            self._append_tool_trace(kwargs["context"], "get_sales_formula_catalog")
            self._append_tool_trace(kwargs["context"], "get_sales_overview")
            return SimpleNamespace(final_output=self._neutral_sales_output())

        mock_run_sync.side_effect = _run_sync

        output = _run_agent(
            lambda: self._tool_agent("get_sales_formula_catalog", "get_sales_overview"),
            user_input="почему упали продажи",
            context=context,
            hooks="hooks",
            max_turns=8,
            tool_policy=_domain_tool_policy(
                target=QueryIntent.SALES,
                user_query="почему упали продажи",
                round_name="primary",
                question_id=13,
            ),
        )

        self.assertIsInstance(output, SalesDiagnosticResult)
        report = context.last_tool_report
        self.assertIsNotNone(report)
        self.assertEqual(report["tool_names"], ["get_sales_formula_catalog", "get_sales_overview"])
        self.assertTrue(report["formula_catalog_used"])
        self.assertEqual(report["tools"][0]["duration_ms"], 7)
        self.assertNotIn("raw_output", report)
        self.assertNotIn("sql", json.dumps(report).lower())

    @patch("agi_ecom_app.runner.Runner.run_sync")
    def test_run_agent_does_not_retry_when_formula_catalog_is_missing(self, mock_run_sync):
        context = self._context()

        def _run_sync(_, prompt, **kwargs):
            self._append_tool_trace(kwargs["context"], "get_sales_overview")
            return SimpleNamespace(final_output=outputs.pop(0))

        outputs = [self._neutral_sales_output()]
        mock_run_sync.side_effect = _run_sync

        _run_agent(
            lambda: self._tool_agent("get_sales_formula_catalog", "get_sales_overview"),
            user_input="почему упали продажи",
            context=context,
            hooks="hooks",
            max_turns=8,
            tool_policy=_domain_tool_policy(
                target=QueryIntent.SALES,
                user_query="почему упали продажи",
                round_name="primary",
                question_id=14,
            ),
        )

        self.assertEqual(mock_run_sync.call_count, 1)
        self.assertEqual(context.last_tool_report["tool_names"], ["get_sales_overview"])
        self.assertEqual(context.last_tool_report["required_tools"], [])
        self.assertEqual(context.last_tool_report["policy_violations"], [])
        self.assertFalse(context.last_tool_report["formula_catalog_used"])

    @patch("agi_ecom_app.runner.Runner.run_sync")
    def test_run_agent_retries_price_decision_when_decision_tool_is_missing(self, mock_run_sync):
        context = self._context(
            user_query="Какую цену на товар с низкой оборачиваемостью мы можем поставить?",
            intent=QueryIntent.PRICE,
        )
        outputs = [
            self._agent_output("Price Agent"),
            AgentDiagnosticResult(
                agent_name="Price Agent",
                summary="Сценарная цена рассчитана.",
                tables=[
                    ReportTable(
                        title="Ценовой сценарий",
                        columns=[
                            "SKU",
                            "Текущая цена",
                            "Рекомендованная цена",
                            "expected_orders_7d",
                            "Уверенность",
                        ],
                        rows=[["46224", "2075", "1765", "5.53", "low"]],
                    )
                ],
            ),
        ]

        def _run_sync(_, prompt, **kwargs):
            if len(outputs) == 2:
                self._append_tool_trace(kwargs["context"], "get_price_competitor_overview")
            else:
                self.assertIn("Служебная повторная попытка", prompt)
                self._append_tool_trace(kwargs["context"], "get_price_decision_overview")
            return SimpleNamespace(final_output=outputs.pop(0))

        mock_run_sync.side_effect = _run_sync

        output = _run_agent(
            lambda: self._tool_agent("get_price_decision_overview", "get_price_competitor_overview"),
            user_input="Какую цену на товар с низкой оборачиваемостью мы можем поставить?",
            context=context,
            hooks="hooks",
            max_turns=8,
            tool_policy=_domain_tool_policy(
                target=QueryIntent.PRICE,
                user_query="Какую цену на товар с низкой оборачиваемостью мы можем поставить?",
                round_name="primary",
                question_id=16,
            ),
        )

        self.assertEqual(mock_run_sync.call_count, 2)
        self.assertEqual(output.summary, "Сценарная цена рассчитана.")
        self.assertEqual(context.last_tool_report["tool_names"], ["get_price_decision_overview"])
        self.assertEqual(context.last_tool_report["required_tools"], ["get_price_decision_overview"])
        self.assertEqual(context.last_tool_report["policy_violations"], [])

    @patch("agi_ecom_app.runner.Runner.run_sync")
    def test_run_agent_allows_formula_catalog_after_data_tool(self, mock_run_sync):
        context = self._context()

        def _run_sync(_, __, **kwargs):
            self._append_tool_trace(kwargs["context"], "get_sales_overview")
            self._append_tool_trace(kwargs["context"], "get_sales_formula_catalog")
            return SimpleNamespace(final_output=self._neutral_sales_output())

        mock_run_sync.side_effect = _run_sync

        _run_agent(
            lambda: self._tool_agent("get_sales_formula_catalog", "get_sales_overview"),
            user_input="почему упали продажи",
            context=context,
            hooks="hooks",
            max_turns=8,
            tool_policy=_domain_tool_policy(
                target=QueryIntent.SALES,
                user_query="почему упали продажи",
                round_name="primary",
                question_id=15,
            ),
        )

        self.assertEqual(mock_run_sync.call_count, 1)
        self.assertEqual(
            context.last_tool_report["tool_names"],
            ["get_sales_overview", "get_sales_formula_catalog"],
        )
        self.assertEqual(context.last_tool_report["policy_violations"], [])
        self.assertTrue(context.last_tool_report["formula_catalog_used"])

    @patch("agi_ecom_app.runner.Runner.run_sync")
    def test_diagnostic_orchestrator_run_is_exempt_from_formula_policy(self, mock_run_sync):
        context = self._context(intent=QueryIntent.SALES)
        mock_run_sync.return_value = SimpleNamespace(final_output="# Итог")

        output = _run_agent(
            lambda: self._tool_agent("get_sales_overview"),
            user_input="собери финальный ответ",
            context=context,
            hooks="hooks",
            max_turns=4,
        )

        self.assertEqual(output, "# Итог")
        self.assertEqual(mock_run_sync.call_count, 1)
        self.assertIsNone(context.last_tool_report)

    @patch("agi_ecom_app.runner.save_agent_run_gaps")
    @patch("agi_ecom_app.runner.save_agent_run_response")
    def test_successful_domain_run_persists_tool_report_in_raw_output(
        self,
        mock_save_response,
        mock_save_gaps,
    ):
        context = self._context()
        context.last_tool_report = {
            "agent_name": "sales_agent",
            "target_agent": "sales",
            "round": "primary",
            "question_id": 13,
            "tool_names": ["get_sales_formula_catalog", "get_sales_overview"],
            "tools": [
                {"name": "get_sales_formula_catalog", "status": "completed", "duration_ms": 7},
                {"name": "get_sales_overview", "status": "completed", "duration_ms": 7},
            ],
            "required_tools": [],
            "policy_violations": [],
            "formula_catalog_used": True,
        }
        mock_save_response.return_value = SimpleNamespace(id=1)
        payload = _attach_tool_report(_diagnostic_payload(self._neutral_sales_output()), context)

        _persist_agent_run_response_safely(
            run_id=101,
            question_record=SimpleNamespace(id=13),
            agent_name="sales",
            payload=payload,
        )

        raw_output = mock_save_response.call_args.kwargs["raw_output"]
        report = raw_output["_tool_report"]
        self.assertEqual(report["required_tools"], [])
        self.assertTrue(report["formula_catalog_used"])
        self.assertEqual(report["tool_names"], ["get_sales_formula_catalog", "get_sales_overview"])

    def test_tool_policy_failed_evidence_is_not_confirmed_fact(self):
        failed_output = self._agent_output("Marketing Agent")

        reduced = _reduce_evidence(
            primary_agent=QueryIntent.SALES,
            primary_output=self._neutral_sales_output(),
            round_1_evidence=[
                {
                    "round": "round_1",
                    "target_agent": "marketing",
                    "question": "Проверь рекламу.",
                    "output": _diagnostic_payload(failed_output),
                    "runtime_failure": {
                        "kind": "tool_policy",
                        "violations": ["missing_required_tool:get_marketing_formula_catalog"],
                    },
                }
            ],
            followup_evidence=[],
        )

        self.assertFalse(
            any("marketing" in fact.lower() for fact in reduced["confirmed_facts"])
        )
        self.assertIn("tool_policy", json.dumps(reduced["runtime_failures"]))

    def test_sales_diagnostic_payload_includes_report_snapshot(self):
        sales_output = self._neutral_sales_output()
        sales_output.sales_report_snapshot = self._sales_snapshot_table()

        payload = _diagnostic_payload(sales_output)

        self.assertIsNotNone(payload["sales_report_snapshot"])
        self.assertEqual(payload["sales_report_snapshot"]["title"], "Sales-сводка")
        self.assertEqual(payload["sales_report_snapshot"]["rows"][0][0], "Вчера к позавчера")

    def test_marketing_diagnostic_payload_includes_report_snapshot(self):
        marketing_output = MarketingDiagnosticResult(
            agent_name="Marketing Agent",
            summary="Реклама за период собрана.",
            marketing_report_snapshot=self._marketing_snapshot_table(),
            tables=[
                ReportTable(
                    title="Реклама",
                    columns=["Метрика", "Значение"],
                    rows=[["ДРР", "10%"]],
                )
            ],
        )

        payload = _diagnostic_payload(marketing_output)

        self.assertIsNotNone(payload["marketing_report_snapshot"])
        self.assertEqual(payload["marketing_report_snapshot"]["title"], "Marketing-сводка")
        self.assertEqual(payload["marketing_report_snapshot"]["rows"][0][0], "Вчера к позавчера")

    def test_sales_report_snapshot_is_prepended_to_final_answer(self):
        sales_output = self._neutral_sales_output()
        sales_output.sales_report_snapshot = self._sales_snapshot_table()
        synthesis_package = {
            "primary_evidence": {
                "sales_report_snapshot": _diagnostic_payload(sales_output)["sales_report_snapshot"]
            }
        }

        rendered = _prepend_sales_report_snapshot("# Отчет по продажам\n\nКраткий вывод.", synthesis_package)

        self.assertTrue(rendered.startswith("**Sales-сводка**"))
        self.assertLess(rendered.index("**Sales-сводка**"), rendered.index("# Отчет по продажам"))
        self.assertEqual(_prepend_sales_report_snapshot(rendered, synthesis_package), rendered)

    def test_marketing_report_snapshot_is_prepended_to_final_answer(self):
        marketing_output = MarketingDiagnosticResult(
            agent_name="Marketing Agent",
            summary="Реклама за период собрана.",
            marketing_report_snapshot=self._marketing_snapshot_table(),
            tables=[
                ReportTable(
                    title="Реклама",
                    columns=["Метрика", "Значение"],
                    rows=[["ДРР", "10%"]],
                )
            ],
        )
        synthesis_package = {
            "primary_evidence": {
                "marketing_report_snapshot": _diagnostic_payload(marketing_output)["marketing_report_snapshot"]
            }
        }

        rendered = _prepend_marketing_report_snapshot("# Отчет по рекламе\n\nКраткий вывод.", synthesis_package)

        self.assertTrue(rendered.startswith("**Marketing-сводка**"))
        self.assertLess(rendered.index("**Marketing-сводка**"), rendered.index("# Отчет по рекламе"))
        self.assertEqual(_prepend_marketing_report_snapshot(rendered, synthesis_package), rendered)

    def test_marketing_report_snapshot_moves_existing_markdown_table_to_top(self):
        marketing_output = MarketingDiagnosticResult(
            agent_name="Marketing Agent",
            summary="Реклама за период собрана.",
            marketing_report_snapshot=self._marketing_snapshot_table(),
            tables=[
                ReportTable(
                    title="Реклама",
                    columns=["Метрика", "Значение"],
                    rows=[["ДРР", "10%"]],
                )
            ],
        )
        synthesis_package = {
            "primary_evidence": {
                "marketing_report_snapshot": _diagnostic_payload(marketing_output)["marketing_report_snapshot"]
            }
        }
        answer = (
            "# Отчет по рекламе\n\n"
            "Краткий вывод.\n\n"
            "| Сравнение | Периоды | ДРР тек./пред. |\n"
            "| --- | --- | --- |\n"
            "| Вчера к позавчера | 2026-04-19 к 2026-04-18 | 10.00% / 5.00% |\n"
            "| Неделя к неделе | 2026-04-13..2026-04-19 к 2026-04-06..2026-04-12 | 20.00% / 10.00% |\n"
            "| Месяц к месяцу | 2026-03-21..2026-04-19 к 2026-02-19..2026-03-20 | 20.00% / 10.00% |\n\n"
            "Что делать дальше."
        )

        rendered = _prepend_marketing_report_snapshot(answer, synthesis_package)

        self.assertTrue(rendered.startswith("**Marketing-сводка**"))
        self.assertLess(rendered.index("**Marketing-сводка**"), rendered.index("# Отчет по рекламе"))
        self.assertEqual(rendered.lower().count("вчера к позавчера"), 1)
        self.assertEqual(rendered.lower().count("дрр"), 1)
        self.assertIn("Что делать дальше.", rendered)

    def test_sales_report_snapshot_moves_existing_html_table_to_top(self):
        sales_output = self._neutral_sales_output()
        sales_output.sales_report_snapshot = self._sales_snapshot_table()
        synthesis_package = {
            "primary_evidence": {
                "sales_report_snapshot": _diagnostic_payload(sales_output)["sales_report_snapshot"]
            }
        }
        answer = (
            "# Отчет по продажам\n\n"
            "<table><tr><th>Сравнение</th></tr>"
            "<tr><td>Вчера к позавчера</td></tr>"
            "<tr><td>Неделя к неделе</td></tr>"
            "<tr><td>Месяц к месяцу</td></tr></table>\n\n"
            "Краткий вывод."
        )

        rendered = _prepend_sales_report_snapshot(answer, synthesis_package)

        self.assertTrue(rendered.startswith("**Sales-сводка**"))
        self.assertLess(rendered.index("**Sales-сводка**"), rendered.index("# Отчет по продажам"))
        self.assertNotIn("<table>", rendered)

    def test_sales_report_snapshot_moves_existing_markdown_table_to_top(self):
        sales_output = self._neutral_sales_output()
        sales_output.sales_report_snapshot = self._sales_snapshot_table()
        synthesis_package = {
            "primary_evidence": {
                "sales_report_snapshot": _diagnostic_payload(sales_output)["sales_report_snapshot"]
            }
        }
        answer = (
            "# Отчет по продажам\n\n"
            "Краткий вывод.\n\n"
            "| Сравнение | Периоды | Продажи тек./пред. |\n"
            "| --- | --- | --- |\n"
            "| Вчера к позавчера | 2026-04-19 к 2026-04-18 | 15 / 13 |\n"
            "| Неделя к неделе | 2026-04-13..2026-04-19 к 2026-04-06..2026-04-12 | 90 / 84 |\n"
            "| Месяц к месяцу | 2026-03-21..2026-04-19 к 2026-02-19..2026-03-20 | 300 / 280 |\n\n"
            "Что делать дальше."
        )

        rendered = _prepend_sales_report_snapshot(answer, synthesis_package)

        self.assertTrue(rendered.startswith("**Sales-сводка**"))
        self.assertLess(rendered.index("**Sales-сводка**"), rendered.index("# Отчет по продажам"))
        self.assertEqual(rendered.lower().count("вчера к позавчера"), 1)
        self.assertEqual(rendered.lower().count("неделя к неделе"), 1)
        self.assertEqual(rendered.lower().count("месяц к месяцу"), 1)
        self.assertIn("Что делать дальше.", rendered)

    def _agent_output(self, agent_name="Supply Agent", cross_agent_requests=None):
        return AgentDiagnosticResult(
            agent_name=agent_name,
            summary="Остатки проверены.",
            tables=[ReportTable(title="Остатки", columns=["Артикул", "Остаток"], rows=[["46313", 5]])],
            findings=[self._finding("Supply", "Есть локальный риск доступности.")],
            cross_agent_requests=cross_agent_requests or [],
        )

    def test_final_markdown_quality_validator_flags_nm_id_and_unconditional_actions(self):
        bad_answer = (
            "# Диагностика\n\n"
            "Краткий вывод: главный фактор — снижение объёма продаж.\n\n"
            "| nm_id | Артикул | Δ маржи |\n"
            "| --- | --- | ---: |\n"
            "| 171742855 | 46303 | -41335 |\n\n"
            "Что делать:\n"
            "1. Обновить кампанию 26700049: сократить дорогой трафик.\n"
        )

        violations = validate_final_diagnostic_markdown(
            bad_answer,
            user_query="Покажи подтверждённые и неподтверждённые причины.",
        )
        sanitized = sanitize_user_facing_markdown(bad_answer)

        self.assertIn("final markdown exposes nm_id", violations)
        self.assertTrue(any("risky action lacks condition" in item for item in violations))
        self.assertNotIn("nm_id", sanitized.lower())
        self.assertNotIn("171742855", sanitized)

    def test_sanitize_user_facing_markdown_does_not_partially_translate_free_prose(self):
        answer = (
            "Нет price-слоя, supply-tools, marketing spend и finance data.\n"
            "| Гипотеза | Результат |\n"
            "| --- | --- |\n"
            "| Цена | price-evidence нет, supply-срез неполный |\n"
        )

        sanitized = sanitize_user_facing_markdown(answer)

        self.assertEqual(sanitized, answer.rstrip("\n"))
        self.assertNotIn("цену-слоя", sanitized)
        self.assertNotIn("остатки-tools", sanitized)
        self.assertNotIn("рекламу spend", sanitized)
        self.assertNotIn("экономику data", sanitized)
        self.assertFalse(_sku_only_display_was_sanitized(answer, sanitized))

    def test_final_quality_validator_rejects_broken_domain_localization(self):
        violations = validate_final_diagnostic_markdown(
            "Нет цену-слоя; остатки-tools вернули рекламу spend.",
            user_query="Проверь данные.",
        )

        self.assertIn("final markdown contains broken domain localization", violations)

    def test_sku_only_display_sanitization_tracks_nm_id_removal(self):
        answer = "| nm_id | SKU | Остаток |\n| --- | --- | --- |\n| 171742855 | 46303 | 228 |"
        sanitized = sanitize_user_facing_markdown(answer)

        self.assertTrue(_sku_only_display_was_sanitized(answer, sanitized))

    def test_empty_markdown_table_is_replaced_with_explicit_status(self):
        answer = (
            "## Сценарий\n\n"
            "| Вариант | Эффект |\n"
            "|---|---:|\n"
            "\nСледующий раздел."
        )

        normalized, removed = remove_empty_markdown_tables(answer)

        self.assertEqual(removed, 1)
        self.assertNotIn("| Вариант | Эффект |", normalized)
        self.assertIn("пустая таблица не публикуется", normalized)

    def test_non_diagnostic_turnover_strips_unrequested_sku_cause_matrix(self):
        answer = (
            "# Оборачиваемость склада\n\n"
            "Краткий вывод.\n\n"
            "## Почему это произошло\n\n"
            "| SKU | спрос | реклама |\n"
            "| --- | --- | --- |\n"
            "| 46312 | подтверждено | не проверялось |\n\n"
            "## Подтверждено\n\n"
            "Остатки и продажи проверены."
        )

        cleaned = _strip_unrequested_sku_cause_matrix(
            answer,
            user_query="Посчитай оборачиваемость склада",
            sku_evidence_matrix=[],
        )

        self.assertNotIn("Почему это произошло", cleaned)
        self.assertNotIn("не проверялось", cleaned)
        self.assertIn("## Подтверждено", cleaned)

    def test_non_diagnostic_turnover_softens_cause_overclaim(self):
        answer = (
            "# Оборачиваемость склада\n\n"
            "Краткий вывод: по части SKU причина замедления подтверждена, "
            "а по бренду экономика положительная."
        )

        cleaned = _sanitize_inventory_turnover_cause_claims(
            answer,
            user_query="Посчитай оборачиваемость склада",
        )

        self.assertNotIn("причина замедления подтверждена", cleaned)
        self.assertIn("подтвержден механический вклад в медленную оборачиваемость", cleaned)
        self.assertIn("экономика положительная", cleaned)

    def test_diagnostic_turnover_keeps_cause_language_for_explicit_cause_query(self):
        answer = (
            "# Оборачиваемость склада\n\n"
            "Краткий вывод: по части SKU причина замедления подтверждена."
        )

        cleaned = _sanitize_inventory_turnover_cause_claims(
            answer,
            user_query="Почему замедлилась оборачиваемость склада?",
        )

        self.assertEqual(cleaned, answer)

    def test_inventory_turnover_causal_query_requires_core_domains(self):
        plan = _inventory_turnover_coverage_plan(
            user_input="Почему оборачиваемость склада медленная?",
            primary_agent=QueryIntent.SUPPLY,
            existing_items=[],
            max_secondary_agents=5,
        )

        self.assertIsNotNone(plan)
        self.assertEqual(
            [item["target_agent"] for item in plan["recommended_secondary_agents"]],
            ["sales", "price", "marketing", "finance"],
        )
        self.assertTrue(
            all(item["source"] == "coverage_policy" for item in plan["recommended_secondary_agents"])
        )

    def test_simple_turnover_query_does_not_create_causal_coverage_plan(self):
        plan = _inventory_turnover_coverage_plan(
            user_input="Посчитай оборачиваемость склада",
            primary_agent=QueryIntent.SUPPLY,
            existing_items=[],
            max_secondary_agents=5,
        )

        self.assertIsNone(plan)

    def test_query_plan_selects_cross_checks_without_keyword_parsing(self):
        plan = _inventory_turnover_coverage_plan(
            user_input="Сделай то, что мы обсуждали",
            primary_agent=QueryIntent.SUPPLY,
            existing_items=[],
            max_secondary_agents=5,
            query_plan={
                "query_mode": "decision",
                "requested_analyses": ["inventory_turnover", "inventory_matrix"],
                "required_domains": ["supply", "finance", "marketing", "price", "sales"],
            },
        )

        self.assertEqual(
            [item["target_agent"] for item in plan["recommended_secondary_agents"]],
            ["finance", "marketing", "price", "sales"],
        )

    def test_synthesis_package_marks_missing_claim_verification_domains(self):
        package = _build_synthesis_package(
            user_input="Почему оборачиваемость склада медленная?",
            primary_agent=QueryIntent.SUPPLY,
            primary_output=self._agent_output("Supply Agent"),
            round_1_plan={},
            round_1_evidence=[
                {
                    "round": "round_1",
                    "target_agent": "sales",
                    "question": "Проверь спрос.",
                    "output": _diagnostic_payload(self._agent_output("Sales Agent")),
                }
            ],
            round_2_plan={},
            followup_evidence=[],
        )

        verification = package["claim_verification"]
        self.assertEqual(verification["status"], "incomplete")
        self.assertIn("sales", verification["covered_domains"])
        self.assertEqual(
            verification["missing_domains"],
            ["price", "marketing", "finance"],
        )

    def test_claim_verification_plan_targets_missing_domains_only(self):
        plan = _claim_verification_plan(
            user_input="Почему оборачиваемость склада медленная?",
            primary_agent=QueryIntent.SUPPLY,
            primary_output=self._agent_output("Supply Agent"),
            round_1_evidence=[
                {
                    "round": "round_1",
                    "target_agent": "sales",
                    "question": "Проверь спрос.",
                    "output": _diagnostic_payload(self._agent_output("Sales Agent")),
                },
                {
                    "round": "round_1",
                    "target_agent": "finance",
                    "question": "Проверь экономику.",
                    "output": _diagnostic_payload(self._agent_output("Finance Agent")),
                },
            ],
            followup_evidence=[],
            max_secondary_agents=4,
        )

        self.assertIsNotNone(plan)
        self.assertEqual(
            [item["target_agent"] for item in plan["recommended_secondary_agents"]],
            ["price", "marketing"],
        )
        self.assertTrue(
            all(item["source"] == "claim_verification" for item in plan["recommended_secondary_agents"])
        )

    def test_diagnostic_query_keeps_requested_sku_cause_matrix(self):
        answer = (
            "# Причины\n\n"
            "## Почему это произошло\n\n"
            "| SKU | спрос | реклама |\n"
            "| --- | --- | --- |\n"
            "| 46312 | подтверждено | нет данных |"
        )

        cleaned = _strip_unrequested_sku_cause_matrix(
            answer,
            user_query="Почему изменились продажи по SKU?",
            sku_evidence_matrix=[{"sku": "46312"}],
        )

        self.assertEqual(cleaned, answer)

    def test_gap_resolution_no_data_becomes_accepted_limitation(self):
        payload = {
            "summary": "Проверка не закрыта данными: в источнике нет строк по ставкам. Результат — no_data.",
            "tables": [{"title": "Проверка", "columns": ["Факт"], "rows": [["нет строк"]]}],
            "findings": [],
            "data_gaps": [
                {
                    "metric": "Ставки",
                    "reason": "В источнике отсутствуют строки.",
                    "impact": "Нельзя подтвердить влияние ставки.",
                }
            ],
        }

        self.assertEqual(classify_gap_resolution_response(payload), "accepted_limitation")

    def test_gap_resolution_partial_evidence_resolves_original_gap(self):
        payload = {
            "summary": "Gap закрыт не как no_data: маркетинговый фактор подтвержден частично.",
            "tables": [{"title": "Воронка", "columns": ["SKU", "Заказы"], "rows": [["46311", 2]]}],
            "findings": [self._finding("Marketing", "Воронка после показа ослабла.")],
            "data_gaps": [
                {
                    "metric": "Кампании по 46311",
                    "reason": "Детальный кампанийный срез вернул пустой результат.",
                }
            ],
        }

        self.assertEqual(classify_gap_resolution_response(payload), "resolved")

    def test_gap_resolution_typed_no_data_wins_over_evidence_heuristic(self):
        payload = {
            "summary": "Проверка завершена.",
            "tables": [{"title": "Snapshot", "columns": ["SKU"], "rows": [["46227"]]}],
            "findings": [self._finding("Supply", "Доступен только идентификатор SKU.").model_dump(mode="json")],
            "data_gaps": [],
            "gap_resolution_status": "no_data",
            "_round_result": {
                "round_type": "gap_resolution",
                "verdict": "no_data",
                "facts": [],
            },
        }

        self.assertEqual(
            classify_gap_resolution_response(payload),
            "accepted_limitation",
        )

    def test_gap_resolution_typed_outcome_precedes_conflicting_text(self):
        resolved_payload = {
            "summary": "В пояснении упомянуто no_data, но typed result подтвержден evidence.",
            "tables": [],
            "findings": [],
            "data_gaps": [],
            "_round_result": {
                "round_type": "gap_resolution",
                "verdict": "resolved",
                "facts": [],
            },
        }
        limited_payload = {
            "summary": "Формулировка похожа на подтверждение.",
            "tables": [{"title": "Проверка", "columns": ["Статус"], "rows": [["Есть snapshot"]]}],
            "findings": [],
            "data_gaps": [],
            "gap_resolution_status": "accepted_limitation",
        }

        self.assertEqual(classify_gap_resolution_response(resolved_payload), "resolved")
        self.assertEqual(
            classify_gap_resolution_response(limited_payload),
            "accepted_limitation",
        )

    def test_gap_resolution_filters_no_data_refinement_gap(self):
        original_gap = SimpleNamespace(
            id=1,
            sku="46311",
            source_agent="marketing",
            target_agent="marketing",
            metric="Рекламные кампании по артикулу 46311",
            reason="sku_not_in_source",
        )
        new_gap = {
            "sku": "46311",
            "source_agent": "marketing",
            "target_agent": "marketing",
            "metric": "Кампании по 46311",
            "reason": "Готовый срез кампаний вернул пустой результат.",
            "impact": "Нельзя детально проверить кампании.",
        }

        self.assertFalse(should_persist_gap_from_gap_resolution(new_gap, original_gap))

    def test_gap_resolution_allows_independent_new_gap(self):
        original_gap = SimpleNamespace(
            id=1,
            sku="46311",
            source_agent="marketing",
            target_agent="marketing",
            metric="Рекламные кампании по артикулу 46311",
            reason="sku_not_in_source",
        )
        new_gap = {
            "sku": "46300",
            "source_agent": "supply",
            "target_agent": "supply",
            "metric": "Поставка",
            "reason": "Нужно проверить входящие поставки.",
            "impact": "Нельзя масштабировать рекламу без запаса.",
        }

        self.assertTrue(should_persist_gap_from_gap_resolution(new_gap, original_gap))

    def test_final_markdown_quality_validator_allows_conditional_marketing_task(self):
        answer = (
            "# Диагностика\n\n"
            "Краткий вывод: прибыль механически упала из-за объёма, первичная причина частично подтверждена.\n\n"
            "Подтверждено: цена и воронка проверены.\n\n"
            "Неподтверждено: остатки не подтвердились.\n\n"
            "Что делать:\n"
            "1. Проверить кампании 26700049 и 29223909 по SKU 46303; "
            "если ДРР выше порога или клики без заказов — снизить ставку или остановить кампанию.\n"
        )

        self.assertEqual(validate_final_diagnostic_markdown(answer), [])

    def test_final_markdown_quality_validator_accepts_five_action_table(self):
        answer = (
            "# Диагностика SKU\n\n"
            "Краткий вывод: прибыль механически изменилась через объём, причины проверены.\n\n"
            "Подтверждено: причины собраны.\n\n"
            "Неподтверждено: часть гипотез снята.\n\n"
            "Что делать:\n"
            "| # | Задача | Объект | Почему |\n"
            "| --- | --- | --- | --- |\n"
            "| 1 | Проверить доступный остаток | 46230, 46306, 46225 + ещё 2 | OOS или критично низкий запас |\n"
            "| 2 | Остановить слабую рекламу | 46312 | расход есть, заказов 0 |\n"
            "| 3 | Рассчитать ценовой тест | 46230, 46303 | маржа есть, цена выше рынка |\n"
            "| 4 | Проверить карточку и корзину | 46303, 46230 | заказы и конверсия просели |\n"
            "| 5 | Удержать рекламу без расширения | 46227 | реклама рабочая |\n"
        )

        self.assertEqual(
            validate_final_diagnostic_markdown(
                answer,
                user_query="В конце дай ровно 5 задач.",
            ),
            [],
        )

    def test_final_markdown_quality_validator_flags_action_table_fallback_phrasing(self):
        answer = (
            "# Диагностика SKU\n\n"
            "Краткий вывод: причины проверены.\n\n"
            "Что делать:\n"
            "| # | Задача | Объект | Почему |\n"
            "| --- | --- | --- | --- |\n"
            "| 1 | Проверить кампанию | 46312 | если ДРР выше порога иначе зафиксировать |\n"
        )

        violations = validate_final_diagnostic_markdown(answer)

        self.assertTrue(
            any("action table row contains fallback phrasing" in item for item in violations)
        )

    def test_final_markdown_quality_validator_flags_bare_action_table_task(self):
        answer = (
            "# Диагностика SKU\n\n"
            "Краткий вывод: причины проверены.\n\n"
            "Что делать:\n"
            "| # | Задача | Объект | Почему |\n"
            "| --- | --- | --- | --- |\n"
            "| 1 | Проверить | 46230 | полка пустая |\n"
        )

        violations = validate_final_diagnostic_markdown(answer)

        self.assertTrue(any("action table row is a bare verb" in item for item in violations))

    def test_analytic_chat_uses_diagnostic_orchestrator(self):
        patches, started = self._start_mocks(
            "Почему изменились продажи?",
            QueryIntent.SALES,
            output=[
                self._sales_output(),
                self._agent_output(),
                self._agent_output("Marketing Agent"),
                "# Почему изменились продажи\n\nКраткий вывод: продажи разобраны.",
            ],
        )
        try:
            mock_build_diagnostic_orchestrator = started[3]
            mock_build_sales = started[4]
            mock_build_marketing = started[5]
            mock_build_supply = started[6]
            mock_build_domain_router = started[8]
            mock_run_sync = started[10]
            mock_append_message = started[14]
            mock_refresh_chat_summary = started[15]
            mock_create_trace = started[17]
            mock_complete_trace = started[18]

            result = run_main_agent("Почему изменились продажи?", chat_id="chat-1")

            self.assertIn("Краткий вывод", result)
            self.assertNotIn("Проверяли:", result)
            self.assertNotIn("cross_agent_requests", result)
            self.assertNotIn("Marketing Agent", result)
            self.assertNotIn("Supply Agent", result)
            mock_build_diagnostic_orchestrator.assert_called_once_with()
            mock_build_sales.assert_called_once_with()
            mock_build_marketing.assert_called_once_with()
            mock_build_supply.assert_called_once_with()
            mock_build_domain_router.assert_not_called()
            self.assertEqual(
                mock_run_sync.call_args_list[0].args[:2],
                ("sales-agent", "Почему изменились продажи?"),
            )
            self.assertEqual(mock_run_sync.call_args_list[1].args[0], "supply-agent")
            supply_prompt = mock_run_sync.call_args_list[1].args[1]
            self.assertIn("Служебная смежная проверка", supply_prompt)
            self.assertIn("Почему изменились продажи?", supply_prompt)
            self.assertIn("Проверь остатки.", supply_prompt)
            self.assertIn("primary_evidence", supply_prompt)
            self.assertEqual(
                mock_run_sync.call_args_list[2].args[0],
                "marketing-agent",
            )
            self.assertEqual(
                mock_run_sync.call_args_list[3].args[0],
                "diagnostic-orchestrator",
            )
            metadata = mock_append_message.call_args.kwargs["metadata"]
            self.assertEqual(metadata["intent"], "sales")
            self.assertEqual(metadata["primary_agent"], "sales_agent")
            self.assertEqual(metadata["secondary_agents"], ["supply_agent", "marketing_agent"])
            self.assertEqual(metadata["followup_agents"], [])
            self.assertEqual(mock_run_sync.call_count, 4)
            self.assertEqual(metadata["max_rounds"], MAX_DIAGNOSTIC_ROUNDS)
            self.assertEqual(
                metadata["max_total_followup_domain_calls"],
                MAX_TOTAL_FOLLOWUP_DOMAIN_CALLS,
            )
            mock_refresh_chat_summary.assert_called_once_with(chat_id="chat-1")
            mock_create_trace.assert_called_once_with(
                chat_id="chat-1",
                intent="sales",
                client_message_id=None,
                primary_agent="sales_agent",
                summary=ANY,
            )
            self.assertEqual(mock_create_trace.call_args.kwargs["summary"]["status"], "running")
            self.assertEqual(mock_create_trace.call_args.kwargs["summary"]["model_label"], "openai:gpt-test")
            self.assertIn("intent_router", mock_create_trace.call_args.kwargs["summary"])
            trace_kwargs = mock_complete_trace.call_args.kwargs
            self.assertEqual(trace_kwargs["trace_id"], 101)
            self.assertEqual(trace_kwargs["assistant_message_seq"], 2)
            self.assertEqual(trace_kwargs["status"], "completed")
            trace_events = [event["event"] for event in trace_kwargs["events"]]
            self.assertIn("primary_selected", trace_events)
            self.assertIn("dependency_plan_built", trace_events)
            self.assertIn("secondary_agent_requested", trace_events)
            self.assertIn("secondary_agent_completed", trace_events)
            self.assertIn("synthesis_started", trace_events)
            self.assertIn("synthesis_completed", trace_events)
            self.assertEqual(trace_kwargs["summary"]["primary_agent"], "sales_agent")
        finally:
            for patcher in reversed(patches):
                patcher.stop()

    def test_diagnostic_workflow_runs_round_two_for_new_followup_question(self):
        primary_output = self._sales_output()
        primary_output.cross_agent_requests = [
            CrossAgentRequest(
                target_agent=QueryIntent.MARKETING,
                reason="Нужно проверить трафик.",
                question="Проверь маркетинг.",
            ),
            CrossAgentRequest(
                target_agent=QueryIntent.SUPPLY,
                reason="Нужно проверить остатки.",
                question="Проверь остатки.",
            ),
            CrossAgentRequest(
                target_agent=QueryIntent.FINANCE,
                reason="Нужно проверить маржу.",
                question="Проверь маржу.",
            ),
        ]
        marketing_output = self._agent_output(
            "Marketing Agent",
            cross_agent_requests=[
                CrossAgentRequest(
                    target_agent=QueryIntent.MARKETING,
                    reason="Нужно уточнить поисковую видимость.",
                    question="Проверь поисковую видимость точнее.",
                )
            ],
        )
        patches, started = self._start_mocks(
            "Почему изменились продажи?",
            QueryIntent.SALES,
            output=[
                primary_output,
                marketing_output,
                self._agent_output("Supply Agent"),
                self._agent_output("Finance Agent"),
                self._agent_output("Marketing Agent"),
                "# Почему изменились продажи\n\nКраткий вывод: диагностика собрана.",
            ],
        )
        try:
            mock_run_sync = started[10]
            mock_append_message = started[14]
            mock_persist_tasks = started[19]

            result = run_main_agent("Почему изменились продажи?", chat_id="chat-1")

            self.assertIn("Краткий вывод", result)
            self.assertEqual(mock_run_sync.call_count, 6)
            self.assertEqual(mock_run_sync.call_args_list[0].args[:2], ("sales-agent", "Почему изменились продажи?"))
            self.assertEqual(mock_run_sync.call_args_list[1].args[0], "marketing-agent")
            self.assertIn("Проверь маркетинг.", mock_run_sync.call_args_list[1].args[1])
            self.assertEqual(mock_run_sync.call_args_list[2].args[0], "supply-agent")
            self.assertIn("Проверь остатки.", mock_run_sync.call_args_list[2].args[1])
            self.assertEqual(mock_run_sync.call_args_list[3].args[0], "finance-agent")
            self.assertIn("Проверь маржу.", mock_run_sync.call_args_list[3].args[1])
            self.assertEqual(mock_run_sync.call_args_list[4].args[0], "marketing-agent")
            followup_prompt = mock_run_sync.call_args_list[4].args[1]
            self.assertIn("Проверь поисковую видимость точнее.", followup_prompt)
            self.assertIn("known_cross_evidence", followup_prompt)
            metadata = mock_append_message.call_args.kwargs["metadata"]
            self.assertEqual(
                metadata["secondary_agents"],
                ["marketing_agent", "supply_agent", "finance_agent"],
            )
            self.assertEqual(metadata["followup_agents"], ["marketing_agent"])
            self.assertLessEqual(
                len(metadata["secondary_agents"]) + len(metadata["followup_agents"]),
                MAX_TOTAL_FOLLOWUP_DOMAIN_CALLS,
            )
        finally:
            for patcher in reversed(patches):
                patcher.stop()

    def test_cross_agent_request_metadata_survives_dependency_plan(self):
        plan = build_cross_domain_dependency_plan(
            user_query="Посчитай оборачиваемость склада",
            primary_agent="sales",
            cross_agent_requests=[
                {
                    "target_agent": "supply",
                    "reason": "Нужен средний запас.",
                    "question": "Проверь остатки и средний запас.",
                    "period": "2026-05-24..2026-06-22",
                    "vendor_codes": ["46227", "46303"],
                    "nm_ids": [70194686, 171742855],
                    "required_checks": ["средний запас", "остатки по складам"],
                }
            ],
            max_secondary_agents=3,
        )

        item = plan["recommended_secondary_agents"][0]
        self.assertEqual(item["target_agent"], "supply")
        self.assertEqual(item["period"], "2026-05-24..2026-06-22")
        self.assertEqual(item["vendor_codes"], ["46227", "46303"])
        self.assertEqual(item["nm_ids"], [70194686, 171742855])
        self.assertEqual(item["required_checks"], ["средний запас", "остатки по складам"])

    def test_secondary_task_prompt_preserves_requested_targets(self):
        prompt = _build_secondary_task_prompt(
            user_input="Посчитай оборачиваемость склада",
            item={
                "target_agent": "supply",
                "reason": "Нужен средний запас.",
                "question": "Проверь остатки и средний запас.",
                "period": "2026-05-24..2026-06-22",
                "vendor_codes": ["46227", "46303"],
                "nm_ids": [70194686, 171742855],
                "required_checks": ["средний запас", "остатки по складам"],
            },
            primary_output=self._sales_output(),
        )

        self.assertIn('"period": "2026-05-24..2026-06-22"', prompt)
        self.assertIn('"vendor_codes"', prompt)
        self.assertIn('"46227"', prompt)
        self.assertIn('"nm_ids"', prompt)
        self.assertIn('"средний запас"', prompt)

    def test_secondary_turnover_prompt_uses_default_full_30_day_period(self):
        prompt = _build_secondary_task_prompt(
            user_input="Посчитай оборачиваемость склада и деньги в текущих остатках.",
            item={
                "target_agent": "supply",
                "question": "Проверь оборачиваемость и текущий sellable-остаток.",
            },
            primary_output=self._sales_output(),
            workflow_version=WorkflowVersion.V2_OPTIMIZED,
            round_type=RoundType.CROSS_CHECK,
            required_tools=["get_inventory_turnover", "get_current_inventory_value"],
        )

        payload = json.loads(prompt)
        self.assertEqual(
            payload["round_task"]["period"],
            "last complete 30-day period ending at the latest available database date",
        )

        prompt_with_snapshot = _build_secondary_task_prompt(
            user_input="Посчитай оборачиваемость склада и деньги в текущих остатках.",
            item={
                "target_agent": "supply",
                "question": "Проверь оборачиваемость и текущий sellable-остаток.",
                "period": "current snapshot / latest available",
            },
            primary_output=self._sales_output(),
            workflow_version=WorkflowVersion.V2_OPTIMIZED,
            round_type=RoundType.CROSS_CHECK,
            required_tools=["get_inventory_turnover", "get_current_inventory_value"],
        )
        snapshot_payload = json.loads(prompt_with_snapshot)
        self.assertIn("turnover: last complete 30-day period", snapshot_payload["round_task"]["period"])

    def test_broad_sales_profit_diagnostic_runs_action_safety_round_two(self):
        query = (
            "Почему за последнюю неделю изменились продажи и прибыль по ключевым SKU? "
            "Проверь гипотезы: спрос, реклама, цена, остатки и экономика. "
            "Разложи вклад по SKU. Покажи подтверждённые и неподтверждённые причины. "
            "Для каждого предлагаемого действия сделай второй раунд проверки: "
            "если снижать цену - проверь маржу; если масштабировать рекламу - проверь запас и экономику; "
            "если дозаказывать - проверь спрос; если распродавать - проверь цену и маржу. "
            "В конце дай ровно 5 задач, которые можно сразу поставить в работу."
        )
        final_text = (
            "# Диагностика SKU\n\n"
            "Краткий вывод: прибыль механически изменилась через объём, первичные причины проверены по SKU.\n\n"
            "Подтверждено: причины собраны.\n\n"
            "Неподтверждено: часть гипотез снята.\n\n"
            "Что делать:\n"
            "| # | Задача | Объект | Почему |\n"
            "| --- | --- | --- | --- |\n"
            "| 1 | Проверить доступный остаток | 46230, 46306, 46225 + ещё 2 | OOS или критично низкий запас |\n"
            "| 2 | Остановить слабую рекламу | 46312 | расход есть, заказов 0 |\n"
            "| 3 | Рассчитать ценовой тест | 46230, 46303 | маржа есть, цена выше рынка |\n"
            "| 4 | Проверить карточку и корзину | 46303, 46230 | заказы и конверсия просели |\n"
            "| 5 | Удержать рекламу без расширения | 46227 | реклама рабочая, причина падения не подтверждена |"
        )
        patches, started = self._start_mocks(
            query,
            QueryIntent.FINANCE,
            output=[
                self._agent_output("Finance Agent"),
                self._neutral_sales_output(),
                self._agent_output("Marketing Agent"),
                self._agent_output("Price Agent"),
                self._agent_output("Supply Agent"),
                self._agent_output("Price Agent"),
                self._agent_output("Supply Agent"),
                self._agent_output("Marketing Agent"),
                self._agent_output("Sales Agent"),
                final_text,
            ],
        )
        try:
            mock_run_sync = started[10]
            mock_append_message = started[14]
            mock_persist_tasks = started[19]

            result = run_main_agent(query, chat_id="chat-1")

            self.assertIn("Диагностика SKU", result)
            self.assertNotIn("nm_id", result.lower())
            self.assertNotIn("171742855", result)
            self.assertEqual(mock_run_sync.call_count, 10)
            self.assertEqual(mock_run_sync.call_args_list[0].args[0], "finance-agent")
            self.assertEqual(
                [call.args[0] for call in mock_run_sync.call_args_list[1:5]],
                ["sales-agent", "marketing-agent", "price-agent", "supply-agent"],
            )
            self.assertEqual(
                [call.args[0] for call in mock_run_sync.call_args_list[5:9]],
                ["price-agent", "supply-agent", "marketing-agent", "sales-agent"],
            )
            metadata = mock_append_message.call_args.kwargs["metadata"]
            self.assertEqual(metadata["primary_agent"], "finance_agent")
            self.assertEqual(
                metadata["secondary_agents"],
                ["sales_agent", "marketing_agent", "price_agent", "supply_agent"],
            )
            self.assertEqual(
                metadata["followup_agents"],
                ["price_agent", "supply_agent", "marketing_agent", "sales_agent"],
            )
            self.assertTrue(
                all(
                    item["source"] == "action_safety"
                    for item in metadata["dependency_plans"][1]["recommended_secondary_agents"]
                )
            )
            self.assertGreaterEqual(len(metadata["action_safety_checks"]), 4)
            self.assertEqual(len(metadata["feature3_actions"]), 5)
            self.assertTrue(
                all(item["source"] == "final_table_action" for item in metadata["feature3_actions"])
            )
            self.assertTrue(
                all("nm_id" not in item["action"].lower() for item in metadata["feature3_actions"])
            )
            self.assertEqual(metadata["final_quality_violations"], [])
            mock_persist_tasks.assert_called_once()
            self.assertEqual(mock_persist_tasks.call_args.kwargs["chat_id"], "chat-1")
            self.assertEqual(mock_persist_tasks.call_args.kwargs["server_message_id"], 2)
            self.assertEqual(mock_persist_tasks.call_args.kwargs["intent"], "finance")
            self.assertEqual(len(mock_persist_tasks.call_args.kwargs["raw_actions"]), 5)
        finally:
            for patcher in reversed(patches):
                patcher.stop()

    def test_diagnostic_workflow_does_not_run_round_two_for_data_gaps_only(self):
        supply_output = self._agent_output("Supply Agent")
        supply_output.data_gaps = [
            DataGap(
                metric="Маркетинговая видимость",
                reason="Это не supply-метрика.",
                impact="Не влияет на supply-проверку остатков.",
            )
        ]
        patches, started = self._start_mocks(
            "Почему изменились продажи?",
            QueryIntent.SALES,
            output=[
                self._sales_output(),
                supply_output,
                self._agent_output("Marketing Agent"),
                "# Почему изменились продажи\n\nКраткий вывод: диагностика собрана.",
            ],
        )
        try:
            mock_run_sync = started[10]
            mock_append_message = started[14]

            result = run_main_agent("Почему изменились продажи?", chat_id="chat-1")

            self.assertIn("Краткий вывод", result)
            self.assertEqual(mock_run_sync.call_count, 4)
            metadata = mock_append_message.call_args.kwargs["metadata"]
            self.assertEqual(metadata["secondary_agents"], ["supply_agent", "marketing_agent"])
            self.assertEqual(metadata["followup_agents"], [])
            self.assertEqual(metadata["runtime_failures"], [])
            self.assertEqual(
                metadata["resolved_gaps"][0]["metric"],
                "Маркетинговая видимость",
            )
        finally:
            for patcher in reversed(patches):
                patcher.stop()

    def test_cross_domain_gap_is_resolved_when_neighbor_evidence_exists(self):
        marketing_output = AgentDiagnosticResult(
            agent_name="Marketing Agent",
            summary="ДРР требует фактическую выручку продаж.",
            tables=[
                ReportTable(
                    title="Расходы рекламы",
                    columns=["Период", "Расход"],
                    rows=[["2026-04", 1000]],
                )
            ],
            data_gaps=[
                DataGap(
                    metric="Фактическая выручка и ДРР",
                    reason="Нельзя корректно посчитать ДРР без данных sales.",
                    impact="Нельзя связать рекламные расходы с фактической выручкой продаж.",
                )
            ],
        )
        sales_output = AgentDiagnosticResult(
            agent_name="Sales Agent",
            summary="Фактическая выручка продаж за период получена.",
            tables=[
                ReportTable(
                    title="Продажи",
                    columns=["Период", "Выручка"],
                    rows=[["2026-04", 5000]],
                )
            ],
            findings=[self._finding("Sales revenue", "Фактическая выручка подтверждена.")],
        )

        reduced = _reduce_evidence(
            primary_agent=QueryIntent.MARKETING,
            primary_output=marketing_output,
            round_1_evidence=[
                {
                    "round": "round_1",
                    "target_agent": "sales",
                    "reason": "ДРР требует sales-выручку.",
                    "question": "Проверь фактическую выручку.",
                    "output": _diagnostic_payload(sales_output),
                }
            ],
            followup_evidence=[],
        )

        self.assertEqual(reduced["blocking_gaps"], [])
        self.assertEqual(reduced["resolved_gaps"][0]["metric"], "Фактическая выручка и ДРР")
        self.assertEqual(reduced["resolved_gaps"][0]["resolved_by"], "sales")

    def test_sales_layer_gaps_resolve_to_marketing_and_finance(self):
        sales_output = self._neutral_sales_output()
        sales_output.data_gaps = [
            DataGap(
                metric="traffic source attribution",
                reason="В sales-слое нет явной привязки продаж к рекламным кампаниям.",
                impact="Нельзя подтвердить рекламную инкрементальность внутри sales-домена.",
            ),
            DataGap(
                metric="profitability",
                reason="В sales-слое нет комиссий, логистики, хранения, штрафов и COGS.",
                impact="Нельзя подтвердить profitability after costs внутри sales-домена.",
            ),
        ]
        marketing_output = AgentDiagnosticResult(
            agent_name="Marketing Agent",
            summary="Рекламная проверка выполнена.",
            tables=[ReportTable(title="Реклама", columns=["Метрика"], rows=[["Клики"]])],
            findings=[self._finding("Marketing", "Рекламные метрики проверены.")],
        )
        finance_output = AgentDiagnosticResult(
            agent_name="Finance Agent",
            summary="Финансовая проверка выполнена.",
            tables=[ReportTable(title="Финансы", columns=["Метрика"], rows=[["Маржа"]])],
            findings=[self._finding("Finance", "Маржа и комиссии проверены.")],
        )

        reduced = _reduce_evidence(
            primary_agent=QueryIntent.SALES,
            primary_output=sales_output,
            round_1_evidence=[
                {"round": "round_1", "target_agent": "marketing", "output": _diagnostic_payload(marketing_output)},
                {"round": "round_1", "target_agent": "finance", "output": _diagnostic_payload(finance_output)},
            ],
            followup_evidence=[],
        )

        resolved_by_metric = {
            gap["metric"]: gap["resolved_by"]
            for gap in reduced["resolved_gaps"]
        }
        self.assertEqual(reduced["blocking_gaps"], [])
        self.assertEqual(resolved_by_metric["traffic source attribution"], "marketing")
        self.assertEqual(resolved_by_metric["profitability"], "finance")

    def test_sales_data_gap_entries_target_neighbor_domains(self):
        payload = {
            "data_gaps": [
                {
                    "metric": "traffic source attribution",
                    "reason": "В sales-слое нет привязки продаж к рекламным кампаниям.",
                    "impact": "Нельзя подтвердить рекламную инкрементальность.",
                },
                {
                    "metric": "profitability",
                    "reason": "В sales-слое нет комиссий, логистики и COGS.",
                    "impact": "Нельзя подтвердить прибыльность после всех затрат.",
                },
            ]
        }

        entries = _gap_entries_from_payload(
            payload=payload,
            source_agent="sales",
            blocking=True,
        )

        self.assertEqual(entries[0]["target_agent"], "marketing")
        self.assertEqual(entries[1]["target_agent"], "finance")

    def test_sku_text_does_not_route_price_or_supply_gaps_to_sales(self):
        price_entries = _gap_entries_from_payload(
            payload={
                "data_gaps": [
                    {
                        "metric": "current price / promo price / market median for SKU 46227",
                        "reason": "Конкурентный price-срез по SKU пустой.",
                        "impact": "Нельзя подтвердить ценовую первопричину.",
                    }
                ]
            },
            source_agent="price",
            blocking=True,
        )
        supply_entries = _gap_entries_from_payload(
            payload={
                "data_gaps": [
                    {
                        "metric": "остатки и распределение по складам для SKU 46311",
                        "reason": "В supply-срезе нет подтвержденных строк по SKU.",
                        "impact": "Нельзя подтвердить складскую доступность.",
                    }
                ]
            },
            source_agent="supply",
            blocking=True,
        )
        marketing_entries = _gap_entries_from_payload(
            payload={
                "data_gaps": [
                    {
                        "metric": "SKU-уровень: показы, клики, CTR, CPC, заказы",
                        "reason": "Доступный marketing snapshot не вернул строки.",
                        "impact": "Нельзя подтвердить рекламную детализацию.",
                    }
                ]
            },
            source_agent="marketing",
            blocking=True,
        )

        self.assertEqual(price_entries[0]["target_agent"], "price")
        self.assertEqual(supply_entries[0]["target_agent"], "supply")
        self.assertEqual(marketing_entries[0]["target_agent"], "marketing")

    def test_gap_routing_distinguishes_impressions_from_generic_indicator(self):
        sales_entries = _gap_entries_from_payload(
            payload={
                "data_gaps": [
                    {
                        "metric": "Показатель выручки по SKU",
                        "reason": "Нужно проверить продажи и фактическую выручку.",
                        "impact": "Нельзя подтвердить динамику продаж.",
                    }
                ]
            },
            source_agent="finance",
            blocking=True,
        )
        supply_entries = _gap_entries_from_payload(
            payload={
                "data_gaps": [
                    {
                        "metric": "Показатель оборачиваемости и покрытия запасом",
                        "reason": "Нет расчета days of stock по SKU.",
                        "impact": "Нельзя оценить потребность в поставке.",
                    }
                ]
            },
            source_agent="finance",
            blocking=True,
        )
        marketing_entries = _gap_entries_from_payload(
            payload={
                "data_gaps": [
                    {
                        "metric": "Показы и CTR по SKU",
                        "reason": "Нет строк рекламной кампании.",
                        "impact": "Нельзя проверить видимость.",
                    }
                ]
            },
            source_agent="sales",
            blocking=True,
        )

        self.assertEqual(sales_entries[0]["target_agent"], "sales")
        self.assertEqual(supply_entries[0]["target_agent"], "supply")
        self.assertEqual(marketing_entries[0]["target_agent"], "marketing")

    def test_net_profit_gap_with_marketplace_word_stays_in_finance(self):
        entries = _gap_entries_from_payload(
            payload={
                "data_gaps": [
                    {
                        "metric": "net profit",
                        "reason": "Нет полного P&L и расходов вне маркетплейса.",
                        "impact": "Нельзя подтвердить чистую прибыль.",
                    }
                ]
            },
            source_agent="finance",
            blocking=True,
        )

        self.assertEqual(entries[0]["target_agent"], "finance")

    def test_sync_optimized_gap_dedupe_keeps_no_data_terminal(self):
        self.assertEqual(OPTIMIZED_SECONDARY_QUESTION_BUDGET, 4)
        gaps = [
            SimpleNamespace(
                target_agent="price",
                source_agent="price",
                sku="46227",
                metric="Средняя цена за последние 30 дней",
                reason="Источник не вернул строки за последние 30 дней",
                evidence_status="no_data",
            ),
            SimpleNamespace(
                target_agent="price",
                source_agent="price",
                sku="46227",
                metric="Средняя цена за последние 30 дней",
                reason="Нужно повторить проверку за последние 30 дней",
                evidence_status="unconfirmed",
            ),
        ]

        self.assertEqual(_deduplicate_optimized_gap_records(gaps), [])
        self.assertEqual(_deduplicate_optimized_gap_records(list(reversed(gaps))), [])

    def test_sync_typed_cross_check_no_data_persists_as_terminal_gap(self):
        entries = _gap_entries_from_payload(
            payload={
                "_round_result": {
                    "round_type": "cross_check",
                    "verdict": "no_data",
                    "facts": [],
                },
                "data_gaps": [
                    {
                        "metric": "Средняя цена за последние 30 дней",
                        "reason": "Источник не вернул строки за последние 30 дней.",
                        "impact": "Ценовой фактор не подтвержден.",
                    }
                ],
            },
            source_agent="price",
            blocking=True,
        )

        self.assertEqual(entries[0]["evidence_status"], "no_data")
        terminal_gap = SimpleNamespace(sku="46227", **entries[0])
        stale_unconfirmed_gap = SimpleNamespace(
            target_agent="price",
            source_agent="price",
            sku="46227",
            metric="Средняя цена за последние 30 дней",
            reason="Повторить проверку за последние 30 дней.",
            evidence_status="unconfirmed",
        )
        self.assertEqual(
            _deduplicate_optimized_gap_records([stale_unconfirmed_gap, terminal_gap]),
            [],
        )
        self.assertEqual(
            _deduplicate_optimized_gap_records([terminal_gap, stale_unconfirmed_gap]),
            [],
        )

    def test_optimized_report_does_not_repair_synthesis(self):
        sales_output = self._neutral_sales_output()
        sales_output.cross_agent_requests = []
        patches, started = self._start_mocks(
            "Покажи продажи за неделю",
            QueryIntent.SALES,
            output=[
                sales_output,
                "# Продажи за неделю\n\nЧерновик требует quality repair.",
            ],
        )
        started[0].return_value.agent_workflow_version = "v2_optimized"
        try:
            with (
                patch(
                    "agi_ecom_app.runner.build_domain_agent_for_round",
                    return_value="v2-sales-agent",
                ),
                patch(
                    "agi_ecom_app.runner.validate_prompt_v2_answer",
                    return_value=["forced quality violation"],
                ),
                patch("agi_ecom_app.runner.build_prompt_v2_repair_prompt") as repair_prompt,
            ):
                run_main_agent("Покажи продажи за неделю", chat_id="chat-1")

            self.assertEqual(started[10].call_count, 2)
            repair_prompt.assert_not_called()
        finally:
            for patcher in reversed(patches):
                patcher.stop()

    def test_sync_primary_validation_failure_persists_prompt_report(self):
        class ModelBehaviorError(Exception):
            pass

        patches, started = self._start_mocks(
            "Почему изменились продажи?",
            QueryIntent.SALES,
        )
        started[0].return_value.agent_workflow_version = "v2_optimized"
        started[10].side_effect = ModelBehaviorError("facts.11: unit is required")
        try:
            with patch(
                "agi_ecom_app.runner.build_domain_agent_for_round",
                return_value="v2-sales-agent",
            ):
                with self.assertRaisesRegex(
                    AgentRuntimeError,
                    "Primary domain agent failed validation",
                ):
                    run_main_agent("Почему изменились продажи?", chat_id="chat-1")

            failed_response = started[24].call_args.kwargs
            self.assertEqual(failed_response["status"], "failed")
            self.assertEqual(
                failed_response["runtime_failure"]["error_type"],
                "ModelBehaviorError",
            )
            self.assertGreater(
                failed_response["raw_output"]["_prompt_report"]["input_chars"],
                0,
            )
            self.assertFalse(failed_response["has_business_evidence"])
        finally:
            for patcher in reversed(patches):
                patcher.stop()

    def test_sync_primary_transport_failure_is_not_persisted_as_evidence(self):
        class APIConnectionError(Exception):
            pass

        patches, started = self._start_mocks(
            "Почему изменились продажи?",
            QueryIntent.SALES,
        )
        started[0].return_value.agent_workflow_version = "v2_optimized"
        started[10].side_effect = APIConnectionError("temporary transport failure")
        try:
            with (
                patch(
                    "agi_ecom_app.runner.build_domain_agent_for_round",
                    return_value="v2-sales-agent",
                ),
                patch("agi_ecom_app.runner.time.sleep"),
            ):
                with self.assertRaisesRegex(AgentRuntimeError, "temporary transport failure"):
                    run_main_agent("Почему изменились продажи?", chat_id="chat-1")

            self.assertEqual(started[10].call_count, 2)
            started[24].assert_not_called()
        finally:
            for patcher in reversed(patches):
                patcher.stop()

    def test_persisted_blocking_gaps_are_not_reintroduced_after_neighbor_evidence(self):
        sales_output = self._neutral_sales_output()
        marketing_output = AgentDiagnosticResult(
            agent_name="Marketing Agent",
            summary="Рекламная проверка выполнена.",
            tables=[ReportTable(title="Реклама", columns=["Метрика"], rows=[["Клики"]])],
            findings=[self._finding("Marketing", "Рекламные метрики проверены.")],
        )

        package = _build_synthesis_package(
            user_input="Сформируй отчет по продажам.",
            primary_agent=QueryIntent.SALES,
            primary_output=sales_output,
            round_1_plan={},
            round_1_evidence=[
                {"round": "round_1", "target_agent": "marketing", "output": _diagnostic_payload(marketing_output)}
            ],
            round_2_plan={},
            followup_evidence=[],
            persisted_blocking_gaps=[
                {
                    "source_agent": "sales",
                    "target_agent": "marketing",
                    "metric": "traffic source attribution",
                    "reason": "В sales-слое нет явной привязки продаж к рекламным кампаниям.",
                    "impact": "Нельзя подтвердить рекламную инкрементальность внутри sales-домена.",
                    "blocking": True,
                    "status": "open",
                }
            ],
        )

        self.assertFalse(
            any(gap.get("metric") == "traffic source attribution" for gap in package["blocking_gaps"])
        )
        self.assertTrue(
            any(
                gap.get("metric") == "traffic source attribution"
                and gap.get("resolved_by") == "marketing"
                for gap in package["resolved_gaps"]
            )
        )

    def test_price_scenario_decision_demotes_exact_model_gaps(self):
        price_output = AgentDiagnosticResult(
            agent_name="Price Agent",
            summary="Для медленного SKU есть сценарная стартовая цена.",
            tables=[
                ReportTable(
                    title="Ценовой сценарий",
                    columns=[
                        "SKU",
                        "Текущая цена",
                        "Рекомендованная цена",
                        "expected_orders_7d",
                        "expected_margin_pct",
                        "Уверенность",
                    ],
                    rows=[["46224", "2075", "1765", "5.53", "42.05", "low"]],
                )
            ],
            data_gaps=[
                DataGap(
                    metric="Точный прогноз продаж при скидке -10%",
                    reason="Нет исторического price-volume ряда по SKU.",
                    impact="Нельзя подтвердить точный uplift до единицы продажи.",
                )
            ],
        )

        package = _build_synthesis_package(
            user_input="Какую цену на товар с низкой оборачиваемостью мы можем поставить?",
            primary_agent=QueryIntent.PRICE,
            primary_output=price_output,
            round_1_plan={},
            round_1_evidence=[],
            round_2_plan={},
            followup_evidence=[],
        )

        self.assertEqual(package["decision_mode"]["mode"], "scenario_decision")
        self.assertEqual(package["blocking_gaps"], [])
        self.assertEqual(package["scenario_limitations"][0]["status"], "scenario_limitation")
        self.assertFalse(package["scenario_limitations"][0]["blocking"])
        self.assertTrue(
            any(
                gap.get("metric") == "Точный прогноз продаж при скидке -10%"
                for gap in package["non_blocking_gaps"]
            )
        )

    def test_price_decision_without_scenario_keeps_exact_model_gap_blocking(self):
        price_output = AgentDiagnosticResult(
            agent_name="Price Agent",
            summary="Точной сценарной цены нет.",
            tables=[
                ReportTable(
                    title="Конкурентный срез",
                    columns=["SKU", "Медиана конкурентов"],
                    rows=[["46224", "1800"]],
                )
            ],
            data_gaps=[
                DataGap(
                    metric="Точный прогноз продаж при скидке -10%",
                    reason="Нет исторического price-volume ряда по SKU.",
                    impact="Нельзя подтвердить точный uplift до единицы продажи.",
                )
            ],
        )

        package = _build_synthesis_package(
            user_input="Какую цену на товар с низкой оборачиваемостью мы можем поставить?",
            primary_agent=QueryIntent.PRICE,
            primary_output=price_output,
            round_1_plan={},
            round_1_evidence=[],
            round_2_plan={},
            followup_evidence=[],
        )

        self.assertEqual(package["decision_mode"]["mode"], "decision_needed")
        self.assertTrue(
            any(
                gap.get("metric") == "Точный прогноз продаж при скидке -10%"
                for gap in package["blocking_gaps"]
            )
        )
        self.assertIsNone(package["scenario_limitations"])

    def test_persisted_pseudo_sku_gaps_are_not_reintroduced(self):
        sales_output = self._neutral_sales_output()

        package = _build_synthesis_package(
            user_input="Сформируй отчет по продажам.",
            primary_agent=QueryIntent.SALES,
            primary_output=sales_output,
            round_1_plan={},
            round_1_evidence=[],
            round_2_plan={},
            followup_evidence=[],
            persisted_blocking_gaps=[
                {
                    "source_agent": "finance",
                    "target_agent": "finance",
                    "sku": "46313",
                    "metric": "finance factor",
                    "reason": "В sku-отчёте ряд SKU имеет положительную маржу.",
                    "impact": "Фактор нельзя использовать как подтвержденную причину.",
                    "blocking": True,
                    "status": "open",
                    "evidence_status": "unconfirmed",
                },
                {
                    "source_agent": "price",
                    "target_agent": "price",
                    "sku": "46313",
                    "metric": "price factor",
                    "reason": "sku_not_in_source",
                    "impact": "Фактор нельзя использовать как подтвержденную причину.",
                    "blocking": True,
                    "status": "open",
                    "evidence_status": "unconfirmed",
                },
            ],
        )

        self.assertFalse(
            any(gap.get("metric") == "finance factor" for gap in package["blocking_gaps"])
        )
        self.assertFalse(
            any(gap.get("metric") == "price factor" for gap in package["blocking_gaps"])
        )

    def test_sku_matrix_does_not_persist_pseudo_sku_source_gaps(self):
        gaps = _gap_entries_from_sku_matrix(
            {
                "rows": [
                    {
                        "sku": "46313",
                        "domains": {
                            "finance": {
                                "status": "unconfirmed",
                                "fact": "В sku-отчёте ряд SKU имеет положительную маржу.",
                                "reason": None,
                            },
                            "price": {
                                "status": "unconfirmed",
                                "fact": "фактор не подтвержден по доступным данным",
                                "reason": "sku_not_in_source",
                            },
                        },
                    }
                ]
            }
        )

        self.assertEqual(gaps, [])

    def test_non_diagnostic_turnover_query_omits_sku_evidence_matrix_from_synthesis(self):
        package = _build_synthesis_package(
            user_input="Посчитай оборачиваемость склада",
            primary_agent=QueryIntent.SALES,
            primary_output=self._neutral_sales_output(),
            round_1_plan={},
            round_1_evidence=[],
            round_2_plan={},
            followup_evidence=[],
        )

        self.assertEqual(package["sku_evidence_matrix"], [])
        self.assertEqual(package["domain_coverage"], {})
        self.assertFalse(package["format_policy"]["sku_cause_action_matrix"])
        self.assertTrue(package["sku_metric_matrix_required"])
        self.assertTrue(package["format_policy"]["sku_metric_matrix"])
        self.assertTrue(package["sku_metric_matrix"])

    def test_diagnostic_sales_query_keeps_sku_evidence_matrix_for_synthesis(self):
        package = _build_synthesis_package(
            user_input="Почему изменились продажи по SKU?",
            primary_agent=QueryIntent.SALES,
            primary_output=self._neutral_sales_output(),
            round_1_plan={},
            round_1_evidence=[],
            round_2_plan={},
            followup_evidence=[],
        )

        self.assertTrue(package["sku_evidence_matrix"])
        self.assertTrue(package["format_policy"]["sku_cause_action_matrix"])

    def test_guardrail_failure_with_structured_evidence_is_not_blocking_gap(self):
        finance_output = AgentDiagnosticResult(
            agent_name="Finance Agent",
            summary="Финансы требуют проверки спроса.",
            tables=[ReportTable(title="Финансы", columns=["Метрика"], rows=[["Маржа"]])],
            findings=[self._finding("Finance", "Маржа изменилась.")],
        )
        sales_output = self._neutral_sales_output()
        sales_output.findings = [
            self._finding("Sales demand", "Продажи и выручка по SKU рассчитаны.")
        ]

        reduced = _reduce_evidence(
            primary_agent=QueryIntent.FINANCE,
            primary_output=finance_output,
            round_1_evidence=[
                {
                    "round": "round_1",
                    "target_agent": "sales",
                    "reason": "Нужен спрос.",
                    "question": "Проверь фактические продажи.",
                    "output": _diagnostic_payload(sales_output),
                    "runtime_failure": {
                        "target_agent": "sales",
                        "kind": "guardrail",
                        "message": "Смежная проверка не прошла внутреннюю проверку качества.",
                        "violations": ["sales_decomposition is missing current_sales"],
                    },
                }
            ],
            followup_evidence=[],
        )

        self.assertEqual(reduced["blocking_gaps"], [])
        self.assertEqual(reduced["non_blocking_gaps"][0]["metric"], "sales answer quality")
        self.assertTrue(
            any(fact["source_agent"] == "sales" for fact in reduced["confirmed_facts"])
        )

    def test_sku_evidence_matrix_uses_unconfirmed_for_called_domain_missing_sku(self):
        finance_output = AgentDiagnosticResult(
            agent_name="Finance Agent",
            summary="Маржа по SKU рассчитана.",
            tables=[
                ReportTable(
                    title="Изменение маржи по SKU",
                    columns=["Артикул", "Δ маржи, руб", "Рентабельность тек. период, %"],
                    rows=[
                        ["46227", -41004.31, 57.04],
                        ["46313", 15894.05, 51.14],
                    ],
                )
            ],
            findings=[self._finding("Finance", "SKU 46227 и SKU 46313 рассчитаны.")],
        )
        sales_output = SalesDiagnosticResult(
            agent_name="Sales Agent",
            summary="Продажи по SKU 46227 рассчитаны.",
            sales_overview=ReportTable(title="Обзор", columns=["Артикул", "Заказы"], rows=[["46227", 60]]),
            sales_dynamics=ReportTable(title="Динамика", columns=["Дата", "Продажи"], rows=[["2026-04-16", 10]]),
            sales_decomposition=ReportTable(
                title="Вклад артикулов",
                columns=[
                    "Артикул",
                    "Текущая выручка",
                    "Предыдущая выручка",
                    "Δ выручки",
                    "Текущие продажи",
                    "Предыдущие продажи",
                    "Δ продаж",
                ],
                rows=[["46227", 1000, 1200, -200, 10, 12, -2]],
            ),
            findings=[self._finding("Sales", "SKU 46227: продажи снизились.")],
        )

        matrix = _build_sku_evidence_matrix(
            user_input="Сформируй финансовый отчет.",
            primary_agent=QueryIntent.FINANCE,
            primary_payload=_diagnostic_payload(finance_output),
            round_1_evidence=[
                {
                    "round": "round_1",
                    "target_agent": "sales",
                    "output": _diagnostic_payload(sales_output),
                }
            ],
            followup_evidence=[],
        )

        rows = {row["sku"]: row for row in matrix["rows"]}
        self.assertIn("46313", rows)
        self.assertEqual(rows["46227"]["domains"]["sales"]["status"], "confirmed")
        self.assertEqual(rows["46313"]["domains"]["sales"]["status"], "no_data")
        self.assertEqual(rows["46313"]["domains"]["sales"]["reason"], "sku_not_in_source")
        self.assertEqual(rows["46313"]["domains"]["price"]["status"], "not_checked")
        self.assertEqual(rows["46313"]["domains"]["price"]["reason"], "domain_not_called")
        self.assertEqual(rows["46313"]["domains"]["finance"]["status"], "confirmed")
        self.assertIn("Δ маржи", rows["46313"]["domains"]["finance"]["fact"])

    def test_sku_evidence_matrix_marks_called_failed_domain_as_no_data(self):
        finance_output = AgentDiagnosticResult(
            agent_name="Finance Agent",
            summary="Маржа по SKU рассчитана.",
            tables=[
                ReportTable(
                    title="Изменение маржи по SKU",
                    columns=["Артикул", "Δ маржи, руб"],
                    rows=[["46227", -41004.31]],
                )
            ],
        )

        matrix = _build_sku_evidence_matrix(
            user_input="Сформируй финансовый отчет.",
            primary_agent=QueryIntent.FINANCE,
            primary_payload=_diagnostic_payload(finance_output),
            round_1_evidence=[
                {
                    "round": "round_1",
                    "target_agent": "marketing",
                    "output": {
                        "summary": "Смежную проверку не удалось выполнить.",
                        "tables": [],
                        "findings": [],
                        "data_gaps": [],
                        "cross_agent_requests": [],
                    },
                    "runtime_failure": {"kind": "exception", "target_agent": "marketing"},
                }
            ],
            followup_evidence=[],
        )

        row = matrix["rows"][0]
        self.assertEqual(row["sku"], "46227")
        self.assertEqual(row["domains"]["marketing"]["status"], "no_data")
        self.assertEqual(row["domains"]["marketing"]["reason"], "agent_failed_without_evidence")

    def test_sku_matrix_conclusion_does_not_turn_sales_mechanics_into_cause(self):
        sales_output = SalesDiagnosticResult(
            agent_name="Sales Agent",
            summary="Продажи по SKU рассчитаны.",
            sales_overview=ReportTable(title="Обзор", columns=["Артикул", "Заказы"], rows=[["46227", 60]]),
            sales_dynamics=ReportTable(title="Динамика", columns=["Дата", "Продажи"], rows=[["2026-04-16", 10]]),
            sales_decomposition=ReportTable(
                title="Вклад артикулов",
                columns=["Артикул", "Текущая выручка", "Предыдущая выручка", "Δ выручки"],
                rows=[["46227", 1000, 1200, -200]],
            ),
            findings=[self._finding("Sales", "SKU 46227: выручка и продажи снизились.")],
        )

        matrix = _build_sku_evidence_matrix(
            user_input="Почему упали продажи по SKU 46227?",
            primary_agent=QueryIntent.SALES,
            primary_payload=_diagnostic_payload(sales_output),
            round_1_evidence=[],
            followup_evidence=[],
        )

        row = matrix["rows"][0]
        self.assertEqual(row["sku"], "46227")
        self.assertEqual(row["conclusion"]["status"], "unconfirmed")
        self.assertEqual(row["summary"], "изменение подтверждено, первопричина не закрыта")

    def test_sales_status_workflow_calls_supply_and_marketing(self):
        sales_output = self._neutral_sales_output()
        sales_output.cross_agent_requests = []
        patches, started = self._start_mocks(
            "Что с продажами?",
            QueryIntent.SALES,
            output=[
                sales_output,
                self._agent_output("Supply Agent"),
                self._agent_output("Marketing Agent"),
                "# Что с продажами\n\nКраткий вывод: диагностика собрана.",
            ],
        )
        try:
            mock_run_sync = started[10]
            mock_append_message = started[14]

            result = run_main_agent("Что с продажами?", chat_id="chat-1")

            self.assertIn("Краткий вывод", result)
            self.assertEqual(mock_run_sync.call_count, 4)
            self.assertEqual(mock_run_sync.call_args_list[0].args[0], "sales-agent")
            self.assertEqual(mock_run_sync.call_args_list[1].args[0], "supply-agent")
            self.assertEqual(mock_run_sync.call_args_list[2].args[0], "marketing-agent")
            metadata = mock_append_message.call_args.kwargs["metadata"]
            self.assertEqual(
                metadata["secondary_agents"],
                ["supply_agent", "marketing_agent"],
            )
            self.assertNotIn("price_agent", metadata["secondary_agents"])
            self.assertEqual(metadata["followup_agents"], [])
        finally:
            for patcher in reversed(patches):
                patcher.stop()

    def test_inventory_turnover_causal_workflow_adds_coverage_and_claim_round(self):
        supply_output = self._agent_output("Supply Agent")
        supply_output.cross_agent_requests = []
        patches, started = self._start_mocks(
            "Почему оборачиваемость склада медленная?",
            QueryIntent.SUPPLY,
            output=[
                supply_output,
                self._agent_output("Price Agent"),
                self._agent_output("Marketing Agent"),
                self._sales_output(),
                self._agent_output("Finance Agent"),
                "# Почему оборачиваемость медленная\n\nКраткий вывод: причины проверены.",
            ],
        )
        try:
            mock_create_round = started[20]
            mock_create_question = started[22]
            mock_append_message = started[14]

            run_main_agent("Почему оборачиваемость склада медленная?", chat_id="chat-1")

            round_types = [call.kwargs["round_type"] for call in mock_create_round.call_args_list]
            self.assertIn("cross_check", round_types)
            self.assertIn("claim_verification", round_types)
            self.assertIn("synthesis", round_types)
            target_agents = [call.kwargs["target_agent"] for call in mock_create_question.call_args_list]
            for target in ("sales", "price", "marketing", "finance"):
                self.assertIn(target, target_agents)
            metadata = mock_append_message.call_args.kwargs["metadata"]
            self.assertEqual(metadata["claim_verification"]["status"], "complete")
            self.assertEqual(metadata["claim_verification"]["missing_domains"], [])
        finally:
            for patcher in reversed(patches):
                patcher.stop()

    def test_diagnostic_workflow_persists_round_questions_and_responses(self):
        sales_output = self._neutral_sales_output()
        sales_output.cross_agent_requests = []
        patches, started = self._start_mocks(
            "Что с продажами?",
            QueryIntent.SALES,
            output=[
                sales_output,
                self._agent_output("Supply Agent"),
                self._agent_output("Marketing Agent"),
                "# Что с продажами\n\nКраткий вывод: диагностика собрана.",
            ],
        )
        try:
            mock_create_round = started[20]
            mock_create_question = started[22]
            mock_save_response = started[24]

            run_main_agent("Что с продажами?", chat_id="chat-1")

            round_types = [call.kwargs["round_type"] for call in mock_create_round.call_args_list]
            self.assertIn("primary", round_types)
            self.assertIn("cross_check", round_types)
            self.assertIn("synthesis", round_types)
            target_agents = [call.kwargs["target_agent"] for call in mock_create_question.call_args_list]
            self.assertIn("sales", target_agents)
            self.assertIn("supply", target_agents)
            self.assertIn("marketing", target_agents)
            self.assertGreaterEqual(mock_save_response.call_count, 4)
        finally:
            for patcher in reversed(patches):
                patcher.stop()

    def test_gap_resolution_round_is_capped_at_configured_blocking_gaps(self):
        gaps = [
            SimpleNamespace(
                id=index,
                run_id=101,
                response_id=None,
                sku=f"463{index:02d}",
                source_agent="supply",
                target_agent="supply",
                metric="supply factor",
                reason="Нет подтверждения остатка.",
                impact="Нельзя делать сильный вывод.",
                blocking=True,
                status="open",
                evidence_status="no_data",
                created_at=None,
                updated_at=None,
            )
            for index in range(1, ROUND_3_GAP_RESOLUTION_LIMIT + 3)
        ]
        sales_output = self._neutral_sales_output()
        sales_output.cross_agent_requests = []
        patches, started = self._start_mocks(
            "Покажи продажи за неделю",
            QueryIntent.SALES,
            output=[
                sales_output,
                *[
                    self._agent_output("Supply Agent")
                    for _ in range(ROUND_3_GAP_RESOLUTION_LIMIT)
                ],
                "# Продажи за неделю\n\nКраткий вывод: отчёт собран.",
            ],
        )
        try:
            mock_run_sync = started[10]
            mock_create_round = started[20]
            mock_create_question = started[22]
            mock_list_gaps = started[26]
            mock_update_gap = started[28]
            mock_list_gaps.return_value = gaps

            run_main_agent("Покажи продажи за неделю", chat_id="chat-1")

            self.assertEqual(mock_run_sync.call_count, ROUND_3_GAP_RESOLUTION_LIMIT + 2)
            round_types = [call.kwargs["round_type"] for call in mock_create_round.call_args_list]
            self.assertIn("gap_resolution", round_types)
            gap_questions = [
                call
                for call in mock_create_question.call_args_list
                if call.kwargs["verification_goal"].get("round") == "round_3"
            ]
            self.assertEqual(len(gap_questions), ROUND_3_GAP_RESOLUTION_LIMIT)
            self.assertEqual(mock_update_gap.call_count, ROUND_3_GAP_RESOLUTION_LIMIT)
        finally:
            for patcher in reversed(patches):
                patcher.stop()

    def test_price_decision_workflow_calls_sales_and_finance(self):
        patches, started = self._start_mocks(
            "Нужно ли снизить цену?",
            QueryIntent.PRICE,
            output=[
                self._agent_output("Price Agent"),
                self._agent_output("Sales Agent"),
                self._agent_output("Finance Agent"),
                "# Цена\n\nКраткий вывод: решение по цене собрано.",
            ],
        )
        try:
            mock_run_sync = started[10]
            mock_append_message = started[14]

            result = run_main_agent("Нужно ли снизить цену?", chat_id="chat-1")

            self.assertIn("Краткий вывод", result)
            self.assertEqual(mock_run_sync.call_count, 4)
            self.assertEqual(mock_run_sync.call_args_list[0].args[0], "price-agent")
            self.assertEqual(mock_run_sync.call_args_list[1].args[0], "sales-agent")
            self.assertEqual(mock_run_sync.call_args_list[2].args[0], "finance-agent")
            metadata = mock_append_message.call_args.kwargs["metadata"]
            self.assertEqual(metadata["primary_agent"], "price_agent")
            self.assertEqual(metadata["secondary_agents"], ["sales_agent", "finance_agent"])
        finally:
            for patcher in reversed(patches):
                patcher.stop()

    def test_llm_router_primary_intent_overrides_keyword_fallback(self):
        query = "Какую цену на товар с низкой оборачиваемостью мы можем поставить?"
        patches, started = self._start_mocks(
            query,
            QueryIntent.SUPPLY,
            route_intent=QueryIntent.PRICE,
            output=[
                self._agent_output("Price Agent"),
                "# Цена\n\nКраткий вывод: решение по цене собрано.",
            ],
        )
        try:
            mock_run_sync = started[10]
            mock_append_message = started[14]

            result = run_main_agent(query, chat_id="chat-1")

            self.assertIn("Краткий вывод", result)
            self.assertEqual(mock_run_sync.call_args_list[0].args[0], "price-agent")
            metadata = mock_append_message.call_args.kwargs["metadata"]
            self.assertEqual(metadata["intent"], "price")
            self.assertEqual(metadata["primary_agent"], "price_agent")
            self.assertEqual(metadata["intent_router"]["intent"], "price")
        finally:
            for patcher in reversed(patches):
                patcher.stop()

    def test_marketing_conversion_with_price_signal_calls_price(self):
        patches, started = self._start_mocks(
            "Клики есть, заказов нет, возможно цена высокая",
            QueryIntent.MARKETING,
            output=[
                self._agent_output("Marketing Agent"),
                self._agent_output("Price Agent"),
                self._agent_output("Sales Agent"),
                self._agent_output("Supply Agent"),
                "# Конверсия\n\nКраткий вывод: проверка собрана.",
            ],
        )
        try:
            mock_run_sync = started[10]
            mock_append_message = started[14]

            result = run_main_agent(
                "Клики есть, заказов нет, возможно цена высокая",
                chat_id="chat-1",
            )

            self.assertIn("Краткий вывод", result)
            self.assertEqual(mock_run_sync.call_args_list[0].args[0], "marketing-agent")
            self.assertEqual(mock_run_sync.call_args_list[1].args[0], "price-agent")
            metadata = mock_append_message.call_args.kwargs["metadata"]
            self.assertEqual(metadata["primary_agent"], "marketing_agent")
            self.assertIn("price_agent", metadata["secondary_agents"])
            self.assertLessEqual(len(metadata["secondary_agents"]), 3)
        finally:
            for patcher in reversed(patches):
                patcher.stop()

    def test_sales_report_workflow_does_not_call_secondary_agents(self):
        sales_output = self._neutral_sales_output()
        sales_output.cross_agent_requests = []
        patches, started = self._start_mocks(
            "Покажи продажи за неделю",
            QueryIntent.SALES,
            output=[
                sales_output,
                "# Продажи за неделю\n\nКраткий вывод: отчёт собран.",
            ],
        )
        try:
            mock_run_sync = started[10]
            mock_append_message = started[14]

            result = run_main_agent("Покажи продажи за неделю", chat_id="chat-1")

            self.assertIn("Краткий вывод", result)
            self.assertEqual(mock_run_sync.call_count, 2)
            self.assertEqual(mock_run_sync.call_args_list[0].args[0], "sales-agent")
            self.assertEqual(mock_run_sync.call_args_list[1].args[0], "diagnostic-orchestrator")
            metadata = mock_append_message.call_args.kwargs["metadata"]
            self.assertEqual(metadata["secondary_agents"], [])
            self.assertEqual(metadata["followup_agents"], [])
        finally:
            for patcher in reversed(patches):
                patcher.stop()

    def test_simple_chat_uses_light_router(self):
        patches, started = self._start_mocks(
            "Привет",
            QueryIntent.FINANCE,
            output="Привет! Я на связи.",
            has_domain_signal=False,
        )
        try:
            mock_build_diagnostic_orchestrator = started[3]
            mock_build_domain_router = started[8]
            mock_run_sync = started[10]
            mock_create_trace = started[17]
            mock_complete_trace = started[18]

            result = run_main_agent("Привет", chat_id="chat-1")

            self.assertEqual(result, "Привет! Я на связи.")
            mock_build_diagnostic_orchestrator.assert_not_called()
            mock_build_domain_router.assert_called_once_with()
            self.assertEqual(mock_run_sync.call_args.args[:2], ("domain-router-agent", "Привет"))
            mock_create_trace.assert_called_once_with(
                chat_id="chat-1",
                intent="finance",
                client_message_id=None,
                primary_agent=None,
                summary=ANY,
            )
            self.assertEqual(mock_create_trace.call_args.kwargs["summary"]["status"], "running")
            self.assertEqual(mock_create_trace.call_args.kwargs["summary"]["model_label"], "openai:gpt-test")
            self.assertIn("intent_router", mock_create_trace.call_args.kwargs["summary"])
            trace_kwargs = mock_complete_trace.call_args.kwargs
            self.assertEqual(trace_kwargs["status"], "completed")
            trace_events = [event["event"] for event in trace_kwargs["events"]]
            self.assertIn("run_started", trace_events)
            self.assertIn("run_completed", trace_events)
        finally:
            for patcher in reversed(patches):
                patcher.stop()

    def test_summary_refresh_failure_does_not_block_main_answer(self):
        patches, started = self._start_mocks("Да, покажи расходы", QueryIntent.FINANCE)
        try:
            mock_append_message = started[14]
            mock_refresh_chat_summary = started[15]
            mock_refresh_chat_summary.side_effect = RuntimeError("summary timeout")

            result = run_main_agent("Да, покажи расходы", chat_id="chat-1")

            self.assertEqual(result, "Ответ собран.")
            self.assertIn("Ответ собран.", mock_append_message.call_args.kwargs["content"])
            mock_refresh_chat_summary.assert_called_once_with(chat_id="chat-1")
        finally:
            for patcher in reversed(patches):
                patcher.stop()

    def test_output_guardrail_tripwire_returns_rendered_agent_output(self):
        patches, started = self._start_mocks("Почему упали продажи?", QueryIntent.SALES)
        try:
            mock_run_sync = started[10]
            mock_append_message = started[14]
            mock_complete_trace = started[18]
            mock_run_sync.side_effect = OutputGuardrailTripwireTriggered(
                SimpleNamespace(
                    guardrail=SimpleNamespace(),
                    output=SimpleNamespace(output_info={"violations": ["missing evidence"]}),
                    agent_output="Продажи снизились из-за падения заказов.",
                )
            )

            result = run_main_agent("Почему упали продажи?", chat_id="chat-1")

            self.assertEqual(result, "Продажи снизились из-за падения заказов.")
            self.assertEqual(mock_append_message.call_args.kwargs["content"], result)
            self.assertEqual(mock_append_message.call_args.kwargs["metadata"]["intent"], "sales")
            self.assertEqual(
                mock_append_message.call_args.kwargs["metadata"]["guardrail_violations"],
                ["missing evidence"],
            )
            trace_kwargs = mock_complete_trace.call_args.kwargs
            self.assertEqual(trace_kwargs["status"], "guardrail_failed")
            self.assertIn(
                "guardrail_failed",
                [event["event"] for event in trace_kwargs["events"]],
            )
        finally:
            for patcher in reversed(patches):
                patcher.stop()

    @patch("agi_ecom_app.runner.create_agent_run_trace")
    @patch("agi_ecom_app.runner.get_assistant_reply_for_client_message_id")
    @patch("agi_ecom_app.runner.determine_query_intent")
    def test_duplicate_client_message_reuses_reply_without_new_trace(
        self,
        mock_determine_query_intent,
        mock_get_assistant_reply,
        mock_create_trace,
    ):
        mock_determine_query_intent.return_value = QueryIntent.FINANCE
        mock_get_assistant_reply.return_value = SimpleNamespace(
            content="Уже готово",
            metadata={"intent": "finance"},
            seq=7,
        )

        result = run_chat_turn(
            user_input="Повтори",
            chat_id="chat-1",
            client_message_id="client-dup-1",
        )

        self.assertEqual(result.answer, "Уже готово")
        self.assertEqual(result.assistant_message_seq, 7)
        self.assertTrue(result.reused_existing)
        mock_create_trace.assert_not_called()


if __name__ == "__main__":
    unittest.main()
