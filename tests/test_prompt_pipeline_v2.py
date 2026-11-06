from __future__ import annotations

import json
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import ValidationError

from agi_ecom_app.agent_queue import (
    MAX_OPTIMIZED_FAST_PATH_QUESTION_ROWS,
    MAX_OPTIMIZED_MODEL_EXECUTIONS,
    MAX_OPTIMIZED_PERSISTED_QUESTION_ROWS,
    OPTIMIZED_REPAIR_EXECUTION_RESERVE,
    OPTIMIZED_ROUTER_EXECUTIONS,
    OPTIMIZED_SYNTHESIS_QUESTION_RESERVE,
    _deduplicate_actionable_gaps,
    _question_to_plan_item,
    _remaining_optimized_question_slots,
    _synthesis_repair_allowed,
    _trace_workflow_version,
)
from agi_ecom_app.main_agent import (
    SUPPLY_PROMPT_PATH,
    _build_domain_agent_for_round,
    _diagnostic_orchestrator_instructions,
    _dynamic_instructions,
)
from agi_ecom_app.llm_runtime import LLMRuntimeSpec
from agi_ecom_app.prompt_pipeline import (
    DOMAIN_SYSTEM_MAX_CHARS,
    ROUND_CONTRACT_MAX_CHARS,
    SECONDARY_PROMPT_MAX_CHARS,
    SYNTHESIS_PACKAGE_MAX_CHARS,
    adapt_round_result_to_legacy,
    build_round_prompt,
    compact_secondary_context,
    compact_synthesis_package,
    determine_query_mode,
    domain_base_contract,
    memory_mode_for_round,
    round_contract,
)
from agi_ecom_app.prompt_quality import validate_prompt_v2_answer
from agi_ecom_app.runner import (
    ROUND_RESULT_TYPES,
    _build_secondary_task_prompt,
    _finalize_action_safety_checks,
    _load_persisted_evidence_safely,
    _reduce_evidence,
    _run_agent_once,
)
from agi_ecom_app.schemas import (
    ClaimVerificationResult,
    ChatMemoryMessage,
    ChatRole,
    CrossAgentRequest,
    CrossCheckResult,
    EvidenceFact,
    EcomRunContext,
    FollowupResult,
    GapResolutionResult,
    MemoryMode,
    QueryIntent,
    QueryMode,
    RoundTask,
    RoundType,
    WorkflowVersion,
)
from agi_ecom_app.services.memory import AgentRunResponseRecord


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "prompt_v2"


def _load_goldens() -> list[dict]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(FIXTURE_ROOT.glob("run_*.json"))
    ]


class PromptPipelineV2Tests(unittest.TestCase):
    def _task(
        self,
        round_type: RoundType,
        *,
        question: str = "Проверь оборачиваемость и остаток SKU 46227 за июнь.",
        verification_goal: dict | str | None = None,
        gap_id: int | str | None = None,
        claim_policy: str | None = None,
    ) -> RoundTask:
        return RoundTask(
            round_type=round_type,
            target_agent="supply",
            question=question,
            original_user_query="Покажи отчет по остаткам",
            verification_goal=verification_goal
            or {"metric": "inventory_turnover", "sku": "46227"},
            required_checks=["inventory_turnover", "stock_qty"],
            source="offline_regression",
            gap_id=gap_id,
            claim_policy=claim_policy,
            period="2026-06-01/2026-06-25",
            sku="46227",
            vendor_codes=["46227"],
            nm_ids=[],
            allowed_tools=["get_inventory_turnover", "get_supply_overview"],
            required_tools=["get_inventory_turnover"],
        )

    def test_each_round_has_a_contract_memory_mode_and_secondary_schema(self):
        expected_schemas = {
            RoundType.CROSS_CHECK: CrossCheckResult,
            RoundType.FOLLOWUP: FollowupResult,
            RoundType.GAP_RESOLUTION: GapResolutionResult,
            RoundType.CLAIM_VERIFICATION: ClaimVerificationResult,
        }
        self.assertEqual(ROUND_RESULT_TYPES, expected_schemas)

        expected_memory = {
            RoundType.ROUTER: MemoryMode.SUMMARY_RECENT,
            RoundType.PRIMARY: MemoryMode.SUMMARY_RECENT,
            RoundType.CROSS_CHECK: MemoryMode.NONE,
            RoundType.FOLLOWUP: MemoryMode.NONE,
            RoundType.GAP_RESOLUTION: MemoryMode.NONE,
            RoundType.CLAIM_VERIFICATION: MemoryMode.NONE,
            RoundType.SYNTHESIS: MemoryMode.SUMMARY_ONLY,
            RoundType.TERMINAL_ACTION: MemoryMode.FULL,
        }

        for round_type in RoundType:
            with self.subTest(round_type=round_type.value):
                task = self._task(round_type)
                result = build_round_prompt(
                    task,
                    {"summary": "Primary summary", "confirmed_facts": []},
                    [],
                    WorkflowVersion.V2_PROMPT,
                )
                contract = round_contract(round_type)
                self.assertTrue(contract)
                self.assertLessEqual(len(contract), ROUND_CONTRACT_MAX_CHARS)
                # Static contracts live in system instructions and must not be
                # duplicated in the per-question model input.
                self.assertNotIn(contract, result.text)
                self.assertEqual(
                    result.report.system_chars,
                    len(domain_base_contract(task.target_agent)) + 1 + len(contract),
                )
                self.assertEqual(result.report.round_type, round_type)
                self.assertEqual(result.report.memory_mode, expected_memory[round_type])
                self.assertEqual(memory_mode_for_round(round_type), expected_memory[round_type])

        for domain in ("sales", "marketing", "price", "supply", "finance", "synthesis"):
            with self.subTest(domain=domain):
                self.assertLessEqual(len(domain_base_contract(domain)), DOMAIN_SYSTEM_MAX_CHARS)

    def test_domain_contract_distinguishes_portfolio_focus_from_sku_list_filter(self):
        contract = domain_base_contract("supply")

        self.assertIn("`scope=portfolio` всегда означает анализ всей товарной матрицы", contract)
        self.assertIn("дополнительной углублённой проверки", contract)
        self.assertIn("`scope=sku_list`", contract)

    def test_cached_agent_uses_round_specific_output_schema(self):
        spec = LLMRuntimeSpec(
            profile="offline-openai",
            provider="openai",
            model_name="gpt-offline",
            api_style="responses",
            base_url=None,
            default_headers=(),
            experimental=False,
            missing_env_vars=(),
        )
        expected = {
            RoundType.CROSS_CHECK: CrossCheckResult,
            RoundType.FOLLOWUP: FollowupResult,
            RoundType.GAP_RESOLUTION: GapResolutionResult,
            RoundType.CLAIM_VERIFICATION: ClaimVerificationResult,
        }
        _build_domain_agent_for_round.cache_clear()
        for round_type, output_schema in expected.items():
            with self.subTest(round_type=round_type.value):
                first = _build_domain_agent_for_round(
                    spec,
                    QueryIntent.SUPPLY.value,
                    WorkflowVersion.V2_PROMPT.value,
                    round_type.value,
                )
                second = _build_domain_agent_for_round(
                    spec,
                    QueryIntent.SUPPLY.value,
                    WorkflowVersion.V2_PROMPT.value,
                    round_type.value,
                )
                self.assertIs(first, second)
                self.assertIs(first.output_type, output_schema)

    def test_round_task_round_trips_and_persisted_goal_fields_are_not_lost(self):
        question = SimpleNamespace(
            target_agent="supply",
            reason="Закрыть gap оборачиваемости.",
            question="Проверь точную оборачиваемость SKU 46227.",
            priority=90,
            verification_goal={
                "round": "round_3",
                "source": "gap_resolution",
                "gap_id": 1521,
                "sku": "46227",
                "period": "2026-06-01/2026-06-25",
                "required_checks": ["inventory_turnover"],
                "claim_policy": "inventory_turnover_causal",
            },
        )

        restored = _question_to_plan_item(question)
        self.assertEqual(restored["round"], "round_3")
        self.assertEqual(restored["gap_id"], 1521)
        self.assertEqual(restored["claim_policy"], "inventory_turnover_causal")

        task = self._task(
            RoundType.GAP_RESOLUTION,
            question=restored["question"],
            verification_goal=question.verification_goal,
            gap_id=restored["gap_id"],
            claim_policy=restored["claim_policy"],
        )
        round_tripped = RoundTask.model_validate_json(task.model_dump_json())
        self.assertEqual(round_tripped.round_type, RoundType.GAP_RESOLUTION)
        self.assertEqual(round_tripped.gap_id, 1521)
        self.assertEqual(round_tripped.claim_policy, "inventory_turnover_causal")

        prompt = build_round_prompt(task, {"summary": "Evidence"}, [], "v2_prompt")
        self.assertIn('"round_type":"gap_resolution"', prompt.text)
        self.assertIn('"gap_id":1521', prompt.text)
        self.assertIn('"claim_policy":"inventory_turnover_causal"', prompt.text)

    def test_secondary_evidence_selection_is_deterministic_and_narrow(self):
        task = self._task(RoundType.CROSS_CHECK)
        primary = {
            "summary": "Проверяется оборачиваемость склада.",
            "findings": [
                {
                    "id": "best",
                    "factor": "оборачиваемость остатка",
                    "observation": "SKU 46227 за июнь",
                },
                {"id": "second", "factor": "остаток SKU 46227", "observation": "июнь"},
                {"id": "third", "factor": "оборачиваемость", "observation": "склад"},
                {"id": "noise-a", "factor": "реклама", "observation": "CTR"},
                {"id": "noise-b", "factor": "цена", "observation": "скидка"},
            ],
            "tables": [
                {
                    "title": "Остаток и оборачиваемость SKU 46227",
                    "columns": ["SKU", "Остаток", "Дни"],
                    "rows": [["46227", 1131, 59.4]] * 12,
                },
                {
                    "title": "Рекламные кампании",
                    "columns": ["Кампания", "CTR"],
                    "rows": [["A", 1.2]],
                },
            ],
        }
        prior = [
            {
                "round": "followup",
                "target_agent": "supply",
                "question": "Оборачиваемость SKU 46227 за июнь",
                "output": {"summary": "релевантно", "facts": [{"metric": "days_on_hand"}]},
            },
            {
                "round": "followup",
                "target_agent": "finance",
                "question": "Остаток SKU 46227 в деньгах",
                "output": {"summary": "релевантно", "facts": [{"metric": "stock_value"}]},
            },
            {
                "round": "followup",
                "target_agent": "marketing",
                "question": "CTR кампании",
                "output": {"summary": "не относится к задаче", "facts": []},
            },
        ]

        first_payload, first_stats = compact_secondary_context(task, primary, prior)
        second_payload, second_stats = compact_secondary_context(task, primary, prior)

        self.assertEqual(first_payload, second_payload)
        self.assertEqual(first_stats, second_stats)
        self.assertEqual(
            [item["id"] for item in first_payload["relevant_findings"]],
            ["best", "second", "third"],
        )
        self.assertEqual(len(first_payload["relevant_tables"]), 1)
        self.assertEqual(len(first_payload["relevant_tables"][0]["rows"]), 8)
        self.assertIn("Остаток", first_payload["relevant_tables"][0]["title"])
        self.assertEqual(len(first_payload["prior_evidence"]), 2)
        self.assertEqual(first_stats["dropped_findings"], 2)
        self.assertEqual(first_stats["dropped_tables"], 1)
        self.assertEqual(first_stats["dropped_prior_evidence"], 1)

    def test_secondary_and_synthesis_have_hard_size_limits(self):
        task = self._task(
            RoundType.GAP_RESOLUTION,
            question="Закрой gap 1521 по точной оборачиваемости SKU 46227. " + "метрика " * 2_000,
            gap_id=1521,
            claim_policy="inventory_turnover_causal",
        )
        primary = {
            "summary": "summary " * 8_000,
            "findings": [
                {"id": index, "observation": "оборачиваемость остаток SKU 46227 " * 300}
                for index in range(40)
            ],
            "tables": [
                {
                    "title": f"Остатки {index}",
                    "columns": ["SKU", "Описание"],
                    "rows": [["46227", "row " * 2_000] for _ in range(20)],
                }
                for index in range(5)
            ],
        }
        prior = [
            {
                "round": "followup",
                "target_agent": "supply",
                "question": "оборачиваемость SKU 46227 " * 100,
                "output": {"summary": "prior " * 2_000},
            }
            for _ in range(30)
        ]

        secondary = build_round_prompt(task, primary, prior, WorkflowVersion.V2_PROMPT)
        self.assertLessEqual(len(secondary.text), SECONDARY_PROMPT_MAX_CHARS)
        self.assertLessEqual(secondary.report.total_chars, SECONDARY_PROMPT_MAX_CHARS)
        self.assertEqual(secondary.report.max_chars, SECONDARY_PROMPT_MAX_CHARS)
        self.assertGreater(secondary.report.dropped_evidence, 0)
        self.assertIn('"gap_id":1521', secondary.text)
        self.assertIn('"claim_policy":"inventory_turnover_causal"', secondary.text)

        synthesis_source = {
            "workflow_version": "v2_prompt",
            "query_mode": "diagnostic",
            "original_user_query": "Почему упала оборачиваемость? " + "контекст " * 2_000,
            "primary_agent": "supply",
            "query_plan": {"scope": "portfolio", "targets": [{"value": "46227"}]},
            "decision_mode": {"mode": "scenario_decision"},
            "claim_verification": {"status": "incomplete", "missing_domains": ["price"]},
            "domain_coverage": {"supply": "confirmed", "price": "not_checked"},
            "evidence_quality": {"missing_domain_reasons": ["price_not_called"]},
            "display_policy": {"user_facing_identifier": "sku_only"},
            "format_policy": {"action_task_count": 5},
            "final_answer_rules": ["Use only confirmed evidence."],
            "confirmed_facts": [
                {
                    "metric": f"metric_{index}",
                    "statement": "confirmed evidence " * 500,
                    "status": "confirmed",
                }
                for index in range(120)
            ],
            "refuted_facts": [
                {"metric": f"refuted_{index}", "statement": "refuted " * 500, "status": "refuted"}
                for index in range(40)
            ],
            "blocking_gaps": [
                {"metric": f"gap_{index}", "reason": "missing " * 500, "blocking": True}
                for index in range(40)
            ],
            "sku_metric_matrix": [
                {"sku": str(index), "details": "metric matrix " * 500} for index in range(40)
            ],
            "sku_evidence_matrix": [
                {"sku": str(index), "details": "evidence matrix " * 500} for index in range(40)
            ],
            "action_safety_checks": [
                {"sku": str(index), "details": "safety " * 500} for index in range(40)
            ],
        }
        compact, compact_report = compact_synthesis_package(synthesis_source)
        compact_json = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
        self.assertLessEqual(len(compact_json), SYNTHESIS_PACKAGE_MAX_CHARS)
        self.assertLessEqual(compact_report.total_chars, SYNTHESIS_PACKAGE_MAX_CHARS)
        self.assertTrue(compact_report.truncated)
        self.assertEqual(
            set(compact),
            {
                "query_mode",
                "original_user_query",
                "workflow_version",
                "primary_agent",
                "query_plan",
                "decision_mode",
                "primary_summary",
                "confirmed_facts",
                "refuted_facts",
                "blocking_gaps",
                "accepted_limitations",
                "non_blocking_gaps",
                "resolved_gaps",
                "sku_metric_matrix_required",
                "sku_metric_matrix",
                "sku_evidence_matrix",
                "evidence_tables",
                "action_safety_checks",
                "claim_verification",
                "domain_coverage",
                "evidence_quality",
                "display_policy",
                "format_policy",
                "final_answer_rules",
            },
        )
        self.assertEqual(compact["query_plan"]["scope"], "portfolio")
        self.assertEqual(compact["decision_mode"]["mode"], "scenario_decision")
        self.assertEqual(compact["claim_verification"]["status"], "incomplete")
        self.assertEqual(compact["format_policy"]["action_task_count"], 5)

        synthesis_task = self._task(RoundType.SYNTHESIS)
        synthesis = build_round_prompt(
            synthesis_task,
            synthesis_source,
            [],
            WorkflowVersion.V2_PROMPT,
        )
        self.assertLessEqual(len(synthesis.text), SYNTHESIS_PACKAGE_MAX_CHARS)
        self.assertLessEqual(synthesis.report.total_chars, SYNTHESIS_PACKAGE_MAX_CHARS)

    def test_hard_limits_hold_for_single_pathologically_nested_evidence_item(self):
        nested = {
            f"group_{outer}": {
                f"field_{inner}": "nested-evidence " * 500
                for inner in range(24)
            }
            for outer in range(24)
        }
        task = self._task(RoundType.CROSS_CHECK)
        secondary = build_round_prompt(
            task,
            {"summary": "Primary", "findings": [nested]},
            [],
            WorkflowVersion.V2_PROMPT,
        )
        self.assertLessEqual(secondary.report.total_chars, SECONDARY_PROMPT_MAX_CHARS)
        secondary_payload = json.loads(secondary.text)
        self.assertTrue(
            secondary_payload.get("relevant_findings")
            or secondary_payload.get("current_evidence")
        )

        compact, report = compact_synthesis_package(
            {
                "workflow_version": "v2_prompt",
                "query_mode": "diagnostic",
                "original_user_query": "Почему изменился показатель?",
                "confirmed_facts": [nested],
            }
        )
        self.assertLessEqual(
            len(json.dumps(compact, ensure_ascii=False, separators=(",", ":"))),
            SYNTHESIS_PACKAGE_MAX_CHARS,
        )
        self.assertLessEqual(report.total_chars, SYNTHESIS_PACKAGE_MAX_CHARS)

    def test_synthesis_compactor_keeps_table_only_primary_evidence(self):
        compact, _ = compact_synthesis_package(
            {
                "workflow_version": "v2_prompt",
                "query_mode": "lookup",
                "original_user_query": "Сколько остатка у SKU 46227?",
                "primary_evidence": {
                    "summary": "Остаток найден.",
                    "tables": [
                        {
                            "title": "Остатки",
                            "columns": ["SKU", "Остаток, шт."],
                            "rows": [["46227", 1131]],
                        }
                    ],
                },
            }
        )
        self.assertEqual(len(compact["evidence_tables"]), 1)
        serialized = json.dumps(compact["evidence_tables"], ensure_ascii=False)
        self.assertIn("46227", serialized)
        self.assertIn("1131", serialized)

    def test_synthesis_compactor_preserves_machine_contract_without_size_pressure(self):
        source = {
            "workflow_version": "v2_optimized",
            "query_mode": "decision",
            "original_user_query": "Сформируй портфельный план.",
            "primary_agent": "supply",
            "query_plan": {
                "scope": "portfolio",
                "decision_kind": "single_action",
                "targets": [{"type": "sku", "value": "46227"}],
            },
            "decision_mode": {"mode": "scenario_decision", "exact_model_required": False},
            "confirmed_facts": [{"metric": "days_on_hand", "value": 45, "unit": "days"}],
            "blocking_gaps": [],
            "claim_verification": {"status": "complete", "missing_domains": []},
            "domain_coverage": {"supply": "confirmed"},
            "evidence_quality": {"missing_domain_reasons": []},
            "sku_metric_matrix_required": True,
            "sku_metric_matrix": [{"sku": "46227", "days_on_hand": 45}],
            "sku_evidence_matrix": [{"sku": "46227", "supply": {"status": "confirmed"}}],
            "action_safety_checks": [{"action": "reorder", "status": "needs_review"}],
            "display_policy": {"user_facing_identifier": "sku_only"},
            "format_policy": {"action_task_count": 5, "sku_metric_matrix": True},
            "final_answer_rules": ["Do not expose technical JSON."],
        }

        compact, report = compact_synthesis_package(source)

        for key in (
            "query_plan",
            "decision_mode",
            "claim_verification",
            "domain_coverage",
            "evidence_quality",
            "sku_metric_matrix",
            "sku_evidence_matrix",
            "action_safety_checks",
            "display_policy",
            "format_policy",
            "final_answer_rules",
        ):
            self.assertEqual(compact[key], source[key])
        self.assertFalse(report.truncated)
        self.assertLessEqual(report.total_chars, SYNTHESIS_PACKAGE_MAX_CHARS)

    def test_typed_secondary_adapters_forbid_new_cross_requests(self):
        fact = EvidenceFact(
            metric="inventory_turnover",
            value=59.4,
            unit="days",
            period="2026-06-01/2026-06-25",
            evidence_ref="offline:turnover",
        )
        request = CrossAgentRequest(
            target_agent=QueryIntent.FINANCE,
            reason="Нужна себестоимость.",
            question="Проверь себестоимость SKU 46227.",
        )
        cross_check = CrossCheckResult(
            agent_name="supply",
            summary="Проверено.",
            verdict="confirmed",
            facts=[fact],
            followup_requests=[request],
        )
        self.assertEqual(
            len(adapt_round_result_to_legacy(cross_check)["cross_agent_requests"]),
            1,
        )

        typed_results = [
            FollowupResult(
                agent_name="supply",
                summary="Безопасность проверена.",
                status="safe",
                facts=[fact],
            ),
            GapResolutionResult(
                agent_name="supply",
                gap_id=1521,
                summary="Gap закрыт.",
                outcome="resolved",
                fact=fact,
            ),
            ClaimVerificationResult(
                agent_name="supply",
                claim="Остаток влияет на оборачиваемость.",
                summary="Проверено.",
                verdict="confirmed",
                facts=[fact],
            ),
        ]
        for result in typed_results:
            with self.subTest(result=type(result).__name__):
                self.assertEqual(
                    adapt_round_result_to_legacy(result)["cross_agent_requests"],
                    [],
                )
                payload = result.model_dump(mode="json")
                payload["cross_agent_requests"] = []
                with self.assertRaises(ValidationError):
                    type(result).model_validate(payload)

        with self.assertRaises(ValidationError):
            GapResolutionResult(
                agent_name="supply",
                gap_id=1521,
                summary="Ошибочно закрыт без evidence.",
                outcome="resolved",
            )

        limited = GapResolutionResult(
            agent_name="supply",
            gap_id=1522,
            summary="Истории нет, но текущий snapshot подтверждён.",
            outcome="no_data",
            fact=fact,
            reason="Доступен только один snapshot.",
            impact="Динамика не проверена.",
        )
        limited_envelope = adapt_round_result_to_legacy(limited)
        self.assertEqual(len(limited_envelope["findings"]), 1)
        self.assertEqual(len(limited_envelope["data_gaps"]), 1)
        self.assertEqual(limited_envelope["gap_resolution_status"], "no_data")

        reduced = _reduce_evidence(
            primary_agent=QueryIntent.SUPPLY,
            primary_output={
                "summary": "Primary",
                "tables": [],
                "findings": [],
                "data_gaps": [],
                "cross_agent_requests": [],
            },
            round_1_evidence=[
                {
                    "round": "round_1",
                    "target_agent": "supply",
                    "question": "Проверь остаток",
                    "output": adapt_round_result_to_legacy(cross_check),
                }
            ],
            followup_evidence=[],
        )
        self.assertEqual(len(reduced["confirmed_facts"]), 1)
        self.assertEqual(reduced["confirmed_facts"][0]["metric"], "inventory_turnover")
        self.assertEqual(reduced["confirmed_facts"][0]["value"], 59.4)
        self.assertEqual(reduced["confirmed_facts"][0]["unit"], "days")
        self.assertEqual(reduced["contradictions"], [])

    def test_evidence_fact_allows_categorical_value_without_unit(self):
        fact = EvidenceFact(
            metric="driver_role",
            value="growth_driver",
            period="2026-07-06/2026-07-12",
            evidence_ref="sales:sku-change",
        )

        self.assertIsNone(fact.unit)
        self.assertEqual(fact.value, "growth_driver")
        sku_fact = EvidenceFact(
            metric="sku",
            value="46227",
            period="2026-07-06/2026-07-12",
            evidence_ref="sales:sku-change",
        )
        self.assertIsNone(sku_fact.unit)
        with self.assertRaises(ValidationError):
            EvidenceFact(
                metric="sales_units",
                value=12,
                period="2026-07-06/2026-07-12",
                evidence_ref="sales:overview",
            )

        for metric, value in (
            ("nm_id", 70194686),
            ("SKU", 46227),
            ("vendor_code", 46227),
            ("campaign_id", 1234),
        ):
            with self.subTest(identifier_metric=metric):
                identifier = EvidenceFact(
                    metric=metric,
                    value=value,
                    evidence_ref=f"source:{metric}",
                )
                self.assertIsNone(identifier.unit)
                self.assertIsNone(identifier.period)

        with self.assertRaises(ValidationError):
            EvidenceFact(
                metric="sales_units",
                value=12,
                unit="шт.",
                evidence_ref="sales:overview",
            )

        unsafe = FollowupResult(
            agent_name="supply",
            summary="Запаса недостаточно.",
            status="unsafe",
            facts=[fact],
        )
        finalized = _finalize_action_safety_checks(
            [{"check_id": "scale:supply:1", "status": "missing"}],
            [
                {
                    "target_agent": "supply",
                    "safety_check_ids": ["scale:supply:1"],
                    "output": adapt_round_result_to_legacy(unsafe),
                }
            ],
        )
        self.assertEqual(finalized[0]["status"], "unsafe")

    def test_query_mode_precedence_is_decision_then_diagnostic_then_report_lookup(self):
        cases = {
            "Покажи отчет: почему упали продажи и что делать?": QueryMode.DECISION,
            "Покажи отчет и объясни, почему упали продажи": QueryMode.DIAGNOSTIC,
            "Покажи отчет: сколько продаж было": QueryMode.REPORT,
            "Сколько продаж было вчера?": QueryMode.LOOKUP,
            "Поднять ли цену?": QueryMode.DECISION,
            "Заказывать ли товар?": QueryMode.DECISION,
            "Отключать ли рекламу?": QueryMode.DECISION,
            "Какую безопасную цену поставить?": QueryMode.DECISION,
            "Посчитай оборачиваемость, деньги в остатках и прибыль": QueryMode.DIAGNOSTIC,
            "Обычный запрос без специальных маркеров": QueryMode.REPORT,
        }
        for query, expected in cases.items():
            with self.subTest(query=query):
                self.assertEqual(determine_query_mode(query), expected)

    def test_manager_action_portfolio_is_decision_without_promoting_reports(self):
        decision_queries = (
            "Сформируй единый план из 3–7 приоритетных рекламных действий",
            "Составь портфель из пяти рекламных действий на сегодня",
            "Подготовь единый план из приоритетных действий по рекламе и продажам",
        )
        report_queries = (
            "Сформируй единый отчет по 7 рекламным кампаниям",
            "Покажи отчет с 3–7 приоритетными рекламными показателями",
            "Подготовь сводку по выполненным рекламным действиям",
        )

        for query in decision_queries:
            with self.subTest(query=query):
                self.assertEqual(determine_query_mode(query), QueryMode.DECISION)
        for query in report_queries:
            with self.subTest(query=query):
                self.assertEqual(determine_query_mode(query), QueryMode.REPORT)

    def test_openai_typed_and_openrouter_text_outputs_normalize_identically(self):
        payload = {
            "agent_name": "supply",
            "summary": "Остаток проверен.",
            "verdict": "confirmed",
            "facts": [
                {
                    "metric": "stock_qty",
                    "value": 1131,
                    "unit": "units",
                    "period": "2026-06-25",
                    "sku": "46227",
                    "evidence_ref": "get_supply_overview:46227",
                    "status": "confirmed",
                    "confidence": "high",
                }
            ],
            "tables": [],
            "data_gaps": [],
            "followup_requests": [],
        }
        typed = CrossCheckResult.model_validate(payload)
        context = EcomRunContext(
            user_query="Проверь остаток SKU 46227",
            intent=QueryIntent.SUPPLY,
            today=date(2026, 7, 13),
            db_target="offline",
            chat_id="offline",
            workflow_version=WorkflowVersion.V2_PROMPT,
            round_type=RoundType.CROSS_CHECK,
            query_mode=QueryMode.DIAGNOSTIC,
            memory_mode=MemoryMode.NONE,
        )
        agent = SimpleNamespace(tools=[], instructions="")

        normalized = []
        for raw_output in (typed, f"```json\n{json.dumps(payload, ensure_ascii=False)}\n```"):
            with patch(
                "agi_ecom_app.runner.Runner.run_sync",
                return_value=SimpleNamespace(final_output=raw_output),
            ):
                _, output = _run_agent_once(
                    lambda: agent,
                    "{}",
                    context,
                    hooks=None,
                    max_turns=1,
                    expected_output_type=CrossCheckResult,
                )
            normalized.append(output.model_dump(mode="json"))

        self.assertEqual(normalized[0], normalized[1])

    def test_actual_v2_instructions_apply_round_memory_policy(self):
        messages = [
            ChatMemoryMessage(
                seq=index,
                role=ChatRole.USER,
                content="old-secret" if index == 1 else f"recent-{index}",
            )
            for index in range(1, 14)
        ]
        context = EcomRunContext(
            user_query="Проверь остаток SKU 46227",
            intent=QueryIntent.SUPPLY,
            today=date(2026, 7, 13),
            db_target="offline",
            chat_id="offline",
            workflow_version=WorkflowVersion.V2_PROMPT,
            round_type=RoundType.CROSS_CHECK,
            query_mode=QueryMode.DIAGNOSTIC,
            chat_summary="summary-secret",
            recent_messages=messages,
        )
        instructions = _dynamic_instructions(SUPPLY_PROMPT_PATH)
        secondary = instructions(SimpleNamespace(context=context), None)
        self.assertNotIn("summary-secret", secondary)
        self.assertNotIn("recent-13", secondary)

        context.round_type = RoundType.PRIMARY
        primary = instructions(SimpleNamespace(context=context), None)
        self.assertIn("summary-secret", primary)
        self.assertIn("recent-13", primary)
        self.assertNotIn("old-secret", primary)

        context.round_type = RoundType.SYNTHESIS
        synthesis = _diagnostic_orchestrator_instructions()(
            SimpleNamespace(context=context),
            None,
        )
        self.assertIn("summary-secret", synthesis)
        self.assertNotIn("recent-13", synthesis)

    def test_queue_and_synchronous_paths_share_builder_and_pinned_version(self):
        import agi_ecom_app.agent_queue as queue_module

        self.assertIs(queue_module._build_secondary_task_prompt, _build_secondary_task_prompt)
        trace = SimpleNamespace(
            summary={
                "workflow_version": "v2_prompt",
                "query_mode": "diagnostic",
            }
        )
        self.assertEqual(_trace_workflow_version(trace), "v2_prompt")
        # A settings change cannot alter an already persisted run.
        trace.summary["unrelated_live_default"] = "v1"
        self.assertEqual(_trace_workflow_version(trace), "v2_prompt")

    def test_optimized_budget_and_no_data_gap_deduplication(self):
        rounds = [
            {"questions": [1]},
            {"questions": [2, 3, 4]},
        ]
        remaining = _remaining_optimized_question_slots(
            rounds,
            reserve=OPTIMIZED_SYNTHESIS_QUESTION_RESERVE,
        )
        self.assertEqual(
            sum(len(item["questions"]) for item in rounds)
            + remaining
            + OPTIMIZED_SYNTHESIS_QUESTION_RESERVE,
            MAX_OPTIMIZED_PERSISTED_QUESTION_ROWS,
        )
        self.assertEqual(
            MAX_OPTIMIZED_PERSISTED_QUESTION_ROWS
            + OPTIMIZED_ROUTER_EXECUTIONS
            + OPTIMIZED_REPAIR_EXECUTION_RESERVE,
            MAX_OPTIMIZED_MODEL_EXECUTIONS,
        )
        self.assertEqual(MAX_OPTIMIZED_FAST_PATH_QUESTION_ROWS, 2)

        gaps = [
            SimpleNamespace(
                target_agent="price",
                source_agent="price",
                sku="46227",
                metric="Средняя цена",
                reason="Нет данных за 30 дней",
                evidence_status="no_data",
            ),
            SimpleNamespace(
                target_agent="price",
                source_agent="price",
                sku="46227",
                metric="Средняя цена",
                reason="Проверить период 30 дней",
                evidence_status="unconfirmed",
            ),
            SimpleNamespace(
                target_agent="price",
                source_agent="price",
                sku="46227",
                metric="Средняя цена",
                reason="Проверить период 30 дней",
                evidence_status="unconfirmed",
            ),
        ]
        deduplicated = _deduplicate_actionable_gaps(gaps)
        self.assertEqual(deduplicated, [])

        # A no_data result blocks the same key regardless of row order.
        reverse_order = _deduplicate_actionable_gaps(list(reversed(gaps)))
        self.assertEqual(reverse_order, [])

    def test_optimized_lookup_and_report_never_attempt_synthesis_repair(self):
        for query_mode in (QueryMode.LOOKUP, QueryMode.REPORT):
            with self.subTest(query_mode=query_mode):
                self.assertFalse(
                    _synthesis_repair_allowed(
                        WorkflowVersion.V2_OPTIMIZED,
                        query_mode,
                    )
                )
        self.assertTrue(
            _synthesis_repair_allowed(
                WorkflowVersion.V2_OPTIMIZED,
                QueryMode.DIAGNOSTIC,
            )
        )
        self.assertTrue(
            _synthesis_repair_allowed(
                WorkflowVersion.V2_PROMPT,
                QueryMode.LOOKUP,
            )
        )

    def test_queue_reconstruction_keeps_safety_and_claim_metadata(self):
        response = AgentRunResponseRecord(
            id=7,
            question_id=6,
            agent_name="supply",
            status="completed",
            summary="Проверено",
            raw_output={"summary": "Проверено", "findings": []},
            has_business_evidence=True,
            runtime_failure=None,
        )
        rows = [
            {
                "round_number": 2,
                "round_type": "followup",
                "target_agent": "supply",
                "question": "Проверь запас",
                "reason": "Safety",
                "verification_goal": {
                    "source": "action_safety",
                    "safety_check_ids": ["scale:supply:1"],
                    "claim_policy": "inventory_turnover_causal",
                    "gap_id": 42,
                },
                "response": response,
            }
        ]
        with patch("agi_ecom_app.runner.list_agent_run_responses", return_value=rows):
            # A primary row is required for reconstruction.
            primary = {
                **rows[0],
                "round_number": 0,
                "round_type": "primary",
                "target_agent": "supply",
                "verification_goal": {},
            }
            with patch(
                "agi_ecom_app.runner.list_agent_run_responses",
                return_value=[primary, rows[0]],
            ):
                restored = _load_persisted_evidence_safely(99)
        evidence = restored["followup_evidence"][0]
        self.assertEqual(evidence["safety_check_ids"], ["scale:supply:1"])
        self.assertEqual(evidence["claim_policy"], "inventory_turnover_causal")
        self.assertEqual(evidence["gap_id"], 42)

    def test_offline_goldens_cover_runs_190_191_196_197(self):
        goldens = _load_goldens()
        self.assertEqual([fixture["run_id"] for fixture in goldens], [190, 191, 196, 197])

        all_hazards: set[str] = set()
        for fixture in goldens:
            with self.subTest(run_id=fixture["run_id"]):
                self.assertEqual(
                    determine_query_mode(fixture["user_query"]).value,
                    fixture["expected_query_mode"],
                )
                self.assertEqual(
                    validate_prompt_v2_answer(
                        fixture["hazardous_draft"],
                        fixture["evidence_package"],
                    ),
                    fixture["expected_violation_codes"],
                )
                self.assertEqual(
                    validate_prompt_v2_answer(
                        fixture["corrected_draft"],
                        fixture["evidence_package"],
                    ),
                    [],
                )
                for fragment in fixture["required_corrected_fragments"]:
                    self.assertIn(fragment, fixture["corrected_draft"])
                for fragment in fixture["forbidden_corrected_fragments"]:
                    self.assertNotIn(fragment, fixture["corrected_draft"])
                all_hazards.update(fixture["acceptance_hazards"])

        self.assertTrue(
            {
                "contribution_margin_is_not_net_profit",
                "sales_units_pct_is_not_revenue_pct",
                "missing_sku_row_is_not_zero",
                "exact_turnover_must_use_available_tool_fact",
                "current_stock_value_is_stock_times_unit_cost",
                "sold_units_times_cost_is_not_current_stock_value",
                "same_sku_cannot_have_stop_and_maintain_reorder_actions",
            }.issubset(all_hazards)
        )

        finance_contract = domain_base_contract("finance").lower()
        sales_contract = domain_base_contract("sales").lower()
        synthesis_base_contract = domain_base_contract("synthesis").lower()
        synthesis_contract = round_contract(RoundType.SYNTHESIS).lower()
        self.assertIn("contribution margin чистой прибылью", finance_contract)
        self.assertIn("остаток supply", finance_contract)
        self.assertIn("отсутствие sku", sales_contract)
        self.assertIn("no_data", synthesis_base_contract)
        self.assertIn("противоречивые действия", synthesis_contract)

    def test_goldens_build_with_both_v2_workflow_versions(self):
        for fixture in _load_goldens():
            for version in (WorkflowVersion.V2_PROMPT, WorkflowVersion.V2_OPTIMIZED):
                with self.subTest(run_id=fixture["run_id"], version=version.value):
                    facts = list(fixture["evidence_package"].get("confirmed_facts") or [])
                    task = RoundTask(
                        round_type=RoundType.CROSS_CHECK,
                        target_agent=fixture["primary_domain"],
                        question=f"Проверь evidence для replay run {fixture['run_id']}.",
                        original_user_query=fixture["user_query"],
                        verification_goal={"run_id": fixture["run_id"]},
                        required_checks=[str(item.get("metric")) for item in facts],
                        source="golden_replay",
                        gap_id=None,
                        claim_policy=None,
                        period=None,
                        sku=None,
                        vendor_codes=[],
                        nm_ids=[],
                        allowed_tools=["offline_fixture"],
                        required_tools=[],
                    )
                    prompt = build_round_prompt(
                        task,
                        {
                            "summary": f"Golden run {fixture['run_id']}",
                            "findings": facts,
                        },
                        [],
                        version,
                    )
                    self.assertLessEqual(prompt.report.total_chars, SECONDARY_PROMPT_MAX_CHARS)
                    self.assertEqual(prompt.report.workflow_version, version)

                    _, report = compact_synthesis_package(
                        {
                            **fixture["evidence_package"],
                            "workflow_version": version.value,
                            "query_mode": fixture["expected_query_mode"],
                            "original_user_query": fixture["user_query"],
                        }
                    )
                    self.assertLessEqual(report.total_chars, SYNTHESIS_PACKAGE_MAX_CHARS)
                    self.assertEqual(report.workflow_version, version)


if __name__ == "__main__":
    unittest.main()
