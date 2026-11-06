from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agents import OutputGuardrailTripwireTriggered

from agi_ecom_app.llm_runtime import (
    LLMRuntimeConfigurationError,
    build_sync_client,
    get_active_llm_runtime_spec,
    validate_model_profile_configured,
)
from agi_ecom_app.llm_output_parsing import coerce_model_from_json_text
from agi_ecom_app.llm_routing import (
    DomainRouteDecision,
    QueryPlan,
    _coerce_domain_route_decision,
    route_query_intent_with_llm,
)
from agi_ecom_app.main_agent import _build_sales_agent
from agi_ecom_app.model_profiles import (
    OPENAI_DEFAULT_PROFILE,
    OPENROUTER_GEMMA_FREE_PROFILE,
    build_model_profiles,
    describe_model_profile,
)
from agi_ecom_app.schemas import QueryIntent
from agi_ecom_app.services.runtime_settings import get_active_model_profile_name


def _settings(**overrides):
    base = {
        "openai_api_key": "sk-openai-test",
        "openai_model": "gpt-test",
        "openai_memory_model": "gpt-memory",
        "openrouter_api_key": None,
        "openrouter_base_url": "https://openrouter.ai/api/v1",
        "openrouter_model": "google/gemma-4-31b-it:free",
        "openrouter_http_referer": "https://example.test",
        "openrouter_title": "AI-ECOM Test",
        "llm_model_profile": OPENAI_DEFAULT_PROFILE,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class ModelProfileTests(unittest.TestCase):
    def test_registry_describes_profiles_without_secret_values(self):
        settings = _settings(openrouter_api_key="sk-or-secret")
        profiles = build_model_profiles(settings)

        self.assertIn(OPENAI_DEFAULT_PROFILE, profiles)
        self.assertIn(OPENROUTER_GEMMA_FREE_PROFILE, profiles)
        description = describe_model_profile(profiles[OPENROUTER_GEMMA_FREE_PROFILE], settings)

        self.assertEqual(description["provider"], "openrouter")
        self.assertEqual(description["model"], "google/gemma-4-31b-it:free")
        self.assertEqual(description["required_env_vars"], ["OPENROUTER_API_KEY"])
        self.assertNotIn("sk-or-secret", str(description))

    def test_runtime_settings_falls_back_to_env_profile_when_db_unavailable(self):
        settings = _settings(llm_model_profile=OPENROUTER_GEMMA_FREE_PROFILE)

        with patch(
            "agi_ecom_app.services.runtime_settings._load_runtime_profile_from_db",
            return_value=None,
        ):
            self.assertEqual(get_active_model_profile_name(settings), OPENROUTER_GEMMA_FREE_PROFILE)

    def test_openrouter_profile_reports_missing_key(self):
        settings = _settings(llm_model_profile=OPENROUTER_GEMMA_FREE_PROFILE, openrouter_api_key=None)

        with patch(
            "agi_ecom_app.services.runtime_settings._load_runtime_profile_from_db",
            return_value=None,
        ):
            spec = get_active_llm_runtime_spec(settings=settings)

        self.assertEqual(spec.provider, "openrouter")
        self.assertEqual(spec.missing_env_vars, ("OPENROUTER_API_KEY",))
        with self.assertRaises(LLMRuntimeConfigurationError):
            validate_model_profile_configured(OPENROUTER_GEMMA_FREE_PROFILE, settings=settings)

    def test_openrouter_sync_client_uses_openrouter_base_url_and_headers(self):
        settings = _settings(
            llm_model_profile=OPENROUTER_GEMMA_FREE_PROFILE,
            openrouter_api_key="sk-or-test",
        )

        with patch(
            "agi_ecom_app.services.runtime_settings._load_runtime_profile_from_db",
            return_value=None,
        ):
            spec = get_active_llm_runtime_spec(settings=settings)
        client = build_sync_client(spec, settings=settings)

        self.assertEqual(str(client.base_url), "https://openrouter.ai/api/v1/")
        self.assertEqual(client.default_headers["HTTP-Referer"], "https://example.test")
        self.assertEqual(client.default_headers["X-Title"], "AI-ECOM Test")

    def test_lenient_json_parser_accepts_gemma_wrapped_fenced_output(self):
        output = """```json
        {
          "domain_route_decision": {
            "route_to_domain": true,
            "intent": "sales",
            "confidence": "high",
            "reason": "Запрос про продажи.",
            "supporting_agents": []
          }
        }
        ```"""

        decision = coerce_model_from_json_text(output, DomainRouteDecision)

        self.assertTrue(decision.route_to_domain)
        self.assertEqual(decision.intent, QueryIntent.SALES)
        self.assertEqual(decision.confidence, "high")

    def test_domain_router_adapter_normalizes_gemma_freeform_fields(self):
        fallback = DomainRouteDecision(
            route_to_domain=True,
            intent=QueryIntent.SALES,
            confidence="low",
            reason="fallback",
            supporting_agents=[],
        )
        output = {
            "route_to_domain": "sales",
            "intent": "check_recent_sales_performance",
            "confidence": 1.0,
            "reason": "Запрос про продажи.",
            "supporting_agents": ["finance", "unknown_domain"],
            "query_plan": {
                "query_mode": "decision",
                "scope": "portfolio",
                "decision_kind": "action_portfolio",
                "objective": "reduce_inventory_days",
                "targets": [],
                "requested_analyses": [
                    "inventory_matrix",
                    "price_scenarios",
                    "forecast",
                ],
                "scenario_requests": ["compare price variants"],
                "required_domains": ["supply", "finance", "price"],
                "constraints": {
                    "target_cover_days": 30,
                    "horizon_days": 60,
                    "avoid_stockout": True,
                    "preserve_contribution_profit": True,
                },
            },
        }

        decision = _coerce_domain_route_decision(output, fallback)

        self.assertTrue(decision.route_to_domain)
        self.assertEqual(decision.intent, QueryIntent.SALES)
        self.assertEqual(decision.confidence, "high")
        self.assertEqual(decision.supporting_agents, [QueryIntent.FINANCE])
        self.assertEqual(decision.query_plan.scope, "portfolio")
        self.assertEqual(decision.query_plan.constraints.target_cover_days, 30)
        self.assertEqual(decision.query_plan.targets, [])

    def test_domain_router_repairs_portfolio_focus_targets_without_keyword_fallback(self):
        fallback = DomainRouteDecision(
            route_to_domain=True,
            intent=QueryIntent.FINANCE,
            confidence="low",
            reason="fallback",
        )
        decision = _coerce_domain_route_decision(
            {
                "route_to_domain": False,
                "intent": "supply",
                "confidence": "high",
                "reason": "Цель — покрытие всей матрицы.",
                "supporting_agents": ["supply", "price", "marketing"],
                "query_plan": {
                    "query_mode": "report",
                    "scope": "portfolio",
                    "decision_kind": "action_portfolio",
                    "objective": "reduce_cover_to_30_days",
                    "targets": [
                        {"target_type": "vendor_code", "value": "46227"},
                        {"target_type": "vendor_code", "value": "46303"},
                    ],
                    "requested_analyses": [
                        "inventory_matrix",
                        "price_scenarios",
                        "advertising",
                        "conversions",
                        "forecast",
                    ],
                    "scenario_requests": ["price, marketing and supply variants"],
                    "required_domains": [],
                    "constraints": {"target_cover_days": 30, "horizon_days": 30},
                },
            },
            fallback,
        )

        self.assertTrue(decision.route_to_domain)
        self.assertEqual(decision.intent, QueryIntent.SUPPLY)
        self.assertEqual(decision.supporting_agents, [QueryIntent.PRICE, QueryIntent.MARKETING])
        self.assertEqual(decision.query_plan.scope, "portfolio")
        self.assertEqual(decision.query_plan.query_mode.value, "decision")
        self.assertEqual(
            [target.value for target in decision.query_plan.targets],
            ["46227", "46303"],
        )

    def test_router_recovers_structured_llm_plan_from_guardrail_exception(self):
        recovered_output = DomainRouteDecision(
            route_to_domain=True,
            intent=QueryIntent.SUPPLY,
            confidence="high",
            reason="portfolio inventory decision",
            supporting_agents=[QueryIntent.PRICE, QueryIntent.MARKETING],
            query_plan=QueryPlan(
                query_mode="decision",
                scope="portfolio",
                decision_kind="action_portfolio",
                targets=[
                    {"target_type": "vendor_code", "value": "46227"},
                    {"target_type": "vendor_code", "value": "46303"},
                ],
                requested_analyses=["inventory_matrix", "price_scenarios"],
                required_domains=[QueryIntent.SUPPLY, QueryIntent.PRICE],
            ),
        )
        tripwire = OutputGuardrailTripwireTriggered(
            SimpleNamespace(
                guardrail=SimpleNamespace(),
                output=SimpleNamespace(output_info={"violations": ["repairable"]}),
                agent_output=recovered_output,
            )
        )
        settings = _settings(db_host="db", db_port=5432, db_name="analytics")
        runtime = SimpleNamespace(provider="openai", missing_env_vars=())
        with (
            patch("agi_ecom_app.llm_routing.get_settings", return_value=settings),
            patch(
                "agi_ecom_app.llm_routing.get_active_llm_runtime_spec",
                return_value=runtime,
            ),
            patch("agi_ecom_app.llm_routing.build_intent_router_agent", return_value="router"),
            patch("agi_ecom_app.llm_routing.Runner.run_sync", side_effect=tripwire),
        ):
            decision = route_query_intent_with_llm(
                "Снизить покрытие всей матрицы до 30 дней; отдельно проверить 46227 и 46303"
            )

        self.assertEqual(decision.intent, QueryIntent.SUPPLY)
        self.assertEqual(decision.query_plan.scope, "portfolio")
        self.assertEqual(decision.query_plan.decision_kind, "action_portfolio")

    def test_openrouter_experimental_agents_use_plain_text_output_contract(self):
        settings = _settings(
            llm_model_profile=OPENROUTER_GEMMA_FREE_PROFILE,
            openrouter_api_key="sk-or-test",
        )

        with patch(
            "agi_ecom_app.services.runtime_settings._load_runtime_profile_from_db",
            return_value=None,
        ):
            spec = get_active_llm_runtime_spec(settings=settings)

        with patch("agi_ecom_app.main_agent.build_agent_model", return_value="fake-openrouter-model"):
            agent = _build_sales_agent(spec)

        self.assertFalse(spec.supports_agent_structured_outputs)
        self.assertIsNone(agent.output_type)
        self.assertEqual(agent.output_guardrails, [])


if __name__ == "__main__":
    unittest.main()
