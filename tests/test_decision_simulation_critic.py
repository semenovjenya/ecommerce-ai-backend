from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agi_ecom_app.decision_simulation.critic import (
    DecisionCriticInput,
    build_decision_critic_agent,
    run_decision_critic,
)
from agi_ecom_app.llm_runtime import LLMRuntimeSpec
from agi_ecom_app.schemas import QueryIntent


_RUNTIME = LLMRuntimeSpec(
    profile="test",
    provider="openai",
    model_name="test-model",
    api_style="responses",
    base_url=None,
    default_headers=(),
    experimental=False,
    missing_env_vars=(),
)


def _critic_input(*, recalculation_attempt: int = 0) -> DecisionCriticInput:
    candidates = []
    for rank, candidate_id in enumerate(("candidate-1", "candidate-2", "candidate-3"), 1):
        candidates.append(
            {
                "candidate_id": candidate_id,
                "action_id": f"action-{rank}",
                "rank": rank,
                "is_winner": rank == 1,
                "evidence_refs": [f"evidence:{candidate_id}"],
                "distribution": {
                    "per_day": {"p10": 100.0 - rank, "mean": 150.0 - rank}
                },
                "score_breakdown": {"cvar10_per_day": 90.0 - rank},
            }
        )
    return DecisionCriticInput(
        model_card={
            "name": "advertising-bootstrap-v1",
            "sampleCount": 500,
            "batchSize": 50,
        },
        snapshot_freshness={
            "status": "fresh",
            "evidenceRefs": ["evidence:snapshot-freshness"],
        },
        candidate_aggregates=candidates,
        top_three=["candidate-1", "candidate-2", "candidate-3"],
        leader_delta={
            "leader_candidate_id": "candidate-1",
            "runner_up_candidate_id": "candidate-2",
            "cvar10_per_day_delta": 1.0,
            "evidence_refs": ["evidence:leader-delta"],
        },
        hard_constraints=[
            {
                "code": "zero_stock_pause_only",
                "status": "passed",
                "evidence_refs": ["evidence:hard-constraint"],
            }
        ],
        evidence_refs=["evidence:monitoring-spec"],
        recalculation_attempt=recalculation_attempt,
    )


def _run_with_output(output, *, mode="enforced", critic_input=None):
    with (
        patch(
            "agi_ecom_app.decision_simulation.critic.get_active_llm_runtime_spec",
            return_value=_RUNTIME,
        ),
        patch(
            "agi_ecom_app.decision_simulation.critic.Runner.run_sync",
            return_value=SimpleNamespace(final_output=output),
        ) as runner,
    ):
        execution = run_decision_critic(
            critic_input=critic_input or _critic_input(),
            mode=mode,
        )
    return execution, runner


class DecisionSimulationCriticTests(unittest.TestCase):
    def test_advertising_critic_keeps_marketing_route(self):
        execution, runner = _run_with_output(
            {
                "verdict": "approved",
                "issues": [],
                "summary": "Модель и evidence согласованы.",
            }
        )

        self.assertTrue(execution.llm_available)
        context = runner.call_args.kwargs["context"]
        self.assertEqual(context.intent, QueryIntent.MARKETING)
        self.assertEqual(
            context.user_query,
            "Проверь численное рекламное решение, не изменяя ranking.",
        )

    def test_pricing_model_card_routes_critic_to_price_domain(self):
        critic_input = _critic_input().model_copy(
            update={
                "model_card": {
                    "domain": "pricing",
                    "name": "pricing-elasticity-bootstrap-v1",
                    "sampleCount": 500,
                    "batchSize": 50,
                }
            }
        )
        execution, runner = _run_with_output(
            {
                "verdict": "approved",
                "issues": [],
                "summary": "Ценовая модель и evidence согласованы.",
            },
            critic_input=critic_input,
        )

        self.assertTrue(execution.llm_available)
        context = runner.call_args.kwargs["context"]
        self.assertEqual(context.intent, QueryIntent.PRICE)
        self.assertEqual(
            context.user_query,
            "Проверь численное ценовое решение, не изменяя ranking.",
        )

    def test_approved_output_continues_without_exposing_ranking_mutations(self):
        execution, runner = _run_with_output(
            {
                "verdict": "approved",
                "issues": [],
                "summary": "Модель, evidence и monitoring согласованы.",
            }
        )

        self.assertTrue(execution.llm_available)
        self.assertEqual(execution.result.verdict, "approved")
        self.assertEqual(execution.effective_action, "continue")
        self.assertFalse(execution.blocks_decision)
        result_fields = type(execution.result).model_fields
        self.assertNotIn("winner", result_fields)
        self.assertNotIn("rank", result_fields)
        self.assertNotIn("score", result_fields)

        prompt = json.loads(runner.call_args.args[1])
        self.assertEqual(prompt["criticInput"]["objective"], "incremental_contribution_profit")
        self.assertEqual(prompt["criticInput"]["top3"][0], "candidate-1")
        self.assertIn("modelCard", prompt["criticInput"])
        self.assertIn("snapshotFreshness", prompt["criticInput"])
        self.assertIn("candidateAggregates", prompt["criticInput"])
        self.assertIn("leaderDelta", prompt["criticInput"])
        self.assertTrue(prompt["rankingIntegrity"]["immutable"])

    def test_valid_grounded_blocker_blocks_enforced_decision(self):
        execution, _ = _run_with_output(
            {
                "verdict": "blocked",
                "issues": [
                    {
                        "code": "unsafe_execution",
                        "message": "Guardrail исполнения не покрывает подтверждённый риск.",
                        "evidenceRefs": ["evidence:hard-constraint"],
                        "affectedCandidates": ["candidate-1"],
                    }
                ],
                "summary": "Исполнение небезопасно до исправления guardrail.",
            }
        )

        self.assertTrue(execution.llm_available)
        self.assertEqual(execution.result.verdict, "blocked")
        self.assertEqual(execution.effective_action, "block")
        self.assertTrue(execution.blocks_decision)

    def test_blocker_without_evidence_is_rejected_as_unavailable(self):
        execution, _ = _run_with_output(
            {
                "verdict": "blocked",
                "issues": [
                    {
                        "code": "unsafe_execution",
                        "message": "Неподтверждённая блокировка.",
                        "evidenceRefs": [],
                        "affectedCandidates": ["candidate-1"],
                    }
                ],
                "summary": "Невалидный blocker.",
            }
        )

        self.assertFalse(execution.llm_available)
        self.assertEqual(execution.result.verdict, "blocked")
        self.assertEqual(execution.result.issues[0].code, "llm_critic_unavailable")
        self.assertEqual(execution.effective_action, "block")
        self.assertEqual(execution.debug["mode"], "unavailable")

    def test_unknown_evidence_or_candidate_cannot_ground_blocker(self):
        execution, _ = _run_with_output(
            {
                "verdict": "blocked",
                "issues": [
                    {
                        "code": "candidate_conflict",
                        "message": "Ссылка не принадлежит входному snapshot.",
                        "evidenceRefs": ["evidence:invented"],
                        "affectedCandidates": ["candidate-invented"],
                    }
                ],
                "summary": "Невалидная ссылка.",
            }
        )

        self.assertFalse(execution.llm_available)
        self.assertEqual(execution.result.issues[0].code, "llm_critic_unavailable")

    def test_output_cannot_override_winner_or_score(self):
        execution, _ = _run_with_output(
            {
                "verdict": "approved",
                "issues": [],
                "summary": "Попытка изменить winner.",
                "winnerCandidateId": "candidate-2",
                "score": 999999,
            }
        )

        self.assertFalse(execution.llm_available)
        self.assertEqual(execution.result.issues[0].code, "llm_critic_unavailable")
        self.assertEqual(execution.effective_action, "block")

    def test_shadow_failure_is_recorded_but_does_not_block(self):
        with (
            patch(
                "agi_ecom_app.decision_simulation.critic.get_active_llm_runtime_spec",
                return_value=_RUNTIME,
            ),
            patch(
                "agi_ecom_app.decision_simulation.critic.Runner.run_sync",
                side_effect=RuntimeError("provider unavailable"),
            ),
        ):
            execution = run_decision_critic(
                critic_input=_critic_input(),
                mode="shadow",
            )

        self.assertFalse(execution.llm_available)
        self.assertEqual(execution.result.issues[0].code, "llm_critic_unavailable")
        self.assertEqual(execution.effective_action, "record_only")
        self.assertFalse(execution.blocks_decision)

    def test_enforced_failure_blocks_without_changing_numeric_ranking(self):
        with (
            patch(
                "agi_ecom_app.decision_simulation.critic.get_active_llm_runtime_spec",
                return_value=_RUNTIME,
            ),
            patch(
                "agi_ecom_app.decision_simulation.critic.Runner.run_sync",
                side_effect=RuntimeError("provider unavailable"),
            ),
        ):
            execution = run_decision_critic(
                critic_input=_critic_input(),
                mode="enforced",
            )

        self.assertFalse(execution.llm_available)
        self.assertEqual(execution.effective_action, "block")
        self.assertTrue(execution.blocks_decision)
        self.assertEqual(execution.ranking_fingerprint, _critic_input().ranking_fingerprint)

    def test_recalculate_is_allowed_once_then_becomes_blocked(self):
        output = {
            "verdict": "recalculate",
            "issues": [
                {
                    "code": "stale_evidence",
                    "message": "Snapshot рекламных расходов устарел.",
                    "evidenceRefs": ["evidence:snapshot-freshness"],
                    "affectedCandidates": ["candidate-1", "candidate-2"],
                }
            ],
            "summary": "Нужен один пересчёт со свежим snapshot.",
        }

        first, _ = _run_with_output(output, critic_input=_critic_input(recalculation_attempt=0))
        second, _ = _run_with_output(output, critic_input=_critic_input(recalculation_attempt=1))

        self.assertEqual(first.result.verdict, "recalculate")
        self.assertEqual(first.effective_action, "recalculate")
        self.assertTrue(first.blocks_decision)
        self.assertEqual(second.result.verdict, "blocked")
        self.assertEqual(second.effective_action, "block")
        self.assertTrue(second.debug["recalculation_exhausted"])

    def test_agent_uses_dynamic_instructions_and_no_tools(self):
        agent = build_decision_critic_agent(_RUNTIME)
        instructions = agent.instructions(SimpleNamespace(context=None), None)

        self.assertIn("не изменяй числа", instructions)
        self.assertIn("evidenceRefs", instructions)
        self.assertEqual(agent.tools, [])


if __name__ == "__main__":
    unittest.main()
