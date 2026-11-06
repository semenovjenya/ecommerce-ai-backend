from __future__ import annotations

import json
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any
from uuid import uuid4

from agents import (
    Agent,
    AgentOutputSchema,
    GuardrailFunctionOutput,
    ModelSettings,
    OutputGuardrailTripwireTriggered,
    Runner,
    output_guardrail,
)

from agi_ecom_app.config import get_settings
from agi_ecom_app.forecast.models import ForecastRouteProposal
from agi_ecom_app.llm_runtime import (
    LLMRuntimeConfigurationError,
    LLMRuntimeSpec,
    build_agent_model,
    get_active_llm_runtime_spec,
)
from agi_ecom_app.llm_output_parsing import coerce_model_from_json_text
from agi_ecom_app.main_agent import build_run_hooks
from agi_ecom_app.schemas import EcomRunContext, QueryIntent

PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "forecast_router_agent.md"


def _load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


def _dynamic_instructions(context_wrapper, agent) -> str:
    del context_wrapper, agent
    return _load_prompt()


@output_guardrail
def forecast_router_output_guardrail(context, agent, output: Any) -> GuardrailFunctionOutput:
    del context, agent
    violations: list[str] = []
    if not isinstance(output, ForecastRouteProposal):
        violations.append("Final output is not ForecastRouteProposal.")
    elif output.route and not output.action_family:
        violations.append("route=true requires action_family.")
    return GuardrailFunctionOutput(
        output_info={"violations": violations},
        tripwire_triggered=bool(violations),
    )


def build_forecast_router_agent() -> Agent[EcomRunContext]:
    return _build_forecast_router_agent(get_active_llm_runtime_spec())


@lru_cache(maxsize=8)
def _build_forecast_router_agent(runtime_spec: LLMRuntimeSpec) -> Agent[EcomRunContext]:
    return Agent(
        name="Forecast Router Agent",
        instructions=_dynamic_instructions,
        model=build_agent_model(runtime_spec),
        model_settings=ModelSettings(tool_choice="auto"),
        tools=[],
        output_type=(
            AgentOutputSchema(ForecastRouteProposal, strict_json_schema=False)
            if runtime_spec.supports_agent_structured_outputs
            else None
        ),
        output_guardrails=(
            [forecast_router_output_guardrail]
            if runtime_spec.supports_agent_structured_outputs
            else []
        ),
    )


def route_task_with_llm(
    *,
    task_snapshot: dict[str, Any],
    candidate_families: list[str] | None = None,
    today: date | None = None,
) -> tuple[ForecastRouteProposal | None, dict[str, Any]]:
    key = _task_snapshot_cache_key(task_snapshot)
    runtime_spec = get_active_llm_runtime_spec(require_configured=False)
    route_json, debug = _route_task_with_llm_cached(
        key,
        json.dumps(candidate_families or [], ensure_ascii=False, sort_keys=True),
        today.isoformat() if today else "",
        runtime_spec.cache_key,
    )
    if route_json is None:
        return None, debug
    return ForecastRouteProposal.model_validate(route_json), debug


@lru_cache(maxsize=256)
def _route_task_with_llm_cached(
    task_snapshot_key: str,
    candidates_key: str,
    today_iso: str,
    runtime_cache_key: str,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    del runtime_cache_key
    settings = get_settings()
    runtime_spec = get_active_llm_runtime_spec(require_configured=False, settings=settings)
    if runtime_spec.missing_env_vars:
        if runtime_spec.provider != "openai":
            missing = ", ".join(runtime_spec.missing_env_vars)
            raise LLMRuntimeConfigurationError(
                f"Model profile `{runtime_spec.profile}` requires missing environment variable(s): {missing}."
            )
        return None, {
            "mode": "disabled",
            "reason": "missing_llm_api_key",
            "provider": runtime_spec.provider,
            "profile": runtime_spec.profile,
            "missing_env_vars": list(runtime_spec.missing_env_vars),
        }

    task_snapshot = json.loads(task_snapshot_key)
    candidate_families = json.loads(candidates_key)
    payload = {
        "title": task_snapshot.get("title"),
        "action": task_snapshot.get("action"),
        "targets": task_snapshot.get("targets"),
        "actionKind": task_snapshot.get("actionKind"),
        "decisionType": task_snapshot.get("decisionType"),
        "businessGoal": task_snapshot.get("businessGoal"),
        "requiredMetrics": task_snapshot.get("requiredMetrics"),
        "forecastSpec": task_snapshot.get("forecastSpec"),
        "observation": task_snapshot.get("observation"),
        "impact": task_snapshot.get("impact"),
        "cause": task_snapshot.get("cause"),
        "executionPlan": task_snapshot.get("executionPlan"),
        "evidence": task_snapshot.get("evidence"),
        "candidateFamilies": candidate_families,
    }
    context = EcomRunContext(
        user_query="Определи action family для forecast planner.",
        intent=QueryIntent.MARKETING,
        today=date.fromisoformat(today_iso) if today_iso else date.today(),
        db_target="analytics",
        chat_id=f"forecast-router-{uuid4()}",
        evidence_budget=0,
    )
    hooks = build_run_hooks()
    try:
        raw_output = Runner.run_sync(
            build_forecast_router_agent(),
            json.dumps(payload, ensure_ascii=False, indent=2),
            context=context,
            hooks=hooks,
            max_turns=4,
        ).final_output
        result = coerce_model_from_json_text(raw_output, ForecastRouteProposal)
        debug = {
            "mode": "agent",
            "used_tools": list(dict.fromkeys(context.tool_trace)),
            "agent_trace": list(context.agent_trace),
            "decision_trace": list(context.decision_trace),
        }
        return result.model_dump(mode="json"), debug
    except (OutputGuardrailTripwireTriggered, Exception) as exc:
        if runtime_spec.provider != "openai":
            raise
        return None, {
            "mode": "fallback",
            "reason": type(exc).__name__,
            "error": str(exc),
            "used_tools": list(dict.fromkeys(context.tool_trace)),
            "agent_trace": list(context.agent_trace),
            "decision_trace": list(context.decision_trace),
        }


build_forecast_router_agent.cache_clear = _build_forecast_router_agent.cache_clear  # type: ignore[attr-defined]


def _task_snapshot_cache_key(task_snapshot: dict[str, Any]) -> str:
    return json.dumps(task_snapshot, ensure_ascii=False, sort_keys=True, default=str)
