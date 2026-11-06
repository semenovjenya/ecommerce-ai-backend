from __future__ import annotations

from langgraph.runtime import Runtime

from commerce_agent.context import CommerceRuntimeContext
from commerce_agent.models import QueryPlan, WorkflowError
from commerce_agent.safety import assert_secret_free
from commerce_agent.state import CommerceState, empty_state_update


class RuntimeIdentityMismatchError(ValueError):
    """Browser-supplied identity disagrees with the verified BFF context."""


class M1WritePathDisabledError(RuntimeError):
    """M1 must never accept or execute an action proposal."""


def intake(
    state: CommerceState,
    runtime: Runtime[CommerceRuntimeContext],
) -> CommerceState:
    assert_secret_free(state)
    context = runtime.context
    for field_name in ("tenant_id", "user_id", "request_id"):
        supplied = state.get(field_name)
        verified = getattr(context, field_name)
        if supplied not in (None, "", verified):
            raise RuntimeIdentityMismatchError(
                f"{field_name} does not match verified runtime context"
            )

    update = empty_state_update()
    update.update(
        {
            "tenant_id": context.tenant_id,
            "user_id": context.user_id,
            "request_id": context.request_id,
        }
    )
    assert_secret_free(update)
    return update


def router_stub(_: CommerceState) -> CommerceState:
    return {}


def query_plan_stub(state: CommerceState) -> CommerceState:
    existing = state.get("query_plan")
    if existing is None:
        plan = QueryPlan()
    elif isinstance(existing, QueryPlan):
        plan = existing
    else:
        plan = QueryPlan.model_validate(existing)
    return {"query_plan": plan}


def evidence_gate_stub(_: CommerceState) -> CommerceState:
    return {}


def gap_resolution_stub(_: CommerceState) -> CommerceState:
    return {}


def deep_research_stub(_: CommerceState) -> CommerceState:
    return {
        "errors": [
            WorkflowError(
                error_id="m1:deep-research:disabled",
                code="M1_DEEP_RESEARCH_DISABLED",
                message="Deep research is intentionally disabled until M5.",
                node="deep_research",
                retryable=False,
                severity="info",
            )
        ]
    }


def synthesis_stub(_: CommerceState) -> CommerceState:
    return {
        "errors": [
            WorkflowError(
                error_id="m1:synthesis:not-implemented",
                code="M1_GRAPH_SKELETON_ONLY",
                message="Synthesis is intentionally unavailable until M4.",
                node="synthesis",
                retryable=False,
                severity="info",
            )
        ]
    }


def approval_interrupt_stub(_: CommerceState) -> CommerceState:
    return {}


def action_execution_stub(state: CommerceState) -> CommerceState:
    if state.get("action_tasks") or state.get("pending_interrupt"):
        raise M1WritePathDisabledError(
            "action execution is unavailable in the M1 skeleton"
        )
    return {}


def effect_measurement_schedule_stub(_: CommerceState) -> CommerceState:
    return {}


__all__ = [
    "RuntimeIdentityMismatchError",
    "M1WritePathDisabledError",
    "action_execution_stub",
    "approval_interrupt_stub",
    "deep_research_stub",
    "effect_measurement_schedule_stub",
    "evidence_gate_stub",
    "gap_resolution_stub",
    "intake",
    "query_plan_stub",
    "router_stub",
    "synthesis_stub",
]
