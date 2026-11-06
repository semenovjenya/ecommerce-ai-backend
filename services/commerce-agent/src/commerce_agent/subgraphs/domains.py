from __future__ import annotations

from collections.abc import Callable

from langgraph.graph import END, START, StateGraph

from commerce_agent.context import CommerceRuntimeContext
from commerce_agent.models import (
    AgentStatus,
    DataGap,
    DomainAnalysis,
    DomainName,
    QueryPlan,
    WorkflowError,
)
from commerce_agent.state import (
    DOMAIN_SUBGRAPH_SCHEMAS,
    CommerceState,
    DomainSubgraphInput,
)


def _is_required(state: CommerceState, domain: DomainName) -> bool:
    raw_plan = state.get("query_plan")
    if raw_plan is None:
        return True
    plan = raw_plan if isinstance(raw_plan, QueryPlan) else QueryPlan.model_validate(raw_plan)
    return domain in plan.required_domains


def _status(
    domain: DomainName,
    status: str,
    node: str,
    detail: str,
    *,
    attempt: int,
) -> CommerceState:
    return {
        "agent_statuses": {
            domain.value: AgentStatus(
                domain=domain,
                status=status,
                attempt=attempt,
                node=node,
                detail=detail,
            )
        }
    }


def _prepare(domain: DomainName) -> Callable[[CommerceState], CommerceState]:
    def prepare(state: CommerceState) -> CommerceState:
        if not _is_required(state, domain):
            return _status(
                domain, "skipped", "prepare", "domain_not_required", attempt=1
            )
        return _status(domain, "running", "prepare", "m1_skeleton", attempt=1)

    return prepare


def _collect_evidence(domain: DomainName) -> Callable[[CommerceState], CommerceState]:
    def collect_evidence(state: CommerceState) -> CommerceState:
        if not _is_required(state, domain):
            return {}
        return {
            **_status(
                domain,
                "running",
                "collect_evidence",
                "tools_not_connected",
                attempt=2,
            ),
            "data_gaps": [
                DataGap(
                    gap_id=f"m1:{domain.value}:tools-not-connected",
                    domain=domain,
                    reason="Domain tool adapters are intentionally deferred to M2.",
                    blocking=True,
                )
            ],
            "errors": [
                WorkflowError(
                    error_id=f"m1:{domain.value}:not-implemented",
                    code="M1_DOMAIN_NOT_IMPLEMENTED",
                    message=f"{domain.value} subgraph has no tools or model in M1.",
                    node=f"{domain.value}.collect_evidence",
                    retryable=False,
                    severity="info",
                )
            ],
        }

    return collect_evidence


def _analyze(domain: DomainName) -> Callable[[CommerceState], CommerceState]:
    def analyze(state: CommerceState) -> CommerceState:
        if not _is_required(state, domain):
            return {}
        analysis_field = f"{domain.value}_analysis"
        return {
            **_status(
                domain,
                "planned",
                "analyze",
                "implementation_deferred_to_m3",
                attempt=3,
            ),
            analysis_field: DomainAnalysis(
                domain=domain,
                revision=1,
                status="not_implemented",
                summary="M1 topology placeholder; no business conclusion was produced.",
            ),
        }

    return analyze


def _validate_and_publish(domain: DomainName) -> Callable[[CommerceState], CommerceState]:
    def validate_and_publish(state: CommerceState) -> CommerceState:
        if not _is_required(state, domain):
            return {}
        return _status(
            domain,
            "planned",
            "validate_and_publish",
            "awaiting_m2_m3",
            attempt=4,
        )

    return validate_and_publish


def build_domain_subgraph(domain: DomainName):
    state_schema, output_schema = DOMAIN_SUBGRAPH_SCHEMAS[domain]
    builder = StateGraph(
        state_schema,
        context_schema=CommerceRuntimeContext,
        input_schema=DomainSubgraphInput,
        output_schema=output_schema,
    )
    builder.add_node("prepare", _prepare(domain))
    builder.add_node("collect_evidence", _collect_evidence(domain))
    builder.add_node("analyze", _analyze(domain))
    builder.add_node("validate_and_publish", _validate_and_publish(domain))
    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", "collect_evidence")
    builder.add_edge("collect_evidence", "analyze")
    builder.add_edge("analyze", "validate_and_publish")
    builder.add_edge("validate_and_publish", END)
    return builder.compile(name=f"{domain.value}_subgraph")


__all__ = ["build_domain_subgraph"]
