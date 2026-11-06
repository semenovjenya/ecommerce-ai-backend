from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langgraph.graph import END, START, StateGraph

from commerce_agent.context import CommerceRuntimeContext
from commerce_agent.models import DOMAIN_NAMES
from commerce_agent.nodes.core import (
    action_execution_stub,
    approval_interrupt_stub,
    deep_research_stub,
    effect_measurement_schedule_stub,
    evidence_gate_stub,
    gap_resolution_stub,
    intake,
    query_plan_stub,
    router_stub,
    synthesis_stub,
)
from commerce_agent.observability import configure_graph_observability
from commerce_agent.routing import route_after_evidence_gate, route_after_gap_resolution
from commerce_agent.state import CommerceInput, CommerceState
from commerce_agent.subgraphs import build_domain_subgraph


GRAPH_ID = "commerce_agent"


def build_graph(
    *,
    checkpointer: Any = None,
    interrupt_before: Sequence[str] | None = None,
    interrupt_after: Sequence[str] | None = None,
):
    """Compile the M1 graph; the caller owns the checkpointer lifecycle."""

    builder = StateGraph(
        CommerceState,
        context_schema=CommerceRuntimeContext,
        input_schema=CommerceInput,
    )
    builder.add_node("intake", intake)
    builder.add_node("router", router_stub)
    builder.add_node("query_plan", query_plan_stub)

    domain_nodes = [domain.value for domain in DOMAIN_NAMES]
    for domain in DOMAIN_NAMES:
        builder.add_node(domain.value, build_domain_subgraph(domain))

    builder.add_node("evidence_gate", evidence_gate_stub)
    builder.add_node("gap_resolution", gap_resolution_stub)
    builder.add_node("deep_research", deep_research_stub)
    builder.add_node("synthesis", synthesis_stub)
    builder.add_node("approval_interrupt", approval_interrupt_stub)
    builder.add_node("action_execution", action_execution_stub)
    builder.add_node(
        "effect_measurement_schedule",
        effect_measurement_schedule_stub,
    )
    builder.add_edge(START, "intake")
    builder.add_edge("intake", "router")
    builder.add_edge("router", "query_plan")
    for domain_node in domain_nodes:
        builder.add_edge("query_plan", domain_node)
    builder.add_edge(domain_nodes, "evidence_gate")
    builder.add_conditional_edges(
        "evidence_gate",
        route_after_evidence_gate,
        {
            "gap_resolution": "gap_resolution",
            "synthesis": "synthesis",
        },
    )
    builder.add_conditional_edges(
        "gap_resolution",
        route_after_gap_resolution,
        {
            "deep_research": "deep_research",
            "synthesis": "synthesis",
        },
    )
    builder.add_edge("deep_research", "synthesis")
    builder.add_edge("synthesis", "approval_interrupt")
    builder.add_edge("approval_interrupt", "action_execution")
    builder.add_edge("action_execution", "effect_measurement_schedule")
    builder.add_edge("effect_measurement_schedule", END)
    compiled_graph = builder.compile(
        checkpointer=checkpointer,
        interrupt_before=list(interrupt_before) if interrupt_before else None,
        interrupt_after=list(interrupt_after) if interrupt_after else None,
        name=GRAPH_ID,
    )
    return configure_graph_observability(compiled_graph)


graph = build_graph()


__all__ = ["GRAPH_ID", "build_graph", "graph"]
