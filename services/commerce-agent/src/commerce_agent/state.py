from __future__ import annotations

from typing import Annotated, Any, Literal

from langchain_core.messages import AnyMessage
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import TypedDict

from commerce_agent.models import (
    DOMAIN_NAMES,
    ActionReceipt,
    ActionTask,
    AgentStatus,
    ApprovalRequest,
    ConfidenceReport,
    DataGap,
    DomainAnalysis,
    DomainName,
    EvidenceItem,
    Incident,
    ManagerDecision,
    QueryPlan,
    ResearchResult,
    SchemaVersion,
    WorkflowError,
)
from commerce_agent.reducers import (
    merge_agent_statuses,
    merge_finance_analysis,
    merge_gaps,
    merge_immutable_items,
    merge_marketing_analysis,
    merge_price_analysis,
    merge_sales_analysis,
    merge_supply_analysis,
    schema_version_reducer,
    safe_add_messages,
    set_once_equal,
)


class CommerceState(TypedDict, total=False):
    schema_version: Annotated[SchemaVersion, schema_version_reducer]
    messages: Annotated[list[AnyMessage], safe_add_messages]

    tenant_id: Annotated[str, set_once_equal]
    user_id: Annotated[str, set_once_equal]
    request_id: Annotated[str, set_once_equal]
    query_plan: QueryPlan | None

    agent_statuses: Annotated[dict[str, AgentStatus], merge_agent_statuses]
    sales_analysis: Annotated[DomainAnalysis | None, merge_sales_analysis]
    marketing_analysis: Annotated[DomainAnalysis | None, merge_marketing_analysis]
    supply_analysis: Annotated[DomainAnalysis | None, merge_supply_analysis]
    price_analysis: Annotated[DomainAnalysis | None, merge_price_analysis]
    finance_analysis: Annotated[DomainAnalysis | None, merge_finance_analysis]
    research_result: ResearchResult | None

    detected_incidents: Annotated[list[Incident], merge_immutable_items]
    evidence_matrix: Annotated[list[EvidenceItem], merge_immutable_items]
    data_gaps: Annotated[list[DataGap], merge_gaps]
    action_tasks: Annotated[list[ActionTask], merge_immutable_items]
    confidence: ConfidenceReport | None
    manager_decision: ManagerDecision | None

    pending_interrupt: ApprovalRequest | None
    execution_receipts: Annotated[list[ActionReceipt], merge_immutable_items]
    errors: Annotated[list[WorkflowError], merge_immutable_items]


class CommerceStateSnapshot(BaseModel):
    """Versioned validation/JSON-schema view of the durable graph state."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    schema_version: Literal[1] = 1
    messages: list[Any] = Field(default_factory=list)
    tenant_id: str
    user_id: str
    request_id: str
    query_plan: QueryPlan | None = None
    agent_statuses: dict[str, AgentStatus] = Field(default_factory=dict)
    sales_analysis: DomainAnalysis | None = None
    marketing_analysis: DomainAnalysis | None = None
    supply_analysis: DomainAnalysis | None = None
    price_analysis: DomainAnalysis | None = None
    finance_analysis: DomainAnalysis | None = None
    research_result: ResearchResult | None = None
    detected_incidents: list[Incident] = Field(default_factory=list)
    evidence_matrix: list[EvidenceItem] = Field(default_factory=list)
    data_gaps: list[DataGap] = Field(default_factory=list)
    action_tasks: list[ActionTask] = Field(default_factory=list)
    confidence: ConfidenceReport | None = None
    manager_decision: ManagerDecision | None = None
    pending_interrupt: ApprovalRequest | None = None
    execution_receipts: list[ActionReceipt] = Field(default_factory=list)
    errors: list[WorkflowError] = Field(default_factory=list)


class CommerceInput(TypedDict):
    """Only browser-originated data accepted by the deployed graph."""

    schema_version: Annotated[SchemaVersion, schema_version_reducer]
    messages: Annotated[list[AnyMessage], safe_add_messages]


class DomainSubgraphInput(TypedDict, total=False):
    """Narrow projection available to every fixed domain branch."""

    query_plan: QueryPlan | None


class DomainSubgraphSharedOutput(TypedDict, total=False):
    """Append-only fields shared by owner-specific domain output schemas."""

    agent_statuses: Annotated[dict[str, AgentStatus], merge_agent_statuses]
    data_gaps: Annotated[list[DataGap], merge_gaps]
    errors: Annotated[list[WorkflowError], merge_immutable_items]


class SalesSubgraphOutput(DomainSubgraphSharedOutput, total=False):
    sales_analysis: Annotated[DomainAnalysis | None, merge_sales_analysis]


class MarketingSubgraphOutput(DomainSubgraphSharedOutput, total=False):
    marketing_analysis: Annotated[DomainAnalysis | None, merge_marketing_analysis]


class SupplySubgraphOutput(DomainSubgraphSharedOutput, total=False):
    supply_analysis: Annotated[DomainAnalysis | None, merge_supply_analysis]


class PriceSubgraphOutput(DomainSubgraphSharedOutput, total=False):
    price_analysis: Annotated[DomainAnalysis | None, merge_price_analysis]


class FinanceSubgraphOutput(DomainSubgraphSharedOutput, total=False):
    finance_analysis: Annotated[DomainAnalysis | None, merge_finance_analysis]


class SalesSubgraphState(DomainSubgraphInput, SalesSubgraphOutput):
    pass


class MarketingSubgraphState(DomainSubgraphInput, MarketingSubgraphOutput):
    pass


class SupplySubgraphState(DomainSubgraphInput, SupplySubgraphOutput):
    pass


class PriceSubgraphState(DomainSubgraphInput, PriceSubgraphOutput):
    pass


class FinanceSubgraphState(DomainSubgraphInput, FinanceSubgraphOutput):
    pass


DOMAIN_SUBGRAPH_SCHEMAS = {
    DomainName.SALES: (SalesSubgraphState, SalesSubgraphOutput),
    DomainName.MARKETING: (MarketingSubgraphState, MarketingSubgraphOutput),
    DomainName.SUPPLY: (SupplySubgraphState, SupplySubgraphOutput),
    DomainName.PRICE: (PriceSubgraphState, PriceSubgraphOutput),
    DomainName.FINANCE: (FinanceSubgraphState, FinanceSubgraphOutput),
}


STATE_FIELD_NAMES = frozenset(CommerceState.__annotations__)


def initial_agent_statuses() -> dict[str, AgentStatus]:
    return {
        domain.value: AgentStatus(domain=domain, status="pending", attempt=0)
        for domain in DOMAIN_NAMES
    }


def empty_state_update() -> CommerceState:
    return {
        "schema_version": 1,
        "agent_statuses": initial_agent_statuses(),
        "detected_incidents": [],
        "evidence_matrix": [],
        "data_gaps": [],
        "action_tasks": [],
        "execution_receipts": [],
        "errors": [],
    }


__all__ = [
    "STATE_FIELD_NAMES",
    "CommerceState",
    "CommerceStateSnapshot",
    "CommerceInput",
    "DOMAIN_SUBGRAPH_SCHEMAS",
    "DomainSubgraphInput",
    "empty_state_update",
    "initial_agent_statuses",
]
