from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic_core import core_schema


class SchemaVersion(int):
    """Reducer identity that makes even LangGraph's step -1 checkpoint versioned."""

    def __new__(cls, value: int = 1):
        if int(value) != 1:
            raise ValueError("only CommerceState schema_version=1 is supported")
        return super().__new__(cls, 1)

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: Any):
        del source_type, handler
        return core_schema.no_info_after_validator_function(
            cls,
            core_schema.literal_schema([1]),
        )


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DomainName(str, Enum):
    SALES = "sales"
    MARKETING = "marketing"
    SUPPLY = "supply"
    PRICE = "price"
    FINANCE = "finance"


DOMAIN_NAMES: tuple[DomainName, ...] = tuple(DomainName)


class QueryPlan(FrozenModel):
    objective: str = "answer_user_request"
    intent: str = "commerce_analysis"
    required_domains: tuple[DomainName, ...] = DOMAIN_NAMES
    requested_analyses: tuple[str, ...] = ()
    allowed_action_families: tuple[Literal["price", "marketing", "supply"], ...] = ()
    constraints: dict[str, Any] = Field(default_factory=dict)


class AgentStatus(FrozenModel):
    domain: DomainName
    status: Literal[
        "pending",
        "running",
        "planned",
        "completed",
        "failed",
        "skipped",
        "interrupted",
        "cancelled",
    ] = "pending"
    attempt: int = Field(default=0, ge=0)
    node: str | None = None
    detail: str | None = None


class DomainAnalysis(FrozenModel):
    domain: DomainName
    revision: int = Field(default=1, ge=1)
    status: Literal["not_implemented", "completed", "partial", "failed"]
    summary: str
    evidence_refs: tuple[str, ...] = ()


class EvidenceItem(FrozenModel):
    evidence_ref: str = Field(min_length=1)
    domain: DomainName
    claim: str
    source: str
    period: str | None = None
    freshness: Literal["fresh", "stale", "future", "unknown"] = "unknown"


class DataGap(FrozenModel):
    gap_id: str = Field(min_length=1)
    domain: DomainName | None = None
    reason: str
    blocking: bool = True
    status: Literal["open", "resolved", "accepted_limitation"] = "open"
    revision: int = Field(default=1, ge=1)


class Incident(FrozenModel):
    incident_id: str = Field(min_length=1)
    severity: Literal["info", "warning", "critical"]
    summary: str


class ActionTask(FrozenModel):
    action_id: str = Field(min_length=1)
    action_family: Literal["price", "marketing", "supply"]
    summary: str
    status: Literal["proposed", "blocked", "approved", "rejected", "executed"] = "proposed"


class ConfidenceReport(FrozenModel):
    score: float = Field(ge=0.0, le=1.0)
    level: Literal["low", "medium", "high"]
    limitations: tuple[str, ...] = ()


class ManagerDecision(FrozenModel):
    decision_id: str
    status: Literal["needs_data", "hold", "ready_for_approval"]
    summary: str


class ApprovalRequest(FrozenModel):
    interrupt_id: str
    action_id: str
    summary: str
    idempotency_key: str


class ActionReceipt(FrozenModel):
    receipt_id: str
    action_id: str
    idempotency_key: str
    status: Literal["executed", "verified", "failed", "unknown"]
    audit_ref: str | None = None
    verification_ref: str | None = None


class WorkflowError(FrozenModel):
    error_id: str
    code: str
    message: str
    node: str
    retryable: bool = False
    severity: Literal["info", "warning", "error", "critical"] = "error"


class ResearchResult(FrozenModel):
    brief_id: str
    status: Literal["completed", "partial", "failed", "cancelled"]
    findings: tuple[str, ...] = ()
    evidence: tuple[EvidenceItem, ...] = ()
    limitations: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()
    proposed_checks: tuple[str, ...] = ()
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


__all__ = [
    "DOMAIN_NAMES",
    "ActionReceipt",
    "ActionTask",
    "AgentStatus",
    "ApprovalRequest",
    "ConfidenceReport",
    "DataGap",
    "DomainAnalysis",
    "DomainName",
    "EvidenceItem",
    "Incident",
    "ManagerDecision",
    "QueryPlan",
    "ResearchResult",
    "SchemaVersion",
    "WorkflowError",
]
