from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class QueryIntent(str, Enum):
    FINANCE = "finance"
    MARKETING = "marketing"
    PRICE = "price"
    SALES = "sales"
    SUPPLY = "supply"


class WorkflowVersion(str, Enum):
    V1 = "v1"
    V2_PROMPT = "v2_prompt"
    V2_OPTIMIZED = "v2_optimized"


class RoundType(str, Enum):
    ROUTER = "router"
    PRIMARY = "primary"
    CROSS_CHECK = "cross_check"
    FOLLOWUP = "followup"
    GAP_RESOLUTION = "gap_resolution"
    CLAIM_VERIFICATION = "claim_verification"
    SYNTHESIS = "synthesis"
    TERMINAL_ACTION = "terminal_action"


class QueryMode(str, Enum):
    DECISION = "decision"
    DIAGNOSTIC = "diagnostic"
    REPORT = "report"
    LOOKUP = "lookup"


class MemoryMode(str, Enum):
    FULL = "full"
    SUMMARY_RECENT = "summary_recent"
    SUMMARY_ONLY = "summary_only"
    NONE = "none"


class ChatRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class DataGap(BaseModel):
    metric: str = Field(..., description="Missing metric or business dimension.")
    reason: str = Field(..., description="Why the metric is unavailable.")
    impact: str = Field(..., description="How the missing metric limits the report.")


class CrossAgentRequest(BaseModel):
    target_agent: QueryIntent = Field(..., description="Domain agent that can close the evidence gap.")
    reason: str = Field(..., description="Why the primary agent needs this cross-domain check.")
    question: str = Field(..., description="Specific narrow question for the target agent.")
    period: str | None = Field(default=None, description="Requested period, if known.")
    vendor_codes: list[str] = Field(default_factory=list, description="Vendor codes to check.")
    nm_ids: list[int | str] = Field(default_factory=list, description="WB nm_id values to check.")
    required_checks: list[str] = Field(
        default_factory=list,
        description="Specific facts or metrics the target agent should verify.",
    )


class DriverClass(str, Enum):
    PRIMARY_DRIVER = "primary_driver"
    SECONDARY_DRIVER = "secondary_driver"
    AMPLIFIER = "amplifier"
    NOT_A_DRIVER = "not_a_driver"


class ReportTable(BaseModel):
    title: str
    columns: list[str]
    rows: list[list[str | int | float | None]] = Field(default_factory=list)


class CrossAgentEvidence(BaseModel):
    target_agent: QueryIntent
    request_question: str
    summary: str
    facts: list[str] = Field(default_factory=list)
    tables: list[ReportTable] = Field(default_factory=list)
    data_gaps: list[DataGap] = Field(default_factory=list)


class CrossDomainDependencyItem(BaseModel):
    target_agent: QueryIntent
    reason: str
    question: str
    priority: int
    source: str
    period: str | None = None
    vendor_codes: list[str] = Field(default_factory=list)
    nm_ids: list[int | str] = Field(default_factory=list)
    required_checks: list[str] = Field(default_factory=list)


class CrossDomainDependencyPlan(BaseModel):
    primary_agent: QueryIntent
    recommended_secondary_agents: list[CrossDomainDependencyItem] = Field(default_factory=list)
    skipped_agents: list[CrossDomainDependencyItem] = Field(default_factory=list)
    max_secondary_agents: int = 2


class DiagnosticFinding(BaseModel):
    factor: str
    observation: str
    impact: str
    cause: str
    action: str
    evidence: list[str] = Field(default_factory=list)
    confidence: str
    driver_class: DriverClass

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(cls, value):
        normalized = str(value or "").strip().lower().replace("ё", "е")
        aliases = {
            "высокая": "high",
            "высокий": "high",
            "высокое": "high",
            "high": "high",
            "средняя": "medium",
            "средний": "medium",
            "среднее": "medium",
            "medium": "medium",
            "низкая": "low",
            "низкий": "low",
            "низкое": "low",
            "low": "low",
        }
        return aliases.get(normalized, value)


class AgentDiagnosticResult(BaseModel):
    agent_name: str
    summary: str
    facts: list["EvidenceFact"] = Field(default_factory=list)
    tables: list[ReportTable] = Field(default_factory=list)
    findings: list[DiagnosticFinding] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    data_gaps: list[DataGap] = Field(default_factory=list)
    cross_agent_requests: list[CrossAgentRequest] = Field(default_factory=list)


class RoundTask(BaseModel):
    round_type: RoundType
    target_agent: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    original_user_query: str = Field(..., min_length=1)
    verification_goal: dict[str, Any] | str
    required_checks: list[str]
    source: str
    gap_id: int | str | None
    claim_policy: str | None
    period: str | None
    sku: str | None
    vendor_codes: list[str]
    nm_ids: list[int | str]
    allowed_tools: list[str]
    required_tools: list[str]
    query_plan: dict[str, Any] = Field(
        default_factory=dict,
        description="Semantic query plan produced by the LLM router.",
    )


class PromptReport(BaseModel):
    workflow_version: WorkflowVersion
    round_type: RoundType
    query_mode: QueryMode
    memory_mode: MemoryMode
    system_chars: int = 0
    input_chars: int = 0
    total_chars: int = 0
    max_chars: int | None = None
    included_evidence: int = 0
    dropped_evidence: int = 0
    included_findings: int = 0
    dropped_findings: int = 0
    included_tables: int = 0
    dropped_tables: int = 0
    included_prior_evidence: int = 0
    dropped_prior_evidence: int = 0
    allowed_tools: list[str] = Field(default_factory=list)
    required_tools: list[str] = Field(default_factory=list)
    truncated: bool = False

    def as_metadata(self) -> dict[str, Any]:
        return {"_prompt_report": self.model_dump(mode="json")}


class PromptBuildResult(BaseModel):
    text: str
    report: PromptReport


class EvidenceFact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric: str = Field(..., min_length=1)
    value: str | int | float | bool | None = None
    unit: str | None = None
    period: str | None = None
    period_start: date | datetime | str | None = None
    period_end: date | datetime | str | None = None
    sku: str | None = None
    target_type: str | None = None
    target_id: str | None = None
    source_table: str | None = None
    source_snapshot_at: date | datetime | str | None = None
    retrieved_at: datetime | str | None = None
    metric_definition: str | None = None
    freshness_status: Literal["fresh", "stale", "future", "unknown"] = "unknown"
    freshness_lag_days: int | None = None
    statement: str | None = None
    evidence_ref: str = Field(..., min_length=1)
    source_agent: str | None = None
    status: Literal["confirmed", "refuted", "unconfirmed"] = "confirmed"
    confidence: Literal["low", "medium", "high"] = "medium"

    @staticmethod
    def _metric_is_identifier(metric: str) -> bool:
        normalized = re.sub(
            r"[^a-zа-я0-9]+",
            "_",
            metric.lower().replace("ё", "е"),
        ).strip("_")
        if normalized in {
            "nm_id",
            "nmid",
            "sku",
            "sku_id",
            "vendor_code",
            "vendorcode",
            "артикул",
            "артикул_продавца",
            "код_продавца",
        }:
            return True
        return bool(re.fullmatch(r"[a-z][a-z0-9_]*_id", normalized))

    @model_validator(mode="after")
    def validate_numeric_context(self) -> "EvidenceFact":
        metric_is_identifier = self._metric_is_identifier(self.metric)
        value_is_numeric = isinstance(self.value, (int, float)) and not isinstance(
            self.value,
            bool,
        )
        if (
            value_is_numeric
            and not metric_is_identifier
            and not str(self.unit or "").strip()
        ):
            raise ValueError("unit is required when numeric value is present")
        if (
            self.value is not None
            and not metric_is_identifier
            and not str(self.period or "").strip()
        ):
            raise ValueError("period is required when value is present")
        return self


class CrossCheckResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_name: str
    summary: str
    verdict: Literal["confirmed", "refuted", "no_data"]
    facts: list[EvidenceFact] = Field(default_factory=list)
    tables: list[ReportTable] = Field(default_factory=list)
    data_gaps: list[DataGap] = Field(default_factory=list)
    followup_requests: list[CrossAgentRequest] = Field(default_factory=list, max_length=2)


class FollowupResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_name: str
    summary: str
    status: Literal["safe", "unsafe", "needs_review", "no_data"]
    facts: list[EvidenceFact] = Field(default_factory=list)
    tables: list[ReportTable] = Field(default_factory=list)
    data_gaps: list[DataGap] = Field(default_factory=list)


class GapResolutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_name: str
    gap_id: int | str
    summary: str
    outcome: Literal["resolved", "no_data", "accepted_limitation"]
    fact: EvidenceFact | None = None
    reason: str | None = None
    impact: str | None = None

    @model_validator(mode="after")
    def validate_resolution_evidence(self) -> "GapResolutionResult":
        if self.outcome == "resolved" and self.fact is None:
            raise ValueError("fact is required when outcome=resolved")
        if self.outcome == "resolved" and self.fact is not None and self.fact.value is None:
            raise ValueError("fact.value is required when outcome=resolved")
        return self


class ClaimVerificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_name: str
    claim: str
    summary: str
    verdict: Literal["confirmed", "refuted", "unverified"]
    facts: list[EvidenceFact] = Field(default_factory=list)
    unchecked_checks: list[str] = Field(default_factory=list)
    data_gaps: list[DataGap] = Field(default_factory=list)


class MarketingDiagnosticResult(AgentDiagnosticResult):
    marketing_report_snapshot: ReportTable | None = Field(
        default=None,
        description="Mandatory top table for marketing reports: yesterday/day-before, week/week, month/month with ДРР.",
    )


class SalesDiagnosticResult(BaseModel):
    agent_name: str
    summary: str
    facts: list[EvidenceFact] = Field(default_factory=list)
    sales_report_snapshot: ReportTable | None = Field(
        default=None,
        description="Mandatory top table for sales reports: yesterday/day-before, week/week, month/month.",
    )
    sales_overview: ReportTable
    sales_dynamics: ReportTable
    sales_decomposition: ReportTable
    findings: list[DiagnosticFinding] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    data_gaps: list[DataGap] = Field(default_factory=list)
    cross_agent_requests: list[CrossAgentRequest] = Field(default_factory=list)

    def as_tables(self) -> list[ReportTable]:
        return [self.sales_overview, self.sales_dynamics, self.sales_decomposition]


class ActionExecutionCheck(BaseModel):
    name: str
    status: Literal["ok", "warning", "blocked", "unknown"]
    summary: str
    evidence: list[str] = Field(default_factory=list)


class ActionExecutionForecast(BaseModel):
    salesVelocity: float | None = Field(
        default=None,
        description="Recent average sales units per active day for the target SKU/object.",
    )
    activeDays: int | None = None
    availableStock: float | None = None
    inboundStock: float | None = None
    coverageDays: float | None = None
    leadTimeDays: int | None = None
    safetyStockUnits: float | None = None
    reorderQty: float | None = None
    stockoutRisk: Literal["low", "medium", "high", "unknown"] = "unknown"
    assumptions: list[str] = Field(default_factory=list)


class ActionExecutionResult(BaseModel):
    businessMeaning: str
    checks: list[ActionExecutionCheck] = Field(default_factory=list)
    forecast: ActionExecutionForecast | None = None
    decision: str
    decisionStatus: Literal["completed", "needs_review", "blocked"]
    confidence: Literal["low", "medium", "high"]
    blockedBy: list[str] = Field(default_factory=list)
    finalRecommendation: str
    summary: str | None = Field(
        default=None,
        description="One short card-friendly summary of the execution result.",
    )


GLOBAL_DATA_GAPS: list[DataGap] = [
    DataGap(
        metric="COGS",
        reason="Закупочная себестоимость доступна только для SKU, загруженных в finance.wb_product_coast.",
        impact="Нельзя подтверждать contribution после COGS и чистую прибыльность после закупки для SKU без строки себестоимости.",
    ),
    DataGap(
        metric="cash-flow / working capital",
        reason="Нет отдельного слоя оборотки, платежного календаря и кассового планирования.",
        impact="Нельзя честно оценивать cash gap, working capital pressure и liquidity planning.",
    ),
    DataGap(
        metric="SLA / fulfillment latency",
        reason="Нет полноценного слоя фактических SLA и времени исполнения доставки/фулфилмента.",
        impact="Нельзя подтверждать риск SLA и fulfillment latency beyond stock and supply signals.",
    ),
    DataGap(
        metric="retention / CRM / new-vs-repeat",
        reason="Нет когорт, CRM-слоя и признака repeat/new customer behavior.",
        impact="Нельзя подтверждать CAC, ROMI, MER, retention economics и reactivation effects.",
    ),
]


@dataclass
class EcomRunContext:
    user_query: str
    intent: QueryIntent
    today: date
    db_target: str
    chat_id: str
    run_id: int | None = None
    workflow_version: WorkflowVersion = WorkflowVersion.V1
    round_type: RoundType = RoundType.PRIMARY
    query_mode: QueryMode = QueryMode.REPORT
    memory_mode: MemoryMode = MemoryMode.FULL
    chat_summary: str | None = None
    recent_messages: list["ChatMemoryMessage"] = field(default_factory=list)
    evidence_budget: int = 5
    data_gaps: list[DataGap] = field(
        default_factory=lambda: [gap.model_copy(deep=True) for gap in GLOBAL_DATA_GAPS]
    )
    agent_trace: list[str] = field(default_factory=list)
    tool_trace: list[str] = field(default_factory=list)
    cross_agent_trace: list[str] = field(default_factory=list)
    decision_trace: list[dict] = field(default_factory=list)
    last_tool_report: dict[str, Any] | None = None
    last_prompt_report: dict[str, Any] | None = None
    query_plan: dict[str, Any] = field(default_factory=dict)

    def add_decision_trace(
        self,
        event: str,
        *,
        agent: str | None = None,
        tool: str | None = None,
        round: str | None = None,
        target_agent: str | None = None,
        question: str | None = None,
        status: str = "ok",
        duration_ms: int | None = None,
        details: dict | None = None,
    ) -> None:
        self.decision_trace.append(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "event": event,
                "agent": agent,
                "tool": tool,
                "round": round,
                "target_agent": target_agent,
                "question": question,
                "status": status,
                "duration_ms": duration_ms,
                "details": details or {},
            }
        )

    def as_chat_context_window(self) -> "ChatContextWindow":
        return ChatContextWindow(
            chat_id=self.chat_id,
            summary_text=self.chat_summary,
            recent_messages=list(self.recent_messages),
        )


@dataclass(frozen=True)
class ChatMemoryMessage:
    seq: int
    role: ChatRole
    content: str
    created_at: datetime | None = None


@dataclass(frozen=True)
class ChatSummaryRecord:
    summary_text: str
    summarized_through_seq: int
    updated_at: datetime | None = None


@dataclass(frozen=True)
class ChatContextWindow:
    chat_id: str
    summary_text: str | None
    recent_messages: list[ChatMemoryMessage]


AgentDiagnosticResult.model_rebuild()
