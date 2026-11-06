from __future__ import annotations

import json
import logging
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from agents import (
    Agent,
    GuardrailFunctionOutput,
    ModelSettings,
    RunHooks,
    RunResult,
    RunResultStreaming,
    output_guardrail,
)
from pydantic import BaseModel

from agi_ecom_app.config import get_settings  # kept for existing test patch points
from agi_ecom_app.llm_runtime import LLMRuntimeSpec, build_agent_model, get_active_llm_runtime_spec
from agi_ecom_app.prompt_pipeline import (
    domain_base_contract,
    memory_mode_for_round,
    resolve_workflow_version,
    round_contract,
)
from agi_ecom_app.rendering import (
    render_action_execution_result,
    render_agent_diagnostic,
    render_sales_diagnostic,
)
from agi_ecom_app.schemas import (
    ActionExecutionResult,
    AgentDiagnosticResult,
    ClaimVerificationResult,
    ChatContextWindow,
    CrossCheckResult,
    DataGap,
    DiagnosticFinding,
    DriverClass,
    EcomRunContext,
    EvidenceFact,
    FollowupResult,
    GapResolutionResult,
    MarketingDiagnosticResult,
    MemoryMode,
    QueryIntent,
    QueryMode,
    ReportTable,
    RoundType,
    SalesDiagnosticResult,
    WorkflowVersion,
)
from agi_ecom_app.services import strategic_db
from agi_ecom_app.services.memory import build_memory_context_block
from agi_ecom_app.tool_cache import trace_cached_tools
from agi_ecom_app.tools.db_tools import (
    get_finance_marketplace_report,
    get_finance_marketplace_report_text,
    get_finance_sku_report,
    get_finance_sku_report_text,
    get_finance_sku_profit_change_report,
    get_finance_sku_profit_change_report_text,
    get_database_time,
)
from agi_ecom_app.tools.role_db_tools import (
    FINANCE_SCHEMAS,
    MARKETING_SCHEMAS,
    PRICE_SCHEMAS,
    SALES_SCHEMAS,
    SUPPLY_SCHEMAS,
    build_role_db_tools,
)
from agi_ecom_app.tools.strategic_tools import (
    get_finance_formula_catalog,
    get_finance_overview,
    get_marketing_formula_catalog,
    get_marketing_action_matrix,
    get_marketing_campaign_period_report,
    get_marketing_bid_pressure_snapshot,
    get_marketing_campaign_efficiency_snapshot,
    get_marketing_deep_diagnostic,
    get_marketing_drr_report,
    get_marketing_overview,
    get_marketing_query_performance_snapshot,
    get_marketing_report_snapshot,
    get_marketing_search_demand_snapshot,
    get_marketing_sku_period_comparison_report,
    get_marketing_table_catalog,
    get_marketing_visibility_snapshot,
    get_price_formula_catalog,
    get_current_inventory_value,
    get_inventory_turnover,
    get_price_competitor_details,
    get_price_competitor_overview,
    get_price_context,
    get_price_decision_overview,
    get_price_table_catalog,
    get_sales_formula_catalog,
    get_sales_overview,
    get_sales_report_snapshot,
    get_sales_sku_revenue_change_report,
    get_sku_daily_price_sales_comparison,
    get_sales_table_catalog,
    get_supply_formula_catalog,
    get_supply_overview,
    get_supply_table_catalog,
)

logger = logging.getLogger(__name__)

PROMPT_ROOT = Path(__file__).resolve().parent / "prompts"
SALES_PROMPT_PATH = PROMPT_ROOT / "sales_agent.md"
MARKETING_PROMPT_PATH = PROMPT_ROOT / "marketing_agent.md"
SUPPLY_PROMPT_PATH = PROMPT_ROOT / "supply_agent.md"
FINANCE_PROMPT_PATH = PROMPT_ROOT / "finance_agent.md"
PRICE_PROMPT_PATH = PROMPT_ROOT / "price_agent.md"
DOMAIN_PROMPT_PATHS = frozenset(
    {
        SALES_PROMPT_PATH,
        MARKETING_PROMPT_PATH,
        SUPPLY_PROMPT_PATH,
        FINANCE_PROMPT_PATH,
        PRICE_PROMPT_PATH,
    }
)
DOMAIN_BY_PROMPT_PATH: dict[Path, QueryIntent] = {
    SALES_PROMPT_PATH: QueryIntent.SALES,
    MARKETING_PROMPT_PATH: QueryIntent.MARKETING,
    SUPPLY_PROMPT_PATH: QueryIntent.SUPPLY,
    FINANCE_PROMPT_PATH: QueryIntent.FINANCE,
    PRICE_PROMPT_PATH: QueryIntent.PRICE,
}
PROMPT_PATH_BY_DOMAIN: dict[QueryIntent, Path] = {
    domain: path for path, domain in DOMAIN_BY_PROMPT_PATH.items()
}

DERIVED_FEATURE_PROTOCOL = """
Общий протокол производных признаков:
- Для аналитических запросов работай циклом: гипотеза -> производный признак -> расчет по данным -> статус -> рекомендация.
- Для диагностических вопросов сам выбери 3-5 релевантных признаков. Formula catalog используй как справочник готовых идей, а не как список разрешённых мыслей.
- Сначала определи метод проверки и нужные данные; затем выбери самый короткий надежный путь: готовый reader, scoped read-only SQL или уже переданный evidence/context.
- Ты можешь создавать временные производные признаки внутри read-only SQL через CTE, CASE, lag/lead, rolling windows, ratios и классификации.
- Не сохраняй признаки в БД и не предлагай CREATE/INSERT/UPDATE/DELETE; признаки существуют только внутри текущего расчета.
- Для каждого важного признака укажи, что он проверяет, формулу, источник данных, grain, период, результат и статус: `подтверждено`, `частично подтверждено`, `не подтверждено` или `нет данных`.
- Не называй производный признак причиной сам по себе. Причина появляется только если признак подтверждает гипотезу и не опровергнут смежными данными.
- Если гипотеза зависит от другого домена, создай узкий `cross_agent_requests`, а не объявляй вывод доказанным.
""".strip()

COMMON_DOMAIN_WORKFLOW_RULES = """
COMMON_DOMAIN_WORKFLOW_RULES:
- Ты не вызываешь других агентов напрямую.
- Для смежных проверок заполняй `cross_agent_requests`; runner сам запустит следующий раунд.
- Не подтверждай факты чужого домена самостоятельно.
- Не показывай пользователю служебные запросы, имена агентов, JSON или Pydantic-структуры.
- Пользователю показывай SKU/артикул; `nm_id` используй только во внутренних полях.
- Не придумывай причинность там, где есть только связь.
- `data_gaps` используй только после попытки закрыть пробел доступными данными, read-only SQL или смежным доменом.
- Выводы отделяй от рекомендаций; рекомендации формулируй коротко и операционно.
""".strip()

FORBIDDEN_METRIC_PATTERNS: tuple[tuple[str, str], ...] = (
    ("cac", r"\bcac\b"),
    ("romi", r"\bromi\b"),
    ("mer", r"\bmer\b"),
    ("retention", r"\bretention\b"),
    ("new-vs-repeat", r"\bnew\s*-\s*vs\s*-\s*repeat\b"),
    ("cash-flow", r"\bcash[\s-]*flow\b"),
    ("working capital", r"\bworking\s+capital\b"),
    ("sla", r"\bsla\b"),
    ("contribution after cogs", r"\bcontribution\s+after\s+cogs\b"),
    ("profitability after procurement cost", r"\bprofitability\s+after\s+procurement\s+cost\b"),
)
MISSING_METRIC_CONTEXT_MARKERS: tuple[str, ...] = (
    "known_data_gap",
    "data_gap",
    "no ",
    "without ",
    "cannot ",
    "can't ",
    "нет ",
    "без ",
    "отсутств",
    "недоступ",
    "не подтверж",
    "не рассчит",
)
NM_ID_TOKEN_PATTERN = re.compile(r"\b\d{7,12}\b")
CONTEXTUAL_NM_ID_PATTERN = re.compile(
    r"(?:(?:sku|артикул|nm[_\s-]?id)\s*[:#№-]?\s*)?(\b\d{7,12}\b)(?=\s*:)|"
    r"(?:sku|артикул|nm[_\s-]?id)\s*[:#№-]?\s*(\b\d{7,12}\b)",
    re.IGNORECASE,
)
USER_FACING_DOMAIN_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bprice\b", re.IGNORECASE), "цену"),
    (re.compile(r"\bmarketing\b", re.IGNORECASE), "рекламу"),
    (re.compile(r"\bsupply\b", re.IGNORECASE), "остатки"),
    (re.compile(r"\bfinance\b", re.IGNORECASE), "экономику"),
    (re.compile(r"\bsales\b", re.IGNORECASE), "продажи"),
)
MECHANICAL_CAUSE_PATTERN = re.compile(
    r"механик|механическ|снижен\w*\s+(?:продаж|реализац|выручк)|"
    r"паден\w*\s+(?:продаж|реализац|выручк)|спад\w*\s+(?:фактическ\w*\s+)?(?:продаж|реализац|выручк)|"
    r"сжати\w*\s+продаж|"
    r"рост\w*\s+внутри\s+негативн\w*\s+тренд|рост\w*\s+отдельн\w*\s+sku|"
    r"краткосрочн\w*\s+отскок|компенсационн\w*\s+эффект|локальн\w*\s+улучшени",
    re.IGNORECASE,
)

def _load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _build_context_header(context: EcomRunContext) -> str:
    gap_lines = "\n".join(f"- {gap.metric}: {gap.reason}" for gap in context.data_gaps)
    return (
        f"Runtime context:\n"
        f"- today: {context.today.isoformat()}\n"
        f"- intent: {context.intent.value}\n"
        f"- db_target: {context.db_target}\n"
        f"- chat_id: {context.chat_id}\n"
        f"- evidence_budget: {context.evidence_budget}\n"
        f"- user_query: {context.user_query}\n"
        f"- known_data_gaps:\n{gap_lines}"
    )


def _enum_value(value: Any, default: str) -> str:
    return str(getattr(value, "value", value) or default)


def _v2_memory_block(context: EcomRunContext, mode: MemoryMode) -> str:
    if mode == MemoryMode.NONE:
        return ""
    if mode == MemoryMode.SUMMARY_ONLY:
        summary = (context.chat_summary or "").strip() or "No prior summary."
        return f"Chat memory summary:\n{summary}"
    return build_memory_context_block(
        ChatContextWindow(
            chat_id=context.chat_id,
            summary_text=context.chat_summary,
            recent_messages=list(context.recent_messages[-12:]),
        )
    )


def _build_v2_context_header(context: EcomRunContext, mode: MemoryMode) -> str:
    lines = [
        "Runtime context:",
        f"- today: {context.today.isoformat()}",
        f"- intent: {context.intent.value}",
        f"- round_type: {_enum_value(context.round_type, RoundType.PRIMARY.value)}",
        f"- query_mode: {_enum_value(context.query_mode, QueryMode.REPORT.value)}",
        f"- user_query: {context.user_query}",
    ]
    if context.round_type == RoundType.PRIMARY and context.data_gaps:
        lines.append("- known_data_gaps:")
        lines.extend(f"  - {gap.metric}: {gap.reason}" for gap in context.data_gaps)
    memory_block = _v2_memory_block(context, mode)
    if memory_block:
        lines.extend(["", memory_block])
    return "\n".join(lines)


def _dynamic_instructions(prompt_path: Path):
    legacy_base_prompt = _load_prompt(prompt_path)
    if prompt_path in DOMAIN_PROMPT_PATHS:
        legacy_base_prompt = (
            f"{COMMON_DOMAIN_WORKFLOW_RULES}\n\n"
            f"{legacy_base_prompt}\n\n{DERIVED_FEATURE_PROTOCOL}"
        )

    def _instructions(context_wrapper, agent) -> str:
        context = context_wrapper.context
        if isinstance(context, EcomRunContext):
            workflow_version = resolve_workflow_version(context.workflow_version)
            if workflow_version != WorkflowVersion.V1 and prompt_path in DOMAIN_BY_PROMPT_PATH:
                round_type = RoundType(_enum_value(context.round_type, RoundType.PRIMARY.value))
                memory_mode = memory_mode_for_round(round_type)
                context.memory_mode = memory_mode
                parts = [
                    domain_base_contract(DOMAIN_BY_PROMPT_PATH[prompt_path].value),
                    round_contract(round_type),
                ]
                if round_type == RoundType.PRIMARY and context.query_mode == QueryMode.DIAGNOSTIC:
                    parts.append(DERIVED_FEATURE_PROTOCOL)
                parts.append(_build_v2_context_header(context, memory_mode))
                return "\n\n".join(part for part in parts if part).strip()
            return (
                f"{legacy_base_prompt}\n\n"
                f"{_build_context_header(context)}\n\n"
                f"{build_memory_context_block(context.as_chat_context_window())}"
            )
        return legacy_base_prompt

    return _instructions


def _extract_text_from_model(value: BaseModel) -> str:
    return value.model_dump_json(indent=2)


def _compact_table_payload(table: ReportTable, max_rows: int = 8) -> dict[str, Any]:
    return {
        "title": table.title,
        "columns": table.columns,
        "rows": table.rows[:max_rows],
    }


def _compact_finding_payload(finding: DiagnosticFinding) -> dict[str, Any]:
    return {
        "factor": finding.factor,
        "observation": finding.observation,
        "impact": finding.impact,
        "cause": finding.cause,
        "evidence": finding.evidence[:5],
        "confidence": finding.confidence,
        "driver_class": finding.driver_class.value,
    }


def _diagnostic_tool_payload(output: Any) -> dict[str, Any] | str:
    if isinstance(output, SalesDiagnosticResult):
        return {
            "agent_name": output.agent_name,
            "summary": output.summary,
            "facts": [fact.model_dump(mode="json") for fact in output.facts[:20]],
            "sales_report_snapshot": (
                _compact_table_payload(output.sales_report_snapshot)
                if output.sales_report_snapshot
                else None
            ),
            "tables": [_compact_table_payload(table) for table in output.as_tables()],
            "findings": [_compact_finding_payload(finding) for finding in output.findings[:6]],
            "recommendations": output.recommendations[:6],
            "data_gaps": [gap.model_dump(mode="json") for gap in output.data_gaps],
            "cross_agent_requests": [
                request.model_dump(mode="json") for request in output.cross_agent_requests
            ],
        }
    if isinstance(output, MarketingDiagnosticResult):
        return {
            "agent_name": output.agent_name,
            "summary": output.summary,
            "facts": [fact.model_dump(mode="json") for fact in output.facts[:20]],
            "marketing_report_snapshot": (
                _compact_table_payload(output.marketing_report_snapshot)
                if output.marketing_report_snapshot
                else None
            ),
            "tables": [_compact_table_payload(table) for table in output.tables[:3]],
            "findings": [_compact_finding_payload(finding) for finding in output.findings[:6]],
            "recommendations": output.recommendations[:6],
            "data_gaps": [gap.model_dump(mode="json") for gap in output.data_gaps],
            "cross_agent_requests": [
                request.model_dump(mode="json") for request in output.cross_agent_requests
            ],
        }
    if isinstance(output, AgentDiagnosticResult):
        return {
            "agent_name": output.agent_name,
            "summary": output.summary,
            "facts": [fact.model_dump(mode="json") for fact in output.facts[:20]],
            "tables": [_compact_table_payload(table) for table in output.tables[:3]],
            "findings": [_compact_finding_payload(finding) for finding in output.findings[:6]],
            "recommendations": output.recommendations[:6],
            "data_gaps": [gap.model_dump(mode="json") for gap in output.data_gaps],
            "cross_agent_requests": [
                request.model_dump(mode="json") for request in output.cross_agent_requests
            ],
        }
    if isinstance(output, BaseModel):
        return output.model_dump(mode="json")
    if isinstance(output, str):
        return output
    return json.loads(json.dumps(output, ensure_ascii=False, default=str))


async def _structured_output_extractor(run_result: RunResult | RunResultStreaming) -> str:
    output = run_result.final_output
    if isinstance(output, ActionExecutionResult):
        return render_action_execution_result(output)
    if isinstance(output, SalesDiagnosticResult):
        return render_sales_diagnostic(output)
    if isinstance(output, MarketingDiagnosticResult):
        return render_agent_diagnostic(output)
    if isinstance(output, AgentDiagnosticResult):
        return render_agent_diagnostic(output)
    if isinstance(output, str):
        return output
    if isinstance(output, BaseModel):
        return _extract_text_from_model(output)
    return json.dumps(output, ensure_ascii=False, indent=2, default=str)


async def _orchestrator_tool_output_extractor(run_result: RunResult | RunResultStreaming) -> str:
    payload = _diagnostic_tool_payload(run_result.final_output)
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def _table_violations(table: ReportTable, label: str) -> list[str]:
    violations: list[str] = []
    if not table.title.strip():
        violations.append(f"{label} has empty title.")
    if not table.columns:
        violations.append(f"{label} has no columns.")
    for index, row in enumerate(table.rows, start=1):
        if len(row) != len(table.columns):
            violations.append(
                f"{label} row {index} length {len(row)} does not match columns {len(table.columns)}."
            )
    return violations


def _normalized_header(value: str) -> str:
    return value.strip().lower().replace("ё", "е")


def _is_empty_decomposition_value(value: str | int | float | None) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() in {"", "-", "—"}
    return False


def _find_column_index(columns: list[str], *patterns: tuple[str, ...]) -> int | None:
    normalized_columns = [_normalized_header(column) for column in columns]
    for pattern in patterns:
        for index, column in enumerate(normalized_columns):
            if all(token in column for token in pattern):
                return index
    return None


def _sales_decomposition_violations(table: ReportTable, label: str) -> list[str]:
    required_columns = {
        "current_revenue": (("current_revenue",), ("текущ", "выруч")),
        "previous_revenue": (("previous_revenue",), ("предыдущ", "выруч"), ("прошл", "выруч")),
        "delta_revenue": (
            ("delta_revenue",),
            ("delta", "выруч"),
            ("дельт", "выруч"),
            ("измен", "выруч"),
            ("δ", "выруч"),
        ),
        "current_sales": (("current_sales",), ("текущ", "продаж"), ("текущ", "шт")),
        "previous_sales": (
            ("previous_sales",),
            ("предыдущ", "продаж"),
            ("предыдущ", "шт"),
            ("прошл", "продаж"),
            ("прошл", "шт"),
        ),
        "delta_sales": (
            ("delta_sales",),
            ("delta", "продаж"),
            ("delta", "шт"),
            ("дельт", "продаж"),
            ("дельт", "шт"),
            ("измен", "продаж"),
            ("измен", "шт"),
            ("δ", "продаж"),
        ),
    }
    violations: list[str] = []
    column_indexes: dict[str, int] = {}

    for column_name, patterns in required_columns.items():
        index = _find_column_index(table.columns, *patterns)
        if index is None:
            violations.append(f"{label} is missing required decomposition column `{column_name}`.")
            continue
        column_indexes[column_name] = index

    for row_index, row in enumerate(table.rows, start=1):
        for column_name, column_index in column_indexes.items():
            if column_index >= len(row) or _is_empty_decomposition_value(row[column_index]):
                violations.append(
                    f"{label} row {row_index} has empty `{column_name}`; use the deterministic sales decomposition tool."
                )

    return violations


def _finding_violations(findings: list[DiagnosticFinding], prefix: str) -> list[str]:
    violations: list[str] = []
    for index, finding in enumerate(findings, start=1):
        finding_fields = [
            finding.factor,
            finding.observation,
            finding.impact,
            finding.cause,
            finding.action,
            *finding.evidence,
        ]
        for label, pattern in FORBIDDEN_METRIC_PATTERNS:
            unsupported_claim = any(
                any(
                    not _unsupported_metric_mention_is_limitation(
                        str(field).lower(),
                        match,
                    )
                    for match in re.finditer(pattern, str(field).lower())
                )
                for field in finding_fields
            )
            if unsupported_claim:
                violations.append(f"{prefix} finding {index} references unsupported metric: {label}")
        if not finding.evidence:
            violations.append(f"{prefix} finding {index} has no evidence.")
        if finding.confidence.strip().lower() not in {"high", "medium", "low"}:
            violations.append(
                f"{prefix} finding {index} has invalid confidence `{finding.confidence}`."
            )
    return violations


def _unsupported_metric_mention_is_limitation(
    text: str,
    match: re.Match[str],
) -> bool:
    """Return true only when the matched metric itself is explicitly unavailable.

    Missing-data text elsewhere in the same field must not legitimise a positive
    metric claim (for example, ``working capital improved, but no source exists``).
    """

    clause_start = max(
        [
            text.rfind(separator, 0, match.start())
            for separator in (".", ";", "!", "?", "\n")
        ],
        default=-1,
    )
    before = text[clause_start + 1 : match.start()]
    before = re.split(
        r"\b(?:but|however|whereas|while|но|однако|тогда как|при этом)\b",
        before,
    )[-1]

    # A missing-data qualifier before a metric can govern a short list, e.g.
    # ``no working capital / cash-flow layer``.
    if any(marker in before[-120:] for marker in MISSING_METRIC_CONTEXT_MARKERS):
        return True
    if re.search(
        r"(?:\bmissing\b|\bunavailable\b|\bunknown\b|\black(?:s|ing)?\b)"
        r"[^.;!?\n]{0,80}$",
        before,
    ):
        return True

    after = text[match.end() :]
    after = re.split(
        r"[.;!?\n]|\b(?:but|however|whereas|while|но|однако|тогда как|при этом)\b",
        after,
        maxsplit=1,
    )[0]
    # A qualifier after the metric is accepted only when it is the immediate
    # predicate of that metric. A later ``no source`` cannot mask ``improved``.
    return bool(
        re.match(
            r"^\s*(?:(?:metric|data|layer|source|value|calculation|данн\w*|"
            r"сло\w*|источник\w*|значени\w*|расч[её]т\w*)\s+){0,3}"
            r"(?:(?:is|are|was|were|remains?|оказал\w*|являет\w*)\s+)?"
            r"(?:not\s+(?:available|known|confirmed|calculated|provided)|"
            r"unavailable|unknown|missing|cannot\s+be\s+(?:calculated|confirmed)|"
            r"can't\s+be\s+(?:calculated|confirmed)|нет\b|без\b|отсутств\w*|"
            r"недоступ\w*|не\s+(?:подтвержд\w*|рассчит\w*|извест\w*|доступ\w*))",
            after,
        )
    )


def _text_has_nm_id(value: Any) -> bool:
    return bool(CONTEXTUAL_NM_ID_PATTERN.search(str(value or "")))


def _replace_nm_id_tokens(
    value: Any,
    vendor_code_by_nm_id: dict[str, str],
    *,
    force: bool = False,
) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    text = str(value)

    def _replacement(match: re.Match[str]) -> str:
        nm_id = match.group(0)
        if nm_id not in vendor_code_by_nm_id and not force:
            return nm_id
        return vendor_code_by_nm_id.get(nm_id, "SKU не сопоставлен")

    replaced = NM_ID_TOKEN_PATTERN.sub(_replacement, text)
    return replaced


def _replace_contextual_nm_ids(value: Any, vendor_code_by_nm_id: dict[str, str]) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    text = str(value)

    def _replacement(match: re.Match[str]) -> str:
        nm_id = next(group for group in match.groups() if group)
        vendor_code = vendor_code_by_nm_id.get(nm_id, "SKU не сопоставлен")
        return match.group(0).replace(nm_id, vendor_code)

    contextual_replaced = CONTEXTUAL_NM_ID_PATTERN.sub(_replacement, text)
    return _replace_nm_id_tokens(contextual_replaced, vendor_code_by_nm_id)


def _is_user_identifier_column(column: str) -> bool:
    normalized = _normalized_header(column)
    return (
        "sku" in normalized
        or "артикул" in normalized
        or "nm_id" in normalized
        or "nm id" in normalized
    )


def _replace_table_nm_ids(table: ReportTable, vendor_code_by_nm_id: dict[str, str]) -> None:
    identifier_indexes = {
        index
        for index, column in enumerate(table.columns)
        if _is_user_identifier_column(column)
    }
    table.rows = [
        [
            _replace_nm_id_tokens(cell, vendor_code_by_nm_id, force=True)
            if index in identifier_indexes
            else cell
            for index, cell in enumerate(row)
        ]
        for row in table.rows
    ]


def _replace_user_domain_names(text: str) -> str:
    updated = text
    for pattern, replacement in USER_FACING_DOMAIN_REPLACEMENTS:
        updated = pattern.sub(replacement, updated)
    return updated


def _clean_duplicate_sku_aliases(text: str) -> str:
    return re.sub(r"\b(\d{4,6})/\1\b", r"\1", text)


def _normalize_sales_finding(finding: DiagnosticFinding, vendor_code_by_nm_id: dict[str, str]) -> None:
    finding.factor = _clean_duplicate_sku_aliases(
        _replace_user_domain_names(_replace_contextual_nm_ids(finding.factor, vendor_code_by_nm_id))
    )
    finding.observation = _clean_duplicate_sku_aliases(
        _replace_user_domain_names(
            _replace_contextual_nm_ids(finding.observation, vendor_code_by_nm_id)
        )
    )
    finding.impact = _clean_duplicate_sku_aliases(
        _replace_user_domain_names(_replace_contextual_nm_ids(finding.impact, vendor_code_by_nm_id))
    )
    finding.action = _clean_duplicate_sku_aliases(
        _replace_user_domain_names(_replace_contextual_nm_ids(finding.action, vendor_code_by_nm_id))
    )
    finding.evidence = [
        _clean_duplicate_sku_aliases(
            _replace_user_domain_names(_replace_contextual_nm_ids(evidence, vendor_code_by_nm_id))
        )
        for evidence in finding.evidence
    ]
    if MECHANICAL_CAUSE_PATTERN.search(finding.cause):
        finding.cause = (
            "Sales-данные подтверждают изменение метрик; первопричина требует смежной проверки."
        )
    else:
        finding.cause = _clean_duplicate_sku_aliases(
            _replace_user_domain_names(_replace_contextual_nm_ids(finding.cause, vendor_code_by_nm_id))
        )


def _collect_sales_user_facing_nm_ids(result: SalesDiagnosticResult) -> list[str]:
    values: list[Any] = []
    for table in [
        result.sales_report_snapshot,
        result.sales_overview,
        result.sales_dynamics,
        result.sales_decomposition,
    ]:
        if table:
            identifier_indexes = {
                index
                for index, column in enumerate(table.columns)
                if _is_user_identifier_column(column)
            }
            values.extend(
                cell
                for row in table.rows
                for index, cell in enumerate(row)
                if index in identifier_indexes
            )
    for finding in result.findings:
        values.extend(
            [
                finding.factor,
                finding.observation,
                finding.impact,
                finding.cause,
                finding.action,
                *finding.evidence,
            ]
        )
    values.extend(result.recommendations)
    nm_ids: list[str] = []
    for value in values:
        text = str(value or "")
        contextual_matches = [
            next(group for group in match if group)
            for match in CONTEXTUAL_NM_ID_PATTERN.findall(text)
            if any(match)
        ]
        token_matches = NM_ID_TOKEN_PATTERN.findall(text)
        for token in [*contextual_matches, *token_matches]:
            if token not in nm_ids:
                nm_ids.append(token)
    return nm_ids


def enforce_sales_user_facing_contract(output: Any) -> Any:
    if not isinstance(output, SalesDiagnosticResult):
        return output

    nm_ids = _collect_sales_user_facing_nm_ids(output)
    vendor_code_by_nm_id = strategic_db.get_sales_vendor_code_map(nm_ids) if nm_ids else {}

    for table in [
        output.sales_report_snapshot,
        output.sales_overview,
        output.sales_dynamics,
        output.sales_decomposition,
    ]:
        if table:
            _replace_table_nm_ids(table, vendor_code_by_nm_id)
    for finding in output.findings:
        _normalize_sales_finding(finding, vendor_code_by_nm_id)
    output.recommendations = [
        _clean_duplicate_sku_aliases(
            _replace_user_domain_names(
                _replace_contextual_nm_ids(recommendation, vendor_code_by_nm_id)
            )
        )
        for recommendation in output.recommendations
    ]
    return output


def _build_sales_decomposition_table_from_tool() -> ReportTable | None:
    payload = strategic_db.get_sales_sku_revenue_change(limit=10)
    rows = payload.get("rows") or []
    if not rows:
        return None
    return ReportTable(
        title="Вклад артикулов в изменение выручки",
        columns=[
            "SKU/артикул",
            "Бренд",
            "Категория",
            "Текущая выручка",
            "Предыдущая выручка",
            "Δ выручки",
            "Текущие продажи",
            "Предыдущие продажи",
            "Δ продаж",
            "Роль",
        ],
        rows=[
            [
                row.get("vendor_code") or "SKU не сопоставлен",
                row.get("brand_name"),
                row.get("subject_name"),
                row.get("current_revenue"),
                row.get("previous_revenue"),
                row.get("delta_revenue"),
                row.get("current_sales"),
                row.get("previous_sales"),
                row.get("delta_sales"),
                row.get("driver_role"),
            ]
            for row in rows
        ],
    )


def ensure_sales_decomposition_contract(output: Any) -> Any:
    if not isinstance(output, SalesDiagnosticResult):
        return output
    if not _sales_decomposition_violations(
        output.sales_decomposition,
        f"{output.agent_name} sales_decomposition",
    ):
        return output
    table = _build_sales_decomposition_table_from_tool()
    if table:
        output.sales_decomposition = table
    return output


def _sales_user_facing_contract_violations(result: SalesDiagnosticResult) -> list[str]:
    violations: list[str] = []
    user_text_parts: list[str] = []
    for table in [
        result.sales_report_snapshot,
        result.sales_overview,
        result.sales_dynamics,
        result.sales_decomposition,
    ]:
        if table:
            for index, column in enumerate(table.columns):
                if _is_user_identifier_column(column):
                    user_text_parts.extend(
                        str(row[index])
                        for row in table.rows
                        if index < len(row)
                    )
    for finding in result.findings:
        user_text_parts.extend(
            [
                finding.factor,
                finding.observation,
                finding.impact,
                finding.cause,
                finding.action,
            ]
        )
        if MECHANICAL_CAUSE_PATTERN.search(finding.cause):
            violations.append(f"{result.agent_name} finding cause treats mechanics as cause.")
    user_text_parts.extend(result.recommendations)
    user_text = " ".join(user_text_parts).lower()
    if _text_has_nm_id(user_text):
        violations.append(f"{result.agent_name} exposes nm_id in user-facing fields.")
    for domain_name in ("price", "marketing", "supply", "finance"):
        if re.search(rf"\b{domain_name}\b", user_text, re.IGNORECASE):
            violations.append(
                f"{result.agent_name} exposes internal domain name `{domain_name}` in user-facing fields."
            )
    return violations


def validate_agent_diagnostic_result(result: AgentDiagnosticResult) -> list[str]:
    violations: list[str] = []
    if not result.tables:
        violations.append(f"{result.agent_name} returned no tables.")
    for index, table in enumerate(result.tables, start=1):
        violations.extend(_table_violations(table, f"{result.agent_name} table {index}"))
    violations.extend(_finding_violations(result.findings, result.agent_name))
    return violations


def _is_sales_report_request(user_query: str | None) -> bool:
    text = (user_query or "").lower().replace("ё", "е")
    if not text:
        return False
    has_report_word = bool(re.search(r"\bотчет\w*\b|\bсводк\w*\b|\breport\b|\bsnapshot\b", text))
    has_sales_word = bool(re.search(r"\bпродаж\w*\b|\bзаказ\w*\b|\bвыручк\w*\b|\bsales\b", text))
    has_sales_comparison = bool(
        re.search(
            r"вчера\s+к\s+позавчера|недел\w*\s+к\s+недел\w*|месяц\w*\s+к\s+месяц\w*|"
            r"\bweek\s+over\s+week\b|\bmonth\s+over\s+month\b|\bwow\b|\bmom\b",
            text,
        )
    )
    return has_report_word and (has_sales_word or has_sales_comparison)


def _is_marketing_report_request(user_query: str | None) -> bool:
    text = (user_query or "").lower().replace("ё", "е")
    if not text:
        return False
    has_report_word = bool(re.search(r"\bотчет\w*\b|\bсводк\w*\b|\breport\b|\bsnapshot\b", text))
    has_marketing_word = bool(
        re.search(
            r"\bмаркетинг\w*\b|\bреклам\w*\b|\bкампани\w*\b|\bдрр\b|\bdrr\b|"
            r"\bcpc\b|\bклик\w*\b|\bпоказ\w*\b|\bрасход\w*\b|\bspend\b",
            text,
        )
    )
    has_comparison = bool(
        re.search(
            r"вчера\s+к\s+позавчера|недел\w*\s+к\s+недел\w*|месяц\w*\s+к\s+месяц\w*|"
            r"\bweek\s+over\s+week\b|\bmonth\s+over\s+month\b|\bwow\b|\bmom\b",
            text,
        )
    )
    return (has_report_word and (has_marketing_word or has_comparison)) or (
        has_marketing_word and has_comparison
    )


def _supply_current_stock_lookup_key(
    query_plan: dict[str, Any] | None,
) -> str | None:
    """Read a one-SKU target only from the semantic LLM routing contract."""

    plan = dict(query_plan or {})
    if str(plan.get("scope") or "") != "single_sku":
        return None
    targets = [target for target in list(plan.get("targets") or []) if isinstance(target, dict)]
    if len(targets) != 1:
        return None
    target = targets[0]
    if str(target.get("target_type") or "") not in {"vendor_code", "nm_id"}:
        return None
    return str(target.get("value") or "").strip() or None


def _supply_lookup_table(lookup: dict[str, Any], requested_sku: str) -> ReportTable:
    columns = [
        "snapshot_date",
        "SKU/артикул",
        "nm_id",
        "available_stock_qty",
        "full_stock_qty",
        "status",
    ]
    matches = list(lookup.get("matches") or [])
    if lookup.get("status") == "found":
        matches = [lookup]
    rows = [
        [
            (
                str(match.get("snapshot_date"))
                if match.get("snapshot_date") is not None
                else None
            ),
            match.get("vendor_code") or requested_sku,
            match.get("nm_id"),
            match.get("available_stock_qty"),
            match.get("full_stock_qty"),
            lookup.get("status"),
        ]
        for match in matches
    ]
    if not rows:
        rows = [[None, requested_sku, None, None, None, "no_data"]]
    return ReportTable(
        title="Агрегированный текущий доступный остаток",
        columns=columns,
        rows=rows,
    )


def ensure_supply_current_stock_lookup(
    output: Any,
    user_query: str | None = None,
    query_plan: dict[str, Any] | None = None,
) -> Any:
    """Replace a one-SKU stock lookup with the deterministic aggregated fact."""

    if not isinstance(output, AgentDiagnosticResult):
        return output
    del user_query
    requested_sku = _supply_current_stock_lookup_key(query_plan)
    if not requested_sku:
        return output
    if any(
        fact.evidence_ref == "get_supply_overview.sku_stock_lookup"
        for fact in output.facts
    ):
        return output

    overview = strategic_db.get_supply_overview(product_keyword=requested_sku, limit=10)
    lookup = dict(overview.get("sku_stock_lookup") or {})
    status = str(lookup.get("status") or "no_data")
    output.tables = [_supply_lookup_table(lookup, requested_sku)]
    output.findings = []
    output.recommendations = []
    output.cross_agent_requests = []

    if status == "found":
        available_stock = lookup.get("available_stock_qty")
        snapshot_date = lookup.get("snapshot_date")
        vendor_code = str(lookup.get("vendor_code") or requested_sku)
        output.summary = (
            f"На актуальном срезе {snapshot_date} агрегированный доступный остаток "
            f"по SKU {vendor_code} — {available_stock} шт. Значение рассчитано как "
            "сумма quantity по всем складским строкам SKU, а не как максимум одной строки."
        )
        output.facts = [
            EvidenceFact(
                metric="current_available_stock",
                value=available_stock,
                unit="pcs",
                period=str(snapshot_date),
                sku=vendor_code,
                target_type="nm_id",
                target_id=(
                    str(lookup.get("nm_id")) if lookup.get("nm_id") is not None else None
                ),
                source_table="supply.wb_supplier_stocks",
                source_snapshot_at=snapshot_date,
                metric_definition=(
                    "sum(quantity) across all warehouse rows for the SKU on the latest snapshot"
                ),
                statement=(
                    f"Агрегированный доступный остаток SKU {vendor_code}: "
                    f"{available_stock} шт."
                ),
                evidence_ref="get_supply_overview.sku_stock_lookup",
                source_agent="supply",
                status="confirmed",
                confidence="high",
            )
        ]
        output.data_gaps = []
        return output

    output.facts = []
    if status == "multiple_matches":
        output.summary = (
            f"Запрос {requested_sku} соответствует нескольким SKU. Единый текущий остаток "
            "не рассчитан: агрегированные строки показаны отдельно."
        )
        reason = "По ключу найдено несколько SKU; нужен точный артикул или nm_id."
    else:
        output.summary = (
            f"Для SKU {requested_sku} в последнем доступном снимке строка не найдена. "
            "Текущий доступный остаток не подтверждён и не считается равным нулю."
        )
        reason = "В последнем доступном снимке нет строки для запрошенного SKU."
    output.data_gaps = [
        DataGap(
            metric="current_available_stock",
            reason=reason,
            impact="Нельзя назвать текущий доступный остаток числом.",
        )
    ]
    return output


SUPPLY_AGGREGATED_STOCK_FACT_FIELDS: dict[str, tuple[str, str]] = {
    "stock_quantity": ("available_stock_qty", "available_stock_qty"),
    "current_stock_qty": ("available_stock_qty", "available_stock_qty"),
    "current_available_stock": ("available_stock_qty", "available_stock_qty"),
    "available_stock_qty": ("available_stock_qty", "available_stock_qty"),
    "full_stock_quantity": ("full_stock_qty", "full_stock_qty"),
    "full_stock_qty": ("full_stock_qty", "full_stock_qty"),
    "in_transit_to_client": ("in_way_to_client", "in_way_to_client"),
    "in_way_to_client": ("in_way_to_client", "in_way_to_client"),
    "in_transit_from_client": ("in_way_from_client", "in_way_from_client"),
    "in_way_from_client": ("in_way_from_client", "in_way_from_client"),
}
SUPPLY_STOCK_IDENTIFIER_COLUMNS = frozenset(
    {
        "sku",
        "sku_артикул",
        "артикул",
        "vendor_code",
        "vendorcode",
        "nm_id",
        "nmid",
    }
)
SUPPLY_GENERIC_STOCK_QUANTITY_COLUMNS = frozenset(
    {"quantity", "qty", "количество"}
)
SUPPLY_STOCK_TABLE_TITLE_PATTERN = re.compile(
    r"остат|склад|\bstock\b|\binventory\b|в\s+пути|in[\s_-]*transit",
    re.IGNORECASE,
)


def _supply_stock_metric_key(value: Any) -> str:
    return re.sub(r"[^a-zа-я0-9]+", "_", str(value or "").lower()).strip("_")


def _supply_stock_table_contract(
    table: ReportTable,
) -> tuple[int | None, dict[int, tuple[str, str]]] | None:
    column_keys = [_supply_stock_metric_key(column) for column in table.columns]
    identifier_index = next(
        (
            index
            for index, key in enumerate(column_keys)
            if key in SUPPLY_STOCK_IDENTIFIER_COLUMNS
        ),
        None,
    )
    stock_fields = {
        index: SUPPLY_AGGREGATED_STOCK_FACT_FIELDS[key]
        for index, key in enumerate(column_keys)
        if key in SUPPLY_AGGREGATED_STOCK_FACT_FIELDS
    }
    generic_quantity_indexes = [
        index
        for index, key in enumerate(column_keys)
        if key in SUPPLY_GENERIC_STOCK_QUANTITY_COLUMNS
    ]
    if identifier_index is not None and generic_quantity_indexes:
        for index in generic_quantity_indexes:
            stock_fields[index] = ("available_stock_qty", "available_stock_qty")
    elif SUPPLY_STOCK_TABLE_TITLE_PATTERN.search(table.title):
        for index in generic_quantity_indexes:
            stock_fields[index] = ("available_stock_qty", "available_stock_qty")
    if not stock_fields:
        return None
    return identifier_index, stock_fields


def _supply_fact_lookup_key(fact: Any) -> str:
    if isinstance(fact, dict):
        sku = fact.get("sku")
        target_type = fact.get("target_type")
        target_id = fact.get("target_id")
    else:
        sku = getattr(fact, "sku", None)
        target_type = getattr(fact, "target_type", None)
        target_id = getattr(fact, "target_id", None)
    if str(sku or "").strip():
        return str(sku).strip()
    if _supply_stock_metric_key(target_type) in SUPPLY_STOCK_IDENTIFIER_COLUMNS:
        return str(target_id or "").strip()
    return ""


def _canonical_supply_stock_fact(
    *,
    lookup: dict[str, Any],
    requested_sku: str,
    canonical_metric: str,
    lookup_field: str,
) -> EvidenceFact | None:
    value = lookup.get(lookup_field)
    snapshot_date = lookup.get("snapshot_date")
    if value is None or snapshot_date is None:
        return None
    vendor_code = str(lookup.get("vendor_code") or requested_sku)
    return EvidenceFact(
        metric=canonical_metric,
        value=value,
        unit="pcs",
        period=str(snapshot_date),
        sku=vendor_code,
        target_type="nm_id",
        target_id=(
            str(lookup.get("nm_id")) if lookup.get("nm_id") is not None else None
        ),
        source_table="supply.wb_supplier_stocks",
        source_snapshot_at=snapshot_date,
        metric_definition="aggregated latest SKU stock lookup across all warehouse rows",
        statement=f"SKU {vendor_code}: {canonical_metric}={value} на {snapshot_date}.",
        evidence_ref=f"get_supply_overview.sku_stock_lookup.{lookup_field}",
        source_agent="supply",
        status="confirmed",
        confidence="high",
    )


def ensure_supply_aggregated_stock_facts(
    output: Any,
    *,
    replace_summary: bool = False,
) -> Any:
    """Fail closed on raw current-stock evidence and replace it with SKU aggregates."""

    facts = getattr(output, "facts", None)
    source_facts = list(facts) if isinstance(facts, list) else []
    source_tables = list(getattr(output, "tables", []) or [])
    corrected_facts: list[Any] = []
    requested_fields_by_sku: dict[str, set[tuple[str, str]]] = {}
    unresolved_stock_fields: set[tuple[str, str]] = set()
    changed = False

    for fact in source_facts:
        metric = fact.get("metric") if isinstance(fact, dict) else getattr(fact, "metric", None)
        field_contract = SUPPLY_AGGREGATED_STOCK_FACT_FIELDS.get(
            _supply_stock_metric_key(metric)
        )
        if not field_contract:
            corrected_facts.append(fact)
            continue
        changed = True
        sku = _supply_fact_lookup_key(fact)
        if sku:
            requested_fields_by_sku.setdefault(sku, set()).add(field_contract)
        else:
            unresolved_stock_fields.add(field_contract)

    clean_tables: list[Any] = []
    for table in source_tables:
        if not isinstance(table, ReportTable):
            clean_tables.append(table)
            continue
        table_contract = _supply_stock_table_contract(table)
        if table_contract is None:
            clean_tables.append(table)
            continue
        changed = True
        identifier_index, stock_fields = table_contract
        table_had_sku = False
        if identifier_index is not None:
            for row in table.rows:
                if identifier_index >= len(row):
                    continue
                sku = str(row[identifier_index] or "").strip()
                if not sku:
                    continue
                table_had_sku = True
                requested_fields_by_sku.setdefault(sku, set()).update(
                    stock_fields.values()
                )
        if not table_had_sku:
            unresolved_stock_fields.update(stock_fields.values())

    # A pure turnover fact/table is not current-stock evidence and stays intact.
    if not changed:
        return output

    has_unresolved_stock = bool(unresolved_stock_fields)
    if has_unresolved_stock and len(requested_fields_by_sku) == 1:
        only_sku = next(iter(requested_fields_by_sku))
        requested_fields_by_sku[only_sku].update(unresolved_stock_fields)
        has_unresolved_stock = False

    corrected_rows: dict[str, list[Any]] = {}
    missing_reasons: dict[str, str] = {}
    canonical_fact_keys: set[tuple[str, str]] = set()
    simulated_snapshot_dates: set[str] = set()
    for sku in sorted(requested_fields_by_sku):
        try:
            overview = strategic_db.get_supply_overview(product_keyword=sku, limit=10)
            lookup = dict(overview.get("sku_stock_lookup") or {})
            simulation = dict(overview.get("simulation") or {})
        except Exception:
            logger.exception("supply_aggregated_stock_fact_lookup_failed sku=%s", sku)
            missing_reasons[sku] = (
                "Агрегированный sku_stock_lookup завершился ошибкой; "
                "исходное число остатка отброшено."
            )
            continue
        if str(lookup.get("status") or "no_data") != "found":
            missing_reasons[sku] = (
                "Агрегированный sku_stock_lookup не вернул строку; "
                "остаток не считается равным нулю."
            )
            continue

        simulated_current = bool(simulation.get("enabled")) and str(
            simulation.get("effective_status") or ""
        ) == "simulated_current"
        if simulated_current and simulation.get("source_snapshot_date") is not None:
            simulated_snapshot_dates.add(str(simulation["source_snapshot_date"]))

        vendor_code = str(lookup.get("vendor_code") or sku)
        corrected_for_sku = False
        for canonical_metric, lookup_field in sorted(requested_fields_by_sku[sku]):
            canonical_fact = _canonical_supply_stock_fact(
                lookup=lookup,
                requested_sku=sku,
                canonical_metric=canonical_metric,
                lookup_field=lookup_field,
            )
            if canonical_fact is None:
                missing_reasons[sku] = (
                    f"В агрегированном sku_stock_lookup нет поля {lookup_field} "
                    "или даты снимка; исходное число остатка отброшено."
                )
                continue
            fact_key = (canonical_fact.sku or sku, canonical_metric)
            if fact_key not in canonical_fact_keys:
                corrected_facts.append(canonical_fact)
                canonical_fact_keys.add(fact_key)
            corrected_for_sku = True
        if corrected_for_sku:
            corrected_rows[vendor_code] = [
                vendor_code,
                lookup.get("available_stock_qty"),
                lookup.get("full_stock_qty"),
                lookup.get("in_way_to_client"),
                lookup.get("in_way_from_client"),
                str(lookup.get("snapshot_date")),
                "simulated_current" if simulated_current else "found",
            ]

    if has_unresolved_stock:
        missing_reasons["unknown"] = (
            "В raw evidence нет SKU/nm_id для агрегированной проверки; "
            "исходное число остатка отброшено."
        )

    output.facts = corrected_facts
    data_gaps = list(getattr(output, "data_gaps", []) or [])
    known_gap_metrics = {str(gap.metric) for gap in data_gaps if isinstance(gap, DataGap)}
    for sku, reason in sorted(missing_reasons.items()):
        metric = f"current_available_stock:{sku}"
        if metric in known_gap_metrics:
            continue
        data_gaps.append(
            DataGap(
                metric=metric,
                reason=reason,
                impact="Нельзя использовать складскую строку как общий остаток SKU.",
            )
        )
    output.data_gaps = data_gaps

    if hasattr(output, "tables"):
        rows = list(corrected_rows.values())
        rows.extend(
            [sku, None, None, None, None, None, "no_data"]
            for sku in sorted(missing_reasons)
        )
        if rows:
            clean_tables.insert(
                0,
                ReportTable(
                    title="Агрегированный текущий остаток по SKU",
                    columns=[
                        "SKU",
                        "available_stock_qty",
                        "full_stock_qty",
                        "in_way_to_client",
                        "in_way_from_client",
                        "snapshot_date",
                        "status",
                    ],
                    rows=rows,
                ),
            )
        output.tables = clean_tables

    if replace_summary:
        confirmed_parts = [
            f"SKU {sku}: доступно {row[1]} шт., полный остаток {row[2]} шт."
            for sku, row in sorted(corrected_rows.items())
        ]
        missing_parts = [
            f"SKU {sku}: остаток не подтверждён и не равен нулю по умолчанию."
            for sku in sorted(missing_reasons)
        ]
        source_dates = ", ".join(sorted(simulated_snapshot_dates))
        intro = (
            "Эмуляция: последний известный складской снимок"
            f" от {source_dates} используется как текущий."
            if source_dates
            else "Текущий остаток перепроверен как сумма по всем складским строкам SKU."
        )
        output.summary = " ".join(
            [
                intro,
                *confirmed_parts,
                *missing_parts,
                (
                    "Эмулированный текущий срез сам по себе не доказывает причинность изменения продаж."
                    if source_dates
                    else "Один текущий срез сам по себе не доказывает причинность изменения продаж."
                ),
            ]
        ).strip()
    return output


def _sales_report_snapshot_violations(table: ReportTable | None) -> list[str]:
    if table is None:
        return ["Sales report request is missing `sales_report_snapshot`."]

    violations = _table_violations(table, "Sales Agent sales_report_snapshot")
    table_text = " ".join(
        [
            table.title,
            " ".join(table.columns),
            " ".join(str(cell) for row in table.rows for cell in row),
        ]
    ).lower()
    required_period_labels = (
        ("вчера к позавчера", "последний день в данных"),
        ("неделя к неделе", "последние 7 дней"),
        ("месяц к месяцу", "последние 30 дней"),
    )
    for labels in required_period_labels:
        if not any(label in table_text for label in labels):
            violations.append(
                f"Sales Agent sales_report_snapshot is missing `{' / '.join(labels)}`."
            )
    if "продаж" not in table_text:
        violations.append("Sales Agent sales_report_snapshot is missing sales units.")
    if "выруч" not in table_text:
        violations.append("Sales Agent sales_report_snapshot is missing revenue.")
    return violations


def _marketing_report_snapshot_violations(table: ReportTable | None) -> list[str]:
    if table is None:
        return ["Marketing report request is missing `marketing_report_snapshot`."]

    violations = _table_violations(table, "Marketing Agent marketing_report_snapshot")
    table_text = " ".join(
        [
            table.title,
            " ".join(table.columns),
            " ".join(str(cell) for row in table.rows for cell in row),
        ]
    ).lower()
    required_period_labels = (
        ("вчера к позавчера", "последний день в данных"),
        ("неделя к неделе", "последние 7 дней"),
        ("месяц к месяцу", "последние 30 дней"),
    )
    for labels in required_period_labels:
        if not any(label in table_text for label in labels):
            violations.append(
                f"Marketing Agent marketing_report_snapshot is missing `{' / '.join(labels)}`."
            )
    if "дрр" not in table_text:
        violations.append("Marketing Agent marketing_report_snapshot is missing ДРР.")
    if "расход" not in table_text:
        violations.append("Marketing Agent marketing_report_snapshot is missing ad spend.")
    if "выруч" not in table_text:
        violations.append("Marketing Agent marketing_report_snapshot is missing promoted SKU revenue.")
    return violations


def validate_sales_diagnostic_result(
    result: SalesDiagnosticResult,
    user_query: str | None = None,
) -> list[str]:
    violations: list[str] = []
    if result.sales_report_snapshot:
        if _is_sales_report_request(user_query):
            violations.extend(_sales_report_snapshot_violations(result.sales_report_snapshot))
        else:
            violations.extend(
                _table_violations(
                    result.sales_report_snapshot,
                    "Sales Agent sales_report_snapshot",
                )
            )
    elif _is_sales_report_request(user_query):
        violations.extend(_sales_report_snapshot_violations(None))
    tables = {
        "sales_overview": result.sales_overview,
        "sales_dynamics": result.sales_dynamics,
        "sales_decomposition": result.sales_decomposition,
    }
    for label, table in tables.items():
        violations.extend(_table_violations(table, f"{result.agent_name} {label}"))
    violations.extend(
        _sales_decomposition_violations(
            result.sales_decomposition,
            f"{result.agent_name} sales_decomposition",
        )
    )
    violations.extend(_finding_violations(result.findings, result.agent_name))
    violations.extend(_sales_user_facing_contract_violations(result))
    return violations


def validate_marketing_diagnostic_result(
    result: MarketingDiagnosticResult,
    user_query: str | None = None,
) -> list[str]:
    violations = validate_agent_diagnostic_result(result)
    if result.marketing_report_snapshot:
        if _is_marketing_report_request(user_query):
            violations.extend(_marketing_report_snapshot_violations(result.marketing_report_snapshot))
        else:
            violations.extend(
                _table_violations(
                    result.marketing_report_snapshot,
                    "Marketing Agent marketing_report_snapshot",
                )
            )
    elif _is_marketing_report_request(user_query):
        violations.extend(_marketing_report_snapshot_violations(None))
    return violations


def ensure_sales_report_snapshot(output: Any, user_query: str | None = None) -> Any:
    if not isinstance(output, SalesDiagnosticResult):
        return output

    if not output.sales_report_snapshot and _is_sales_report_request(user_query):
        snapshot = strategic_db.get_sales_report_snapshot()
        report_table = snapshot.get("report_table")
        if report_table:
            output.sales_report_snapshot = ReportTable.model_validate(report_table)
    ensure_sales_decomposition_contract(output)
    return enforce_sales_user_facing_contract(output)


def ensure_marketing_report_snapshot(output: Any, user_query: str | None = None) -> Any:
    if not isinstance(output, MarketingDiagnosticResult):
        return output

    if not output.marketing_report_snapshot and _is_marketing_report_request(user_query):
        snapshot = strategic_db.get_marketing_report_snapshot()
        report_table = snapshot.get("report_table")
        if report_table:
            output.marketing_report_snapshot = ReportTable.model_validate(report_table)
    cleaned_tables: list[ReportTable] = []
    dropped_rows = 0
    for table in output.tables:
        original_rows = list(table.rows)
        valid_rows = [row for row in original_rows if len(row) == len(table.columns)]
        dropped_rows += len(original_rows) - len(valid_rows)
        table.rows = valid_rows
        if valid_rows or not original_rows:
            cleaned_tables.append(table)
    output.tables = cleaned_tables
    if not output.tables and output.marketing_report_snapshot is not None:
        if not _table_violations(
            output.marketing_report_snapshot,
            "Marketing Agent marketing_report_snapshot",
        ):
            output.tables = [output.marketing_report_snapshot.model_copy(deep=True)]
    if dropped_rows and not any(
        gap.metric == "marketing_table_shape" for gap in output.data_gaps
    ):
        output.data_gaps.append(
            DataGap(
                metric="marketing_table_shape",
                reason=(
                    f"{dropped_rows} строк исключено: число ячеек не совпало с числом колонок."
                ),
                impact="Исключённые строки не используются как evidence.",
            )
        )
    return output


def ensure_domain_report_snapshots(
    output: Any,
    user_query: str | None = None,
    query_plan: dict[str, Any] | None = None,
) -> Any:
    output = ensure_sales_report_snapshot(output, user_query=user_query)
    output = ensure_marketing_report_snapshot(output, user_query=user_query)
    return ensure_supply_current_stock_lookup(
        output,
        user_query=user_query,
        query_plan=query_plan,
    )


@output_guardrail
def diagnostic_output_guardrail(context, agent, output: Any) -> GuardrailFunctionOutput:
    if not isinstance(output, AgentDiagnosticResult):
        return GuardrailFunctionOutput(
            output_info={"violations": ["Final output is not AgentDiagnosticResult."]},
            tripwire_triggered=True,
        )

    violations = validate_agent_diagnostic_result(output)
    return GuardrailFunctionOutput(
        output_info={"violations": violations},
        tripwire_triggered=bool(violations),
    )


@output_guardrail
def supply_output_guardrail(context, agent, output: Any) -> GuardrailFunctionOutput:
    if not isinstance(output, AgentDiagnosticResult):
        return GuardrailFunctionOutput(
            output_info={"violations": ["Final output is not AgentDiagnosticResult."]},
            tripwire_triggered=True,
        )

    run_context = getattr(context, "context", None)
    ensure_supply_current_stock_lookup(
        output,
        user_query=getattr(run_context, "user_query", None),
        query_plan=getattr(run_context, "query_plan", None),
    )
    violations = validate_agent_diagnostic_result(output)
    return GuardrailFunctionOutput(
        output_info={"violations": violations},
        tripwire_triggered=bool(violations),
    )


@output_guardrail
def marketing_output_guardrail(context, agent, output: Any) -> GuardrailFunctionOutput:
    if not isinstance(output, MarketingDiagnosticResult):
        return GuardrailFunctionOutput(
            output_info={"violations": ["Final output is not MarketingDiagnosticResult."]},
            tripwire_triggered=True,
        )

    run_context = getattr(context, "context", None)
    ensure_marketing_report_snapshot(
        output,
        user_query=getattr(run_context, "user_query", None),
    )
    violations = validate_marketing_diagnostic_result(
        output,
        user_query=getattr(run_context, "user_query", None),
    )
    return GuardrailFunctionOutput(
        output_info={"violations": violations},
        tripwire_triggered=bool(violations),
    )


@output_guardrail
def sales_output_guardrail(context, agent, output: Any) -> GuardrailFunctionOutput:
    if not isinstance(output, SalesDiagnosticResult):
        return GuardrailFunctionOutput(
            output_info={"violations": ["Final output is not SalesDiagnosticResult."]},
            tripwire_triggered=True,
        )

    run_context = getattr(context, "context", None)
    ensure_sales_report_snapshot(
        output,
        user_query=getattr(run_context, "user_query", None),
    )
    violations = validate_sales_diagnostic_result(
        output,
        user_query=getattr(run_context, "user_query", None),
    )
    return GuardrailFunctionOutput(
        output_info={"violations": violations},
        tripwire_triggered=bool(violations),
    )


class StrategicRunHooks(RunHooks[EcomRunContext]):
    def __init__(self) -> None:
        self._agent_started_at: dict[str, float] = {}
        self._tool_started_at: dict[tuple[str, str], float] = {}

    async def on_agent_start(self, context, agent) -> None:
        run_context = context.context
        if isinstance(run_context, EcomRunContext):
            run_context.agent_trace.append(f"start:{agent.name}")
            self._agent_started_at[agent.name] = time.monotonic()
            run_context.add_decision_trace(
                "agent_start",
                agent=agent.name,
                status="started",
            )
        logger.info("agent_start name=%s", agent.name)

    async def on_agent_end(self, context, agent, output: Any) -> None:
        run_context = context.context
        if isinstance(run_context, EcomRunContext):
            run_context.agent_trace.append(f"end:{agent.name}")
            started_at = self._agent_started_at.pop(agent.name, None)
            run_context.add_decision_trace(
                "agent_end",
                agent=agent.name,
                status="completed",
                duration_ms=(
                    int((time.monotonic() - started_at) * 1000)
                    if started_at is not None
                    else None
                ),
                details={"output_type": type(output).__name__},
            )
        logger.info("agent_end name=%s output_type=%s", agent.name, type(output).__name__)

    async def on_tool_start(self, context, agent, tool) -> None:
        run_context = context.context
        if isinstance(run_context, EcomRunContext):
            run_context.tool_trace.append(tool.name)
            self._tool_started_at[(agent.name, tool.name)] = time.monotonic()
            run_context.add_decision_trace(
                "tool_start",
                agent=agent.name,
                tool=tool.name,
                status="started",
            )
        logger.info("tool_start agent=%s tool=%s", agent.name, tool.name)

    async def on_tool_end(self, context, agent, tool, result: str) -> None:
        run_context = context.context
        if isinstance(run_context, EcomRunContext):
            started_at = self._tool_started_at.pop((agent.name, tool.name), None)
            run_context.add_decision_trace(
                "tool_end",
                agent=agent.name,
                tool=tool.name,
                status="completed",
                duration_ms=(
                    int((time.monotonic() - started_at) * 1000)
                    if started_at is not None
                    else None
                ),
            )
        logger.info("tool_end agent=%s tool=%s", agent.name, tool.name)


def build_run_hooks() -> StrategicRunHooks:
    return StrategicRunHooks()


def _structured_output_type(runtime_spec: LLMRuntimeSpec, output_type: Any) -> Any:
    if runtime_spec.supports_agent_structured_outputs:
        return output_type
    return None


def _structured_output_guardrails(runtime_spec: LLMRuntimeSpec, guardrails: list[Any]) -> list[Any]:
    if runtime_spec.supports_agent_structured_outputs:
        return guardrails
    return []


def build_sales_agent() -> Agent[EcomRunContext]:
    return _build_sales_agent(get_active_llm_runtime_spec())


@lru_cache(maxsize=8)
def _build_sales_agent(runtime_spec: LLMRuntimeSpec) -> Agent[EcomRunContext]:
    return Agent(
        name="Sales Agent",
        instructions=_dynamic_instructions(SALES_PROMPT_PATH),
        model=build_agent_model(runtime_spec),
        model_settings=ModelSettings(tool_choice="auto"),
        tools=trace_cached_tools([
            get_database_time,
            get_sales_formula_catalog,
            get_sales_report_snapshot,
            get_sales_table_catalog,
            get_sales_overview,
            get_sku_daily_price_sales_comparison,
            get_sales_sku_revenue_change_report,
            *build_role_db_tools("sales", SALES_SCHEMAS),
        ]),
        output_type=_structured_output_type(runtime_spec, SalesDiagnosticResult),
        output_guardrails=_structured_output_guardrails(runtime_spec, [sales_output_guardrail]),
    )


def _build_domain_agents() -> tuple[Agent[EcomRunContext], Agent[EcomRunContext]]:
    return _build_domain_agents_for_runtime(get_active_llm_runtime_spec())


@lru_cache(maxsize=8)
def _build_domain_agents_for_runtime(
    runtime_spec: LLMRuntimeSpec,
) -> tuple[Agent[EcomRunContext], Agent[EcomRunContext]]:
    marketing_agent = Agent(
        name="Marketing Agent",
        instructions=_dynamic_instructions(MARKETING_PROMPT_PATH),
        model=build_agent_model(runtime_spec),
        model_settings=ModelSettings(tool_choice="auto"),
        tools=trace_cached_tools([
            get_database_time,
            get_marketing_report_snapshot,
            get_marketing_formula_catalog,
            get_marketing_table_catalog,
            get_marketing_deep_diagnostic,
            get_marketing_overview,
            get_marketing_campaign_period_report,
            get_marketing_drr_report,
            get_marketing_sku_period_comparison_report,
            get_marketing_campaign_efficiency_snapshot,
            get_marketing_query_performance_snapshot,
            get_marketing_bid_pressure_snapshot,
            get_marketing_visibility_snapshot,
            get_marketing_search_demand_snapshot,
            get_marketing_action_matrix,
            *build_role_db_tools("marketing", MARKETING_SCHEMAS),
        ]),
        output_type=_structured_output_type(runtime_spec, MarketingDiagnosticResult),
        output_guardrails=_structured_output_guardrails(runtime_spec, [marketing_output_guardrail]),
    )
    finance_agent = Agent(
        name="Finance Agent",
        instructions=_dynamic_instructions(FINANCE_PROMPT_PATH),
        model=build_agent_model(runtime_spec),
        model_settings=ModelSettings(tool_choice="auto"),
        tools=trace_cached_tools([
            get_database_time,
            get_finance_formula_catalog,
            get_finance_overview,
            get_finance_marketplace_report,
            get_finance_marketplace_report_text,
            get_finance_sku_report,
            get_finance_sku_report_text,
            get_finance_sku_profit_change_report,
            get_finance_sku_profit_change_report_text,
            *build_role_db_tools("finance", FINANCE_SCHEMAS),
        ]),
        output_type=_structured_output_type(runtime_spec, AgentDiagnosticResult),
        output_guardrails=_structured_output_guardrails(runtime_spec, [diagnostic_output_guardrail]),
    )

    return finance_agent, marketing_agent


def build_marketing_agent() -> Agent[EcomRunContext]:
    return _build_domain_agents()[1]


def build_price_agent() -> Agent[EcomRunContext]:
    return _build_price_agent(get_active_llm_runtime_spec())


@lru_cache(maxsize=8)
def _build_price_agent(runtime_spec: LLMRuntimeSpec) -> Agent[EcomRunContext]:
    return Agent(
        name="Price Agent",
        instructions=_dynamic_instructions(PRICE_PROMPT_PATH),
        model=build_agent_model(runtime_spec),
        model_settings=ModelSettings(tool_choice="auto"),
        tools=trace_cached_tools([
            get_database_time,
            get_price_formula_catalog,
            get_price_table_catalog,
            get_price_context,
            get_price_decision_overview,
            get_price_competitor_overview,
            get_price_competitor_details,
            get_sku_daily_price_sales_comparison,
            *build_role_db_tools("price", PRICE_SCHEMAS),
        ]),
        output_type=_structured_output_type(runtime_spec, AgentDiagnosticResult),
        output_guardrails=_structured_output_guardrails(runtime_spec, [diagnostic_output_guardrail]),
    )


def build_supply_agent() -> Agent[EcomRunContext]:
    return _build_supply_agent(get_active_llm_runtime_spec())


@lru_cache(maxsize=8)
def _build_supply_agent(runtime_spec: LLMRuntimeSpec) -> Agent[EcomRunContext]:
    return Agent(
        name="Supply Agent",
        instructions=_dynamic_instructions(SUPPLY_PROMPT_PATH),
        model=build_agent_model(runtime_spec),
        model_settings=ModelSettings(tool_choice="auto"),
        tools=trace_cached_tools([
            get_database_time,
            get_supply_formula_catalog,
            get_supply_table_catalog,
            get_supply_overview,
            get_current_inventory_value,
            get_inventory_turnover,
            *build_role_db_tools("supply", SUPPLY_SCHEMAS),
        ]),
        output_type=_structured_output_type(runtime_spec, AgentDiagnosticResult),
        output_guardrails=_structured_output_guardrails(runtime_spec, [supply_output_guardrail]),
    )


def build_finance_agent() -> Agent[EcomRunContext]:
    return _build_domain_agents()[0]


build_sales_agent.cache_clear = _build_sales_agent.cache_clear  # type: ignore[attr-defined]
_build_domain_agents.cache_clear = _build_domain_agents_for_runtime.cache_clear  # type: ignore[attr-defined]
build_marketing_agent.cache_clear = _build_domain_agents_for_runtime.cache_clear  # type: ignore[attr-defined]
build_finance_agent.cache_clear = _build_domain_agents_for_runtime.cache_clear  # type: ignore[attr-defined]
build_price_agent.cache_clear = _build_price_agent.cache_clear  # type: ignore[attr-defined]
build_supply_agent.cache_clear = _build_supply_agent.cache_clear  # type: ignore[attr-defined]


def _base_domain_agent_for_runtime(
    runtime_spec: LLMRuntimeSpec,
    intent: QueryIntent,
) -> Agent[EcomRunContext]:
    if intent == QueryIntent.SALES:
        return _build_sales_agent(runtime_spec)
    if intent == QueryIntent.MARKETING:
        return _build_domain_agents_for_runtime(runtime_spec)[1]
    if intent == QueryIntent.FINANCE:
        return _build_domain_agents_for_runtime(runtime_spec)[0]
    if intent == QueryIntent.PRICE:
        return _build_price_agent(runtime_spec)
    return _build_supply_agent(runtime_spec)


def _round_output_type(round_type: RoundType) -> type[BaseModel] | None:
    return {
        RoundType.CROSS_CHECK: CrossCheckResult,
        RoundType.FOLLOWUP: FollowupResult,
        RoundType.GAP_RESOLUTION: GapResolutionResult,
        RoundType.CLAIM_VERIFICATION: ClaimVerificationResult,
    }.get(round_type)


def _tool_name(tool: Any) -> str:
    return str(getattr(tool, "name", "") or "")


@lru_cache(maxsize=128)
def _build_domain_agent_for_round(
    runtime_spec: LLMRuntimeSpec,
    intent_value: str,
    workflow_version_value: str,
    round_type_value: str,
) -> Agent[EcomRunContext]:
    intent = QueryIntent(intent_value)
    workflow_version = resolve_workflow_version(workflow_version_value)
    round_type = RoundType(round_type_value)
    base_agent = _base_domain_agent_for_runtime(runtime_spec, intent)
    if workflow_version == WorkflowVersion.V1:
        return base_agent

    output_type = _round_output_type(round_type)
    if output_type is None:
        structured_output = base_agent.output_type
        output_guardrails = list(base_agent.output_guardrails)
    else:
        structured_output = _structured_output_type(runtime_spec, output_type)
        output_guardrails = []

    return base_agent.clone(
        instructions=_dynamic_instructions(PROMPT_PATH_BY_DOMAIN[intent]),
        tools=list(base_agent.tools),
        output_type=structured_output,
        output_guardrails=output_guardrails,
    )


def build_domain_agent_for_round(
    intent: QueryIntent,
    *,
    workflow_version: WorkflowVersion | str = WorkflowVersion.V1,
    round_type: RoundType | str = RoundType.PRIMARY,
    allowed_tools: list[str] | tuple[str, ...] = (),
) -> Agent[EcomRunContext]:
    normalized_workflow = resolve_workflow_version(workflow_version)
    normalized_round = RoundType(_enum_value(round_type, RoundType.PRIMARY.value))
    cached_agent = _build_domain_agent_for_round(
        get_active_llm_runtime_spec(),
        intent.value,
        normalized_workflow.value,
        normalized_round.value,
    )
    if normalized_workflow == WorkflowVersion.V1 or not allowed_tools:
        return cached_agent
    allow = {str(name) for name in allowed_tools if name}
    selected_tools = [tool for tool in cached_agent.tools if _tool_name(tool) in allow]
    return cached_agent.clone(tools=selected_tools)


def _domain_router_tools() -> dict[str, Any]:
    return {
        "sales_agent": build_sales_agent().as_tool(
            tool_name="sales_agent",
            tool_description=(
                "Используй для вопросов по продажам: продажи, заказы, выручка продаж, "
                "динамика по артикулам, регионы/склады, отмены, выкупы и воронка карточки."
            ),
            custom_output_extractor=_structured_output_extractor,
            max_turns=12,
        ),
        "finance_agent": build_finance_agent().as_tool(
            tool_name="finance_agent",
            tool_description=(
                "Используй для финансовых вопросов: выручка, расходы маркетплейса, комиссия, "
                "эквайринг, логистика, хранение, штрафы, выплаты, финансовые отчёты и ограничения."
            ),
            custom_output_extractor=_structured_output_extractor,
            max_turns=12,
        ),
        "marketing_agent": build_marketing_agent().as_tool(
            tool_name="marketing_agent",
            tool_description=(
                "Используй для маркетинговых вопросов: реклама, кампании, кликабельность, "
                "цена клика, конверсия клика в заказ, видимость, поисковые запросы, ставки, "
                "распределение затрат и таблицы артикулов в кампаниях."
            ),
            custom_output_extractor=_structured_output_extractor,
            max_turns=12,
        ),
        "price_agent": build_price_agent().as_tool(
            tool_name="price_agent",
            tool_description=(
                "Используй для вопросов по товарным ценам: цены конкурентов, скидки, "
                "ценовой сегмент, дороже/дешевле рынка, переоценка и конкурентное "
                "ценовое позиционирование."
            ),
            custom_output_extractor=_structured_output_extractor,
            max_turns=12,
        ),
        "supply_agent": build_supply_agent().as_tool(
            tool_name="supply_agent",
            tool_description=(
                "Используй для вопросов по остаткам и поставкам: наличие, дефицит, "
                "покрытие запасом, дозаказ, поставки, товары в пути и риск out-of-stock."
            ),
            custom_output_extractor=_structured_output_extractor,
            max_turns=12,
        ),
    }


def _domain_router_instructions(strict_rules: str) -> Any:
    base_prompt = (
        "Ты лёгкий русскоязычный оркестратор для аналитики e-commerce.\n"
        "Твоя задача: для простых сообщений ответить напрямую, а для аналитики направить запрос "
        "в профильный управляемый workflow. Не подменяй профильных агентов свободным пересказом.\n\n"
        "Доступные специалисты:\n"
        "- sales_agent: продажи, заказы, выручка продаж, динамика по артикулам, регионы/склады, отмены, выкупы, воронка карточки.\n"
        "- finance_agent: финансы, расходы маркетплейса, выручка, комиссия, эквайринг, логистика, хранение, штрафы, выплаты, финансовые ограничения.\n"
        "- marketing_agent: реклама, кампании, кликабельность, цена клика, конверсия клика в заказ, поисковые запросы, видимость, ставки, распределение затрат, таблицы артикулов в кампаниях.\n"
        "- price_agent: товарные цены, цены конкурентов, скидки, ценовой сегмент, дороже/дешевле рынка, переоценка и конкурентное ценовое позиционирование.\n"
        "- supply_agent: остатки, наличие, дефицит, покрытие запасом, дозаказ, поставки, товары в пути и риск out-of-stock.\n\n"
        "Правила:\n"
        "- Используй Chat memory, чтобы понимать продолжения диалога и короткие подтверждения.\n"
        "- Если текущее сообщение короткое подтверждение вроде `да`, `сделай`, `покажи`, `ок`, `давай`, "
        "и предыдущий ответ ассистента предлагал конкретное следующее действие, трактуй это как просьбу выполнить предложенное действие.\n"
        "- Для такого follow-up выбери профильного агента по смыслу предыдущего предложения в Chat memory; "
        "не отвечай `чем помочь?` и не считай это простым приветствием.\n"
        "- Если текущий Runtime context intent конфликтует с явным продолжением в Chat memory, доверяй Chat memory.\n"
        "- На приветствия, проверки доступности, уточнения и простые разговорные сообщения отвечай напрямую по-русски.\n"
        "- На вопросы по продажам, заказам, динамике продаж, SKU, регионам, складам, отменам и воронке карточки вызывай sales_agent.\n"
        "- На финансовые вопросы и вопросы по финансовым данным вызывай finance_agent.\n"
        "- На маркетинговые вопросы и вопросы по рекламным данным вызывай marketing_agent.\n"
        "- На вопросы по товарным ценам, скидкам, конкурентным ценам, ценовому сегменту, дороже/дешевле рынка и переоценке вызывай price_agent.\n"
        "- На вопросы про цену клика, CPC и ставки рекламы вызывай marketing_agent, а не price_agent.\n"
        "- На вопросы по остаткам, поставкам, наличию, дефициту, покрытию запасом и дозаказу вызывай supply_agent.\n"
        "- Не вызывай нескольких специалистов напрямую; мультиагентную диагностику выполняет runner workflow.\n"
        "- Не создавай бизнес-репорт или 4P-отчёт.\n"
        "- Если вызвал профильного агента, верни его ответ пользователю без потери структуры: "
        "сохраняй таблицы, выводы, ограничения данных и рекомендации.\n"
        "- Не сжимай детальные таблицы в краткую выжимку, если пользователь попросил полный отчёт или таблицу.\n"
        "- Не пересказывай результат профильного агента своими словами, если от этого теряются доказательства или структура.\n"
        "- Финальный ответ делай практичным и без лишних англицизмов."
        f"{strict_rules}"
    )

    def _instructions(context_wrapper, agent) -> str:
        context = context_wrapper.context
        if isinstance(context, EcomRunContext):
            if resolve_workflow_version(context.workflow_version) != WorkflowVersion.V1:
                context.round_type = RoundType.ROUTER
                memory_mode = memory_mode_for_round(RoundType.ROUTER)
                context.memory_mode = memory_mode
                compact_router = (
                    "Ты Router e-commerce. Определи один основной домен либо ответь напрямую только на "
                    "простое разговорное сообщение. sales: продажи/заказы/выручка; finance: расходы/маржа; "
                    "marketing: реклама/CTR/CPC/ДРР; price: цена/скидка/конкуренты; supply: остатки/поставки. "
                    "Для короткого продолжения используй summary и последние сообщения."
                )
                return (
                    f"{compact_router}\n{round_contract(RoundType.ROUTER)}\n\n"
                    f"{_build_v2_context_header(context, memory_mode)}"
                ).strip()
            return (
                f"{base_prompt}\n\n"
                f"{_build_context_header(context)}\n\n"
                f"{build_memory_context_block(context.as_chat_context_window())}"
            )
        return base_prompt

    return _instructions


def _diagnostic_orchestrator_instructions() -> Any:
    legacy_base_prompt = (
        "Ты скрытый Diagnostic Orchestrator для аналитики e-commerce.\n"
        "Доменных агентов уже вызвал Python-controlled workflow; тебе передан очищенный synthesis package.\n"
        "Твоя задача: синтезировать один финальный бизнес-ответ пользователю, не вызывая tools.\n\n"
        "Как читать вход:\n"
        "- `primary_evidence` — основной доменный разбор.\n"
        "- `supporting_evidence` — факты соседних направлений; используй их как подтверждение или опровержение факторов.\n"
        "- `confirmed_facts` — факты, которые можно использовать в выводе.\n"
        "- `contradictions` — факторы, которые скорее не являются причиной.\n"
        "- `sku_metric_matrix` — обязательные расчетные строки по SKU/артикулам для метрик, которые спросил "
        "пользователь: остатки, оборачиваемость, продажи, цена, реклама, маржа. Если "
        "`sku_metric_matrix_required=true` или `format_policy.sku_metric_matrix=true`, финальный ответ обязан "
        "содержать SKU-матрицу с рассчитанными значениями и коротким выводом по каждой SKU-строке.\n"
        "- `sku_evidence_matrix` — детерминированная матрица SKU × домен. Если она есть и "
        "`format_policy.sku_cause_action_matrix=true`, матрицу `Почему это произошло` строй по ней, "
        "не меняй смысл статусов ячеек и не добавляй причины без evidence. Если "
        "`format_policy.sku_cause_action_matrix=false` или `sku_evidence_matrix` пустая, не добавляй раздел "
        "`Почему это произошло`.\n"
        "Статусы переводь так: `confirmed` = `подтверждено`, `partial` = `частично подтверждено`, "
        "`refuted` = `опровергнуто`, `unconfirmed` = `не подтверждено после проверки`, "
        "`no_data` = `нет данных`, `not_checked` = `не проверялось`.\n"
        "- В строках `sku_evidence_matrix` поле `conclusion`/`summary` — это готовый смысл для колонки "
        "`итоговая причина`; используй его, чтобы не превращать sales-механику в причину.\n"
        "- `domain_coverage` и `evidence_quality` — служебная проверка покрытия. Пользователю их не показывай.\n"
        "- `claim_verification` — служебная проверка сильных причинных утверждений перед synthesis. "
        "Если `status=incomplete`, не называй причину подтвержденной по доменам из `missing_domains`; "
        "пиши `не подтверждено по доступным данным` или смягчай до механического вклада.\n"
        "- `decision_mode` — режим управленческого ответа. Если `mode=scenario_decision`, пользователь просит "
        "решение/действие, а не только аудит доказательств: отсутствие точной эластичности, точного uplift "
        "или окончательного floor снижает уверенность, но не отменяет сценарное решение.\n"
        "- `blocking_gaps` — единственные пробелы, которые можно показывать пользователю.\n"
        "- `non_blocking_gaps` — не перечисляй в ответе, если они не меняют главный вывод.\n\n"
        "- `resolved_gaps` — уже закрытые соседними проверками ограничения; пользователю их не показывай.\n\n"
        "- `action_safety_checks` — второй раунд проверки действий; используй его перед рекомендациями "
        "по цене, рекламе, дозаказу и распродаже.\n"
        "- `format_policy.action_task_count` — максимальное число строк в таблице `Что делать`, если поле задано.\n\n"
        "- `display_policy.user_facing_identifier=sku_only` означает: в пользовательском ответе, таблицах, "
        "заголовках и задачах показывай только SKU/артикул; не пиши `nm_id`.\n\n"
        "- Если `primary_evidence.sales_report_snapshot` или `primary_evidence.marketing_report_snapshot` "
        "заполнен, поставь эту таблицу самым первым блоком ответа, перед заголовком и кратким выводом.\n\n"
        "Финальный формат:\n"
        "# Короткий заголовок по смыслу запроса\n\n"
        "Краткий вывод: 1-2 предложения. Не называй падение объёма продаж первопричиной; "
        "пиши, что это механика изменения прибыли, а первичная причина подтверждена, частично подтверждена "
        "или не подтверждена по SKU. Для простых расчётных запросов про оборачиваемость склада/запаса "
        "без явного вопроса `почему` не пиши `причина замедления подтверждена`; пиши "
        "`подтвержден механический вклад в медленную оборачиваемость` или `избыточный запас подтвержден`.\n\n"
        "Что изменилось механически: 1 короткая таблица по SKU с Δ продаж/заказов, Δ выручки и Δ маржи, "
        "если такие данные есть.\n\n"
        "SKU-матрица расчёта: если `sku_metric_matrix_required=true`, добавь таблицу по каждой SKU из "
        "`sku_metric_matrix` или из SKU-таблиц `primary_evidence`. Не показывай только портфельный итог. "
        "Обязательная последняя колонка — `Вывод`, где коротко интерпретируешь строку этой SKU. Для "
        "оборачиваемости используй колонки по смыслу: SKU, продажи, средний доступный остаток за период, "
        "средний полный остаток за период, текущий доступный остаток, текущий полный остаток, "
        "оборачиваемость, дни запаса, вывод. Если текущего остатка нет в evidence, явно не выдумывай его.\n\n"
        "Сценарное решение: если `decision_mode.mode=scenario_decision`, сразу после краткого вывода "
        "дай таблицу решения. Для price decision колонки: SKU, текущая цена, стартовая/рекомендованная цена, "
        "изменение цены, ожидаемые заказы/маржа если есть, уверенность/статус, стоп-условие. "
        "Не заменяй эту таблицу разделом `Неподтверждено` или `Ограничения`. Если confidence low, пиши "
        "`сценарный тест`, а не `решение невозможно`. Если `decision_mode.mode=decision_needed`, не выдумывай "
        "цену без сценарных price-данных: коротко покажи, какого price-decision факта не хватило.\n\n"
        "Проверка гипотез: добавляй только для диагностических вопросов про `почему`, `причины`, `гипотезы`, "
        "`упало`, `выросло` или `изменилось`. Формат — компактная markdown-таблица с колонками "
        "`Гипотеза`, `Признак`, `Результат`, `Статус`, `Смысл`. Если пользователь просит простой отчёт, "
        "не добавляй эту таблицу и не раздувай ответ формулами.\n\n"
        "Почему это произошло: добавляй только если `format_policy.sku_cause_action_matrix=true` "
        "и `sku_evidence_matrix` не пустая. Это компактная матрица гипотез по SKU. Колонки: SKU, спрос, реклама, цена, "
        "остатки, экономика, итоговая причина. В ячейках используй статусы `подтверждено`, "
        "`частично подтверждено`, `опровергнуто`, `не подтверждено после проверки`, `нет данных`, "
        "`не проверялось` и короткий факт. В итоговой причине не пиши `подтверждено`, если подтверждена "
        "только механика продаж; пиши `изменение подтверждено, первопричина не закрыта`.\n\n"
        "Проверка действий: если `format_policy.show_action_safety_summary=true`, добавь короткую таблицу "
        "`Действие`, `Что проверили во втором раунде`, `Статус`, `Что разрешено`.\n\n"
        "Ключевые данные: 1-3 самые важные таблицы из evidence package, если они есть и полезны; "
        "не включай технические идентификаторы.\n\n"
        "Подтверждено: 3-5 коротких предложений без маркированного списка. "
        "Объединяй близкие факты в одну строку, не раскладывай каждый факт отдельным пунктом.\n\n"
        "Ограничения: максимум 1-2 короткие строки только из `blocking_gaps`. "
        "Если `blocking_gaps` пустой, не добавляй этот раздел.\n\n"
        "Неподтверждено: 1-3 короткие строки только по важным факторам, которые проверялись, "
        "но не получили подтверждения. Если таких факторов нет, не добавляй этот раздел.\n\n"
        "Что делать:\n"
        "1. Выводи задачи только markdown-таблицей с колонками: `#`, `Задача`, `Объект`, `Почему`.\n"
        "2. Покажи максимум 5 строк; если `format_policy.action_task_count` задан, используй его как максимум строк, "
        "а не как число отдельных SKU.\n"
        "3. Одна строка = одно действие + одна причина. Не создавай отдельную строку на каждый SKU, "
        "если действие и причина одинаковые.\n"
        "4. Если под задачу попадает несколько объектов, сгруппируй их в ячейке `Объект`; покажи до 5 объектов "
        "и добавь `+ ещё N`, например `46230, 46306, 46225 + ещё 12`.\n"
        "5. `Задача` должна быть полной короткой фразой из глагола и бизнес-объекта, а не одним словом. "
        "Правильно: `Проверить доступный остаток`, `Остановить слабую рекламу`, "
        "`Рассчитать ценовой тест`, `Проверить карточку и корзину`, `Удержать рекламу без расширения`. "
        "Неправильно: `Проверить`, `Остановить`, `Рассчитать`, `Удержать`.\n"
        "6. `Почему` — короткий бизнес-факт, а не протокол проверки: `полка пустая`, "
        "`расход есть, заказов 0`, `маржа есть, цена выше рынка`.\n"
        "7. Не пиши в таблице задач длинные условные конструкции и fallback-фразы: `иначе зафиксировать`, "
        "`при повторной сверке`, `если метрика... иначе...`.\n"
        "8. Не рекомендуй снижение цены, масштабирование рекламы, дозаказ или распродажу как готовое "
        "действие, если нужный `action_safety_checks` имеет статус `missing` или `unconfirmed`; "
        "в этом случае формулируй строку как полную короткую фразу с объектом: `Проверить маржу перед снижением`, "
        "`Рассчитать безопасную цену`, `Сверить доступность перед запуском`.\n"
        "9. Не начинай задачи со слов `Срочно`, `Важно`, `В приоритете`, `Критично`, "
        "`Обратить внимание`; срочность выражай смыслом задачи, а не эмоциональным маркером.\n\n"
        "Правила качества:\n"
        "- В пользовательском markdown запрещено писать `nm_id`; если SKU не сопоставлен, добавь это в `Ограничения`.\n"
        "- Не придумывай причинность, если данные показывают только связь.\n"
        "- Для оборачиваемости склада отделяй расчетный факт от причинности: высокий запас в днях "
        "подтверждает механический вклад SKU в медленный оборот, но не подтверждает причину спроса, цены "
        "или рекламы без соответствующей смежной проверки.\n"
        "- Если `sku_metric_matrix_required=true`, запрещено ограничиться строкой `Портфель` или топ-итогом: "
        "по каждой SKU-строке, которую вернул расчетный источник, должен быть числовой ряд и вывод. "
        "Не смешивай средний остаток за период и текущий остаток на дату; подписывай их разными колонками.\n"
        "- Если `sku_evidence_matrix` содержит для ячейки статус `not_checked`, пиши `не проверялось`, "
        "а не `не подтверждено`. Если статус `no_data`, пиши `нет данных`. Если статус `refuted`, "
        "пиши `опровергнуто`. Если статус `unconfirmed`, пиши `не подтверждено после проверки`.\n"
        "- В матрице причин не смешивай `не проверялось`, `нет данных`, `опровергнуто` и "
        "`не подтверждено после проверки`; это разные статусы.\n"
        "- Не называй `снижение объёма продаж` причиной без пояснения, что это механический вклад в прибыль; "
        "всегда отделяй механику от первичной причины.\n"
        "- Не пиши безусловные команды `сократить`, `остановить`, `снизить`, `масштабировать`, "
        "если соответствующий safety-check не подтверждён. Если не подтверждён — формулируй задачу как проверку.\n"
        "- Сохраняй важные таблицы, но не перегружай ответ служебными деталями.\n"
        "- Не превращай раздел `Подтверждено` в длинную простыню bullet-list. "
        "В этом разделе запрещены строки, начинающиеся с `- ` или `* `.\n"
        "- Не перечисляй все ограничения подряд; показывай только `blocking_gaps`, связанные с вопросом пользователя.\n"
        "- Для `decision_mode.mode=scenario_decision` не выводи отсутствие точной эластичности, точного uplift "
        "или окончательного safe floor как блокирующее ограничение, если есть сценарная цена/рекомендация. "
        "Такие факты формулируй коротко как `уверенность низкая/сценарный тест`, с guardrail и стоп-условием.\n"
        "- Не показывай `resolved_gaps`: это ограничения, уже закрытые другим доменом в текущем workflow.\n"
        "- Если соседняя проверка не дала фактов, пиши коротко: `фактор не подтверждён по доступным данным`, без технической причины.\n"
        "- Не выводи сырые JSON, названия service tools, `cross_agent_requests`, `request_question`, `Проверяли:`, "
        "`инструмент вернул ошибку`, `guardrail`, `Marketing Agent`, `Price Agent`, `Supply Agent`, `Sales Agent`, `Finance Agent` как заголовки служебных блоков.\n"
        "- Если после лимита проверок причина не закрыта, напиши `не подтверждено по доступным данным`.\n"
        "- Отвечай по-русски, без лишних англицизмов.\n"
        "- Не создавай 4P/CEO/BusinessReport."
    )

    def _instructions(context_wrapper, agent) -> str:
        context = context_wrapper.context
        if isinstance(context, EcomRunContext):
            workflow_version = resolve_workflow_version(context.workflow_version)
            if workflow_version != WorkflowVersion.V1:
                memory_mode = memory_mode_for_round(RoundType.SYNTHESIS)
                context.memory_mode = memory_mode
                return (
                    f"{round_contract(RoundType.SYNTHESIS)}\n\n"
                    f"{_build_v2_context_header(context, memory_mode)}"
                ).strip()
            return (
                f"{legacy_base_prompt}\n\n"
                f"{_build_context_header(context)}\n\n"
                f"{build_memory_context_block(context.as_chat_context_window())}"
            )
        return legacy_base_prompt

    return _instructions


def _action_executor_instructions() -> Any:
    base_prompt = (
        "Ты скрытый Action Executor для AI KAM. Ты выполняешь одну карточку Action Board "
        "как финальный операционный анализ, а не как новую диагностику.\n\n"
        "Главное правило: не создавай новый раздел `Что делать`, не генерируй новые задачи, "
        "не предлагай еще один pipeline. Верни финальное решение по переданной задаче.\n\n"
        "Метод работы:\n"
        "1. Сам определи объект задачи: SKU/артикул, кампанию, период или другой target.\n"
        "2. Сам выбери путь проверки: использовать переданный evidence, готовый reader или scoped read-only SQL. "
        "Не делай лишних вызовов, если данных уже хватает.\n"
        "3. Если задача про наличие/продажи, найди достаточные факты про остаток, товары в пути, скорость продаж, "
        "покрытие в днях, риск out-of-stock, маржинальность и маркетинговый спрос.\n"
        "4. Для supply forecast базовый метод: salesVelocity = sales_units_recent / active_days; "
        "coverageDays = available_stock / salesVelocity; reorderQty = 14 дней спроса + safety stock "
        "- доступный остаток - товары в пути. Если нужен другой метод, выбери его и объясни почему.\n"
        "5. Marketing guardrail: не разрешай масштабирование рекламы, если запас не покрывает горизонт "
        "или маржа не подтверждена.\n"
        "6. Если прямых данных не хватает, сначала ищи альтернативный способ оценки. Только если альтернативы "
        "тоже не закрывают расчет, верни decisionStatus=`blocked` или `needs_review` и конкретные blockedBy. "
        "Не выдумывай недостающие числа.\n\n"
        "Формат ответа строго ActionExecutionResult. Поле summary должно быть коротким текстом для карточки."
    )

    def _instructions(context_wrapper, agent) -> str:
        context = context_wrapper.context
        if isinstance(context, EcomRunContext):
            return (
                f"{base_prompt}\n\n"
                f"{_build_context_header(context)}\n\n"
                f"{build_memory_context_block(context.as_chat_context_window())}"
            )
        return base_prompt

    return _instructions


def _action_result_formatter_instructions() -> Any:
    base_prompt = (
        "Ты скрытый Action Result Formatter для AI KAM. На вход ты получаешь уже собранный "
        "deterministic evidence package и draft result по одной карточке Action Board.\n\n"
        "Главные правила:\n"
        "1. Не запускай новую диагностику и не предлагай новый pipeline.\n"
        "2. Не создавай раздел `Что делать` и не генерируй новые задачи.\n"
        "3. Не меняй числа, decisionStatus, blockedBy, confidence, forecast и статусы проверок.\n"
        "4. Разрешено только улучшить формулировки businessMeaning, decision, finalRecommendation, "
        "summary и текст summary у checks.\n"
        "5. Верни строго ActionExecutionResult."
    )

    def _instructions(context_wrapper, agent) -> str:
        context = context_wrapper.context
        if isinstance(context, EcomRunContext):
            return (
                f"{base_prompt}\n\n"
                f"{_build_context_header(context)}\n\n"
                f"{build_memory_context_block(context.as_chat_context_window())}"
            )
        return base_prompt

    return _instructions


def build_domain_router_agent(required_tool_name: str | None = None) -> Agent[EcomRunContext]:
    return _build_domain_router_agent(get_active_llm_runtime_spec(), required_tool_name)


@lru_cache(maxsize=16)
def _build_domain_router_agent(
    runtime_spec: LLMRuntimeSpec,
    required_tool_name: str | None = None,
) -> Agent[EcomRunContext]:
    all_tools = _domain_router_tools()
    if required_tool_name is not None and required_tool_name not in all_tools:
        raise ValueError(f"Unknown domain router tool: {required_tool_name}")

    tools = [all_tools[required_tool_name]] if required_tool_name else list(all_tools.values())
    strict_rules = ""
    if required_tool_name:
        strict_rules = (
            "\n\nСтрогий режим маршрутизации:\n"
            f"- Этот запрос уже классифицирован как доменный; единственный профильный специалист в tools — `{required_tool_name}`.\n"
            "- Для аналитики направь запрос к профильному специалисту, но короткие уточнения и служебные ответы можешь закрыть напрямую.\n"
            "- Если данных не хватает, профильный агент должен сначала попробовать закрыть пробел доступными данными или read-only SQL."
        )

    return Agent(
        name="Domain Router Orchestrator",
        instructions=_domain_router_instructions(strict_rules),
        model=build_agent_model(runtime_spec),
        model_settings=ModelSettings(tool_choice="auto") if required_tool_name else ModelSettings(),
        tools=tools,
    )


def build_diagnostic_orchestrator_agent() -> Agent[EcomRunContext]:
    return _build_diagnostic_orchestrator_agent(get_active_llm_runtime_spec())


@lru_cache(maxsize=8)
def _build_diagnostic_orchestrator_agent(runtime_spec: LLMRuntimeSpec) -> Agent[EcomRunContext]:
    return Agent(
        name="Diagnostic Orchestrator",
        instructions=_diagnostic_orchestrator_instructions(),
        model=build_agent_model(runtime_spec),
        model_settings=ModelSettings(),
        tools=[],
    )


def build_action_executor_agent() -> Agent[EcomRunContext]:
    return _build_action_executor_agent(get_active_llm_runtime_spec())


@lru_cache(maxsize=8)
def _build_action_executor_agent(runtime_spec: LLMRuntimeSpec) -> Agent[EcomRunContext]:
    return Agent(
        name="Action Executor",
        instructions=_action_executor_instructions(),
        model=build_agent_model(runtime_spec),
        model_settings=ModelSettings(tool_choice="auto"),
        tools=[
            get_database_time,
            get_sales_formula_catalog,
            get_sales_report_snapshot,
            get_sales_table_catalog,
            get_sales_overview,
            get_sku_daily_price_sales_comparison,
            get_sales_sku_revenue_change_report,
            get_supply_formula_catalog,
            get_supply_table_catalog,
            get_supply_overview,
            get_inventory_turnover,
            get_finance_formula_catalog,
            get_finance_sku_report,
            get_finance_sku_report_text,
            get_finance_sku_profit_change_report,
            get_finance_sku_profit_change_report_text,
            get_marketing_formula_catalog,
            get_marketing_report_snapshot,
            get_marketing_sku_period_comparison_report,
            get_marketing_campaign_efficiency_snapshot,
            get_marketing_action_matrix,
            get_price_formula_catalog,
            get_price_context,
            get_price_competitor_details,
            *build_role_db_tools("sales", SALES_SCHEMAS),
            *build_role_db_tools("supply", SUPPLY_SCHEMAS),
            *build_role_db_tools("finance", FINANCE_SCHEMAS),
            *build_role_db_tools("marketing", MARKETING_SCHEMAS),
            *build_role_db_tools("price", PRICE_SCHEMAS),
        ],
        output_type=_structured_output_type(runtime_spec, ActionExecutionResult),
    )


def build_action_result_formatter_agent() -> Agent[EcomRunContext]:
    return _build_action_result_formatter_agent(get_active_llm_runtime_spec())


@lru_cache(maxsize=8)
def _build_action_result_formatter_agent(runtime_spec: LLMRuntimeSpec) -> Agent[EcomRunContext]:
    return Agent(
        name="Action Result Formatter",
        instructions=_action_result_formatter_instructions(),
        model=build_agent_model(runtime_spec),
        model_settings=ModelSettings(),
        tools=[],
        output_type=_structured_output_type(runtime_spec, ActionExecutionResult),
    )


build_domain_router_agent.cache_clear = _build_domain_router_agent.cache_clear  # type: ignore[attr-defined]
build_diagnostic_orchestrator_agent.cache_clear = _build_diagnostic_orchestrator_agent.cache_clear  # type: ignore[attr-defined]
build_action_executor_agent.cache_clear = _build_action_executor_agent.cache_clear  # type: ignore[attr-defined]
build_action_result_formatter_agent.cache_clear = _build_action_result_formatter_agent.cache_clear  # type: ignore[attr-defined]
