from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel

from agi_ecom_app.schemas import (
    ClaimVerificationResult,
    CrossCheckResult,
    EvidenceFact,
    FollowupResult,
    GapResolutionResult,
    MemoryMode,
    PromptBuildResult,
    PromptReport,
    QueryMode,
    RoundTask,
    RoundType,
    WorkflowVersion,
)


SECONDARY_PROMPT_MAX_CHARS = 12_000
SYNTHESIS_PACKAGE_MAX_CHARS = 32_000
DOMAIN_SYSTEM_MAX_CHARS = 4_000
ROUND_CONTRACT_MAX_CHARS = 1_500

_WORD_RE = re.compile(r"[a-zа-яё0-9]+", re.IGNORECASE)


def resolve_workflow_version(value: Any) -> WorkflowVersion:
    if isinstance(value, WorkflowVersion):
        return value
    normalized = str(value or "").strip().lower()
    try:
        return WorkflowVersion(normalized)
    except ValueError:
        return WorkflowVersion.V1


def _resolve_round_type(value: RoundType | str) -> RoundType:
    if isinstance(value, RoundType):
        return value
    try:
        return RoundType(str(value or "").strip().lower())
    except ValueError:
        return RoundType.PRIMARY


def determine_query_mode(user_query: str) -> QueryMode:
    """Classify deterministically, with decision > diagnostic > report/lookup precedence."""

    normalized = " ".join(str(user_query or "").lower().replace("ё", "е").split())
    decision_markers = (
        "что делать",
        "как поступить",
        "рекомендуй",
        "рекомендац",
        "стоит ли",
        "можно ли",
        "нужно ли",
        "следует ли",
        "пора ли",
        "безопасно ли",
        "какое решение",
        "план действий",
        "какую цену",
        "установить цену",
        "поднять цену",
        "поднять ли цен",
        "снизить ли цен",
        "менять ли цен",
        "изменять ли цен",
        "увеличить цену",
        "снизить цену",
        "изменить цену",
        "заказать товар",
        "заказать ли",
        "заказывать ли",
        "закупить ли",
        "заказать поставку",
        "дозаказать",
        "дозаказ",
        "пополн",
        "остановить реклам",
        "отключить реклам",
        "отключать ли реклам",
        "запускать ли реклам",
        "отключить кампанию",
        "запустить реклам",
        "what should",
        "recommend",
        "should i",
    )
    decision_patterns = (
        r"\bкакую(?:\s+[a-zа-я0-9]+){0,3}\s+цену\s+(?:поставить|установить)\b",
        r"\b(?:сформируй|составь|предложи|подготовь)\b.{0,120}\b(?:план|портфел[ья])\b.{0,120}\bдейств",
    )
    diagnostic_markers = (
        "почему",
        "причин",
        "за счет чего",
        "что повлияло",
        "диагност",
        "проанализируй",
        "разбери",
        "объясни падение",
        "объясни рост",
        "why",
        "root cause",
        "diagnos",
        "analy",
    )
    report_markers = (
        "отчет",
        "отчёт",
        "сводк",
        "динамик",
        "изменени",
        "сравни период",
        "покажи ключевые",
        "report",
        "summary",
        "trend",
    )
    lookup_markers = (
        "сколько",
        "какой товар",
        "какой sku",
        "кто лидер",
        "покажи остат",
        "значение",
        "на какую сумму",
        "what is",
        "how many",
        "which sku",
    )
    if any(marker in normalized for marker in decision_markers) or any(
        re.search(pattern, normalized, flags=re.IGNORECASE)
        for pattern in decision_patterns
    ):
        return QueryMode.DECISION
    if any(marker in normalized for marker in diagnostic_markers):
        return QueryMode.DIAGNOSTIC
    domain_signal_groups = (
        ("продаж", "заказ", "выруч", "sales", "revenue"),
        ("реклам", "кампани", "дrr", "дрр", "ctr", "cpc", "marketing"),
        ("цен", "скид", "промо", "price"),
        ("остат", "оборач", "склад", "запас", "постав", "supply", "stock"),
        ("прибыл", "марж", "себестоим", "финанс", "деньги", "finance", "cost"),
    )
    domain_count = sum(
        1 for markers in domain_signal_groups if any(marker in normalized for marker in markers)
    )
    if domain_count >= 2 and any(
        marker in normalized
        for marker in ("посчитай", "рассчитай", "сравни", "разлож", "вместе", "сводн")
    ):
        return QueryMode.DIAGNOSTIC
    if any(marker in normalized for marker in report_markers):
        return QueryMode.REPORT
    if any(marker in normalized for marker in lookup_markers):
        return QueryMode.LOOKUP
    return QueryMode.REPORT


def _resolve_query_mode(value: Any, *, user_query: str = "") -> QueryMode:
    if isinstance(value, QueryMode):
        return value
    try:
        return QueryMode(str(value or "").strip().lower())
    except ValueError:
        return determine_query_mode(user_query)


def memory_mode_for_round(round_type: RoundType | str) -> MemoryMode:
    resolved = _resolve_round_type(round_type)
    if resolved in {RoundType.ROUTER, RoundType.PRIMARY}:
        return MemoryMode.SUMMARY_RECENT
    if resolved == RoundType.SYNTHESIS:
        return MemoryMode.SUMMARY_ONLY
    if resolved in {
        RoundType.CROSS_CHECK,
        RoundType.FOLLOWUP,
        RoundType.GAP_RESOLUTION,
        RoundType.CLAIM_VERIFICATION,
    }:
        return MemoryMode.NONE
    return MemoryMode.FULL


def domain_base_contract(domain: str) -> str:
    normalized = str(domain or "").strip().lower().replace("-", "_").replace(" ", "_")
    normalized = normalized.removesuffix("_agent")
    responsibilities = {
        "sales": (
            "Ты Sales-агент. Проверяй продажи в штуках, заказы, выкупы, выручку и их динамику. "
            "Не подменяй продажи выручкой и не считай отсутствие SKU в ограниченной выборке нулём. "
            "Остатки, себестоимость и рекламные причины подтверждай только данными соответствующего домена. "
            "Верни три обязательные таблицы SalesDiagnosticResult; decomposition должна содержать текущие, "
            "предыдущие и delta-значения отдельно для выручки и продаж. Optional snapshot для узкого периода "
            "может содержать только запрошенный период; три сравнения нужны лишь для явного полного отчёта."
        ),
        "marketing": (
            "Ты Marketing-агент. Проверяй рекламу, показы, клики, CTR, CPC, расходы, заказы, ДРР и сравнение кампаний. "
            "Не объявляй рекламу причиной изменения общих продаж без смежной проверки. "
            "Разделяй отсутствие данных, нулевое значение и кампанию вне выбранного периода. "
            "Верни MarketingDiagnosticResult минимум с одной доказательной таблицей. Optional snapshot для узкого "
            "периода может содержать только этот период; три сравнения нужны лишь для явного полного отчёта."
        ),
        "price": (
            "Ты Price-агент. Проверяй цену, скидку, промо и доступные конкурентные ценовые сигналы. "
            "Не называй ценовой эффект доказанным без сравнимого периода и не рассчитывай прибыль без Finance. "
            "Не принимай слово `оценка` за запрос цены: работай по явно указанной метрике."
        ),
        "supply": (
            "Ты Supply-агент. Проверяй остатки, товары в пути, скорость продаж, покрытие, оборачиваемость и риск out-of-stock. "
            "Для исторического периода используй согласованный as-of срез. "
            "Если период оборачиваемости не задан, используй последний полный 30-дневный период до последней даты БД; "
            "если round_task.period задан, считай его обязательным и не наследуй однодневный период из primary evidence. "
            "Любой текущий остаток SKU бери из get_supply_overview.sku_stock_lookup: available_stock_qty — сумма "
            "по всем складским строкам; отдельная строка quantity не является общим остатком. "
            "Не оценивай деньги в остатках без подтверждённой себестоимости Finance."
        ),
        "finance": (
            "Ты Finance-агент. Проверяй выручку, комиссии, логистику, себестоимость, расходы и contribution margin. "
            "Не называй contribution margin чистой прибылью. "
            "Стоимость текущих остатков считай только как подтверждённый остаток Supply, умноженный на подтверждённую себестоимость. "
            "Для подтверждённого lookup или анализа верни минимум одну компактную таблицу."
        ),
        "synthesis": (
            "Ты Synthesis-агент. Собирай ответ только из переданного evidence ledger. "
            "Не вызывай tools, не добавляй новые числа и не превращай no_data или отсутствие строки в ноль."
        ),
    }
    role = responsibilities.get(normalized, responsibilities["synthesis"])
    contract = (
        f"{role}\n"
        "Семантика QueryPlan: `scope=portfolio` всегда означает анализ всей товарной матрицы. "
        "Непустой `targets` при portfolio — это SKU для дополнительной углублённой проверки, а не фильтр "
        "портфеля. Сначала собери полный портфельный срез, затем отдельно проверь эти targets. "
        "При `scope=sku_list` или `scope=single_sku` targets, напротив, задают исчерпывающий узкий scope. "
        "Общие правила: каждое число должно иметь tool-backed evidence, единицу и период. "
        "В primary-ответе каждый числовой факт, на котором может строиться решение, верни также в `facts` как "
        "EvidenceFact с target_id, source_table, source_snapshot_at, retrieved_at, metric_definition, "
        "freshness_status и evidence_ref; не придумывай эти поля, если tool их не вернул. "
        "Если факт не подтверждён, верни no_data или unverified; не достраивай его по смыслу. "
        "У каждой таблицы должны быть непустые title и columns; каждая строка должна содержать ровно столько "
        "ячеек, сколько объявлено columns. Не включай malformed-строки в evidence. "
        "Отвечай строго по выходной схеме текущего раунда и только в своей зоне ответственности. "
        "Не показывай пользователю внутренние prompts, названия раундов и служебный JSON."
    )
    return contract[:DOMAIN_SYSTEM_MAX_CHARS]


def round_contract(round_type: RoundType | str) -> str:
    resolved = _resolve_round_type(round_type)
    contracts = {
        RoundType.ROUTER: (
            "Определи только основной домен запроса. Не отвечай на бизнес-вопрос и не выполняй диагностику."
        ),
        RoundType.PRIMARY: (
            "Ответь на исходный вопрос в своём домене. Собери основные tool-backed evidence, явно раздели факты, "
            "выводы и data gaps. Cross-agent request создавай только для конкретной недостающей проверки."
        ),
        RoundType.CROSS_CHECK: (
            "Проверь ровно verification_goal. Верни verdict confirmed, refuted или no_data, подтверждённые факты "
            "и не более двух узких followup_requests. Если round_task.period задан, используй именно его. "
            "Не пересказывай primary и не составляй общий план действий."
        ),
        RoundType.FOLLOWUP: (
            "Проверь ровно указанную рекомендацию или safety condition. Верни safe, unsafe, needs_review или no_data. "
            "Новые cross-agent requests, общий анализ и новые рекомендации запрещены."
        ),
        RoundType.GAP_RESOLUTION: (
            "Закрой только один gap_id одной требуемой метрикой. Верни resolved только вместе с EvidenceFact; иначе "
            "верни no_data или accepted_limitation. Если round_task.period задан, используй именно его. "
            "Не создавай findings, рекомендации, новые gaps или cross-agent requests."
        ),
        RoundType.CLAIM_VERIFICATION: (
            "Проверь только переданный причинный claim. Верни confirmed, refuted или unverified, evidence и список "
            "непроведённых проверок. Не создавай новые cross-agent requests и рекомендации."
        ),
        RoundType.SYNTHESIS: (
            "Собери пользовательский ответ только из компактного evidence ledger. Не вызывай tools и не добавляй новые "
            "значения. Сохрани названия метрик, единицы, периоды и статусы ограничений; устрани противоречивые действия. "
            "Статус unsafe запрещает соответствующее действие, needs_review разрешает только условный следующий шаг. "
            "Соблюдай `query_plan`, `decision_mode`, `claim_verification`, `domain_coverage`, `display_policy`, "
            "`format_policy` и `final_answer_rules` как обязательный контракт. Не смешивай `sku_metric_matrix`, "
            "`sku_evidence_matrix` и `evidence_tables`: это разные типы доказательств."
        ),
        RoundType.TERMINAL_ACTION: (
            "Выполни только явно заданный action contract. Этот раунд не использует v2 diagnostic orchestration."
        ),
    }
    return contracts[resolved][:ROUND_CONTRACT_MAX_CHARS]


def _as_plain(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): _as_plain(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_as_plain(item) for item in value]
    return value


def _as_mapping(value: Any) -> dict[str, Any]:
    plain = _as_plain(value)
    return dict(plain) if isinstance(plain, Mapping) else {}


def _json_text(value: Any) -> str:
    return json.dumps(_as_plain(value), ensure_ascii=False, separators=(",", ":"), default=str)


def _clip_text(value: Any, max_chars: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    if max_chars <= 1:
        return text[:max_chars]
    return f"{text[: max_chars - 1].rstrip()}…"


def _clip_value(value: Any, *, text_limit: int = 600, depth: int = 0) -> Any:
    if depth >= 5:
        return _clip_text(value, text_limit)
    if isinstance(value, str):
        return _clip_text(value, text_limit)
    if isinstance(value, Mapping):
        return {
            _clip_text(key, 80): _clip_value(item, text_limit=text_limit, depth=depth + 1)
            for key, item in list(value.items())[:24]
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_clip_value(item, text_limit=text_limit, depth=depth + 1) for item in list(value)[:24]]
    return value


def _bounded_value(value: Any, max_chars: int, *, text_limit: int = 600) -> Any:
    """Keep arbitrary nested evidence deterministic and below a JSON character budget."""

    budget = max(96, int(max_chars))
    clipped = _clip_value(value, text_limit=text_limit)
    serialized = _json_text(clipped)
    if len(serialized) <= budget:
        return clipped
    marker = {"_truncated": True, "json": _clip_text(_json_text(value), max(16, budget - 40))}
    while len(_json_text(marker)) > budget and len(marker["json"]) > 8:
        marker["json"] = marker["json"][: max(8, len(marker["json"]) - 32)]
    return marker


def _bounded_contract_value(value: Any, max_chars: int, *, text_limit: int) -> Any:
    """Keep a small machine contract byte-for-byte equivalent when it already fits."""

    plain = _as_plain(value)
    if len(_json_text(plain)) <= max_chars:
        return plain
    return _bounded_value(plain, max_chars, text_limit=text_limit)


def _tokens(value: Any) -> set[str]:
    return {
        token
        for token in _WORD_RE.findall(_json_text(value).lower().replace("ё", "е"))
        if len(token) > 2
    }


def _relevance_score(value: Any, task_tokens: set[str], index: int) -> tuple[int, int]:
    overlap = len(_tokens(value) & task_tokens)
    return overlap, -index


def _select_relevant(values: Sequence[Any], task_tokens: set[str], limit: int) -> list[Any]:
    ranked = sorted(
        enumerate(values),
        key=lambda pair: _relevance_score(pair[1], task_tokens, pair[0]),
        reverse=True,
    )
    return [values[index] for index, _value in ranked[:limit]]


def _compact_table(table: Any) -> dict[str, Any]:
    payload = _as_mapping(table)
    columns = list(payload.get("columns") or [])[:16]
    rows = []
    for row in list(payload.get("rows") or [])[:8]:
        row_values = list(row) if isinstance(row, Sequence) and not isinstance(row, str) else [row]
        rows.append([_bounded_value(cell, 320, text_limit=160) for cell in row_values[:16]])
    compact = {
        "title": _clip_text(payload.get("title"), 240),
        "columns": [_clip_text(column, 120) for column in columns],
        "rows": rows,
    }
    if len(_json_text(compact)) > 4_000:
        return _bounded_value(compact, 4_000, text_limit=120)
    return compact


def _compact_prior_evidence(item: Any) -> dict[str, Any]:
    payload = _as_mapping(item)
    output = _as_mapping(payload.get("output")) or payload
    round_result = _as_mapping(output.get("_round_result"))
    facts = list(round_result.get("facts") or output.get("facts") or output.get("findings") or [])[:2]
    compact = {
        "round": _clip_text(payload.get("round") or payload.get("round_type"), 80),
        "target_agent": _clip_text(payload.get("target_agent") or output.get("agent_name"), 80),
        "question": _clip_text(payload.get("question"), 500),
        "summary": _clip_text(output.get("summary"), 900),
        "facts": [_bounded_value(fact, 700, text_limit=240) for fact in facts],
    }
    return _bounded_value(compact, 2_000, text_limit=360)


def compact_secondary_context(
    task: RoundTask,
    primary_output: dict[str, Any] | BaseModel | None,
    prior_evidence: list[Any] | None,
    *,
    max_chars: int = SECONDARY_PROMPT_MAX_CHARS,
    reserved_chars: int = 0,
) -> tuple[dict[str, Any], dict[str, int | bool]]:
    primary = _as_mapping(primary_output)
    primary_round_result = _as_mapping(primary.get("_round_result"))
    findings = list(
        primary_round_result.get("facts") or primary.get("facts") or primary.get("findings") or []
    )
    tables = list(primary.get("tables") or [])
    prior = list(prior_evidence or [])
    task_tokens = _tokens(
        {
            "question": task.question,
            "verification_goal": task.verification_goal,
            "required_checks": task.required_checks,
            "sku": task.sku,
            "period": task.period,
        }
    )
    selected_findings = _select_relevant(findings, task_tokens, 3)
    selected_tables = _select_relevant(tables, task_tokens, 1)
    selected_prior = _select_relevant(prior, task_tokens, 2)
    payload: dict[str, Any] = {
        "original_user_query": _clip_text(task.original_user_query, 2_400),
        "round_task": {
            "round_type": task.round_type.value,
            "target_agent": task.target_agent,
            "question": _clip_text(task.question, 2_000),
            "verification_goal": _clip_value(task.verification_goal, text_limit=900),
            "required_checks": [_clip_text(value, 300) for value in task.required_checks[:12]],
            "source": _clip_text(task.source, 160),
            "gap_id": task.gap_id,
            "claim_policy": _clip_text(task.claim_policy, 240) if task.claim_policy else None,
            "period": task.period,
            "sku": task.sku,
            "vendor_codes": [_clip_text(value, 120) for value in task.vendor_codes[:20]],
            "nm_ids": [_clip_value(value, text_limit=120) for value in task.nm_ids[:20]],
            "allowed_tools": [_clip_text(value, 160) for value in task.allowed_tools],
            "required_tools": [_clip_text(value, 160) for value in task.required_tools],
            "query_plan": _clip_value(task.query_plan, text_limit=900),
        },
        "primary_summary": _clip_text(primary.get("summary"), 2_000),
        "relevant_findings": [
            _bounded_value(item, 1_600, text_limit=480) for item in selected_findings
        ],
        "relevant_tables": [_compact_table(item) for item in selected_tables],
        "prior_evidence": [_compact_prior_evidence(item) for item in selected_prior],
    }
    target_chars = max(512, max_chars - max(0, reserved_chars))
    truncated = False

    def too_large() -> bool:
        return len(_json_text(payload)) > target_chars

    while too_large() and len(payload["prior_evidence"]) > 1:
        payload["prior_evidence"].pop()
        truncated = True
    table_rows = payload["relevant_tables"][0]["rows"] if payload["relevant_tables"] else []
    while too_large() and len(table_rows) > 1:
        table_rows.pop()
        truncated = True
    while too_large() and len(payload["relevant_findings"]) > 1:
        payload["relevant_findings"].pop()
        truncated = True
    for limit in (1_200, 800, 480, 240):
        if not too_large():
            break
        payload["primary_summary"] = _clip_text(payload["primary_summary"], limit)
        payload["round_task"]["question"] = _clip_text(payload["round_task"]["question"], max(240, limit))
        payload["original_user_query"] = _clip_text(payload["original_user_query"], max(240, limit))
        payload["round_task"]["verification_goal"] = _clip_value(
            payload["round_task"]["verification_goal"], text_limit=max(120, limit // 2)
        )
        payload["relevant_findings"] = [
            _bounded_value(item, max(240, limit), text_limit=max(120, limit // 2))
            for item in payload["relevant_findings"]
        ]
        payload["prior_evidence"] = [
            _bounded_value(item, max(240, limit), text_limit=max(120, limit // 2))
            for item in payload["prior_evidence"]
        ]
        truncated = True
    if too_large():
        payload["relevant_tables"] = []
        payload["prior_evidence"] = []
        truncated = True
    if too_large():
        payload["relevant_findings"] = [
            _bounded_value(payload["relevant_findings"][0], 320, text_limit=120)
        ] if payload["relevant_findings"] else []
        payload["round_task"]["required_checks"] = payload["round_task"]["required_checks"][:3]
        payload["round_task"]["allowed_tools"] = payload["round_task"]["allowed_tools"][:12]
        payload["round_task"]["required_tools"] = payload["round_task"]["required_tools"][:8]
        truncated = True

    stats: dict[str, int | bool] = {
        "included_findings": len(payload["relevant_findings"]),
        "dropped_findings": max(0, len(findings) - len(payload["relevant_findings"])),
        "included_tables": len(payload["relevant_tables"]),
        "dropped_tables": max(0, len(tables) - len(payload["relevant_tables"])),
        "included_prior_evidence": len(payload["prior_evidence"]),
        "dropped_prior_evidence": max(0, len(prior) - len(payload["prior_evidence"])),
        "truncated": truncated,
    }
    return payload, stats


def build_round_prompt(
    task: RoundTask,
    primary_output: dict[str, Any] | BaseModel | None,
    prior_evidence: list[Any] | None,
    workflow_version: WorkflowVersion | str,
) -> PromptBuildResult:
    version = resolve_workflow_version(workflow_version)
    try:
        query_mode = QueryMode(str(task.query_plan.get("query_mode")))
    except (TypeError, ValueError):
        query_mode = determine_query_mode(task.original_user_query)
    memory_mode = memory_mode_for_round(task.round_type)
    system_text = f"{domain_base_contract(task.target_agent)}\n{round_contract(task.round_type)}"
    stats: dict[str, int | bool] = {
        "included_findings": 0,
        "dropped_findings": 0,
        "included_tables": 0,
        "dropped_tables": 0,
        "included_prior_evidence": 0,
        "dropped_prior_evidence": 0,
        "truncated": False,
    }
    max_chars: int | None = None

    if task.round_type in {
        RoundType.CROSS_CHECK,
        RoundType.FOLLOWUP,
        RoundType.GAP_RESOLUTION,
        RoundType.CLAIM_VERIFICATION,
    }:
        max_chars = SECONDARY_PROMPT_MAX_CHARS
        payload, stats = compact_secondary_context(
            task,
            primary_output,
            prior_evidence,
            max_chars=max_chars,
            reserved_chars=len(system_text) + 2,
        )
    elif task.round_type == RoundType.SYNTHESIS:
        package = {
            "workflow_version": version.value,
            "query_mode": query_mode.value,
            "original_user_query": task.original_user_query,
            **_as_mapping(primary_output),
        }
        if prior_evidence:
            package.setdefault("evidence_ledger", _as_plain(prior_evidence))
        payload, compact_report = compact_synthesis_package(package)
        stats = {
            "included_findings": compact_report.included_findings,
            "dropped_findings": compact_report.dropped_findings,
            "included_tables": compact_report.included_tables,
            "dropped_tables": compact_report.dropped_tables,
            "included_prior_evidence": compact_report.included_prior_evidence,
            "dropped_prior_evidence": compact_report.dropped_prior_evidence,
            "truncated": compact_report.truncated,
        }
        max_chars = SYNTHESIS_PACKAGE_MAX_CHARS
    else:
        payload = {
            "original_user_query": _clip_text(task.original_user_query, 8_000),
            "round_task": task.model_dump(mode="json"),
            "query_mode": query_mode.value,
        }

    input_text = _json_text(payload)
    # `text` is the model input. Domain and round contracts are installed as the
    # agent's system instructions by `main_agent`; keeping them out of the input
    # avoids paying for the same static contract twice on every round.
    text = input_text
    if max_chars is not None and len(system_text) + 2 + len(text) > max_chars:
        # The structured compactors normally fit before this point. Preserve valid JSON
        # and the task itself in the extremely small-budget fallback.
        current_evidence: Any = None
        for collection in (
            payload.get("relevant_findings") or [],
            payload.get("relevant_tables") or [],
            payload.get("prior_evidence") or [],
        ):
            if collection:
                current_evidence = _bounded_value(collection[0], 1_000, text_limit=240)
                break
        fallback = {
            "original_user_query": _clip_text(task.original_user_query, 480),
            "round_task": {
                "round_type": task.round_type.value,
                "target_agent": _clip_text(task.target_agent, 80),
                "question": _clip_text(task.question, 600),
                "verification_goal": _bounded_value(
                    task.verification_goal,
                    800,
                    text_limit=200,
                ),
                "required_checks": [_clip_text(item, 120) for item in task.required_checks[:3]],
                "gap_id": _clip_text(task.gap_id, 80) if task.gap_id is not None else None,
                "claim_policy": _clip_text(task.claim_policy, 240),
                "period": _clip_text(task.period, 160),
                "sku": _clip_text(task.sku, 120),
                "allowed_tools": [_clip_text(item, 100) for item in task.allowed_tools[:8]],
                "required_tools": [_clip_text(item, 100) for item in task.required_tools[:8]],
                "query_plan": _bounded_value(task.query_plan, 800, text_limit=160),
            },
            "current_evidence": current_evidence,
        }
        input_text = _json_text(fallback)
        text = input_text
        stats["truncated"] = True
        if len(system_text) + 2 + len(text) > max_chars:
            fallback["original_user_query"] = _clip_text(task.original_user_query, 160)
            fallback["round_task"]["question"] = _clip_text(task.question, 240)
            fallback["round_task"]["verification_goal"] = _bounded_value(
                task.verification_goal,
                240,
                text_limit=80,
            )
            fallback["current_evidence"] = _bounded_value(current_evidence, 240, text_limit=80)
            input_text = _json_text(fallback)
            text = input_text

    included_evidence = int(stats["included_findings"]) + int(stats["included_tables"]) + int(
        stats["included_prior_evidence"]
    )
    dropped_evidence = int(stats["dropped_findings"]) + int(stats["dropped_tables"]) + int(
        stats["dropped_prior_evidence"]
    )
    report = PromptReport(
        workflow_version=version,
        round_type=task.round_type,
        query_mode=query_mode,
        memory_mode=memory_mode,
        system_chars=len(system_text),
        input_chars=len(input_text),
        total_chars=len(system_text) + 2 + len(text),
        max_chars=max_chars,
        included_evidence=included_evidence,
        dropped_evidence=dropped_evidence,
        included_findings=int(stats["included_findings"]),
        dropped_findings=int(stats["dropped_findings"]),
        included_tables=int(stats["included_tables"]),
        dropped_tables=int(stats["dropped_tables"]),
        included_prior_evidence=int(stats["included_prior_evidence"]),
        dropped_prior_evidence=int(stats["dropped_prior_evidence"]),
        allowed_tools=list(task.allowed_tools),
        required_tools=list(task.required_tools),
        truncated=bool(stats["truncated"]),
    )
    return PromptBuildResult(text=text, report=report)


def _fact_status(item: Any) -> str:
    return str(_as_mapping(item).get("status") or "confirmed").strip().lower()


def compact_synthesis_package(
    package: dict[str, Any] | BaseModel,
    max_chars: int = SYNTHESIS_PACKAGE_MAX_CHARS,
) -> tuple[dict[str, Any], PromptReport]:
    """Bound synthesis evidence without discarding its machine-readable contract.

    Contract fields stay present even in the emergency size fallback.  The
    compactor spends the character budget on evidence rows instead of silently
    removing QueryPlan, verification, safety, or rendering semantics.
    """

    source = _as_mapping(package)
    original_query = str(source.get("original_user_query") or source.get("user_query") or "")
    query_mode = _resolve_query_mode(source.get("query_mode"), user_query=original_query)
    version = resolve_workflow_version(source.get("workflow_version"))

    confirmed = list(source.get("confirmed_facts") or [])
    refuted = list(source.get("refuted_facts") or source.get("contradictions") or [])
    ledger = list(source.get("evidence_ledger") or [])
    for item in ledger:
        status = _fact_status(item)
        if status == "confirmed":
            confirmed.append(item)
        elif status == "refuted":
            refuted.append(item)

    all_gaps = list(source.get("blocking_gaps") or source.get("gaps") or source.get("data_gaps") or [])
    blocking_gaps = []
    for gap in all_gaps:
        payload = _as_mapping(gap)
        status = str(payload.get("status") or "open").lower()
        if payload.get("blocking", True) and status not in {"resolved", "closed"}:
            blocking_gaps.append(gap)

    sku_metric_matrix = list(
        source.get("sku_metric_matrix")
        or source.get("sku_matrices")
        or source.get("sku_matrix")
        or []
    )
    sku_evidence_matrix = list(source.get("sku_evidence_matrix") or [])
    evidence_tables: list[dict[str, Any]] = []
    primary_evidence = _as_mapping(source.get("primary_evidence"))
    primary_tables = list(primary_evidence.get("tables") or [])
    for snapshot_key in ("sales_report_snapshot", "marketing_report_snapshot"):
        if primary_evidence.get(snapshot_key):
            primary_tables.insert(0, primary_evidence[snapshot_key])
    for table in primary_tables[:3]:
        evidence_tables.append(
            {
                "source": "primary",
                "table": _compact_table(table),
            }
        )
    for evidence in list(source.get("supporting_evidence") or [])[:5]:
        evidence_payload = _as_mapping(evidence)
        for table in list(evidence_payload.get("tables") or [])[:1]:
            evidence_tables.append(
                {
                    "source": _clip_text(evidence_payload.get("target_agent"), 80),
                    "table": _compact_table(table),
                }
            )
    safety = list(
        source.get("action_safety_checks")
        or source.get("safety_statuses")
        or source.get("safety_checks")
        or []
    )
    non_blocking_gaps = list(source.get("non_blocking_gaps") or [])
    resolved_gaps = list(source.get("resolved_gaps") or [])
    accepted_limitations = list(source.get("accepted_limitations") or [])

    compact = {
        "workflow_version": version.value,
        "query_mode": query_mode.value,
        "original_user_query": _clip_text(original_query, 4_000),
        "primary_agent": _clip_text(source.get("primary_agent"), 80),
        "query_plan": _bounded_contract_value(
            source.get("query_plan") or {}, 6_000, text_limit=800
        ),
        "decision_mode": _bounded_contract_value(
            source.get("decision_mode") or {}, 2_400, text_limit=500
        ),
        "primary_summary": _clip_text(primary_evidence.get("summary"), 1_200),
        "confirmed_facts": [
            _bounded_value(item, 1_600, text_limit=600) for item in confirmed
        ],
        "refuted_facts": [
            _bounded_value(item, 1_600, text_limit=600) for item in refuted
        ],
        "blocking_gaps": [
            _bounded_value(item, 1_600, text_limit=600) for item in blocking_gaps
        ],
        "accepted_limitations": [
            _bounded_value(item, 1_000, text_limit=360) for item in accepted_limitations
        ],
        "non_blocking_gaps": [
            _bounded_value(item, 900, text_limit=320) for item in non_blocking_gaps
        ],
        "resolved_gaps": [
            _bounded_value(item, 900, text_limit=320) for item in resolved_gaps
        ],
        "sku_metric_matrix_required": bool(source.get("sku_metric_matrix_required")),
        "sku_metric_matrix": [
            _bounded_value(item, 1_800, text_limit=480) for item in sku_metric_matrix
        ],
        "sku_evidence_matrix": [
            _bounded_value(item, 1_800, text_limit=480) for item in sku_evidence_matrix
        ],
        "evidence_tables": [
            _bounded_value(item, 1_800, text_limit=400) for item in evidence_tables
        ],
        "action_safety_checks": [
            _bounded_value(item, 1_600, text_limit=400) for item in safety
        ],
        "claim_verification": _bounded_contract_value(
            source.get("claim_verification") or {}, 2_400, text_limit=500
        ),
        "domain_coverage": _bounded_contract_value(
            source.get("domain_coverage") or {}, 2_400, text_limit=400
        ),
        "evidence_quality": _bounded_contract_value(
            source.get("evidence_quality") or {}, 2_400, text_limit=400
        ),
        "display_policy": _bounded_contract_value(
            source.get("display_policy") or {}, 1_600, text_limit=320
        ),
        "format_policy": _bounded_contract_value(
            source.get("format_policy") or {}, 1_600, text_limit=320
        ),
        "final_answer_rules": [
            _clip_text(item, 600) for item in list(source.get("final_answer_rules") or [])[:24]
        ],
    }
    original_counts = {key: len(value) for key, value in compact.items() if isinstance(value, list)}
    contract_sources = {
        "query_plan": source.get("query_plan") or {},
        "decision_mode": source.get("decision_mode") or {},
        "claim_verification": source.get("claim_verification") or {},
        "domain_coverage": source.get("domain_coverage") or {},
        "evidence_quality": source.get("evidence_quality") or {},
        "display_policy": source.get("display_policy") or {},
        "format_policy": source.get("format_policy") or {},
    }
    truncated = any(
        _json_text(compact[key]) != _json_text(_as_plain(value))
        for key, value in contract_sources.items()
    ) or len(original_query) > 4_000
    list_keys = (
        "resolved_gaps",
        "non_blocking_gaps",
        "evidence_tables",
        "accepted_limitations",
        "refuted_facts",
        "sku_evidence_matrix",
        "sku_metric_matrix",
        "action_safety_checks",
        "blocking_gaps",
        "confirmed_facts",
    )
    while len(_json_text(compact)) > max_chars:
        candidates = [key for key in list_keys if len(compact[key]) > 1]
        if not candidates:
            break
        selected = max(candidates, key=lambda key: len(compact[key]))
        compact[selected].pop()
        truncated = True
    if len(_json_text(compact)) > max_chars:
        for key in list_keys:
            compact[key] = [
                _bounded_value(item, 1_000, text_limit=160) for item in compact[key]
            ]
        compact["original_user_query"] = _clip_text(compact["original_user_query"], 800)
        compact["primary_summary"] = _clip_text(compact["primary_summary"], 480)
        truncated = True
    if len(_json_text(compact)) > max_chars:
        for key in list_keys:
            compact[key] = [
                _bounded_value(item, 480, text_limit=100) for item in compact[key][:1]
            ]
        compact["original_user_query"] = _clip_text(compact["original_user_query"], 240)
        compact["primary_summary"] = _clip_text(compact["primary_summary"], 240)
        truncated = True
    if len(_json_text(compact)) > max_chars:
        for key in list_keys:
            compact[key] = [
                _bounded_value(item, 160, text_limit=48) for item in compact[key][:1]
            ]
        compact["original_user_query"] = _clip_text(compact["original_user_query"], 80)
        compact["primary_summary"] = _clip_text(compact["primary_summary"], 80)
        truncated = True

    if len(_json_text(compact)) > max_chars:
        # Last-resort protection for pathologically large nested contracts.  Do
        # not remove contract keys: retain a bounded marker/value under each one.
        scalar_keys = (
            "query_plan",
            "decision_mode",
            "claim_verification",
            "domain_coverage",
            "evidence_quality",
            "display_policy",
            "format_policy",
        )
        for key in scalar_keys:
            compact[key] = _bounded_value(compact[key], 320, text_limit=96)
        compact["final_answer_rules"] = [
            _clip_text(item, 160) for item in compact["final_answer_rules"][:3]
        ]
        truncated = True

    included = sum(len(compact[key]) for key in list_keys)
    original_total = sum(original_counts.get(key, 0) for key in list_keys)
    report = PromptReport(
        workflow_version=version,
        round_type=RoundType.SYNTHESIS,
        query_mode=query_mode,
        memory_mode=MemoryMode.SUMMARY_ONLY,
        input_chars=len(_json_text(compact)),
        total_chars=len(_json_text(compact)),
        max_chars=max_chars,
        included_evidence=included,
        dropped_evidence=max(0, original_total - included),
        included_findings=len(compact["confirmed_facts"]) + len(compact["refuted_facts"]),
        dropped_findings=max(
            0,
            original_counts.get("confirmed_facts", 0)
            + original_counts.get("refuted_facts", 0)
            - len(compact["confirmed_facts"])
            - len(compact["refuted_facts"]),
        ),
        included_tables=(
            len(compact["sku_metric_matrix"])
            + len(compact["sku_evidence_matrix"])
            + len(compact["evidence_tables"])
        ),
        dropped_tables=max(
            0,
            original_counts.get("sku_metric_matrix", 0)
            + original_counts.get("sku_evidence_matrix", 0)
            + original_counts.get("evidence_tables", 0)
            - len(compact["sku_metric_matrix"])
            - len(compact["sku_evidence_matrix"])
            - len(compact["evidence_tables"]),
        ),
        included_prior_evidence=included,
        dropped_prior_evidence=max(0, original_total - included),
        truncated=truncated,
    )
    return compact, report


def _fact_as_finding(
    fact: EvidenceFact,
    *,
    verification_verdict: str | None = None,
) -> dict[str, Any]:
    if fact.statement:
        observation = fact.statement
    elif fact.value is None:
        observation = fact.metric
    else:
        observation = f"{fact.value}{f' {fact.unit}' if fact.unit else ''}"
    evidence = [fact.evidence_ref]
    if fact.period:
        evidence.append(f"period={fact.period}")
    if fact.sku:
        evidence.append(f"sku={fact.sku}")
    return {
        "factor": fact.metric,
        "observation": observation,
        "impact": f"Evidence status: {fact.status}.",
        "cause": "Не устанавливается этой проверкой.",
        "action": "Использовать только в пределах указанного evidence status.",
        "evidence": evidence,
        "confidence": fact.confidence,
        "driver_class": (
            "not_a_driver"
            if fact.status == "refuted" or verification_verdict == "refuted"
            else "secondary_driver"
        ),
        "evidence_status": fact.status,
        "verification_verdict": verification_verdict,
    }


def _legacy_envelope(
    *,
    agent_name: str,
    summary: str,
    facts: list[EvidenceFact] | None = None,
    tables: list[Any] | None = None,
    data_gaps: list[Any] | None = None,
    cross_agent_requests: list[Any] | None = None,
    verification_verdict: str | None = None,
) -> dict[str, Any]:
    return {
        "agent_name": agent_name,
        "summary": summary,
        "tables": [_as_plain(table) for table in (tables or [])],
        "findings": [
            _fact_as_finding(fact, verification_verdict=verification_verdict)
            for fact in (facts or [])
        ],
        "recommendations": [],
        "data_gaps": [_as_plain(gap) for gap in (data_gaps or [])],
        "cross_agent_requests": [_as_plain(request) for request in (cross_agent_requests or [])],
    }


def _round_result_metadata(
    *,
    round_type: RoundType,
    verdict: str,
    facts: list[EvidenceFact] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "round_type": round_type.value,
        "verdict": verdict,
        "facts": [fact.model_dump(mode="json") for fact in (facts or [])],
        **_as_plain(extra),
    }


def adapt_round_result_to_legacy(result: Any) -> dict[str, Any]:
    if isinstance(result, CrossCheckResult):
        envelope = _legacy_envelope(
            agent_name=result.agent_name,
            summary=result.summary,
            facts=result.facts,
            tables=result.tables,
            data_gaps=result.data_gaps,
            cross_agent_requests=result.followup_requests,
            verification_verdict=result.verdict,
        )
        envelope["_round_result"] = _round_result_metadata(
            round_type=RoundType.CROSS_CHECK,
            verdict=result.verdict,
            facts=result.facts,
        )
        return envelope
    if isinstance(result, FollowupResult):
        envelope = _legacy_envelope(
            agent_name=result.agent_name,
            summary=result.summary,
            facts=result.facts,
            tables=result.tables,
            data_gaps=result.data_gaps,
        )
        envelope["_round_result"] = _round_result_metadata(
            round_type=RoundType.FOLLOWUP,
            verdict=result.status,
            facts=result.facts,
        )
        return envelope
    if isinstance(result, GapResolutionResult):
        gaps = []
        facts = [result.fact] if result.fact is not None else []
        if result.outcome != "resolved":
            gaps.append(
                {
                    "metric": f"gap:{result.gap_id}",
                    "reason": result.reason or result.summary,
                    "impact": result.impact or "Проверка не подтверждена данными.",
                    "outcome": result.outcome,
                }
            )
        envelope = _legacy_envelope(
            agent_name=result.agent_name,
            summary=result.summary,
            facts=facts,
            data_gaps=gaps,
            verification_verdict=("confirmed" if result.outcome == "resolved" else None),
        )
        envelope["gap_id"] = result.gap_id
        envelope["gap_resolution_status"] = result.outcome
        envelope["_round_result"] = _round_result_metadata(
            round_type=RoundType.GAP_RESOLUTION,
            verdict=result.outcome,
            facts=facts,
            gap_id=result.gap_id,
        )
        return envelope
    if isinstance(result, ClaimVerificationResult):
        gaps = [_as_plain(gap) for gap in result.data_gaps]
        gaps.extend(
            {
                "metric": check,
                "reason": "Проверка причинного утверждения не выполнена.",
                "impact": "Claim нельзя считать полностью подтверждённым.",
            }
            for check in result.unchecked_checks
        )
        envelope = _legacy_envelope(
            agent_name=result.agent_name,
            summary=result.summary,
            facts=result.facts,
            data_gaps=gaps,
            verification_verdict=result.verdict,
        )
        envelope["claim"] = result.claim
        envelope["claim_verdict"] = result.verdict
        envelope["_round_result"] = _round_result_metadata(
            round_type=RoundType.CLAIM_VERIFICATION,
            verdict=result.verdict,
            facts=result.facts,
            unchecked_checks=result.unchecked_checks,
        )
        return envelope

    payload = _as_mapping(result)
    if not payload:
        raise TypeError("Round result must be a typed result or a structured legacy mapping")
    if "outcome" in payload and "gap_id" in payload:
        return adapt_round_result_to_legacy(GapResolutionResult.model_validate(payload))
    if "claim" in payload and "verdict" in payload:
        return adapt_round_result_to_legacy(ClaimVerificationResult.model_validate(payload))
    if payload.get("status") in {"safe", "unsafe", "needs_review", "no_data"}:
        return adapt_round_result_to_legacy(FollowupResult.model_validate(payload))
    if payload.get("verdict") in {"confirmed", "refuted", "no_data"}:
        return adapt_round_result_to_legacy(CrossCheckResult.model_validate(payload))
    if any(key in payload for key in ("summary", "tables", "findings", "data_gaps")):
        normalized = dict(payload)
        normalized.setdefault("summary", "")
        normalized.setdefault("tables", [])
        normalized.setdefault("findings", [])
        normalized.setdefault("recommendations", [])
        normalized.setdefault("data_gaps", [])
        normalized.setdefault("cross_agent_requests", [])
        return normalized
    raise ValueError("Round result does not match a typed result or legacy evidence envelope")


__all__ = [
    "DOMAIN_SYSTEM_MAX_CHARS",
    "ROUND_CONTRACT_MAX_CHARS",
    "SECONDARY_PROMPT_MAX_CHARS",
    "SYNTHESIS_PACKAGE_MAX_CHARS",
    "adapt_round_result_to_legacy",
    "build_round_prompt",
    "compact_secondary_context",
    "compact_synthesis_package",
    "determine_query_mode",
    "domain_base_contract",
    "memory_mode_for_round",
    "resolve_workflow_version",
    "round_contract",
]
