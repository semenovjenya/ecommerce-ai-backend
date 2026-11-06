from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional

from agents import function_tool
from pydantic import Field

from agi_ecom_app.schemas import (
    CrossDomainDependencyItem,
    CrossDomainDependencyPlan,
    QueryIntent,
)
from agi_ecom_app.services import strategic_db

MARKETING_TABLE_CATALOG_PATH = (
    Path(__file__).resolve().parents[1] / "prompts" / "marketing_table_catalog.json"
)
SALES_TABLE_CATALOG_PATH = (
    Path(__file__).resolve().parents[1] / "prompts" / "sales_table_catalog.json"
)
SUPPLY_TABLE_CATALOG_PATH = (
    Path(__file__).resolve().parents[1] / "prompts" / "supply_table_catalog.json"
)
PRICE_TABLE_CATALOG_PATH = (
    Path(__file__).resolve().parents[1] / "prompts" / "price_table_catalog.json"
)
DERIVED_FEATURE_CATALOG_PATH = (
    Path(__file__).resolve().parents[1] / "prompts" / "derived_feature_catalog.json"
)

DOMAIN_ALIASES: Dict[str, QueryIntent] = {
    "sales": QueryIntent.SALES,
    "sales_agent": QueryIntent.SALES,
    "sales agent": QueryIntent.SALES,
    "продажи": QueryIntent.SALES,
    "marketing": QueryIntent.MARKETING,
    "marketing_agent": QueryIntent.MARKETING,
    "marketing agent": QueryIntent.MARKETING,
    "маркетинг": QueryIntent.MARKETING,
    "price": QueryIntent.PRICE,
    "price_agent": QueryIntent.PRICE,
    "price agent": QueryIntent.PRICE,
    "цены": QueryIntent.PRICE,
    "цена": QueryIntent.PRICE,
    "ценообразование": QueryIntent.PRICE,
    "supply": QueryIntent.SUPPLY,
    "supply_agent": QueryIntent.SUPPLY,
    "supply agent": QueryIntent.SUPPLY,
    "поставки": QueryIntent.SUPPLY,
    "остатки": QueryIntent.SUPPLY,
    "finance": QueryIntent.FINANCE,
    "finance_agent": QueryIntent.FINANCE,
    "finance agent": QueryIntent.FINANCE,
    "финансы": QueryIntent.FINANCE,
}

DOMAIN_QUESTIONS: Dict[QueryIntent, str] = {
    QueryIntent.SALES: (
        "Проверь фактические продажи, заказы, выкупы, отмены и SKU-драйверы за тот же период."
    ),
    QueryIntent.MARKETING: (
        "Проверь рекламный трафик, показы, клики, кликабельность, цену клика, поисковую видимость и промо за тот же период."
    ),
    QueryIntent.PRICE: (
        "Проверь товарные цены, скидки, ценовой сегмент и цены конкурентов за тот же период."
    ),
    QueryIntent.SUPPLY: (
        "Проверь остатки, доступность по складам, дефицит, поставки, концентрацию запаса и риск out-of-stock за тот же период."
    ),
    QueryIntent.FINANCE: (
        "Проверь маржу, комиссии, логистику, хранение, штрафы, выплаты, себестоимость и финансовую экономику за тот же период."
    ),
}


def _get_formula_catalog(domain: QueryIntent) -> Dict[str, Any]:
    catalog = json.loads(DERIVED_FEATURE_CATALOG_PATH.read_text(encoding="utf-8"))
    domain_key = domain.value
    formulas = catalog.get("domains", {}).get(domain_key, [])
    return {
        "catalog_name": catalog.get("catalog_name"),
        "version": catalog.get("version"),
        "language": catalog.get("language"),
        "domain": domain_key,
        "usage_rule": catalog.get("usage_rule"),
        "formulas": formulas,
    }

BROAD_SALES_PROFIT_DIAGNOSTIC_TARGETS: tuple[QueryIntent, ...] = (
    QueryIntent.SALES,
    QueryIntent.MARKETING,
    QueryIntent.PRICE,
    QueryIntent.SUPPLY,
    QueryIntent.FINANCE,
)
EXPLICIT_HYPOTHESIS_TARGET_RULES: tuple[tuple[QueryIntent, tuple[str, ...]], ...] = (
    (QueryIntent.SALES, ("спрос", "продаж", "выруч", "заказ")),
    (QueryIntent.MARKETING, ("реклам", "маркетинг", "кампан", "трафик", "клик", "показ")),
    (QueryIntent.PRICE, ("цена", "цену", "ценов", "скидк", "дороже", "дешевле")),
    (QueryIntent.SUPPLY, ("остат", "склад", "налич", "доступн", "дефиц")),
    (QueryIntent.FINANCE, ("экономик", "финанс", "прибыл", "марж", "рентаб")),
)

METRIC_RULES: tuple[tuple[tuple[str, ...], tuple[QueryIntent, ...], str, str], ...] = (
    (
        (
            "цены конкурентов",
            "цена конкурентов",
            "конкурентные цены",
            "конкуренты дешевле",
            "дороже конкурентов",
            "дешевле конкурентов",
            "дороже рынка",
            "дешевле рынка",
            "ценовой сегмент",
            "цена высокая",
            "высокая цена",
            "цена низкая",
            "низкая цена",
        ),
        (QueryIntent.PRICE,),
        "formula",
        "Конкурентное ценовое позиционирование требует проверки price-слоя.",
    ),
    (
        (
            "менять цену",
            "изменить цену",
            "поднять цену",
            "снизить цену",
            "переоценка",
            "скидка",
            "промо цена",
        ),
        (QueryIntent.PRICE, QueryIntent.SALES, QueryIntent.FINANCE),
        "formula",
        "Решение по товарной цене требует цен конкурентов, влияния на продажи и финансовой экономики.",
    ),
    (
        (
            "цена влияет на продажи",
            "из-за цены продажи",
            "продажи из-за цены",
            "дорого для рынка",
            "дешево для рынка",
        ),
        (QueryIntent.PRICE, QueryIntent.SALES),
        "formula",
        "Связь цены и продаж требует price-сравнения и фактических продаж.",
    ),
    (
        (
            "прибыльная цена",
            "минимальная цена",
            "минимально допустимая цена",
            "цена и маржа",
            "маржа цены",
        ),
        (QueryIntent.PRICE, QueryIntent.FINANCE),
        "formula",
        "Минимальная или прибыльная цена требует ценового слоя и финансовой экономики.",
    ),
    (
        (
            "что с продаж",
            "почему продаж",
            "продаж измен",
            "изменились продаж",
            "упали продаж",
            "падают продаж",
            "выросли продаж",
            "растут продаж",
            "просели продаж",
        ),
        (QueryIntent.SALES, QueryIntent.SUPPLY, QueryIntent.MARKETING),
        "diagnostic",
        "Диагностика изменения продаж требует проверки продаж, доступности и маркетингового спроса.",
    ),
    (
        ("дрр", "доля рекламных расходов"),
        (QueryIntent.MARKETING, QueryIntent.SALES),
        "formula",
        "ДРР считается из рекламных расходов и выручки продаж.",
    ),
    (
        ("romi", "mer", "окупаемость рекламы", "реклама съедает прибыль"),
        (QueryIntent.MARKETING, QueryIntent.SALES, QueryIntent.FINANCE),
        "formula",
        "Окупаемость рекламы требует рекламных расходов, продаж и финансовой экономики.",
    ),
    (
        ("маржа по sku", "маржа по артик", "маржинальность по sku", "маржинальность по артик"),
        (QueryIntent.FINANCE, QueryIntent.SALES),
        "formula",
        "Маржа по SKU требует финансовой экономики и фактических продаж.",
    ),
    (
        (
            "хватит ли остатков на маркетинг",
            "хватит ли остатков на рекламу",
            "хватит ли остатков на промо",
            "остатков на маркетинг",
            "остатков на рекламу",
            "остатков на промо",
        ),
        (QueryIntent.SUPPLY, QueryIntent.SALES, QueryIntent.MARKETING),
        "formula",
        "Проверка остатков под маркетинг требует запасов, темпа продаж и маркетинговой нагрузки.",
    ),
    (
        ("дозаказ", "сколько заказать", "сколько дозаказать", "хватит ли остатков", "покрытие запасом"),
        (QueryIntent.SUPPLY, QueryIntent.SALES),
        "formula",
        "Дозаказ и покрытие запасом требуют остатков и скорости продаж.",
    ),
    (
        ("клики есть", "клики есть, а заказов нет", "есть клики", "нет заказов"),
        (QueryIntent.MARKETING, QueryIntent.SALES, QueryIntent.SUPPLY),
        "formula",
        "Разрыв между кликами и заказами требует проверки маркетинга, продаж и доступности.",
    ),
    (
        (
            "выручка выросла",
            "прибыль нет",
            "прибыль не вырос",
            "прибыль не раст",
            "прибыль упала",
        ),
        (QueryIntent.FINANCE, QueryIntent.SALES, QueryIntent.MARKETING),
        "formula",
        "Расхождение выручки и прибыли требует проверки продаж, финансов и рекламных расходов.",
    ),
    (
        ("открытия карточ", "падение открытий", "просели открытия"),
        (QueryIntent.SALES, QueryIntent.MARKETING),
        "formula",
        "Падение открытий карточки требует проверки воронки продаж и маркетинговой видимости.",
    ),
    (
        ("товар есть, но не прода", "есть остатки, но не прода", "не продается"),
        (QueryIntent.SUPPLY, QueryIntent.SALES, QueryIntent.MARKETING),
        "formula",
        "Товар с остатком без продаж требует проверки наличия, продаж и спроса.",
    ),
    (
        ("склад влияет", "склады влияют", "влияние склада"),
        (QueryIntent.SUPPLY, QueryIntent.SALES),
        "formula",
        "Влияние склада требует проверки доступности и продаж по складам.",
    ),
    (
        ("акция повлияла", "промо повлия", "promo"),
        (QueryIntent.MARKETING, QueryIntent.SALES),
        "formula",
        "Промо требует проверки маркетинга и продаж; финансовая экономика нужна только при явном финансовом вопросе.",
    ),
    (
        ("overstock", "остатки зависли", "зависшие остатки"),
        (QueryIntent.SUPPLY, QueryIntent.SALES, QueryIntent.FINANCE),
        "formula",
        "Overstock требует проверки запасов, темпа продаж и финансовой нагрузки.",
    ),
    (
        ("какие sku тянут", "sku тянут", "артикулы тянут", "драйверы роста", "драйверы падения"),
        (QueryIntent.SALES, QueryIntent.SUPPLY, QueryIntent.MARKETING),
        "formula",
        "SKU-драйверы требуют продаж, а причины могут быть в наличии или маркетинге.",
    ),
    (
        ("какие кампании масштабировать", "масштабировать кампании", "масштабировать рекламу"),
        (QueryIntent.MARKETING, QueryIntent.SALES),
        "formula",
        "Масштабирование кампаний требует маркетинговых метрик и продаж; финансовый лимит нужен только при явном финансовом вопросе.",
    ),
)

DOMAIN_SIGNAL_RULES: Dict[QueryIntent, tuple[tuple[QueryIntent, tuple[str, ...], str], ...]] = {
    QueryIntent.SALES: (
        (
            QueryIntent.PRICE,
            ("цена конкурент", "ценов", "скидк", "промо-цен", "дороже", "дешевле", "ценовой сегмент"),
            "В sales-ответе есть товарный ценовой фактор вне sales-слоя.",
        ),
        (
            QueryIntent.MARKETING,
            ("трафик", "показ", "клик", "ctr", "cpc", "видим", "поисков", "реклам", "промо", "открытия"),
            "В sales-ответе есть сигнал верхней воронки или маркетингового трафика.",
        ),
        (
            QueryIntent.SUPPLY,
            ("остат", "склад", "доступн", "out-of-stock", "дефицит", "постав", "концентрация запаса", "наличие"),
            "В sales-ответе есть сигнал доступности, склада или остатка.",
        ),
        (
            QueryIntent.FINANCE,
            ("марж", "комис", "логист", "хранен", "штраф", "выплат", "cogs", "себестоим", "прибыл"),
            "В sales-ответе есть финансовый фактор вне sales-слоя.",
        ),
    ),
    QueryIntent.MARKETING: (
        (
            QueryIntent.PRICE,
            ("цена конкурент", "промо-цен", "скидк", "дороже", "дешевле", "ценовой сегмент"),
            "Маркетинговый вывод зависит от товарного ценового позиционирования.",
        ),
        (
            QueryIntent.SALES,
            ("дрр", "выруч", "продаж", "заказ", "выкуп", "отмен", "post-click", "sku", "артикул"),
            "Маркетинговый вывод зависит от фактических продаж или SKU-драйверов.",
        ),
        (
            QueryIntent.SUPPLY,
            ("доступн", "налич", "остат", "out-of-stock", "дефиц", "нет заказов"),
            "Маркетинговый трафик может упираться в наличие или доступность.",
        ),
        (
            QueryIntent.FINANCE,
            ("romi", "mer", "марж", "окупаем", "расход", "себестоим", "прибыл"),
            "Маркетинговый вывод зависит от финансовой экономики.",
        ),
    ),
    QueryIntent.SUPPLY: (
        (
            QueryIntent.PRICE,
            ("цена конкурент", "скидк", "переоцен", "дороже", "дешевле", "ценовой сегмент"),
            "Supply-вывод зависит от товарного ценового позиционирования.",
        ),
        (
            QueryIntent.SALES,
            ("скорость продаж", "продаж", "дозаказ", "покрытие", "forecast", "прогноз", "спрос"),
            "Supply-вывод зависит от темпа продаж и SKU-спроса.",
        ),
        (
            QueryIntent.MARKETING,
            ("реклам", "промо", "видим", "трафик", "спрос", "кампан"),
            "Supply-вывод зависит от маркетингового спроса или промо.",
        ),
        (
            QueryIntent.FINANCE,
            ("деньги в запас", "себестоим", "марж", "хранен", "overstock", "заморож"),
            "Supply-вывод зависит от финансовой нагрузки запасов.",
        ),
    ),
    QueryIntent.FINANCE: (
        (
            QueryIntent.PRICE,
            ("цена конкурент", "ценов", "скидк", "дороже", "дешевле", "переоцен", "минимальная цена"),
            "Финансовый вывод зависит от текущей цены и конкурентного ценового позиционирования.",
        ),
        (
            QueryIntent.SALES,
            ("выруч", "продаж", "заказ", "возврат", "sku", "артикул", "вклад"),
            "Финансовый вывод зависит от фактических продаж или SKU-вклада.",
        ),
        (
            QueryIntent.MARKETING,
            ("реклам", "дрр", "кампан", "маркетинг", "romi", "mer"),
            "Финансовый вывод зависит от рекламных расходов или кампаний.",
        ),
        (
            QueryIntent.SUPPLY,
            ("остат", "overstock", "дефиц", "закупочн", "товары в пути", "запас"),
            "Финансовый вывод зависит от остатков или закупочной нагрузки.",
        ),
    ),
    QueryIntent.PRICE: (
        (
            QueryIntent.SALES,
            ("продаж", "заказ", "выкуп", "отмен", "спрос", "конверси", "оборот"),
            "Price-вывод зависит от фактических продаж или спроса.",
        ),
        (
            QueryIntent.FINANCE,
            ("марж", "прибыл", "себестоим", "комис", "логист", "минимальная цена", "окупаем"),
            "Price-вывод зависит от финансовой экономики.",
        ),
        (
            QueryIntent.MARKETING,
            ("видим", "позици", "поисков", "трафик", "клик", "реклам", "промо"),
            "Price-вывод зависит от маркетинговой видимости или рекламного спроса.",
        ),
    ),
}


def _normalize_text(value: Any) -> str:
    return str(value or "").lower().replace("ё", "е")


def _normalize_agent(value: Any) -> QueryIntent:
    if isinstance(value, QueryIntent):
        return value
    normalized = _normalize_text(value).replace("-", "_")
    if normalized in DOMAIN_ALIASES:
        return DOMAIN_ALIASES[normalized]
    for key, agent in DOMAIN_ALIASES.items():
        if key in normalized:
            return agent
    return QueryIntent.SALES


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def _is_broad_sales_profit_diagnostic_query(text: str) -> bool:
    has_sales = _contains_any(text, ("продаж", "выруч", "заказ", "sku", "артикул"))
    has_profit = _contains_any(text, ("прибыл", "марж", "экономик", "рентаб"))
    has_diagnostic = _contains_any(
        text,
        (
            "почему",
            "причин",
            "гипотез",
            "проверь",
            "разлож",
            "вклад",
            "изменил",
            "подтвержден",
            "неподтвержден",
            "фактор",
        ),
    )
    has_sku_or_factor_scope = _contains_any(
        text,
        (
            "sku",
            "артикул",
            "ключев",
            "вклад по sku",
            "спрос",
            "реклам",
            "цена",
            "остат",
        ),
    )
    return has_sales and has_profit and has_diagnostic and has_sku_or_factor_scope


def _is_profit_leader_hypothesis_query(text: str) -> bool:
    has_profit = _contains_any(text, ("прибыл", "марж", "экономик", "рентаб"))
    has_product_scope = _contains_any(
        text,
        ("товар", "sku", "артикул", "номенклатур", "позици"),
    )
    asks_for_leader = _contains_any(
        text,
        (
            "какой",
            "какие",
            "больше прибыли",
            "больше всего прибыли",
            "принес",
            "принёс",
            "лидер",
            "топ",
            "самый прибыль",
            "наиболее прибыль",
        ),
    )
    return has_profit and has_product_scope and asks_for_leader


def _is_explicit_hypothesis_query(text: str) -> bool:
    return _contains_any(
        text,
        (
            "почему",
            "причин",
            "гипотез",
            "проверь",
            "проверить",
            "проверка",
            "подтвержд",
            "фактор",
            "разлож",
        ),
    )


def _explicit_hypothesis_targets(text: str) -> list[QueryIntent]:
    if not _is_explicit_hypothesis_query(text):
        return []
    targets: list[QueryIntent] = []
    for target, keywords in EXPLICIT_HYPOTHESIS_TARGET_RULES:
        if _contains_any(text, keywords) and target not in targets:
            targets.append(target)
    return targets


def _flatten_payload(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(_flatten_payload(item) for item in value.values())
    if isinstance(value, list):
        return " ".join(_flatten_payload(item) for item in value)
    return str(value)


def _parse_json_list(value: str) -> List[Dict[str, Any]]:
    if not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return [{"text": value}]
    if isinstance(parsed, list):
        return [item if isinstance(item, dict) else {"text": item} for item in parsed]
    if isinstance(parsed, dict):
        return [parsed]
    return [{"text": parsed}]


def _question_key(value: Any) -> str:
    return " ".join(_normalize_text(value).split())


def _asked_keys(already_asked: Optional[List[Dict[str, Any]]]) -> set[tuple[QueryIntent, str]]:
    keys: set[tuple[QueryIntent, str]] = set()
    for item in already_asked or []:
        target = _normalize_agent(item.get("target_agent"))
        question = _question_key(item.get("question"))
        if question:
            keys.add((target, question))
    return keys


def _add_dependency_candidate(
    candidates: Dict[QueryIntent, CrossDomainDependencyItem],
    skipped: list[CrossDomainDependencyItem],
    *,
    target_agent: QueryIntent,
    primary_agent: QueryIntent,
    reason: str,
    question: Optional[str],
    priority: int,
    source: str,
    already_asked_keys: set[tuple[QueryIntent, str]],
    period: Optional[str] = None,
    vendor_codes: Optional[List[str]] = None,
    nm_ids: Optional[List[int | str]] = None,
    required_checks: Optional[List[str]] = None,
) -> None:
    if target_agent == primary_agent:
        return
    resolved_question = question or DOMAIN_QUESTIONS[target_agent]
    item = CrossDomainDependencyItem(
        target_agent=target_agent,
        reason=reason,
        question=resolved_question,
        priority=priority,
        source=source,
        period=period,
        vendor_codes=[str(value) for value in (vendor_codes or [])],
        nm_ids=list(nm_ids or []),
        required_checks=[str(value) for value in (required_checks or [])],
    )
    if (target_agent, _question_key(resolved_question)) in already_asked_keys:
        skipped.append(item)
        return
    current = candidates.get(target_agent)
    if current is None or item.priority > current.priority:
        candidates[target_agent] = item


def build_cross_domain_dependency_plan(
    user_query: str,
    primary_agent: str,
    primary_summary: str = "",
    findings: Optional[List[Dict[str, Any]]] = None,
    data_gaps: Optional[List[Dict[str, Any]]] = None,
    cross_agent_requests: Optional[List[Dict[str, Any]]] = None,
    already_asked: Optional[List[Dict[str, Any]]] = None,
    max_secondary_agents: int = 2,
) -> Dict[str, Any]:
    """Build a deterministic secondary-agent plan from requests, formulas, and domain signals."""
    primary = _normalize_agent(primary_agent)
    max_agents = max(0, min(int(max_secondary_agents or 2), 5))
    candidates: Dict[QueryIntent, CrossDomainDependencyItem] = {}
    skipped: list[CrossDomainDependencyItem] = []
    existing_question_keys = _asked_keys(already_asked)

    for index, request in enumerate(cross_agent_requests or []):
        target = _normalize_agent(request.get("target_agent"))
        reason = str(request.get("reason") or "Primary agent requested cross-domain evidence.")
        question = str(request.get("question") or DOMAIN_QUESTIONS.get(target, "Проверь смежный фактор."))
        _add_dependency_candidate(
            candidates,
            skipped,
            target_agent=target,
            primary_agent=primary,
            reason=reason,
            question=question,
            priority=100 - index,
            source="cross_agent_requests",
            already_asked_keys=existing_question_keys,
            period=request.get("period"),
            vendor_codes=request.get("vendor_codes") or [],
            nm_ids=request.get("nm_ids") or [],
            required_checks=request.get("required_checks") or [],
        )

    query_text = _normalize_text(user_query)
    evidence_text = _normalize_text(
        " ".join(
            [
                user_query,
                primary_summary,
                _flatten_payload(findings or []),
                _flatten_payload(data_gaps or []),
            ]
        )
    )

    if _is_broad_sales_profit_diagnostic_query(query_text) or (
        primary == QueryIntent.FINANCE and _is_profit_leader_hypothesis_query(query_text)
    ):
        for target in BROAD_SALES_PROFIT_DIAGNOSTIC_TARGETS:
            _add_dependency_candidate(
                candidates,
                skipped,
                target_agent=target,
                primary_agent=primary,
                reason=(
                    "Широкая диагностика продаж и прибыли по SKU требует проверки "
                    "спроса, рекламы, цены, остатков и экономики."
                ),
                question=DOMAIN_QUESTIONS[target],
                priority=95,
                source="broad_diagnostic",
                already_asked_keys=existing_question_keys,
            )

    for target in _explicit_hypothesis_targets(query_text):
        _add_dependency_candidate(
            candidates,
            skipped,
            target_agent=target,
            primary_agent=primary,
            reason="Пользователь явно попросил проверить эту гипотезу в матрице причин.",
            question=DOMAIN_QUESTIONS[target],
            priority=90,
            source="explicit_hypothesis",
            already_asked_keys=existing_question_keys,
        )

    for keywords, targets, source, reason in METRIC_RULES:
        if _contains_any(query_text, keywords):
            for target in targets:
                _add_dependency_candidate(
                    candidates,
                    skipped,
                    target_agent=target,
                    primary_agent=primary,
                    reason=reason,
                    question=DOMAIN_QUESTIONS[target],
                    priority=80,
                    source=source,
                    already_asked_keys=existing_question_keys,
                )

    for target, keywords, reason in DOMAIN_SIGNAL_RULES.get(primary, ()):
        if _contains_any(evidence_text, keywords):
            _add_dependency_candidate(
                candidates,
                skipped,
                target_agent=target,
                primary_agent=primary,
                reason=reason,
                question=DOMAIN_QUESTIONS[target],
                priority=60,
                source="domain_signal",
                already_asked_keys=existing_question_keys,
            )

    ranked_items = sorted(
        candidates.values(),
        key=lambda item: -item.priority,
    )
    plan = CrossDomainDependencyPlan(
        primary_agent=primary,
        recommended_secondary_agents=ranked_items[:max_agents],
        skipped_agents=[*ranked_items[max_agents:], *skipped],
        max_secondary_agents=max_agents,
    )
    return plan.model_dump(mode="json")


@function_tool
def get_cross_domain_dependency_plan(
    user_query: str,
    primary_agent: str,
    primary_summary: str = "",
    findings_json: str = "[]",
    data_gaps_json: str = "[]",
    cross_agent_requests_json: str = "[]",
    already_asked_json: str = "[]",
    max_secondary_agents: int = 2,
) -> Dict[str, Any]:
    """Вернуть детерминированный план смежных проверок для скрытого диагностического оркестратора."""
    return build_cross_domain_dependency_plan(
        user_query=user_query,
        primary_agent=primary_agent,
        primary_summary=primary_summary,
        findings=_parse_json_list(findings_json),
        data_gaps=_parse_json_list(data_gaps_json),
        cross_agent_requests=_parse_json_list(cross_agent_requests_json),
        already_asked=_parse_json_list(already_asked_json),
        max_secondary_agents=max_secondary_agents,
    )


@function_tool
def get_sales_table_catalog() -> Dict[str, Any]:
    """Вернуть краткий каталог выбора таблиц sales и формат ответа агента."""
    return json.loads(SALES_TABLE_CATALOG_PATH.read_text(encoding="utf-8"))


@function_tool
def get_sales_overview(
    product_keyword: Optional[str] = None,
    limit: Annotated[int, Field(ge=1, le=100)] = 100,
) -> Dict[str, Any]:
    """Return sales signals for the complete portfolio unless an explicit smaller limit is requested."""
    bounded_limit = max(1, min(int(limit), 100))
    return strategic_db.get_sales_overview(
        product_keyword=product_keyword,
        limit=bounded_limit,
    )


@function_tool
def get_sku_daily_price_sales_comparison(
    product_keyword: str,
    days: int = 14,
) -> Dict[str, Any]:
    """Compare daily seller price, discounts, funnel conversion, orders, and realized sales for one SKU."""
    return strategic_db.get_sku_daily_price_sales_comparison(
        product_keyword=product_keyword,
        days=days,
    )


@function_tool
def get_sales_report_snapshot(
    product_keyword: Optional[str] = None,
) -> Dict[str, Any]:
    """Вернуть обязательную верхнюю таблицу отчёта: последний день, 7 дней и 30 дней к предыдущим периодам."""
    return strategic_db.get_sales_report_snapshot(product_keyword=product_keyword)


@function_tool
def get_marketing_report_snapshot(
    product_keyword: Optional[str] = None,
) -> Dict[str, Any]:
    """Вернуть обязательную верхнюю таблицу маркетингового отчёта: последний день, 7 дней, 30 дней и ДРР."""
    return strategic_db.get_marketing_report_snapshot(product_keyword=product_keyword)


@function_tool
def get_sales_sku_revenue_change_report(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    product_keyword: Optional[str] = None,
    limit: int = 20,
) -> Dict[str, Any]:
    """Вернуть детерминированный вклад артикулов в изменение выручки: текущий период, предыдущий период и дельты."""
    return strategic_db.get_sales_sku_revenue_change(
        date_from=date_from,
        date_to=date_to,
        product_keyword=product_keyword,
        limit=limit,
    )


@function_tool
def get_sales_formula_catalog() -> Dict[str, Any]:
    """Вернуть справочник производных признаков sales для проверки гипотез."""
    return _get_formula_catalog(QueryIntent.SALES)


@function_tool
def get_supply_overview(
    product_keyword: Optional[str] = None,
    limit: Annotated[int, Field(ge=1, le=100)] = 100,
) -> Dict[str, Any]:
    """Return latest stock, reorder, and inbound supply risks.

    product_keyword accepts one SKU/nm_id or a comma-separated list. When the
    configured simulation mode is enabled, the latest known snapshot is marked
    as current for simulation while its original source date is preserved.
    limit must be between 1 and 100. For portfolio analysis, keep the default so
    the full SKU matrix is returned instead of a short watchlist.
    """
    bounded_limit = max(1, min(int(limit), 100))
    return strategic_db.get_supply_overview(
        product_keyword=product_keyword,
        limit=bounded_limit,
    )


@function_tool
def get_current_inventory_value(
    product_keyword: Optional[str] = None,
    limit: int = 100,
) -> Dict[str, Any]:
    """Value current sellable stock as on-hand quantity times confirmed unit cost.

    Sold units, full stock, and customer transit are never used as the quantity
    side of this calculation. Rows without a confirmed cost remain unvalued.
    """
    return strategic_db.get_current_inventory_value(
        product_keyword=product_keyword,
        limit=limit,
    )


@function_tool
def get_inventory_turnover(
    date_from: str,
    date_to: Optional[str] = None,
    vendor_codes: Optional[str] = None,
    limit: int = 100,
) -> Dict[str, Any]:
    """Calculate inventory turnover from sales units and average stock over a period.

    vendor_codes accepts comma- or semicolon-separated supplier articles.
    """
    return strategic_db.get_inventory_turnover(
        date_from=date_from,
        date_to=date_to,
        vendor_codes=vendor_codes,
        limit=limit,
    )


@function_tool
def get_supply_formula_catalog() -> Dict[str, Any]:
    """Вернуть справочник производных признаков supply для проверки гипотез."""
    return _get_formula_catalog(QueryIntent.SUPPLY)


@function_tool
def get_supply_table_catalog() -> Dict[str, Any]:
    """Вернуть краткий каталог выбора таблиц supply и формат ответа агента."""
    return json.loads(SUPPLY_TABLE_CATALOG_PATH.read_text(encoding="utf-8"))


@function_tool
def get_marketing_formula_catalog() -> Dict[str, Any]:
    """Вернуть справочник производных признаков marketing для проверки гипотез."""
    return _get_formula_catalog(QueryIntent.MARKETING)


@function_tool
def get_marketing_overview(
    product_keyword: Optional[str] = None,
    limit: int = 5,
) -> Dict[str, Any]:
    """Вернуть последние маркетинговые затраты, эффективность кампаний и распределение по артикулам."""
    return strategic_db.get_marketing_overview(product_keyword=product_keyword, limit=limit)


@function_tool
def get_marketing_table_catalog() -> Dict[str, Any]:
    """Вернуть краткий каталог выбора таблиц marketing и формат ответа агента."""
    return json.loads(MARKETING_TABLE_CATALOG_PATH.read_text(encoding="utf-8"))


@function_tool
def get_price_table_catalog() -> Dict[str, Any]:
    """Вернуть краткий каталог выбора таблиц price и формат ответа агента."""
    return json.loads(PRICE_TABLE_CATALOG_PATH.read_text(encoding="utf-8"))


@function_tool
def get_price_formula_catalog() -> Dict[str, Any]:
    """Вернуть справочник производных признаков price для проверки гипотез."""
    return _get_formula_catalog(QueryIntent.PRICE)


@function_tool
def get_price_context(
    nm_id: Optional[str] = None,
    vendor_code: Optional[str] = None,
    product_keyword: Optional[str] = None,
    limit: int = 20,
    include_evidence: bool = False,
) -> Dict[str, Any]:
    """Вернуть однозначные слои текущей цены, свежесть и качество конкурентов."""
    return strategic_db.get_price_context(
        nm_id=nm_id,
        vendor_code=vendor_code,
        product_keyword=product_keyword,
        limit=limit,
        include_evidence=include_evidence,
    )


@function_tool
def get_price_competitor_overview(
    product_keyword: Optional[str] = None,
    limit: int = 20,
) -> Dict[str, Any]:
    """Вернуть позиционирование наших товарных цен относительно дедуплицированных конкурентов."""
    return strategic_db.get_price_competitor_overview(
        product_keyword=product_keyword,
        limit=limit,
    )


@function_tool
def get_price_decision_overview(
    product_keyword: Optional[str] = None,
    limit: int = 20,
) -> Dict[str, Any]:
    """Вернуть готовое решение repricing из pricing.mart_price_decision_latest."""
    return strategic_db.get_price_decision_overview(
        product_keyword=product_keyword,
        limit=limit,
    )


@function_tool
def get_price_competitor_details(
    nm_id: Optional[str] = None,
    vendor_code: Optional[str] = None,
    product_keyword: Optional[str] = None,
    limit: int = 50,
) -> Dict[str, Any]:
    """Вернуть дедуплицированный список конкурентов по nm_id, артикулу или поисковому фокусу."""
    return strategic_db.get_price_competitor_details(
        nm_id=nm_id,
        vendor_code=vendor_code,
        product_keyword=product_keyword,
        limit=limit,
    )


@function_tool
def get_marketing_campaign_period_report(
    date_from: str,
    date_to: Optional[str] = None,
    limit: int = 200,
) -> Dict[str, Any]:
    """Вернуть точную таблицу кампаний за период: артикулы, цены продажи и долю затрат."""
    return strategic_db.get_marketing_campaign_period_report(
        date_from=date_from,
        date_to=date_to,
        limit=limit,
    )


@function_tool
def get_marketing_drr_report(
    date_from: str,
    date_to: Optional[str] = None,
    limit: int = 50,
) -> Dict[str, Any]:
    """Детерминированно посчитать ДРР: рекламные затраты / sales-выручка за период."""
    return strategic_db.get_marketing_drr_report(
        date_from=date_from,
        date_to=date_to,
        limit=limit,
    )


@function_tool
def get_marketing_sku_period_comparison_report(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    sku_list: Optional[str] = None,
    product_keyword: Optional[str] = None,
    period_days: int = 7,
    limit: int = 100,
) -> Dict[str, Any]:
    """Сравнить рекламные метрики по SKU за текущий период к предыдущему сопоставимому периоду."""
    return strategic_db.get_marketing_sku_period_comparison_report(
        date_from=date_from,
        date_to=date_to,
        sku_list=sku_list,
        product_keyword=product_keyword,
        period_days=period_days,
        limit=limit,
    )


@function_tool
def get_marketing_campaign_efficiency_snapshot(
    product_keyword: Optional[str] = None,
    days: int = 7,
    limit: int = 20,
) -> Dict[str, Any]:
    """Вернуть оценку эффективности кампаний, лишние затраты и кандидатов на масштабирование."""
    return strategic_db.get_marketing_campaign_efficiency_snapshot(
        product_keyword=product_keyword,
        days=days,
        limit=limit,
    )


@function_tool
def get_marketing_query_performance_snapshot(
    product_keyword: Optional[str] = None,
    days: int = 7,
    limit: int = 30,
) -> Dict[str, Any]:
    """Return query-level demand, waste, and position-risk rows from normquery stats."""
    return strategic_db.get_marketing_query_performance_snapshot(
        product_keyword=product_keyword,
        days=days,
        limit=limit,
    )


@function_tool
def get_marketing_bid_pressure_snapshot(
    product_keyword: Optional[str] = None,
    days: int = 7,
    limit: int = 30,
) -> Dict[str, Any]:
    """Return bid adequacy rows together with overbid and underbid candidates."""
    return strategic_db.get_marketing_bid_pressure_snapshot(
        product_keyword=product_keyword,
        days=days,
        limit=limit,
    )


@function_tool
def get_marketing_visibility_snapshot(
    product_keyword: Optional[str] = None,
    limit: int = 30,
) -> Dict[str, Any]:
    """Вернуть видимость, среднюю позицию и прокси воронки после клика."""
    return strategic_db.get_marketing_visibility_snapshot(
        product_keyword=product_keyword,
        limit=limit,
    )


@function_tool
def get_marketing_search_demand_snapshot(
    product_keyword: Optional[str] = None,
    days: int = 7,
    limit: int = 30,
) -> Dict[str, Any]:
    """Return search-text demand, order-driving queries, and high-frequency low-order rows."""
    return strategic_db.get_marketing_search_demand_snapshot(
        product_keyword=product_keyword,
        days=days,
        limit=limit,
    )


@function_tool
def get_marketing_action_matrix(
    product_keyword: Optional[str] = None,
    days: int = 7,
    limit: int = 20,
) -> Dict[str, Any]:
    """Вернуть действия: масштабировать, оставить, сократить, остановить или проверить."""
    return strategic_db.get_marketing_action_matrix(
        product_keyword=product_keyword,
        days=days,
        limit=limit,
    )


@function_tool
def get_marketing_campaign_daily_mart(
    product_keyword: Optional[str] = None,
    days: int = 7,
    limit: int = 20,
) -> Dict[str, Any]:
    """Вернуть campaign-level mart со статусом качества и readiness materialized view."""
    return strategic_db.get_marketing_campaign_daily_mart(
        product_keyword=product_keyword,
        days=days,
        limit=limit,
    )


@function_tool
def get_marketing_query_bands_mart(
    product_keyword: Optional[str] = None,
    limit: int = 30,
) -> Dict[str, Any]:
    """Вернуть query bands mart: winner, traffic_only и waste по нормализованным запросам."""
    return strategic_db.get_marketing_query_bands_mart(
        product_keyword=product_keyword,
        limit=limit,
    )


@function_tool
def get_marketing_forecast_ready_mart(
    product_keyword: Optional[str] = None,
    limit: int = 20,
) -> Dict[str, Any]:
    """Вернуть финальный marketing decision mart с decision_hint, confidence и safe_growth_pct."""
    return strategic_db.get_marketing_forecast_ready_mart(
        product_keyword=product_keyword,
        limit=limit,
    )


@function_tool
def get_marketing_query_demand_daily_mart(
    product_keyword: Optional[str] = None,
    limit: int = 30,
) -> Dict[str, Any]:
    """Вернуть спрос по запросам из materialized view; если не populated, source_status это покажет."""
    return strategic_db.get_marketing_query_demand_daily_mart(
        product_keyword=product_keyword,
        limit=limit,
    )


@function_tool
def get_marketing_query_share_daily_mart(
    product_keyword: Optional[str] = None,
    limit: int = 30,
) -> Dict[str, Any]:
    """Вернуть query share mart; если не populated, source_status это покажет."""
    return strategic_db.get_marketing_query_share_daily_mart(
        product_keyword=product_keyword,
        limit=limit,
    )


@function_tool
def get_marketing_query_forecast_ready_mart(
    product_keyword: Optional[str] = None,
    limit: int = 30,
) -> Dict[str, Any]:
    """Вернуть query-level marketing forecast mart; если не populated, source_status это покажет."""
    return strategic_db.get_marketing_query_forecast_ready_mart(
        product_keyword=product_keyword,
        limit=limit,
    )


@function_tool
def get_marketing_campaign_demand_forecast_mart(
    product_keyword: Optional[str] = None,
    limit: int = 20,
) -> Dict[str, Any]:
    """Вернуть campaign demand forecast mart; если не populated, source_status это покажет."""
    return strategic_db.get_marketing_campaign_demand_forecast_mart(
        product_keyword=product_keyword,
        limit=limit,
    )


@function_tool
def get_marketing_deep_diagnostic(
    product_keyword: Optional[str] = None,
    days: int = 7,
    limit: int = 20,
) -> Dict[str, Any]:
    """Вернуть общий маркетинговый пакет: кампании, запросы, ставки, видимость, спрос и действия."""
    return strategic_db.get_marketing_deep_diagnostic(
        product_keyword=product_keyword,
        days=days,
        limit=limit,
    )


@function_tool
def get_finance_overview(
    product_keyword: Optional[str] = None,
    limit: Annotated[int, Field(ge=1, le=100)] = 100,
) -> Dict[str, Any]:
    """Вернуть финансовые сигналы по всей матрице; COGS требует строк finance.wb_product_coast."""
    bounded_limit = max(1, min(int(limit), 100))
    return strategic_db.get_finance_overview(
        product_keyword=product_keyword,
        limit=bounded_limit,
    )


@function_tool
def get_finance_formula_catalog() -> Dict[str, Any]:
    """Вернуть справочник производных признаков finance для проверки гипотез."""
    return _get_formula_catalog(QueryIntent.FINANCE)
