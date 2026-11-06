from __future__ import annotations

import re

from agi_ecom_app.schemas import QueryIntent

_STRATEGIC_PATTERNS = [
    r"\bмаркетинг\w*\b",
    r"\bреклам\w*\b",
    r"\bкампан\w*\b",
    r"\badvert_id\b",
    r"\bctr\b",
    r"\bcpc\b",
    r"\bcr\b",
    r"\bпоказ\w*\b",
    r"\bклик\w*\b",
    r"\bконверси\w*\b",
    r"\bпоиск\w*\b",
    r"\bзапрос\w*\b",
    r"\bвидимост\w*\b",
    r"\bставк\w*\b",
    r"\bsku в кампани\w*\b",
    r"\bдоля затрат\b",
    r"\bдрр\b",
    r"\bддр\b",
]

_SALES_PATTERNS = [
    r"\bпродаж\w*\b",
    r"\bзаказ\w*\b",
    r"\bвыручк\w*\b",
    r"\bвыкуп\w*\b",
    r"\bотмен\w*\b",
    r"\bворонк\w*\b",
    r"\bартикул\w*\b",
    r"\bsku\b",
    r"\bnm[_\s-]?id\b",
    r"\bсклад\w*\b",
    r"\bрегион\w*\b",
    r"\bкарточк\w*\b",
    r"\bоткрыти\w*\s+карточк\w*\b",
    r"\bкорзин\w*\b",
    r"\bsales\b",
]

_SALES_REPORT_COMPARISON_PATTERNS = [
    r"вчера\s+к\s+позавчера",
    r"недел\w*\s+к\s+недел\w*",
    r"месяц\w*\s+к\s+месяц\w*",
    r"\bweek\s+over\s+week\b",
    r"\bmonth\s+over\s+month\b",
    r"\bwow\b",
    r"\bmom\b",
]

_PRICE_PATTERNS = [
    r"\bцен[аыеуой]*\b",
    r"\bценов\w*\b",
    r"\bпрайс\w*\b",
    r"\bскидк\w*\b",
    r"\bдисконт\w*\b",
    r"\bдороже\b",
    r"\bдешевле\b",
    r"\bконкурент\w*\b",
    r"\bпереоценк\w*\b",
    r"\bуценк\w*\b",
    r"\bсегмент\w*\s+цен\w*\b",
    r"\bprice\b",
    r"\bdiscount\b",
]

_MARKETING_PRICE_PATTERNS = [
    r"\bцен[ауыой]*\s+клик\w*\b",
    r"\bcpc\b",
    r"\bставк\w*\b",
]

_SUPPLY_PATTERNS = [
    r"\bостат\w*\b",
    r"\bоборачиваем\w*\b",
    r"\bзапас\w*\b",
    r"\bскладск\w*\b",
    r"\bденьг\w*.*\bсклад\w*\b",
    r"\bдерж\w*.*\bденьг\w*.*\bсклад\w*\b",
    r"\bпоставк\w*\b",
    r"\bпоставщ\w*\b",
    r"\bналичи\w*\b",
    r"\bдефицит\w*\b",
    r"\bперезаказ\w*\b",
    r"\bдозаказ\w*\b",
    r"\bотгрузк\w*\b",
    r"\bпри[её]мк\w*\b",
    r"\bв\s+пути\b",
    r"\bout[\s-]?of[\s-]?stock\b",
    r"\boos\b",
    r"\bstock\b",
    r"\breorder\b",
    r"\binbound\b",
]

_FACTUAL_HINTS = [
    r"\bфин\b",
    r"\bфин\.\b",
    r"\bфинанс\w*\b",
    r"\bэкономик\w*\b",
    r"\bкомисси\w*\b",
    r"\bэквайринг\b",
    r"\bлогистик\w*\b",
    r"\bхранени\w*\b",
    r"\bкросс[\s-]?докинг\b",
    r"\bштраф\w*\b",
    r"\bудержани\w*\b",
    r"\bначислен\w*\b",
    r"\bк оплате\b",
    r"\bрасход\w*\b",
    r"\bмаржин\w*\b",
    r"\bприбыл\w*\b",
    r"\bрентаб\w*\b",
    r"\bpayout\b",
]

_BROAD_DIAGNOSTIC_PATTERNS = [
    r"\bпочему\b",
    r"\bпричин\w*\b",
    r"\bгипотез\w*\b",
    r"\bпроверь\b",
    r"\bразлож\w*\b",
    r"\bвклад\w*\b",
    r"\bизменил\w*\b",
    r"\bизменени\w*\b",
    r"\bподтвержден\w*\b",
    r"\bподтвержд[её]н\w*\b",
    r"\bнеподтвержден\w*\b",
    r"\bнеподтвержд[её]н\w*\b",
    r"\bфактор\w*\b",
]

_SKU_DIAGNOSTIC_PATTERNS = [
    r"\bsku\b",
    r"\bnm[_\s-]?id\b",
    r"\bартикул\w*\b",
    r"\bключев\w*\b",
    r"\bтовар\w*\b",
]

_FINANCE_FACTUAL_HINTS = [
    r"\bфин[\s-]?отч[её]т\b",
    r"\bмп\b",
    r"\bwb report\b",
]

_PERIOD_HINTS = [
    r"\bза\b",
    r"\bпериод\b",
    r"\bс\b",
    r"\bпо\b",
    r"\bпредыдущ\w*\b",
    r"\bпрошл\w*\b",
    r"\bнедел\w*\b",
    r"\bмесяц\w*\b",
    r"\bотч[её]т\b",
    r"\d{1,2}[./-]\d{1,2}(?:[./-]\d{2,4})?",
]


def determine_query_intent(user_input: str) -> QueryIntent:
    text = user_input.lower().strip()
    if not text:
        return QueryIntent.FINANCE

    marketing_hits = sum(bool(re.search(pattern, text)) for pattern in _STRATEGIC_PATTERNS)
    sales_hits = sum(bool(re.search(pattern, text)) for pattern in _SALES_PATTERNS)
    price_hits = sum(bool(re.search(pattern, text)) for pattern in _PRICE_PATTERNS)
    marketing_price_hits = sum(
        bool(re.search(pattern, text)) for pattern in _MARKETING_PRICE_PATTERNS
    )
    supply_hits = sum(bool(re.search(pattern, text)) for pattern in _SUPPLY_PATTERNS)
    finance_hits = (
        sum(bool(re.search(pattern, text)) for pattern in _FACTUAL_HINTS)
        + sum(bool(re.search(pattern, text)) for pattern in _FINANCE_FACTUAL_HINTS)
    )
    period_hits = sum(bool(re.search(pattern, text)) for pattern in _PERIOD_HINTS)
    sales_report_comparison_hits = sum(
        bool(re.search(pattern, text)) for pattern in _SALES_REPORT_COMPARISON_PATTERNS
    )

    if _is_broad_sales_profit_diagnostic(
        text=text,
        sales_hits=sales_hits,
        finance_hits=finance_hits,
        marketing_hits=marketing_hits,
        price_hits=price_hits,
        supply_hits=supply_hits,
    ):
        return QueryIntent.FINANCE

    if supply_hits and supply_hits >= sales_hits and supply_hits >= marketing_hits:
        return QueryIntent.SUPPLY

    if marketing_hits and sales_hits and re.search(r"\bклик\w*\b|\bтрафик\w*\b", text):
        return QueryIntent.MARKETING

    if marketing_hits and marketing_price_hits:
        return QueryIntent.MARKETING

    if price_hits and price_hits >= marketing_hits:
        return QueryIntent.PRICE

    if sales_hits > finance_hits and sales_hits >= marketing_hits:
        return QueryIntent.SALES

    if sales_report_comparison_hits and not any(
        [marketing_hits, price_hits, supply_hits, finance_hits]
    ):
        return QueryIntent.SALES

    if marketing_hits > finance_hits:
        return QueryIntent.MARKETING

    if finance_hits > marketing_hits:
        return QueryIntent.FINANCE

    if marketing_hits and period_hits:
        return QueryIntent.MARKETING

    return QueryIntent.FINANCE


def _is_broad_sales_profit_diagnostic(
    *,
    text: str,
    sales_hits: int,
    finance_hits: int,
    marketing_hits: int,
    price_hits: int,
    supply_hits: int,
) -> bool:
    if not sales_hits or not finance_hits:
        return False

    has_diagnostic_word = any(re.search(pattern, text) for pattern in _BROAD_DIAGNOSTIC_PATTERNS)
    has_sku_scope = any(re.search(pattern, text) for pattern in _SKU_DIAGNOSTIC_PATTERNS)
    has_cross_domain_scope = (
        marketing_hits > 0
        or price_hits > 0
        or supply_hits > 0
        or all(word in text for word in ("спрос", "реклама", "цена", "остат"))
    )
    return has_diagnostic_word and (has_sku_scope or has_cross_domain_scope)


def has_domain_routing_signal(user_input: str) -> bool:
    text = user_input.lower().strip()
    if not text:
        return False

    patterns = (
        *_STRATEGIC_PATTERNS,
        *_SALES_PATTERNS,
        *_SALES_REPORT_COMPARISON_PATTERNS,
        *_PRICE_PATTERNS,
        *_SUPPLY_PATTERNS,
        *_FACTUAL_HINTS,
        *_FINANCE_FACTUAL_HINTS,
    )
    return any(re.search(pattern, text) for pattern in patterns)
