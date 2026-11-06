from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any


SALES_UNITS_PCT_REVENUE_PCT_LABEL_SWAP = (
    "sales_units_pct_revenue_pct_label_swap"
)
CONTRIBUTION_MARGIN_AS_NET_PROFIT = "contribution_margin_as_net_profit"
NO_DATA_METRIC_STATED_AS_ZERO = "no_data_metric_stated_as_zero"
CONTRADICTORY_SKU_ACTIONS = "contradictory_sku_actions"
AVAILABLE_METRIC_STATED_UNAVAILABLE = "available_metric_stated_unavailable"
STOCK_VALUE_USES_SOLD_UNITS = "stock_value_uses_sold_units"
INCOMPLETE_TARGET_COVERAGE = "incomplete_target_coverage"
MISSING_REQUIRED_METRICS = "missing_required_metrics"
PERIOD_CONFLICT = "period_conflict"

PROMPT_V2_VIOLATION_ORDER = (
    SALES_UNITS_PCT_REVENUE_PCT_LABEL_SWAP,
    CONTRIBUTION_MARGIN_AS_NET_PROFIT,
    NO_DATA_METRIC_STATED_AS_ZERO,
    AVAILABLE_METRIC_STATED_UNAVAILABLE,
    STOCK_VALUE_USES_SOLD_UNITS,
    CONTRADICTORY_SKU_ACTIONS,
    INCOMPLETE_TARGET_COVERAGE,
    MISSING_REQUIRED_METRICS,
    PERIOD_CONFLICT,
)

MAX_REPAIR_PROMPT_CHARS = 12_000

_UNAVAILABLE_STATUSES = {
    "accepted_limitation",
    "missing",
    "no_data",
    "not_available",
    "not_checked",
    "unavailable",
    "unconfirmed",
    "unknown",
}
_AVAILABLE_STATUSES = {"available", "confirmed", "ok", "partial", "resolved"}
_STATUS_KEYS = ("evidence_status", "outcome", "status")
_METRIC_KEYS = ("metric", "metric_name", "name", "code", "label")
_VALUE_KEYS = (
    "value",
    "metric_value",
    "current_value",
    "delta_pct",
    "change_pct",
    "pct_change",
)
_SKU_KEYS = ("sku", "vendor_code", "vendorCode", "артикул")
_KNOWN_DOMAINS = {"finance", "marketing", "price", "sales", "supply"}

_NUMBER_PATTERN = re.compile(
    r"[-+−]?(?:\d{1,3}(?:[\s\u00a0\u202f]\d{3})+|\d+)(?:[.,]\d+)?"
)
_PERCENT_PATTERN = re.compile(
    r"(?P<number>[-+−]?(?:\d{1,3}(?:[\s\u00a0\u202f]\d{3})+|\d+)(?:[.,]\d+)?)\s*%"
)
_ZERO_PATTERN = re.compile(r"(?<![\d.,])0(?:[.,]0+)?(?!\d)")
_EXPLICIT_SKU_PATTERN = re.compile(
    r"\b(?:sku|артикул)\s*(?:#|№|:|-)?\s*([a-zа-я0-9][a-zа-я0-9._/-]{1,63})",
    re.IGNORECASE,
)

_REVENUE_MARKERS = ("revenue", "выруч", "оборот", "продаж в руб", "sales rub")
_SALES_MARKERS = ("sales", "продаж", "реализац", "заказ")
_UNIT_MARKERS = ("unit", "quantity", " qty", "шт", "единиц", "количеств")
_PCT_MARKERS = ("pct", "percent", "%", "change", "delta", "динамик", "измен")
_CONTRIBUTION_MARKERS = (
    "contribution_margin",
    "contribution margin",
    "маржинальн",
    "маржа",
)
_NET_PROFIT_MARKERS = (
    "net_profit",
    "net profit",
    "чистая прибыль",
    "прибыль после всех расходов",
    "итоговая прибыль",
)
_NET_PROFIT_TERM_PATTERN = re.compile(
    r"(?:net[_ ]profit|чист\w*\s+прибыл\w*)",
    re.IGNORECASE,
)
_CONTRIBUTION_TERM_RE = (
    r"(?:contribution[_ ]margin|маржинальн\w*(?:\s+прибыл\w*)?|марж\w*)"
)

_METRIC_ALIASES = {
    "finance": ("марж", "прибыл", "себестоим", "расход", "финанс"),
    "marketing": ("реклам", "дрр", "drr", "расход", "клик", "показ"),
    "price": ("цен", "price", "скид"),
    "sales": ("продаж", "заказ", "выруч", "sales", "revenue", "unit"),
    "supply": (
        "остат",
        "запас",
        "склад",
        "постав",
        "пополн",
        "дозаказ",
        "stock",
        "inventory",
        "coverage",
    ),
    "stock": ("остат", "запас", "склад", "stock", "inventory"),
    "revenue": ("выруч", "оборот", "revenue"),
    "sales_units": ("продаж", "заказ", "шт", "единиц", "sales", "unit"),
}

_STOP_REORDER_PATTERN = re.compile(
    r"(?:"
    r"(?:приостанов|останов|замороз|отмен|прекрат|сократ)\w*[^\n|.;]{0,40}"
    r"(?:дозаказ|пополн|постав|закуп)"
    r"|не\s+(?:дозаказ\w*|заказ\w*|пополн\w*|постав\w*|закуп\w*)"
    r")",
    re.IGNORECASE,
)
_MAINTAIN_REORDER_PATTERN = re.compile(
    r"(?:"
    r"(?:продолж|поддерж|сохран|возобнов|увелич|удерж)\w*[^\n|.;]{0,40}"
    r"(?:дозаказ|пополн|постав|закуп)"
    r"|(?:дозаказать|пополнить|возобновить\s+постав|увеличить\s+постав)\w*"
    r")",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _MetricValue:
    kind: str
    value: float
    sku: str | None = None


@dataclass(frozen=True)
class _NoDataMetric:
    sku: str
    metric: str
    domain: str | None = None


def _normalize(value: Any) -> str:
    return re.sub(
        r"\s+",
        " ",
        str(value or "")
        .lower()
        .replace("ё", "е")
        .replace("\u2212", "-")
        .replace("\u00a0", " ")
        .replace("\u202f", " "),
    ).strip()


def _status(node: Mapping[str, Any]) -> str:
    for key in _STATUS_KEYS:
        if key in node and node[key] is not None:
            return _normalize(node[key])
    return ""


def _node_sku(node: Mapping[str, Any], inherited: str | None = None) -> str | None:
    for key in _SKU_KEYS:
        value = node.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return inherited


def _node_metric(node: Mapping[str, Any], fallback: str = "") -> str:
    for key in _METRIC_KEYS:
        value = node.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return fallback


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = _NUMBER_PATTERN.fullmatch(str(value).strip())
    if not match:
        return None
    normalized = match.group(0).replace("\u2212", "-")
    normalized = re.sub(r"[\s\u00a0\u202f]", "", normalized).replace(",", ".")
    try:
        return float(normalized)
    except ValueError:
        return None


def _metric_kind(label: str) -> str | None:
    text = _normalize(label)
    if any(marker in text for marker in _NET_PROFIT_MARKERS):
        return "net_profit"
    if any(marker in text for marker in _CONTRIBUTION_MARKERS):
        return "contribution_margin"
    has_pct = any(marker in text for marker in _PCT_MARKERS)
    if has_pct and any(marker in text for marker in _REVENUE_MARKERS):
        return "revenue_pct"
    if has_pct and any(marker in text for marker in _SALES_MARKERS):
        if any(marker in text for marker in _UNIT_MARKERS) or "revenue" not in text:
            return "sales_units_pct"
    return None


def _iter_metric_values(
    value: Any,
    *,
    path: tuple[str, ...] = (),
    inherited_sku: str | None = None,
) -> Iterable[_MetricValue]:
    if isinstance(value, Mapping):
        sku = _node_sku(value, inherited_sku)
        status = _status(value)
        unavailable = status in _UNAVAILABLE_STATUSES
        metric_label = _node_metric(value)

        if not unavailable:
            for key, candidate in value.items():
                number = _as_float(candidate)
                if number is None:
                    continue
                label = " ".join((*path[-2:], metric_label, str(key)))
                kind = _metric_kind(label)
                if kind:
                    yield _MetricValue(kind=kind, value=number, sku=sku)

        columns = value.get("columns")
        rows = value.get("rows")
        if isinstance(columns, list) and isinstance(rows, list):
            for row in rows:
                if isinstance(row, list):
                    row = {str(column): cell for column, cell in zip(columns, row)}
                yield from _iter_metric_values(
                    row,
                    path=(*path, "table_row"),
                    inherited_sku=sku,
                )

        for key, child in value.items():
            if key in {"columns", "rows"}:
                continue
            if isinstance(child, (Mapping, list, tuple)):
                yield from _iter_metric_values(
                    child,
                    path=(*path, str(key)),
                    inherited_sku=sku,
                )
        return

    if isinstance(value, (list, tuple)):
        for child in value:
            yield from _iter_metric_values(
                child,
                path=path,
                inherited_sku=inherited_sku,
            )


def _text_segments(markdown: str) -> list[str]:
    segments: list[str] = []
    table_header: str | None = None
    for line in str(markdown or "").splitlines():
        stripped = line.strip()
        if not stripped:
            table_header = None
            continue
        if stripped.startswith("|") and stripped.endswith("|"):
            if table_header is None:
                table_header = stripped
                continue
            if re.fullmatch(r"\|[\s:|-]+\|", stripped):
                continue
            stripped = f"{table_header} {stripped}"
        else:
            table_header = None
        parts = re.split(
            r"(?<=[.!?;])\s+|,\s*(?=(?:а\s+)?(?:выруч|оборот|продаж|sales|revenue))",
            stripped,
            flags=re.IGNORECASE,
        )
        segments.extend(part for part in parts if part.strip())
    return segments


def _segment_pct_kind(segment: str) -> str | None:
    text = _normalize(segment)
    if any(marker in text for marker in _REVENUE_MARKERS):
        return "revenue_pct"
    if any(marker in text for marker in _SALES_MARKERS):
        return "sales_units_pct"
    return None


def _percent_values(segment: str) -> list[float]:
    text = _normalize(segment)
    values: list[float] = []
    for match in _PERCENT_PATTERN.finditer(segment):
        value = _as_float(match.group("number"))
        if value is None:
            continue
        prefix = _normalize(segment[max(0, match.start() - 80) : match.start()])
        if value >= 0 and any(
            marker in prefix
            for marker in ("сниз", "паден", "сократ", "уменьш", "просел", "declin", "decreas")
        ):
            value = -value
        elif value <= 0 and any(
            marker in prefix
            for marker in ("вырос", "рост", "увелич", "rise", "growth", "increas")
        ):
            value = abs(value)
        values.append(value)
    return values


def _same_metric_value(left: float, right: float) -> bool:
    return abs(left - right) <= max(0.05, abs(right) * 0.0005)


def _has_sales_revenue_label_swap(
    markdown: str,
    metrics: list[_MetricValue],
) -> bool:
    sales_values = [item.value for item in metrics if item.kind == "sales_units_pct"]
    revenue_values = [item.value for item in metrics if item.kind == "revenue_pct"]
    if not sales_values or not revenue_values:
        return False

    for segment in _text_segments(markdown):
        kind = _segment_pct_kind(segment)
        if kind not in {"sales_units_pct", "revenue_pct"}:
            continue
        values = _percent_values(segment)
        if kind == "sales_units_pct":
            expected, swapped = sales_values, revenue_values
        else:
            expected, swapped = revenue_values, sales_values
        for value in values:
            matches_expected = any(_same_metric_value(value, item) for item in expected)
            matches_swapped = any(_same_metric_value(value, item) for item in swapped)
            if matches_swapped and not matches_expected:
                return True
    return False


def _segment_skus(segment: str, known_skus: set[str]) -> set[str]:
    found = {match.group(1) for match in _EXPLICIT_SKU_PATTERN.finditer(segment)}
    for sku in known_skus:
        if re.search(rf"(?<![\w]){re.escape(sku)}(?![\w])", segment, re.IGNORECASE):
            found.add(sku)
    return {str(item).strip() for item in found if str(item).strip()}


def _has_net_profit_evidence(metrics: list[_MetricValue], sku: str | None) -> bool:
    net_profit = [item for item in metrics if item.kind == "net_profit"]
    if sku:
        return any(item.sku in {None, sku} for item in net_profit)
    return bool(net_profit)


def _net_profit_mention_is_disclaimed(
    text: str,
    mention: re.Match[str],
) -> bool:
    prefix = text[max(0, mention.start() - 240) : mention.start()]
    suffix = text[mention.end() : mention.end() + 80]
    clause = r"[^\n.!?;]"
    relation_tail = (
        r"(?:\s+(?:именно|подтвержденн\w*|фактическ\w*|полн\w*|реальн\w*|"
        r"показател\w*|эквивалент\w*|расчет\w*))*\s*$"
    )

    if re.search(
        rf"{_CONTRIBUTION_TERM_RE}{clause}{{0,120}}(?:"
        rf"не\s+(?:явля\w*|равн\w*|означа\w*|счита\w*)"
        rf"|нельзя\s+(?:счита\w*|называ\w*|трактова\w*|выдава\w*)"
        rf"|(?:это\s+)?не"
        rf"){relation_tail}",
        prefix,
    ):
        return True

    if re.search(
        rf"(?:не\s+|нельзя\s+|не\s+следует\s+|некорректно\s+|ошибочно\s+)"
        rf"(?:счита\w*|называ\w*|трактова\w*|выдава\w*)"
        rf"{clause}{{0,100}}{_CONTRIBUTION_TERM_RE}{relation_tail}",
        prefix,
    ):
        return True

    if re.search(
        rf"(?:нет|отсутств\w*|недоступ\w*|не\s+(?:хвата\w*|подтвержд\w*|рассчит\w*))"
        rf"{clause}{{0,120}}(?:p&l|данн\w*|сло\w*|расчет\w*|метрик\w*)"
        rf"{clause}{{0,80}}(?:для|чтобы|котор\w*\s+бы|необходим\w*\s+для)"
        rf"{clause}{{0,60}}$",
        prefix,
    ):
        return True

    unavailable = (
        r"(?:не\s+(?:рассчит\w*|подтвержд\w*|доступ\w*)"
        r"|нет\s+данн\w*|отсутств\w*|недоступ\w*|no_data)"
    )
    return bool(
        re.search(rf"{unavailable}\s*$", prefix)
        or re.search(rf"^{clause}{{0,50}}{unavailable}", suffix)
    )


def _contribution_margin_as_profit(
    markdown: str,
    metrics: list[_MetricValue],
) -> bool:
    contribution = [item for item in metrics if item.kind == "contribution_margin"]
    if not contribution:
        return False
    known_skus = {item.sku for item in contribution if item.sku}

    for segment in _text_segments(markdown):
        text = _normalize(segment)
        definite_net_claim = any(
            not _net_profit_mention_is_disclaimed(text, mention)
            for mention in _NET_PROFIT_TERM_PATTERN.finditer(text)
        )
        generic_profit_claim = "прибыл" in text and not any(
            marker in text
            for marker in (
                "маржинальн",
                "валов",
                "contribution",
                "чистая прибыль отсутств",
                "чистая прибыль не рассчит",
            )
        )
        if not definite_net_claim and not generic_profit_claim:
            continue

        segment_skus = _segment_skus(segment, known_skus)
        scoped_sku = next(iter(segment_skus), None) if len(segment_skus) == 1 else None
        if _has_net_profit_evidence(metrics, scoped_sku):
            continue

        relevant = [
            item
            for item in contribution
            if not scoped_sku or item.sku in {None, scoped_sku}
        ]
        if definite_net_claim and relevant:
            return True

        numbers = [number for number in (_as_float(match.group(0)) for match in _NUMBER_PATTERN.finditer(segment)) if number is not None]
        if any(
            _same_metric_value(number, item.value)
            for number in numbers
            for item in relevant
        ):
            return True
    return False


def _domain_from_path(path: tuple[str, ...]) -> str | None:
    for part in reversed(path):
        normalized = _normalize(part)
        if normalized in _KNOWN_DOMAINS:
            return normalized
    return None


def _iter_no_data_metrics(
    value: Any,
    *,
    path: tuple[str, ...] = (),
    inherited_sku: str | None = None,
) -> Iterable[_NoDataMetric]:
    if isinstance(value, Mapping):
        sku = _node_sku(value, inherited_sku)
        status = _status(value)
        reason = _normalize(value.get("reason"))
        unavailable = status in _UNAVAILABLE_STATUSES or reason in {
            "metric_absent",
            "sku_not_in_source",
        }
        domain = _domain_from_path(path)
        metric = _node_metric(value, fallback=domain or "")
        if unavailable and sku and (metric or domain):
            yield _NoDataMetric(sku=sku, metric=metric or domain or "", domain=domain)

        for key, child in value.items():
            if isinstance(child, (Mapping, list, tuple)):
                yield from _iter_no_data_metrics(
                    child,
                    path=(*path, str(key)),
                    inherited_sku=sku,
                )
        return

    if isinstance(value, (list, tuple)):
        for child in value:
            yield from _iter_no_data_metrics(
                child,
                path=path,
                inherited_sku=inherited_sku,
            )


def _metric_aliases(metric: str, domain: str | None) -> tuple[str, ...]:
    normalized = _normalize(metric)
    aliases: list[str] = []
    if any(marker in normalized for marker in ("stock", "inventory", "остат", "запас")):
        aliases.extend(_METRIC_ALIASES["stock"])
    if any(marker in normalized for marker in ("revenue", "выруч", "оборот")):
        aliases.extend(_METRIC_ALIASES["revenue"])
    if any(marker in normalized for marker in ("sales", "unit", "продаж", "заказ")):
        aliases.extend(_METRIC_ALIASES["sales_units"])
    if domain in _METRIC_ALIASES:
        aliases.extend(_METRIC_ALIASES[domain])
    if len(normalized) >= 4:
        aliases.append(normalized)
    return tuple(dict.fromkeys(aliases))


def _segment_states_zero(segment: str, sku: str) -> bool:
    without_sku = re.sub(
        rf"(?<![\w]){re.escape(sku)}(?![\w])",
        " ",
        _normalize(segment),
        flags=re.IGNORECASE,
    )
    if _ZERO_PATTERN.search(without_sku):
        return True
    return any(
        marker in without_sku
        for marker in (
            "нет остат",
            "остатки отсутств",
            "продаж нет",
            "выручки нет",
            "нулевой остат",
            "нулевая выруч",
        )
    )


def _no_data_stated_as_zero(markdown: str, package: Mapping[str, Any]) -> bool:
    refs = list(dict.fromkeys(_iter_no_data_metrics(package)))
    if not refs:
        return False
    known_skus = {ref.sku for ref in refs}
    for segment in _text_segments(markdown):
        segment_skus = _segment_skus(segment, known_skus)
        if not segment_skus:
            continue
        text = _normalize(segment)
        for ref in refs:
            if ref.sku not in segment_skus or not _segment_states_zero(segment, ref.sku):
                continue
            aliases = _metric_aliases(ref.metric, ref.domain)
            if aliases and any(alias in text for alias in aliases):
                return True
    return False


def _known_skus(package: Mapping[str, Any]) -> set[str]:
    skus: set[str] = set()

    def visit(value: Any, inherited_sku: str | None = None) -> None:
        if isinstance(value, Mapping):
            sku = _node_sku(value, inherited_sku)
            if sku:
                skus.add(sku)
            for child in value.values():
                if isinstance(child, (Mapping, list, tuple)):
                    visit(child, sku)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child, inherited_sku)

    visit(package)
    return skus


def _has_contradictory_sku_actions(markdown: str, package: Mapping[str, Any]) -> bool:
    known_skus = _known_skus(package)
    actions: dict[str, set[str]] = {}
    active_skus: set[str] = set()
    for line in str(markdown or "").splitlines():
        if not line.strip():
            active_skus = set()
            continue
        line_skus = _segment_skus(line, known_skus)
        if line_skus:
            active_skus = line_skus
        if not active_skus:
            continue
        action_kinds: set[str] = set()
        if _STOP_REORDER_PATTERN.search(line):
            action_kinds.add("stop")
        if _MAINTAIN_REORDER_PATTERN.search(line):
            action_kinds.add("maintain")
        for sku in active_skus:
            actions.setdefault(sku, set()).update(action_kinds)
    return any({"stop", "maintain"}.issubset(kinds) for kinds in actions.values())


def _has_confirmed_metric(package: Mapping[str, Any], markers: tuple[str, ...]) -> bool:
    def visit(value: Any) -> bool:
        if isinstance(value, Mapping):
            metric = _normalize(_node_metric(value))
            status = _status(value)
            if (
                metric
                and status in _AVAILABLE_STATUSES
                and any(marker in metric for marker in markers)
                and any(value.get(key) is not None for key in _VALUE_KEYS)
            ):
                return True
            return any(visit(child) for child in value.values() if isinstance(child, (Mapping, list, tuple)))
        if isinstance(value, (list, tuple)):
            return any(visit(child) for child in value)
        return False

    return visit(package)


def _available_turnover_stated_unavailable(markdown: str, package: Mapping[str, Any]) -> bool:
    if not _has_confirmed_metric(
        package,
        ("inventory_days", "days_on_hand", "turnover", "оборачиваем"),
    ):
        return False
    text = _normalize(markdown)
    unavailable = r"(?:нельзя|невозможно|не удалось|не рассчит|нет данных|отсутств)"
    return bool(
        re.search(rf"оборачиваем\w*.{{0,100}}{unavailable}", text)
        or re.search(rf"{unavailable}.{{0,100}}оборачиваем\w*", text)
    )


def _stock_value_uses_sold_units(markdown: str, package: Mapping[str, Any]) -> bool:
    if not _has_confirmed_metric(
        package,
        ("current_stock_value", "stock_value", "inventory_value", "стоимост", "деньги в остат"),
    ):
        return False
    text = _normalize(markdown)
    claims_stock_money = any(
        marker in text
        for marker in ("деньги в остат", "стоимость остат", "остатки в деньгах", "stock value")
    ) or bool(re.search(r"деньги\s+в\s+(?:текущ\w*\s+)?остат", text))
    sold_cost_formula = bool(
        re.search(r"(?:себестоим|cost).{0,50}(?:продан|реализован|sold)", text)
        or re.search(r"(?:продан|реализован|sold).{0,50}(?:себестоим|cost)", text)
    )
    return claims_stock_money and sold_cost_formula


def _target_id(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for key in ("target", "target_id", "id", "sku", "vendor_code", "entity_id"):
            candidate = value.get(key)
            if candidate is not None and str(candidate).strip():
                return str(candidate).strip()
        return None
    if value is None or not str(value).strip():
        return None
    return str(value).strip()


def _target_set(value: Any) -> set[str]:
    if isinstance(value, (list, tuple, set)):
        return {target for item in value if (target := _target_id(item))}
    target = _target_id(value)
    return {target} if target else set()


def _coverage_is_incomplete(package: Mapping[str, Any]) -> bool:
    coverage = package.get("request_coverage")
    coverage = coverage if isinstance(coverage, Mapping) else {}
    requested = _target_set(
        coverage.get("requested_targets")
        or coverage.get("requested")
        or package.get("requested_targets")
    )
    covered = _target_set(
        coverage.get("covered_targets")
        or coverage.get("resolved_targets")
        or package.get("covered_targets")
        or package.get("resolved_targets")
    )
    missing = _target_set(coverage.get("missing_targets") or package.get("missing_targets"))
    if missing:
        return True
    return bool(requested and requested - covered)


def _flatten_required_metrics(value: Any) -> set[str]:
    result: set[str] = set()
    if isinstance(value, str):
        normalized = _normalize(value)
        if normalized:
            result.add(normalized)
    elif isinstance(value, Mapping):
        metric = _node_metric(value)
        if metric:
            result.add(_normalize(metric))
        else:
            for child in value.values():
                result.update(_flatten_required_metrics(child))
    elif isinstance(value, (list, tuple, set)):
        for child in value:
            result.update(_flatten_required_metrics(child))
    return result


def _confirmed_metric_names(value: Any) -> set[str]:
    names: set[str] = set()
    if isinstance(value, Mapping):
        metric = _node_metric(value)
        status = _status(value)
        if metric and status in _AVAILABLE_STATUSES:
            names.add(_normalize(metric))
        for child in value.values():
            if isinstance(child, (Mapping, list, tuple)):
                names.update(_confirmed_metric_names(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            names.update(_confirmed_metric_names(child))
    return names


def _required_metrics_are_missing(package: Mapping[str, Any]) -> bool:
    required = _flatten_required_metrics(package.get("required_metrics"))
    if not required:
        return False
    confirmed = _confirmed_metric_names(
        package.get("canonical_facts")
        or package.get("confirmed_facts")
        or package
    )
    return bool(required - confirmed)


def _has_period_conflict(package: Mapping[str, Any]) -> bool:
    conflicts = package.get("period_conflicts")
    if isinstance(conflicts, (list, tuple, set, Mapping)) and bool(conflicts):
        return True
    return package.get("period_compatible") is False


def validate_evidence_contract(
    evidence_package: Mapping[str, Any] | None,
) -> list[str]:
    """Validate explicit request/evidence invariants without inferring missing schema.

    The checks activate only when callers provide the corresponding contract
    fields, so legacy lookup/report payloads remain backwards compatible.
    """

    package = evidence_package if isinstance(evidence_package, Mapping) else {}
    violations: list[str] = []
    if _coverage_is_incomplete(package):
        violations.append(INCOMPLETE_TARGET_COVERAGE)
    if _required_metrics_are_missing(package):
        violations.append(MISSING_REQUIRED_METRICS)
    if _has_period_conflict(package):
        violations.append(PERIOD_CONFLICT)
    return violations


def validate_prompt_v2_answer(
    markdown: str,
    evidence_package: Mapping[str, Any] | None,
) -> list[str]:
    """Return deterministic, stable violation codes for a synthesized answer.

    The validator is intentionally pure: it only compares the draft with its
    compact evidence package and never calls a model, database, or tool.
    """

    package = evidence_package if isinstance(evidence_package, Mapping) else {}
    metrics = list(dict.fromkeys(_iter_metric_values(package)))
    found: set[str] = set()

    if _has_sales_revenue_label_swap(markdown, metrics):
        found.add(SALES_UNITS_PCT_REVENUE_PCT_LABEL_SWAP)
    if _contribution_margin_as_profit(markdown, metrics):
        found.add(CONTRIBUTION_MARGIN_AS_NET_PROFIT)
    if _no_data_stated_as_zero(markdown, package):
        found.add(NO_DATA_METRIC_STATED_AS_ZERO)
    if _available_turnover_stated_unavailable(markdown, package):
        found.add(AVAILABLE_METRIC_STATED_UNAVAILABLE)
    if _stock_value_uses_sold_units(markdown, package):
        found.add(STOCK_VALUE_USES_SOLD_UNITS)
    if _has_contradictory_sku_actions(markdown, package):
        found.add(CONTRADICTORY_SKU_ACTIONS)
    found.update(validate_evidence_contract(package))

    return [code for code in PROMPT_V2_VIOLATION_ORDER if code in found]


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except (TypeError, ValueError):
        if isinstance(value, Mapping):
            return {str(key): _json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_json_safe(item) for item in value]
        return str(value)


def _canonical_repair_facts(package: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "query_mode",
        "original_user_query",
        "canonical_facts",
        "confirmed_facts",
        "refuted_facts",
        "contradictions",
        "blocking_gaps",
        "sku_metric_matrix",
        "sku_evidence_matrix",
        "sku_matrices",
        "action_safety_checks",
        "safety_statuses",
    )
    selected = {key: package[key] for key in keys if key in package}
    if selected:
        return _json_safe(selected)
    return {"evidence": _json_safe(package)}


def _clip(value: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(value) <= limit:
        return value
    marker = "\n...[truncated]"
    if limit <= len(marker):
        return value[:limit]
    return value[: limit - len(marker)] + marker


def build_prompt_v2_repair_prompt(
    draft_markdown: str,
    violations: Iterable[str],
    evidence_package: Mapping[str, Any] | None,
    *,
    max_chars: int = MAX_REPAIR_PROMPT_CHARS,
) -> str:
    """Build one bounded repair request from a draft and canonical evidence."""

    limit = max(1, min(int(max_chars), MAX_REPAIR_PROMPT_CHARS))
    package = evidence_package if isinstance(evidence_package, Mapping) else {}
    codes = list(dict.fromkeys(str(item) for item in violations if str(item).strip()))
    violations_json = json.dumps(codes, ensure_ascii=False, separators=(",", ":"))
    facts_json = json.dumps(
        _canonical_repair_facts(package),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    prefix = (
        "Исправь черновик, не добавляя новых фактов. Используй только канонические факты ниже. "
        "Устрани все коды нарушений; если данных нет, явно напиши об ограничении вместо нуля.\n"
        f"VIOLATIONS:\n{violations_json}\n"
        "CANONICAL_FACTS:\n"
    )
    middle = "\nDRAFT:\n"
    facts_budget = max(0, int((limit - len(prefix) - len(middle)) * 0.55))
    clipped_facts = _clip(facts_json, facts_budget)
    draft_budget = max(0, limit - len(prefix) - len(clipped_facts) - len(middle))
    clipped_draft = _clip(str(draft_markdown or ""), draft_budget)
    prompt = f"{prefix}{clipped_facts}{middle}{clipped_draft}"
    return prompt[:limit]


# Descriptive aliases for callers that do not need the workflow-version name.
validate_final_answer_evidence = validate_prompt_v2_answer
build_repair_prompt = build_prompt_v2_repair_prompt


__all__ = [
    "AVAILABLE_METRIC_STATED_UNAVAILABLE",
    "CONTRADICTORY_SKU_ACTIONS",
    "CONTRIBUTION_MARGIN_AS_NET_PROFIT",
    "INCOMPLETE_TARGET_COVERAGE",
    "MAX_REPAIR_PROMPT_CHARS",
    "NO_DATA_METRIC_STATED_AS_ZERO",
    "MISSING_REQUIRED_METRICS",
    "PERIOD_CONFLICT",
    "PROMPT_V2_VIOLATION_ORDER",
    "SALES_UNITS_PCT_REVENUE_PCT_LABEL_SWAP",
    "STOCK_VALUE_USES_SOLD_UNITS",
    "build_prompt_v2_repair_prompt",
    "build_repair_prompt",
    "validate_final_answer_evidence",
    "validate_evidence_contract",
    "validate_prompt_v2_answer",
]
