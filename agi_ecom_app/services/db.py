from __future__ import annotations

import atexit
import os
import re
from contextlib import AbstractContextManager
from datetime import date, datetime, timedelta
from dataclasses import dataclass
from decimal import Decimal
from threading import Lock
from typing import Any

import psycopg
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row

from agi_ecom_app.config import get_settings

SAFE_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
SCHEMA_REF_PATTERN = re.compile(
    r'(?i)\b(?:from|join)\s+("?([A-Za-z_][A-Za-z0-9_]*)"?)\s*\.\s*"?([A-Za-z_][A-Za-z0-9_]*)"?'
)
FORBIDDEN_SQL = (
    "insert ",
    "update ",
    "delete ",
    "alter ",
    "drop ",
    "truncate ",
    "create ",
    "grant ",
    "revoke ",
    "copy ",
)


_POSTGRES_POOL: ConnectionPool | None = None
_POSTGRES_POOL_CONFIG: tuple[str, int, int, float] | None = None
_POSTGRES_POOL_PID: int | None = None
_POSTGRES_POOL_LOCK = Lock()


def _postgres_pool_config() -> tuple[str, int, int, float]:
    settings = get_settings()
    min_size = max(1, int(getattr(settings, "postgres_pool_min_size", 1)))
    max_size = max(
        min_size,
        int(getattr(settings, "postgres_pool_max_size", 10)),
    )
    timeout = max(
        1.0,
        float(getattr(settings, "postgres_pool_timeout_seconds", 10.0)),
    )
    return settings.postgres_dsn, min_size, max_size, timeout


def _discard_inherited_postgres_pool() -> None:
    """Forget parent-owned connections after fork without touching their sockets."""

    global _POSTGRES_POOL, _POSTGRES_POOL_CONFIG, _POSTGRES_POOL_PID
    global _POSTGRES_POOL_LOCK
    _POSTGRES_POOL = None
    _POSTGRES_POOL_CONFIG = None
    _POSTGRES_POOL_PID = None
    # A lock held by another thread at fork remains locked forever in the child.
    _POSTGRES_POOL_LOCK = Lock()


def _get_postgres_pool() -> ConnectionPool:
    global _POSTGRES_POOL, _POSTGRES_POOL_CONFIG, _POSTGRES_POOL_PID
    config = _postgres_pool_config()
    pid = os.getpid()
    if (
        _POSTGRES_POOL is not None
        and _POSTGRES_POOL_CONFIG == config
        and _POSTGRES_POOL_PID == pid
    ):
        return _POSTGRES_POOL

    # `register_at_fork()` covers normal forks. Keep this fallback for runtimes
    # that forked before importing this module or do not expose that hook.
    if _POSTGRES_POOL is not None and _POSTGRES_POOL_PID != pid:
        _discard_inherited_postgres_pool()

    with _POSTGRES_POOL_LOCK:
        if (
            _POSTGRES_POOL is not None
            and _POSTGRES_POOL_CONFIG == config
            and _POSTGRES_POOL_PID == pid
        ):
            return _POSTGRES_POOL
        if _POSTGRES_POOL is not None:
            _POSTGRES_POOL.close()
        dsn, min_size, max_size, timeout = config
        _POSTGRES_POOL = ConnectionPool(
            conninfo=dsn,
            min_size=min_size,
            max_size=max_size,
            timeout=timeout,
            kwargs={"row_factory": dict_row},
            open=True,
            name="agi-ecom-analytics",
        )
        _POSTGRES_POOL_CONFIG = config
        _POSTGRES_POOL_PID = pid
        return _POSTGRES_POOL


def close_postgres_connection_pool() -> None:
    """Close the analytics pool owned by the current process, if it exists."""

    global _POSTGRES_POOL, _POSTGRES_POOL_CONFIG, _POSTGRES_POOL_PID
    pid = os.getpid()
    with _POSTGRES_POOL_LOCK:
        pool = _POSTGRES_POOL
        owner_pid = _POSTGRES_POOL_PID
        _POSTGRES_POOL = None
        _POSTGRES_POOL_CONFIG = None
        _POSTGRES_POOL_PID = None
        if pool is not None and owner_pid == pid:
            pool.close()


def get_connection() -> AbstractContextManager[psycopg.Connection]:
    """Borrow a transaction-scoped analytics connection from the local pool."""

    return _get_postgres_pool().connection()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_discard_inherited_postgres_pool)

atexit.register(close_postgres_connection_pool)


def ping_database() -> None:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("select 1 as ok")
        cur.fetchone()


def _validate_identifier(name: str) -> str:
    if not SAFE_IDENTIFIER.match(name):
        raise ValueError(f"Invalid SQL identifier: {name}")
    return name


def _normalize_allowed_schemas(allowed_schemas: list[str] | tuple[str, ...] | set[str]) -> list[str]:
    normalized = sorted({_validate_identifier(schema) for schema in allowed_schemas})
    if not normalized:
        raise ValueError("allowed_schemas must not be empty")
    return normalized


def _latest_full_day_expr(column_sql: str) -> str:
    return (
        f"coalesce("
        f"max({column_sql}) filter (where {column_sql} < current_date), "
        f"max({column_sql})"
        f")"
    )


def _parse_report_date(value: str | None) -> date | None:
    if value is None:
        return None

    cleaned = value.strip()
    if not cleaned:
        return None

    today = date.today()
    formats = (
        ("%Y-%m-%d", False),
        ("%Y/%m/%d", False),
        ("%d.%m.%Y", False),
        ("%d.%m.%y", False),
        ("%d-%m-%Y", False),
        ("%d-%m-%y", False),
        ("%d.%m", True),
        ("%d-%m", True),
    )

    for fmt, inject_current_year in formats:
        try:
            parsed = datetime.strptime(cleaned, fmt).date()
            if inject_current_year:
                return parsed.replace(year=today.year)
            return parsed
        except ValueError:
            continue

    raise ValueError(
        "Invalid date format. Use YYYY-MM-DD, DD.MM.YYYY, or DD.MM."
    )


def _parse_report_period(
    date_from: str,
    date_to: str | None = None,
) -> tuple[date | None, date | None]:
    if date_to is not None:
        return _parse_report_date(date_from), _parse_report_date(date_to)

    cleaned = date_from.strip()
    if not cleaned:
        return None, None

    date_token = r"(?:\d{4}[/-]\d{2}[/-]\d{2}|\d{1,2}[./-]\d{1,2}(?:[./-]\d{2,4})?)"
    patterns = (
        r"^\s*с\s+(.+?)\s+по\s+(.+?)\s*$",
        r"^\s*(.+?)\s*[–—]\s*(.+?)\s*$",
        rf"^\s*({date_token})\s*-\s*({date_token})\s*$",
    )

    for pattern in patterns:
        match = re.match(pattern, cleaned, flags=re.IGNORECASE)
        if not match:
            continue
        left, right = match.groups()
        return _parse_report_date(left), _parse_report_date(right)

    parsed = _parse_report_date(cleaned)
    return parsed, parsed


def validate_readonly_query(query: str) -> str:
    cleaned = query.strip()
    lowered = cleaned.lower()

    if not lowered.startswith(("select", "with")):
        raise ValueError("Only SELECT and WITH queries are allowed.")

    if any(token in lowered for token in FORBIDDEN_SQL):
        raise ValueError("Query contains forbidden SQL statements.")

    return cleaned.rstrip(";")


def serialize_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, list):
        return [serialize_value(item) for item in value]
    if isinstance(value, dict):
        return {key: serialize_value(item) for key, item in value.items()}
    return value


def serialize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [serialize_value(row) for row in rows]


def list_tables(schema: str | None = None) -> list[dict[str, Any]]:
    query = """
        select table_schema, table_name, table_type
        from information_schema.tables
        where table_schema not in ('pg_catalog', 'information_schema')
    """
    params: list[Any] = []
    if schema:
        query += " and table_schema = %s"
        params.append(schema)

    query += " order by table_schema, table_name"

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(query, params)
        return serialize_rows(cur.fetchall())


def list_tables_in_schemas(allowed_schemas: list[str] | tuple[str, ...] | set[str]) -> list[dict[str, Any]]:
    schemas = _normalize_allowed_schemas(allowed_schemas)
    query = """
        select table_schema, table_name, table_type
        from information_schema.tables
        where table_schema = any(%s)
        order by table_schema, table_name
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(query, [schemas])
        return serialize_rows(cur.fetchall())


def describe_table(table_name: str, schema: str = "public") -> list[dict[str, Any]]:
    query = """
        select
            column_name,
            data_type,
            is_nullable,
            column_default
        from information_schema.columns
        where table_schema = %s and table_name = %s
        order by ordinal_position
    """

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(query, [schema, table_name])
        return serialize_rows(cur.fetchall())


def describe_table_in_schemas(
    table_name: str,
    schema: str,
    allowed_schemas: list[str] | tuple[str, ...] | set[str],
) -> list[dict[str, Any]]:
    schemas = _normalize_allowed_schemas(allowed_schemas)
    if schema not in schemas:
        raise ValueError(f"Schema {schema} is not allowed.")
    return describe_table(table_name=table_name, schema=schema)


def search_columns(pattern: str, schema: str | None = None) -> list[dict[str, Any]]:
    query = """
        select table_schema, table_name, column_name, data_type
        from information_schema.columns
        where table_schema not in ('pg_catalog', 'information_schema')
          and column_name ilike %s
    """
    params: list[Any] = [f"%{pattern}%"]

    if schema:
        query += " and table_schema = %s"
        params.append(schema)

    query += " order by table_schema, table_name, ordinal_position"

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(query, params)
        return serialize_rows(cur.fetchall())


def search_columns_in_schemas(
    pattern: str,
    allowed_schemas: list[str] | tuple[str, ...] | set[str],
    schema: str | None = None,
) -> list[dict[str, Any]]:
    schemas = _normalize_allowed_schemas(allowed_schemas)
    if schema and schema not in schemas:
        raise ValueError(f"Schema {schema} is not allowed.")

    query = """
        select table_schema, table_name, column_name, data_type
        from information_schema.columns
        where table_schema = any(%s)
          and column_name ilike %s
    """
    params: list[Any] = [schemas, f"%{pattern}%"]
    if schema:
        query += " and table_schema = %s"
        params.append(schema)
    query += " order by table_schema, table_name, ordinal_position"

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(query, params)
        return serialize_rows(cur.fetchall())


def sample_rows(table_name: str, schema: str = "public", limit: int = 5) -> list[dict[str, Any]]:
    if not 1 <= limit <= 50:
        raise ValueError("limit must be between 1 and 50")

    safe_schema = _validate_identifier(schema)
    safe_table = _validate_identifier(table_name)
    query = f'SELECT * FROM "{safe_schema}"."{safe_table}" LIMIT %s'

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(query, [limit])
        return serialize_rows(cur.fetchall())


def sample_rows_in_schemas(
    table_name: str,
    schema: str,
    allowed_schemas: list[str] | tuple[str, ...] | set[str],
    limit: int = 5,
) -> list[dict[str, Any]]:
    schemas = _normalize_allowed_schemas(allowed_schemas)
    if schema not in schemas:
        raise ValueError(f"Schema {schema} is not allowed.")
    return sample_rows(table_name=table_name, schema=schema, limit=limit)


def run_readonly_sql(query: str, limit: int = 200) -> list[dict[str, Any]]:
    if not 1 <= limit <= 500:
        raise ValueError("limit must be between 1 and 500")

    safe_query = validate_readonly_query(query)
    wrapped_query = f"select * from ({safe_query}) as subq limit %s"

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(wrapped_query, [limit])
        return serialize_rows(cur.fetchall())


def _extract_explicit_schemas(query: str) -> set[str]:
    return {match.group(2) for match in SCHEMA_REF_PATTERN.finditer(query)}


def _validate_scoped_query(
    query: str,
    allowed_schemas: list[str] | tuple[str, ...] | set[str],
) -> tuple[str, list[str]]:
    safe_query = validate_readonly_query(query)
    schemas = _normalize_allowed_schemas(allowed_schemas)
    explicit_schemas = _extract_explicit_schemas(safe_query)

    disallowed = sorted(schema for schema in explicit_schemas if schema not in schemas)
    if disallowed:
        raise ValueError(
            "Query references schemas outside the allowed role scope: " + ", ".join(disallowed)
        )

    return safe_query, schemas


def run_scoped_readonly_sql(
    query: str,
    allowed_schemas: list[str] | tuple[str, ...] | set[str],
    limit: int = 200,
) -> list[dict[str, Any]]:
    if not 1 <= limit <= 500:
        raise ValueError("limit must be between 1 and 500")

    safe_query, schemas = _validate_scoped_query(query, allowed_schemas)
    wrapped_query = f"select * from ({safe_query}) as subq limit %s"
    search_path = ", ".join(f'"{schema}"' for schema in schemas)

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(f"set local search_path to {search_path}")
        cur.execute(wrapped_query, [limit])
        return serialize_rows(cur.fetchall())


def get_database_time() -> dict[str, Any]:
    query = "select current_date as db_current_date, now() as db_now"
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(query)
        row = cur.fetchone()
        return serialize_value(row)


def get_sales_summary(
    date_from: str,
    date_to: str | None = None,
    product_keyword: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")

    params: list[Any] = [date_from, date_to or date_from]
    keyword_filter = ""

    if product_keyword:
        keyword_filter = """
            and (
                coalesce(subject, '') ilike %s
                or coalesce(category, '') ilike %s
                or coalesce(brand, '') ilike %s
                or coalesce(supplier_article, '') ilike %s
            )
        """
        like = f"%{product_keyword}%"
        params.extend([like, like, like, like])

    summary_query = f"""
        with sales as (
            select
                date_ord::date as sale_date,
                nm_id,
                supplier_article,
                subject,
                category,
                brand,
                count(*) filter (where is_realization) as sales_count,
                sum(case when is_realization then coalesce(for_pay, 0) else 0 end) as revenue
            from sales.wb_supplier_sales
            where date_ord::date between %s and %s
            {keyword_filter}
            group by date_ord::date, nm_id, supplier_article, subject, category, brand
        )
        select *
        from sales
        where sales_count > 0 or revenue > 0
        order by revenue desc, sales_count desc, sale_date desc
        limit %s
    """
    summary_params = [*params, limit]

    totals_query = f"""
        select
            count(*) filter (where is_realization) as total_sales,
            sum(case when is_realization then coalesce(for_pay, 0) else 0 end) as total_revenue,
            max(date_ord::date) as latest_sale_date
        from sales.wb_supplier_sales
        where date_ord::date between %s and %s
        {keyword_filter}
    """

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(summary_query, summary_params)
        rows = serialize_rows(cur.fetchall())

        cur.execute(totals_query, params)
        totals = serialize_value(cur.fetchone())

    return {
        "date_from": date_from,
        "date_to": date_to or date_from,
        "product_keyword": product_keyword,
        "totals": totals,
        "rows": rows,
    }


def get_sales_report_text(
    date_from: str,
    date_to: str | None = None,
    product_keyword: str | None = None,
    top_n: int = 10,
) -> str:
    summary = get_sales_summary(
        date_from=date_from,
        date_to=date_to,
        product_keyword=product_keyword,
        limit=top_n,
    )

    totals = summary["totals"]
    rows = summary["rows"]
    date_to_value = summary["date_to"]

    header = [
        f"Период: {summary['date_from']} - {date_to_value}",
        f"Фильтр: {product_keyword or 'без фильтра'}",
        f"Всего продаж: {int(totals['total_sales'] or 0)}",
        f"Сумма for_pay: {float(totals['total_revenue'] or 0):.2f}",
        f"Последняя доступная дата в выборке: {totals['latest_sale_date']}",
    ]

    if not rows:
        return "\n".join(header + ["Топ товаров: данных нет"])

    lines = header + ["Топ товаров:"]
    for row in rows[:top_n]:
        lines.append(
            "- "
            f"{row['supplier_article']} | subject={row['subject']} | "
            f"sales_count={row['sales_count']} | for_pay_sum={row['revenue']}"
        )

    return "\n".join(lines)


def get_finance_marketplace_report(
    date_from: str,
    date_to: str | None = None,
) -> dict[str, Any]:
    start_date, end_date = _parse_report_period(date_from=date_from, date_to=date_to)

    if start_date is None or end_date is None:
        raise ValueError("date_from must not be empty")
    if end_date < start_date:
        raise ValueError("date_to must be greater than or equal to date_from")

    query = """
        with report_rows as (
            select
                rr_dt,
                nullif(sa_name, '') as vendor_code,
                nm_id,
                doc_type_name,
                coalesce(quantity, 0) as quantity,
                coalesce(retail_price_withdisc_rub, 0) as retail_price_withdisc_rub,
                coalesce(ppvz_for_pay, 0) as ppvz_for_pay,
                coalesce(delivery_rub, 0) as delivery_rub,
                coalesce(acceptance, 0) as acceptance,
                coalesce(storage_fee, 0) as storage_fee,
                coalesce(rebill_logistic_cost, 0) as rebill_logistic_cost,
                coalesce(penalty, 0) as penalty,
                coalesce(deduction, 0) as deduction,
                supplier_oper_name
            from finance.wb_report_detail_daily
            where rr_dt between %s and %s
        ),
        costed_rows as (
            select
                r.*,
                cost.cost_price_rub
            from report_rows r
            left join lateral (
                select c.cost_price_rub
                from finance.wb_product_coast c
                where (
                    (r.nm_id is not null and c.nm_id = r.nm_id)
                    or (r.vendor_code is not null and c.vendor_code = r.vendor_code)
                )
                order by
                    case when r.nm_id is not null and c.nm_id = r.nm_id then 0 else 1 end,
                    (c.upload_date <= %s) desc,
                    case when c.upload_date <= %s then c.upload_date end desc nulls last,
                    c.upload_date desc,
                    c.updated_at desc
                limit 1
            ) cost on true
        ),
        sales as (
            select
                coalesce(sum(retail_price_withdisc_rub) filter (
                    where doc_type_name = 'Продажа'
                ), 0) as revenue,
                abs(coalesce(sum(retail_price_withdisc_rub) filter (
                    where doc_type_name = 'Возврат'
                ), 0)) as returns,
                coalesce(sum(ppvz_for_pay) filter (
                    where doc_type_name = 'Продажа'
                ), 0) as payout_sales
            from report_rows
        ),
        cogs as (
            select
                coalesce(sum(cost_price_rub * quantity) filter (
                    where doc_type_name = 'Продажа' and cost_price_rub is not null
                ), 0) as sales_cogs,
                coalesce(sum(cost_price_rub * abs(quantity)) filter (
                    where doc_type_name = 'Возврат' and cost_price_rub is not null
                ), 0) as return_cogs,
                count(*) filter (
                    where doc_type_name in ('Продажа', 'Возврат')
                        and cost_price_rub is null
                        and (
                            quantity <> 0
                            or retail_price_withdisc_rub <> 0
                        )
                ) as missing_cost_row_count,
                coalesce(sum(quantity) filter (
                    where doc_type_name = 'Продажа'
                        and cost_price_rub is null
                        and (
                            quantity <> 0
                            or retail_price_withdisc_rub <> 0
                        )
                ), 0) as missing_cost_sales_quantity,
                coalesce(sum(abs(quantity)) filter (
                    where doc_type_name = 'Возврат'
                        and cost_price_rub is null
                        and (
                            quantity <> 0
                            or retail_price_withdisc_rub <> 0
                        )
                ), 0) as missing_cost_return_quantity
            from costed_rows
        ),
        ops as (
            select
                coalesce(sum(deduction) filter (
                    where supplier_oper_name = 'Удержание'
                ), 0) as marketing,
                coalesce(sum(delivery_rub) filter (
                    where supplier_oper_name = 'Логистика'
                ), 0) as logistics,
                coalesce(sum(acceptance), 0) as acceptance,
                coalesce(sum(storage_fee) filter (
                    where supplier_oper_name = 'Хранение'
                ), 0) as storage_fee,
                coalesce(sum(rebill_logistic_cost) filter (
                    where supplier_oper_name = 'Возмещение издержек по перевозке/по складским операциям с товаром'
                ), 0) as cross_docking,
                coalesce(sum(penalty + deduction) filter (
                    where supplier_oper_name not in ('Удержание')
                ), 0) as penalties_and_withholdings,
                coalesce(sum(ppvz_for_pay) filter (
                    where supplier_oper_name = 'Добровольная компенсация при возврате'
                ), 0) as misc_accruals
            from report_rows
        ),
        calc as (
            select
                s.revenue,
                s.returns,
                (s.revenue - s.payout_sales) as commission_and_acquiring,
                o.marketing,
                o.logistics,
                o.acceptance,
                o.storage_fee,
                o.cross_docking,
                o.penalties_and_withholdings,
                o.misc_accruals,
                (
                    (s.revenue - s.payout_sales)
                    + o.marketing
                    + o.logistics
                    + o.acceptance
                    + o.storage_fee
                    + o.cross_docking
                    + o.penalties_and_withholdings
                ) as total_marketplace_expenses,
                c.sales_cogs,
                c.return_cogs,
                (c.sales_cogs - c.return_cogs) as net_cogs,
                c.missing_cost_row_count,
                c.missing_cost_sales_quantity,
                c.missing_cost_return_quantity
            from sales s
            cross join ops o
            cross join cogs c
        )
        select
            round(revenue::numeric, 2) as revenue,
            round(returns::numeric, 2) as returns,
            round(sales_cogs::numeric, 2) as sales_cogs,
            round(return_cogs::numeric, 2) as return_cogs,
            round(net_cogs::numeric, 2) as net_cogs,
            round(total_marketplace_expenses::numeric, 2) as total_marketplace_expenses,
            round(commission_and_acquiring::numeric, 2) as commission_and_acquiring,
            round(marketing::numeric, 2) as marketing,
            round(logistics::numeric, 2) as logistics,
            round(acceptance::numeric, 2) as acceptance,
            round(storage_fee::numeric, 2) as storage_fee,
            round(cross_docking::numeric, 2) as cross_docking,
            round(penalties_and_withholdings::numeric, 2) as penalties_and_withholdings,
            round(misc_accruals::numeric, 2) as misc_accruals,
            round((revenue - returns - total_marketplace_expenses + misc_accruals)::numeric, 2) as total_to_pay,
            round((
                revenue
                - returns
                - total_marketplace_expenses
                + misc_accruals
                - net_cogs
            )::numeric, 2) as margin_rub,
            case
                when (revenue - returns) <> 0
                    then round(
                        100 * (
                            revenue
                            - returns
                            - total_marketplace_expenses
                            + misc_accruals
                            - net_cogs
                        ) / (revenue - returns),
                        2
                    )
            end as margin_percent,
            missing_cost_row_count,
            missing_cost_sales_quantity,
            missing_cost_return_quantity
        from calc
    """

    params = [start_date, end_date, end_date, end_date]

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(query, params)
        row = serialize_value(cur.fetchone() or {})

    metrics = [
        ("ИТОГО К ОПЛАТЕ", row.get("total_to_pay", 0.0)),
        ("Выручка", row.get("revenue", 0.0)),
        ("Возвраты", row.get("returns", 0.0)),
        ("Себестоимость продаж", row.get("sales_cogs", 0.0)),
        ("Себестоимость возвратов", row.get("return_cogs", 0.0)),
        ("Себестоимость нетто", row.get("net_cogs", 0.0)),
        ("ИТОГО РАСХОДЫ МП", row.get("total_marketplace_expenses", 0.0)),
        ("КОМИССИЯ и ЭКВАЙРИНГ", row.get("commission_and_acquiring", 0.0)),
        ("МАРКЕТИНГ", row.get("marketing", 0.0)),
        ("ЛОГИСТИКА", row.get("logistics", 0.0)),
        ("Приемка", row.get("acceptance", 0.0)),
        ("Хранение", row.get("storage_fee", 0.0)),
        ("Кросс-докинг", row.get("cross_docking", 0.0)),
        ("ШТРАФЫ и УДЕРЖАНИЯ", row.get("penalties_and_withholdings", 0.0)),
        ("ПРОЧИЕ НАЧИСЛЕНИЯ", row.get("misc_accruals", 0.0)),
        ("МАРЖА", row.get("margin_rub", 0.0)),
    ]

    return {
        "date_from": start_date.isoformat(),
        "date_to": end_date.isoformat(),
        "source_table": "finance.wb_report_detail_daily",
        "metrics": [
            {
                "label": label,
                "value": float(value or 0.0),
                **(
                    {"percent": float(row["margin_percent"])}
                    if label == "МАРЖА" and row.get("margin_percent") is not None
                    else {}
                ),
            }
            for label, value in metrics
        ],
        "totals": row,
    }


def get_finance_marketplace_report_text(
    date_from: str,
    date_to: str | None = None,
) -> str:
    summary = get_finance_marketplace_report(date_from=date_from, date_to=date_to)
    start_date = datetime.strptime(summary["date_from"], "%Y-%m-%d").date()
    end_date = datetime.strptime(summary["date_to"], "%Y-%m-%d").date()
    period_label = f"{start_date.strftime('%d.%m')}–{end_date.strftime('%d.%m')}"

    lines = [
        f"Период: {period_label} ({summary['date_from']} - {summary['date_to']})",
        f"Источник: {summary['source_table']}",
    ]

    for metric in summary["metrics"]:
        value_text = f"{metric['value']:,.2f} ₽".replace(",", " ")
        if metric["label"] == "МАРЖА" and metric.get("percent") is not None:
            value_text = f"{value_text} ({metric['percent']:,.2f}%)".replace(",", " ")
        lines.append(f"{metric['label']}: {value_text}")

    totals = summary.get("totals") or {}
    missing_cost_rows = int(totals.get("missing_cost_row_count") or 0)
    if missing_cost_rows:
        lines.append(
            "Пробел в данных: себестоимость не найдена для "
            f"{missing_cost_rows} строк продаж/возвратов."
        )

    return "\n".join(lines)


FINANCE_SKU_REPORT_COLUMNS = [
    "nm_id",
    "Артикул",
    "Себестоимость",
    "Доставлено, шт",
    "Выручка, руб (после возвратов)",
    "Возвраты, шт",
    "Возвраты, руб",
    "ИТОГО Расходы МП на артикул",
    "КОМИССИИ и ЭКВАЙРИНГ",
    "МАРКЕТИНГ",
    "ЛОГИСТИКА",
    "Себестоимость возвратов",
    "Доля себестоимости в выручке до возвратов",
    "ABC по выручке",
    "Маржа, руб",
    "Рентабельность, %",
]


def get_finance_sku_report(
    date_from: str,
    date_to: str | None = None,
    product_keyword: str | None = None,
    limit: int = 100,
    sort_by: str = "net_revenue",
) -> dict[str, Any]:
    if not 1 <= limit <= 500:
        raise ValueError("limit must be between 1 and 500")
    sort_options = {
        "net_revenue": "net_revenue desc",
        "margin_rub": "margin_rub desc",
        "profitability_pct": "profitability_pct desc nulls last",
    }
    if sort_by not in sort_options:
        raise ValueError("sort_by must be one of: net_revenue, margin_rub, profitability_pct")
    order_by_sql = sort_options[sort_by]

    start_date, end_date = _parse_report_period(date_from=date_from, date_to=date_to)
    if start_date is None or end_date is None:
        raise ValueError("date_from must not be empty")
    if end_date < start_date:
        raise ValueError("date_to must be greater than or equal to date_from")

    keyword_filter = ""
    params: list[Any] = [start_date, end_date]
    if product_keyword:
        keyword_filter = """
              and (
                  coalesce(nm_id::text, '') ilike %s
                  or coalesce(sa_name, '') ilike %s
                  or coalesce(brand_name, '') ilike %s
                  or coalesce(subject_name, '') ilike %s
              )
        """
        like = f"%{product_keyword}%"
        params.extend([like, like, like, like])

    query = f"""
        with report_rows as (
            select
                rr_dt,
                nullif(sa_name, '') as vendor_code,
                nm_id,
                doc_type_name,
                coalesce(quantity, 0) as quantity,
                coalesce(retail_price_withdisc_rub, 0) as retail_price_withdisc_rub,
                coalesce(ppvz_for_pay, 0) as ppvz_for_pay,
                coalesce(delivery_rub, 0) as delivery_rub,
                coalesce(acceptance, 0) as acceptance,
                coalesce(storage_fee, 0) as storage_fee,
                coalesce(rebill_logistic_cost, 0) as rebill_logistic_cost,
                coalesce(penalty, 0) as penalty,
                coalesce(deduction, 0) as deduction,
                supplier_oper_name
            from finance.wb_report_detail_daily
            where rr_dt between %s and %s
              and (nm_id is not null or nullif(sa_name, '') is not null)
              {keyword_filter}
        ),
        per_sku as (
            select
                nm_id,
                vendor_code,
                coalesce(sum(quantity) filter (
                    where doc_type_name = 'Продажа'
                ), 0) as delivered_qty,
                coalesce(sum(retail_price_withdisc_rub) filter (
                    where doc_type_name = 'Продажа'
                ), 0) as revenue_before_returns,
                abs(coalesce(sum(quantity) filter (
                    where doc_type_name = 'Возврат'
                ), 0)) as return_qty,
                abs(coalesce(sum(retail_price_withdisc_rub) filter (
                    where doc_type_name = 'Возврат'
                ), 0)) as returns_rub,
                coalesce(sum(ppvz_for_pay) filter (
                    where doc_type_name = 'Продажа'
                ), 0) as payout_sales,
                coalesce(sum(deduction) filter (
                    where supplier_oper_name = 'Удержание'
                ), 0) as marketing,
                coalesce(sum(delivery_rub) filter (
                    where supplier_oper_name = 'Логистика'
                ), 0) as logistics,
                coalesce(sum(acceptance), 0) as acceptance,
                coalesce(sum(storage_fee) filter (
                    where supplier_oper_name = 'Хранение'
                ), 0) as storage_fee,
                coalesce(sum(rebill_logistic_cost) filter (
                    where supplier_oper_name = 'Возмещение издержек по перевозке/по складским операциям с товаром'
                ), 0) as cross_docking,
                coalesce(sum(penalty + deduction) filter (
                    where supplier_oper_name not in ('Удержание')
                ), 0) as penalties_and_withholdings
            from report_rows
            group by nm_id, vendor_code
        ),
        costed as (
            select
                p.*,
                cost.cost_price_rub
            from per_sku p
            left join lateral (
                select c.cost_price_rub
                from finance.wb_product_coast c
                where (
                    (p.nm_id is not null and c.nm_id = p.nm_id)
                    or (p.vendor_code is not null and c.vendor_code = p.vendor_code)
                )
                order by
                    case when p.nm_id is not null and c.nm_id = p.nm_id then 0 else 1 end,
                    (c.upload_date <= %s) desc,
                    case when c.upload_date <= %s then c.upload_date end desc nulls last,
                    c.upload_date desc,
                    c.updated_at desc
                limit 1
            ) cost on true
        ),
        calc as (
            select
                nm_id,
                vendor_code,
                cost_price_rub,
                delivered_qty,
                revenue_before_returns,
                (revenue_before_returns - returns_rub) as net_revenue,
                return_qty,
                returns_rub,
                (revenue_before_returns - payout_sales) as commission_and_acquiring,
                marketing,
                logistics,
                acceptance,
                storage_fee,
                cross_docking,
                penalties_and_withholdings,
                (
                    (revenue_before_returns - payout_sales)
                    + marketing
                    + logistics
                    + acceptance
                    + storage_fee
                    + cross_docking
                    + penalties_and_withholdings
                ) as total_marketplace_expenses,
                coalesce(cost_price_rub * delivered_qty, 0) as sales_cogs,
                coalesce(cost_price_rub * return_qty, 0) as return_cogs,
                cost_price_rub is null and (delivered_qty <> 0 or return_qty <> 0) as missing_cost
            from costed
        ),
        margin_calc as (
            select
                *,
                (sales_cogs - return_cogs) as net_cogs,
                (
                    net_revenue
                    - total_marketplace_expenses
                    - (sales_cogs - return_cogs)
                ) as margin_rub,
                case
                    when revenue_before_returns <> 0
                        then round(100 * sales_cogs / revenue_before_returns, 2)
                end as cost_share_before_returns_pct
            from calc
        ),
        abc_calc as (
            select
                *,
                sum(net_revenue) over () as total_net_revenue,
                sum(net_revenue) over (
                    order by net_revenue desc, nm_id nulls last, vendor_code nulls last
                    rows between unbounded preceding and current row
                ) as cumulative_net_revenue
            from margin_calc
        ),
        classified as (
            select
                *,
                case
                    when total_net_revenue <= 0 then 'C'
                    when (cumulative_net_revenue - net_revenue) < total_net_revenue * 0.80 then 'A'
                    when (cumulative_net_revenue - net_revenue) < total_net_revenue * 0.95 then 'B'
                    else 'C'
                end as abc_revenue_class,
                case
                    when net_revenue <> 0 then round(100 * margin_rub / net_revenue, 2)
                end as profitability_pct,
                count(*) over () as total_sku_count,
                count(*) filter (where missing_cost) over () as missing_cost_sku_count
            from abc_calc
        )
        select
            nm_id,
            vendor_code,
            round(cost_price_rub::numeric, 2) as unit_cost_rub,
            round(delivered_qty::numeric, 2) as delivered_qty,
            round(net_revenue::numeric, 2) as net_revenue,
            round(return_qty::numeric, 2) as return_qty,
            round(returns_rub::numeric, 2) as returns_rub,
            round(total_marketplace_expenses::numeric, 2) as total_marketplace_expenses,
            round(commission_and_acquiring::numeric, 2) as commission_and_acquiring,
            round(marketing::numeric, 2) as marketing,
            round(logistics::numeric, 2) as logistics,
            round(return_cogs::numeric, 2) as return_cogs,
            round(cost_share_before_returns_pct::numeric, 2) as cost_share_before_returns_pct,
            abc_revenue_class,
            round(margin_rub::numeric, 2) as margin_rub,
            profitability_pct,
            missing_cost,
            total_sku_count,
            missing_cost_sku_count
        from classified
        order by {order_by_sql}, nm_id nulls last, vendor_code nulls last
        limit %s
    """
    params.extend([end_date, end_date, limit])

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(query, params)
        rows = serialize_rows(cur.fetchall())

    return {
        "date_from": start_date.isoformat(),
        "date_to": end_date.isoformat(),
        "product_keyword": product_keyword,
        "sort_by": sort_by,
        "source_tables": ["finance.wb_report_detail_daily", "finance.wb_product_coast"],
        "columns": FINANCE_SKU_REPORT_COLUMNS,
        "rows": [
            {
                "nm_id": row.get("nm_id"),
                "Артикул": row.get("vendor_code"),
                "Себестоимость": row.get("unit_cost_rub"),
                "Доставлено, шт": row.get("delivered_qty"),
                "Выручка, руб (после возвратов)": row.get("net_revenue"),
                "Возвраты, шт": row.get("return_qty"),
                "Возвраты, руб": row.get("returns_rub"),
                "ИТОГО Расходы МП на артикул": row.get("total_marketplace_expenses"),
                "КОМИССИИ и ЭКВАЙРИНГ": row.get("commission_and_acquiring"),
                "МАРКЕТИНГ": row.get("marketing"),
                "ЛОГИСТИКА": row.get("logistics"),
                "Себестоимость возвратов": row.get("return_cogs"),
                "Доля себестоимости в выручке до возвратов": row.get(
                    "cost_share_before_returns_pct"
                ),
                "ABC по выручке": row.get("abc_revenue_class"),
                "Маржа, руб": row.get("margin_rub"),
                "Рентабельность, %": row.get("profitability_pct"),
            }
            for row in rows
        ],
        "totals": {
            "total_sku_count": int(rows[0].get("total_sku_count") or 0) if rows else 0,
            "returned_sku_count": len(rows),
            "missing_cost_sku_count": (
                int(rows[0].get("missing_cost_sku_count") or 0) if rows else 0
            ),
        },
    }


def _format_finance_sku_cell(column: str, value: Any) -> str:
    if value is None or value == "":
        return "—"

    if column == "ABC по выручке":
        return str(value)

    if column == "nm_id":
        return str(value)

    if isinstance(value, float):
        formatted = f"{value:,.2f}".replace(",", " ")
        return f"{formatted}%" if column.endswith("%") else formatted

    return str(value).replace("|", "/")


def get_finance_sku_report_text(
    date_from: str,
    date_to: str | None = None,
    product_keyword: str | None = None,
    limit: int = 50,
    sort_by: str = "net_revenue",
) -> str:
    report = get_finance_sku_report(
        date_from=date_from,
        date_to=date_to,
        product_keyword=product_keyword,
        limit=limit,
        sort_by=sort_by,
    )

    start_date = datetime.strptime(report["date_from"], "%Y-%m-%d").date()
    end_date = datetime.strptime(report["date_to"], "%Y-%m-%d").date()
    period_label = f"{start_date.strftime('%d.%m')}–{end_date.strftime('%d.%m')}"
    filter_label = report["product_keyword"] or "без фильтра"
    lines = [
        f"Период: {period_label} ({report['date_from']} - {report['date_to']})",
        f"Фильтр: {filter_label}",
        "Источник: " + ", ".join(report["source_tables"]),
        f"Сортировка: {report.get('sort_by', 'net_revenue')}",
        f"SKU в выборке: {report['totals']['returned_sku_count']} из {report['totals']['total_sku_count']}",
    ]

    if report["totals"]["missing_cost_sku_count"]:
        lines.append(
            "Пробел в данных: себестоимость не найдена для "
            f"{report['totals']['missing_cost_sku_count']} SKU."
        )

    columns = report["columns"]
    lines.extend(
        [
            "",
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
        ]
    )

    if not report["rows"]:
        lines.append("| " + " | ".join(["Нет данных", *["—"] * (len(columns) - 1)]) + " |")
        return "\n".join(lines)

    for row in report["rows"]:
        lines.append(
            "| "
            + " | ".join(_format_finance_sku_cell(column, row.get(column)) for column in columns)
            + " |"
        )

    return "\n".join(lines)


FINANCE_SKU_PROFIT_CHANGE_COLUMNS = [
    "nm_id",
    "Артикул",
    "Маржа тек. период, руб",
    "Маржа пред. период, руб",
    "Δ маржи, руб",
    "Δ маржи, %",
    "Выручка тек. период, руб",
    "Выручка пред. период, руб",
    "Рентабельность тек. период, %",
    "Роль",
]


def _finance_sku_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("nm_id") or ""), str(row.get("Артикул") or ""))


def _numeric_or_zero(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def _finance_delta_pct(current: float, previous: float) -> float | None:
    if previous == 0:
        return None
    return round(100 * (current - previous) / abs(previous), 2)


def get_finance_sku_profit_change_report(
    current_date_from: str,
    current_date_to: str | None = None,
    previous_date_from: str | None = None,
    previous_date_to: str | None = None,
    product_keyword: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")

    current_start, current_end = _parse_report_period(
        date_from=current_date_from,
        date_to=current_date_to,
    )
    if current_start is None or current_end is None:
        raise ValueError("current_date_from must not be empty")
    if current_end < current_start:
        raise ValueError("current_date_to must be greater than or equal to current_date_from")

    if previous_date_from:
        previous_start, previous_end = _parse_report_period(
            date_from=previous_date_from,
            date_to=previous_date_to,
        )
        if previous_start is None or previous_end is None:
            raise ValueError("previous_date_from must not be empty")
    else:
        period_days = (current_end - current_start).days + 1
        previous_end = current_start - timedelta(days=1)
        previous_start = previous_end - timedelta(days=period_days - 1)

    if previous_end < previous_start:
        raise ValueError("previous_date_to must be greater than or equal to previous_date_from")

    current_report = get_finance_sku_report(
        current_start.isoformat(),
        current_end.isoformat(),
        product_keyword=product_keyword,
        limit=500,
        sort_by="margin_rub",
    )
    previous_report = get_finance_sku_report(
        previous_start.isoformat(),
        previous_end.isoformat(),
        product_keyword=product_keyword,
        limit=500,
        sort_by="margin_rub",
    )

    current_rows = {_finance_sku_key(row): row for row in current_report["rows"]}
    previous_rows = {_finance_sku_key(row): row for row in previous_report["rows"]}
    rows: list[dict[str, Any]] = []

    for key in sorted(set(current_rows) | set(previous_rows)):
        current = current_rows.get(key, {})
        previous = previous_rows.get(key, {})
        source = current or previous
        current_margin = _numeric_or_zero(current.get("Маржа, руб"))
        previous_margin = _numeric_or_zero(previous.get("Маржа, руб"))
        delta_margin = round(current_margin - previous_margin, 2)
        if previous and not current:
            role = "lost_in_current"
        elif current and not previous:
            role = "new_in_current"
        elif delta_margin > 0:
            role = "growth_driver"
        elif delta_margin < 0:
            role = "decline_driver"
        else:
            role = "stable"

        rows.append(
            {
                "nm_id": source.get("nm_id"),
                "Артикул": source.get("Артикул"),
                "Маржа тек. период, руб": round(current_margin, 2),
                "Маржа пред. период, руб": round(previous_margin, 2),
                "Δ маржи, руб": delta_margin,
                "Δ маржи, %": _finance_delta_pct(current_margin, previous_margin),
                "Выручка тек. период, руб": current.get("Выручка, руб (после возвратов)", 0),
                "Выручка пред. период, руб": previous.get("Выручка, руб (после возвратов)", 0),
                "Рентабельность тек. период, %": current.get("Рентабельность, %"),
                "Роль": role,
            }
        )

    rows.sort(
        key=lambda row: (
            -abs(_numeric_or_zero(row.get("Δ маржи, руб"))),
            -abs(_numeric_or_zero(row.get("Маржа тек. период, руб"))),
            str(row.get("Артикул") or ""),
        )
    )

    return {
        "current_period_start": current_start.isoformat(),
        "current_period_end": current_end.isoformat(),
        "previous_period_start": previous_start.isoformat(),
        "previous_period_end": previous_end.isoformat(),
        "product_keyword": product_keyword,
        "source_tables": ["finance.wb_report_detail_daily", "finance.wb_product_coast"],
        "columns": FINANCE_SKU_PROFIT_CHANGE_COLUMNS,
        "rows": rows[:limit],
        "totals": {
            "current_total_sku_count": current_report["totals"]["total_sku_count"],
            "previous_total_sku_count": previous_report["totals"]["total_sku_count"],
            "returned_sku_count": min(limit, len(rows)),
            "missing_cost_sku_count": max(
                current_report["totals"]["missing_cost_sku_count"],
                previous_report["totals"]["missing_cost_sku_count"],
            ),
        },
    }


def get_finance_sku_profit_change_report_text(
    current_date_from: str,
    current_date_to: str | None = None,
    previous_date_from: str | None = None,
    previous_date_to: str | None = None,
    product_keyword: str | None = None,
    limit: int = 20,
) -> str:
    report = get_finance_sku_profit_change_report(
        current_date_from=current_date_from,
        current_date_to=current_date_to,
        previous_date_from=previous_date_from,
        previous_date_to=previous_date_to,
        product_keyword=product_keyword,
        limit=limit,
    )
    lines = [
        (
            "Периоды: "
            f"{report['current_period_start']} - {report['current_period_end']} "
            f"vs {report['previous_period_start']} - {report['previous_period_end']}"
        ),
        f"Фильтр: {report['product_keyword'] or 'без фильтра'}",
        "Источник: " + ", ".join(report["source_tables"]),
        (
            "SKU в выборке: "
            f"{report['totals']['returned_sku_count']} "
            f"(текущий период: {report['totals']['current_total_sku_count']}, "
            f"предыдущий период: {report['totals']['previous_total_sku_count']})"
        ),
    ]
    if report["totals"]["missing_cost_sku_count"]:
        lines.append(
            "Пробел в данных: себестоимость не найдена для "
            f"{report['totals']['missing_cost_sku_count']} SKU."
        )

    columns = report["columns"]
    lines.extend(
        [
            "",
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
        ]
    )

    if not report["rows"]:
        lines.append("| " + " | ".join(["Нет данных", *["—"] * (len(columns) - 1)]) + " |")
        return "\n".join(lines)

    for row in report["rows"]:
        lines.append(
            "| "
            + " | ".join(_format_finance_sku_cell(column, row.get(column)) for column in columns)
            + " |"
        )

    return "\n".join(lines)


def get_1pager_ai_summary(
    report_date: str | None = None,
    product_keyword: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")

    with get_connection() as conn, conn.cursor() as cur:
        if report_date is None:
            cur.execute(
                f"select {_latest_full_day_expr('report_date')} as latest_report_date "
                "from reports.v_1pager_ai"
            )
            latest = cur.fetchone()
            report_date = latest["latest_report_date"].isoformat() if latest and latest["latest_report_date"] else None

        if report_date is None:
            return {
                "report_date": None,
                "product_keyword": product_keyword,
                "rows": [],
            }

        params: list[Any] = [report_date]
        keyword_filter = ""
        if product_keyword:
            keyword_filter = """
                and (
                    coalesce(sku, '') ilike %s
                    or coalesce(product_name, '') ilike %s
                )
            """
            like = f"%{product_keyword}%"
            params.extend([like, like])

        query = f"""
            select
                report_date,
                sku,
                product_name,
                qty_on_stock,
                days_on_stock,
                orders_1d,
                sales_1d,
                sales_7d,
                avg_sales_7d,
                sales_14d,
                avg_sales_14d,
                sales_21d,
                avg_sales_21d,
                sales_31d,
                avg_sales_31d,
                stock_turnover,
                in_promo,
                promo_names,
                in_marketing_campaigns,
                marketing_campaign_count,
                marketing_spend,
                marketing_spend_7d,
                marketing_orders,
                marketing_orders_7d,
                price_before_discount,
                price_after_discount,
                price_with_spp,
                promo_nearest_end_date
            from reports.v_1pager_ai
            where report_date = %s
            {keyword_filter}
            order by coalesce(sales_7d, 0) desc, coalesce(sales_1d, 0) desc, product_name
            limit %s
        """
        params.append(limit)
        cur.execute(query, params)
        rows = serialize_rows(cur.fetchall())

    return {
        "report_date": report_date,
        "product_keyword": product_keyword,
        "rows": rows,
    }


def get_1pager_ai_report_text(
    report_date: str | None = None,
    product_keyword: str | None = None,
    top_n: int = 10,
) -> str:
    summary = get_1pager_ai_summary(
        report_date=report_date,
        product_keyword=product_keyword,
        limit=top_n,
    )

    if not summary["rows"]:
        return (
            f"Дата отчета: {summary['report_date']}\n"
            f"Фильтр: {product_keyword or 'без фильтра'}\n"
            "Данные в v_1pager_ai не найдены."
        )

    lines = [
        f"Дата отчета: {summary['report_date']}",
        f"Фильтр: {product_keyword or 'без фильтра'}",
        "Топ строк из v_1pager_ai:",
    ]
    for row in summary["rows"][:top_n]:
        lines.append(
            "- "
            f"{row['product_name']} | sku={row['sku']} | "
            f"stock={row['qty_on_stock']} | sales_1d={row['sales_1d']} | "
            f"sales_7d={row['sales_7d']} | avg_sales_7d={row['avg_sales_7d']} | "
            f"in_promo={row['in_promo']} | in_marketing={row['in_marketing_campaigns']}"
        )

    return "\n".join(lines)


def get_1pager_ai_sales_summary(
    report_date: str | None = None,
    product_keyword: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")

    with get_connection() as conn, conn.cursor() as cur:
        if report_date is None:
            cur.execute(
                f"select {_latest_full_day_expr('report_date')} as latest_report_date "
                "from reports.v_1pager_ai"
            )
            latest = cur.fetchone()
            report_date = latest["latest_report_date"].isoformat() if latest and latest["latest_report_date"] else None

        if report_date is None:
            return {
                "report_date": None,
                "product_keyword": product_keyword,
                "totals": {
                    "product_count": 0,
                    "total_sales_1d": 0,
                    "total_sales_7d": 0,
                },
                "rows": [],
            }

        params: list[Any] = [report_date]
        keyword_filter = ""
        if product_keyword:
            keyword_filter = """
                and (
                    coalesce(sku, '') ilike %s
                    or coalesce(product_name, '') ilike %s
                )
            """
            like = f"%{product_keyword}%"
            params.extend([like, like])

        rows_query = f"""
            select
                report_date,
                sku,
                product_name,
                qty_on_stock,
                sales_1d,
                sales_7d,
                avg_sales_7d,
                in_promo,
                in_marketing_campaigns
            from reports.v_1pager_ai
            where report_date = %s
            {keyword_filter}
              and coalesce(sales_1d, 0) > 0
            order by coalesce(sales_1d, 0) desc, coalesce(sales_7d, 0) desc, product_name
            limit %s
        """
        row_params = [*params, limit]

        totals_query = f"""
            select
                count(*) filter (where coalesce(sales_1d, 0) > 0) as product_count,
                sum(coalesce(sales_1d, 0)) as total_sales_1d,
                sum(coalesce(sales_7d, 0)) as total_sales_7d
            from reports.v_1pager_ai
            where report_date = %s
            {keyword_filter}
        """

        cur.execute(rows_query, row_params)
        rows = serialize_rows(cur.fetchall())

        cur.execute(totals_query, params)
        totals = serialize_value(cur.fetchone())

    return {
        "report_date": report_date,
        "product_keyword": product_keyword,
        "totals": totals,
        "rows": rows,
    }


def get_1pager_ai_sales_report_text(
    report_date: str | None = None,
    product_keyword: str | None = None,
    top_n: int = 10,
) -> str:
    summary = get_1pager_ai_sales_summary(
        report_date=report_date,
        product_keyword=product_keyword,
        limit=top_n,
    )

    totals = summary["totals"]
    rows = summary["rows"]
    lines = [
        f"Дата отчета: {summary['report_date']}",
        f"Фильтр: {product_keyword or 'без фильтра'}",
        f"Продажи за 1 день из v_1pager_ai: {int(totals['total_sales_1d'] or 0)}",
        f"Продажи за 7 дней из v_1pager_ai: {int(totals['total_sales_7d'] or 0)}",
        f"Товаров с продажами за 1 день: {int(totals['product_count'] or 0)}",
    ]

    if not rows:
        return "\n".join(lines + ["Строк с sales_1d > 0 не найдено."])

    lines.append("Топ товары по sales_1d:")
    for row in rows[:top_n]:
        lines.append(
            "- "
            f"{row['product_name']} | sku={row['sku']} | stock={row['qty_on_stock']} | "
            f"sales_1d={row['sales_1d']} | sales_7d={row['sales_7d']} | "
            f"in_promo={row['in_promo']} | in_marketing={row['in_marketing_campaigns']}"
        )

    return "\n".join(lines)


@dataclass(frozen=True)
class CliDashboardKpis:
    sales_units: int | None
    sales_rub: float | None
    orders_today: int | None
    sales_units_delta_pct: float | None
    sales_rub_delta_pct: float | None
    in_way_to_client_units: int | None
    sales_date: date | None
    prev_sales_date: date | None
    orders_date: date | None
    stocks_snapshot_hour: datetime | None
    is_partial: bool = False
    warning: str | None = None

    @classmethod
    def unavailable(cls, warning: str = "Данные KPI временно недоступны") -> "CliDashboardKpis":
        return cls(
            sales_units=None,
            sales_rub=None,
            orders_today=None,
            sales_units_delta_pct=None,
            sales_rub_delta_pct=None,
            in_way_to_client_units=None,
            sales_date=None,
            prev_sales_date=None,
            orders_date=None,
            stocks_snapshot_hour=None,
            is_partial=True,
            warning=warning,
        )


def _coerce_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _compute_delta_pct(current: int | float | None, previous: int | float | None) -> float | None:
    if current is None or previous in (None, 0):
        return None
    return round(((current - previous) / previous) * 100, 2)


def get_cli_dashboard_kpis() -> CliDashboardKpis:
    query = """
        with sales_dates as (
            select {_latest_full_day_expr('date_ord::date')} as sales_date
            from reports.wb_supplier_sales
        ),
        sales_current as (
            select
                count(*) filter (where is_realization) as sales_units,
                sum(case when is_realization then coalesce(for_pay, 0) else 0 end) as sales_rub
            from reports.wb_supplier_sales s
            join sales_dates d
              on s.date_ord::date = d.sales_date
        ),
        sales_previous as (
            select
                count(*) filter (where is_realization) as prev_sales_units,
                sum(case when is_realization then coalesce(for_pay, 0) else 0 end) as prev_sales_rub
            from reports.wb_supplier_sales s
            join sales_dates d
              on s.date_ord::date = d.sales_date - interval '1 day'
        ),
        orders_dates as (
            select {_latest_full_day_expr('snapshot_date')} as orders_date
            from reports.wb_supplier_orders
        ),
        orders_current as (
            select
                count(*) filter (where coalesce(is_cancel, false) = false) as orders_today
            from reports.wb_supplier_orders o
            join orders_dates d
              on o.snapshot_date = d.orders_date
        ),
        stocks_dates as (
            select coalesce(
                max(snapshot_hour) filter (where snapshot_hour::date < current_date),
                max(snapshot_hour)
            ) as stocks_snapshot_hour
            from reports.wb_supplier_stocks
        ),
        stocks_current as (
            select
                sum(coalesce(in_way_to_client, 0)) as in_way_to_client_units
            from reports.wb_supplier_stocks s
            join stocks_dates d
              on s.snapshot_hour = d.stocks_snapshot_hour
        )
        select
            sd.sales_date,
            (sd.sales_date - interval '1 day')::date as prev_sales_date,
            sc.sales_units,
            sc.sales_rub,
            sp.prev_sales_units,
            sp.prev_sales_rub,
            od.orders_date,
            oc.orders_today,
            td.stocks_snapshot_hour,
            tc.in_way_to_client_units
        from sales_dates sd
        cross join sales_current sc
        cross join sales_previous sp
        cross join orders_dates od
        cross join orders_current oc
        cross join stocks_dates td
        cross join stocks_current tc
    """

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(query)
        row = cur.fetchone()

    if not row:
        return CliDashboardKpis.unavailable()

    sales_date = row["sales_date"]
    prev_sales_date = row["prev_sales_date"] if sales_date else None
    orders_date = row["orders_date"]
    stocks_snapshot_hour = row["stocks_snapshot_hour"]

    sales_units = _coerce_optional_int(row["sales_units"]) if sales_date else None
    sales_rub = _coerce_optional_float(row["sales_rub"]) if sales_date else None
    prev_sales_units = _coerce_optional_int(row["prev_sales_units"]) if prev_sales_date else None
    prev_sales_rub = _coerce_optional_float(row["prev_sales_rub"]) if prev_sales_date else None
    orders_today = _coerce_optional_int(row["orders_today"]) if orders_date else None
    in_way_to_client_units = (
        _coerce_optional_int(row["in_way_to_client_units"]) if stocks_snapshot_hour else None
    )

    missing_sources = []
    if sales_date is None:
        missing_sources.append("продажи")
    if orders_date is None:
        missing_sources.append("заказы")
    if stocks_snapshot_hour is None:
        missing_sources.append("остатки")

    warning = None
    if missing_sources:
        warning = (
            "Данные KPI временно недоступны"
            if len(missing_sources) == 3
            else f"Часть KPI недоступна: {', '.join(missing_sources)}"
        )

    return CliDashboardKpis(
        sales_units=sales_units,
        sales_rub=sales_rub,
        orders_today=orders_today,
        sales_units_delta_pct=_compute_delta_pct(sales_units, prev_sales_units),
        sales_rub_delta_pct=_compute_delta_pct(sales_rub, prev_sales_rub),
        in_way_to_client_units=in_way_to_client_units,
        sales_date=sales_date,
        prev_sales_date=prev_sales_date,
        orders_date=orders_date,
        stocks_snapshot_hour=stocks_snapshot_hour,
        is_partial=bool(missing_sources),
        warning=warning,
    )
