from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from agi_ecom_app.config import get_settings
from agi_ecom_app.inventory_policy import (
    calculate_coverage_days,
    classify_availability,
    classify_coverage_days,
    normalize_current_coverage_row,
    normalize_turnover_row,
)
from agi_ecom_app.services.db import (
    _parse_report_period,
    get_connection,
    serialize_rows,
    serialize_value,
)


PORTFOLIO_ANALYSIS_LIMIT = 100


def _latest_full_day_sql(column_sql: str) -> str:
    return (
        f"coalesce("
        f"max({column_sql}) filter (where {column_sql} < current_date), "
        f"max({column_sql})"
        f")"
    )


def _validate_limit(limit: int, upper: int = 20) -> int:
    if not 1 <= limit <= upper:
        raise ValueError(f"limit must be between 1 and {upper}")
    return limit


def _validate_days(days: int, upper: int = 30) -> int:
    if not 1 <= days <= upper:
        raise ValueError(f"days must be between 1 and {upper}")
    return days


def _validate_period_limit(limit: int, upper: int = 1000) -> int:
    if not 1 <= limit <= upper:
        raise ValueError(f"limit must be between 1 and {upper}")
    return limit


def _validate_period_days(days: int, upper: int = 31) -> int:
    if not 1 <= days <= upper:
        raise ValueError(f"period_days must be between 1 and {upper}")
    return days


def _keyword_clause(keyword: str | None, expressions: list[str]) -> tuple[str, list[Any]]:
    if not keyword:
        return "", []

    like = f"%{keyword}%"
    params = [like] * len(expressions)
    clause = " and (" + " or ".join(f"coalesce({expr}, '') ilike %s" for expr in expressions) + ")"
    return clause, params


def _parse_sku_tokens(sku_list: str | None) -> list[str]:
    if not sku_list:
        return []
    tokens: list[str] = []
    for token in str(sku_list).replace(";", ",").split(","):
        cleaned = token.strip()
        if cleaned and cleaned not in tokens:
            tokens.append(cleaned)
    return tokens


def _supply_keyword_clause(
    product_keyword: str | None,
    *,
    vendor_code_expression: str,
    nm_id_expression: str,
    fallback_expressions: list[str],
) -> tuple[str, list[Any]]:
    """Filter a supply query by one free-text keyword or multiple exact SKU ids."""

    tokens = _parse_sku_tokens(product_keyword)
    if len(tokens) <= 1:
        return _keyword_clause(product_keyword, fallback_expressions)

    conditions: list[str] = []
    params: list[Any] = []
    for token in tokens:
        conditions.append(
            f"(coalesce({vendor_code_expression}, '') = %s "
            f"or coalesce({nm_id_expression}, '') = %s)"
        )
        params.extend([token, token])
    return " and (" + " or ".join(conditions) + ")", params


def get_sales_vendor_code_map(nm_ids: list[int | str]) -> dict[str, str]:
    normalized_ids: list[int] = []
    for nm_id in nm_ids:
        try:
            value = int(str(nm_id).strip())
        except (TypeError, ValueError):
            continue
        if value > 0 and value not in normalized_ids:
            normalized_ids.append(value)

    if not normalized_ids:
        return {}

    values_sql = ", ".join(["(%s::bigint)"] * len(normalized_ids))
    query = f"""
        with requested(nm_id) as (
            values {values_sql}
        ),
        sales_codes as (
            select
                s.nm_id,
                max(nullif(s.supplier_article, '')) as vendor_code
            from sales.wb_supplier_sales s
            join requested r on r.nm_id = s.nm_id
            group by s.nm_id
        ),
        funnel_codes as (
            select
                f.nm_id,
                max(nullif(f.vendor_code, '')) as vendor_code
            from sales.wb_sales_funnel_products_daily f
            join requested r on r.nm_id = f.nm_id
            group by f.nm_id
        )
        select
            r.nm_id,
            coalesce(s.vendor_code, f.vendor_code) as vendor_code
        from requested r
        left join sales_codes s on s.nm_id = r.nm_id
        left join funnel_codes f on f.nm_id = r.nm_id
        where coalesce(s.vendor_code, f.vendor_code) is not null
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(query, normalized_ids)
        rows = serialize_rows(cur.fetchall())

    return {str(row["nm_id"]): str(row["vendor_code"]) for row in rows if row.get("vendor_code")}


def get_sku_identity_map(nm_ids: list[int | str]) -> dict[str, dict[str, Any]]:
    requested_keys: list[str] = []
    for nm_id in nm_ids:
        candidate = str(nm_id).strip()
        if not candidate:
            continue
        try:
            value = int(candidate)
        except (TypeError, ValueError):
            continue
        if value <= 0:
            continue
        normalized = str(value)
        if normalized not in requested_keys:
            requested_keys.append(normalized)

    if not requested_keys:
        return {}

    values_sql = ", ".join(["(%s::text)"] * len(requested_keys))
    query = f"""
        with requested(raw_key) as (
            values {values_sql}
        ),
        latest_catalog as (
            select distinct on (nm_id)
                nm_id,
                nullif(vendor_code, '') as vendor_code,
                nullif(brand, '') as brand_name,
                nullif(subject_name, '') as subject_name,
                nullif(title, '') as title
            from analytics_ai.wb_sku_ai_analysis
            where nm_id::text in (select raw_key from requested)
               or nullif(vendor_code, '') in (select raw_key from requested)
            order by nm_id, snapshot_date desc, last_synced_at desc nulls last, id desc
        ),
        sales_identity as (
            select
                s.nm_id,
                max(nullif(s.supplier_article, '')) as vendor_code,
                max(nullif(s.brand, '')) as brand_name,
                max(nullif(s.subject, '')) as subject_name
            from sales.wb_supplier_sales s
            join requested r on r.raw_key = s.nm_id::text or r.raw_key = s.supplier_article
            group by s.nm_id
        ),
        funnel_identity as (
            select
                f.nm_id,
                max(nullif(f.vendor_code, '')) as vendor_code,
                max(nullif(f.brand_name, '')) as brand_name,
                max(nullif(f.subject_name, '')) as subject_name
            from sales.wb_sales_funnel_products_daily f
            join requested r on r.raw_key = f.nm_id::text or r.raw_key = f.vendor_code
            group by f.nm_id
        ),
        finance_identity as (
            select
                d.nm_id,
                max(nullif(d.sa_name, '')) as vendor_code,
                max(nullif(d.brand_name, '')) as brand_name,
                max(nullif(d.subject_name, '')) as subject_name
            from finance.wb_report_detail_daily d
            join requested r on r.raw_key = d.nm_id::text or r.raw_key = d.sa_name
            group by d.nm_id
        ),
        merged_identity as (
            select
                r.raw_key as request_key,
                i.nm_id,
                i.vendor_code,
                i.brand_name,
                i.subject_name,
                i.title,
                case
                    when r.raw_key = i.nm_id::text then 0
                    when r.raw_key = i.vendor_code then 1
                    else 2
                end as match_rank
            from requested r
            join (
                select
                    r2.raw_key,
                    coalesce(c.nm_id, s.nm_id, f.nm_id, fi.nm_id) as nm_id,
                    coalesce(c.vendor_code, s.vendor_code, f.vendor_code, fi.vendor_code) as vendor_code,
                    coalesce(c.brand_name, s.brand_name, f.brand_name, fi.brand_name) as brand_name,
                    coalesce(c.subject_name, s.subject_name, f.subject_name, fi.subject_name) as subject_name,
                    c.title
                from requested r2
                left join latest_catalog c
                    on r2.raw_key = c.nm_id::text or r2.raw_key = c.vendor_code
                left join sales_identity s
                    on r2.raw_key = s.nm_id::text or r2.raw_key = s.vendor_code
                left join funnel_identity f
                    on r2.raw_key = f.nm_id::text or r2.raw_key = f.vendor_code
                left join finance_identity fi
                    on r2.raw_key = fi.nm_id::text or r2.raw_key = fi.vendor_code
            ) i on i.raw_key = r.raw_key
            where i.nm_id is not null
        )
        select
            distinct on (request_key)
            request_key,
            nm_id,
            vendor_code,
            brand_name,
            subject_name,
            title
        from merged_identity
        order by request_key, match_rank, nm_id desc
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(query, requested_keys)
        rows = serialize_rows(cur.fetchall())

    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        request_key = row.get("request_key")
        if request_key is None:
            continue
        result[str(request_key)] = {
            "nm_id": row.get("nm_id"),
            "vendor_code": row.get("vendor_code"),
            "brand_name": row.get("brand_name"),
            "subject_name": row.get("subject_name"),
            "title": row.get("title"),
        }
    return result


MARKETING_CAMPAIGN_DAILY_CTE = """
agent_campaign_daily as (
    with latest_campaigns as (
        select distinct on (external_id)
            external_id as advert_id,
            snapshot_date as campaign_snapshot_date,
            campaign_type,
            name as campaign_name,
            status,
            type,
            payment_type,
            brand,
            created_at,
            updated_at,
            started_at,
            deleted_at,
            change_time,
            end_time,
            settings_json,
            last_synced_at as campaign_last_synced_at
        from marketing.wb_marketing_campaigns
        order by external_id, snapshot_date desc, last_synced_at desc, id desc
    ),
    latest_budget as (
        select distinct on (advert_id)
            advert_id,
            snapshot_date as budget_snapshot_date,
            cash,
            netting,
            total as budget_total,
            last_synced_at as budget_last_synced_at
        from marketing.wb_marketing_budget
        order by advert_id, snapshot_date desc, last_synced_at desc, id desc
    ),
    latest_nms as (
        select
            advert_id,
            count(distinct nm_id) as active_nms_count,
            (array_agg(subject_name order by nm_id))[1] as top_subject_name,
            max(last_synced_at) as nms_last_synced_at
        from (
            select distinct on (advert_id, nm_id)
                advert_id,
                nm_id,
                subject_name,
                last_synced_at
            from marketing.wb_marketing_campaign_nms
            order by advert_id, nm_id, snapshot_date desc, last_synced_at desc, id desc
        ) n
        group by advert_id
    )
    select
        s.stat_date,
        s.advert_id,
        coalesce(c.campaign_name, 'campaign_' || s.advert_id::text) as campaign_name,
        c.campaign_type,
        c.status,
        c.type,
        c.payment_type,
        c.brand,
        s.views,
        s.clicks,
        s.ctr,
        s.cpc,
        s.orders,
        s.shks,
        s.sum_spend,
        s.atbs,
        s.canceled,
        s.cr,
        b.cash,
        b.netting,
        b.budget_total,
        coalesce(n.active_nms_count, 0) as active_nms_count,
        n.top_subject_name,
        c.campaign_snapshot_date,
        b.budget_snapshot_date,
        greatest(
            s.last_synced_at,
            coalesce(c.campaign_last_synced_at, s.last_synced_at),
            coalesce(b.budget_last_synced_at, s.last_synced_at),
            coalesce(n.nms_last_synced_at, s.last_synced_at)
        ) as last_synced_at
    from marketing.wb_marketing_campaign_stats_daily s
    left join latest_campaigns c
        on c.advert_id = s.advert_id
    left join latest_budget b
        on b.advert_id = s.advert_id
    left join latest_nms n
        on n.advert_id = s.advert_id
)
"""

MARKETING_SEARCH_TEXT_DAILY_CTE = """
agent_search_text_daily as (
    with latest_details as (
        select distinct on (nm_id)
            nm_id,
            name,
            vendor_code,
            subject_name,
            brand_name,
            last_synced_at as details_last_synced_at
        from marketing.wb_search_query_details
        order by nm_id, snapshot_hour desc, id desc
    )
    select
        t.dt as snapshot_date,
        t.request_key,
        t.snapshot_hour,
        t.current_period_start,
        t.current_period_end,
        t.nm_id,
        t.text as query_text,
        t.frequency,
        t.avg_position as avg_position_text,
        t.orders_count,
        d.name,
        d.vendor_code,
        d.subject_name,
        d.brand_name,
        greatest(
            t.last_synced_at,
            coalesce(d.details_last_synced_at, t.last_synced_at)
        ) as last_synced_at
    from marketing.wb_search_query_orders_text_daily t
    left join latest_details d
        on d.nm_id = t.nm_id
)
"""

MARKETING_SEARCH_SKU_DAILY_CTE = """
agent_search_sku_daily as (
    with details_latest as (
        select distinct on (nm_id)
            current_period_end as snapshot_date,
            request_key,
            snapshot_hour,
            current_period_start,
            current_period_end,
            past_period_start,
            past_period_end,
            nm_id,
            name,
            vendor_code,
            subject_name,
            brand_name,
            main_photo,
            is_advertised,
            is_substituted_sku,
            is_card_rated,
            rating,
            feedback_rating,
            min_price,
            max_price,
            avg_position_current,
            avg_position_dynamics,
            open_card_current,
            open_card_dynamics,
            add_to_cart_current,
            add_to_cart_dynamics,
            open_to_cart_current,
            open_to_cart_dynamics,
            orders_current,
            orders_dynamics,
            cart_to_order_current,
            cart_to_order_dynamics,
            visibility_current,
            visibility_dynamics,
            currency,
            last_synced_at as details_last_synced_at
        from marketing.wb_search_query_details
        where current_period_end is not null
        order by nm_id, snapshot_hour desc, id desc
    ),
    overview_latest as (
        select distinct on (current_period_end)
            current_period_end as snapshot_date,
            request_key,
            snapshot_hour,
            current_period_start,
            current_period_end,
            past_period_start,
            past_period_end,
            total_products,
            supplier_rating_current,
            supplier_rating_dynamics,
            advertised_products_current,
            advertised_products_dynamics,
            position_average_current,
            position_average_dynamics,
            position_median_current,
            position_median_dynamics,
            cluster_first_hundred_current,
            cluster_first_hundred_dynamics,
            cluster_second_hundred_current,
            cluster_second_hundred_dynamics,
            cluster_below_current,
            cluster_below_dynamics,
            visibility_current as overview_visibility_current,
            visibility_dynamics as overview_visibility_dynamics,
            open_card_current as overview_open_card_current,
            open_card_dynamics as overview_open_card_dynamics,
            currency as overview_currency,
            last_synced_at as overview_last_synced_at
        from marketing.wb_search_query_report_overview
        where current_period_end is not null
        order by current_period_end desc, snapshot_hour desc, id desc
    )
    select
        d.snapshot_date,
        d.request_key,
        d.snapshot_hour,
        d.current_period_start,
        d.current_period_end,
        d.past_period_start,
        d.past_period_end,
        d.nm_id,
        d.name,
        d.vendor_code,
        d.subject_name,
        d.brand_name,
        d.main_photo,
        d.is_advertised,
        d.is_substituted_sku,
        d.is_card_rated,
        d.rating,
        d.feedback_rating,
        d.min_price,
        d.max_price,
        d.avg_position_current,
        d.avg_position_dynamics,
        d.open_card_current,
        d.open_card_dynamics,
        d.add_to_cart_current,
        d.add_to_cart_dynamics,
        d.open_to_cart_current,
        d.open_to_cart_dynamics,
        d.orders_current,
        d.orders_dynamics,
        d.cart_to_order_current,
        d.cart_to_order_dynamics,
        d.visibility_current,
        d.visibility_dynamics,
        o.total_products,
        o.supplier_rating_current,
        o.supplier_rating_dynamics,
        o.advertised_products_current,
        o.advertised_products_dynamics,
        o.position_average_current,
        o.position_average_dynamics,
        o.position_median_current,
        o.position_median_dynamics,
        o.cluster_first_hundred_current,
        o.cluster_first_hundred_dynamics,
        o.cluster_second_hundred_current,
        o.cluster_second_hundred_dynamics,
        o.cluster_below_current,
        o.cluster_below_dynamics,
        o.overview_visibility_current,
        o.overview_visibility_dynamics,
        o.overview_open_card_current,
        o.overview_open_card_dynamics,
        coalesce(d.currency, o.overview_currency) as currency,
        greatest(
            d.details_last_synced_at,
            coalesce(o.overview_last_synced_at, d.details_last_synced_at)
        ) as last_synced_at
    from details_latest d
    left join overview_latest o
        on o.snapshot_date = d.snapshot_date
)
"""


def get_sales_overview(
    product_keyword: str | None = None,
    limit: int = PORTFOLIO_ANALYSIS_LIMIT,
) -> dict[str, Any]:
    limit = _validate_limit(limit, upper=PORTFOLIO_ANALYSIS_LIMIT)
    raw_sales_keyword_clause, raw_sales_keyword_params = _keyword_clause(
        product_keyword,
        [
            "s.supplier_article",
            "s.brand",
            "s.subject",
            "s.nm_id::text",
        ],
    )
    grouped_sales_keyword_clause, grouped_sales_keyword_params = _keyword_clause(
        product_keyword,
        [
            "vendor_code",
            "brand_name",
            "subject_name",
            "nm_id::text",
        ],
    )
    funnel_keyword_clause, funnel_keyword_params = _keyword_clause(
        product_keyword,
        [
            "vendor_code",
            "brand_name",
            "subject_name",
            "nm_id::text",
        ],
    )

    totals_query = f"""
        with latest_sales_day as (
            select {_latest_full_day_sql('date_ord::date')} as sales_date
            from sales.wb_supplier_sales
        )
        select
            d.sales_date as sales_date,
            count(*) filter (where s.is_realization and s.date_ord::date = d.sales_date) as sales_1d_units,
            count(*) filter (
                where s.is_realization
                  and s.date_ord::date between d.sales_date - interval '6 day' and d.sales_date
            ) as sales_7d_units,
            count(distinct s.nm_id) filter (
                where s.is_realization
                  and s.date_ord::date between d.sales_date - interval '6 day' and d.sales_date
            ) as sold_sku_7d,
            sum(case when s.is_realization and s.date_ord::date = d.sales_date then coalesce(s.for_pay, 0) else 0 end) as revenue_1d,
            sum(case when s.is_realization and s.date_ord::date between d.sales_date - interval '6 day' and d.sales_date then coalesce(s.for_pay, 0) else 0 end) as revenue_7d,
            avg(case when s.is_realization and s.date_ord::date between d.sales_date - interval '6 day' and d.sales_date then s.price_with_disc end) as avg_price_with_disc_7d
        from sales.wb_supplier_sales s
        cross join latest_sales_day d
        where 1 = 1
        {raw_sales_keyword_clause}
        group by d.sales_date
    """
    top_revenue_query = f"""
        with latest_sales_day as (
            select {_latest_full_day_sql('date_ord::date')} as sales_date
            from sales.wb_supplier_sales
        )
        select
            (select sales_date from latest_sales_day) as report_date,
            nm_id,
            vendor_code,
            brand_name,
            subject_name,
            sales_7d_units,
            revenue_7d,
            avg_price_with_disc_7d
        from (
            select
                s.nm_id,
                s.supplier_article as vendor_code,
                s.brand as brand_name,
                s.subject as subject_name,
                count(*) filter (where s.is_realization) as sales_7d_units,
                sum(case when s.is_realization then coalesce(s.for_pay, 0) else 0 end) as revenue_7d,
                avg(case when s.is_realization then s.price_with_disc end) as avg_price_with_disc_7d
            from sales.wb_supplier_sales s
            where s.date_ord::date between (select sales_date - interval '6 day' from latest_sales_day)
                                       and (select sales_date from latest_sales_day)
            group by s.nm_id, s.supplier_article, s.brand, s.subject
        ) t
        where coalesce(revenue_7d, 0) > 0
        {grouped_sales_keyword_clause}
        order by coalesce(revenue_7d, 0) desc, coalesce(sales_7d_units, 0) desc
        limit %s
    """
    conversion_watchlist_query = f"""
        with latest_day as (
            select {_latest_full_day_sql('report_date')} as report_date
            from sales.wb_sales_funnel_products_daily
        )
        select
            report_date,
            nm_id,
            vendor_code,
            brand_name,
            subject_name,
            open_count,
            cart_count,
            order_count,
            add_to_cart_conversion,
            cart_to_order_conversion,
            buyout_percent
        from sales.wb_sales_funnel_products_daily
        where report_date = (select report_date from latest_day)
          and coalesce(open_count, 0) >= 10
        {funnel_keyword_clause}
        order by coalesce(cart_to_order_conversion, 999999) asc, coalesce(open_count, 0) desc
        limit %s
    """
    sales_raw_query = f"""
        with latest_day as (
            select {_latest_full_day_sql("date_ord::date")} as sales_date
            from sales.wb_supplier_sales
        )
        select
            (select sales_date from latest_day) as sales_date,
            s.nm_id,
            s.supplier_article as vendor_code,
            s.brand,
            s.subject,
            count(*) filter (where s.is_realization) as sales_count,
            sum(case when s.is_realization then coalesce(s.for_pay, 0) else 0 end) as revenue
        from sales.wb_supplier_sales s
        where s.date_ord::date = (select sales_date from latest_day)
        {raw_sales_keyword_clause}
        group by s.nm_id, s.supplier_article, s.brand, s.subject
        order by revenue desc nulls last, sales_count desc
        limit %s
    """

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(totals_query, raw_sales_keyword_params)
        totals = serialize_value(cur.fetchone())

        cur.execute(top_revenue_query, [*grouped_sales_keyword_params, limit])
        top_revenue = serialize_rows(cur.fetchall())

        cur.execute(conversion_watchlist_query, [*funnel_keyword_params, limit])
        conversion_watchlist = serialize_rows(cur.fetchall())

        cur.execute(sales_raw_query, [*raw_sales_keyword_params, limit])
        sales_raw_rows = serialize_rows(cur.fetchall())

    return {
        "focus_keyword": product_keyword,
        "totals": totals,
        "top_revenue_rows": top_revenue,
        "conversion_watchlist": conversion_watchlist,
        "sales_raw_rows": sales_raw_rows,
    }


def get_sku_daily_price_sales_comparison(
    product_keyword: str,
    days: int = 14,
) -> dict[str, Any]:
    """Compare seller price, discounts, funnel conversion, orders and sales by day."""

    target = str(product_keyword or "").strip()
    if not target:
        raise ValueError("product_keyword must not be empty")
    days = _validate_days(days, upper=31)
    query = """
        with latest_day as (
            select max(report_date) as report_date
            from agent_marts.v_agent_price_sku_daily
            where nm_id::text = %s or vendor_code = %s
        )
        select
            p.report_date,
            p.nm_id,
            p.vendor_code,
            p.base_price_rub,
            p.seller_discount_pct,
            p.seller_effective_price_rub,
            p.observed_transaction_seller_price_avg_rub,
            p.observed_buyer_price_avg_rub,
            p.observed_spp_pct_avg,
            case
                when p.price_snapshot_exact_day and p.seller_effective_price_rub is not null
                    then 'seller_snapshot_exact_day'
                when p.observed_transaction_seller_price_avg_rub is not null
                    then 'observed_transaction_average'
                else 'missing'
            end as price_basis,
            p.non_cancelled_order_units,
            p.realized_sales_units,
            p.realized_for_pay_rub,
            f.open_count,
            f.cart_count,
            f.order_count as funnel_order_count,
            f.add_to_cart_conversion,
            f.cart_to_order_conversion,
            f.buyout_percent,
            p.business_day_complete,
            p.marketing_day_complete,
            p.sales_day_complete
        from agent_marts.v_agent_price_sku_daily p
        left join sales.wb_sales_funnel_products_daily f
          on f.report_date = p.report_date
         and f.nm_id = p.nm_id
        where (p.nm_id::text = %s or p.vendor_code = %s)
          and p.report_date between
              (select report_date - (%s::int - 1) from latest_day)
              and (select report_date from latest_day)
        order by p.report_date desc
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(query, [target, target, target, target, days])
        rows = serialize_rows(cur.fetchall())
    return {
        "focus_keyword": target,
        "days": days,
        "source_tables": [
            "agent_marts.v_agent_price_sku_daily",
            "sales.wb_sales_funnel_products_daily",
        ],
        "row_count": len(rows),
        "rows": rows,
        "causal_valid": False,
    }


def _delta_pct(current: int | float | None, previous: int | float | None) -> float | None:
    if current is None or previous in (None, 0):
        return None
    return round(((float(current) - float(previous)) / float(previous)) * 100, 2)


def _format_report_date(value: date) -> str:
    return value.strftime("%d.%m.%Y")


def _format_date_range(start: date, end: date) -> str:
    if start == end:
        return _format_report_date(start)
    return f"{_format_report_date(start)} - {_format_report_date(end)}"


def _format_delta(value: int | float | None, pct: float | None, *, suffix: str = "") -> str:
    if value is None:
        return "нет данных"
    sign = "+" if value > 0 else ""
    pct_text = "" if pct is None else f" ({pct:+.1f}%)"
    if isinstance(value, float):
        value_text = f"{sign}{value:,.2f}".replace(",", " ")
    else:
        value_text = f"{sign}{value}"
    return f"{value_text}{suffix}{pct_text}"


def _format_money(value: int | float | None) -> str:
    if value is None:
        return "нет данных"
    return f"{float(value):,.0f} ₽".replace(",", " ")


def _format_number(value: int | float | None, *, decimals: int = 0) -> str:
    if value is None:
        return "нет данных"
    if decimals <= 0:
        return f"{float(value):,.0f}".replace(",", " ")
    return f"{float(value):,.{decimals}f}".replace(",", " ")


def _format_percent(value: int | float | None) -> str:
    if value is None:
        return "нет данных"
    return f"{float(value):,.2f}%".replace(",", " ")


def get_sales_report_snapshot(product_keyword: str | None = None) -> dict[str, Any]:
    """Return a deterministic top-of-report sales comparison snapshot."""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            select {_latest_full_day_sql('date_ord::date')} as latest_sales_date
            from sales.wb_supplier_sales
            """
        )
        latest_row = serialize_value(cur.fetchone()) or {}
        latest_sales_date = latest_row.get("latest_sales_date")
        if latest_sales_date is None:
            return {
                "report_date": None,
                "focus_keyword": product_keyword,
                "source_table": "sales.wb_supplier_sales",
                "required_fields": [
                    "period",
                    "current_period",
                    "previous_period",
                    "current_sales_units",
                    "previous_sales_units",
                    "sales_units_delta",
                    "sales_units_delta_pct",
                    "current_revenue_rub",
                    "previous_revenue_rub",
                    "revenue_delta_rub",
                    "revenue_delta_pct",
                ],
                "calculations": {
                    "sales_units_delta": "current_sales_units - previous_sales_units",
                    "sales_units_delta_pct": "(current_sales_units - previous_sales_units) / previous_sales_units * 100",
                    "revenue_delta_rub": "current_revenue_rub - previous_revenue_rub",
                    "revenue_delta_pct": "(current_revenue_rub - previous_revenue_rub) / previous_revenue_rub * 100",
                },
                "rows": [],
                "report_table": {
                    "title": "Sales-сводка",
                    "columns": [
                        "Сравнение",
                        "Периоды",
                        "Продажи тек./пред.",
                        "Δ продаж",
                        "Выручка тек./пред.",
                        "Δ выручки",
                    ],
                    "rows": [],
                },
                "markdown_table": "Данные продаж не найдены.",
            }

        report_date = date.fromisoformat(str(latest_sales_date))
        period_specs = [
            (
                "Последний день в данных к предыдущему",
                report_date,
                report_date,
                report_date - timedelta(days=1),
                report_date - timedelta(days=1),
            ),
            (
                "Последние 7 дней к предыдущим 7",
                report_date - timedelta(days=6),
                report_date,
                report_date - timedelta(days=13),
                report_date - timedelta(days=7),
            ),
            (
                "Последние 30 дней к предыдущим 30",
                report_date - timedelta(days=29),
                report_date,
                report_date - timedelta(days=59),
                report_date - timedelta(days=30),
            ),
        ]
        values_sql = []
        period_params: list[Any] = []
        for index, (label, current_start, current_end, previous_start, previous_end) in enumerate(
            period_specs,
            start=1,
        ):
            values_sql.append("(%s, %s, 'current', %s::date, %s::date)")
            period_params.extend([label, index, current_start, current_end])
            values_sql.append("(%s, %s, 'previous', %s::date, %s::date)")
            period_params.extend([label, index, previous_start, previous_end])

        keyword_clause, keyword_params = _keyword_clause(
            product_keyword,
            [
                "s.supplier_article",
                "s.brand",
                "s.subject",
                "s.nm_id::text",
            ],
        )
        query = f"""
            with periods(label, sort_order, side, date_from, date_to) as (
                values {", ".join(values_sql)}
            )
            select
                p.label,
                p.sort_order,
                p.side,
                p.date_from,
                p.date_to,
                coalesce(sum(case when s.is_realization then 1 else 0 end), 0)::int as sales_units,
                round(coalesce(sum(case when s.is_realization then coalesce(s.for_pay, 0) else 0 end), 0)::numeric, 2) as revenue_rub
            from periods p
            left join sales.wb_supplier_sales s
              on s.date_ord::date between p.date_from and p.date_to
              {keyword_clause}
            group by p.label, p.sort_order, p.side, p.date_from, p.date_to
            order by p.sort_order, p.side
        """
        cur.execute(query, [*period_params, *keyword_params])
        raw_rows = serialize_rows(cur.fetchall())

    rows_by_label: dict[str, dict[str, Any]] = {}
    for raw in raw_rows:
        label = str(raw["label"])
        rows_by_label.setdefault(label, {})[str(raw["side"])] = raw

    rows: list[dict[str, Any]] = []
    for label, current_start, current_end, previous_start, previous_end in period_specs:
        current = rows_by_label.get(label, {}).get("current", {})
        previous = rows_by_label.get(label, {}).get("previous", {})
        current_sales = int(current.get("sales_units") or 0)
        previous_sales = int(previous.get("sales_units") or 0)
        current_revenue = float(current.get("revenue_rub") or 0)
        previous_revenue = float(previous.get("revenue_rub") or 0)
        sales_delta = current_sales - previous_sales
        revenue_delta = round(current_revenue - previous_revenue, 2)
        rows.append(
            {
                "period": label,
                "current_period": {
                    "date_from": current_start.isoformat(),
                    "date_to": current_end.isoformat(),
                },
                "previous_period": {
                    "date_from": previous_start.isoformat(),
                    "date_to": previous_end.isoformat(),
                },
                "current_sales_units": current_sales,
                "previous_sales_units": previous_sales,
                "sales_units_delta": sales_delta,
                "sales_units_delta_pct": _delta_pct(current_sales, previous_sales),
                "current_revenue_rub": current_revenue,
                "previous_revenue_rub": previous_revenue,
                "revenue_delta_rub": revenue_delta,
                "revenue_delta_pct": _delta_pct(current_revenue, previous_revenue),
            }
        )

    report_table_rows: list[list[str | int | float | None]] = []
    for row in rows:
        current_period = _format_date_range(
            date.fromisoformat(row["current_period"]["date_from"]),
            date.fromisoformat(row["current_period"]["date_to"]),
        )
        previous_period = _format_date_range(
            date.fromisoformat(row["previous_period"]["date_from"]),
            date.fromisoformat(row["previous_period"]["date_to"]),
        )
        report_table_rows.append(
            [
                str(row["period"]),
                f"{current_period} к {previous_period}",
                f"{row['current_sales_units']} / {row['previous_sales_units']}",
                _format_delta(
                    row["sales_units_delta"],
                    row["sales_units_delta_pct"],
                ),
                f"{_format_money(row['current_revenue_rub'])} / {_format_money(row['previous_revenue_rub'])}",
                _format_delta(
                    row["revenue_delta_rub"],
                    row["revenue_delta_pct"],
                    suffix=" ₽",
                ),
            ]
        )
    report_table = {
        "title": "Sales-сводка",
        "columns": [
            "Сравнение",
            "Периоды",
            "Продажи тек./пред.",
            "Δ продаж",
            "Выручка тек./пред.",
            "Δ выручки",
        ],
        "rows": report_table_rows,
    }
    markdown_lines = [
        "| " + " | ".join(report_table["columns"]) + " |",
        "|---|---|---|---|---|---|",
    ]
    for row in report_table_rows:
        markdown_lines.append("| " + " | ".join(str(cell) for cell in row) + " |")

    return {
        "report_date": report_date.isoformat(),
        "focus_keyword": product_keyword,
        "source_table": "sales.wb_supplier_sales",
        "required_fields": [
            "period",
            "current_period",
            "previous_period",
            "current_sales_units",
            "previous_sales_units",
            "sales_units_delta",
            "sales_units_delta_pct",
            "current_revenue_rub",
            "previous_revenue_rub",
            "revenue_delta_rub",
            "revenue_delta_pct",
        ],
        "calculations": {
            "sales_units_delta": "current_sales_units - previous_sales_units",
            "sales_units_delta_pct": "(current_sales_units - previous_sales_units) / previous_sales_units * 100",
            "revenue_delta_rub": "current_revenue_rub - previous_revenue_rub",
            "revenue_delta_pct": "(current_revenue_rub - previous_revenue_rub) / previous_revenue_rub * 100",
            "month_to_month_default": "rolling 30 days vs previous 30 days",
        },
        "rows": rows,
        "report_table": report_table,
        "markdown_table": "\n".join(markdown_lines),
    }


def get_marketing_report_snapshot(product_keyword: str | None = None) -> dict[str, Any]:
    """Return a deterministic top-of-report marketing comparison snapshot with ДРР."""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            select least(marketing.latest_marketing_date, sales.latest_sales_date) as report_date
            from (
                select {_latest_full_day_sql('stat_date')} as latest_marketing_date
                from marketing.wb_marketing_campaign_stats_daily
            ) marketing
            cross join (
                select {_latest_full_day_sql('date_ord::date')} as latest_sales_date
                from sales.wb_supplier_sales
                where is_realization = true
            ) sales
            """
        )
        latest_row = serialize_value(cur.fetchone()) or {}
        latest_report_date = latest_row.get("report_date")
        report_table_columns = [
            "Сравнение",
            "Периоды",
            "Расход тек./пред.",
            "Δ расхода",
            "Клики тек./пред.",
            "CPC тек./пред.",
            "Заказы рекламы тек./пред.",
            "Выручка SKU тек./пред.",
            "ДРР тек./пред.",
            "Δ ДРР п.п.",
        ]
        if latest_report_date is None:
            return {
                "report_date": None,
                "focus_keyword": product_keyword,
                "source_tables": [
                    "marketing.wb_marketing_campaign_stats_daily",
                    "marketing.wb_marketing_campaign_nms",
                    "sales.wb_supplier_sales",
                    "sales.wb_sales_funnel_products_daily",
                ],
                "required_fields": [
                    "period",
                    "current_period",
                    "previous_period",
                    "current_spend_rub",
                    "previous_spend_rub",
                    "spend_delta_rub",
                    "spend_delta_pct",
                    "current_clicks",
                    "previous_clicks",
                    "current_cpc_rub",
                    "previous_cpc_rub",
                    "current_ad_orders",
                    "previous_ad_orders",
                    "current_promoted_sku_revenue_rub",
                    "previous_promoted_sku_revenue_rub",
                    "current_drr_pct",
                    "previous_drr_pct",
                    "drr_delta_pp",
                ],
                "calculations": {
                    "drr_pct": "marketing_spend_rub / promoted_sku_revenue_rub * 100",
                    "drr_delta_pp": "current_drr_pct - previous_drr_pct",
                    "cpc_rub": "marketing_spend_rub / clicks",
                    "spend_delta_pct": "(current_spend_rub - previous_spend_rub) / previous_spend_rub * 100",
                },
                "rows": [],
                "report_table": {
                    "title": "Marketing-сводка",
                    "columns": report_table_columns,
                    "rows": [],
                },
                "markdown_table": "Данные маркетинга или продаж не найдены.",
            }

        report_date = date.fromisoformat(str(latest_report_date))
        period_specs = [
            (
                "Последний день в данных к предыдущему",
                report_date,
                report_date,
                report_date - timedelta(days=1),
                report_date - timedelta(days=1),
            ),
            (
                "Последние 7 дней к предыдущим 7",
                report_date - timedelta(days=6),
                report_date,
                report_date - timedelta(days=13),
                report_date - timedelta(days=7),
            ),
            (
                "Последние 30 дней к предыдущим 30",
                report_date - timedelta(days=29),
                report_date,
                report_date - timedelta(days=59),
                report_date - timedelta(days=30),
            ),
        ]
        values_sql = []
        period_params: list[Any] = []
        for index, (label, current_start, current_end, previous_start, previous_end) in enumerate(
            period_specs,
            start=1,
        ):
            values_sql.append("(%s, %s, 'current', %s::date, %s::date)")
            period_params.extend([label, index, current_start, current_end])
            values_sql.append("(%s, %s, 'previous', %s::date, %s::date)")
            period_params.extend([label, index, previous_start, previous_end])

        keyword_clause, keyword_params = _keyword_clause(
            product_keyword,
            [
                "si.nm_id::text",
                "si.vendor_code",
                "si.subject_name",
                "si.brand_name",
            ],
        )
        query = f"""
            with periods(label, sort_order, side, date_from, date_to) as (
                values {", ".join(values_sql)}
            ),
            latest_campaign_nms as (
                select
                    advert_id,
                    nm_id,
                    max(nullif(subject_name, '')) as campaign_subject_name
                from (
                    select distinct on (advert_id, nm_id)
                        advert_id,
                        nm_id,
                        subject_name,
                        snapshot_date,
                        last_synced_at,
                        id
                    from marketing.wb_marketing_campaign_nms
                    order by advert_id, nm_id, snapshot_date desc, last_synced_at desc, id desc
                ) n
                group by advert_id, nm_id
            ),
            promoted_nms as (
                select distinct nm_id
                from latest_campaign_nms
            ),
            sales_codes as (
                select
                    s.nm_id,
                    max(nullif(s.supplier_article, '')) as vendor_code,
                    max(nullif(s.subject, '')) as subject_name,
                    max(nullif(s.brand, '')) as brand_name
                from sales.wb_supplier_sales s
                join promoted_nms p using (nm_id)
                group by s.nm_id
            ),
            funnel_codes as (
                select
                    f.nm_id,
                    max(nullif(f.vendor_code, '')) as vendor_code,
                    max(nullif(f.subject_name, '')) as subject_name,
                    max(nullif(f.brand_name, '')) as brand_name
                from sales.wb_sales_funnel_products_daily f
                join promoted_nms p using (nm_id)
                group by f.nm_id
            ),
            sku_identity as (
                select
                    p.nm_id,
                    coalesce(sc.vendor_code, fc.vendor_code) as vendor_code,
                    coalesce(sc.subject_name, fc.subject_name, n.campaign_subject_name) as subject_name,
                    coalesce(sc.brand_name, fc.brand_name) as brand_name
                from promoted_nms p
                left join sales_codes sc using (nm_id)
                left join funnel_codes fc using (nm_id)
                left join (
                    select nm_id, max(campaign_subject_name) as campaign_subject_name
                    from latest_campaign_nms
                    group by nm_id
                ) n using (nm_id)
            ),
            focus_nms as (
                select distinct si.nm_id
                from sku_identity si
                where true
                {keyword_clause}
            ),
            eligible_campaigns as (
                select distinct n.advert_id
                from latest_campaign_nms n
                join focus_nms f using (nm_id)
            ),
            campaign_spend as (
                select
                    p.label,
                    p.sort_order,
                    p.side,
                    s.advert_id,
                    round(sum(coalesce(s.sum_spend, 0))::numeric, 2) as spend_rub,
                    sum(coalesce(s.views, 0))::int as views,
                    sum(coalesce(s.clicks, 0))::int as clicks,
                    sum(coalesce(s.orders, 0))::int as ad_orders
                from periods p
                left join marketing.wb_marketing_campaign_stats_daily s
                  on s.stat_date between p.date_from and p.date_to
                left join eligible_campaigns ec
                  on ec.advert_id = s.advert_id
                where ec.advert_id is not null
                group by p.label, p.sort_order, p.side, s.advert_id
                having sum(coalesce(s.sum_spend, 0)) > 0
            ),
            marketing_by_period as (
                select
                    p.label,
                    p.sort_order,
                    p.side,
                    p.date_from,
                    p.date_to,
                    count(distinct cs.advert_id)::int as campaign_count,
                    round(coalesce(sum(cs.spend_rub), 0)::numeric, 2) as spend_rub,
                    coalesce(sum(cs.views), 0)::int as views,
                    coalesce(sum(cs.clicks), 0)::int as clicks,
                    coalesce(sum(cs.ad_orders), 0)::int as ad_orders
                from periods p
                left join campaign_spend cs
                  on cs.label = p.label
                 and cs.side = p.side
                group by p.label, p.sort_order, p.side, p.date_from, p.date_to
            ),
            campaign_period_nms as (
                select distinct
                    cs.label,
                    cs.side,
                    n.nm_id
                from campaign_spend cs
                join latest_campaign_nms n
                  on n.advert_id = cs.advert_id
                join focus_nms f
                  on f.nm_id = n.nm_id
            ),
            sales_by_period as (
                select
                    p.label,
                    p.side,
                    count(distinct cpn.nm_id)::int as promoted_nm_count,
                    round(coalesce(sum(case when s.is_realization then coalesce(s.for_pay, 0) else 0 end), 0)::numeric, 2)
                        as promoted_sku_revenue_rub
                from periods p
                left join campaign_period_nms cpn
                  on cpn.label = p.label
                 and cpn.side = p.side
                left join sales.wb_supplier_sales s
                  on s.nm_id = cpn.nm_id
                 and s.date_ord::date between p.date_from and p.date_to
                group by p.label, p.side
            )
            select
                mbp.label,
                mbp.sort_order,
                mbp.side,
                mbp.date_from,
                mbp.date_to,
                mbp.campaign_count,
                coalesce(sbp.promoted_nm_count, 0)::int as promoted_nm_count,
                mbp.spend_rub,
                mbp.views,
                mbp.clicks,
                round(case when mbp.clicks > 0 then mbp.spend_rub / mbp.clicks end, 2) as cpc_rub,
                mbp.ad_orders,
                coalesce(sbp.promoted_sku_revenue_rub, 0) as promoted_sku_revenue_rub,
                round(case when coalesce(sbp.promoted_sku_revenue_rub, 0) > 0
                    then mbp.spend_rub * 100 / sbp.promoted_sku_revenue_rub
                end, 2) as drr_pct
            from marketing_by_period mbp
            left join sales_by_period sbp
              on sbp.label = mbp.label
             and sbp.side = mbp.side
            order by mbp.sort_order, mbp.side
        """
        cur.execute(query, [*period_params, *keyword_params])
        raw_rows = serialize_rows(cur.fetchall())

    rows_by_label: dict[str, dict[str, Any]] = {}
    for raw in raw_rows:
        label = str(raw["label"])
        rows_by_label.setdefault(label, {})[str(raw["side"])] = raw

    rows: list[dict[str, Any]] = []
    for label, current_start, current_end, previous_start, previous_end in period_specs:
        current = rows_by_label.get(label, {}).get("current", {})
        previous = rows_by_label.get(label, {}).get("previous", {})
        current_spend = float(current.get("spend_rub") or 0)
        previous_spend = float(previous.get("spend_rub") or 0)
        current_views = int(current.get("views") or 0)
        previous_views = int(previous.get("views") or 0)
        current_clicks = int(current.get("clicks") or 0)
        previous_clicks = int(previous.get("clicks") or 0)
        current_ad_orders = int(current.get("ad_orders") or 0)
        previous_ad_orders = int(previous.get("ad_orders") or 0)
        current_revenue = float(current.get("promoted_sku_revenue_rub") or 0)
        previous_revenue = float(previous.get("promoted_sku_revenue_rub") or 0)
        current_drr = current.get("drr_pct")
        previous_drr = previous.get("drr_pct")
        current_drr_value = float(current_drr) if current_drr is not None else None
        previous_drr_value = float(previous_drr) if previous_drr is not None else None
        rows.append(
            {
                "period": label,
                "current_period": {
                    "date_from": current_start.isoformat(),
                    "date_to": current_end.isoformat(),
                },
                "previous_period": {
                    "date_from": previous_start.isoformat(),
                    "date_to": previous_end.isoformat(),
                },
                "current_campaign_count": int(current.get("campaign_count") or 0),
                "previous_campaign_count": int(previous.get("campaign_count") or 0),
                "current_promoted_nm_count": int(current.get("promoted_nm_count") or 0),
                "previous_promoted_nm_count": int(previous.get("promoted_nm_count") or 0),
                "current_spend_rub": current_spend,
                "previous_spend_rub": previous_spend,
                "spend_delta_rub": round(current_spend - previous_spend, 2),
                "spend_delta_pct": _delta_pct(current_spend, previous_spend),
                "current_views": current_views,
                "previous_views": previous_views,
                "views_delta": current_views - previous_views,
                "views_delta_pct": _delta_pct(current_views, previous_views),
                "current_clicks": current_clicks,
                "previous_clicks": previous_clicks,
                "clicks_delta": current_clicks - previous_clicks,
                "clicks_delta_pct": _delta_pct(current_clicks, previous_clicks),
                "current_cpc_rub": float(current["cpc_rub"]) if current.get("cpc_rub") is not None else None,
                "previous_cpc_rub": float(previous["cpc_rub"]) if previous.get("cpc_rub") is not None else None,
                "current_ad_orders": current_ad_orders,
                "previous_ad_orders": previous_ad_orders,
                "ad_orders_delta": current_ad_orders - previous_ad_orders,
                "ad_orders_delta_pct": _delta_pct(current_ad_orders, previous_ad_orders),
                "current_promoted_sku_revenue_rub": current_revenue,
                "previous_promoted_sku_revenue_rub": previous_revenue,
                "promoted_sku_revenue_delta_rub": round(current_revenue - previous_revenue, 2),
                "promoted_sku_revenue_delta_pct": _delta_pct(current_revenue, previous_revenue),
                "current_drr_pct": current_drr_value,
                "previous_drr_pct": previous_drr_value,
                "drr_delta_pp": (
                    round(current_drr_value - previous_drr_value, 2)
                    if current_drr_value is not None and previous_drr_value is not None
                    else None
                ),
            }
        )

    report_table_rows: list[list[str | int | float | None]] = []
    for row in rows:
        current_period = _format_date_range(
            date.fromisoformat(row["current_period"]["date_from"]),
            date.fromisoformat(row["current_period"]["date_to"]),
        )
        previous_period = _format_date_range(
            date.fromisoformat(row["previous_period"]["date_from"]),
            date.fromisoformat(row["previous_period"]["date_to"]),
        )
        report_table_rows.append(
            [
                str(row["period"]),
                f"{current_period} к {previous_period}",
                f"{_format_money(row['current_spend_rub'])} / {_format_money(row['previous_spend_rub'])}",
                _format_delta(row["spend_delta_rub"], row["spend_delta_pct"], suffix=" ₽"),
                f"{row['current_clicks']} / {row['previous_clicks']}",
                f"{_format_money(row['current_cpc_rub'])} / {_format_money(row['previous_cpc_rub'])}",
                f"{_format_number(row['current_ad_orders'])} / {_format_number(row['previous_ad_orders'])}",
                (
                    f"{_format_money(row['current_promoted_sku_revenue_rub'])} / "
                    f"{_format_money(row['previous_promoted_sku_revenue_rub'])}"
                ),
                f"{_format_percent(row['current_drr_pct'])} / {_format_percent(row['previous_drr_pct'])}",
                _format_delta(row["drr_delta_pp"], None, suffix=" п.п."),
            ]
        )
    report_table = {
        "title": "Marketing-сводка",
        "columns": report_table_columns,
        "rows": report_table_rows,
    }
    markdown_lines = [
        "| " + " | ".join(report_table["columns"]) + " |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in report_table_rows:
        markdown_lines.append("| " + " | ".join(str(cell) for cell in row) + " |")

    return {
        "report_date": report_date.isoformat(),
        "focus_keyword": product_keyword,
        "source_tables": [
            "marketing.wb_marketing_campaign_stats_daily",
            "marketing.wb_marketing_campaign_nms",
            "sales.wb_supplier_sales",
            "sales.wb_sales_funnel_products_daily",
        ],
        "required_fields": [
            "period",
            "current_period",
            "previous_period",
            "current_spend_rub",
            "previous_spend_rub",
            "spend_delta_rub",
            "spend_delta_pct",
            "current_clicks",
            "previous_clicks",
            "current_cpc_rub",
            "previous_cpc_rub",
            "current_ad_orders",
            "previous_ad_orders",
            "current_promoted_sku_revenue_rub",
            "previous_promoted_sku_revenue_rub",
            "current_drr_pct",
            "previous_drr_pct",
            "drr_delta_pp",
        ],
        "calculations": {
            "drr_pct": "marketing_spend_rub / promoted_sku_revenue_rub * 100",
            "drr_delta_pp": "current_drr_pct - previous_drr_pct",
            "cpc_rub": "marketing_spend_rub / clicks",
            "spend_delta_rub": "current_spend_rub - previous_spend_rub",
            "spend_delta_pct": "(current_spend_rub - previous_spend_rub) / previous_spend_rub * 100",
            "promoted_sku_revenue": "sales.wb_supplier_sales.for_pay for SKU linked to campaigns with spend in the same period side",
            "month_to_month_default": "rolling 30 days vs previous 30 days",
        },
        "interpretation_note": (
            "ДРР считается как расходы рекламы / выручка продвигаемых SKU * 100. "
            "Выручка продвигаемых SKU означает все фактические продажи SKU, "
            "которые связаны с кампаниями со spend > 0 в этой стороне периода."
        ),
        "rows": rows,
        "report_table": report_table,
        "markdown_table": "\n".join(markdown_lines),
    }


def get_sales_sku_revenue_change(
    date_from: str | None = None,
    date_to: str | None = None,
    product_keyword: str | None = None,
    limit: int = 20,
    period_days: int = 7,
) -> dict[str, Any]:
    limit = _validate_limit(limit, upper=100)
    period_days = _validate_period_days(period_days)

    with get_connection() as conn, conn.cursor() as cur:
        if date_from:
            current_start, current_end = _parse_report_period(date_from=date_from, date_to=date_to)
            if current_start is None or current_end is None:
                raise ValueError("date_from must not be empty")
            if current_end < current_start:
                raise ValueError("date_to must be greater than or equal to date_from")
        else:
            cur.execute(
                f"""
                select {_latest_full_day_sql('date_ord::date')} as latest_sales_date
                from sales.wb_supplier_sales
                """
            )
            latest_row = serialize_value(cur.fetchone()) or {}
            latest_sales_date = latest_row.get("latest_sales_date")
            if latest_sales_date is None:
                return {
                    "date_from": None,
                    "date_to": None,
                    "current_period_start": None,
                    "current_period_end": None,
                    "previous_period_start": None,
                    "previous_period_end": None,
                    "period_days": period_days,
                    "focus_keyword": product_keyword,
                    "source_table": "sales.wb_supplier_sales",
                    "row_count": 0,
                    "rows": [],
                }
            current_end = date.fromisoformat(str(latest_sales_date))
            current_start = current_end - timedelta(days=period_days - 1)

        previous_end = current_start - timedelta(days=1)
        previous_start = previous_end - (current_end - current_start)

        keyword_clause, keyword_params = _keyword_clause(
            product_keyword,
            [
                "vendor_code",
                "brand_name",
                "subject_name",
                "nm_id::text",
            ],
        )

        query = f"""
            with current_sku as (
                select
                    nm_id,
                    max(nullif(supplier_article, '')) as vendor_code,
                    max(nullif(brand, '')) as brand_name,
                    max(nullif(subject, '')) as subject_name,
                    round(sum(coalesce(for_pay, 0))::numeric, 2) as current_revenue,
                    count(*) as current_sales
                from sales.wb_supplier_sales
                where is_realization
                  and date_ord::date between %s and %s
                group by nm_id
            ),
            previous_sku as (
                select
                    nm_id,
                    max(nullif(supplier_article, '')) as vendor_code,
                    max(nullif(brand, '')) as brand_name,
                    max(nullif(subject, '')) as subject_name,
                    round(sum(coalesce(for_pay, 0))::numeric, 2) as previous_revenue,
                    count(*) as previous_sales
                from sales.wb_supplier_sales
                where is_realization
                  and date_ord::date between %s and %s
                group by nm_id
            ),
            combined as (
                select
                    coalesce(c.nm_id, p.nm_id) as nm_id,
                    coalesce(c.vendor_code, p.vendor_code) as vendor_code,
                    coalesce(c.brand_name, p.brand_name) as brand_name,
                    coalesce(c.subject_name, p.subject_name) as subject_name,
                    coalesce(c.current_revenue, 0) as current_revenue,
                    coalesce(p.previous_revenue, 0) as previous_revenue,
                    round((coalesce(c.current_revenue, 0) - coalesce(p.previous_revenue, 0))::numeric, 2) as delta_revenue,
                    case
                        when coalesce(p.previous_revenue, 0) <> 0 then round(
                            100 * (coalesce(c.current_revenue, 0) - p.previous_revenue)
                            / p.previous_revenue,
                            2
                        )
                    end as delta_revenue_pct,
                    coalesce(c.current_sales, 0) as current_sales,
                    coalesce(p.previous_sales, 0) as previous_sales,
                    coalesce(c.current_sales, 0) - coalesce(p.previous_sales, 0) as delta_sales
                from current_sku c
                full outer join previous_sku p
                  on p.nm_id = c.nm_id
            )
            select
                %s::date as current_period_start,
                %s::date as current_period_end,
                %s::date as previous_period_start,
                %s::date as previous_period_end,
                nm_id,
                vendor_code,
                brand_name,
                subject_name,
                current_revenue,
                previous_revenue,
                delta_revenue,
                delta_revenue_pct,
                current_sales,
                previous_sales,
                delta_sales,
                case
                    when previous_sales = 0 and current_sales > 0 then 'new_in_current'
                    when previous_sales > 0 and current_sales = 0 then 'lost_in_current'
                    when delta_revenue > 0 then 'growth_driver'
                    when delta_revenue < 0 then 'decline_driver'
                    else 'stable'
                end as driver_role
            from combined
            where 1 = 1
            {keyword_clause}
            order by abs(delta_revenue) desc, abs(delta_sales) desc, nm_id
            limit %s
        """

        params = [
            current_start,
            current_end,
            previous_start,
            previous_end,
            current_start,
            current_end,
            previous_start,
            previous_end,
            *keyword_params,
            limit,
        ]
        cur.execute(query, params)
        rows = serialize_rows(cur.fetchall())

    return {
        "date_from": current_start.isoformat(),
        "date_to": current_end.isoformat(),
        "current_period_start": current_start.isoformat(),
        "current_period_end": current_end.isoformat(),
        "previous_period_start": previous_start.isoformat(),
        "previous_period_end": previous_end.isoformat(),
        "period_days": (current_end - current_start).days + 1,
        "focus_keyword": product_keyword,
        "source_table": "sales.wb_supplier_sales",
        "row_count": len(rows),
        "rows": rows,
    }


def get_price_context(
    nm_id: int | str | None = None,
    vendor_code: str | None = None,
    product_keyword: str | None = None,
    limit: int = 20,
    include_evidence: bool = False,
) -> dict[str, Any]:
    """Return the stable latest price-layer contract used by price agents."""

    limit = _validate_limit(limit, upper=100)
    nm_id_value = int(nm_id) if nm_id not in (None, "") else None
    vendor_code_value = (
        vendor_code.strip()
        if isinstance(vendor_code, str) and vendor_code.strip()
        else None
    )
    keyword_value = (
        product_keyword.strip()
        if isinstance(product_keyword, str) and product_keyword.strip()
        else None
    )

    filters: list[str] = []
    params: list[Any] = []
    if nm_id_value is not None:
        filters.append("p.nm_id = %s")
        params.append(nm_id_value)
    if vendor_code_value is not None:
        filters.append("p.vendor_code = %s")
        params.append(vendor_code_value)
    if keyword_value is not None:
        pattern = f"%{keyword_value}%"
        filters.append(
            "("
            "p.nm_id::text ilike %s "
            "or coalesce(p.vendor_code, '') ilike %s "
            "or coalesce(p.brand, '') ilike %s "
            "or coalesce(p.subject_name, '') ilike %s "
            "or coalesce(p.title, '') ilike %s"
            ")"
        )
        params.extend([pattern] * 5)
    where_sql = " and " + " and ".join(filters) if filters else ""

    query = f"""
        select p.*
        from analytics_ai.v_agent_price_context_latest p
        where 1 = 1
        {where_sql}
        order by
            p.context_as_of_date desc nulls last,
            p.nm_id
        limit %s
    """

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(query, [*params, limit])
        rows = serialize_rows(cur.fetchall())

        evidence: list[dict[str, Any]] = []
        selected_nm_ids = [int(row["nm_id"]) for row in rows if row.get("nm_id")]
        if include_evidence and selected_nm_ids:
            cur.execute(
                """
                with ranked_runs as (
                    select
                        r.source_nm_id,
                        r.run_id::text as run_id,
                        r.batch_id::text as batch_id,
                        r.query,
                        r.observed_at,
                        r.completed_at,
                        r.source_found_any,
                        r.source_position_organic,
                        row_number() over (
                            partition by r.source_nm_id
                            order by r.observed_at desc, r.completed_at desc nulls last
                        ) as row_number
                    from analytics_ai.wb_serp_runs r
                    where r.source_nm_id = any(%s::bigint[])
                )
                select
                    source_nm_id,
                    run_id,
                    batch_id,
                    query,
                    observed_at,
                    completed_at,
                    source_found_any,
                    source_position_organic,
                    'analytics_ai.wb_serp_runs:' || run_id as evidence_ref
                from ranked_runs
                where row_number <= 5
                order by source_nm_id, observed_at desc
                """,
                [selected_nm_ids],
            )
            evidence = serialize_rows(cur.fetchall())

    for row in rows:
        row["evidence_refs"] = [
            (
                "pricing.wb_supplier_prices:"
                f"{row['nm_id']}:{row.get('seller_snapshot_date')}"
            ),
            (
                "analytics_ai.wb_price_ai_analysis:"
                f"{row.get('analysis_id')}:{row.get('analysis_run_date')}"
            ),
            (
                "analytics_ai.mart_price_context_daily:"
                f"{row['nm_id']}:{row.get('context_snapshot_date')}"
            ),
        ]

    return {
        "contract_version": "price-context-v1",
        "source_view": "analytics_ai.v_agent_price_context_latest",
        "grain": "one latest row per nm_id",
        "price_semantics": {
            "seller_controlled_price": "Цена, управляемая продавцом в отчёте WB.",
            "customer_price_display": "Обычная цена на странице товара.",
            "customer_price_wallet": "Цена при оплате WB Кошельком.",
            "customer_price_effective": "Минимальная подтверждённая цена покупателя.",
            "broad_market_price_median": "Широкий рынок из поисковой выдачи; не подтверждённые аналоги.",
            "strict_competitor_price_median": "Медиана строгих аналогов после проверки качества.",
        },
        "quality_rule": (
            "price_status is directional only when analysis_status=success and "
            "strict_competitor_count>=5; otherwise it is "
            "insufficient_competitor_quality"
        ),
        "row_count": len(rows),
        "rows": rows,
        "evidence": evidence,
    }


def get_price_competitor_overview(
    product_keyword: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """Return latest price positioning by SKU from competitor raw rows.

    Competitors are deduplicated by source_nm_id + competitor_nm_id because the
    raw table can contain the same competitor across several search keywords.
    """
    limit = _validate_limit(limit, upper=100)
    keyword_clause, keyword_params = _keyword_clause(
        product_keyword,
        [
            "coalesce(s.vendor_code, '')",
            "coalesce(d.source_brand, s.brand, '')",
            "coalesce(d.source_subject_name, s.subject_name, '')",
            "coalesce(d.source_title, s.title, '')",
            "d.source_nm_id::text",
        ],
    )

    latest_run_query = """
        select max(run_date) as run_date
        from analytics_ai.wb_price_ai_competitors_raw
    """
    query = f"""
        with latest_run as (
            select max(run_date) as run_date
            from analytics_ai.wb_price_ai_competitors_raw
        ),
        latest_sku as (
            select distinct on (nm_id)
                nm_id,
                vendor_code,
                brand,
                subject_name,
                title,
                discounted_price,
                club_discounted_price,
                wb_link_price_promo,
                wb_link_price_regular
            from analytics_ai.wb_sku_ai_analysis
            order by nm_id, snapshot_date desc, last_synced_at desc nulls last, id desc
        ),
        keyword_coverage as (
            select
                source_nm_id,
                count(distinct keyword) as keyword_count,
                count(*) as raw_rows
            from analytics_ai.wb_price_ai_competitors_raw
            where run_date = (select run_date from latest_run)
            group by source_nm_id
        ),
        deduped as (
            select distinct on (source_nm_id, competitor_nm_id)
                run_date,
                source_nm_id,
                source_brand,
                source_subject_name,
                source_title,
                source_price_regular,
                source_price_promo,
                source_price_segment_min,
                source_price_segment_max,
                competitor_nm_id,
                competitor_price_promo,
                price_delta_abs,
                price_delta_percent,
                within_price_segment,
                same_subject,
                same_color,
                photo_ai_is_competitor,
                photo_ai_confidence,
                catalog_position_absolute,
                keyword
            from analytics_ai.wb_price_ai_competitors_raw
            where run_date = (select run_date from latest_run)
              and competitor_price_promo is not null
            order by
                source_nm_id,
                competitor_nm_id,
                within_price_segment desc,
                same_subject desc,
                photo_ai_is_competitor desc nulls last,
                catalog_position_absolute nulls last,
                keyword_order,
                keyword
        ),
        aggregated as (
            select
                d.run_date,
                d.source_nm_id as nm_id,
                max(s.vendor_code) as vendor_code,
                max(coalesce(d.source_brand, s.brand)) as brand_name,
                max(coalesce(d.source_subject_name, s.subject_name)) as subject_name,
                max(coalesce(d.source_title, s.title)) as title,
                coalesce(
                    max(d.source_price_promo),
                    max(s.wb_link_price_promo),
                    max(s.club_discounted_price),
                    max(s.discounted_price)
                ) as source_price_promo,
                coalesce(
                    max(d.source_price_regular),
                    max(s.wb_link_price_regular)
                ) as source_price_regular,
                max(d.source_price_segment_min) as source_price_segment_min,
                max(d.source_price_segment_max) as source_price_segment_max,
                count(*)::int as competitor_count,
                count(*) filter (where d.competitor_price_promo < d.source_price_promo)::int as cheaper_competitor_count,
                count(*) filter (where d.competitor_price_promo > d.source_price_promo)::int as pricier_competitor_count,
                count(*) filter (where d.within_price_segment)::int as within_price_segment_count,
                count(*) filter (where d.same_subject)::int as same_subject_count,
                count(*) filter (where d.photo_ai_is_competitor)::int as photo_ai_competitor_count,
                count(*) filter (where d.photo_ai_is_competitor is null)::int as photo_ai_unchecked_count,
                min(d.competitor_price_promo) as competitor_price_min,
                percentile_cont(0.5) within group (order by d.competitor_price_promo) as competitor_price_median,
                max(d.competitor_price_promo) as competitor_price_max,
                round(avg(d.competitor_price_promo)::numeric, 2) as competitor_price_avg,
                coalesce(max(k.keyword_count), 0)::int as keyword_count,
                coalesce(max(k.raw_rows), 0)::int as raw_rows
            from deduped d
            left join latest_sku s on s.nm_id = d.source_nm_id
            left join keyword_coverage k on k.source_nm_id = d.source_nm_id
            where 1 = 1
            {keyword_clause}
            group by d.run_date, d.source_nm_id
        ),
        scored as (
            select
                *,
                round(
                    case when competitor_count > 0
                        then cheaper_competitor_count::numeric * 100 / competitor_count
                    end,
                    2
                ) as cheaper_competitor_share_pct,
                round(
                    case when competitor_count > 0
                        then pricier_competitor_count::numeric * 100 / competitor_count
                    end,
                    2
                ) as pricier_competitor_share_pct,
                round(
                    case when competitor_price_median > 0
                        then ((source_price_promo - competitor_price_median) * 100 / competitor_price_median)::numeric
                    end,
                    2
                ) as source_vs_median_gap_pct,
                round(
                    case when competitor_price_median is not null
                        then (source_price_promo - competitor_price_median)::numeric
                    end,
                    2
                ) as source_vs_median_gap_rub
            from aggregated
        )
        select
            *,
            case
                when competitor_count < 5 or source_price_promo is null or competitor_price_median is null
                    then 'thin_data'
                when source_vs_median_gap_pct >= 15 and cheaper_competitor_share_pct >= 60
                    then 'above_market'
                when source_vs_median_gap_pct <= -15 and pricier_competitor_share_pct >= 60
                    then 'below_market'
                else 'market_aligned'
            end as price_status
        from scored
        order by
            case
                when competitor_count < 5 or source_price_promo is null or competitor_price_median is null then 3
                when source_vs_median_gap_pct >= 15 and cheaper_competitor_share_pct >= 60 then 0
                when source_vs_median_gap_pct <= -15 and pricier_competitor_share_pct >= 60 then 1
                else 2
            end,
            abs(coalesce(source_vs_median_gap_pct, 0)) desc,
            competitor_count desc,
            nm_id
        limit %s
    """

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(latest_run_query)
        latest_run = serialize_value(cur.fetchone()) or {}
        if latest_run.get("run_date") is None:
            return {
                "run_date": None,
                "source_table": "analytics_ai.wb_price_ai_competitors_raw",
                "sku_lookup_table": "analytics_ai.wb_sku_ai_analysis",
                "dedupe_key": "run_date + source_nm_id + competitor_nm_id",
                "focus_keyword": product_keyword,
                "row_count": 0,
                "rows": [],
            }

        cur.execute(query, [*keyword_params, limit])
        rows = serialize_rows(cur.fetchall())

    return {
        "run_date": latest_run["run_date"],
        "source_table": "analytics_ai.wb_price_ai_competitors_raw",
        "sku_lookup_table": "analytics_ai.wb_sku_ai_analysis",
        "current_price_table": "pricing.wb_supplier_prices",
        "dedupe_key": "run_date + source_nm_id + competitor_nm_id",
        "focus_keyword": product_keyword,
        "row_count": len(rows),
        "rows": rows,
    }


def get_price_competitor_details(
    nm_id: int | str | None = None,
    vendor_code: str | None = None,
    product_keyword: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """Return deduplicated competitor rows for one SKU or a focused subset."""
    limit = _validate_limit(limit, upper=100)
    nm_id_value = int(nm_id) if nm_id not in (None, "") else None
    vendor_code_value = vendor_code.strip() if isinstance(vendor_code, str) and vendor_code.strip() else None
    keyword_clause, keyword_params = _keyword_clause(
        product_keyword,
        [
            "coalesce(s.vendor_code, '')",
            "coalesce(r.source_brand, s.brand, '')",
            "coalesce(r.source_subject_name, s.subject_name, '')",
            "coalesce(r.source_title, s.title, '')",
            "coalesce(r.keyword, '')",
            "r.source_nm_id::text",
            "r.competitor_nm_id::text",
            "coalesce(r.competitor_brand, '')",
            "coalesce(r.competitor_title, '')",
        ],
    )

    filter_clauses = []
    filter_params: list[Any] = []
    if nm_id_value is not None:
        filter_clauses.append("r.source_nm_id = %s")
        filter_params.append(nm_id_value)
    if vendor_code_value is not None:
        filter_clauses.append("s.vendor_code = %s")
        filter_params.append(vendor_code_value)
    filter_sql = ""
    if filter_clauses:
        filter_sql = " and " + " and ".join(filter_clauses)

    latest_run_query = """
        select max(run_date) as run_date
        from analytics_ai.wb_price_ai_competitors_raw
    """
    query = f"""
        with latest_run as (
            select max(run_date) as run_date
            from analytics_ai.wb_price_ai_competitors_raw
        ),
        latest_sku as (
            select distinct on (nm_id)
                nm_id,
                vendor_code,
                brand,
                subject_name,
                title
            from analytics_ai.wb_sku_ai_analysis
            order by nm_id, snapshot_date desc, last_synced_at desc nulls last, id desc
        ),
        ranked as (
            select
                r.*,
                s.vendor_code,
                coalesce(r.source_brand, s.brand) as source_brand_name,
                coalesce(r.source_subject_name, s.subject_name) as source_subject_name_resolved,
                coalesce(r.source_title, s.title) as source_title_resolved,
                row_number() over (
                    partition by r.source_nm_id, r.competitor_nm_id
                    order by
                        r.within_price_segment desc,
                        r.same_subject desc,
                        r.photo_ai_is_competitor desc nulls last,
                        r.catalog_position_absolute nulls last,
                        r.keyword_order,
                        r.keyword
                ) as dedupe_rank
            from analytics_ai.wb_price_ai_competitors_raw r
            left join latest_sku s on s.nm_id = r.source_nm_id
            where r.run_date = (select run_date from latest_run)
              and r.competitor_price_promo is not null
              {filter_sql}
              {keyword_clause}
        )
        select
            run_date,
            keyword,
            source_nm_id,
            vendor_code,
            source_brand_name as source_brand,
            source_subject_name_resolved as source_subject_name,
            source_title_resolved as source_title,
            source_price_regular,
            source_price_promo,
            source_price_segment_min,
            source_price_segment_max,
            competitor_nm_id,
            competitor_brand,
            competitor_supplier_name,
            competitor_subject_name,
            competitor_title,
            competitor_product_url,
            competitor_price_regular,
            competitor_price_promo,
            competitor_discount_percent,
            price_delta_abs,
            price_delta_percent,
            within_price_segment,
            same_subject,
            same_color,
            photo_ai_is_competitor,
            photo_ai_confidence,
            competitor_rating,
            competitor_feedbacks,
            catalog_position_absolute
        from ranked
        where dedupe_rank = 1
        order by
            within_price_segment desc,
            same_subject desc,
            photo_ai_is_competitor desc nulls last,
            catalog_position_absolute nulls last,
            abs(coalesce(price_delta_percent, 0)) desc,
            competitor_nm_id
        limit %s
    """

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(latest_run_query)
        latest_run = serialize_value(cur.fetchone()) or {}
        if latest_run.get("run_date") is None:
            return {
                "run_date": None,
                "source_table": "analytics_ai.wb_price_ai_competitors_raw",
                "sku_lookup_table": "analytics_ai.wb_sku_ai_analysis",
                "dedupe_key": "run_date + source_nm_id + competitor_nm_id",
                "nm_id": nm_id_value,
                "vendor_code": vendor_code_value,
                "focus_keyword": product_keyword,
                "row_count": 0,
                "rows": [],
            }

        cur.execute(query, [*filter_params, *keyword_params, limit])
        rows = serialize_rows(cur.fetchall())

    return {
        "run_date": latest_run["run_date"],
        "source_table": "analytics_ai.wb_price_ai_competitors_raw",
        "sku_lookup_table": "analytics_ai.wb_sku_ai_analysis",
        "dedupe_key": "run_date + source_nm_id + competitor_nm_id",
        "nm_id": nm_id_value,
        "vendor_code": vendor_code_value,
        "focus_keyword": product_keyword,
        "row_count": len(rows),
        "rows": rows,
    }


def get_price_decision_overview(
    product_keyword: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    limit = _validate_limit(limit, upper=100)
    keyword_clause, keyword_params = _keyword_clause(
        product_keyword,
        ["vendor_code", "brand_name", "subject_name", "nm_id::text"],
    )

    query = f"""
        with latest_price_context as (
            select max(context_as_of_date) as context_as_of_date
            from analytics_ai.v_agent_price_context_latest
        ),
        scored as (
            select
                snapshot_date,
                nm_id,
                vendor_code,
                brand_name,
                subject_name,
                price_current,
                stock_qty,
                baseline_orders_7d,
                baseline_revenue_gross_7d,
                baseline_revenue_net_7d,
                baseline_buyouts_7d,
                baseline_adv_spend_7d,
                elasticity_to_apply,
                elasticity_source,
                marketing_decision_hint,
                marketing_safe_growth_pct,
                marketing_marginal_quality_coef,
                marketing_winner_query_share_7d,
                marketing_waste_query_share_7d,
                marketing_confidence,
                competitor_count,
                competitor_price_min,
                competitor_price_median,
                competitor_price_max,
                price_ai_run_date,
                scenario_id,
                sales_model,
                price_after_discount_rub,
                net_profit_rub_current,
                margin_pct_current,
                cost_price_rub,
                ad_percent,
                tax_percent,
                expected_outbound_per_buyout_rub,
                expected_return_logistics_rub,
                storage_total_rub,
                other_expenses_rub,
                candidate_price_min,
                candidate_price_max,
                candidate_step_rub,
                recommended_price,
                recommended_price_delta_abs,
                recommended_price_delta_pct,
                expected_orders_7d,
                expected_net_profit_7d,
                expected_margin_pct,
                decision_status,
                decision_confidence,
                decision_reason_code,
                market_ceiling_enabled,
                current_net_profit_7d,
                latest_price_context.context_as_of_date
                    as price_context_as_of_date,
                latest_price_context.context_as_of_date - snapshot_date
                    as decision_lag_days,
                case
                    when latest_price_context.context_as_of_date is null
                        then 'context_unavailable'
                    when snapshot_date is null
                        then 'decision_date_missing'
                    when latest_price_context.context_as_of_date - snapshot_date > 2
                        then 'stale'
                    else 'fresh'
                end as decision_freshness,
                round(
                    case when competitor_price_median > 0
                        then ((price_current - competitor_price_median) * 100 / competitor_price_median)::numeric
                    end,
                    2
                ) as source_vs_median_gap_pct,
                round(
                    case when competitor_price_median is not null
                        then (price_current - competitor_price_median)::numeric
                    end,
                    2
                ) as source_vs_median_gap_rub,
                case
                    when competitor_count < 5 or price_current is null or competitor_price_median is null
                        then 'thin_data'
                    when coalesce(recommended_price_delta_pct, 0) <= -3
                        then 'above_market'
                    when coalesce(recommended_price_delta_pct, 0) >= 3
                        then 'below_market'
                    else 'market_aligned'
                end as price_status
            from pricing.mart_price_decision_latest
            cross join latest_price_context
            where 1 = 1
            {keyword_clause}
        )
        select
            snapshot_date,
            nm_id,
            vendor_code,
            brand_name,
            subject_name,
            price_current as source_price_promo,
            price_after_discount_rub as source_price_regular,
            competitor_count,
            competitor_price_min,
            competitor_price_median,
            competitor_price_max,
            source_vs_median_gap_pct,
            source_vs_median_gap_rub,
            price_status,
            elasticity_to_apply,
            elasticity_source,
            marketing_decision_hint,
            marketing_safe_growth_pct,
            marketing_marginal_quality_coef,
            marketing_winner_query_share_7d,
            marketing_waste_query_share_7d,
            marketing_confidence,
            scenario_id,
            sales_model,
            net_profit_rub_current,
            margin_pct_current,
            cost_price_rub,
            ad_percent,
            tax_percent,
            expected_outbound_per_buyout_rub,
            expected_return_logistics_rub,
            storage_total_rub,
            other_expenses_rub,
            stock_qty,
            baseline_orders_7d,
            baseline_revenue_gross_7d,
            baseline_revenue_net_7d,
            baseline_buyouts_7d,
            baseline_adv_spend_7d,
            candidate_price_min as source_price_segment_min,
            candidate_price_max as source_price_segment_max,
            candidate_price_min,
            candidate_price_max,
            candidate_step_rub,
            recommended_price,
            recommended_price_delta_abs,
            recommended_price_delta_pct,
            expected_orders_7d,
            expected_net_profit_7d,
            expected_margin_pct,
            decision_status,
            decision_confidence,
            decision_reason_code,
            market_ceiling_enabled,
            current_net_profit_7d,
            price_ai_run_date,
            price_context_as_of_date,
            decision_lag_days,
            decision_freshness
        from scored
        order by
            case decision_status
                when 'lower' then 0
                when 'raise' then 1
                when 'hold' then 2
                else 3
            end,
            abs(coalesce(recommended_price_delta_pct, 0)) desc,
            competitor_count desc,
            nm_id
        limit %s
    """

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(query, [*keyword_params, limit])
        rows = serialize_rows(cur.fetchall())

    return {
        "focus_keyword": product_keyword,
        "source_table": "pricing.mart_price_decision_latest",
        "freshness_rule": (
            "A decision more than 2 days behind price_context_as_of_date is "
            "historical only and must not be presented as current."
        ),
        "row_count": len(rows),
        "rows": rows,
    }


def get_supply_overview(
    product_keyword: str | None = None,
    limit: int = PORTFOLIO_ANALYSIS_LIMIT,
) -> dict[str, Any]:
    """Return the latest supply overview with warehouse rows aggregated by SKU.

    When ``product_keyword`` identifies one SKU, ``sku_stock_lookup`` is the
    authoritative current-stock contract. Its ``available_stock_qty`` is the
    sum of ``quantity`` across every row for that SKU on the latest snapshot.
    ``status=no_data`` keeps the quantity as ``None``; a missing row is never
    represented as a zero balance.
    """

    limit = _validate_limit(limit, upper=PORTFOLIO_ANALYSIS_LIMIT)
    keyword_clause, keyword_params = _supply_keyword_clause(
        product_keyword,
        vendor_code_expression="vendor_code",
        nm_id_expression="nm_id::text",
        fallback_expressions=["vendor_code", "brand_name", "subject_name", "nm_id::text"],
    )
    stock_keyword_clause, stock_keyword_params = _supply_keyword_clause(
        product_keyword,
        vendor_code_expression="supplier_article",
        nm_id_expression="nm_id::text",
        fallback_expressions=["supplier_article", "brand", "subject", "nm_id::text"],
    )

    inventory_cte = """
        with latest_stock_day as (
            select {latest_stock_date} as snapshot_date
            from supply.wb_supplier_stocks
        ),
        latest_sales_day as (
            select {latest_sales_date} as sales_date
            from sales.wb_supplier_sales
        ),
        latest_supply_day as (
            select {latest_supply_date} as snapshot_date
            from supply.wb_supplies
        ),
        supplier_inbound as (
            select
                g.nm_id,
                s.snapshot_date as supplier_snapshot_date,
                sum(
                    greatest(
                        coalesce(g.quantity, 0) - coalesce(g.ready_for_sale_quantity, 0),
                        0
                    )
                )::numeric as supplier_inbound_qty,
                min(s.supply_date) as supplier_eta_date,
                count(*)::int as supplier_goods_rows
            from supply.wb_supplies s
            join supply.wb_supplies_goods g
              on g.snapshot_date = s.snapshot_date
             and g.supply_id = s.supply_id
            where s.snapshot_date = (select snapshot_date from latest_supply_day)
              and g.nm_id is not null
              and (
                    coalesce(g.quantity, 0) > coalesce(g.ready_for_sale_quantity, 0)
                    or coalesce(g.unloading_quantity, 0) > 0
                  )
            group by g.nm_id, s.snapshot_date
        ),
        sales_7d as (
            select
                s.nm_id,
                s.supplier_article as vendor_code,
                s.brand as brand_name,
                s.subject as subject_name,
                count(*) filter (where s.is_realization and s.date_ord::date = (select sales_date from latest_sales_day)) as sales_1d_units,
                count(*) filter (where s.is_realization) as sales_7d_units
            from sales.wb_supplier_sales s
            where s.date_ord::date between (select sales_date - interval '6 day' from latest_sales_day)
                                       and (select sales_date from latest_sales_day)
            group by s.nm_id, s.supplier_article, s.brand, s.subject
        ),
        stocks as (
            select
                snapshot_date,
                nm_id,
                supplier_article as vendor_code,
                brand as brand_name,
                subject as subject_name,
                sum(coalesce(quantity, 0)) as stock_qty,
                sum(coalesce(quantity_full, 0)) as stock_qty_full,
                sum(coalesce(in_way_to_client, 0)) as in_way_to_client,
                sum(coalesce(in_way_from_client, 0)) as in_way_from_client
            from supply.wb_supplier_stocks
            where snapshot_date = (select snapshot_date from latest_stock_day)
            group by snapshot_date, nm_id, supplier_article, brand, subject
        ),
        inventory as (
            select
                st.snapshot_date as report_date,
                st.nm_id,
                st.vendor_code,
                st.brand_name,
                st.subject_name,
                st.stock_qty,
                st.stock_qty_full,
                greatest(coalesce(st.stock_qty_full, 0) - coalesce(st.stock_qty, 0), 0) as unavailable_stock_qty,
                st.in_way_to_client,
                st.in_way_from_client,
                supply_day.snapshot_date as supplier_audit_snapshot_date,
                case
                    when supply_day.snapshot_date is not null
                    then coalesce(inbound.supplier_inbound_qty, 0::numeric)
                end as supplier_inbound_qty,
                case
                    when supply_day.snapshot_date is not null
                    then coalesce(inbound.supplier_goods_rows, 0)
                end as supplier_goods_rows,
                inbound.supplier_eta_date,
                coalesce(s7.sales_1d_units, 0) as sales_1d_units,
                coalesce(s7.sales_7d_units, 0) as sales_7d_units,
                case
                    when coalesce(s7.sales_7d_units, 0) > 0 then round(s7.sales_7d_units::numeric / 7, 2)
                end as avg_daily_sales_7d,
                case
                    when coalesce(s7.sales_7d_units, 0) > 0 then round(
                        st.stock_qty::numeric / (s7.sales_7d_units::numeric / 7),
                        1
                    )
                end as days_of_stock,
                case
                    when coalesce(s7.sales_7d_units, 0) > 0 then round(
                        st.stock_qty_full::numeric / (s7.sales_7d_units::numeric / 7),
                        1
                    )
                end as total_stock_days_for_audit,
                case
                    when coalesce(s7.sales_7d_units, 0) > 0
                         and st.stock_qty::numeric / (s7.sales_7d_units::numeric / 7) < 7 then 'critical'
                    when coalesce(s7.sales_7d_units, 0) > 0
                         and st.stock_qty::numeric / (s7.sales_7d_units::numeric / 7) < 21 then 'warning'
                    when coalesce(s7.sales_7d_units, 0) = 0 and coalesce(st.stock_qty, 0) = 0 then 'critical'
                    when coalesce(s7.sales_7d_units, 0) = 0 and coalesce(st.stock_qty, 0) >= 60 then 'overstock'
                    else 'ok'
                end as reorder_flag,
                case
                    when coalesce(s7.sales_7d_units, 0) > 0 and st.stock_qty::numeric / (s7.sales_7d_units::numeric / 7) < 7
                        then greatest(s7.sales_7d_units * 8 - st.stock_qty, 0)
                    when coalesce(s7.sales_7d_units, 0) > 0 and st.stock_qty::numeric / (s7.sales_7d_units::numeric / 7) < 21
                        then greatest(s7.sales_7d_units * 3 - st.stock_qty, 0)
                    else 0
                end as reorder_qty_recommended
            from stocks st
            left join sales_7d s7
              on s7.nm_id = st.nm_id
             and s7.vendor_code = st.vendor_code
            left join latest_supply_day supply_day on true
            left join supplier_inbound inbound on inbound.nm_id = st.nm_id
        )
    """.format(
        latest_stock_date=_latest_full_day_sql("snapshot_date"),
        latest_sales_date=_latest_full_day_sql("date_ord::date"),
        latest_supply_date=_latest_full_day_sql("snapshot_date"),
    )
    totals_query = inventory_cte + f"""
        select
            report_date,
            count(*) filter (where reorder_flag in ('critical', 'warning')) as alert_sku_count,
            count(*) filter (where reorder_flag = 'critical') as critical_count,
            count(*) filter (where reorder_flag = 'warning') as warning_count,
            sum(coalesce(reorder_qty_recommended, 0)) filter (where reorder_flag in ('critical', 'warning')) as reorder_qty_total
        from inventory
        where 1 = 1
        {keyword_clause}
        group by report_date
    """
    critical_query = inventory_cte + f"""
        select
            report_date,
            nm_id,
            vendor_code,
            brand_name,
            subject_name,
            stock_qty,
            stock_qty_full,
            unavailable_stock_qty,
            sales_1d_units,
            sales_7d_units,
            avg_daily_sales_7d,
            days_of_stock,
            total_stock_days_for_audit,
            reorder_flag,
            reorder_qty_recommended,
            in_way_to_client,
            in_way_from_client,
            supplier_audit_snapshot_date,
            supplier_inbound_qty,
            supplier_goods_rows,
            supplier_eta_date
        from inventory
        where reorder_flag in ('critical', 'warning')
        {keyword_clause}
        order by
            case reorder_flag when 'critical' then 0 when 'warning' then 1 else 2 end,
            coalesce(days_of_stock, 999999) asc,
            coalesce(stock_qty, 0) asc
        limit %s
    """
    overstock_query = inventory_cte + f"""
        select
            report_date,
            nm_id,
            vendor_code,
            brand_name,
            subject_name,
            stock_qty,
            stock_qty_full,
            unavailable_stock_qty,
            sales_7d_units,
            avg_daily_sales_7d,
            days_of_stock,
            total_stock_days_for_audit,
            reorder_flag
        from inventory
        where reorder_flag = 'overstock'
        {keyword_clause}
        order by coalesce(stock_qty, 0) desc, nm_id
        limit %s
    """
    rows_query = inventory_cte + f"""
        select
            report_date,
            nm_id,
            vendor_code,
            brand_name,
            subject_name,
            stock_qty,
            stock_qty_full,
            unavailable_stock_qty,
            in_way_to_client,
            in_way_from_client,
            supplier_audit_snapshot_date,
            supplier_inbound_qty,
            supplier_goods_rows,
            supplier_eta_date,
            sales_1d_units,
            sales_7d_units,
            avg_daily_sales_7d,
            days_of_stock,
            total_stock_days_for_audit,
            reorder_flag,
            reorder_qty_recommended
        from inventory
        where 1 = 1
        {keyword_clause}
        order by
            case reorder_flag
                when 'critical' then 0
                when 'warning' then 1
                when 'ok' then 2
                when 'overstock' then 3
                else 4
            end,
            coalesce(days_of_stock, 999999) asc,
            coalesce(stock_qty, 0) asc,
            nm_id
        limit %s
    """
    pipeline_query = """
        with latest_day as (
            select {latest_snapshot_date} as snapshot_date
            from supply.wb_supplies
        )
        select
            s.snapshot_date,
            s.supply_id,
            s.supply_date,
            s.status_id,
            s.warehouse_name,
            s.quantity,
            s.ready_for_sale_quantity,
            s.accepted_quantity,
            s.unloading_quantity,
            count(distinct g.nm_id) as sku_count,
            sum(coalesce(g.quantity, 0)) as goods_quantity
        from supply.wb_supplies s
        left join supply.wb_supplies_goods g
          on g.snapshot_date = s.snapshot_date
         and g.supply_id = s.supply_id
        where s.snapshot_date = (select snapshot_date from latest_day)
          and (
                coalesce(s.quantity, 0) > coalesce(s.ready_for_sale_quantity, 0)
                or coalesce(s.unloading_quantity, 0) > 0
              )
        group by
            s.snapshot_date,
            s.supply_id,
            s.supply_date,
            s.status_id,
            s.warehouse_name,
            s.quantity,
            s.ready_for_sale_quantity,
            s.accepted_quantity,
            s.unloading_quantity
        order by coalesce(s.quantity, 0) desc, s.supply_id desc
        limit %s
    """.format(latest_snapshot_date=_latest_full_day_sql("snapshot_date"))
    stock_snapshot_query = f"""
        with latest_day as (
            select {_latest_full_day_sql('snapshot_date')} as snapshot_date
            from supply.wb_supplier_stocks
        )
        select
            snapshot_date,
            nm_id,
            supplier_article,
            brand,
            subject,
            sum(coalesce(quantity, 0)) as quantity,
            sum(coalesce(quantity_full, 0)) as quantity_full,
            sum(coalesce(in_way_to_client, 0)) as in_way_to_client,
            sum(coalesce(in_way_from_client, 0)) as in_way_from_client
        from supply.wb_supplier_stocks
        where snapshot_date = (select snapshot_date from latest_day)
        {stock_keyword_clause}
        group by snapshot_date, nm_id, supplier_article, brand, subject
        order by sum(coalesce(quantity_full, 0)) asc, nm_id
        limit %s
    """

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(totals_query, keyword_params)
        totals = serialize_value(cur.fetchone())

        cur.execute(critical_query, [*keyword_params, limit])
        critical = serialize_rows(cur.fetchall())

        cur.execute(overstock_query, [*keyword_params, limit])
        overstock = serialize_rows(cur.fetchall())

        cur.execute(rows_query, [*keyword_params, limit])
        rows = serialize_rows(cur.fetchall())

        cur.execute(pipeline_query, [limit])
        pipeline = serialize_rows(cur.fetchall())

        cur.execute(stock_snapshot_query, [*stock_keyword_params, limit])
        stock_snapshot_rows = serialize_rows(cur.fetchall())

    rows = [normalize_current_coverage_row(row) for row in rows]
    critical = [normalize_current_coverage_row(row) for row in critical]
    overstock = [normalize_current_coverage_row(row) for row in overstock]

    simulation_enabled = get_settings().supply_latest_known_as_current
    lookup = _build_supply_stock_lookup(product_keyword, rows)
    source_snapshot_date = _latest_supply_snapshot_date(
        totals,
        rows,
        stock_snapshot_rows,
    )
    data_mode = (
        "simulation_latest_known"
        if simulation_enabled
        else "observed_latest_known"
    )
    simulation = {
        "enabled": simulation_enabled,
        "policy": "latest_known_as_current",
        "source_snapshot_date": source_snapshot_date,
        "effective_status": (
            "simulated_current"
            if simulation_enabled and source_snapshot_date is not None
            else "observed_latest_known"
        ),
    }
    lookup = {
        **lookup,
        "data_mode": data_mode,
        "simulation": simulation,
    }
    return {
        "focus_keyword": product_keyword,
        "data_mode": data_mode,
        "simulation": simulation,
        "sku_stock_lookup": lookup,
        "totals": totals,
        "rows": rows,
        "critical_reorder_rows": critical,
        "overstock_rows": overstock,
        "pipeline_rows": pipeline,
        "stock_snapshot_rows": stock_snapshot_rows,
        "calculation_methods": {
            "days_of_stock": "quantity / (sales_7d_units / 7); available stock days, verified from latest stock snapshot and latest 7 full sales days",
            "coverage_band": "deterministic thresholds: deficit <14 days; normal 14-30; moderate excess >30-60; critical excess >60",
            "availability_status": "deterministic current sellable-stock state, kept separate from historical coverage",
            "total_stock_days_for_audit": "quantity_full / (sales_7d_units / 7); audit-only coverage including unavailable units, not user-facing available stock",
            "reorder_flag": "based on available-stock days_of_stock thresholds: critical < 7, warning < 21",
            "supplier_inbound_qty": "sum(max(goods.quantity - goods.ready_for_sale_quantity, 0)) on the latest supply audit snapshot; an audited absent SKU is explicit zero",
        },
    }


def _latest_supply_snapshot_date(
    totals: dict[str, Any] | None,
    rows: list[dict[str, Any]],
    stock_snapshot_rows: list[dict[str, Any]],
) -> Any | None:
    candidates = [
        (totals or {}).get("report_date"),
        *(row.get("report_date") for row in rows),
        *(row.get("snapshot_date") for row in stock_snapshot_rows),
    ]
    known = [candidate for candidate in candidates if candidate is not None]
    return max(known, key=str) if known else None


def _build_supply_stock_lookup(
    product_keyword: str | None,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build an explicit no-ambiguity contract from aggregated overview rows."""

    matches = [
        {
            "snapshot_date": row.get("report_date"),
            "nm_id": row.get("nm_id"),
            "vendor_code": row.get("vendor_code"),
            "available_stock_qty": row.get("stock_qty"),
            "full_stock_qty": row.get("stock_qty_full"),
            "unavailable_stock_qty": row.get("unavailable_stock_qty"),
            "in_way_to_client": row.get("in_way_to_client"),
            "in_way_from_client": row.get("in_way_from_client"),
            "supplier_audit_snapshot_date": row.get("supplier_audit_snapshot_date"),
            "supplier_inbound_qty": row.get("supplier_inbound_qty"),
            "supplier_goods_rows": row.get("supplier_goods_rows"),
            "supplier_eta_date": row.get("supplier_eta_date"),
        }
        for row in rows
    ]
    base = {
        "query": product_keyword,
        "match_count": len(matches),
        "aggregation_method": (
            "sum(quantity) across all warehouse rows for each SKU on the latest snapshot"
        ),
        "matches": matches,
    }
    requested_skus = _parse_sku_tokens(product_keyword)
    if len(requested_skus) > 1:
        matched_queries = [
            token
            for token in requested_skus
            if any(
                token == str(match.get("nm_id"))
                or token == str(match.get("vendor_code"))
                for match in matches
            )
        ]
        missing_queries = [
            token for token in requested_skus if token not in matched_queries
        ]
        return {
            **base,
            "status": (
                "found_multiple"
                if matches and not missing_queries
                else "partial"
                if matches
                else "no_data"
            ),
            "requested_skus": requested_skus,
            "matched_queries": matched_queries,
            "missing_queries": missing_queries,
            "snapshot_date": max(
                (
                    match.get("snapshot_date")
                    for match in matches
                    if match.get("snapshot_date") is not None
                ),
                key=str,
                default=None,
            ),
            "nm_id": None,
            "vendor_code": None,
            "available_stock_qty": None,
            "full_stock_qty": None,
        }
    if not product_keyword:
        return {**base, "status": "not_requested", "available_stock_qty": None}
    if not matches:
        return {
            **base,
            "status": "no_data",
            "snapshot_date": None,
            "nm_id": None,
            "vendor_code": None,
            "available_stock_qty": None,
            "full_stock_qty": None,
        }
    if len(matches) > 1:
        return {
            **base,
            "status": "multiple_matches",
            "snapshot_date": None,
            "nm_id": None,
            "vendor_code": None,
            "available_stock_qty": None,
            "full_stock_qty": None,
        }

    match = matches[0]
    return {
        **base,
        "status": "found",
        "snapshot_date": match["snapshot_date"],
        "nm_id": match["nm_id"],
        "vendor_code": match["vendor_code"],
        "available_stock_qty": match["available_stock_qty"],
        "full_stock_qty": match["full_stock_qty"],
        "unavailable_stock_qty": match["unavailable_stock_qty"],
        "in_way_to_client": match["in_way_to_client"],
        "in_way_from_client": match["in_way_from_client"],
    }


def get_current_inventory_value(
    product_keyword: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """Value current sellable on-hand stock using the latest known unit cost.

    This deliberately uses `quantity`, never sold units, `quantity_full`, or
    customer transit. Rows without a confirmed cost remain unvalued instead of
    being coerced to zero.
    """

    limit = _validate_limit(limit, upper=100)
    keyword_clause, keyword_params = _keyword_clause(
        product_keyword,
        ["st.vendor_code", "st.brand_name", "st.subject_name", "st.nm_id::text"],
    )
    query = f"""
        with latest_stock_day as (
            select {_latest_full_day_sql('snapshot_date')} as snapshot_date
            from supply.wb_supplier_stocks
        ),
        stocks as (
            select
                snapshot_date,
                nm_id,
                supplier_article as vendor_code,
                brand as brand_name,
                subject as subject_name,
                sum(coalesce(quantity, 0)) as available_stock_qty
            from supply.wb_supplier_stocks
            where snapshot_date = (select snapshot_date from latest_stock_day)
            group by snapshot_date, nm_id, supplier_article, brand, subject
        )
        select
            st.snapshot_date as report_date,
            st.nm_id,
            st.vendor_code,
            st.brand_name,
            st.subject_name,
            st.available_stock_qty,
            round(cost.cost_price_rub::numeric, 2) as unit_cost_rub,
            case
                when cost.cost_price_rub is not null
                    then round((st.available_stock_qty * cost.cost_price_rub)::numeric, 2)
            end as current_stock_value_rub,
            case when cost.cost_price_rub is null then 'missing_cogs' else 'confirmed' end as valuation_status
        from stocks st
        left join lateral (
            select c.cost_price_rub
            from finance.wb_product_coast c
            where c.nm_id = st.nm_id
               or nullif(c.vendor_code, '') = st.vendor_code
            order by
                case when c.nm_id = st.nm_id then 0 else 1 end,
                (c.upload_date <= st.snapshot_date) desc,
                case when c.upload_date <= st.snapshot_date then c.upload_date end desc nulls last,
                c.upload_date desc,
                c.updated_at desc
            limit 1
        ) cost on true
        where 1 = 1
        {keyword_clause}
        order by current_stock_value_rub desc nulls last, st.available_stock_qty desc, st.nm_id
        limit %s
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(query, [*keyword_params, limit])
        rows = serialize_rows(cur.fetchall())

    confirmed_values = [
        float(row["current_stock_value_rub"])
        for row in rows
        if row.get("current_stock_value_rub") is not None
    ]
    return {
        "report_date": rows[0].get("report_date") if rows else None,
        "focus_keyword": product_keyword,
        "rows": rows,
        "totals": {
            "confirmed_stock_value_rub": round(sum(confirmed_values), 2),
            "valued_sku_count": len(confirmed_values),
            "missing_cogs_sku_count": sum(
                1 for row in rows if row.get("valuation_status") == "missing_cogs"
            ),
        },
        "calculation_method": "current_stock_value_rub = sellable on-hand quantity * confirmed unit_cost_rub",
    }


def _inventory_turnover_row_conclusion(row: dict[str, Any]) -> str:
    sales_units = float(row.get("sales_units") or 0)
    days_on_hand = row.get("days_on_hand")
    current_available = float(row.get("current_stock_qty") or 0)
    try:
        days = float(days_on_hand) if days_on_hand is not None else None
    except (TypeError, ValueError):
        days = None

    if sales_units <= 0 and current_available > 0:
        return "есть остаток без продаж за период"
    if current_available <= 0 and sales_units > 0:
        return "продажи были, текущий доступный остаток отсутствует"
    return classify_coverage_days(days)


def get_inventory_turnover(
    date_from: str,
    date_to: str | None = None,
    vendor_codes: str | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    start_date, end_date = _parse_report_period(date_from=date_from, date_to=date_to)
    limit = _validate_period_limit(limit)
    if start_date is None or end_date is None:
        raise ValueError("date_from must not be empty")
    if end_date < start_date:
        raise ValueError("date_to must be greater than or equal to date_from")

    sku_tokens = _parse_sku_tokens(vendor_codes)
    sku_filter_sql = ""
    sku_filter_params: list[Any] = []
    if sku_tokens:
        sku_filter_sql = "and supplier_article = any(%s::text[])"
        sku_filter_params.append(sku_tokens)

    stock_query = f"""
        with params as (
            select %s::date as date_from, %s::date as date_to
        ),
        stock_daily as (
            select
                s.snapshot_date::date as snapshot_date,
                s.nm_id,
                s.supplier_article as vendor_code,
                max(nullif(s.brand, '')) as brand_name,
                max(nullif(s.subject, '')) as subject_name,
                sum(coalesce(s.quantity, 0)) as stock_qty,
                sum(coalesce(s.quantity_full, 0)) as stock_qty_full
            from supply.wb_supplier_stocks s
            join params p on true
            where s.snapshot_date::date between p.date_from and p.date_to
              {sku_filter_sql}
            group by s.snapshot_date::date, s.nm_id, s.supplier_article
        ),
        stock_by_sku as (
            select
                nm_id,
                vendor_code,
                max(brand_name) as brand_name,
                max(subject_name) as subject_name,
                count(*) as stock_snapshot_days,
                round(avg(stock_qty)::numeric, 2) as avg_stock_qty,
                round(avg(stock_qty_full)::numeric, 2) as avg_stock_qty_full,
                min(stock_qty_full) as min_stock_qty_full,
                max(stock_qty_full) as max_stock_qty_full
            from stock_daily
            group by nm_id, vendor_code
        )
        select * from stock_by_sku
    """
    sales_query = f"""
        with params as (
            select %s::date as date_from, %s::date as date_to
        )
        select
            s.nm_id,
            s.supplier_article as vendor_code,
            count(*) filter (where s.is_realization) as sales_units,
            sum(coalesce(s.for_pay, 0)) filter (where s.is_realization) as sales_revenue
        from sales.wb_supplier_sales s
        join params p on true
        where s.date_ord::date between p.date_from and p.date_to
          {sku_filter_sql}
        group by s.nm_id, s.supplier_article
    """
    rows_query = f"""
        with params as (
            select %s::date as date_from, %s::date as date_to
        ),
        stock_daily as (
            select
                s.snapshot_date::date as snapshot_date,
                s.nm_id,
                s.supplier_article as vendor_code,
                max(nullif(s.brand, '')) as brand_name,
                max(nullif(s.subject, '')) as subject_name,
                sum(coalesce(s.quantity, 0)) as stock_qty,
                sum(coalesce(s.quantity_full, 0)) as stock_qty_full
            from supply.wb_supplier_stocks s
            join params p on true
            where s.snapshot_date::date between p.date_from and p.date_to
              {sku_filter_sql}
            group by s.snapshot_date::date, s.nm_id, s.supplier_article
        ),
        stock_by_sku as (
            select
                nm_id,
                vendor_code,
                max(brand_name) as brand_name,
                max(subject_name) as subject_name,
                count(*) as stock_snapshot_days,
                round(avg(stock_qty)::numeric, 2) as avg_stock_qty,
                round(avg(stock_qty_full)::numeric, 2) as avg_stock_qty_full,
                min(stock_qty_full) as min_stock_qty_full,
                max(stock_qty_full) as max_stock_qty_full
            from stock_daily
            group by nm_id, vendor_code
        ),
        sales_by_sku as (
            select
                s.nm_id,
                s.supplier_article as vendor_code,
                count(*) filter (where s.is_realization) as sales_units,
                count(*) filter (
                    where s.is_realization
                      and s.date_ord::date between
                          (select date_to - interval '6 day' from params)
                          and (select date_to from params)
                ) as sales_7d_units,
                round((sum(coalesce(s.for_pay, 0)) filter (where s.is_realization))::numeric, 2) as sales_revenue
            from sales.wb_supplier_sales s
            join params p on true
            where s.date_ord::date between p.date_from and p.date_to
              {sku_filter_sql}
            group by s.nm_id, s.supplier_article
        ),
        latest_stock_day as (
            select max(s.snapshot_date::date) as latest_snapshot_date
            from supply.wb_supplier_stocks s
            where true
              {sku_filter_sql}
        ),
        current_stock_by_sku as (
            select
                s.snapshot_date::date as latest_snapshot_date,
                s.nm_id,
                s.supplier_article as vendor_code,
                sum(coalesce(s.quantity, 0)) as current_stock_qty,
                sum(coalesce(s.quantity_full, 0)) as current_stock_qty_full,
                sum(coalesce(s.in_way_to_client, 0)) as current_in_way_to_client,
                sum(coalesce(s.in_way_from_client, 0)) as current_in_way_from_client
            from supply.wb_supplier_stocks s
            join latest_stock_day d
              on d.latest_snapshot_date = s.snapshot_date::date
            where true
              {sku_filter_sql}
            group by s.snapshot_date::date, s.nm_id, s.supplier_article
        ),
        joined as (
            select
                coalesce(st.nm_id, sa.nm_id, cs.nm_id) as nm_id,
                coalesce(st.vendor_code, sa.vendor_code, cs.vendor_code) as vendor_code,
                st.brand_name,
                st.subject_name,
                coalesce(sa.sales_units, 0) as sales_units,
                coalesce(sa.sales_7d_units, 0) as sales_7d_units,
                coalesce(sa.sales_revenue, 0) as sales_revenue,
                st.stock_snapshot_days,
                st.avg_stock_qty,
                st.avg_stock_qty_full,
                st.min_stock_qty_full,
                st.max_stock_qty_full,
                cs.latest_snapshot_date,
                coalesce(cs.current_stock_qty, 0) as current_stock_qty,
                coalesce(cs.current_stock_qty_full, 0) as current_stock_qty_full,
                coalesce(cs.current_in_way_to_client, 0) as current_in_way_to_client,
                coalesce(cs.current_in_way_from_client, 0) as current_in_way_from_client,
                case
                    when st.avg_stock_qty > 0
                        then round(coalesce(sa.sales_units, 0)::numeric / st.avg_stock_qty, 4)
                end as turnover_ratio,
                case
                    when coalesce(sa.sales_units, 0) > 0
                        then round(st.avg_stock_qty / (coalesce(sa.sales_units, 0)::numeric / greatest((p.date_to - p.date_from + 1), 1)), 1)
                end as days_on_hand
            from stock_by_sku st
            full outer join sales_by_sku sa
              on sa.nm_id = st.nm_id
             and sa.vendor_code = st.vendor_code
            full outer join current_stock_by_sku cs
              on cs.nm_id = coalesce(st.nm_id, sa.nm_id)
             and cs.vendor_code = coalesce(st.vendor_code, sa.vendor_code)
            cross join params p
        )
        select *
        from joined
        order by coalesce(sales_units, 0) desc, vendor_code nulls last, nm_id
        limit %s
    """
    totals_query = f"""
        with params as (
            select %s::date as date_from, %s::date as date_to
        ),
        stock_daily as (
            select
                s.snapshot_date::date as snapshot_date,
                sum(coalesce(s.quantity, 0)) as stock_qty,
                sum(coalesce(s.quantity_full, 0)) as stock_qty_full
            from supply.wb_supplier_stocks s
            join params p on true
            where s.snapshot_date::date between p.date_from and p.date_to
              {sku_filter_sql}
            group by s.snapshot_date::date
        ),
        stock_total as (
            select
                count(*) as stock_snapshot_days,
                round(avg(stock_qty)::numeric, 2) as avg_stock_qty,
                round(avg(stock_qty_full)::numeric, 2) as avg_stock_qty_full
            from stock_daily
        ),
        sales_total as (
            select
                count(*) filter (where s.is_realization) as sales_units,
                round((sum(coalesce(s.for_pay, 0)) filter (where s.is_realization))::numeric, 2) as sales_revenue
            from sales.wb_supplier_sales s
            join params p on true
            where s.date_ord::date between p.date_from and p.date_to
              {sku_filter_sql}
        ),
        latest_stock_day as (
            select max(s.snapshot_date::date) as latest_snapshot_date
            from supply.wb_supplier_stocks s
            where true
              {sku_filter_sql}
        ),
        current_stock_total as (
            select
                d.latest_snapshot_date,
                sum(coalesce(s.quantity, 0)) as current_stock_qty,
                sum(coalesce(s.quantity_full, 0)) as current_stock_qty_full,
                sum(coalesce(s.in_way_to_client, 0)) as current_in_way_to_client,
                sum(coalesce(s.in_way_from_client, 0)) as current_in_way_from_client
            from latest_stock_day d
            left join supply.wb_supplier_stocks s
              on s.snapshot_date::date = d.latest_snapshot_date
             {sku_filter_sql}
            group by d.latest_snapshot_date
        )
        select
            p.date_from,
            p.date_to,
            greatest((p.date_to - p.date_from + 1), 1) as period_days,
            st.stock_snapshot_days,
            st.avg_stock_qty,
            st.avg_stock_qty_full,
            coalesce(sa.sales_units, 0) as sales_units,
            coalesce(sa.sales_revenue, 0) as sales_revenue,
            case
                when st.avg_stock_qty > 0
                    then round(coalesce(sa.sales_units, 0)::numeric / st.avg_stock_qty, 4)
            end as turnover_ratio,
            case
                when coalesce(sa.sales_units, 0) > 0
                    then round(st.avg_stock_qty / (coalesce(sa.sales_units, 0)::numeric / greatest((p.date_to - p.date_from + 1), 1)), 1)
            end as days_on_hand,
            cst.latest_snapshot_date,
            coalesce(cst.current_stock_qty, 0) as current_stock_qty,
            coalesce(cst.current_stock_qty_full, 0) as current_stock_qty_full,
            coalesce(cst.current_in_way_to_client, 0) as current_in_way_to_client,
            coalesce(cst.current_in_way_from_client, 0) as current_in_way_from_client
        from params p
        cross join stock_total st
        cross join sales_total sa
        cross join current_stock_total cst
    """

    single_scope_params: list[Any] = [start_date, end_date, *sku_filter_params]
    double_scope_params: list[Any] = [
        start_date,
        end_date,
        *sku_filter_params,
        *sku_filter_params,
    ]
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(rows_query, [*double_scope_params, *sku_filter_params, *sku_filter_params, limit])
        rows = serialize_rows(cur.fetchall())

        cur.execute(totals_query, [*double_scope_params, *sku_filter_params, *sku_filter_params])
        totals = serialize_value(cur.fetchone())

        cur.execute(stock_query, single_scope_params)
        stock_rows = serialize_rows(cur.fetchall())

        cur.execute(sales_query, single_scope_params)
        sales_rows = serialize_rows(cur.fetchall())

    exact_available = bool(
        totals
        and totals.get("avg_stock_qty")
        and totals.get("stock_snapshot_days")
        and totals.get("sales_units") is not None
    )
    alternative_methods = [
        {
            "name": "boundary_snapshot_average",
            "meaning": "оценить средний остаток как (остаток около начала периода + остаток около конца периода) / 2",
            "when_to_use": "если нет ежедневного среднего остатка внутри периода, но есть снимки рядом с границами периода",
        },
        {
            "name": "movement_reconstruction",
            "meaning": "восстановить остаток по ближайшему снимку, продажам и поставкам/движениям за период",
            "when_to_use": "если есть исходный остаток и события, которыми можно объяснить изменение запаса",
        },
        {
            "name": "latest_stock_velocity_proxy",
            "meaning": "оценить дни запаса как текущий доступный остаток / среднедневные продажи",
            "when_to_use": "если исторический средний запас не собрать, но есть текущий остаток и продажи за период",
        },
    ]
    limitation = None
    if not exact_available:
        limitation = (
            "Прямой расчет по дневному среднему остатку не получился. Не закрывай ответ этим ограничением: "
            "сначала проверь альтернативные методы из alternative_methods; если они тоже не дают данных, "
            "покажи proxy отдельно и явно назови его оценкой."
        )
    period_days = max((end_date - start_date).days + 1, 1)
    rows = [normalize_turnover_row(row, period_days=period_days) for row in rows]
    if totals:
        totals["days_on_hand"] = calculate_coverage_days(
            stock_qty=totals.get("avg_stock_qty"),
            sales_units=totals.get("sales_units"),
            period_days=period_days,
        )
        totals["turnover_band"] = classify_coverage_days(totals.get("days_on_hand"))
        totals["availability_status"] = classify_availability(
            totals.get("current_stock_qty")
        )
    for row in rows:
        row["sku_conclusion"] = _inventory_turnover_row_conclusion(row)

    sku_metric_matrix = [
        {
            "sku": row.get("vendor_code"),
            "sales_units": row.get("sales_units"),
            "sales_7d_units": row.get("sales_7d_units"),
            "avg_stock_qty": row.get("avg_stock_qty"),
            "avg_stock_qty_full": row.get("avg_stock_qty_full"),
            "current_stock_qty": row.get("current_stock_qty"),
            "current_stock_qty_full": row.get("current_stock_qty_full"),
            "current_in_way_to_client": row.get("current_in_way_to_client"),
            "current_in_way_from_client": row.get("current_in_way_from_client"),
            "turnover_ratio": row.get("turnover_ratio"),
            "days_on_hand": row.get("days_on_hand"),
            "turnover_band": row.get("turnover_band"),
            "availability_status": row.get("availability_status"),
            "conclusion": row.get("sku_conclusion"),
        }
        for row in rows
    ]

    return {
        "date_from": start_date,
        "date_to": end_date,
        "vendor_codes": sku_tokens,
        "method": "primary: sales_units / average_sellable_stock_qty from daily stock snapshots over selected period",
        "coverage_formula": "avg_stock_qty / (sales_units / period_days)",
        "coverage_band_policy": {
            "deficit": "<14 days",
            "normal": "14-30 days",
            "moderate_excess": ">30-60 days",
            "critical_excess": ">60 days",
        },
        "calculation_status": "exact" if exact_available else "needs_alternative_search",
        "exact_available": exact_available,
        "limitation": limitation,
        "alternative_methods": alternative_methods,
        "totals": totals,
        "rows": rows,
        "sku_metric_matrix": sku_metric_matrix,
        "sku_metric_matrix_contract": (
            "For every returned SKU row, final answers must preserve the calculated values "
            "and include the per-SKU conclusion; do not collapse SKU rows into only portfolio totals."
        ),
        "source_coverage": {
            "stock_rows": len(stock_rows),
            "sales_rows": len(sales_rows),
        },
    }


def get_marketing_overview(product_keyword: str | None = None, limit: int = 5) -> dict[str, Any]:
    limit = _validate_limit(limit, upper=10)
    campaign_clause, campaign_params = _keyword_clause(
        product_keyword,
        ["campaign_name", "top_subject_name", "advert_id::text"],
    )

    totals_query = f"""
        with {MARKETING_CAMPAIGN_DAILY_CTE},
        latest_day as (
            select {_latest_full_day_sql("stat_date")} as stat_date
            from agent_campaign_daily
        )
        select
            d.stat_date,
            count(distinct advert_id) as campaign_count_7d,
            sum(coalesce(views, 0)) as total_views_7d,
            sum(coalesce(clicks, 0)) as total_clicks_7d,
            sum(coalesce(orders, 0)) as total_orders_7d,
            sum(coalesce(sum_spend, 0)) as total_spend_7d,
            avg(nullif(ctr, 0)) as avg_ctr,
            avg(nullif(cr, 0)) as avg_cr
        from agent_campaign_daily s
        join latest_day d
          on s.stat_date between d.stat_date - interval '6 day' and d.stat_date
        group by d.stat_date
    """
    campaigns_query = f"""
        with {MARKETING_CAMPAIGN_DAILY_CTE},
        latest_stats_day as (
            select {_latest_full_day_sql('stat_date')} as stat_date
            from agent_campaign_daily
        ),
        stats_7d as (
            select
                advert_id,
                max(campaign_name) as campaign_name,
                max(top_subject_name) as top_subject_name,
                sum(coalesce(views, 0)) as views,
                sum(coalesce(clicks, 0)) as clicks,
                sum(coalesce(orders, 0)) as orders,
                sum(coalesce(sum_spend, 0)) as sum_spend,
                avg(nullif(ctr, 0)) as ctr,
                avg(nullif(cpc, 0)) as cpc,
                avg(nullif(cr, 0)) as cr
            from agent_campaign_daily
            where stat_date between (select stat_date - interval '6 day' from latest_stats_day)
                                and (select stat_date from latest_stats_day)
            group by advert_id
        )
        select
            (select stat_date from latest_stats_day) as stat_date,
            advert_id,
            campaign_name,
            top_subject_name as subject_name,
            views,
            clicks,
            orders,
            sum_spend,
            ctr,
            cpc,
            cr
        from stats_7d
        where 1 = 1
        {campaign_clause}
        order by coalesce(sum_spend, 0) desc, coalesce(orders, 0) desc
        limit %s
    """
    inefficient_query = f"""
        with {MARKETING_CAMPAIGN_DAILY_CTE},
        latest_stats_day as (
            select {_latest_full_day_sql('stat_date')} as stat_date
            from agent_campaign_daily
        ),
        stats_7d as (
            select
                advert_id,
                max(campaign_name) as campaign_name,
                max(top_subject_name) as top_subject_name,
                sum(coalesce(views, 0)) as views,
                sum(coalesce(clicks, 0)) as clicks,
                sum(coalesce(orders, 0)) as orders,
                sum(coalesce(sum_spend, 0)) as sum_spend,
                avg(nullif(cpc, 0)) as cpc,
                avg(nullif(cr, 0)) as cr
            from agent_campaign_daily
            where stat_date between (select stat_date - interval '6 day' from latest_stats_day)
                                and (select stat_date from latest_stats_day)
            group by advert_id
        )
        select
            (select stat_date from latest_stats_day) as stat_date,
            advert_id,
            campaign_name,
            top_subject_name as subject_name,
            views,
            clicks,
            orders,
            sum_spend,
            cpc,
            cr
        from stats_7d
        where coalesce(sum_spend, 0) > 0
        {campaign_clause}
        order by
            case when coalesce(orders, 0) = 0 then 0 else 1 end,
            coalesce(sum_spend, 0) desc
        limit %s
    """
    campaign_scope_query = f"""
        with {MARKETING_CAMPAIGN_DAILY_CTE},
        latest_day as (
            select {_latest_full_day_sql('stat_date')} as stat_date
            from agent_campaign_daily
        )
        select
            stat_date,
            advert_id,
            campaign_name,
            top_subject_name as subject_name,
            active_nms_count,
            views,
            clicks,
            orders,
            sum_spend,
            ctr,
            cpc,
            cr
        from agent_campaign_daily
        where stat_date = (select stat_date from latest_day)
        {campaign_clause}
        order by coalesce(sum_spend, 0) desc, coalesce(orders, 0) desc, advert_id
        limit %s
    """
    budget_query = f"""
        with {MARKETING_CAMPAIGN_DAILY_CTE},
        latest_day as (
            select {_latest_full_day_sql("stat_date")} as stat_date
            from agent_campaign_daily
        )
        select
            stat_date as snapshot_date,
            count(distinct advert_id) as advert_count,
            sum(coalesce(budget_total, 0)) as total_budget,
            sum(coalesce(cash, 0)) as total_cash
        from agent_campaign_daily
        where stat_date = (select stat_date from latest_day)
        group by stat_date
    """

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(totals_query)
        totals = serialize_value(cur.fetchone())

        cur.execute(campaigns_query, [*campaign_params, limit])
        top_spend = serialize_rows(cur.fetchall())

        cur.execute(inefficient_query, [*campaign_params, limit])
        inefficient = serialize_rows(cur.fetchall())

        cur.execute(campaign_scope_query, [*campaign_params, limit])
        campaign_scope_rows = serialize_rows(cur.fetchall())

        cur.execute(budget_query)
        budget = serialize_value(cur.fetchone())

    return {
        "focus_keyword": product_keyword,
        "totals": totals,
        "budget": budget,
        "top_spend_campaigns": top_spend,
        "inefficient_campaigns": inefficient,
        "campaign_nm_rows": campaign_scope_rows,
    }


def get_marketing_campaign_period_report(
    date_from: str,
    date_to: str | None = None,
    limit: int = 200,
) -> dict[str, Any]:
    start_date, end_date = _parse_report_period(date_from=date_from, date_to=date_to)
    limit = _validate_period_limit(limit)

    if start_date is None or end_date is None:
        raise ValueError("date_from must not be empty")
    if end_date < start_date:
        raise ValueError("date_to must be greater than or equal to date_from")

    query = """
        with params as (
            select
                %s::date as date_from,
                %s::date as date_to
        ),
        latest_campaigns as (
            select distinct on (external_id)
                external_id as advert_id,
                name,
                campaign_type,
                payment_type,
                status,
                started_at
            from marketing.wb_marketing_campaigns
            order by external_id, snapshot_date desc, last_synced_at desc
        ),
        campaign_skus as (
            select
                advert_id,
                nm_id
            from (
                select distinct on (advert_id, nm_id)
                    advert_id,
                    nm_id
                from marketing.wb_marketing_campaign_nms
                order by advert_id, nm_id, snapshot_date desc, last_synced_at desc
            ) t
        ),
        sku_prices as (
            select
                o.nm_id,
                round(avg(o.price_with_disc)::numeric, 2) as sale_price_without_spp
            from sales.wb_supplier_orders o
            join params p on true
            where o.date_ord::date between p.date_from and p.date_to
            group by o.nm_id
        ),
        campaign_sku_price_strings as (
            select
                cs.advert_id,
                string_agg(cs.nm_id::text, ', ' order by cs.nm_id) as sku_list,
                string_agg(
                    cs.nm_id::text || ': ' ||
                    coalesce(to_char(sp.sale_price_without_spp, 'FM999999990.00'), 'н/д'),
                    '; ' order by cs.nm_id
                ) as sku_price_list
            from campaign_skus cs
            left join sku_prices sp on sp.nm_id = cs.nm_id
            group by cs.advert_id
        ),
        agg as (
            select
                s.advert_id,
                sum(s.views) as impressions,
                sum(s.clicks) as clicks,
                round(
                    case when sum(s.views) > 0
                        then sum(s.clicks)::numeric * 100 / sum(s.views)
                    end, 2
                ) as ctr_pct,
                round(
                    case when sum(s.clicks) > 0
                        then sum(s.sum_spend)::numeric / sum(s.clicks)
                    end, 2
                ) as cpc_rub,
                round(
                    case when sum(s.clicks) > 0
                        then sum(s.orders)::numeric * 100 / sum(s.clicks)
                    end, 2
                ) as cr_pct,
                sum(s.atbs) as all_add_to_cart,
                sum(s.shks) as all_ordered_items,
                round(sum(s.sum_spend)::numeric, 2) as spend_rub
            from marketing.wb_marketing_campaign_stats_daily s
            join params p on true
            where s.stat_date between p.date_from and p.date_to
            group by s.advert_id
        )
        select
            a.advert_id as "ID кампании",
            coalesce(lc.name, 'Без названия') || ', ' ||
                case lc.payment_type
                    when 'cpc' then 'Ручная ставка'
                    when 'cpm' then 'Единая ставка'
                    else coalesce(lc.campaign_type, 'тип не указан')
                end as "Кампания, Тип",
            lc.started_at::date as "Дата начала",
            case lc.status
                when 4 then 'Готово к запуску'
                when 7 then 'Завершена'
                when 8 then 'Отклонена'
                when 9 then 'Активна'
                when 11 then 'Приостановлена'
                else coalesce(lc.status::text, 'нет статуса')
            end as "Статус",
            csp.sku_list as "Артикулы в кампании",
            csp.sku_price_list as "Цена продажи без СПП, ₽",
            a.impressions as "Показы",
            a.clicks as "Клики",
            a.ctr_pct as "Кликабельность, %%",
            a.cpc_rub as "Цена клика, ₽",
            a.cr_pct as "Конверсия клика в заказ, %%",
            a.all_add_to_cart as "Все добавления в корзину",
            a.all_ordered_items as "Все заказанные товары, шт",
            a.spend_rub as "Затраты, ₽",
            round(
                case when sum(a.spend_rub) over () > 0
                    then a.spend_rub * 100 / sum(a.spend_rub) over ()
                end, 2
            ) as "Доля затрат (%%)"
        from agg a
        left join latest_campaigns lc
            on lc.advert_id = a.advert_id
        left join campaign_sku_price_strings csp
            on csp.advert_id = a.advert_id
        order by a.spend_rub desc
        limit %s
    """

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(query, [start_date, end_date, limit])
        rows = serialize_rows(cur.fetchall())

    return {
        "date_from": start_date.isoformat(),
        "date_to": end_date.isoformat(),
        "source_tables": [
            "marketing.wb_marketing_campaigns",
            "marketing.wb_marketing_campaign_nms",
            "marketing.wb_marketing_campaign_stats_daily",
            "sales.wb_supplier_orders",
        ],
        "row_count": len(rows),
        "rows": rows,
    }


def get_marketing_drr_report(
    date_from: str,
    date_to: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    start_date, end_date = _parse_report_period(date_from=date_from, date_to=date_to)
    limit = _validate_period_limit(limit, upper=100)

    if start_date is None or end_date is None:
        raise ValueError("date_from must not be empty")
    if end_date < start_date:
        raise ValueError("date_to must be greater than or equal to date_from")

    totals_query = """
        with params as (
            select %s::date as date_from, %s::date as date_to
        ),
        campaign_spend as (
            select
                advert_id,
                round(sum(coalesce(sum_spend, 0))::numeric, 2) as spend_rub,
                sum(coalesce(views, 0)) as views,
                sum(coalesce(clicks, 0)) as clicks,
                sum(coalesce(orders, 0)) as ad_orders
            from marketing.wb_marketing_campaign_stats_daily s
            join params p on true
            where s.stat_date between p.date_from and p.date_to
            group by advert_id
            having sum(coalesce(sum_spend, 0)) > 0
        ),
        latest_campaign_nms as (
            select
                advert_id,
                nm_id
            from (
                select distinct on (advert_id, nm_id)
                    advert_id,
                    nm_id
                from marketing.wb_marketing_campaign_nms
                order by advert_id, nm_id, snapshot_date desc, last_synced_at desc
            ) t
        ),
        promoted_nms as (
            select distinct n.nm_id
            from campaign_spend s
            join latest_campaign_nms n using (advert_id)
        ),
        sales_totals as (
            select
                round(sum(coalesce(for_pay, 0))::numeric, 2) as store_revenue,
                count(*)::int as store_sales,
                round(
                    sum(coalesce(for_pay, 0)) filter (
                        where nm_id in (select nm_id from promoted_nms)
                    )::numeric,
                    2
                ) as promoted_sku_revenue,
                count(*) filter (
                    where nm_id in (select nm_id from promoted_nms)
                )::int as promoted_sku_sales
            from sales.wb_supplier_sales s
            join params p on true
            where s.date_ord::date between p.date_from and p.date_to
              and s.is_realization = true
        ),
        marketing_totals as (
            select
                count(*)::int as campaign_count,
                round(sum(spend_rub)::numeric, 2) as marketing_spend,
                sum(views)::int as views,
                sum(clicks)::int as clicks,
                sum(ad_orders)::int as ad_orders
            from campaign_spend
        )
        select
            mt.campaign_count,
            (select count(*)::int from promoted_nms) as promoted_nm_count,
            mt.marketing_spend,
            mt.views,
            mt.clicks,
            mt.ad_orders,
            st.store_revenue,
            st.store_sales,
            coalesce(st.promoted_sku_revenue, 0) as promoted_sku_revenue,
            coalesce(st.promoted_sku_sales, 0) as promoted_sku_sales,
            round(
                case when st.store_revenue > 0
                    then coalesce(st.promoted_sku_revenue, 0) * 100 / st.store_revenue
                end,
                2
            ) as promoted_revenue_share_pct,
            round(
                case when st.store_sales > 0
                    then coalesce(st.promoted_sku_sales, 0)::numeric * 100 / st.store_sales
                end,
                2
            ) as promoted_sales_share_pct,
            round(
                case when coalesce(st.promoted_sku_revenue, 0) > 0
                    then mt.marketing_spend * 100 / st.promoted_sku_revenue
                end,
                2
            ) as estimated_drr_promoted_sku_pct,
            round(
                case when st.store_revenue > 0
                    then mt.marketing_spend * 100 / st.store_revenue
                end,
                2
            ) as estimated_drr_store_pct
        from marketing_totals mt
        cross join sales_totals st
    """
    campaigns_query = """
        with params as (
            select %s::date as date_from, %s::date as date_to
        ),
        latest_campaigns as (
            select distinct on (external_id)
                external_id as advert_id,
                name
            from marketing.wb_marketing_campaigns
            order by external_id, snapshot_date desc, last_synced_at desc
        ),
        latest_campaign_nms as (
            select
                advert_id,
                nm_id
            from (
                select distinct on (advert_id, nm_id)
                    advert_id,
                    nm_id
                from marketing.wb_marketing_campaign_nms
                order by advert_id, nm_id, snapshot_date desc, last_synced_at desc
            ) t
        ),
        campaign_spend as (
            select
                s.advert_id,
                round(sum(coalesce(s.sum_spend, 0))::numeric, 2) as spend_rub,
                sum(coalesce(s.views, 0)) as views,
                sum(coalesce(s.clicks, 0)) as clicks,
                sum(coalesce(s.orders, 0)) as ad_orders
            from marketing.wb_marketing_campaign_stats_daily s
            join params p on true
            where s.stat_date between p.date_from and p.date_to
            group by s.advert_id
            having sum(coalesce(s.sum_spend, 0)) > 0
        )
        select
            cs.advert_id,
            coalesce(lc.name, 'Без названия') as campaign_name,
            cs.spend_rub,
            cs.views,
            cs.clicks,
            cs.ad_orders,
            round(case when cs.clicks > 0 then cs.spend_rub / cs.clicks end, 2) as cpc_rub,
            string_agg(distinct n.nm_id::text, ', ' order by n.nm_id::text) as nm_ids
        from campaign_spend cs
        left join latest_campaigns lc using (advert_id)
        left join latest_campaign_nms n using (advert_id)
        group by cs.advert_id, lc.name, cs.spend_rub, cs.views, cs.clicks, cs.ad_orders
        order by cs.spend_rub desc, cs.ad_orders desc, cs.advert_id
        limit %s
    """
    sku_query = """
        with params as (
            select %s::date as date_from, %s::date as date_to
        ),
        campaign_spend as (
            select advert_id
            from marketing.wb_marketing_campaign_stats_daily s
            join params p on true
            where s.stat_date between p.date_from and p.date_to
            group by advert_id
            having sum(coalesce(sum_spend, 0)) > 0
        ),
        latest_campaign_nms as (
            select
                advert_id,
                nm_id
            from (
                select distinct on (advert_id, nm_id)
                    advert_id,
                    nm_id
                from marketing.wb_marketing_campaign_nms
                order by advert_id, nm_id, snapshot_date desc, last_synced_at desc
            ) t
        ),
        promoted_nms as (
            select distinct n.nm_id
            from campaign_spend s
            join latest_campaign_nms n using (advert_id)
        )
        select
            s.nm_id,
            max(s.supplier_article) as vendor_code,
            max(s.subject) as subject_name,
            round(sum(coalesce(s.for_pay, 0))::numeric, 2) as revenue,
            count(*)::int as sales
        from sales.wb_supplier_sales s
        join promoted_nms p using (nm_id)
        join params prm on true
        where s.date_ord::date between prm.date_from and prm.date_to
          and s.is_realization = true
        group by s.nm_id
        order by revenue desc, sales desc, s.nm_id
        limit %s
    """

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(totals_query, [start_date, end_date])
        totals = serialize_value(cur.fetchone())

        cur.execute(campaigns_query, [start_date, end_date, limit])
        campaign_rows = serialize_rows(cur.fetchall())

        cur.execute(sku_query, [start_date, end_date, limit])
        promoted_sku_rows = serialize_rows(cur.fetchall())

    return {
        "date_from": start_date.isoformat(),
        "date_to": end_date.isoformat(),
        "spend_source": "marketing.wb_marketing_campaign_stats_daily.sum_spend",
        "sales_source": "sales.wb_supplier_sales.for_pay where is_realization = true",
        "interpretation_note": (
            "Это оценочный ДРР по всем кампаниям со spend > 0 за период. "
            "Выручка продвигаемых артикулов означает все фактические продажи SKU, "
            "которые были привязаны к рекламным кампаниям, а не строго атрибутированные рекламные продажи."
        ),
        "totals": totals,
        "campaign_rows": campaign_rows,
        "promoted_sku_rows": promoted_sku_rows,
    }


def get_marketing_sku_period_comparison_report(
    date_from: str | None = None,
    date_to: str | None = None,
    sku_list: str | None = None,
    product_keyword: str | None = None,
    period_days: int = 7,
    limit: int = 100,
) -> dict[str, Any]:
    """Compare campaign metrics by promoted SKU for current vs previous period.

    WB campaign stats are stored at campaign-day grain. When one campaign contains
    multiple SKU, campaign metrics are allocated equally across linked SKU and
    marked as estimated rather than exact SKU attribution.
    """
    limit = _validate_limit(limit, upper=200)
    sku_tokens = _parse_sku_tokens(sku_list)
    start_date = None
    end_date = None
    if date_from:
        start_date, end_date = _parse_report_period(date_from=date_from, date_to=date_to)
        if start_date is None or end_date is None:
            raise ValueError("date_from must not be empty")
        if end_date < start_date:
            raise ValueError("date_to must be greater than or equal to date_from")
        period_days = (end_date - start_date).days + 1
    else:
        period_days = _validate_period_days(period_days, upper=31)

    keyword_clause, keyword_params = _keyword_clause(
        product_keyword,
        ["p.vendor_code", "p.subject_name", "p.nm_id::text"],
    )

    query = f"""
        with raw_params as (
            select
                %s::date as requested_date_from,
                %s::date as requested_date_to,
                %s::int as requested_period_days
        ),
        latest_stats_day as (
            select {_latest_full_day_sql('stat_date')} as stat_date
            from marketing.wb_marketing_campaign_stats_daily
        ),
        periods as (
            select
                coalesce(r.requested_date_to, l.stat_date) as current_to,
                coalesce(
                    r.requested_date_from,
                    coalesce(r.requested_date_to, l.stat_date)
                        - ((r.requested_period_days - 1) * interval '1 day')
                )::date as current_from,
                (
                    coalesce(
                        r.requested_date_from,
                        coalesce(r.requested_date_to, l.stat_date)
                            - ((r.requested_period_days - 1) * interval '1 day')
                    )::date - interval '1 day'
                )::date as previous_to,
                (
                    coalesce(
                        r.requested_date_from,
                        coalesce(r.requested_date_to, l.stat_date)
                            - ((r.requested_period_days - 1) * interval '1 day')
                    )::date - (r.requested_period_days * interval '1 day')
                )::date as previous_from,
                r.requested_period_days as period_days
            from raw_params r
            cross join latest_stats_day l
        ),
        latest_campaign_nms as (
            select
                advert_id,
                nm_id,
                max(subject_name) as subject_name
            from (
                select distinct on (advert_id, nm_id)
                    advert_id,
                    nm_id,
                    subject_name,
                    snapshot_date,
                    last_synced_at,
                    id
                from marketing.wb_marketing_campaign_nms
                order by advert_id, nm_id, snapshot_date desc, last_synced_at desc, id desc
            ) n
            group by advert_id, nm_id
        ),
        campaign_nm_counts as (
            select advert_id, count(distinct nm_id)::numeric as nm_count
            from latest_campaign_nms
            group by advert_id
        ),
        campaign_stats as (
            select
                case
                    when s.stat_date between p.current_from and p.current_to then 'current'
                    when s.stat_date between p.previous_from and p.previous_to then 'previous'
                end as period_label,
                s.advert_id,
                sum(coalesce(s.views, 0))::numeric as impressions,
                sum(coalesce(s.clicks, 0))::numeric as clicks,
                sum(coalesce(s.orders, 0))::numeric as ad_orders,
                round(sum(coalesce(s.sum_spend, 0))::numeric, 2) as spend_rub
            from marketing.wb_marketing_campaign_stats_daily s
            cross join periods p
            where s.stat_date between p.previous_from and p.current_to
            group by period_label, s.advert_id
        ),
        promoted_nms as (
            select distinct nm_id
            from latest_campaign_nms
        ),
        sales_codes as (
            select
                s.nm_id,
                max(nullif(s.supplier_article, '')) as vendor_code
            from sales.wb_supplier_sales s
            join promoted_nms p using (nm_id)
            group by s.nm_id
        ),
        funnel_codes as (
            select
                f.nm_id,
                max(nullif(f.vendor_code, '')) as vendor_code
            from sales.wb_sales_funnel_products_daily f
            join promoted_nms p using (nm_id)
            group by f.nm_id
        ),
        sku_codes as (
            select
                p.nm_id,
                coalesce(sc.vendor_code, fc.vendor_code) as vendor_code
            from promoted_nms p
            left join sales_codes sc using (nm_id)
            left join funnel_codes fc using (nm_id)
        ),
        sales_by_sku as (
            select
                case
                    when s.date_ord::date between p.current_from and p.current_to then 'current'
                    when s.date_ord::date between p.previous_from and p.previous_to then 'previous'
                end as period_label,
                s.nm_id,
                round(sum(coalesce(s.for_pay, 0))::numeric, 2) as sales_revenue_rub,
                count(*)::int as sales_units
            from sales.wb_supplier_sales s
            cross join periods p
            where s.date_ord::date between p.previous_from and p.current_to
              and s.is_realization = true
            group by period_label, s.nm_id
        ),
        sku_period_metrics as (
            select
                n.nm_id,
                max(n.subject_name) as subject_name,
                sc.vendor_code,
                cs.period_label,
                count(distinct cs.advert_id)::int as campaign_count,
                sum(case when c.nm_count > 1 then 1 else 0 end)::int as multi_sku_campaign_links,
                round(sum(cs.impressions / nullif(c.nm_count, 0))::numeric, 2) as impressions,
                round(sum(cs.clicks / nullif(c.nm_count, 0))::numeric, 2) as clicks,
                round(sum(cs.ad_orders / nullif(c.nm_count, 0))::numeric, 2) as ad_orders,
                round(sum(cs.spend_rub / nullif(c.nm_count, 0))::numeric, 2) as spend_rub
            from campaign_stats cs
            join latest_campaign_nms n
              on n.advert_id = cs.advert_id
            join campaign_nm_counts c
              on c.advert_id = cs.advert_id
            left join sku_codes sc
              on sc.nm_id = n.nm_id
            group by n.nm_id, sc.vendor_code, cs.period_label
        ),
        pivoted as (
            select
                m.nm_id,
                max(m.vendor_code) as vendor_code,
                max(m.subject_name) as subject_name,
                max(m.campaign_count) filter (where m.period_label = 'current') as current_campaigns,
                max(m.campaign_count) filter (where m.period_label = 'previous') as previous_campaigns,
                sum(m.multi_sku_campaign_links) filter (where m.period_label = 'current') as current_multi_sku_links,
                sum(m.multi_sku_campaign_links) filter (where m.period_label = 'previous') as previous_multi_sku_links,
                coalesce(sum(m.impressions) filter (where m.period_label = 'current'), 0) as current_impressions,
                coalesce(sum(m.impressions) filter (where m.period_label = 'previous'), 0) as previous_impressions,
                coalesce(sum(m.clicks) filter (where m.period_label = 'current'), 0) as current_clicks,
                coalesce(sum(m.clicks) filter (where m.period_label = 'previous'), 0) as previous_clicks,
                coalesce(sum(m.ad_orders) filter (where m.period_label = 'current'), 0) as current_ad_orders,
                coalesce(sum(m.ad_orders) filter (where m.period_label = 'previous'), 0) as previous_ad_orders,
                coalesce(sum(m.spend_rub) filter (where m.period_label = 'current'), 0) as current_spend_rub,
                coalesce(sum(m.spend_rub) filter (where m.period_label = 'previous'), 0) as previous_spend_rub
            from sku_period_metrics m
            group by m.nm_id
        ),
        sales_pivot as (
            select
                nm_id,
                coalesce(sum(sales_revenue_rub) filter (where period_label = 'current'), 0) as current_sales_revenue_rub,
                coalesce(sum(sales_revenue_rub) filter (where period_label = 'previous'), 0) as previous_sales_revenue_rub,
                coalesce(sum(sales_units) filter (where period_label = 'current'), 0)::int as current_sales_units,
                coalesce(sum(sales_units) filter (where period_label = 'previous'), 0)::int as previous_sales_units
            from sales_by_sku
            group by nm_id
        )
        select
            (select current_from from periods) as current_date_from,
            (select current_to from periods) as current_date_to,
            (select previous_from from periods) as previous_date_from,
            (select previous_to from periods) as previous_date_to,
            p.nm_id,
            p.vendor_code,
            p.subject_name,
            coalesce(p.current_campaigns, 0) as current_campaigns,
            coalesce(p.previous_campaigns, 0) as previous_campaigns,
            case
                when coalesce(p.current_multi_sku_links, 0) > 0
                  or coalesce(p.previous_multi_sku_links, 0) > 0
                    then 'estimated_allocated_from_campaign'
                else 'campaign_single_sku'
            end as attribution_quality,
            round(p.current_impressions, 2) as current_impressions,
            round(p.previous_impressions, 2) as previous_impressions,
            round(p.current_impressions - p.previous_impressions, 2) as impressions_delta,
            round(case when p.previous_impressions > 0
                then (p.current_impressions - p.previous_impressions) * 100 / p.previous_impressions
            end, 2) as impressions_delta_pct,
            round(p.current_clicks, 2) as current_clicks,
            round(p.previous_clicks, 2) as previous_clicks,
            round(p.current_clicks - p.previous_clicks, 2) as clicks_delta,
            round(case when p.previous_clicks > 0
                then (p.current_clicks - p.previous_clicks) * 100 / p.previous_clicks
            end, 2) as clicks_delta_pct,
            round(case when p.current_impressions > 0 then p.current_clicks * 100 / p.current_impressions end, 2) as current_ctr_pct,
            round(case when p.previous_impressions > 0 then p.previous_clicks * 100 / p.previous_impressions end, 2) as previous_ctr_pct,
            round(case when p.current_clicks > 0 then p.current_spend_rub / p.current_clicks end, 2) as current_cpc_rub,
            round(case when p.previous_clicks > 0 then p.previous_spend_rub / p.previous_clicks end, 2) as previous_cpc_rub,
            round(p.current_ad_orders, 2) as current_ad_orders,
            round(p.previous_ad_orders, 2) as previous_ad_orders,
            round(p.current_ad_orders - p.previous_ad_orders, 2) as ad_orders_delta,
            round(case when p.previous_ad_orders > 0
                then (p.current_ad_orders - p.previous_ad_orders) * 100 / p.previous_ad_orders
            end, 2) as ad_orders_delta_pct,
            round(p.current_spend_rub, 2) as current_spend_rub,
            round(p.previous_spend_rub, 2) as previous_spend_rub,
            round(p.current_spend_rub - p.previous_spend_rub, 2) as spend_delta_rub,
            round(case when p.previous_spend_rub > 0
                then (p.current_spend_rub - p.previous_spend_rub) * 100 / p.previous_spend_rub
            end, 2) as spend_delta_pct,
            coalesce(sp.current_sales_revenue_rub, 0) as current_sales_revenue_rub,
            coalesce(sp.previous_sales_revenue_rub, 0) as previous_sales_revenue_rub,
            coalesce(sp.current_sales_units, 0) as current_sales_units,
            coalesce(sp.previous_sales_units, 0) as previous_sales_units,
            round(case when coalesce(sp.current_sales_revenue_rub, 0) > 0
                then p.current_spend_rub * 100 / sp.current_sales_revenue_rub
            end, 2) as current_estimated_drr_pct,
            round(case when coalesce(sp.previous_sales_revenue_rub, 0) > 0
                then p.previous_spend_rub * 100 / sp.previous_sales_revenue_rub
            end, 2) as previous_estimated_drr_pct
        from pivoted p
        left join sales_pivot sp
          on sp.nm_id = p.nm_id
        where (
            cardinality(%s::text[]) = 0
            or p.nm_id::text = any(%s::text[])
            or coalesce(p.vendor_code, '') = any(%s::text[])
        )
        {keyword_clause}
        order by p.current_spend_rub desc, abs(p.current_ad_orders - p.previous_ad_orders) desc, p.nm_id
        limit %s
    """

    params: list[Any] = [
        start_date,
        end_date,
        period_days,
        sku_tokens,
        sku_tokens,
        sku_tokens,
        *keyword_params,
        limit,
    ]
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(query, params)
        rows = serialize_rows(cur.fetchall())

    periods = rows[0] if rows else {}
    return {
        "date_from": periods.get("current_date_from"),
        "date_to": periods.get("current_date_to"),
        "previous_date_from": periods.get("previous_date_from"),
        "previous_date_to": periods.get("previous_date_to"),
        "period_days": period_days,
        "focus_sku_list": sku_tokens,
        "focus_keyword": product_keyword,
        "metric_grain": "SKU period comparison from campaign-day stats",
        "allocation_note": (
            "WB campaign stats are campaign-level. For campaigns linked to multiple SKU, "
            "impressions, clicks, spend and ad orders are equally allocated across linked SKU "
            "and marked as estimated_allocated_from_campaign."
        ),
        "source_tables": [
            "marketing.wb_marketing_campaign_stats_daily",
            "marketing.wb_marketing_campaign_nms",
            "sales.wb_supplier_sales",
            "sales.wb_sales_funnel_products_daily",
        ],
        "row_count": len(rows),
        "rows": rows,
    }


def get_marketing_campaign_efficiency_snapshot(
    product_keyword: str | None = None,
    days: int = 7,
    limit: int = 20,
) -> dict[str, Any]:
    days = _validate_days(days)
    limit = _validate_limit(limit, upper=50)
    identity_clause, identity_params = _keyword_clause(
        product_keyword,
        ["si.nm_id::text", "si.vendor_code", "si.subject_name", "si.brand_name"],
    )
    keyword_like = f"%{product_keyword}%" if product_keyword else None
    filter_params = [product_keyword, keyword_like, keyword_like, keyword_like, keyword_like]
    filter_ctes = f"""
        campaign_filter_latest_nms as (
            select
                advert_id,
                nm_id,
                max(nullif(subject_name, '')) as campaign_subject_name
            from (
                select distinct on (advert_id, nm_id)
                    advert_id,
                    nm_id,
                    subject_name,
                    snapshot_date,
                    last_synced_at,
                    id
                from marketing.wb_marketing_campaign_nms
                order by advert_id, nm_id, snapshot_date desc, last_synced_at desc, id desc
            ) n
            group by advert_id, nm_id
        ),
        campaign_filter_promoted_nms as (
            select distinct nm_id
            from campaign_filter_latest_nms
        ),
        campaign_filter_sales_codes as (
            select
                s.nm_id,
                max(nullif(s.supplier_article, '')) as vendor_code,
                max(nullif(s.subject, '')) as subject_name,
                max(nullif(s.brand, '')) as brand_name
            from sales.wb_supplier_sales s
            join campaign_filter_promoted_nms p using (nm_id)
            group by s.nm_id
        ),
        campaign_filter_funnel_codes as (
            select
                f.nm_id,
                max(nullif(f.vendor_code, '')) as vendor_code,
                max(nullif(f.subject_name, '')) as subject_name,
                max(nullif(f.brand_name, '')) as brand_name
            from sales.wb_sales_funnel_products_daily f
            join campaign_filter_promoted_nms p using (nm_id)
            group by f.nm_id
        ),
        campaign_filter_sku_identity as (
            select
                p.nm_id,
                coalesce(sc.vendor_code, fc.vendor_code) as vendor_code,
                coalesce(sc.subject_name, fc.subject_name, n.campaign_subject_name) as subject_name,
                coalesce(sc.brand_name, fc.brand_name) as brand_name
            from campaign_filter_promoted_nms p
            left join campaign_filter_sales_codes sc using (nm_id)
            left join campaign_filter_funnel_codes fc using (nm_id)
            left join (
                select nm_id, max(campaign_subject_name) as campaign_subject_name
                from campaign_filter_latest_nms
                group by nm_id
            ) n using (nm_id)
        ),
        campaign_filter_eligible_campaigns as (
            select distinct n.advert_id
            from campaign_filter_latest_nms n
            join campaign_filter_sku_identity si using (nm_id)
            where true
            {identity_clause}
        )
    """

    def filter_sql(alias: str) -> str:
        return f"""
            and (
                %s::text is null
                or exists (
                    select 1
                    from campaign_filter_eligible_campaigns ec
                    where ec.advert_id = {alias}.advert_id
                )
                or coalesce({alias}.campaign_name, '') ilike %s
                or coalesce({alias}.top_subject_name, '') ilike %s
                or coalesce({alias}.advert_id::text, '') ilike %s
                or coalesce({alias}.brand, '') ilike %s
            )
        """

    query = f"""
        with {MARKETING_CAMPAIGN_DAILY_CTE},
        {filter_ctes},
        latest_stats_day as (
            select {_latest_full_day_sql('stat_date')} as stat_date
            from agent_campaign_daily
        ),
        stats_period as (
            select
                advert_id,
                max(campaign_name) as campaign_name,
                max(top_subject_name) as top_subject_name,
                max(brand) as brand,
                sum(coalesce(views, 0)) as views,
                sum(coalesce(clicks, 0)) as clicks,
                sum(coalesce(orders, 0)) as orders,
                sum(coalesce(sum_spend, 0)) as sum_spend,
                avg(nullif(ctr, 0)) as ctr,
                avg(nullif(cpc, 0)) as cpc,
                avg(nullif(cr, 0)) as cr
            from agent_campaign_daily
            where stat_date between
                (select stat_date - interval '{days - 1} day' from latest_stats_day)
                and (select stat_date from latest_stats_day)
            group by advert_id
        ),
        labeled as (
            select
                (select stat_date from latest_stats_day) as report_date,
                s.advert_id,
                coalesce(s.campaign_name, 'campaign_' || s.advert_id::text) as campaign_name,
                s.top_subject_name,
                s.brand,
                s.views,
                s.clicks,
                s.orders,
                s.sum_spend,
                s.ctr,
                s.cpc,
                s.cr,
                case
                    when coalesce(s.sum_spend, 0) >= 1800 and coalesce(s.orders, 0) <= 1 and coalesce(s.cpc, 0) >= 50
                        then 'negative_outlier'
                    when coalesce(s.sum_spend, 0) >= 1500 and coalesce(s.orders, 0) = 0
                        then 'inefficient'
                    when coalesce(s.orders, 0) >= 5 and coalesce(s.cpc, 0) <= 20 and coalesce(s.cr, 0) >= 5
                        then 'strong_leader'
                    when coalesce(s.orders, 0) >= 1 and coalesce(s.cpc, 0) <= 30
                        then 'working'
                    when coalesce(s.orders, 0) >= 1
                        then 'conditional'
                    else 'investigate'
                end as status
            from stats_period s
        )
        select l.*
        from labeled l
        where 1 = 1
        {filter_sql("l")}
        order by coalesce(sum_spend, 0) desc, coalesce(orders, 0) desc, advert_id
        limit %s
    """
    totals_query = f"""
        with {MARKETING_CAMPAIGN_DAILY_CTE},
        {filter_ctes},
        latest_stats_day as (
            select {_latest_full_day_sql('stat_date')} as stat_date
            from agent_campaign_daily
        )
        select
            (select stat_date from latest_stats_day) as report_date,
            count(distinct advert_id) as campaign_count,
            sum(coalesce(views, 0)) as total_views,
            sum(coalesce(clicks, 0)) as total_clicks,
            sum(coalesce(orders, 0)) as total_orders,
            sum(coalesce(sum_spend, 0)) as total_spend
        from agent_campaign_daily s
        where s.stat_date between
            (select stat_date - interval '{days - 1} day' from latest_stats_day)
            and (select stat_date from latest_stats_day)
        {filter_sql("s")}
    """
    concentration_query = f"""
        with {MARKETING_CAMPAIGN_DAILY_CTE},
        {filter_ctes},
        latest_stats_day as (
            select {_latest_full_day_sql('stat_date')} as stat_date
            from agent_campaign_daily
        ),
        per_campaign as (
            select s.advert_id, sum(coalesce(s.sum_spend, 0)) as sum_spend
            from agent_campaign_daily s
            where s.stat_date between
                (select stat_date - interval '{days - 1} day' from latest_stats_day)
                and (select stat_date from latest_stats_day)
            {filter_sql("s")}
            group by s.advert_id
        )
        select
            coalesce(max(sum_spend), 0) as top_campaign_spend,
            coalesce(sum(sum_spend), 0) as total_spend,
            case
                when coalesce(sum(sum_spend), 0) > 0
                    then round(100 * max(sum_spend) / sum(sum_spend), 2)
            end as top_campaign_spend_share_pct
        from per_campaign
    """

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(query, [*identity_params, *filter_params, limit])
        campaign_rows = serialize_rows(cur.fetchall())

        cur.execute(totals_query, [*identity_params, *filter_params])
        totals = serialize_value(cur.fetchone())

        cur.execute(concentration_query, [*identity_params, *filter_params])
        budget_concentration = serialize_value(cur.fetchone())

    waste_rows = [row for row in campaign_rows if row["status"] in {"inefficient", "negative_outlier"}]
    scale_rows = [row for row in campaign_rows if row["status"] == "strong_leader"]

    return {
        "report_date": totals.get("report_date"),
        "period_days": days,
        "focus_keyword": product_keyword,
        "totals": totals,
        "campaign_rows": campaign_rows,
        "waste_rows": waste_rows[:limit],
        "scale_rows": scale_rows[:limit],
        "budget_concentration": budget_concentration,
    }


def get_marketing_query_performance_snapshot(
    product_keyword: str | None = None,
    days: int = 7,
    limit: int = 30,
) -> dict[str, Any]:
    days = _validate_days(days)
    limit = _validate_limit(limit, upper=100)
    keyword_clause, keyword_params = _keyword_clause(
        product_keyword,
        ["norm_query", "vendor_code", "subject_name", "brand_name", "nm_id::text"],
    )

    query = f"""
        with {MARKETING_SEARCH_TEXT_DAILY_CTE},
        latest_day as (
            select {_latest_full_day_sql('snapshot_date')} as snapshot_date
            from agent_search_text_daily
        ),
        per_query as (
            select
                (select snapshot_date from latest_day) as report_date,
                nm_id,
                query_text as norm_query,
                max(vendor_code) as vendor_code,
                max(subject_name) as subject_name,
                max(brand_name) as brand_name,
                sum(coalesce(frequency, 0)) as views,
                sum(coalesce(frequency, 0)) as clicks,
                sum(coalesce(orders_count, 0)) as orders,
                0::numeric as spend,
                null::numeric as ctr,
                null::numeric as cpc,
                avg(nullif(avg_position_text, 0)) as avg_pos
            from agent_search_text_daily
            where snapshot_date between
                (select snapshot_date - interval '{days - 1} day' from latest_day)
                and (select snapshot_date from latest_day)
            group by nm_id, query_text
        )
        select *
        from per_query
        where 1 = 1
        {keyword_clause}
        order by coalesce(orders, 0) desc, coalesce(spend, 0) desc, norm_query
        limit %s
    """

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(query, [*keyword_params, limit])
        rows = serialize_rows(cur.fetchall())

    return {
        "report_date": rows[0]["report_date"] if rows else None,
        "period_days": days,
        "focus_keyword": product_keyword,
        "top_query_rows": rows,
        "query_waste_rows": [
            row for row in rows if float(row.get("spend") or 0) > 0 and int(row.get("orders") or 0) == 0
        ][:limit],
        "query_scale_rows": [
            row
            for row in rows
            if int(row.get("orders") or 0) > 0
            and float(row.get("cpc") or 0) <= 25
            and float(row.get("avg_pos") or 999999) <= 20
        ][:limit],
        "query_position_risk_rows": [
            row
            for row in rows
            if int(row.get("clicks") or 0) > 0 and float(row.get("avg_pos") or 0) >= 25
        ][:limit],
    }


def get_marketing_bid_pressure_snapshot(
    product_keyword: str | None = None,
    days: int = 7,
    limit: int = 30,
) -> dict[str, Any]:
    days = _validate_days(days)
    limit = _validate_limit(limit, upper=100)
    report_date_query = f"""
        with {MARKETING_CAMPAIGN_DAILY_CTE}
        select {_latest_full_day_sql("stat_date")} as snapshot_date
        from agent_campaign_daily
    """

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(report_date_query)
        row = serialize_value(cur.fetchone()) or {}

    return {
        "report_date": row.get("snapshot_date"),
        "period_days": days,
        "focus_keyword": product_keyword,
        "bid_rows": [],
        "overbid_rows": [],
        "underbid_rows": [],
        "bid_efficiency_rows": [],
    }


def get_marketing_visibility_snapshot(
    product_keyword: str | None = None,
    limit: int = 30,
) -> dict[str, Any]:
    limit = _validate_limit(limit, upper=100)
    keyword_clause, keyword_params = _keyword_clause(
        product_keyword,
        ["name", "vendor_code", "subject_name", "brand_name", "nm_id::text"],
    )

    details_query = f"""
        with {MARKETING_SEARCH_SKU_DAILY_CTE},
        latest_snapshot as (
            select max(snapshot_date) as snapshot_date
            from agent_search_sku_daily
        )
        select
            snapshot_date,
            request_key,
            nm_id,
            name,
            vendor_code,
            subject_name,
            brand_name,
            avg_position_current,
            avg_position_dynamics,
            visibility_current,
            visibility_dynamics,
            open_card_current,
            open_card_dynamics,
            add_to_cart_current,
            add_to_cart_dynamics,
            orders_current,
            orders_dynamics,
            cart_to_order_current,
            cart_to_order_dynamics
        from agent_search_sku_daily
        where snapshot_date = (select snapshot_date from latest_snapshot)
        {keyword_clause}
        order by coalesce(orders_current, 0) desc, coalesce(visibility_current, 0) desc, nm_id
        limit %s
    """
    overview_query = f"""
        with {MARKETING_SEARCH_SKU_DAILY_CTE},
        latest_snapshot as (
            select max(snapshot_date) as snapshot_date
            from agent_search_sku_daily
        )
        select
            snapshot_date,
            max(total_products) as total_products,
            max(supplier_rating_current) as supplier_rating_current,
            max(position_average_current) as position_average_current,
            max(position_average_dynamics) as position_average_dynamics,
            max(overview_visibility_current) as visibility_current,
            max(overview_visibility_dynamics) as visibility_dynamics,
            max(overview_open_card_current) as open_card_current,
            max(overview_open_card_dynamics) as open_card_dynamics
        from agent_search_sku_daily
        where snapshot_date = (select snapshot_date from latest_snapshot)
        group by snapshot_date
    """

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(details_query, [*keyword_params, limit])
        visibility_rows = serialize_rows(cur.fetchall())

        cur.execute(overview_query)
        overview = serialize_value(cur.fetchone())

    return {
        "report_date": visibility_rows[0]["snapshot_date"] if visibility_rows else overview.get("snapshot_date"),
        "focus_keyword": product_keyword,
        "overview": overview,
        "visibility_rows": visibility_rows,
        "position_loss_rows": [
            row
            for row in visibility_rows
            if float(row.get("avg_position_current") or 0) >= 25
            or float(row.get("avg_position_dynamics") or 0) <= -5
        ][:limit],
        "open_card_gap_rows": [
            row
            for row in visibility_rows
            if float(row.get("visibility_current") or 0) > 0
            and float(row.get("open_card_dynamics") or 0) < 0
        ][:limit],
        "post_click_conversion_rows": [
            row
            for row in visibility_rows
            if float(row.get("open_card_current") or 0) > 0
            and float(row.get("orders_current") or 0) <= 0
        ][:limit],
    }


def get_marketing_search_demand_snapshot(
    product_keyword: str | None = None,
    days: int = 7,
    limit: int = 30,
) -> dict[str, Any]:
    days = _validate_days(days)
    limit = _validate_limit(limit, upper=100)
    keyword_clause, keyword_params = _keyword_clause(
        product_keyword,
        ["c.query_text", "c.vendor_code", "c.subject_name", "c.brand_name", "c.nm_id::text"],
    )

    query = f"""
        with {MARKETING_SEARCH_TEXT_DAILY_CTE},
        latest_day as (
            select {_latest_full_day_sql('snapshot_date')} as snapshot_date
            from agent_search_text_daily
        ),
        current_period as (
            select
                query_text,
                nm_id,
                max(vendor_code) as vendor_code,
                max(subject_name) as subject_name,
                max(brand_name) as brand_name,
                sum(coalesce(frequency, 0)) as frequency_sum,
                avg(nullif(avg_position_text, 0)) as avg_position,
                sum(coalesce(orders_count, 0)) as orders_count
            from agent_search_text_daily
            where snapshot_date between
                (select snapshot_date - interval '{days - 1} day' from latest_day)
                and (select snapshot_date from latest_day)
            group by query_text, nm_id
        ),
        prior_period as (
            select
                query_text,
                nm_id,
                sum(coalesce(orders_count, 0)) as prior_orders_count
            from agent_search_text_daily
            where snapshot_date between
                (select snapshot_date - interval '{2 * days - 1} day' from latest_day)
                and (select snapshot_date - interval '{days} day' from latest_day)
            group by query_text, nm_id
        )
        select
            (select snapshot_date from latest_day) as report_date,
            c.query_text,
            c.nm_id,
            c.vendor_code,
            c.subject_name,
            c.brand_name,
            c.frequency_sum,
            c.avg_position,
            c.orders_count,
            p.prior_orders_count,
            coalesce(c.orders_count, 0) - coalesce(p.prior_orders_count, 0) as orders_delta
        from current_period c
        left join prior_period p
          on p.query_text = c.query_text
         and p.nm_id = c.nm_id
        where 1 = 1
        {keyword_clause}
        order by coalesce(c.orders_count, 0) desc, coalesce(c.frequency_sum, 0) desc, c.query_text
        limit %s
    """

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(query, [*keyword_params, limit])
        search_text_rows = serialize_rows(cur.fetchall())

    return {
        "report_date": search_text_rows[0]["report_date"] if search_text_rows else None,
        "period_days": days,
        "focus_keyword": product_keyword,
        "search_text_rows": search_text_rows,
        "growing_query_rows": [
            row for row in search_text_rows if float(row.get("orders_delta") or 0) > 0
        ][:limit],
        "high_frequency_low_order_rows": [
            row
            for row in search_text_rows
            if float(row.get("frequency_sum") or 0) >= 50 and float(row.get("orders_count") or 0) <= 1
        ][:limit],
        "order_driving_query_rows": [
            row for row in search_text_rows if float(row.get("orders_count") or 0) >= 2
        ][:limit],
    }


def _marketing_matview_source_status(schema: str, name: str) -> str:
    query = """
        select
            case
                when c.oid is null then 'missing'
                when c.relkind = 'm' and not c.relispopulated then 'unpopulated'
                else 'ready'
            end as source_status
        from pg_namespace n
        left join pg_class c
          on c.relnamespace = n.oid
         and c.relname = %s
        where n.nspname = %s
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(query, [name, schema])
        row = serialize_value(cur.fetchone()) or {}
    return str(row.get("source_status") or "missing")


def _merge_source_status(*statuses: str) -> str:
    normalized = [str(status or "").strip() for status in statuses if str(status or "").strip()]
    if not normalized:
        return "missing"
    unique = set(normalized)
    if unique == {"ready"}:
        return "ready"
    if "ready" in unique:
        return "partial"
    if "unpopulated" in unique:
        return "unpopulated"
    return normalized[0]


def _coerce_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def _marketing_latest_stats_date() -> date | None:
    query = f"""
        select {_latest_full_day_sql('stat_date')} as report_date
        from marketing.wb_marketing_campaign_stats_daily
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(query)
        row = serialize_value(cur.fetchone()) or {}
    return _coerce_date(row.get("report_date"))


def _source_freshness(
    *,
    report_date: Any,
    reference_date: date | None,
    max_lag_days: int = 2,
) -> dict[str, Any]:
    source_date = _coerce_date(report_date)
    lag_days = None
    status = "unknown"
    if source_date and reference_date:
        lag_days = (reference_date - source_date).days
        if lag_days < 0:
            status = "future"
        elif lag_days <= max_lag_days:
            status = "fresh"
        else:
            status = "stale"
    elif source_date:
        status = "unknown_reference"
    return {
        "report_date": source_date.isoformat() if source_date else None,
        "reference_date": reference_date.isoformat() if reference_date else None,
        "lag_days": lag_days,
        "max_lag_days": max_lag_days,
        "status": status,
    }


def _marketing_matview_rows(
    *,
    table_name: str,
    product_keyword: str | None = None,
    limit: int = 30,
) -> dict[str, Any]:
    limit = _validate_limit(limit, upper=100)
    source_status = _marketing_matview_source_status("marketing", table_name)
    if source_status != "ready":
        return {
            "source_status": source_status,
            "focus_keyword": product_keyword,
            "row_count": 0,
            "rows": [],
        }

    query = f"""
        select *
        from marketing.{table_name} t
        where %s::text is null
           or to_jsonb(t)::text ilike %s::text
        order by 1 desc
        limit %s
    """
    like = f"%{product_keyword}%" if product_keyword else None
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(query, [product_keyword, like, limit])
        rows = serialize_rows(cur.fetchall())

    report_date = None
    if rows:
        for key in ("snapshot_date", "stat_date", "report_date", "date"):
            if key in rows[0]:
                report_date = rows[0].get(key)
                break

    return {
        "source_status": source_status,
        "focus_keyword": product_keyword,
        "report_date": report_date,
        "row_count": len(rows),
        "rows": rows,
    }


def get_marketing_campaign_daily_mart(
    product_keyword: str | None = None,
    days: int = 7,
    limit: int = 20,
) -> dict[str, Any]:
    days = _validate_days(days)
    limit = _validate_limit(limit, upper=100)
    source_status = _marketing_matview_source_status("marketing", "mart_marketing_campaign_daily")
    if source_status != "ready":
        return {
            "source_status": source_status,
            "period_days": days,
            "focus_keyword": product_keyword,
            "report_date": None,
            "row_count": 0,
            "rows": [],
        }

    keyword_clause, keyword_params = _keyword_clause(
        product_keyword,
        ["campaign_name", "advert_id::text", "payment_type"],
    )
    query = f"""
        with latest_day as (
            select {_latest_full_day_sql('stat_date')} as stat_date
            from marketing.mart_marketing_campaign_daily
        ),
        period_rows as (
            select
                (select stat_date from latest_day) as report_date,
                advert_id,
                max(campaign_name) as campaign_name,
                max(payment_type) as payment_type,
                max(campaign_status) as campaign_status,
                sum(coalesce(views, 0)) as views,
                sum(coalesce(clicks, 0)) as clicks,
                sum(coalesce(orders, 0)) as orders,
                sum(coalesce(spend, 0)) as spend,
                avg(nullif(ctr, 0)) as ctr,
                avg(nullif(cpc, 0)) as cpc,
                avg(nullif(cr, 0)) as cr,
                avg(nullif(cpa, 0)) as cpa,
                sum(coalesce(query_views, 0)) as query_views,
                sum(coalesce(query_clicks, 0)) as query_clicks,
                sum(coalesce(query_orders, 0)) as query_orders,
                sum(coalesce(query_spend, 0)) as query_spend,
                avg(coalesce(winner_query_share, 0)) as winner_query_share,
                avg(coalesce(waste_query_share, 0)) as waste_query_share,
                max(coalesce(budget_total, 0)) as budget_total,
                bool_or(coalesce(is_scale_candidate, false)) as is_scale_candidate,
                bool_or(coalesce(is_cut_candidate, false)) as is_cut_candidate
            from marketing.mart_marketing_campaign_daily
            where stat_date between
                (select stat_date - interval '{days - 1} day' from latest_day)
                and (select stat_date from latest_day)
            group by advert_id
        )
        select *
        from period_rows
        where 1 = 1
        {keyword_clause}
        order by is_cut_candidate desc, coalesce(spend, 0) desc, coalesce(orders, 0) asc, advert_id
        limit %s
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(query, [*keyword_params, limit])
        rows = serialize_rows(cur.fetchall())

    return {
        "source_status": source_status,
        "period_days": days,
        "focus_keyword": product_keyword,
        "report_date": rows[0].get("report_date") if rows else None,
        "row_count": len(rows),
        "rows": rows,
    }


def get_marketing_query_bands_mart(
    product_keyword: str | None = None,
    limit: int = 30,
) -> dict[str, Any]:
    limit = _validate_limit(limit, upper=100)
    source_status = _marketing_matview_source_status("marketing", "mart_marketing_query_bands")
    if source_status != "ready":
        return {
            "source_status": source_status,
            "focus_keyword": product_keyword,
            "report_date": None,
            "row_count": 0,
            "rows": [],
        }

    keyword_clause, keyword_params = _keyword_clause(
        product_keyword,
        [
            "q.norm_query",
            "q.campaign_name",
            "q.advert_id::text",
            "q.nm_id::text",
            "coalesce(vm.vendor_code, '')",
            "coalesce(vm.brand_name, '')",
            "coalesce(vm.subject_name, '')",
        ],
    )
    query = f"""
        with latest_snapshot as (
            select max(snapshot_date) as snapshot_date
            from marketing.mart_marketing_query_bands
        ),
        vendor_map as (
            select
                nm_id,
                max(nullif(vendor_code, '')) as vendor_code,
                max(nullif(brand_name, '')) as brand_name,
                max(nullif(subject_name, '')) as subject_name
            from (
                select
                    nm_id,
                    supplier_article as vendor_code,
                    brand as brand_name,
                    subject as subject_name
                from sales.wb_supplier_sales
                union all
                select
                    nm_id,
                    vendor_code,
                    brand_name,
                    subject_name
                from sales.wb_sales_funnel_products_daily
            ) codes
            group by nm_id
        )
        select
            q.*,
            vm.vendor_code,
            vm.brand_name,
            vm.subject_name
        from marketing.mart_marketing_query_bands q
        left join vendor_map vm on vm.nm_id = q.nm_id
        where q.snapshot_date = (select snapshot_date from latest_snapshot)
        {keyword_clause}
        order by
            case q.query_band
                when 'waste' then 1
                when 'traffic_only' then 2
                when 'winner' then 3
                else 4
            end,
            coalesce(q.spend_7d, 0) desc,
            q.norm_query
        limit %s
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(query, [*keyword_params, limit])
        rows = serialize_rows(cur.fetchall())

    return {
        "source_status": source_status,
        "focus_keyword": product_keyword,
        "report_date": rows[0].get("snapshot_date") if rows else None,
        "row_count": len(rows),
        "rows": rows,
    }


def get_marketing_forecast_ready_mart(
    product_keyword: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    limit = _validate_limit(limit, upper=100)
    source_status = _marketing_matview_source_status("marketing", "mart_marketing_forecast_ready")
    if source_status != "ready":
        return {
            "source_status": source_status,
            "focus_keyword": product_keyword,
            "report_date": None,
            "row_count": 0,
            "rows": [],
        }

    keyword_clause, keyword_params = _keyword_clause(
        product_keyword,
        ["campaign_name", "advert_id::text", "payment_type"],
    )
    query = f"""
        with latest_snapshot as (
            select max(snapshot_date) as snapshot_date
            from marketing.mart_marketing_forecast_ready
        )
        select *
        from marketing.mart_marketing_forecast_ready
        where snapshot_date = (select snapshot_date from latest_snapshot)
        {keyword_clause}
        order by coalesce(extra_spend_15pct, 0) desc, advert_id
        limit %s
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(query, [*keyword_params, limit])
        rows = serialize_rows(cur.fetchall())

    return {
        "source_status": source_status,
        "focus_keyword": product_keyword,
        "report_date": rows[0].get("snapshot_date") if rows else None,
        "row_count": len(rows),
        "rows": rows,
    }


def get_marketing_query_demand_daily_mart(
    product_keyword: str | None = None,
    limit: int = 30,
) -> dict[str, Any]:
    return _marketing_matview_rows(
        table_name="mart_marketing_query_demand_daily",
        product_keyword=product_keyword,
        limit=limit,
    )


def get_marketing_query_share_daily_mart(
    product_keyword: str | None = None,
    limit: int = 30,
) -> dict[str, Any]:
    return _marketing_matview_rows(
        table_name="mart_marketing_query_share_daily",
        product_keyword=product_keyword,
        limit=limit,
    )


def get_marketing_query_forecast_ready_mart(
    product_keyword: str | None = None,
    limit: int = 30,
) -> dict[str, Any]:
    return _marketing_matview_rows(
        table_name="mart_marketing_query_forecast_ready",
        product_keyword=product_keyword,
        limit=limit,
    )


def get_marketing_campaign_demand_forecast_mart(
    product_keyword: str | None = None,
    limit: int = 30,
) -> dict[str, Any]:
    return _marketing_matview_rows(
        table_name="mart_marketing_campaign_demand_forecast",
        product_keyword=product_keyword,
        limit=limit,
    )


def get_marketing_action_matrix(
    product_keyword: str | None = None,
    days: int = 7,
    limit: int = 20,
) -> dict[str, Any]:
    days = _validate_days(days)
    limit = _validate_limit(limit, upper=50)
    campaign_mart = get_marketing_campaign_daily_mart(
        product_keyword=product_keyword,
        days=days,
        limit=limit,
    )
    query_bands = get_marketing_query_bands_mart(
        product_keyword=product_keyword,
        limit=limit,
    )
    forecast_ready = get_marketing_forecast_ready_mart(
        product_keyword=product_keyword,
        limit=limit,
    )
    demand_daily = get_marketing_query_demand_daily_mart(product_keyword=product_keyword, limit=limit)
    demand_share = get_marketing_query_share_daily_mart(product_keyword=product_keyword, limit=limit)
    demand_forecast = get_marketing_query_forecast_ready_mart(product_keyword=product_keyword, limit=limit)
    campaign_demand = get_marketing_campaign_demand_forecast_mart(product_keyword=product_keyword, limit=limit)

    marketing_source_status = _merge_source_status(
        campaign_mart.get("source_status"),
        query_bands.get("source_status"),
        forecast_ready.get("source_status"),
    )
    demand_source_status = _merge_source_status(
        demand_daily.get("source_status"),
        demand_share.get("source_status"),
        demand_forecast.get("source_status"),
        campaign_demand.get("source_status"),
    )
    reference_date = _marketing_latest_stats_date()
    source_freshness = {
        "campaign_mart": _source_freshness(
            report_date=campaign_mart.get("report_date"),
            reference_date=reference_date,
        ),
        "query_bands_mart": _source_freshness(
            report_date=query_bands.get("report_date"),
            reference_date=reference_date,
        ),
        "forecast_ready_mart": _source_freshness(
            report_date=forecast_ready.get("report_date"),
            reference_date=reference_date,
        ),
        "query_demand_mart": _source_freshness(
            report_date=demand_daily.get("report_date"),
            reference_date=reference_date,
        ),
        "query_share_mart": _source_freshness(
            report_date=demand_share.get("report_date"),
            reference_date=reference_date,
        ),
        "query_forecast_mart": _source_freshness(
            report_date=demand_forecast.get("report_date"),
            reference_date=reference_date,
        ),
        "campaign_demand_mart": _source_freshness(
            report_date=campaign_demand.get("report_date"),
            reference_date=reference_date,
        ),
    }
    stale_sources = [
        name for name, freshness in source_freshness.items()
        if freshness.get("status") == "stale"
    ]
    invalid_sources = [
        name
        for name, freshness in source_freshness.items()
        if freshness.get("status") != "fresh"
    ]
    campaign_mart_current = source_freshness["campaign_mart"]["status"] == "fresh"
    query_bands_current = source_freshness["query_bands_mart"]["status"] == "fresh"
    forecast_ready_current = source_freshness["forecast_ready_mart"]["status"] == "fresh"
    demand_current = any(
        source_freshness[name]["status"] == "fresh"
        for name in (
            "query_demand_mart",
            "query_share_mart",
            "query_forecast_mart",
            "campaign_demand_mart",
        )
    )
    if demand_source_status == "ready" and not demand_current:
        demand_source_status = "invalid_freshness"

    visibility_fallback = get_marketing_visibility_snapshot(product_keyword=product_keyword, limit=limit)
    demand_fallback = None
    if demand_source_status != "ready":
        demand_fallback = get_marketing_search_demand_snapshot(
            product_keyword=product_keyword,
            days=days,
            limit=limit,
        )

    forecast_rows = {
        str(row.get("advert_id")): row
        for row in forecast_ready.get("rows") or []
        if forecast_ready_current and isinstance(row, dict) and row.get("advert_id") is not None
    }

    campaign_actions = []
    for row in (campaign_mart.get("rows") or []) if campaign_mart_current else []:
        if not isinstance(row, dict):
            continue
        advert_id = str(row.get("advert_id") or "").strip()
        forecast_row = forecast_rows.get(advert_id, {})
        decision_hint = str(forecast_row.get("decision_hint") or "").strip() or None
        confidence = str(forecast_row.get("confidence") or "").strip() or None
        safe_growth_pct = forecast_row.get("safe_growth_pct")
        marginal_quality_coef = forecast_row.get("marginal_quality_coef")
        winner_query_share = forecast_row.get("winner_query_share_7d", row.get("winner_query_share"))
        waste_query_share = forecast_row.get("waste_query_share_7d", row.get("waste_query_share"))
        orders = float(row.get("orders") or 0)
        spend = float(row.get("spend") or 0)
        is_cut_candidate = bool(row.get("is_cut_candidate"))
        is_scale_candidate = bool(row.get("is_scale_candidate"))

        action_mode = "hold"
        status = "оставить"
        suggested_action = "Оставить под контролем."
        reason = f"заказы={row.get('orders')}, затраты={row.get('spend')}, waste_share={waste_query_share}"
        blockers: list[str] = []

        if is_cut_candidate or ((float(waste_query_share or 0) >= 0.9) and orders <= 0 and spend > 0):
            action_mode = "reduce"
            status = "сократить"
            suggested_action = "Сократить или остановить кампанию по waste-сигналу."
        elif decision_hint == "hold":
            action_mode = "hold"
            status = "оставить"
        elif decision_hint == "hold_or_test" and confidence == "low":
            action_mode = "needs_review"
            status = "проверить"
            blockers.append("marketing_low_confidence")
            suggested_action = "Не масштабировать автоматически; сначала ручная проверка."
        elif decision_hint == "hold_or_test":
            action_mode = "test"
            status = "тест"
            blockers.append("marketing_test_only")
            suggested_action = "Разрешен только тестовый шаг без auto-scale."
        elif is_scale_candidate and confidence in {"medium", "high"} and demand_source_status == "ready":
            action_mode = "scale_review"
            status = "проверить масштаб"
            suggested_action = "Можно проверять осторожное расширение после demand-подтверждения."
        elif is_scale_candidate:
            action_mode = "needs_review"
            status = "проверить"
            blockers.append("marketing_demand_mart_unpopulated")
            suggested_action = "Scale-сигнал найден, но demand marts не готовы."

        campaign_actions.append(
            {
                "entity_type": "кампания",
                "entity_id": advert_id,
                "entity_name": row.get("campaign_name"),
                "status": status,
                "action_mode": action_mode,
                "reason": reason,
                "evidence": [
                    f"ctr={row.get('ctr')}",
                    f"cpc={row.get('cpc')}",
                    f"cr={row.get('cr')}",
                ],
                "suggested_action": suggested_action,
                "marketing_source_status": marketing_source_status,
                "demand_source_status": demand_source_status,
                "decision_hint": decision_hint,
                "confidence": confidence,
                "safe_growth_pct": safe_growth_pct,
                "marginal_quality_coef": marginal_quality_coef,
                "winner_query_share": winner_query_share,
                "waste_query_share": waste_query_share,
                "is_cut_candidate": is_cut_candidate,
                "is_scale_candidate": is_scale_candidate,
                "advert_id": row.get("advert_id"),
                "blockers": blockers,
            }
        )

    query_actions = []
    for row in (query_bands.get("rows") or []) if query_bands_current else []:
        if not isinstance(row, dict):
            continue
        band = str(row.get("query_band") or "").strip()
        action_mode = "hold"
        status = "оставить"
        blockers: list[str] = []
        suggested_action = "Оставить запрос под контролем."

        if band == "waste":
            action_mode = "reduce"
            status = "сократить"
            suggested_action = "Снизить ставку, минусовать или отключить waste-запрос."
        elif band == "traffic_only":
            action_mode = "test"
            status = "проверить"
            blockers.append("marketing_test_only")
            suggested_action = "Проверить посадку и конверсию, а не масштабировать."
        elif band == "winner":
            if demand_source_status == "ready":
                action_mode = "scale_review"
                status = "проверить масштаб"
                suggested_action = "Проверить расширение winner-запроса и новые ключи."
            else:
                action_mode = "needs_review"
                status = "проверить"
                blockers.append("marketing_demand_mart_unpopulated")
                suggested_action = "Winner-запрос найден, но demand forecast не готов."

        query_actions.append(
            {
                "entity_type": "поисковый запрос",
                "entity_id": row.get("norm_query"),
                "entity_name": row.get("norm_query"),
                "norm_query": row.get("norm_query"),
                "status": status,
                "action_mode": action_mode,
                "reason": (
                    f"band={band}, заказы={row.get('orders_7d')}, "
                    f"затраты={row.get('spend_7d')}, avg_pos={row.get('avg_pos_7d')}"
                ),
                "evidence": [
                    f"views_7d={row.get('views_7d')}",
                    f"clicks_7d={row.get('clicks_7d')}",
                    f"effective_cpc_7d={row.get('effective_cpc_7d')}",
                ],
                "suggested_action": suggested_action,
                "marketing_source_status": marketing_source_status,
                "demand_source_status": demand_source_status,
                "query_band": band,
                "advert_id": row.get("advert_id"),
                "nm_id": row.get("nm_id"),
                "vendor_code": row.get("vendor_code"),
                "winner_query_share": None,
                "waste_query_share": None,
                "blockers": blockers,
            }
        )

    sku_actions = []
    visibility_rows = visibility_fallback.get("position_loss_rows") or []
    for row in visibility_rows[: max(1, min(5, limit))]:
        if not isinstance(row, dict):
            continue
        sku_actions.append(
            {
                "entity_type": "артикул",
                "entity_id": row.get("vendor_code") or row.get("nm_id"),
                "entity_name": row.get("name"),
                "status": "проверить",
                "action_mode": "needs_review",
                "reason": (
                    f"текущая средняя позиция={row.get('avg_position_current')}, "
                    f"текущая видимость={row.get('visibility_current')}"
                ),
                "evidence": [f"текущие заказы={row.get('orders_current')}"],
                "suggested_action": "Проверить видимость, ставки и качество карточки вместе.",
                "marketing_source_status": marketing_source_status,
                "demand_source_status": demand_source_status,
                "query_band": None,
                "blockers": ["marketing_test_only"],
            }
        )

    legacy_campaign_report_date = None
    legacy_query_report_date = None
    legacy_campaign_freshness: dict[str, Any] | None = None
    legacy_query_freshness: dict[str, Any] | None = None
    if not campaign_actions:
        legacy_campaign = get_marketing_campaign_efficiency_snapshot(
            product_keyword=product_keyword,
            days=days,
            limit=limit,
        )
        legacy_campaign_report_date = legacy_campaign.get("report_date")
        legacy_campaign_freshness = _source_freshness(
            report_date=legacy_campaign_report_date,
            reference_date=reference_date,
        )
        for row in (
            legacy_campaign.get("campaign_rows") or []
            if legacy_campaign_freshness["status"] == "fresh"
            else []
        ):
            if not isinstance(row, dict):
                continue
            status = "сократить" if row.get("status") in {"inefficient", "negative_outlier"} else "оставить"
            suggested_action = (
                "Сократить или остановить кампанию по свежему campaign snapshot."
                if status == "сократить"
                else "Оставить под контролем по свежему campaign snapshot."
            )
            campaign_actions.append(
                {
                    "entity_type": "кампания",
                    "entity_id": row.get("advert_id"),
                    "entity_name": row.get("campaign_name"),
                    "status": status,
                    "action_mode": "reduce" if status == "сократить" else "hold",
                    "reason": f"legacy_snapshot: заказы={row.get('orders')}, затраты={row.get('sum_spend')}",
                    "evidence": [f"cpc={row.get('cpc')}", f"cr={row.get('cr')}"],
                    "suggested_action": suggested_action,
                    "marketing_source_status": "legacy_fallback",
                    "demand_source_status": demand_source_status,
                    "blockers": [],
                }
            )
    if not query_actions:
        legacy_query = get_marketing_query_performance_snapshot(
            product_keyword=product_keyword,
            days=days,
            limit=limit,
        )
        legacy_query_report_date = legacy_query.get("report_date")
        legacy_query_freshness = _source_freshness(
            report_date=legacy_query_report_date,
            reference_date=reference_date,
        )
        for row in (
            legacy_query.get("query_waste_rows") or []
            if legacy_query_freshness["status"] == "fresh"
            else []
        ):
            if not isinstance(row, dict):
                continue
            query_actions.append(
                {
                    "entity_type": "поисковый запрос",
                    "entity_id": row.get("norm_query"),
                    "entity_name": row.get("norm_query"),
                    "status": "сократить",
                    "action_mode": "reduce",
                    "reason": f"legacy_snapshot: затраты={row.get('spend')}, заказы={row.get('orders')}",
                    "evidence": [f"cpc={row.get('cpc')}", f"avg_pos={row.get('avg_pos')}"],
                    "suggested_action": "Legacy fallback из query snapshot.",
                    "marketing_source_status": "legacy_fallback",
                    "demand_source_status": demand_source_status,
                    "query_band": "waste",
                    "blockers": [],
                }
            )

    return {
        "report_date": (
            legacy_campaign_report_date
            or legacy_query_report_date
            or campaign_mart.get("report_date")
            or forecast_ready.get("report_date")
            or query_bands.get("report_date")
        ),
        "period_days": days,
        "focus_keyword": product_keyword,
        "marketing_source_status": marketing_source_status,
        "demand_source_status": demand_source_status,
        "source_freshness": {
            "reference_date": reference_date.isoformat() if reference_date else None,
            "stale_sources": stale_sources,
            "invalid_sources": invalid_sources,
            "sources": source_freshness,
            "legacy_campaign": legacy_campaign_freshness,
            "legacy_query": legacy_query_freshness,
            "interpretation": (
                "Only fresh rows are used for current bid actions; stale, future-dated, or undated rows are blocked."
                if invalid_sources else
                "All action sources are fresh enough for current bid actions."
            ),
        },
        "campaign_actions": campaign_actions[:limit],
        "query_actions": query_actions[:limit],
        "sku_actions": sku_actions[:limit],
        "supporting_sources": {
            "campaign_mart": campaign_mart.get("source_status"),
            "query_bands_mart": query_bands.get("source_status"),
            "forecast_ready_mart": forecast_ready.get("source_status"),
            "query_demand_mart": demand_daily.get("source_status"),
            "query_share_mart": demand_share.get("source_status"),
            "query_forecast_mart": demand_forecast.get("source_status"),
            "campaign_demand_mart": campaign_demand.get("source_status"),
            "legacy_demand_fallback": "used" if demand_fallback else "not_used",
            "legacy_visibility_fallback": "used" if visibility_rows else "not_used",
        },
        "summary": {
            "campaign_scale_count": sum(
                1 for row in campaign_actions if row.get("action_mode") == "scale_review"
            ),
            "campaign_stop_count": sum(
                1 for row in campaign_actions if row.get("action_mode") == "reduce"
            ),
            "query_scale_count": sum(
                1 for row in query_actions if row.get("action_mode") == "scale_review"
            ),
            "query_reduce_count": sum(
                1 for row in query_actions if row.get("action_mode") == "reduce"
            ),
            "visibility_investigate_count": sum(
                1 for row in sku_actions if row.get("action_mode") == "needs_review"
            ),
        },
    }


def get_marketing_deep_diagnostic(
    product_keyword: str | None = None,
    days: int = 7,
    limit: int = 20,
) -> dict[str, Any]:
    days = _validate_days(days)
    limit = _validate_limit(limit, upper=50)

    overview = get_marketing_overview(product_keyword=product_keyword, limit=min(limit, 10))
    campaign = get_marketing_campaign_efficiency_snapshot(
        product_keyword=product_keyword,
        days=days,
        limit=limit,
    )
    query = get_marketing_query_performance_snapshot(
        product_keyword=product_keyword,
        days=days,
        limit=limit,
    )
    bid = get_marketing_bid_pressure_snapshot(
        product_keyword=product_keyword,
        days=days,
        limit=limit,
    )
    visibility = get_marketing_visibility_snapshot(
        product_keyword=product_keyword,
        limit=limit,
    )
    demand = get_marketing_search_demand_snapshot(
        product_keyword=product_keyword,
        days=days,
        limit=limit,
    )
    action_matrix = get_marketing_action_matrix(
        product_keyword=product_keyword,
        days=days,
        limit=limit,
    )

    summary = {
        "top_campaign_scale_candidates": len(campaign["scale_rows"]),
        "top_campaign_waste_candidates": len(campaign["waste_rows"]),
        "query_scale_candidates": len(query["query_scale_rows"]),
        "query_waste_candidates": len(query["query_waste_rows"]),
        "overbid_candidates": len(bid["overbid_rows"]),
        "underbid_candidates": len(bid["underbid_rows"]),
        "visibility_risk_rows": len(visibility["position_loss_rows"]),
        "search_demand_growth_rows": len(demand["growing_query_rows"]),
    }

    data_gaps: list[dict[str, str]] = []
    if not query["top_query_rows"]:
        data_gaps.append(
            {
                "metric": "normquery_performance",
                "reason": "Нет query-level строк за выбранный период.",
                "impact": "Нельзя подтвердить performance по нормализованным запросам.",
            }
        )
    if not visibility["visibility_rows"]:
        data_gaps.append(
            {
                "metric": "search_visibility",
                "reason": "Нет строк по видимости и позиции в слое поисковых запросов.",
                "impact": "Нельзя подтвердить узкое место по позиции и видимости.",
            }
        )
    if not demand["search_text_rows"]:
        data_gaps.append(
            {
                "metric": "search_demand",
                "reason": "Нет search text demand строк за выбранный период.",
                "impact": "Нельзя подтвердить order-driving и high-frequency queries.",
            }
        )

    return {
        "report_date": campaign["report_date"] or overview["totals"].get("stat_date"),
        "period_days": days,
        "focus_keyword": product_keyword,
        "summary": summary,
        "campaign_efficiency": campaign,
        "query_performance": query,
        "bid_pressure": bid,
        "visibility": visibility,
        "search_demand": demand,
        "action_matrix": action_matrix,
        "data_gaps": data_gaps,
    }


def get_finance_overview(
    product_keyword: str | None = None,
    limit: int = PORTFOLIO_ANALYSIS_LIMIT,
) -> dict[str, Any]:
    limit = _validate_limit(limit, upper=PORTFOLIO_ANALYSIS_LIMIT)
    unit_clause, unit_params = _keyword_clause(
        product_keyword,
        ["vendor_code", "brand_name", "subject_name", "nm_id::text"],
    )
    fee_clause, fee_params = _keyword_clause(
        product_keyword,
        ["brand_name", "subject_name", "vendor_code", "nm_id::text"],
    )

    unit_query = f"""
        with latest_rr as (
            select coalesce(
                max(create_dt::date) filter (where create_dt::date < current_date),
                max(create_dt::date)
            ) as latest_date
            from finance.wb_report_detail_daily
        ),
        per_sku as (
            select
                nm_id,
                nullif(sa_name, '') as vendor_code,
                nullif(brand_name, '') as brand_name,
                nullif(subject_name, '') as subject_name,
                sum(coalesce(quantity, 0)) as quantity_sum,
                sum(coalesce(retail_amount, 0)) as retail_amount_sum,
                sum(coalesce(ppvz_for_pay, 0)) as payout_sum,
                sum(coalesce(ppvz_sales_commission, 0)) as commission_sum,
                sum(coalesce(acquiring_fee, 0)) as acquiring_fee_sum,
                sum(coalesce(delivery_rub, 0)) as delivery_rub_sum,
                sum(coalesce(storage_fee, 0)) as storage_fee_sum,
                sum(coalesce(penalty, 0)) as penalty_sum
            from finance.wb_report_detail_daily
            where create_dt::date between (select latest_date - interval '29 day' from latest_rr)
                                     and (select latest_date from latest_rr)
              and coalesce(nm_id, 0) > 0
            group by nm_id, nullif(sa_name, ''), nullif(brand_name, ''), nullif(subject_name, '')
        ),
        calc as (
            select
                (select latest_date from latest_rr) as snapshot_date,
                nm_id,
                vendor_code,
                brand_name,
                subject_name,
                quantity_sum,
                retail_amount_sum,
                payout_sum,
                commission_sum,
                acquiring_fee_sum,
                delivery_rub_sum,
                storage_fee_sum,
                penalty_sum,
                case
                    when coalesce(retail_amount_sum, 0) <> 0
                        then round(100 * payout_sum / retail_amount_sum, 2)
                end as payout_margin_before_cogs_pct,
                case
                    when coalesce(retail_amount_sum, 0) <> 0
                        then round(
                            100 * (
                                commission_sum
                                + acquiring_fee_sum
                                + delivery_rub_sum
                                + storage_fee_sum
                                + penalty_sum
                            ) / retail_amount_sum,
                            2
                        )
                end as fee_burden_pct
            from per_sku
        )
        select
            snapshot_date,
            nm_id,
            vendor_code,
            brand_name,
            subject_name,
            quantity_sum,
            retail_amount_sum,
            payout_sum,
            commission_sum,
            acquiring_fee_sum,
            delivery_rub_sum,
            storage_fee_sum,
            penalty_sum,
            payout_margin_before_cogs_pct,
            fee_burden_pct
        from calc
        where coalesce(retail_amount_sum, 0) > 0
        {unit_clause}
        order by coalesce(payout_margin_before_cogs_pct, 999999) asc, coalesce(retail_amount_sum, 0) desc
        limit %s
    """
    margin_query = f"""
        with latest_rr as (
            select coalesce(
                max(create_dt::date) filter (where create_dt::date < current_date),
                max(create_dt::date)
            ) as latest_date
            from finance.wb_report_detail_daily
        ),
        per_sku as (
            select
                nm_id,
                nullif(sa_name, '') as vendor_code,
                nullif(brand_name, '') as brand_name,
                nullif(subject_name, '') as subject_name,
                sum(coalesce(retail_amount, 0)) as retail_amount_sum,
                sum(coalesce(ppvz_for_pay, 0)) as payout_sum,
                sum(coalesce(quantity, 0)) as quantity_sum
            from finance.wb_report_detail_daily
            where create_dt::date between (select latest_date - interval '29 day' from latest_rr)
                                     and (select latest_date from latest_rr)
              and coalesce(nm_id, 0) > 0
            group by nm_id, nullif(sa_name, ''), nullif(brand_name, ''), nullif(subject_name, '')
        ),
        calc as (
            select
                (select latest_date from latest_rr) as report_date,
                nm_id,
                vendor_code,
                brand_name,
                subject_name,
                quantity_sum,
                retail_amount_sum,
                payout_sum,
                case
                    when coalesce(retail_amount_sum, 0) <> 0
                        then round(100 * payout_sum / retail_amount_sum, 2)
                end as payout_margin_before_cogs_pct
            from per_sku
        )
        select
            report_date,
            nm_id,
            vendor_code,
            brand_name,
            subject_name,
            quantity_sum,
            retail_amount_sum,
            payout_sum,
            payout_margin_before_cogs_pct
        from calc
        where coalesce(retail_amount_sum, 0) > 0
        {unit_clause}
        order by coalesce(payout_margin_before_cogs_pct, 999999) asc, coalesce(retail_amount_sum, 0) desc
        limit %s
    """
    fees_query = f"""
        with latest_rr as (
            select coalesce(
                max(create_dt::date) filter (where create_dt::date < current_date),
                max(create_dt::date)
            ) as latest_date
            from finance.wb_report_detail_daily
        ),
        per_sku as (
            select
                nm_id,
                nullif(sa_name, '') as vendor_code,
                nullif(brand_name, '') as brand_name,
                nullif(subject_name, '') as subject_name,
                sum(coalesce(retail_amount, 0)) as retail_amount_sum,
                sum(coalesce(ppvz_for_pay, 0)) as payout_sum,
                sum(coalesce(acquiring_fee, 0)) as acquiring_fee_sum,
                sum(coalesce(delivery_rub, 0)) as delivery_rub_sum,
                sum(coalesce(storage_fee, 0)) as storage_fee_sum,
                sum(coalesce(penalty, 0)) as penalty_sum
            from finance.wb_report_detail_daily
            where create_dt::date between (select latest_date - interval '29 day' from latest_rr)
                                     and (select latest_date from latest_rr)
              and coalesce(nm_id, 0) > 0
            group by nm_id, nullif(sa_name, ''), nullif(brand_name, ''), nullif(subject_name, '')
        )
        select
            nm_id,
            vendor_code,
            brand_name,
            subject_name,
            retail_amount_sum,
            payout_sum,
            acquiring_fee_sum,
            delivery_rub_sum,
            storage_fee_sum,
            penalty_sum
        from per_sku
        where 1 = 1
        {fee_clause}
        order by
            (
                coalesce(acquiring_fee_sum, 0)
                + coalesce(delivery_rub_sum, 0)
                + coalesce(storage_fee_sum, 0)
                + coalesce(penalty_sum, 0)
            ) desc,
            coalesce(retail_amount_sum, 0) desc
        limit %s
    """

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(unit_query, [*unit_params, limit])
        unit_rows = serialize_rows(cur.fetchall())

        cur.execute(margin_query, [*unit_params, limit])
        margin_rows = serialize_rows(cur.fetchall())

        cur.execute(fees_query, [*fee_params, limit])
        fee_rows = serialize_rows(cur.fetchall())

    return {
        "focus_keyword": product_keyword,
        "unit_economics_watchlist": unit_rows,
        "margin_watchlist": margin_rows,
        "fee_pressure_rows": fee_rows,
        "guardrails_scope": [
            "Проверяем себестоимость по finance.wb_product_coast, комиссии, эквайринг, доставку, хранение, штрафы и профиль выплаты.",
            "Не подтверждаем кассовый разрыв, давление оборотного капитала или прибыльность после закупочной себестоимости для SKU без строки себестоимости.",
        ],
    }
