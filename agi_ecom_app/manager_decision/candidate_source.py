from __future__ import annotations

from datetime import datetime, timezone
import os
from typing import Any

from agi_ecom_app.services.strategic_db import (
    get_connection,
    get_supply_overview,
    serialize_rows,
)


# Feature-off must keep the pre-simulation campaign-level data contract.
_LEGACY_CAMPAIGN_SNAPSHOT_SQL = """
with latest_campaigns as (
    select distinct on (external_id)
        external_id as advert_id,
        snapshot_date as campaign_snapshot_date,
        name as campaign_name,
        status as campaign_status,
        payment_type,
        last_synced_at as campaign_retrieved_at
    from marketing.wb_marketing_campaigns
    order by external_id, snapshot_date desc, last_synced_at desc, id desc
),
latest_nms as (
    select distinct on (advert_id, nm_id)
        advert_id,
        nm_id,
        subject_name,
        bid_search_kopecks,
        bid_recommendations_kopecks,
        snapshot_date as bid_snapshot_date,
        last_synced_at as bid_retrieved_at
    from marketing.wb_marketing_campaign_nms
    order by advert_id, nm_id, snapshot_date desc, last_synced_at desc, id desc
),
nm_counts as (
    select advert_id, count(*)::int as active_sku_count
    from latest_nms
    group by advert_id
),
latest_budget as (
    select distinct on (advert_id)
        advert_id,
        snapshot_date as budget_snapshot_date,
        total as budget_remaining_rub,
        last_synced_at as budget_retrieved_at
    from marketing.wb_marketing_budget
    order by advert_id, snapshot_date desc, last_synced_at desc, id desc
),
latest_stats_day as (
    select max(stat_date) filter (
        where stat_date <= (current_date - interval '1 day')::date
    ) as report_date
    from marketing.wb_marketing_campaign_stats_daily
),
daily_stats as (
    select
        stat_date,
        advert_id,
        sum(coalesce(sum_spend, 0))::numeric as spend_rub,
        sum(coalesce(clicks, 0))::numeric as clicks,
        sum(coalesce(orders, 0))::numeric as orders,
        sum(coalesce(shks, 0))::numeric as units
    from marketing.wb_marketing_campaign_stats_daily
    where stat_date between
        (select report_date - interval '6 day' from latest_stats_day)
        and (select report_date from latest_stats_day)
    group by stat_date, advert_id
),
stats_7d as (
    select
        advert_id,
        min(stat_date) as stats_period_start,
        max(stat_date) as stats_period_end,
        count(distinct stat_date)::int as stats_day_count,
        round(sum(spend_rub), 2) as ad_spend_7d_rub,
        round(sum(clicks), 2) as clicks_7d,
        round(sum(orders), 2) as ad_orders_7d,
        round(sum(units), 2) as ad_units_7d,
        round(min(spend_rub), 2) as daily_spend_min_rub,
        round(percentile_cont(0.5) within group (order by spend_rub)::numeric, 2)
            as daily_spend_median_rub,
        round(max(spend_rub), 2) as daily_spend_max_rub,
        jsonb_agg(round(spend_rub, 2) order by stat_date) as daily_spend_values_rub
    from daily_stats
    group by advert_id
),
latest_stock_day as (
    select max(snapshot_date) filter (
        where snapshot_date <= (current_date - interval '1 day')::date
    ) as snapshot_date
    from supply.wb_supplier_stocks
),
stocks as (
    select
        snapshot_date as stock_snapshot_date,
        nm_id,
        sum(coalesce(quantity, 0))::numeric as stock_on_hand,
        sum(coalesce(quantity_full, 0))::numeric as stock_full_for_audit
    from supply.wb_supplier_stocks
    where snapshot_date = (select snapshot_date from latest_stock_day)
    group by snapshot_date, nm_id
),
latest_supply_day as (
    select max(snapshot_date) filter (
        where snapshot_date <= (current_date - interval '1 day')::date
    ) as snapshot_date
    from supply.wb_supplies
),
supplier_inbound as (
    select
        g.nm_id,
        s.snapshot_date as supplier_snapshot_date,
        sum(greatest(coalesce(g.quantity, 0) - coalesce(g.ready_for_sale_quantity, 0), 0))::numeric
            as supplier_inbound_qty,
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
latest_cost as (
    select distinct on (nm_id)
        nm_id,
        cost_price_rub as unit_cogs_rub,
        upload_date as cogs_snapshot_date,
        updated_at as cogs_retrieved_at
    from finance.wb_product_coast
    where nm_id is not null
    order by nm_id, upload_date desc, updated_at desc
)
select
    c.advert_id,
    c.campaign_name,
    c.campaign_status,
    c.payment_type,
    c.campaign_snapshot_date,
    c.campaign_retrieved_at,
    n.nm_id,
    n.subject_name,
    nc.active_sku_count,
    n.bid_search_kopecks,
    n.bid_recommendations_kopecks,
    n.bid_snapshot_date,
    n.bid_retrieved_at,
    b.budget_remaining_rub,
    b.budget_snapshot_date,
    b.budget_retrieved_at,
    st.stats_period_start,
    st.stats_period_end,
    st.stats_day_count,
    st.ad_spend_7d_rub,
    st.clicks_7d,
    st.ad_orders_7d,
    st.ad_units_7d,
    st.daily_spend_min_rub,
    st.daily_spend_median_rub,
    st.daily_spend_max_rub,
    st.daily_spend_values_rub,
    stock.stock_snapshot_date,
    stock.stock_on_hand,
    stock.stock_full_for_audit,
    inbound.supplier_snapshot_date,
    inbound.supplier_inbound_qty,
    inbound.supplier_goods_rows,
    supply_day.snapshot_date as supplier_audit_snapshot_date,
    case
        when supply_day.snapshot_date is not null
        then coalesce(inbound.supplier_inbound_qty, 0::numeric)
    end as supplier_inbound_audited_qty,
    inbound.supplier_eta_date,
    case
        when supply_day.snapshot_date is not null
        then coalesce(inbound.supplier_goods_rows, 0)
    end as supplier_goods_rows_audited,
    cost.unit_cogs_rub,
    cost.cogs_snapshot_date,
    cost.cogs_retrieved_at,
    case
        when st.stats_day_count = 7
         and st.ad_orders_7d = 0
         and st.ad_units_7d = 0
        then 0::numeric
    end as attributed_revenue_7d_rub,
    case
        when st.stats_day_count = 7
         and st.ad_orders_7d = 0
         and st.ad_units_7d = 0
        then 'derived:zero_attributed_orders(marketing.wb_marketing_campaign_stats_daily)'
    end as attributed_revenue_source_table,
    case
        when st.stats_day_count = 7
         and st.ad_orders_7d = 0
         and st.ad_units_7d = 0
        then -st.ad_spend_7d_rub
    end as campaign_contribution_profit_7d_rub,
    case
        when st.stats_day_count = 7
         and st.ad_orders_7d = 0
         and st.ad_units_7d = 0
        then 'derived:zero_order_contribution_after_ad_spend_v1(marketing.wb_marketing_campaign_stats_daily)'
    end as campaign_contribution_source_table
from latest_campaigns c
join latest_nms n on n.advert_id = c.advert_id
join nm_counts nc on nc.advert_id = c.advert_id
left join latest_budget b on b.advert_id = c.advert_id
left join stats_7d st on st.advert_id = c.advert_id
left join stocks stock on stock.nm_id = n.nm_id
left join latest_supply_day supply_day on true
left join supplier_inbound inbound on inbound.nm_id = n.nm_id
left join latest_cost cost on cost.nm_id = n.nm_id
order by
    case when c.campaign_status = 9 and coalesce(stock.stock_on_hand, 0) = 0 then 0 else 1 end,
    coalesce(st.ad_spend_7d_rub, 0) desc,
    c.advert_id,
    n.nm_id
limit %s
"""


# Public name: scripts/apply_agent_marts.py imports this exact query for EXPLAIN.
_CAMPAIGN_SNAPSHOT_SQL = """
with latest_campaigns as (
    select distinct on (external_id)
        external_id as advert_id,
        snapshot_date as campaign_snapshot_date,
        name as campaign_name,
        status as campaign_status,
        payment_type,
        last_synced_at as campaign_retrieved_at
    from marketing.wb_marketing_campaigns
    order by external_id, snapshot_date desc, last_synced_at desc, id desc
),
latest_nms as (
    select distinct on (advert_id, nm_id)
        advert_id,
        nm_id,
        subject_name,
        bid_search_kopecks,
        bid_recommendations_kopecks,
        snapshot_date as bid_snapshot_date,
        last_synced_at as bid_retrieved_at
    from marketing.wb_marketing_campaign_nms
    order by advert_id, nm_id, snapshot_date desc, last_synced_at desc, id desc
),
nm_counts as (
    select advert_id, count(*)::int as active_sku_count
    from latest_nms
    group by advert_id
),
latest_budget as (
    select distinct on (advert_id)
        advert_id,
        snapshot_date as budget_snapshot_date,
        total as budget_remaining_rub,
        last_synced_at as budget_retrieved_at
    from marketing.wb_marketing_budget
    order by advert_id, snapshot_date desc, last_synced_at desc, id desc
),
latest_complete_stats_day as (
    select max(stat_date) as report_date
    from marketing.wb_marketing_campaign_stats_daily_watermarks
    where stat_date <= (current_date - interval '1 day')::date
      and campaign_stats_complete is true
      and campaign_sku_stats_complete is true
),
latest_raw_sku_stats as (
    select distinct on (raw.stat_date, raw.advert_id, raw.nm_id)
        raw.stat_date,
        raw.advert_id,
        raw.nm_id,
        raw.sum_spend
    from marketing.wb_marketing_campaign_stats_daily_nms raw
    where raw.stat_date <= (current_date - interval '1 day')::date
    order by
        raw.stat_date,
        raw.advert_id,
        raw.nm_id,
        raw.last_synced_at desc nulls last,
        raw.id desc
),
latest_raw_stats_day as (
    select max(stat_date) as report_date
    from latest_raw_sku_stats
),
diagnostic_stats_7d as (
    -- Raw rows are used only to keep an active/spending campaign visible as a
    -- needs-data diagnostic. They never unlock economics or a simulation.
    select
        advert_id,
        nm_id,
        round(sum(sum_spend), 2) as observed_ad_spend_7d_rub
    from latest_raw_sku_stats
    where stat_date between
        (select report_date - interval '6 day' from latest_raw_stats_day)
        and (select report_date from latest_raw_stats_day)
    group by advert_id, nm_id
),
normalized_daily_stats as (
    select
        mart.report_date as stat_date,
        mart.advert_id,
        mart.nm_id,
        mart.stats_row_present,
        mart.stats_evidence_origin,
        mart.stats_day_watermark_complete,
        watermark.campaign_stats_complete,
        watermark.campaign_sku_stats_complete,
        mart.ad_metrics_complete,
        mart.attributed_revenue_complete,
        mart.campaign_attributed_revenue_complete,
        mart.campaign_sku_attribution_reconciled,
        mart.ad_spend_rub as spend_rub,
        mart.clicks,
        mart.ad_orders as orders,
        mart.ad_units as units,
        mart.attributed_order_revenue_rub as attributed_revenue_rub
    from agent_marts.v_agent_ad_campaign_sku_daily mart
    join marketing.wb_marketing_campaign_stats_daily_watermarks watermark
      on watermark.stat_date = mart.report_date
    where mart.report_date between
        (select report_date - interval '6 day' from latest_complete_stats_day)
        and (select report_date from latest_complete_stats_day)
),
stats_rollup as (
    select
        advert_id,
        nm_id,
        min(stat_date) as stats_period_start,
        max(stat_date) as stats_period_end,
        count(distinct stat_date)::int as stats_day_count,
        count(*) filter (
            where stats_evidence_origin = 'complete_day_absence'
        )::int as complete_day_absence_count,
        jsonb_agg(stats_evidence_origin order by stat_date)
            as stats_evidence_origins,
        bool_and(
            stats_day_watermark_complete
            and campaign_stats_complete
            and campaign_sku_stats_complete
        ) as watermark_window_complete,
        bool_and(ad_metrics_complete) as ad_metrics_window_complete,
        bool_and(attributed_revenue_complete)
            as attributed_revenue_window_complete,
        bool_and(campaign_attributed_revenue_complete)
            as campaign_attributed_revenue_window_complete,
        bool_and(campaign_sku_attribution_reconciled is true)
            as campaign_sku_attribution_reconciled,
        sum(spend_rub)::numeric as spend_rub,
        sum(clicks)::numeric as clicks,
        sum(orders)::numeric as orders,
        sum(units)::numeric as units,
        sum(attributed_revenue_rub)::numeric as attributed_revenue_rub,
        min(spend_rub)::numeric as daily_spend_min_rub,
        percentile_cont(0.5) within group (order by spend_rub)::numeric
            as daily_spend_median_rub,
        max(spend_rub)::numeric as daily_spend_max_rub,
        jsonb_agg(round(spend_rub, 2) order by stat_date)
            as daily_spend_values_rub
    from normalized_daily_stats
    group by advert_id, nm_id
),
stats_7d as (
    select
        advert_id,
        nm_id,
        stats_period_start,
        stats_period_end,
        stats_day_count,
        complete_day_absence_count,
        stats_evidence_origins,
        watermark_window_complete,
        ad_metrics_window_complete as spend_window_complete,
        ad_metrics_window_complete as clicks_window_complete,
        ad_metrics_window_complete as orders_window_complete,
        ad_metrics_window_complete as units_window_complete,
        attributed_revenue_window_complete,
        campaign_attributed_revenue_window_complete,
        campaign_sku_attribution_reconciled,
        (
            stats_day_count = 7
            and stats_period_end - stats_period_start = 6
            and watermark_window_complete
            and ad_metrics_window_complete
            and attributed_revenue_window_complete
            and campaign_attributed_revenue_window_complete
            and campaign_sku_attribution_reconciled
        ) as stats_window_complete,
        -- Keep the observed spend for a rejected/needs-data diagnostic row.
        -- It must never be used as deterministic economics unless
        -- stats_window_complete is true.
        round(spend_rub, 2) as observed_ad_spend_7d_rub,
        case
            when stats_day_count = 7
             and stats_period_end - stats_period_start = 6
             and watermark_window_complete
             and ad_metrics_window_complete
             and attributed_revenue_window_complete
             and campaign_attributed_revenue_window_complete
             and campaign_sku_attribution_reconciled
            then round(spend_rub, 2)
        end as ad_spend_7d_rub,
        case
            when stats_day_count = 7
             and stats_period_end - stats_period_start = 6
             and watermark_window_complete
             and ad_metrics_window_complete
             and attributed_revenue_window_complete
             and campaign_attributed_revenue_window_complete
             and campaign_sku_attribution_reconciled
            then round(clicks, 2)
        end as clicks_7d,
        case
            when stats_day_count = 7
             and stats_period_end - stats_period_start = 6
             and watermark_window_complete
             and ad_metrics_window_complete
             and attributed_revenue_window_complete
             and campaign_attributed_revenue_window_complete
             and campaign_sku_attribution_reconciled
            then round(orders, 2)
        end as ad_orders_7d,
        case
            when stats_day_count = 7
             and stats_period_end - stats_period_start = 6
             and watermark_window_complete
             and ad_metrics_window_complete
             and attributed_revenue_window_complete
             and campaign_attributed_revenue_window_complete
             and campaign_sku_attribution_reconciled
            then round(units, 2)
        end as ad_units_7d,
        case
            when stats_day_count = 7
             and stats_period_end - stats_period_start = 6
             and watermark_window_complete
             and ad_metrics_window_complete
             and attributed_revenue_window_complete
             and campaign_attributed_revenue_window_complete
             and campaign_sku_attribution_reconciled
            then round(attributed_revenue_rub, 2)
        end as attributed_revenue_7d_rub,
        case
            when stats_day_count = 7
             and stats_period_end - stats_period_start = 6
             and watermark_window_complete
             and ad_metrics_window_complete
             and attributed_revenue_window_complete
             and campaign_attributed_revenue_window_complete
             and campaign_sku_attribution_reconciled
            then round(daily_spend_min_rub, 2)
        end as daily_spend_min_rub,
        case
            when stats_day_count = 7
             and stats_period_end - stats_period_start = 6
             and watermark_window_complete
             and ad_metrics_window_complete
             and attributed_revenue_window_complete
             and campaign_attributed_revenue_window_complete
             and campaign_sku_attribution_reconciled
            then round(daily_spend_median_rub, 2)
        end as daily_spend_median_rub,
        case
            when stats_day_count = 7
             and stats_period_end - stats_period_start = 6
             and watermark_window_complete
             and ad_metrics_window_complete
             and attributed_revenue_window_complete
             and campaign_attributed_revenue_window_complete
             and campaign_sku_attribution_reconciled
            then round(daily_spend_max_rub, 2)
        end as daily_spend_max_rub,
        case
            when stats_day_count = 7
             and stats_period_end - stats_period_start = 6
             and watermark_window_complete
             and ad_metrics_window_complete
             and attributed_revenue_window_complete
             and campaign_attributed_revenue_window_complete
             and campaign_sku_attribution_reconciled
            then daily_spend_values_rub
        end as daily_spend_values_rub
    from stats_rollup
),
latest_stock_day as (
    select max(snapshot_date) filter (
        where snapshot_date <= (current_date - interval '1 day')::date
    ) as snapshot_date
    from supply.wb_supplier_stocks
),
stocks as (
    select
        snapshot_date as stock_snapshot_date,
        nm_id,
        sum(coalesce(quantity, 0))::numeric as stock_on_hand,
        sum(coalesce(quantity_full, 0))::numeric as stock_full_for_audit
    from supply.wb_supplier_stocks
    where snapshot_date = (select snapshot_date from latest_stock_day)
    group by snapshot_date, nm_id
),
latest_supply_day as (
    select max(snapshot_date) filter (
        where snapshot_date <= (current_date - interval '1 day')::date
    ) as snapshot_date
    from supply.wb_supplies
),
supplier_inbound as (
    select
        g.nm_id,
        s.snapshot_date as supplier_snapshot_date,
        sum(greatest(coalesce(g.quantity, 0) - coalesce(g.ready_for_sale_quantity, 0), 0))::numeric
            as supplier_inbound_qty,
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
latest_cost as (
    select distinct on (nm_id)
        nm_id,
        cost_price_rub as unit_cogs_rub,
        upload_date as cogs_snapshot_date,
        updated_at as cogs_retrieved_at
    from finance.wb_product_coast
    where nm_id is not null
    order by nm_id, upload_date desc, updated_at desc
)
select
    c.advert_id,
    c.campaign_name,
    c.campaign_status,
    c.payment_type,
    c.campaign_snapshot_date,
    c.campaign_retrieved_at,
    n.nm_id,
    n.subject_name,
    nc.active_sku_count,
    n.bid_search_kopecks,
    n.bid_recommendations_kopecks,
    n.bid_snapshot_date,
    n.bid_retrieved_at,
    b.budget_remaining_rub,
    b.budget_snapshot_date,
    b.budget_retrieved_at,
    st.stats_period_start,
    st.stats_period_end,
    st.stats_day_count,
    st.complete_day_absence_count,
    st.stats_evidence_origins,
    st.watermark_window_complete,
    st.stats_window_complete,
    st.spend_window_complete,
    st.clicks_window_complete,
    st.orders_window_complete,
    st.units_window_complete,
    st.attributed_revenue_window_complete,
    st.campaign_attributed_revenue_window_complete,
    st.campaign_sku_attribution_reconciled,
    coalesce(st.observed_ad_spend_7d_rub, diagnostic.observed_ad_spend_7d_rub)
        as observed_ad_spend_7d_rub,
    st.ad_spend_7d_rub,
    st.clicks_7d,
    st.ad_orders_7d,
    st.ad_units_7d,
    st.daily_spend_min_rub,
    st.daily_spend_median_rub,
    st.daily_spend_max_rub,
    st.daily_spend_values_rub,
    stock.stock_snapshot_date,
    stock.stock_on_hand,
    stock.stock_full_for_audit,
    -- Presence in raw snapshots is not proof that the full snapshot completed.
    false::boolean as stock_snapshot_complete,
    inbound.supplier_snapshot_date,
    inbound.supplier_inbound_qty,
    inbound.supplier_goods_rows,
    supply_day.snapshot_date as supplier_audit_snapshot_date,
    case
        when supply_day.snapshot_date is not null
        then coalesce(inbound.supplier_inbound_qty, 0::numeric)
    end as supplier_inbound_audited_qty,
    inbound.supplier_eta_date,
    case
        when supply_day.snapshot_date is not null
        then coalesce(inbound.supplier_goods_rows, 0)
    end as supplier_goods_rows_audited,
    false::boolean as supplier_snapshot_complete,
    cost.unit_cogs_rub,
    cost.cogs_snapshot_date,
    cost.cogs_retrieved_at,
    st.attributed_revenue_7d_rub,
    case
        when st.stats_window_complete is true
        then 'agent_marts.v_agent_ad_campaign_sku_daily.attributed_order_revenue_rub'
    end as attributed_revenue_source_table,
    case
        when st.stats_window_complete is true
         and st.ad_orders_7d = 0
         and st.ad_units_7d = 0
         and st.attributed_revenue_7d_rub = 0
        then -st.ad_spend_7d_rub
    end as campaign_contribution_profit_7d_rub,
    case
        when st.stats_window_complete is true
         and st.ad_orders_7d = 0
         and st.ad_units_7d = 0
         and st.attributed_revenue_7d_rub = 0
        then 'derived:zero_order_contribution_after_ad_spend_v1(agent_marts.v_agent_ad_campaign_sku_daily)'
    end as campaign_contribution_source_table
from latest_campaigns c
join latest_nms n on n.advert_id = c.advert_id
join nm_counts nc on nc.advert_id = c.advert_id
left join latest_budget b on b.advert_id = c.advert_id
left join stats_7d st
  on st.advert_id = c.advert_id
 and st.nm_id = n.nm_id
left join diagnostic_stats_7d diagnostic
  on diagnostic.advert_id = c.advert_id
 and diagnostic.nm_id = n.nm_id
left join stocks stock on stock.nm_id = n.nm_id
left join latest_supply_day supply_day on true
left join supplier_inbound inbound on inbound.nm_id = n.nm_id
left join latest_cost cost on cost.nm_id = n.nm_id
order by
    case when c.campaign_status = 9 and coalesce(stock.stock_on_hand, 0) = 0 then 0 else 1 end,
    coalesce(st.ad_spend_7d_rub, 0) desc,
    c.advert_id,
    n.nm_id
limit %s
"""


_SUPPLY_SUPPORT_SQL = """
with requested as (
    select unnest(%s::bigint[]) as nm_id
),
latest_cost as (
    select distinct on (c.nm_id)
        c.nm_id,
        c.cost_price_rub as unit_cogs_rub,
        c.upload_date as cogs_snapshot_date,
        c.updated_at as cogs_retrieved_at
    from finance.wb_product_coast c
    join requested r on r.nm_id = c.nm_id
    order by c.nm_id, c.upload_date desc, c.updated_at desc
),
latest_supply_day as (
    select max(snapshot_date) filter (
        where snapshot_date <= (current_date - interval '1 day')::date
    ) as snapshot_date
    from supply.wb_supplies
),
supplier_inbound as (
    select
        g.nm_id,
        s.snapshot_date as supplier_snapshot_date,
        sum(greatest(coalesce(g.quantity, 0) - coalesce(g.ready_for_sale_quantity, 0), 0))::numeric
            as supplier_inbound_qty,
        min(s.supply_date) as supplier_eta_date,
        count(*)::int as supplier_goods_rows
    from supply.wb_supplies s
    join supply.wb_supplies_goods g
      on g.snapshot_date = s.snapshot_date
     and g.supply_id = s.supply_id
    join requested r on r.nm_id = g.nm_id
    where s.snapshot_date = (select snapshot_date from latest_supply_day)
      and (
            coalesce(g.quantity, 0) > coalesce(g.ready_for_sale_quantity, 0)
            or coalesce(g.unloading_quantity, 0) > 0
          )
    group by g.nm_id, s.snapshot_date
)
select
    r.nm_id,
    cost.unit_cogs_rub,
    cost.cogs_snapshot_date,
    cost.cogs_retrieved_at,
    inbound.supplier_snapshot_date,
    inbound.supplier_inbound_qty,
    inbound.supplier_goods_rows,
    supply_day.snapshot_date as supplier_audit_snapshot_date,
    case
        when supply_day.snapshot_date is not null
        then coalesce(inbound.supplier_inbound_qty, 0::numeric)
    end as supplier_inbound_audited_qty,
    inbound.supplier_eta_date,
    case
        when supply_day.snapshot_date is not null
        then coalesce(inbound.supplier_goods_rows, 0)
    end as supplier_goods_rows_audited,
    null::numeric as lead_time_days,
    null::numeric as target_cover_days,
    null::numeric as minimum_order_qty,
    null::text as replenishment_policy_version
from requested r
left join latest_cost cost on cost.nm_id = r.nm_id
left join latest_supply_day supply_day on true
left join supplier_inbound inbound on inbound.nm_id = r.nm_id
order by r.nm_id
"""


def load_manager_candidate_snapshot(*, limit: int = 100) -> dict[str, Any]:
    """Read an action-oriented snapshot without calling WB or mutating state."""

    bounded_limit = max(1, min(int(limit), 200))
    retrieved_at = datetime.now(timezone.utc).isoformat()
    simulation_mode = os.getenv("DECISION_SIMULATION_MODE", "off").strip().lower()
    simulation_enabled = simulation_mode in {"shadow", "enforced"}
    campaign_snapshot_sql = (
        _CAMPAIGN_SNAPSHOT_SQL
        if simulation_enabled
        else _LEGACY_CAMPAIGN_SNAPSHOT_SQL
    )
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(campaign_snapshot_sql, [bounded_limit])
        campaign_rows = serialize_rows(cur.fetchall())

    supply_overview = get_supply_overview(limit=100)
    supply_rows = list(supply_overview.get("critical_reorder_rows") or [])
    nm_ids = sorted(
        {
            int(row["nm_id"])
            for row in supply_rows
            if row.get("nm_id") is not None
        }
    )
    support_by_nm: dict[int, dict[str, Any]] = {}
    if nm_ids:
        with get_connection() as conn, conn.cursor() as cur:
            cur.execute(_SUPPLY_SUPPORT_SQL, [nm_ids])
            support_by_nm = {
                int(row["nm_id"]): row
                for row in serialize_rows(cur.fetchall())
                if row.get("nm_id") is not None
            }
    enriched_supply_rows = [
        {
            **row,
            **support_by_nm.get(int(row["nm_id"]), {}),
        }
        for row in supply_rows
        if row.get("nm_id") is not None
    ]
    stats_source_tables = (
        [
            "agent_marts.v_agent_ad_campaign_sku_daily",
            "marketing.wb_marketing_campaign_stats_daily_watermarks",
        ]
        if simulation_enabled
        else ["marketing.wb_marketing_campaign_stats_daily"]
    )
    return {
        "schemaVersion": "1.0",
        "retrievedAt": retrieved_at,
        "advertisingRows": campaign_rows,
        "supplyRows": enriched_supply_rows,
        "sourceTables": [
            "marketing.wb_marketing_campaigns",
            "marketing.wb_marketing_campaign_nms",
            "marketing.wb_marketing_budget",
            *stats_source_tables,
            "supply.wb_supplier_stocks",
            "supply.wb_supplies",
            "supply.wb_supplies_goods",
            "sales.wb_supplier_sales",
            "finance.wb_product_coast",
        ],
    }


__all__ = ["load_manager_candidate_snapshot"]
