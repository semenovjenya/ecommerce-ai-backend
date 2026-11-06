-- One row per WB advertising campaign, SKU and evidence day.
-- Apply 01_add_ad_attribution_columns.sql and the four base agent marts first.
--
-- The date spine includes finance and stock days plus completed advertising
-- watermark days for 35 days after the last observed campaign/SKU membership.
-- This keeps post-pause measurement rows while remaining bounded. Missing ad
-- rows become proven zeroes only when the ETL watermark marks the whole day as
-- complete. Without that watermark, completeness is false and economics stay
-- NULL rather than silently treating missing source data as zero.
--
-- Direct WB fullstats attribution does not prove causal lift. causal_valid is
-- therefore false; causal validation belongs to the experiment layer.

create or replace view {{target_schema}}.v_agent_ad_campaign_sku_daily as
with membership_observations as (
    select
        n.advert_id,
        n.nm_id,
        n.snapshot_date as observed_date
    from marketing.wb_marketing_campaign_nms n
    union all
    select
        s.advert_id,
        s.nm_id,
        s.stat_date as observed_date
    from marketing.wb_marketing_campaign_stats_daily_nms s
),
membership_windows as (
    select
        o.advert_id,
        o.nm_id,
        min(o.observed_date) as first_observed_date,
        max(o.observed_date) as last_observed_date
    from membership_observations o
    where o.nm_id is not null
    group by o.advert_id, o.nm_id
),
latest_campaign_nms as (
    select distinct on (n.advert_id, n.nm_id)
        n.advert_id,
        n.nm_id,
        nullif(n.subject_name, '') as subject_name
    from marketing.wb_marketing_campaign_nms n
    order by
        n.advert_id,
        n.nm_id,
        n.snapshot_date desc,
        n.last_synced_at desc,
        n.id desc
),
stock_daily as (
    select
        s.snapshot_date,
        s.nm_id,
        sum(coalesce(s.quantity, 0))::numeric as stock_on_hand,
        sum(coalesce(s.quantity_full, 0))::numeric as stock_full_for_audit,
        max(s.last_synced_at) as stock_last_synced_at
    from supply.wb_supplier_stocks s
    where s.nm_id is not null
    group by s.snapshot_date, s.nm_id
),
supply_snapshot_days as (
    select
        s.snapshot_date,
        max(s.last_synced_at) as supplier_snapshot_last_synced_at
    from supply.wb_supplies s
    group by s.snapshot_date
),
supplier_inbound_daily as (
    select
        s.snapshot_date,
        g.nm_id,
        sum(
            greatest(
                coalesce(g.quantity, 0) - coalesce(g.ready_for_sale_quantity, 0),
                0
            )
        )::numeric as supplier_inbound_qty,
        min(s.supply_date) filter (
            where coalesce(g.quantity, 0) > coalesce(g.ready_for_sale_quantity, 0)
               or coalesce(g.unloading_quantity, 0) > 0
        ) as supplier_eta_date,
        max(g.last_synced_at) as supplier_goods_last_synced_at
    from supply.wb_supplies s
    join supply.wb_supplies_goods g
      on g.snapshot_date = s.snapshot_date
     and g.supply_id = s.supply_id
    where g.nm_id is not null
      and (
            coalesce(g.quantity, 0) > coalesce(g.ready_for_sale_quantity, 0)
            or coalesce(g.unloading_quantity, 0) > 0
          )
    group by s.snapshot_date, g.nm_id
),
finance_daily as (
    select
        f.report_date,
        f.nm_id,
        sum(f.net_revenue) as sku_net_revenue_rub,
        sum(f.marketplace_costs_total) as sku_marketplace_costs_rub,
        case
            when bool_and(f.cogs_total is not null) then sum(f.cogs_total)
        end as sku_cogs_rub,
        case
            when bool_and(f.contribution_margin is not null)
                then sum(f.contribution_margin)
        end as sku_contribution_before_ads_rub,
        bool_and(f.contribution_margin is not null) as finance_complete,
        count(*)::integer as finance_row_count
    from {{target_schema}}.v_agent_finance_sku_daily f
    where f.nm_id is not null
    group by f.report_date, f.nm_id
),
finance_sync_daily as (
    select
        r.rr_dt as report_date,
        r.nm_id,
        max(r.last_synced_at) as finance_last_synced_at
    from finance.wb_report_detail_daily r
    where r.rr_dt is not null
      and r.nm_id is not null
    group by r.rr_dt, r.nm_id
),
available_sku_days as (
    select s.stat_date as report_date, s.nm_id
    from marketing.wb_marketing_campaign_stats_daily_nms s
    union
    select f.report_date, f.nm_id
    from finance_daily f
    union
    select s.snapshot_date, s.nm_id
    from stock_daily s
),
date_spine as (
    select
        d.report_date,
        membership.advert_id,
        membership.nm_id
    from available_sku_days d
    join membership_windows membership
      on membership.nm_id = d.nm_id
     and d.report_date between membership.first_observed_date
                           and membership.last_observed_date + 35
    union
    select
        watermark.stat_date,
        membership.advert_id,
        membership.nm_id
    from marketing.wb_marketing_campaign_stats_daily_watermarks watermark
    join membership_windows membership
      on watermark.stat_date between membership.first_observed_date
                                and membership.last_observed_date + 35
    where watermark.campaign_sku_stats_complete
),
latest_raw_sku_stats as (
    -- The ETL contract is upsert-based, but the mart still protects its
    -- public grain from historical duplicate deliveries.
    select distinct on (raw.stat_date, raw.advert_id, raw.nm_id)
        raw.*
    from marketing.wb_marketing_campaign_stats_daily_nms raw
    order by
        raw.stat_date,
        raw.advert_id,
        raw.nm_id,
        raw.last_synced_at desc nulls last,
        raw.id desc
),
latest_raw_campaign_stats as (
    select distinct on (raw.stat_date, raw.advert_id)
        raw.*
    from marketing.wb_marketing_campaign_stats_daily raw
    order by
        raw.stat_date,
        raw.advert_id,
        raw.last_synced_at desc nulls last,
        raw.id desc
),
normalized_ad_stats as (
    select
        spine.report_date,
        spine.advert_id,
        spine.nm_id,
        nullif(raw.nm_name, '') as nm_name,
        membership.subject_name,
        (raw.id is not null) as stats_row_present,
        coalesce(watermark.campaign_sku_stats_complete, false)
            as stats_day_watermark_complete,
        watermark.source_run_id as stats_source_run_id,
        watermark.source_completed_at as stats_source_completed_at,
        case
            when raw.id is not null then raw.views
            when watermark.campaign_sku_stats_complete then 0
        end as views,
        case
            when raw.id is not null then raw.clicks
            when watermark.campaign_sku_stats_complete then 0
        end as clicks,
        case
            when raw.id is not null then raw.orders
            when watermark.campaign_sku_stats_complete then 0
        end as orders,
        case
            when raw.id is not null then raw.shks
            when watermark.campaign_sku_stats_complete then 0
        end as shks,
        case
            when raw.id is not null then raw.sum_spend
            when watermark.campaign_sku_stats_complete then 0::numeric
        end as sum_spend,
        case
            when raw.id is not null then raw.sum_price
            when watermark.campaign_sku_stats_complete then 0::numeric
        end as sum_price,
        case
            when raw.id is not null then raw.sum_spend is not null
            else coalesce(watermark.campaign_sku_stats_complete, false)
        end as ad_spend_complete,
        case
            when raw.id is not null then raw.sum_price is not null
            else coalesce(watermark.campaign_sku_stats_complete, false)
        end as attributed_revenue_complete,
        case
            when raw.id is not null
                then raw.views is not null
                 and raw.clicks is not null
                 and raw.orders is not null
                 and raw.shks is not null
                 and raw.sum_spend is not null
            else coalesce(watermark.campaign_sku_stats_complete, false)
        end as ad_metrics_complete,
        coalesce(raw.last_synced_at, watermark.last_synced_at)
            as stats_last_synced_at
    from date_spine spine
    left join latest_raw_sku_stats raw
      on raw.stat_date = spine.report_date
     and raw.advert_id = spine.advert_id
     and raw.nm_id = spine.nm_id
    left join marketing.wb_marketing_campaign_stats_daily_watermarks watermark
      on watermark.stat_date = spine.report_date
    left join latest_campaign_nms membership
      on membership.advert_id = spine.advert_id
     and membership.nm_id = spine.nm_id
),
normalized_campaign_stats as (
    select
        campaign_spine.report_date,
        campaign_spine.advert_id,
        case
            when raw.id is not null then raw.sum_price
            when watermark.campaign_stats_complete then 0::numeric
        end as sum_price,
        case
            when raw.id is not null then raw.sum_price is not null
            else coalesce(watermark.campaign_stats_complete, false)
        end as attributed_revenue_complete,
        coalesce(raw.last_synced_at, watermark.last_synced_at)
            as campaign_stats_last_synced_at
    from (
        select distinct report_date, advert_id
        from date_spine
    ) campaign_spine
    left join latest_raw_campaign_stats raw
      on raw.stat_date = campaign_spine.report_date
     and raw.advert_id = campaign_spine.advert_id
    left join marketing.wb_marketing_campaign_stats_daily_watermarks watermark
      on watermark.stat_date = campaign_spine.report_date
),
campaign_sku_rollup as (
    select
        s.report_date,
        s.advert_id,
        count(*)::integer as sku_row_count,
        count(*) filter (where not s.attributed_revenue_complete)::integer
            as missing_attributed_revenue_sku_count,
        sum(s.sum_price) as sku_attributed_order_revenue_rub,
        max(s.stats_last_synced_at) as sku_stats_last_synced_at
    from normalized_ad_stats s
    group by s.report_date, s.advert_id
),
sku_ad_rollup as (
    select
        s.report_date,
        s.nm_id,
        count(distinct s.advert_id)::integer as sku_campaign_count,
        count(*) filter (where not s.ad_spend_complete)::integer
            as missing_ad_spend_campaign_count,
        sum(s.sum_spend) as sku_ad_spend_rub
    from normalized_ad_stats s
    group by s.report_date, s.nm_id
)
select
    s.report_date,
    s.advert_id,
    s.nm_id,
    s.nm_name,
    s.subject_name,
    campaign.campaign_name,
    campaign.campaign_status,
    campaign.payment_type,
    campaign.campaign_snapshot_date,
    s.stats_row_present,
    s.stats_day_watermark_complete,
    case
        when s.stats_row_present then 'raw_fullstats_row'
        when s.stats_day_watermark_complete then 'complete_day_absence'
        else 'missing'
    end as stats_evidence_origin,
    s.ad_metrics_complete,
    s.views,
    s.clicks,
    s.orders as ad_orders,
    s.shks as ad_units,
    s.sum_spend as ad_spend_rub,
    sku_ads.sku_campaign_count,
    case
        when sku_ads.missing_ad_spend_campaign_count = 0
            then sku_ads.sku_ad_spend_rub
    end as sku_ad_spend_rub,
    (sku_ads.missing_ad_spend_campaign_count = 0) as sku_ad_spend_complete,
    -- Direct WB attribution. NULL and zero have intentionally different meaning.
    s.sum_price as attributed_order_revenue_rub,
    s.attributed_revenue_complete,
    case
        when s.stats_row_present then 'wb_adv_fullstats.sum_price'
        when s.stats_day_watermark_complete
            then 'derived:complete_day_absence(wb_adv_fullstats)'
    end as attributed_revenue_source,
    campaign_stats.sum_price as campaign_attributed_order_revenue_rub,
    campaign_stats.attributed_revenue_complete
        as campaign_attributed_revenue_complete,
    rollup.sku_row_count,
    rollup.missing_attributed_revenue_sku_count,
    rollup.sku_attributed_order_revenue_rub,
    attribution.campaign_sku_attributed_revenue_delta_rub,
    attribution.campaign_sku_attribution_reconciled,
    stock.stock_on_hand,
    stock.stock_full_for_audit,
    source_state.stock_snapshot_complete,
    case
        when supply_day.snapshot_date is not null
            then coalesce(inbound.supplier_inbound_qty, 0::numeric)
    end as supplier_inbound_qty,
    inbound.supplier_eta_date,
    source_state.supplier_snapshot_complete,
    cost.unit_cogs_rub,
    cost.cogs_snapshot_date,
    finance.sku_net_revenue_rub,
    finance.sku_marketplace_costs_rub,
    finance.sku_cogs_rub,
    finance.sku_contribution_before_ads_rub,
    case
        when finance.finance_complete
         and sku_ads.missing_ad_spend_campaign_count = 0
        then finance.sku_contribution_before_ads_rub - sku_ads.sku_ad_spend_rub
    end as sku_contribution_after_ads_rub,
    (
        coalesce(finance.finance_complete, false)
        and sku_ads.missing_ad_spend_campaign_count = 0
    ) as sku_contribution_after_ads_complete,
    coalesce(finance.finance_complete, false) as finance_complete,
    coalesce(finance.finance_row_count, 0) as finance_row_count,
    case
        when s.attributed_revenue_complete
         and s.sum_price = 0
         and s.orders = 0
         and s.shks = 0
         and s.ad_spend_complete
        then -s.sum_spend
        when s.attributed_revenue_complete
         and finance.finance_complete
         and finance.sku_net_revenue_rub <> 0
         and s.ad_spend_complete
        then round(
            s.sum_price
            * finance.sku_contribution_before_ads_rub
            / finance.sku_net_revenue_rub
            - s.sum_spend,
            2
        )
    end as campaign_contribution_profit_estimated_rub,
    case
        when s.attributed_revenue_complete
         and s.sum_price = 0
         and s.orders = 0
         and s.shks = 0
         and s.ad_spend_complete
        then 'zero_attributed_orders_after_ad_spend_v1'
        when s.attributed_revenue_complete
         and finance.finance_complete
         and finance.sku_net_revenue_rub <> 0
         and s.ad_spend_complete
        then 'attributed_revenue_share_estimate_v1'
    end as contribution_estimation_method,
    (
        s.attributed_revenue_complete
        and s.ad_metrics_complete
        and attribution.campaign_sku_attribution_reconciled is true
        and source_state.stock_snapshot_complete
        and source_state.supplier_snapshot_complete
        and (
            (s.sum_price = 0 and s.orders = 0 and s.shks = 0)
            or coalesce(finance.finance_complete, false)
        )
    ) as decision_input_complete,
    false as causal_valid,
    s.stats_source_run_id,
    s.stats_source_completed_at,
    s.stats_last_synced_at,
    campaign_stats.campaign_stats_last_synced_at,
    rollup.sku_stats_last_synced_at,
    stock.stock_last_synced_at,
    greatest(
        supply_day.supplier_snapshot_last_synced_at,
        inbound.supplier_goods_last_synced_at
    ) as supplier_last_synced_at,
    finance_sync.finance_last_synced_at,
    cost.cogs_retrieved_at,
    -- Presence flags are appended to preserve CREATE OR REPLACE VIEW compatibility.
    source_state.stock_snapshot_present,
    source_state.supplier_snapshot_present,
    false::boolean as finance_snapshot_complete
from normalized_ad_stats s
left join normalized_campaign_stats campaign_stats
  on campaign_stats.report_date = s.report_date
 and campaign_stats.advert_id = s.advert_id
left join campaign_sku_rollup rollup
  on rollup.report_date = s.report_date
 and rollup.advert_id = s.advert_id
left join lateral (
    select
        case
            when campaign_stats.attributed_revenue_complete
             and rollup.missing_attributed_revenue_sku_count = 0
            then round(
                campaign_stats.sum_price
                - rollup.sku_attributed_order_revenue_rub,
                2
            )
        end as campaign_sku_attributed_revenue_delta_rub,
        case
            when campaign_stats.attributed_revenue_complete
             and rollup.missing_attributed_revenue_sku_count = 0
            then abs(
                campaign_stats.sum_price
                - rollup.sku_attributed_order_revenue_rub
            ) <= 0.01
        end as campaign_sku_attribution_reconciled
) attribution on true
left join sku_ad_rollup sku_ads
  on sku_ads.report_date = s.report_date
 and sku_ads.nm_id = s.nm_id
left join lateral (
    select
        c.name as campaign_name,
        c.status as campaign_status,
        c.payment_type,
        c.snapshot_date as campaign_snapshot_date
    from marketing.wb_marketing_campaigns c
    where c.external_id = s.advert_id
      and c.snapshot_date <= s.report_date
    order by c.snapshot_date desc, c.last_synced_at desc, c.id desc
    limit 1
) campaign on true
left join stock_daily stock
  on stock.snapshot_date = s.report_date
 and stock.nm_id = s.nm_id
left join supply_snapshot_days supply_day
  on supply_day.snapshot_date = s.report_date
cross join lateral (
    select
        (stock.snapshot_date is not null) as stock_snapshot_present,
        -- Row presence cannot prove a complete warehouse snapshot.
        false::boolean as stock_snapshot_complete,
        (supply_day.snapshot_date is not null) as supplier_snapshot_present,
        -- Row presence cannot prove a complete supplies extraction.
        false::boolean as supplier_snapshot_complete
) source_state
left join supplier_inbound_daily inbound
  on inbound.snapshot_date = s.report_date
 and inbound.nm_id = s.nm_id
left join finance_daily finance
  on finance.report_date = s.report_date
 and finance.nm_id = s.nm_id
left join finance_sync_daily finance_sync
  on finance_sync.report_date = s.report_date
 and finance_sync.nm_id = s.nm_id
left join lateral (
    select
        c.cost_price_rub as unit_cogs_rub,
        c.upload_date as cogs_snapshot_date,
        c.updated_at as cogs_retrieved_at
    from finance.wb_product_coast c
    where c.nm_id = s.nm_id
      and c.upload_date <= s.report_date
    order by c.upload_date desc, c.updated_at desc
    limit 1
) cost on true;
