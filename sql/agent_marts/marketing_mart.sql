create or replace view {{target_schema}}.v_agent_marketing_sku_daily as
with latest_campaigns as (
    select distinct on (c.external_id)
        c.external_id as advert_id,
        c.status,
        nullif(c.brand, '') as campaign_brand_name
    from marketing.wb_marketing_campaigns c
    order by c.external_id, c.snapshot_date desc, c.last_synced_at desc, c.id desc
),
campaign_nms_latest as (
    select distinct on (n.advert_id, n.nm_id)
        n.advert_id,
        n.nm_id,
        nullif(n.subject_name, '') as subject_name
    from marketing.wb_marketing_campaign_nms n
    order by n.advert_id, n.nm_id, n.snapshot_date desc, n.last_synced_at desc, n.id desc
),
ad_daily as (
    select
        s.stat_date as report_date,
        s.nm_id,
        max(lc.campaign_brand_name) as brand_name,
        max(cn.subject_name) as subject_name,
        coalesce(sum(coalesce(s.sum_spend, 0)), 0) as ad_spend,
        coalesce(sum(coalesce(s.views, 0)), 0) as views,
        coalesce(sum(coalesce(s.clicks, 0)), 0) as clicks,
        coalesce(sum(coalesce(s.orders, 0)), 0) as ad_orders,
        count(distinct s.advert_id) as campaign_count,
        count(distinct s.advert_id) filter (where lc.status = 9) as active_campaign_count
    from marketing.wb_marketing_campaign_stats_daily_nms s
    left join latest_campaigns lc
        on lc.advert_id = s.advert_id
    left join campaign_nms_latest cn
        on cn.advert_id = s.advert_id
       and cn.nm_id = s.nm_id
    group by s.stat_date, s.nm_id
),
visibility_daily as (
    select distinct on (d.current_period_end, d.nm_id)
        d.current_period_end as report_date,
        d.nm_id,
        nullif(d.vendor_code, '') as vendor_code,
        nullif(d.brand_name, '') as brand_name,
        nullif(d.subject_name, '') as subject_name,
        d.visibility_current,
        d.visibility_dynamics as visibility_delta,
        d.avg_position_current,
        d.avg_position_dynamics as avg_position_delta,
        d.open_card_current,
        d.add_to_cart_current,
        d.orders_current
    from marketing.wb_search_query_details d
    where d.current_period_end is not null
    order by d.current_period_end, d.nm_id, d.snapshot_hour desc, d.last_synced_at desc, d.id desc
),
search_text_daily as (
    select
        t.dt as report_date,
        t.nm_id,
        coalesce(sum(coalesce(t.frequency, 0)), 0) as search_frequency,
        count(distinct nullif(t.text, '')) as search_query_count
    from marketing.wb_search_query_orders_text_daily t
    group by t.dt, t.nm_id
),
sku_identity as (
    select distinct on (nm_id)
        nm_id,
        vendor_code,
        brand_name,
        subject_name
    from (
        select
            d.nm_id,
            nullif(d.vendor_code, '') as vendor_code,
            nullif(d.brand_name, '') as brand_name,
            nullif(d.subject_name, '') as subject_name,
            d.current_period_end as identity_date,
            0 as source_priority
        from marketing.wb_search_query_details d
        where d.nm_id is not null
        union all
        select
            f.nm_id,
            nullif(f.vendor_code, '') as vendor_code,
            nullif(f.brand_name, '') as brand_name,
            nullif(f.subject_name, '') as subject_name,
            f.report_date as identity_date,
            1 as source_priority
        from sales.wb_sales_funnel_products_daily f
        where f.nm_id is not null
    ) identities
    order by nm_id, source_priority, identity_date desc nulls last
),
mart_keys as (
    select report_date, nm_id from ad_daily
    union
    select report_date, nm_id from visibility_daily
    union
    select report_date, nm_id from search_text_daily
)
select
    k.report_date,
    k.nm_id,
    coalesce(v.vendor_code, i.vendor_code) as vendor_code,
    coalesce(v.brand_name, a.brand_name, i.brand_name) as brand_name,
    coalesce(v.subject_name, a.subject_name, i.subject_name) as subject_name,
    coalesce(a.ad_spend, 0) as ad_spend,
    coalesce(a.views, 0) as views,
    coalesce(a.clicks, 0) as clicks,
    -- Formula: ad CTR is recomputed from clicks divided by ad views.
    case
        when coalesce(a.views, 0) <> 0
            then round(100 * a.clicks::numeric / a.views, 2)
    end as ctr,
    -- Formula: CPC is ad spend divided by clicks.
    case
        when coalesce(a.clicks, 0) <> 0
            then round(a.ad_spend::numeric / a.clicks, 2)
    end as cpc,
    coalesce(a.ad_orders, 0) as ad_orders,
    -- Formula: ad conversion rate is ad orders divided by clicks.
    case
        when coalesce(a.clicks, 0) <> 0
            then round(100 * a.ad_orders::numeric / a.clicks, 2)
    end as ad_cr,
    -- Formula: cost per ad order is ad spend divided by ad-attributed orders.
    case
        when coalesce(a.ad_orders, 0) <> 0
            then round(a.ad_spend::numeric / a.ad_orders, 2)
    end as cost_per_ad_order,
    coalesce(a.campaign_count, 0) as campaign_count,
    coalesce(a.active_campaign_count, 0) as active_campaign_count,
    v.visibility_current,
    v.visibility_delta,
    v.avg_position_current,
    v.avg_position_delta,
    v.open_card_current,
    v.add_to_cart_current,
    v.orders_current,
    coalesce(st.search_frequency, 0) as search_frequency,
    coalesce(st.search_query_count, 0) as search_query_count
from mart_keys k
left join ad_daily a
    on a.report_date = k.report_date
   and a.nm_id is not distinct from k.nm_id
left join visibility_daily v
    on v.report_date = k.report_date
   and v.nm_id is not distinct from k.nm_id
left join search_text_daily st
    on st.report_date = k.report_date
   and st.nm_id is not distinct from k.nm_id
left join sku_identity i
    on i.nm_id is not distinct from k.nm_id;
