create or replace view marketing.v_agent_marketing_campaign_daily as
with latest_campaigns as (
    select distinct on (external_id)
        external_id as advert_id,
        snapshot_date as campaign_snapshot_date,
        snapshot_hour as campaign_snapshot_hour,
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
    from marketing.v_wb_marketing_campaigns_latest_daily
    order by external_id, snapshot_date desc, snapshot_hour desc, id desc
),
latest_budget as (
    select distinct on (advert_id)
        advert_id,
        snapshot_date as budget_snapshot_date,
        snapshot_hour as budget_snapshot_hour,
        cash,
        netting,
        total as budget_total,
        last_synced_at as budget_last_synced_at
    from marketing.v_wb_marketing_budget_latest_daily
    order by advert_id, snapshot_date desc, snapshot_hour desc, id desc
),
latest_nms as (
    select
        advert_id,
        count(distinct nm_id) as active_nms_count,
        (array_agg(subject_name order by nm_id))[1] as top_subject_name,
        max(last_synced_at) as nms_last_synced_at
    from marketing.v_wb_marketing_campaign_nms_latest_daily
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
    c.campaign_snapshot_hour,
    b.budget_snapshot_date,
    b.budget_snapshot_hour,
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
    on n.advert_id = s.advert_id;


drop view if exists marketing.v_agent_marketing_query_daily;
drop view if exists marketing.v_agent_marketing_search_daily;

create or replace view marketing.v_agent_marketing_search_text_daily as
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
    on d.nm_id = t.nm_id;


create or replace view marketing.v_agent_marketing_search_sku_daily as
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
    on o.snapshot_date = d.snapshot_date;
