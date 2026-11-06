-- Stable, compact price contract for LLM agents.
--
-- Keep source tables unchanged: several pipelines already use ambiguous legacy
-- names such as source_price_promo for different price layers.  This view gives
-- every layer an explicit business meaning and exposes one latest row per SKU.
create or replace view analytics_ai.v_agent_price_context_latest as
with latest_seller as (
    select distinct on (p.nm_id)
        p.nm_id,
        p.vendor_code,
        p.snapshot_date as seller_snapshot_date,
        p.price as seller_price_regular,
        p.discounted_price as seller_price_report_promo,
        p.discounted_price as seller_controlled_price,
        p.discount as seller_discount_percent,
        p.club_discount as seller_club_discount_percent,
        p.club_discounted_price as seller_club_discounted_price,
        p.last_synced_at as seller_last_synced_at
    from pricing.wb_supplier_prices p
    where p.nm_id is not null
    order by
        p.nm_id,
        p.snapshot_date desc,
        p.last_synced_at desc nulls last,
        p.id desc
),
latest_analysis as (
    select distinct on (a.source_nm_id)
        a.id as analysis_id,
        a.source_nm_id as nm_id,
        a.run_date as analysis_run_date,
        a.source_snapshot_date,
        a.source_vendor_code,
        a.source_brand,
        a.source_subject_name,
        a.source_title,
        a.source_product_url,
        a.source_price_site_display,
        a.source_price_site_wallet,
        a.source_price_site_effective,
        a.source_platform_discount_percent,
        a.source_wallet_discount_percent,
        a.source_report_to_site_discount_percent,
        a.competitor_selection_mode,
        a.competitor_count,
        a.competitor_price_min,
        a.competitor_price_median,
        a.competitor_price_max,
        a.strict_competitor_count,
        a.strict_competitor_price_min,
        a.strict_competitor_price_median,
        a.strict_competitor_price_max,
        a.market_candidate_count,
        a.market_price_min,
        a.market_price_median,
        a.market_price_max,
        a.llm_checked_count,
        a.llm_kept_count,
        a.vision_checked_count,
        a.vision_absolute_count,
        a.vision_close_count,
        a.vision_similar_count,
        a.vision_not_analog_count,
        a.used_market_fallback,
        a.is_price_outlier,
        a.analysis_status,
        a.last_synced_at as analysis_last_synced_at
    from analytics_ai.wb_price_ai_analysis a
    where a.source_nm_id is not null
    order by
        a.source_nm_id,
        a.run_date desc,
        a.last_synced_at desc nulls last,
        a.id desc
),
latest_context as (
    select distinct on (c.source_nm_id)
        c.source_nm_id as nm_id,
        c.snapshot_date as context_snapshot_date,
        c.source_price_site_display,
        c.source_price_site_wallet,
        c.source_price_site_effective,
        c.source_platform_discount_percent,
        c.source_wallet_discount_percent,
        c.source_report_to_site_discount_percent,
        c.query_count,
        c.source_found_query_count,
        c.source_found_query_share,
        c.weighted_organic_position,
        c.market_price_min,
        c.market_price_median,
        c.market_price_max,
        c.organic_competitors_before_source,
        c.price_before_median,
        c.source_to_before_median_price_ratio
    from analytics_ai.mart_price_context_daily c
    where c.source_nm_id is not null
    order by c.source_nm_id, c.snapshot_date desc
),
sku_keys as (
    select nm_id from latest_seller
    union
    select nm_id from latest_analysis
    union
    select nm_id from latest_context
),
joined as (
    select
        k.nm_id,
        coalesce(s.vendor_code, a.source_vendor_code) as vendor_code,
        a.source_brand as brand,
        a.source_subject_name as subject_name,
        a.source_title as title,
        a.source_product_url as product_url,
        s.seller_snapshot_date,
        a.analysis_run_date,
        a.source_snapshot_date,
        c.context_snapshot_date,
        greatest(
            s.seller_snapshot_date,
            a.analysis_run_date,
            c.context_snapshot_date
        ) as context_as_of_date,
        s.seller_price_regular,
        s.seller_price_report_promo,
        s.seller_controlled_price,
        s.seller_discount_percent,
        s.seller_club_discount_percent,
        s.seller_club_discounted_price,
        coalesce(
            a.source_price_site_display,
            c.source_price_site_display
        ) as customer_price_display,
        coalesce(
            a.source_price_site_wallet,
            c.source_price_site_wallet
        ) as customer_price_wallet,
        coalesce(
            a.source_price_site_effective,
            c.source_price_site_effective,
            s.seller_controlled_price
        ) as customer_price_effective,
        coalesce(
            a.source_platform_discount_percent,
            c.source_platform_discount_percent
        ) as platform_discount_percent,
        coalesce(
            a.source_wallet_discount_percent,
            c.source_wallet_discount_percent
        ) as wallet_discount_percent,
        coalesce(
            a.source_report_to_site_discount_percent,
            c.source_report_to_site_discount_percent
        ) as report_to_customer_discount_percent,
        a.competitor_selection_mode,
        a.competitor_count as broad_competitor_count,
        a.competitor_price_min as broad_competitor_price_min,
        coalesce(
            a.market_price_median,
            a.competitor_price_median,
            c.market_price_median
        ) as broad_market_price_median,
        a.competitor_price_max as broad_competitor_price_max,
        a.strict_competitor_count,
        a.strict_competitor_price_min,
        a.strict_competitor_price_median,
        a.strict_competitor_price_max,
        a.market_candidate_count,
        a.market_price_min,
        a.market_price_max,
        a.llm_checked_count,
        a.llm_kept_count,
        a.vision_checked_count,
        a.vision_absolute_count,
        a.vision_close_count,
        a.vision_similar_count,
        a.vision_not_analog_count,
        a.used_market_fallback,
        a.is_price_outlier,
        a.analysis_status,
        c.query_count,
        c.source_found_query_count,
        c.source_found_query_share,
        c.weighted_organic_position,
        c.organic_competitors_before_source,
        c.price_before_median,
        c.source_to_before_median_price_ratio,
        a.analysis_id,
        s.seller_last_synced_at,
        a.analysis_last_synced_at
    from sku_keys k
    left join latest_seller s on s.nm_id = k.nm_id
    left join latest_analysis a on a.nm_id = k.nm_id
    left join latest_context c on c.nm_id = k.nm_id
),
classified as (
    select
        j.*,
        case
            when j.customer_price_wallet is not null then 'wb_wallet'
            when j.customer_price_display is not null then 'site_display'
            when j.seller_controlled_price is not null then 'seller_report'
            else 'missing'
        end as customer_price_basis,
        case
            when j.analysis_run_date is null then 'missing'
            when coalesce(j.analysis_status, '') <> 'success' then 'insufficient'
            when coalesce(j.strict_competitor_count, 0) < 5 then 'insufficient'
            when j.strict_competitor_price_median is null then 'insufficient'
            else 'trusted'
        end as competitor_quality,
        case
            when j.seller_snapshot_date is null
              or j.analysis_run_date is null
              or j.context_snapshot_date is null
                then 'incomplete'
            when current_date - j.seller_snapshot_date > 1
              or current_date - j.analysis_run_date > 2
              or current_date - j.context_snapshot_date > 2
                then 'stale'
            else 'fresh'
        end as freshness_status
    from joined j
)
select
    c.*,
    current_date - c.seller_snapshot_date as seller_age_days,
    current_date - c.analysis_run_date as analysis_age_days,
    current_date - c.context_snapshot_date as context_age_days,
    case
        when c.competitor_quality <> 'trusted'
            then 'insufficient_competitor_quality'
        when c.customer_price_effective is null
            then 'missing_customer_price'
        when c.customer_price_effective >= c.strict_competitor_price_median * 1.15
            then 'above_market'
        when c.customer_price_effective <= c.strict_competitor_price_median * 0.85
            then 'below_market'
        else 'market_aligned'
    end as price_status,
    case
        when c.strict_competitor_price_median > 0
         and c.customer_price_effective is not null
            then round(
                (
                    c.customer_price_effective
                    - c.strict_competitor_price_median
                ) * 100::numeric
                / c.strict_competitor_price_median,
                2
            )
    end as customer_vs_strict_median_gap_percent,
    'price-context-v1'::text as contract_version
from classified c;
