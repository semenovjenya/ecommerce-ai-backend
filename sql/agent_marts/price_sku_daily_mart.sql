-- One row per WB SKU and completed Moscow business day.
--
-- This mart is evidence for pricing simulation, not an elasticity model. It
-- deliberately does not carry prices, stock or finance values across dates and
-- never invents demand for a day whose extraction watermark is incomplete.
-- `shadow_*_usable` means that an observed historical row is usable for model
-- research. `enforced_input_complete` is stricter and remains false until the
-- pricing, stock and finance loaders publish durable completion watermarks.
--
-- Raw revisions are deduplicated on documented WB business keys before the
-- public nm_id x report_date grain is assembled. Rows lacking a source business
-- key are retained (and flagged) instead of being silently collapsed.

create or replace view {{target_schema}}.v_agent_price_sku_daily as
with business_clock as (
    select (now() at time zone 'Europe/Moscow')::date as current_business_date
),
price_ranked as (
    select
        p.*,
        row_number() over (
            partition by p.snapshot_date, p.nm_id
            order by p.last_synced_at desc nulls last, p.id desc
        ) as source_rank,
        count(*) over (
            partition by p.snapshot_date, p.nm_id
        ) as source_revision_count
    from pricing.wb_supplier_prices p
    where p.snapshot_date is not null
      and p.nm_id is not null
),
price_daily as (
    select
        p.snapshot_date as report_date,
        p.nm_id,
        nullif(p.vendor_code, '') as vendor_code,
        nullif(p.currency_iso_code, '') as currency_iso_code,
        p.price as base_price_rub,
        p.discount as seller_discount_pct,
        p.discounted_price as seller_effective_price_rub,
        p.club_discount as club_discount_pct,
        p.club_discounted_price as club_effective_price_rub,
        p.editable_size_price,
        p.is_bad_turnover,
        p.last_synced_at as price_last_synced_at,
        p.source_revision_count::integer as price_raw_revision_count,
        greatest(p.source_revision_count - 1, 0)::integer
            as price_revisions_dropped
    from price_ranked p
    where p.source_rank = 1
),
orders_source as (
    select
        o.*,
        (
            nullif(o.srid, '') is not null
            or nullif(o.g_number, '') is not null
            or nullif(o.sticker, '') is not null
        ) as business_key_present
    from sales.wb_supplier_orders o
    where o.date_ord is not null
      and o.nm_id is not null
),
orders_ranked as (
    select
        o.*,
        row_number() over (
            partition by
                o.business_key_present,
                case when o.business_key_present then o.date_ord end,
                case when o.business_key_present then o.nm_id end,
                case when o.business_key_present then nullif(o.supplier_article, '') end,
                case when o.business_key_present then nullif(o.srid, '') end,
                case when o.business_key_present then nullif(o.g_number, '') end,
                case when o.business_key_present then nullif(o.sticker, '') end,
                case when not o.business_key_present then o.id end
            order by
                o.last_change_date desc nulls last,
                o.last_synced_at desc nulls last,
                o.id desc
        ) as source_rank,
        count(*) over (
            partition by
                o.business_key_present,
                case when o.business_key_present then o.date_ord end,
                case when o.business_key_present then o.nm_id end,
                case when o.business_key_present then nullif(o.supplier_article, '') end,
                case when o.business_key_present then nullif(o.srid, '') end,
                case when o.business_key_present then nullif(o.g_number, '') end,
                case when o.business_key_present then nullif(o.sticker, '') end,
                case when not o.business_key_present then o.id end
        ) as source_revision_count
    from orders_source o
),
orders_daily as (
    select
        (o.date_ord at time zone 'Europe/Moscow')::date as report_date,
        o.nm_id,
        max(nullif(o.supplier_article, '')) as vendor_code,
        max(nullif(o.brand, '')) as brand_name,
        max(nullif(o.subject, '')) as subject_name,
        count(*)::integer as orders_units,
        count(*) filter (where o.is_cancel is true)::integer as cancel_units,
        count(*) filter (where o.is_cancel is not true)::integer
            as non_cancelled_order_units,
        case
            when bool_and(o.price_with_disc is not null)
                then sum(o.price_with_disc)
        end as order_seller_value_rub,
        avg(o.price_with_disc) as observed_order_seller_price_avg_rub,
        avg(o.finished_price) as observed_order_buyer_price_avg_rub,
        avg(o.spp) as observed_order_spp_pct_avg,
        count(o.finished_price)::integer as buyer_price_observation_count,
        bool_and(o.price_with_disc is not null) as order_seller_price_complete,
        count(*) filter (where not o.business_key_present)::integer
            as orders_missing_business_key_count,
        sum(greatest(o.source_revision_count - 1, 0))::integer
            as order_revisions_dropped,
        max(o.last_change_date) as orders_max_last_change_at,
        max(o.last_synced_at) as orders_last_synced_at
    from orders_ranked o
    where o.source_rank = 1
    group by (o.date_ord at time zone 'Europe/Moscow')::date, o.nm_id
),
orders_watermark as (
    select
        max(p.last_change_date_from) as orders_change_watermark_at,
        max(p.updated_at) as orders_watermark_updated_at
    from sales.wb_supplier_orders_progress p
),
sales_source as (
    select
        s.*,
        (
            nullif(s.srid, '') is not null
            or nullif(s.g_number, '') is not null
            or nullif(s.sticker, '') is not null
        ) as business_key_present
    from sales.wb_supplier_sales s
    where s.date_ord is not null
      and s.nm_id is not null
),
sales_ranked as (
    select
        s.*,
        row_number() over (
            partition by
                s.business_key_present,
                case when s.business_key_present then s.date_ord end,
                case when s.business_key_present then s.nm_id end,
                case when s.business_key_present then nullif(s.supplier_article, '') end,
                case when s.business_key_present then nullif(s.srid, '') end,
                case when s.business_key_present then nullif(s.g_number, '') end,
                case when s.business_key_present then nullif(s.sticker, '') end,
                case when not s.business_key_present then s.id end
            order by
                s.last_change_date desc nulls last,
                s.last_synced_at desc nulls last,
                s.id desc
        ) as source_rank,
        count(*) over (
            partition by
                s.business_key_present,
                case when s.business_key_present then s.date_ord end,
                case when s.business_key_present then s.nm_id end,
                case when s.business_key_present then nullif(s.supplier_article, '') end,
                case when s.business_key_present then nullif(s.srid, '') end,
                case when s.business_key_present then nullif(s.g_number, '') end,
                case when s.business_key_present then nullif(s.sticker, '') end,
                case when not s.business_key_present then s.id end
        ) as source_revision_count
    from sales_source s
),
sales_daily as (
    select
        (s.date_ord at time zone 'Europe/Moscow')::date as report_date,
        s.nm_id,
        max(nullif(s.supplier_article, '')) as vendor_code,
        max(nullif(s.brand, '')) as brand_name,
        max(nullif(s.subject, '')) as subject_name,
        count(*) filter (where s.is_realization is true)::integer
            as realized_sales_units,
        case
            when bool_and(s.for_pay is not null) filter (where s.is_realization is true)
                then sum(s.for_pay) filter (where s.is_realization is true)
        end as realized_for_pay_rub,
        avg(s.price_with_disc) filter (where s.is_realization is true)
            as observed_sale_seller_price_avg_rub,
        avg(s.finished_price) filter (where s.is_realization is true)
            as observed_sale_buyer_price_avg_rub,
        avg(s.spp) filter (where s.is_realization is true)
            as observed_sale_spp_pct_avg,
        count(s.finished_price) filter (where s.is_realization is true)::integer
            as sale_buyer_price_observation_count,
        count(*) filter (where not s.business_key_present)::integer
            as sales_missing_business_key_count,
        sum(greatest(s.source_revision_count - 1, 0))::integer
            as sale_revisions_dropped,
        max(s.last_change_date) as sales_max_last_change_at,
        max(s.last_synced_at) as sales_last_synced_at
    from sales_ranked s
    where s.source_rank = 1
    group by (s.date_ord at time zone 'Europe/Moscow')::date, s.nm_id
),
sales_watermark as (
    select
        max(p.last_change_date_from) as sales_change_watermark_at,
        max(p.updated_at) as sales_watermark_updated_at
    from sales.wb_supplier_sales_progress p
),
stock_ranked as (
    select
        s.*,
        row_number() over (
            partition by
                s.snapshot_date,
                s.nm_id,
                nullif(s.supplier_article, ''),
                nullif(s.warehouse_name, ''),
                nullif(s.barcode, '')
            order by
                s.last_change_date desc nulls last,
                s.last_synced_at desc nulls last,
                s.id desc
        ) as source_rank,
        count(*) over (
            partition by
                s.snapshot_date,
                s.nm_id,
                nullif(s.supplier_article, ''),
                nullif(s.warehouse_name, ''),
                nullif(s.barcode, '')
        ) as source_revision_count
    from supply.wb_supplier_stocks s
    where s.snapshot_date is not null
      and s.nm_id is not null
),
stock_daily as (
    select
        s.snapshot_date as report_date,
        s.nm_id,
        max(nullif(s.supplier_article, '')) as vendor_code,
        max(nullif(s.brand, '')) as brand_name,
        max(nullif(s.subject, '')) as subject_name,
        sum(s.quantity)::numeric as stock_on_hand,
        sum(s.quantity_full)::numeric as stock_full_for_audit,
        sum(s.in_way_to_client)::numeric as in_way_to_client,
        sum(s.in_way_from_client)::numeric as in_way_from_client,
        count(distinct nullif(s.warehouse_name, ''))::integer as warehouse_count,
        count(*)::integer as stock_source_row_count,
        sum(greatest(s.source_revision_count - 1, 0))::integer
            as stock_revisions_dropped,
        max(s.last_synced_at) as stock_last_synced_at
    from stock_ranked s
    where s.source_rank = 1
    group by s.snapshot_date, s.nm_id
),
ad_ranked as (
    select
        a.*,
        row_number() over (
            partition by a.stat_date, a.advert_id, a.nm_id
            order by a.last_synced_at desc nulls last, a.id desc
        ) as source_rank,
        count(*) over (
            partition by a.stat_date, a.advert_id, a.nm_id
        ) as source_revision_count
    from marketing.wb_marketing_campaign_stats_daily_nms a
    where a.stat_date is not null
      and a.nm_id is not null
),
ad_daily as (
    select
        a.stat_date as report_date,
        a.nm_id,
        count(*)::integer as ad_source_row_count,
        count(distinct a.advert_id)::integer as ad_campaign_count,
        sum(a.views)::bigint as ad_views,
        sum(a.clicks)::bigint as ad_clicks,
        sum(a.orders)::bigint as ad_orders,
        sum(a.sum_spend) as ad_spend_observed_rub,
        count(*) filter (where a.sum_spend is null)::integer
            as ad_spend_missing_row_count,
        sum(greatest(a.source_revision_count - 1, 0))::integer
            as ad_revisions_dropped,
        max(a.last_synced_at) as ad_last_synced_at
    from ad_ranked a
    where a.source_rank = 1
    group by a.stat_date, a.nm_id
),
finance_keyed_ranked as (
    select
        r.*,
        row_number() over (
            partition by r.rrd_id
            order by r.last_synced_at desc nulls last
        ) as source_rank,
        count(*) over (partition by r.rrd_id) as source_revision_count
    from finance.wb_report_detail_daily r
    where r.rr_dt is not null
      and r.nm_id is not null
      and r.rrd_id is not null
),
finance_deduped as (
    select r.*
    from finance_keyed_ranked r
    where r.source_rank = 1
    union all
    select
        r.*,
        1::bigint as source_rank,
        1::bigint as source_revision_count
    from finance.wb_report_detail_daily r
    where r.rr_dt is not null
      and r.nm_id is not null
      and r.rrd_id is null
),
finance_costed as (
    select
        r.*,
        cost.cost_price_rub,
        cost.upload_date as cogs_snapshot_date,
        cost.updated_at as cogs_retrieved_at
    from finance_deduped r
    left join lateral (
        select
            c.cost_price_rub,
            c.upload_date,
            c.updated_at
        from finance.wb_product_coast c
        where (
            c.nm_id = r.nm_id
            or (
                c.nm_id is null
                and nullif(r.sa_name, '') is not null
                and nullif(c.vendor_code, '') = nullif(r.sa_name, '')
            )
        )
          and c.upload_date <= r.rr_dt
        order by
            case when c.nm_id = r.nm_id then 0 else 1 end,
            c.upload_date desc,
            c.updated_at desc
        limit 1
    ) cost on true
),
finance_aggregated as (
    select
        r.rr_dt as report_date,
        r.nm_id,
        max(nullif(r.sa_name, '')) as vendor_code,
        max(nullif(r.brand_name, '')) as brand_name,
        max(nullif(r.subject_name, '')) as subject_name,
        coalesce(sum(coalesce(r.retail_price_withdisc_rub, 0)) filter (
            where r.doc_type_name = 'Продажа'
        ), 0) as gross_revenue_rub,
        abs(coalesce(sum(coalesce(r.retail_price_withdisc_rub, 0)) filter (
            where r.doc_type_name = 'Возврат'
        ), 0)) as returns_rub,
        coalesce(sum(coalesce(r.ppvz_sales_commission, 0)) filter (
            where r.doc_type_name = 'Продажа'
        ), 0) as commission_rub,
        coalesce(sum(coalesce(r.acquiring_fee, 0)) filter (
            where r.doc_type_name = 'Продажа'
        ), 0) as acquiring_rub,
        coalesce(sum(coalesce(r.delivery_rub, 0)) filter (
            where r.supplier_oper_name = 'Логистика'
        ), 0) as logistics_rub,
        coalesce(sum(coalesce(r.storage_fee, 0)) filter (
            where r.supplier_oper_name = 'Хранение'
        ), 0) as storage_rub,
        coalesce(sum(coalesce(r.penalty, 0) + coalesce(r.deduction, 0)) filter (
            where r.supplier_oper_name is distinct from 'Удержание'
        ), 0) as penalties_rub,
        coalesce(sum(
            case
                when r.supplier_oper_name = 'Удержание'
                    then coalesce(r.deduction, 0)
                else 0
            end
            + coalesce(r.acceptance, 0)
            + coalesce(r.rebill_logistic_cost, 0)
            + coalesce(r.additional_payment, 0)
        ), 0) as other_charges_rub,
        coalesce(sum(coalesce(r.ppvz_for_pay, 0)), 0) as payout_total_rub,
        count(*) filter (where r.ppvz_for_pay is null)::integer
            as payout_missing_row_count,
        coalesce(sum(r.cost_price_rub * r.quantity) filter (
            where r.doc_type_name = 'Продажа'
              and r.cost_price_rub is not null
        ), 0) as sales_cogs_rub,
        coalesce(sum(r.cost_price_rub * abs(r.quantity)) filter (
            where r.doc_type_name = 'Возврат'
              and r.cost_price_rub is not null
        ), 0) as return_cogs_rub,
        count(*) filter (
            where r.doc_type_name in ('Продажа', 'Возврат')
              and r.cost_price_rub is null
              and (r.quantity <> 0 or r.retail_price_withdisc_rub <> 0)
        )::integer as missing_cost_row_count,
        count(*)::integer as finance_source_row_count,
        count(*) filter (where r.rrd_id is null)::integer
            as finance_missing_business_key_count,
        sum(greatest(r.source_revision_count - 1, 0))::integer
            as finance_revisions_dropped,
        max(r.cogs_snapshot_date) as cogs_snapshot_date,
        max(r.cogs_retrieved_at) as cogs_retrieved_at,
        max(r.last_synced_at) as finance_last_synced_at
    from finance_costed r
    group by r.rr_dt, r.nm_id
),
finance_daily as (
    select
        f.*,
        (f.gross_revenue_rub - f.returns_rub) as net_revenue_rub,
        (
            f.commission_rub
            + f.acquiring_rub
            + f.logistics_rub
            + f.storage_rub
            + f.penalties_rub
            + f.other_charges_rub
        ) as marketplace_costs_rub,
        case
            when f.missing_cost_row_count = 0
                then f.sales_cogs_rub - f.return_cogs_rub
        end as cogs_total_rub
    from finance_aggregated f
),
source_keys as (
    select p.report_date, p.nm_id from price_daily p
    union
    select o.report_date, o.nm_id from orders_daily o
    union
    select s.report_date, s.nm_id from sales_daily s
    union
    select s.report_date, s.nm_id from stock_daily s
    union
    select a.report_date, a.nm_id from ad_daily a
    union
    select f.report_date, f.nm_id from finance_daily f
),
completed_keys as (
    select distinct k.report_date, k.nm_id
    from source_keys k
    cross join business_clock clock
    where k.report_date < clock.current_business_date
),
joined as (
    select
        k.report_date,
        k.nm_id,
        coalesce(
            p.vendor_code,
            o.vendor_code,
            s.vendor_code,
            stock.vendor_code,
            f.vendor_code
        ) as vendor_code,
        coalesce(o.brand_name, s.brand_name, stock.brand_name, f.brand_name)
            as brand_name,
        coalesce(o.subject_name, s.subject_name, stock.subject_name, f.subject_name)
            as subject_name,
        'Europe/Moscow'::text as business_timezone,
        true as business_day_complete,

        p.base_price_rub,
        p.seller_discount_pct,
        p.seller_effective_price_rub,
        p.club_discount_pct,
        p.club_effective_price_rub,
        p.currency_iso_code,
        p.editable_size_price,
        p.is_bad_turnover,
        (p.report_date is not null) as price_snapshot_present,
        (p.report_date is not null) as price_snapshot_exact_day,
        (
            p.report_date is not null
            and p.base_price_rub > 0
            and p.seller_effective_price_rub > 0
        ) as price_values_complete,
        false::boolean as price_snapshot_complete,
        p.price_raw_revision_count,
        p.price_revisions_dropped,
        p.price_last_synced_at,

        coalesce(
            o.observed_order_seller_price_avg_rub,
            s.observed_sale_seller_price_avg_rub
        ) as observed_transaction_seller_price_avg_rub,
        coalesce(
            o.observed_order_buyer_price_avg_rub,
            s.observed_sale_buyer_price_avg_rub
        ) as observed_buyer_price_avg_rub,
        coalesce(
            o.observed_order_spp_pct_avg,
            s.observed_sale_spp_pct_avg
        ) as observed_spp_pct_avg,
        case
            when o.buyer_price_observation_count > 0
                then 'sales.wb_supplier_orders.finished_price'
            when s.sale_buyer_price_observation_count > 0
                then 'sales.wb_supplier_sales.finished_price'
        end as observed_buyer_price_source,
        case
            when o.buyer_price_observation_count > 0
                then o.buyer_price_observation_count
            when s.sale_buyer_price_observation_count > 0
                then s.sale_buyer_price_observation_count
            else 0
        end::integer as observed_buyer_price_count,

        (o.report_date is not null) as orders_source_row_present,
        case
            when o.report_date is not null then 'raw_order_rows'
            when k.report_date < (
                orders_w.orders_change_watermark_at at time zone 'Europe/Moscow'
            )::date then 'complete_day_absence'
            else 'missing'
        end as orders_evidence_origin,
        (
            orders_w.orders_change_watermark_at is not null
            and k.report_date < (
                orders_w.orders_change_watermark_at at time zone 'Europe/Moscow'
            )::date
        ) as orders_day_complete,
        case
            when o.report_date is not null then o.orders_units
            when k.report_date < (
                orders_w.orders_change_watermark_at at time zone 'Europe/Moscow'
            )::date then 0
        end as orders_units,
        case
            when o.report_date is not null then o.cancel_units
            when k.report_date < (
                orders_w.orders_change_watermark_at at time zone 'Europe/Moscow'
            )::date then 0
        end as cancel_units,
        case
            when o.report_date is not null then o.non_cancelled_order_units
            when k.report_date < (
                orders_w.orders_change_watermark_at at time zone 'Europe/Moscow'
            )::date then 0
        end as non_cancelled_order_units,
        o.order_seller_value_rub,
        o.order_seller_price_complete,
        o.orders_missing_business_key_count,
        o.order_revisions_dropped,
        orders_w.orders_change_watermark_at,
        orders_w.orders_watermark_updated_at,
        o.orders_max_last_change_at,
        o.orders_last_synced_at,

        (s.report_date is not null) as sales_source_row_present,
        case
            when s.report_date is not null then 'raw_sale_rows'
            when k.report_date < (
                sales_w.sales_change_watermark_at at time zone 'Europe/Moscow'
            )::date then 'complete_day_absence'
            else 'missing'
        end as sales_evidence_origin,
        (
            sales_w.sales_change_watermark_at is not null
            and k.report_date < (
                sales_w.sales_change_watermark_at at time zone 'Europe/Moscow'
            )::date
        ) as sales_day_complete,
        case
            when s.report_date is not null then s.realized_sales_units
            when k.report_date < (
                sales_w.sales_change_watermark_at at time zone 'Europe/Moscow'
            )::date then 0
        end as realized_sales_units,
        s.realized_for_pay_rub,
        s.sales_missing_business_key_count,
        s.sale_revisions_dropped,
        sales_w.sales_change_watermark_at,
        sales_w.sales_watermark_updated_at,
        s.sales_max_last_change_at,
        s.sales_last_synced_at,

        (stock.report_date is not null) as stock_snapshot_present,
        false::boolean as stock_snapshot_complete,
        stock.stock_on_hand,
        stock.stock_full_for_audit,
        stock.in_way_to_client,
        stock.in_way_from_client,
        stock.warehouse_count,
        stock.stock_source_row_count,
        stock.stock_revisions_dropped,
        stock.stock_last_synced_at,
        case
            when stock.report_date is not null and stock.stock_on_hand = 0
                then true
            when stock.report_date is not null then false
        end as observed_out_of_stock,

        (a.report_date is not null) as marketing_source_row_present,
        coalesce(w.campaign_sku_stats_complete, false)
            as marketing_day_complete,
        a.ad_spend_observed_rub as ad_spend_observed_partial_rub,
        a.ad_views as ad_views_observed_partial,
        a.ad_clicks as ad_clicks_observed_partial,
        a.ad_orders as ad_orders_observed_partial,
        case
            when coalesce(w.campaign_sku_stats_complete, false)
             and coalesce(a.ad_spend_missing_row_count, 0) = 0
                then coalesce(a.ad_spend_observed_rub, 0::numeric)
        end as ad_spend_rub,
        case
            when coalesce(w.campaign_sku_stats_complete, false)
                then coalesce(a.ad_views, 0)
        end as ad_views,
        case
            when coalesce(w.campaign_sku_stats_complete, false)
                then coalesce(a.ad_clicks, 0)
        end as ad_clicks,
        case
            when coalesce(w.campaign_sku_stats_complete, false)
                then coalesce(a.ad_orders, 0)
        end as ad_orders,
        (
            coalesce(w.campaign_sku_stats_complete, false)
            and coalesce(a.ad_spend_missing_row_count, 0) = 0
        ) as ad_spend_complete,
        case
            when a.report_date is not null then 'raw_fullstats_rows'
            when coalesce(w.campaign_sku_stats_complete, false)
                then 'complete_day_absence'
            else 'missing'
        end as marketing_evidence_origin,
        a.ad_campaign_count,
        a.ad_source_row_count,
        a.ad_revisions_dropped,
        w.source_run_id as marketing_source_run_id,
        w.source_completed_at as marketing_source_completed_at,
        greatest(a.ad_last_synced_at, w.last_synced_at)
            as marketing_last_synced_at,

        (f.report_date is not null) as finance_row_present,
        (
            f.report_date is not null
            and f.missing_cost_row_count = 0
            and f.cogs_total_rub is not null
            and f.payout_missing_row_count = 0
        ) as finance_values_complete,
        false::boolean as finance_snapshot_complete,
        f.gross_revenue_rub,
        f.returns_rub,
        f.net_revenue_rub,
        f.marketplace_costs_rub,
        f.cogs_total_rub,
        case
            when f.missing_cost_row_count = 0
             and f.payout_missing_row_count = 0
                then f.payout_total_rub - greatest(f.cogs_total_rub, 0::numeric)
        end as contribution_before_ads_rub,
        f.payout_total_rub,
        f.missing_cost_row_count,
        f.finance_source_row_count,
        f.finance_missing_business_key_count,
        f.finance_revisions_dropped,
        f.cogs_snapshot_date,
        f.cogs_retrieved_at,
        f.finance_last_synced_at,

        'pricing.wb_supplier_prices'::text as price_source_table,
        'sales.wb_supplier_orders'::text as orders_source_table,
        'sales.wb_supplier_sales'::text as sales_source_table,
        'supply.wb_supplier_stocks'::text as stock_source_table,
        'marketing.wb_marketing_campaign_stats_daily_nms'::text
            as marketing_source_table,
        'finance.wb_report_detail_daily'::text as finance_source_table,
        'finance.wb_product_coast'::text as cogs_source_table
    from completed_keys k
    left join price_daily p
      on p.report_date = k.report_date
     and p.nm_id = k.nm_id
    left join orders_daily o
      on o.report_date = k.report_date
     and o.nm_id = k.nm_id
    cross join orders_watermark orders_w
    left join sales_daily s
      on s.report_date = k.report_date
     and s.nm_id = k.nm_id
    cross join sales_watermark sales_w
    left join stock_daily stock
      on stock.report_date = k.report_date
     and stock.nm_id = k.nm_id
    left join ad_daily a
      on a.report_date = k.report_date
     and a.nm_id = k.nm_id
    left join marketing.wb_marketing_campaign_stats_daily_watermarks w
      on w.stat_date = k.report_date
    left join finance_daily f
      on f.report_date = k.report_date
     and f.nm_id = k.nm_id
)
select
    j.*,
    (
        j.business_day_complete
        and j.price_values_complete
        and j.orders_day_complete
        and j.stock_snapshot_present
    ) as shadow_demand_observation_usable,
    (
        j.business_day_complete
        and j.price_values_complete
        and j.orders_day_complete
        and j.stock_snapshot_present
        and j.finance_values_complete
        and j.ad_spend_complete
    ) as shadow_profit_observation_usable,
    case
        when j.finance_values_complete
         and j.realized_sales_units > 0
            then round(
                j.contribution_before_ads_rub / j.realized_sales_units,
                4
            )
    end as contribution_before_ads_per_realized_unit_rub,
    -- Fullstats spend is intentionally retained as evidence but is not
    -- subtracted from ppvz_for_pay: the payout can already contain WB
    -- deductions and the campaign mart is not yet reconciled against them.
    -- Until that reconciliation exists, do not publish a false after-ads
    -- contribution metric.
    null::numeric as contribution_after_ads_rub,
    null::numeric as contribution_after_ads_per_realized_unit_rub,
    (
        j.business_day_complete
        and j.price_snapshot_complete
        and j.orders_day_complete
        and j.sales_day_complete
        and j.stock_snapshot_complete
        and j.marketing_day_complete
        and j.finance_snapshot_complete
        and j.price_values_complete
        and j.finance_values_complete
        and j.ad_spend_complete
    ) as enforced_input_complete,
    false::boolean as causal_valid,
    -- Append-only pricing-finance contract. Keeping these fields after every
    -- pre-existing view column lets CREATE OR REPLACE preserve the durable
    -- column order used by running consumers.
    j.payout_total_rub as pricing_payout_before_cogs_rub,
    case
        when j.finance_values_complete
            then greatest(j.cogs_total_rub, 0::numeric)
    end as pricing_cogs_charged_rub,
    j.contribution_before_ads_rub as pricing_contribution_rub,
    fd.payout_missing_row_count,
    -- Fee fields retain their source signs for audit only. WB report rows
    -- contain both positive and negative commission values, so they must not
    -- be recombined and subtracted to derive pricing contribution.
    fd.commission_rub as commission_signed_raw_rub,
    fd.acquiring_rub as acquiring_signed_raw_rub,
    fd.logistics_rub as logistics_signed_raw_rub,
    fd.storage_rub as storage_signed_raw_rub,
    fd.penalties_rub as penalties_signed_raw_rub,
    fd.other_charges_rub as other_charges_signed_raw_rub,
    j.marketplace_costs_rub as marketplace_costs_signed_raw_rub,
    'finance.wb_report_detail_daily.ppvz_for_pay'::text
        as pricing_contribution_basis,
    false::boolean as partial_ad_spend_subtracted_from_pricing_contribution,
    false::boolean as contribution_after_ads_complete
from joined j
left join finance_daily fd
  on fd.report_date = j.report_date
 and fd.nm_id = j.nm_id;
