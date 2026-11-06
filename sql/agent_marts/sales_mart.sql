create or replace view {{target_schema}}.v_agent_sales_sku_daily as
with sales_daily as (
    select
        s.date_ord::date as report_date,
        s.nm_id,
        nullif(s.supplier_article, '') as vendor_code,
        max(nullif(s.brand, '')) as brand_name,
        max(nullif(s.subject, '')) as subject_name,
        count(*) filter (where s.is_realization) as sales_units,
        coalesce(sum(coalesce(s.for_pay, 0)) filter (where s.is_realization), 0) as sales_revenue,
        avg(nullif(s.price_with_disc, 0)) filter (where s.is_realization) as avg_sales_price
    from sales.wb_supplier_sales s
    group by s.date_ord::date, s.nm_id, nullif(s.supplier_article, '')
),
orders_daily as (
    select
        o.date_ord::date as report_date,
        o.nm_id,
        nullif(o.supplier_article, '') as vendor_code,
        max(nullif(o.brand, '')) as brand_name,
        max(nullif(o.subject, '')) as subject_name,
        count(*) as orders_units,
        coalesce(sum(coalesce(o.price_with_disc, 0)), 0) as orders_sum,
        count(*) filter (where o.is_cancel) as cancel_units,
        avg(nullif(o.price_with_disc, 0)) as avg_order_price
    from sales.wb_supplier_orders o
    group by o.date_ord::date, o.nm_id, nullif(o.supplier_article, '')
),
funnel_daily as (
    select
        f.report_date,
        f.nm_id,
        nullif(f.vendor_code, '') as vendor_code,
        max(nullif(f.brand_name, '')) as brand_name,
        max(nullif(f.subject_name, '')) as subject_name,
        coalesce(sum(coalesce(f.open_count, 0)), 0) as open_count,
        coalesce(sum(coalesce(f.cart_count, 0)), 0) as cart_count,
        coalesce(sum(coalesce(f.order_count, 0)), 0) as order_count,
        coalesce(sum(coalesce(f.order_sum, 0)), 0) as funnel_order_sum,
        coalesce(sum(coalesce(f.buyout_count, 0)), 0) as buyout_units,
        avg(nullif(f.avg_price, 0)) as funnel_avg_price
    from sales.wb_sales_funnel_products_daily f
    group by f.report_date, f.nm_id, nullif(f.vendor_code, '')
),
mart_keys as (
    select report_date, nm_id, vendor_code from sales_daily
    union
    select report_date, nm_id, vendor_code from orders_daily
    union
    select report_date, nm_id, vendor_code from funnel_daily
),
daily as (
    select
        k.report_date,
        k.nm_id,
        k.vendor_code,
        coalesce(sd.brand_name, od.brand_name, fd.brand_name) as brand_name,
        coalesce(sd.subject_name, od.subject_name, fd.subject_name) as subject_name,
        coalesce(sd.sales_units, 0) as sales_units,
        coalesce(sd.sales_revenue, 0) as sales_revenue,
        coalesce(od.orders_units, 0) as orders_units,
        coalesce(od.orders_sum, 0) as orders_sum,
        coalesce(od.cancel_units, 0) as cancel_units,
        coalesce(fd.buyout_units, 0) as buyout_units,
        coalesce(fd.buyout_units, 0) as buyout_count,
        coalesce(fd.open_count, 0) as open_count,
        coalesce(fd.cart_count, 0) as cart_count,
        coalesce(fd.order_count, 0) as order_count,
        -- Formula: open-to-cart conversion is recomputed from summed funnel counts.
        case
            when coalesce(fd.open_count, 0) <> 0
                then round(100 * fd.cart_count::numeric / fd.open_count, 2)
        end as open_to_cart_conversion,
        -- Formula: cart-to-order conversion is recomputed from summed funnel counts.
        case
            when coalesce(fd.cart_count, 0) <> 0
                then round(100 * fd.order_count::numeric / fd.cart_count, 2)
        end as cart_to_order_conversion,
        -- Formula: buyout percent is recomputed from buyouts divided by funnel orders.
        case
            when coalesce(fd.order_count, 0) <> 0
                then round(100 * fd.buyout_units::numeric / fd.order_count, 2)
        end as buyout_percent,
        coalesce(
            fd.funnel_avg_price,
            sd.avg_sales_price,
            od.avg_order_price
        ) as avg_price
    from mart_keys k
    left join sales_daily sd
        on sd.report_date = k.report_date
       and sd.nm_id is not distinct from k.nm_id
       and sd.vendor_code is not distinct from k.vendor_code
    left join orders_daily od
        on od.report_date = k.report_date
       and od.nm_id is not distinct from k.nm_id
       and od.vendor_code is not distinct from k.vendor_code
    left join funnel_daily fd
        on fd.report_date = k.report_date
       and fd.nm_id is not distinct from k.nm_id
       and fd.vendor_code is not distinct from k.vendor_code
)
select
    report_date,
    nm_id,
    vendor_code,
    brand_name,
    subject_name,
    sales_units,
    sales_revenue,
    orders_units,
    orders_sum,
    cancel_units,
    buyout_units,
    open_count,
    cart_count,
    order_count,
    buyout_count,
    open_to_cart_conversion,
    cart_to_order_conversion,
    buyout_percent,
    avg_price,
    -- Formula: rolling 7 calendar days including the current report_date.
    sum(coalesce(sales_units, 0)) over (
        partition by nm_id, vendor_code
        order by report_date
        range between interval '6 days' preceding and current row
    ) as sales_7d_units,
    sum(coalesce(sales_revenue, 0)) over (
        partition by nm_id, vendor_code
        order by report_date
        range between interval '6 days' preceding and current row
    ) as revenue_7d,
    -- Formula: rolling 14 calendar days including the current report_date.
    sum(coalesce(sales_units, 0)) over (
        partition by nm_id, vendor_code
        order by report_date
        range between interval '13 days' preceding and current row
    ) as sales_14d_units,
    sum(coalesce(sales_revenue, 0)) over (
        partition by nm_id, vendor_code
        order by report_date
        range between interval '13 days' preceding and current row
    ) as revenue_14d
from daily;
