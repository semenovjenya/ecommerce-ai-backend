create or replace view {{target_schema}}.v_agent_supply_sku_coverage_daily as
with stock_by_warehouse as (
    select
        s.snapshot_date,
        s.nm_id,
        nullif(s.supplier_article, '') as vendor_code,
        max(nullif(s.brand, '')) as brand_name,
        max(nullif(s.subject, '')) as subject_name,
        nullif(s.warehouse_name, '') as warehouse_name,
        coalesce(sum(coalesce(s.quantity, 0)), 0) as stock_qty,
        coalesce(sum(coalesce(s.quantity_full, 0)), 0) as stock_qty_full,
        coalesce(sum(coalesce(s.in_way_to_client, 0)), 0) as in_way_to_client,
        coalesce(sum(coalesce(s.in_way_from_client, 0)), 0) as in_way_from_client
    from supply.wb_supplier_stocks s
    group by
        s.snapshot_date,
        s.nm_id,
        nullif(s.supplier_article, ''),
        nullif(s.warehouse_name, '')
),
stock_total as (
    select
        snapshot_date,
        nm_id,
        vendor_code,
        max(brand_name) as brand_name,
        max(subject_name) as subject_name,
        sum(stock_qty) as stock_qty,
        sum(stock_qty_full) as stock_qty_full,
        sum(in_way_to_client) as in_way_to_client,
        sum(in_way_from_client) as in_way_from_client,
        count(*) filter (where warehouse_name is not null) as warehouse_count
    from stock_by_warehouse
    group by snapshot_date, nm_id, vendor_code
),
top_warehouse as (
    select
        snapshot_date,
        nm_id,
        vendor_code,
        warehouse_name as top_warehouse_name,
        stock_qty as top_warehouse_stock_qty
    from (
        select
            stock_by_warehouse.*,
            row_number() over (
                partition by snapshot_date, nm_id, vendor_code
                order by stock_qty desc, warehouse_name nulls last
            ) as rn
        from stock_by_warehouse
    ) ranked
    where rn = 1
),
sales_windows as (
    select
        st.snapshot_date,
        st.nm_id,
        st.vendor_code,
        count(ss.id) filter (
            where ss.is_realization
              and ss.date_ord::date = st.snapshot_date
        ) as sales_1d_units,
        count(ss.id) filter (
            where ss.is_realization
              and ss.date_ord::date between st.snapshot_date - interval '6 days' and st.snapshot_date
        ) as sales_7d_units,
        count(ss.id) filter (
            where ss.is_realization
              and ss.date_ord::date between st.snapshot_date - interval '13 days' and st.snapshot_date
        ) as sales_14d_units
    from stock_total st
    left join sales.wb_supplier_sales ss
        on ss.nm_id is not distinct from st.nm_id
       and nullif(ss.supplier_article, '') is not distinct from st.vendor_code
       and ss.date_ord::date between st.snapshot_date - interval '13 days' and st.snapshot_date
    group by st.snapshot_date, st.nm_id, st.vendor_code
),
calc as (
    select
        st.snapshot_date,
        st.nm_id,
        st.vendor_code,
        st.brand_name,
        st.subject_name,
        st.stock_qty,
        st.stock_qty_full,
        coalesce(sw.sales_1d_units, 0) as sales_1d_units,
        coalesce(sw.sales_7d_units, 0) as sales_7d_units,
        round(coalesce(sw.sales_7d_units, 0)::numeric / 7.0, 4) as avg_daily_sales_7d,
        coalesce(sw.sales_14d_units, 0) as sales_14d_units,
        round(coalesce(sw.sales_14d_units, 0)::numeric / 14.0, 4) as avg_daily_sales_14d,
        -- Formula: coverage uses sellable stock only. stock_qty_full can include
        -- goods already in transit to clients and must not inflate availability.
        case
            when coalesce(sw.sales_7d_units, 0) > 0
                then round(st.stock_qty::numeric / (sw.sales_7d_units::numeric / 7.0), 2)
        end as days_of_stock,
        st.in_way_to_client,
        st.in_way_from_client,
        st.warehouse_count,
        tw.top_warehouse_name,
        coalesce(tw.top_warehouse_stock_qty, 0) as top_warehouse_stock_qty,
        -- Formula: concentration share is a percent of sellable stock in the largest warehouse.
        case
            when coalesce(st.stock_qty, 0) <> 0
                then round(100 * coalesce(tw.top_warehouse_stock_qty, 0)::numeric / st.stock_qty, 2)
        end as top_warehouse_share
    from stock_total st
    left join sales_windows sw
        on sw.snapshot_date = st.snapshot_date
       and sw.nm_id is not distinct from st.nm_id
       and sw.vendor_code is not distinct from st.vendor_code
    left join top_warehouse tw
        on tw.snapshot_date = st.snapshot_date
       and tw.nm_id is not distinct from st.nm_id
       and tw.vendor_code is not distinct from st.vendor_code
)
select
    snapshot_date,
    nm_id,
    vendor_code,
    brand_name,
    subject_name,
    stock_qty,
    stock_qty_full,
    sales_1d_units,
    sales_7d_units,
    avg_daily_sales_7d,
    sales_14d_units,
    avg_daily_sales_14d,
    days_of_stock,
    -- Formula: reorder flag is true for stockout, critical, or warning coverage.
    (
        (sales_7d_units > 0 and days_of_stock < 21)
        or (sales_7d_units = 0 and coalesce(stock_qty, 0) = 0)
    ) as reorder_flag,
    case
        when sales_7d_units > 0 and days_of_stock < 7
            then greatest(ceil(avg_daily_sales_7d * 56 - stock_qty), 0)::bigint
        when sales_7d_units > 0 and days_of_stock < 21
            then greatest(ceil(avg_daily_sales_7d * 21 - stock_qty), 0)::bigint
        else 0::bigint
    end as reorder_qty_recommended,
    (
        coalesce(stock_qty, 0) = 0
        or (sales_7d_units > 0 and days_of_stock < 7)
    ) as out_of_stock_risk_flag,
    (sales_14d_units = 0 and coalesce(stock_qty, 0) > 0) as zero_sales_with_stock_flag,
    (sales_7d_units = 0 and coalesce(stock_qty, 0) >= 60) as overstock_flag,
    in_way_to_client,
    in_way_from_client,
    warehouse_count,
    top_warehouse_name,
    top_warehouse_stock_qty,
    top_warehouse_share,
    (coalesce(top_warehouse_share, 0) >= 80) as stock_concentration_flag
from calc;
