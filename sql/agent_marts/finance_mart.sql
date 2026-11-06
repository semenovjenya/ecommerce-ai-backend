create or replace view {{target_schema}}.v_agent_finance_sku_daily as
with report_rows as (
    select
        r.rr_dt as report_date,
        r.nm_id,
        nullif(r.sa_name, '') as vendor_code,
        nullif(r.brand_name, '') as brand_name,
        nullif(r.subject_name, '') as subject_name,
        r.doc_type_name,
        coalesce(r.quantity, 0) as quantity,
        coalesce(r.retail_price_withdisc_rub, 0) as retail_price_withdisc_rub,
        coalesce(r.ppvz_sales_commission, 0) as ppvz_sales_commission,
        coalesce(r.acquiring_fee, 0) as acquiring_fee,
        coalesce(r.delivery_rub, 0) as delivery_rub,
        coalesce(r.storage_fee, 0) as storage_fee,
        coalesce(r.penalty, 0) as penalty,
        coalesce(r.deduction, 0) as deduction,
        coalesce(r.acceptance, 0) as acceptance,
        coalesce(r.rebill_logistic_cost, 0) as rebill_logistic_cost,
        coalesce(r.additional_payment, 0) as additional_payment,
        coalesce(r.ppvz_for_pay, 0) as ppvz_for_pay,
        r.supplier_oper_name
    from finance.wb_report_detail_daily r
    where r.rr_dt is not null
      and (r.nm_id is not null or nullif(r.sa_name, '') is not null)
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
          and c.upload_date <= r.report_date
        order by
            case when r.nm_id is not null and c.nm_id = r.nm_id then 0 else 1 end,
            c.upload_date desc,
            c.updated_at desc
        limit 1
    ) cost on true
),
daily as (
    select
        report_date,
        nm_id,
        vendor_code,
        max(brand_name) as brand_name,
        max(subject_name) as subject_name,
        coalesce(sum(retail_price_withdisc_rub) filter (
            where doc_type_name = 'Продажа'
        ), 0) as gross_revenue,
        abs(coalesce(sum(retail_price_withdisc_rub) filter (
            where doc_type_name = 'Возврат'
        ), 0)) as returns,
        coalesce(sum(ppvz_sales_commission) filter (
            where doc_type_name = 'Продажа'
        ), 0) as commission,
        coalesce(sum(acquiring_fee) filter (
            where doc_type_name = 'Продажа'
        ), 0) as acquiring,
        coalesce(sum(delivery_rub) filter (
            where supplier_oper_name = 'Логистика'
        ), 0) as logistics,
        coalesce(sum(storage_fee) filter (
            where supplier_oper_name = 'Хранение'
        ), 0) as storage,
        coalesce(sum(penalty + deduction) filter (
            where supplier_oper_name is distinct from 'Удержание'
        ), 0) as penalties,
        coalesce(sum(
            case
                when supplier_oper_name = 'Удержание' then deduction
                else 0
            end
            + acceptance
            + rebill_logistic_cost
            + additional_payment
        ), 0) as other_charges,
        coalesce(sum(ppvz_for_pay), 0) as payout_total,
        coalesce(sum(cost_price_rub * quantity) filter (
            where doc_type_name = 'Продажа' and cost_price_rub is not null
        ), 0) as sales_cogs,
        coalesce(sum(cost_price_rub * abs(quantity)) filter (
            where doc_type_name = 'Возврат' and cost_price_rub is not null
        ), 0) as return_cogs,
        coalesce(sum(abs(quantity)) filter (
            where doc_type_name in ('Продажа', 'Возврат') and cost_price_rub is not null
        ), 0) as costed_qty,
        count(*) filter (
            where doc_type_name in ('Продажа', 'Возврат')
              and cost_price_rub is null
              and (quantity <> 0 or retail_price_withdisc_rub <> 0)
        ) as missing_cost_row_count
    from costed_rows
    group by report_date, nm_id, vendor_code
),
calc as (
    select
        *,
        (gross_revenue - returns) as net_revenue,
        (commission + acquiring + logistics + storage + penalties + other_charges) as marketplace_costs_total,
        -- Formula: COGS stays null if any non-empty sale/return row has no cost.
        case
            when missing_cost_row_count > 0 then null
            else (sales_cogs - return_cogs)
        end as cogs_total
    from daily
)
select
    report_date,
    nm_id,
    vendor_code,
    brand_name,
    subject_name,
    gross_revenue,
    returns,
    net_revenue,
    commission,
    acquiring,
    logistics,
    storage,
    penalties,
    other_charges,
    marketplace_costs_total,
    payout_total,
    case
        when missing_cost_row_count = 0 and costed_qty <> 0
            then round((sales_cogs + return_cogs)::numeric / costed_qty, 2)
    end as cogs_per_unit,
    cogs_total,
    -- Formula: gross margin excludes marketplace costs and requires complete COGS.
    case
        when cogs_total is not null then net_revenue - cogs_total
    end as gross_margin,
    case
        when cogs_total is not null and net_revenue <> 0
            then round(100 * (net_revenue - cogs_total)::numeric / net_revenue, 2)
    end as gross_margin_pct,
    -- Formula: contribution margin subtracts marketplace costs and complete COGS.
    case
        when cogs_total is not null
            then net_revenue - marketplace_costs_total - cogs_total
    end as contribution_margin,
    case
        when cogs_total is not null and net_revenue <> 0
            then round(100 * (net_revenue - marketplace_costs_total - cogs_total)::numeric / net_revenue, 2)
    end as contribution_margin_pct
from calc;
