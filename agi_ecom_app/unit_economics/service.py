from __future__ import annotations

import re
from decimal import Decimal
from typing import Any

from agi_ecom_app.services import db
from agi_ecom_app.unit_economics.models import (
    FinanceReportOption,
    FinanceReportOptionsResponse,
    UnitEconomicsColumn,
    UnitEconomicsPeriod,
    UnitEconomicsResponse,
    UnitEconomicsRow,
    UnitEconomicsTotals,
)

UNIT_ECONOMICS_REPORT = FinanceReportOption(
    reportId="unit_economics",
    label="Юнит-экономика",
    sortOrder=1,
)

UNIT_ECONOMICS_PERIOD_LABEL = (
    "Актуальный расчет из finance.wb_profit_calculator_results "
    "+ реклама за 7 дней из finance.wb_report_detail_daily"
)
UNIT_ECONOMICS_SOURCE_TABLE = "finance.wb_profit_calculator_results"

UNIT_ECONOMICS_COLUMNS = [
    UnitEconomicsColumn(key="nmId", label="NM_ID", type="text"),
    UnitEconomicsColumn(key="vendorCode", label="АРТИКУЛ", type="text"),
    UnitEconomicsColumn(key="brandName", label="БРЕНД", type="text"),
    UnitEconomicsColumn(key="subjectName", label="ПРЕДМЕТ", type="text", align="left"),
    UnitEconomicsColumn(key="salePriceRub", label="ЦЕНА ПРОДАЖИ, ₽", type="money"),
    UnitEconomicsColumn(key="unitCostRub", label="СЕБЕСТОИМОСТЬ, ₽/ШТ", type="money"),
    UnitEconomicsColumn(key="commissionPct", label="КОМИССИЯ, %", type="percent"),
    UnitEconomicsColumn(key="commissionRub", label="КОМИССИЯ, ₽", type="money"),
    UnitEconomicsColumn(key="logisticsRub", label="ЛОГИСТИКА, ₽", type="money"),
    UnitEconomicsColumn(key="returnLogisticsRub", label="ВОЗВРАТНАЯ ЛОГИСТИКА, ₽", type="money"),
    UnitEconomicsColumn(key="storageRub", label="ХРАНЕНИЕ, ₽", type="money"),
    UnitEconomicsColumn(key="adRub", label="РЕКЛАМА, ₽", type="money"),
    UnitEconomicsColumn(key="otherExpensesRub", label="ПРОЧИЕ РАСХОДЫ, ₽", type="money"),
    UnitEconomicsColumn(key="taxRub", label="НАЛОГИ, ₽", type="money"),
    UnitEconomicsColumn(key="profitRub", label="ПРИБЫЛЬ, ₽", type="money"),
    UnitEconomicsColumn(key="marginPct", label="МАРЖИНАЛЬНОСТЬ, %", type="percent"),
]


def get_finance_report_options_payload() -> FinanceReportOptionsResponse:
    return FinanceReportOptionsResponse(items=[UNIT_ECONOMICS_REPORT])


def get_unit_economics_payload(
    date_from: str | None = None,
    date_to: str | None = None,
    product_keyword: str | None = None,
    limit: int = 100,
) -> UnitEconomicsResponse:
    if not 1 <= limit <= 500:
        raise ValueError("limit must be between 1 and 500")

    if _normalize_optional_string(date_to) and not _normalize_optional_string(date_from):
        raise ValueError("date_from is required when date_to is provided")

    rows, total_row_count = _fetch_unit_economics_rows(
        date_from=_normalize_optional_string(date_from),
        date_to=_normalize_optional_string(date_to),
        product_keyword=product_keyword,
        limit=limit,
    )

    return UnitEconomicsResponse(
        report=UNIT_ECONOMICS_REPORT,
        period=UnitEconomicsPeriod(
            dateFrom=_normalize_optional_string(date_from),
            dateTo=_normalize_optional_string(date_to),
            label=UNIT_ECONOMICS_PERIOD_LABEL,
            isDefaultPeriod=not _normalize_optional_string(date_from)
            and not _normalize_optional_string(date_to),
        ),
        columns=UNIT_ECONOMICS_COLUMNS,
        rows=[_map_unit_economics_row(row) for row in rows],
        totals=UnitEconomicsTotals(
            totalRowCount=total_row_count,
            returnedRowCount=len(rows),
            sourceTable=UNIT_ECONOMICS_SOURCE_TABLE,
        ),
    )


def _fetch_unit_economics_rows(
    date_from: str | None,
    date_to: str | None,
    product_keyword: str | None,
    limit: int,
) -> tuple[list[dict[str, Any]], int]:
    keyword = _normalize_optional_string(product_keyword)
    keyword_filter = ""
    finance_date_to = date_to or date_from
    params: list[Any] = [date_from, finance_date_to]

    if keyword:
        keyword_filter = """
            where (
                coalesce(p.nm_id::text, '') ilike %s
                or coalesce(p.vendor_code, '') ilike %s
                or coalesce(p.brand_name, '') ilike %s
                or coalesce(p.subject_name, '') ilike %s
                or coalesce(p.scenario_name, '') ilike %s
            )
        """
        like = f"%{keyword}%"
        params.extend([like, like, like, like, like])

    query = f"""
        with report_period as (
            select
                coalesce(%s::date, max(rr_dt::date) - interval '6 days')::date as date_from,
                coalesce(%s::date, max(rr_dt::date))::date as date_to
            from finance.wb_report_detail_daily
        ),
        finance_sales as (
            select
                r.nm_id,
                regexp_replace(coalesce(nullif(r.sa_name, ''), ''), '\s+', '', 'g') as vendor_code_key,
                coalesce(sum(r.quantity) filter (
                    where r.doc_type_name = 'Продажа'
                ), 0) as delivered_qty,
                coalesce(sum(r.retail_price_withdisc_rub) filter (
                    where r.doc_type_name = 'Продажа'
                ), 0) as revenue_sum
            from finance.wb_report_detail_daily r
            cross join report_period rp
            where r.rr_dt::date between rp.date_from and rp.date_to
              and (r.nm_id is not null or nullif(r.sa_name, '') is not null)
            group by
                r.nm_id,
                regexp_replace(coalesce(nullif(r.sa_name, ''), ''), '\s+', '', 'g')
        ),
        ad_deductions as (
            select
                nullif(substring(r.bonus_type_name from 'товар\s+([0-9]+)'), '')::bigint
                    as linked_nm_id,
                coalesce(r.deduction, 0) as deduction
            from finance.wb_report_detail_daily r
            cross join report_period rp
            where r.rr_dt::date between rp.date_from and rp.date_to
              and r.supplier_oper_name = 'Удержание'
              and coalesce(r.deduction, 0) <> 0
              and (
                  r.bonus_type_name ilike '%%WB Продвижение%%'
                  or r.bonus_type_name ilike '%%Списание за отзыв%%'
              )
        ),
        direct_ads as (
            select
                linked_nm_id as nm_id,
                sum(deduction) as direct_marketing_sum
            from ad_deductions
            where linked_nm_id is not null
            group by linked_nm_id
        ),
        ad_totals as (
            select
                coalesce(sum(r.deduction) filter (
                    where linked_nm_id is null
                ), 0) as unlinked_marketing_sum,
                (
                    select coalesce(sum(revenue_sum), 0)
                    from finance_sales
                    where nm_id is not null
                      and nm_id > 0
                ) as total_revenue_sum
            from ad_deductions r
        ),
        finance_ads as (
            select
                fs.nm_id,
                fs.vendor_code_key,
                fs.delivered_qty,
                coalesce(da.direct_marketing_sum, 0)
                + case
                    when totals.total_revenue_sum <> 0
                         and fs.nm_id is not null
                         and fs.nm_id > 0
                        then totals.unlinked_marketing_sum
                            * fs.revenue_sum / totals.total_revenue_sum
                    else 0
                  end as marketing_sum
            from finance_sales fs
            cross join ad_totals totals
            left join direct_ads da on da.nm_id = fs.nm_id
        ),
        filtered as (
            select
                p.scenario_id,
                p.scenario_name,
                p.nm_id,
                p.vendor_code,
                p.brand_name,
                p.subject_name,
                p.price_after_discount_rub,
                coalesce(nullif(p.cost_price_rub, 0), cost.cost_price_rub) as cost_price_rub,
                p.kvv_pct,
                p.kvv_rub,
                p.expected_outbound_per_buyout_rub,
                p.expected_return_logistics_rub,
                p.storage_total_rub,
                case
                    when ads.delivered_qty <> 0
                        then round((ads.marketing_sum / ads.delivered_qty)::numeric, 2)
                    else p.ad_rub
                end as ad_rub,
                p.other_expenses_rub,
                p.tax_rub,
                p.net_profit_rub,
                p.margin_pct,
                p.calculation_status
            from finance.wb_profit_calculator_results p
            left join lateral (
                select c.cost_price_rub
                from finance.wb_product_coast c
                where (
                    (p.nm_id is not null and c.nm_id = p.nm_id)
                    or (
                        nullif(p.vendor_code, '') is not null
                        and regexp_replace(coalesce(c.vendor_code, ''), '\s+', '', 'g')
                            = regexp_replace(p.vendor_code, '\s+', '', 'g')
                    )
                )
                order by
                    case when p.nm_id is not null and c.nm_id = p.nm_id then 0 else 1 end,
                    c.upload_date desc,
                    c.updated_at desc
                limit 1
            ) cost on true
            left join lateral (
                select
                    sum(fa.delivered_qty) as delivered_qty,
                    sum(fa.marketing_sum) as marketing_sum
                from finance_ads fa
                where (
                    p.nm_id is not null
                    and fa.nm_id = p.nm_id
                ) or (
                    p.nm_id is null
                    and nullif(p.vendor_code, '') is not null
                    and fa.vendor_code_key = regexp_replace(p.vendor_code, '\s+', '', 'g')
                )
            ) ads on true
            {keyword_filter}
        )
        select
            *,
            count(*) over () as total_row_count
        from filtered
        order by net_profit_rub desc nulls last, nm_id nulls last, scenario_id desc
        limit %s
    """
    params.append(limit)

    try:
        with db.get_connection() as conn, conn.cursor() as cur:
            cur.execute(query, params)
            rows = db.serialize_rows(cur.fetchall())
    except Exception as exc:
        raise RuntimeError("failed to load finance.wb_profit_calculator_results") from exc

    total_row_count = _int_or_zero(rows[0].get("total_row_count")) if rows else 0
    return rows, total_row_count


def _map_unit_economics_row(row: dict[str, Any]) -> UnitEconomicsRow:
    nm_id = _optional_int(row.get("nm_id"))
    scenario_id = _optional_int(row.get("scenario_id"))
    return UnitEconomicsRow(
        rowId=_row_id(scenario_id=scenario_id, nm_id=nm_id),
        nmId=_sku_string(nm_id),
        vendorCode=_sku_string(row.get("vendor_code")),
        brandName=_optional_string(row.get("brand_name")),
        subjectName=_optional_string(row.get("subject_name")),
        salePriceRub=_optional_number(row.get("price_after_discount_rub")),
        unitCostRub=_optional_number(row.get("cost_price_rub")),
        commissionPct=_optional_number(row.get("kvv_pct")),
        commissionRub=_optional_number(row.get("kvv_rub")),
        logisticsRub=_optional_number(row.get("expected_outbound_per_buyout_rub")),
        returnLogisticsRub=_optional_number(row.get("expected_return_logistics_rub")),
        storageRub=_optional_number(row.get("storage_total_rub")),
        adRub=_optional_number(row.get("ad_rub")),
        otherExpensesRub=_optional_number(row.get("other_expenses_rub")),
        taxRub=_optional_number(row.get("tax_rub")),
        profitRub=_optional_number(row.get("net_profit_rub")),
        marginPct=_optional_number(row.get("margin_pct")),
    )


def _row_id(scenario_id: int | None, nm_id: int | None) -> str:
    scenario_token = str(scenario_id) if scenario_id is not None else "no_scenario"
    nm_token = str(nm_id) if nm_id is not None else "no_nm"
    return f"unit_{_safe_id_token(scenario_token)}_{_safe_id_token(nm_token)}"


def _safe_id_token(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_-]+", "-", value).strip("-") or "empty"


def _normalize_optional_string(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    stripped = str(value).strip()
    return stripped or None


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _sku_string(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return re.sub(r"\s+", "", str(value).strip())


def _optional_number(value: Any) -> int | float | None:
    if value is None or value == "":
        return None
    if isinstance(value, Decimal):
        value = float(value)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else value
    numeric = float(str(value).replace(" ", "").replace(",", "."))
    return int(numeric) if numeric.is_integer() else numeric


def _int_or_zero(value: Any) -> int:
    if value is None or value == "":
        return 0
    return int(value)
