from __future__ import annotations

from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any

from agi_ecom_app.services import db
from agi_ecom_app.commission_calculator.models import (
    CommissionCalculatorColumn,
    CommissionCalculatorPeriod,
    CommissionCalculatorReport,
    CommissionCalculatorResponse,
    CommissionCalculatorRow,
    CommissionCalculatorTotals,
)

COMMISSION_CALCULATOR_REPORT = CommissionCalculatorReport(
    reportId="commission_calculator",
    label="Калькулятор комиссий",
    sortOrder=1,
)

COMMISSION_CALCULATOR_COLUMNS = [
    CommissionCalculatorColumn(key="brand", label="Бренд", type="text"),
    CommissionCalculatorColumn(key="subject", label="Предмет", type="text"),
    CommissionCalculatorColumn(key="productName", label="Наименование", type="text"),
    CommissionCalculatorColumn(key="vendorArticle", label="Артикул продавца", type="text"),
    CommissionCalculatorColumn(key="article", label="Артикул", type="integer"),
    CommissionCalculatorColumn(key="basePriceRub", label="Базовая цена", type="money"),
    CommissionCalculatorColumn(key="discountPct", label="Скидка", type="percent"),
    CommissionCalculatorColumn(
        key="wbSiteDiscountPct",
        label="Скидка с сайта ВБ",
        type="percent",
    ),
    CommissionCalculatorColumn(key="checkStatus", label="Проверка", type="text"),
    CommissionCalculatorColumn(
        key="salePriceWithoutSppRub",
        label="Цена продажи (без СПП)",
        type="money",
    ),
    CommissionCalculatorColumn(key="wbClubDiscountPct", label="ВБ Клуб", type="percent"),
    CommissionCalculatorColumn(
        key="priceWithWbClubRub",
        label="Цена с ВБ клуб",
        type="money",
    ),
    CommissionCalculatorColumn(key="sppPct", label="Размер СПП", type="percent"),
    CommissionCalculatorColumn(key="priceWithSppRub", label="Цена с СПП", type="money"),
    CommissionCalculatorColumn(key="commissionPct", label="Комиссия,%", type="percent"),
    CommissionCalculatorColumn(key="acquiringPct", label="Эквайринг", type="percent"),
    CommissionCalculatorColumn(
        key="totalCommissionPct",
        label="Общая комиссия (вкл эквайринг)",
        type="percent",
    ),
    CommissionCalculatorColumn(key="commissionRub", label="Комиссия,руб.", type="money"),
    CommissionCalculatorColumn(
        key="logisticsTariffRub",
        label="Логистика, тариф",
        type="money",
    ),
    CommissionCalculatorColumn(key="buyoutPct", label="% выкупа", type="percent"),
    CommissionCalculatorColumn(key="logisticsRub", label="Логистика, в руб.", type="money"),
    CommissionCalculatorColumn(
        key="logisticsSalePct",
        label="Логистика, % от продажи",
        type="percent",
    ),
    CommissionCalculatorColumn(
        key="unitCostRub",
        label="Закупочная цена/себестоимость",
        type="money",
    ),
    CommissionCalculatorColumn(key="defectRub", label="Брак, руб", type="money"),
    CommissionCalculatorColumn(
        key="totalCostRub",
        label="ИТОГОВАЯ себестоимость в рублях",
        type="money",
    ),
    CommissionCalculatorColumn(
        key="costSalePct",
        label="Себестоимость, % от продаж",
        type="percent",
    ),
    CommissionCalculatorColumn(
        key="storageRub",
        label="Хранение, в рублях",
        type="money",
    ),
    CommissionCalculatorColumn(
        key="storageSalePct",
        label="Хранение, % от продаж",
        type="percent",
    ),
    CommissionCalculatorColumn(
        key="advertisingRub",
        label="Реклама в рублях",
        type="money",
    ),
    CommissionCalculatorColumn(
        key="advertisingSalePct",
        label="Реклама, % от продаж",
        type="percent",
    ),
    CommissionCalculatorColumn(
        key="taxRub",
        label="Налоги, в рублях",
        type="money",
    ),
    CommissionCalculatorColumn(
        key="taxSalePct",
        label="Налоги, % от продажи",
        type="percent",
    ),
    CommissionCalculatorColumn(
        key="acceptanceRub",
        label="Платная приемка, в рублях",
        type="money",
    ),
    CommissionCalculatorColumn(
        key="acceptanceSalePct",
        label="Платная приемка, % от продаж",
        type="percent",
    ),
    CommissionCalculatorColumn(
        key="profitRub",
        label="Прибыль в рублях",
        type="money",
    ),
    CommissionCalculatorColumn(
        key="marginSalePct",
        label="Маржинальность, прибыль % от продажи",
        type="percent",
    ),
    CommissionCalculatorColumn(
        key="profitCostPct",
        label="Рентабельность, прибыль % от сс",
        type="percent",
    ),
    CommissionCalculatorColumn(
        key="wbPayoutRub",
        label="Получаем на счет от ВБ, р",
        type="money",
    ),
]


class CommissionCalculatorDataError(RuntimeError):
    """Raised when source finance data cannot be read for the calculator."""


def get_commission_calculator_payload(
    nm_id: int | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> CommissionCalculatorResponse:
    if nm_id is not None and nm_id <= 0:
        raise ValueError("nm_id must be a positive integer")

    resolved_date_from, resolved_date_to, is_default_period = _resolve_period(
        date_from=date_from,
        date_to=date_to,
    )

    if nm_id is not None:
        rows, has_price_data, has_finance_data, has_cost_data = _build_calculator_rows(
            nm_ids=[nm_id],
            date_from=resolved_date_from,
            date_to=resolved_date_to,
        )

        if not rows:
            raise LookupError(f"nm_id={nm_id} was not found in calculator sources")
    else:
        rows, has_price_data, has_finance_data, has_cost_data = _build_calculator_rows(
            nm_ids=_fetch_catalog_nm_ids(
                date_from=resolved_date_from,
                date_to=resolved_date_to,
            ),
            date_from=resolved_date_from,
            date_to=resolved_date_to,
        )

    return CommissionCalculatorResponse(
        report=COMMISSION_CALCULATOR_REPORT,
        period=CommissionCalculatorPeriod(
            dateFrom=resolved_date_from,
            dateTo=resolved_date_to,
            isDefaultPeriod=is_default_period,
        ),
        columns=COMMISSION_CALCULATOR_COLUMNS,
        rows=rows,
        totals=CommissionCalculatorTotals(
            requestedNmId=nm_id,
            returnedRowCount=len(rows),
            hasPriceData=has_price_data,
            hasFinanceData=has_finance_data,
            hasCostData=has_cost_data,
        ),
    )


def _build_calculator_rows(
    nm_ids: list[int],
    date_from: str,
    date_to: str,
) -> tuple[list[CommissionCalculatorRow], bool, bool, bool]:
    rows: list[CommissionCalculatorRow] = []
    has_price_data = False
    has_finance_data = False
    has_cost_data = False

    for nm_id in nm_ids:
        context = _fetch_sku_context(nm_id=nm_id, report_date=date_to)
        finance = _fetch_sku_finance_metrics(
            nm_id=nm_id,
            date_from=date_from,
            date_to=date_to,
        )

        if context is None and finance is None:
            continue

        rows.append(
            _build_row(
                nm_id=nm_id,
                context=context or {},
                finance=finance or {},
            )
        )
        has_price_data = has_price_data or bool(
            (context or {}).get("price_before_discount") is not None
            or (context or {}).get("price_after_discount") is not None
            or (context or {}).get("price_with_spp") is not None
            or (finance or {}).get("base_price_rub") is not None
            or (finance or {}).get("sale_price_without_spp_rub") is not None
            or (finance or {}).get("price_with_spp_rub") is not None
        )
        has_finance_data = has_finance_data or bool(finance)
        has_cost_data = has_cost_data or (context or {}).get("cost_price_rub") is not None

    return rows, has_price_data, has_finance_data, has_cost_data


def _fetch_catalog_nm_ids(date_from: str, date_to: str) -> list[int]:
    query = """
        with finance_sku as (
            select
                nm_id,
                coalesce(sum(retail_price_withdisc_rub) filter (
                    where doc_type_name = 'Продажа'
                ), 0) as revenue_sum
            from finance.wb_report_detail_daily
            where rr_dt between %s and %s
              and nm_id is not null
              and nm_id > 0
            group by nm_id
        ),
        catalog_sku as (
            select distinct nm_id
            from finance.wb_product_coast
            where nm_id is not null
              and nm_id > 0
        ),
        sku as (
            select nm_id from catalog_sku
            union
            select nm_id from finance_sku
        )
        select sku.nm_id
        from sku
        left join finance_sku on finance_sku.nm_id = sku.nm_id
        order by coalesce(finance_sku.revenue_sum, 0) desc, sku.nm_id
    """
    try:
        with db.get_connection() as conn, conn.cursor() as cur:
            cur.execute(query, [date_from, date_to])
            rows = cur.fetchall()
    except Exception as exc:
        raise CommissionCalculatorDataError("failed to load calculator catalog") from exc

    return [
        int(row["nm_id"])
        for row in rows
        if row.get("nm_id") is not None
    ]


def _resolve_period(
    date_from: str | None,
    date_to: str | None,
) -> tuple[str, str, bool]:
    normalized_date_from = _normalize_optional_string(date_from)
    normalized_date_to = _normalize_optional_string(date_to)
    if normalized_date_from:
        return normalized_date_from, normalized_date_to or normalized_date_from, False
    if normalized_date_to:
        raise ValueError("date_from is required when date_to is provided")

    latest_date = _get_latest_finance_report_date()
    start_date = latest_date - timedelta(days=6)
    return start_date.isoformat(), latest_date.isoformat(), True


def _get_latest_finance_report_date() -> date:
    try:
        with db.get_connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                select max(rr_dt::date) as latest_rr_date
                from finance.wb_report_detail_daily
                """
            )
            row = cur.fetchone()
    except Exception as exc:
        raise CommissionCalculatorDataError(
            "failed to resolve latest finance report date"
        ) from exc

    latest_date = _coerce_date(row.get("latest_rr_date") if row else None)
    if latest_date is None:
        raise RuntimeError("finance.wb_report_detail_daily does not contain report dates")
    return latest_date


def _fetch_sku_context(nm_id: int, report_date: str) -> dict[str, Any] | None:
    onepager_query = """
        select
            product_name,
            price_before_discount,
            price_after_discount,
            discount_pct,
            spp_pct,
            price_with_spp
        from reports.v_1pager_ai
        where sku = %s
          and report_date <= %s
        order by report_date desc
        limit 1
    """
    supplier_price_query = """
        select
            vendor_code,
            price,
            discounted_price,
            discount,
            club_discount,
            club_discounted_price
        from pricing.wb_supplier_prices
        where nm_id = %s
          and snapshot_date <= %s
        order by snapshot_date desc, last_synced_at desc nulls last
        limit 1
    """
    cost_query = """
        select
            vendor_code,
            product_name,
            brand_name,
            cost_price_rub
        from finance.wb_product_coast
        where nm_id = %s
        order by upload_date desc, updated_at desc
        limit 1
    """
    funnel_query = """
        select
            vendor_code,
            brand_name,
            subject_name,
            title,
            buyout_percent
        from sales.wb_sales_funnel_products_daily
        where nm_id = %s
          and report_date <= %s
        order by report_date desc
        limit 1
    """
    calculator_query = """
        select
            vendor_code,
            brand_name,
            subject_name,
            price_before_wb_discount,
            price_after_discount_rub,
            buyout_pct,
            kvv_pct,
            kvv_rub,
            expected_outbound_per_buyout_rub,
            outbound_dispatch_rub,
            storage_total_rub,
            ad_rub,
            tax_percent,
            tax_rub,
            net_profit_rub,
            margin_pct,
            cost_price_rub
        from finance.wb_profit_calculator_results
        where nm_id = %s
        order by scenario_id desc
        limit 1
    """

    try:
        with db.get_connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                select
                    to_regclass('reports.v_1pager_ai') as onepager_relation,
                    to_regclass('pricing.wb_supplier_prices') as supplier_price_relation,
                    to_regclass('finance.wb_profit_calculator_results') as calculator_relation
                """
            )
            relation_row = cur.fetchone() or {}
            if relation_row.get("onepager_relation"):
                cur.execute(onepager_query, [str(nm_id), report_date])
                onepager_row = db.serialize_value(cur.fetchone() or {})
            else:
                onepager_row = {}

            if relation_row.get("supplier_price_relation"):
                cur.execute(supplier_price_query, [nm_id, report_date])
                supplier_price_row = db.serialize_value(cur.fetchone() or {})
            else:
                supplier_price_row = {}

            cur.execute(cost_query, [nm_id])
            cost_row = db.serialize_value(cur.fetchone() or {})

            cur.execute(funnel_query, [nm_id, report_date])
            funnel_row = db.serialize_value(cur.fetchone() or {})

            if relation_row.get("calculator_relation"):
                cur.execute(calculator_query, [nm_id])
                calculator_row = db.serialize_value(cur.fetchone() or {})
            else:
                calculator_row = {}
    except Exception as exc:
        raise CommissionCalculatorDataError("failed to load SKU context") from exc

    context = {
        "product_name": _first_present(
            onepager_row.get("product_name"),
            cost_row.get("product_name"),
            funnel_row.get("title"),
        ),
        "vendor_code": _first_present(
            funnel_row.get("vendor_code"),
            cost_row.get("vendor_code"),
            supplier_price_row.get("vendor_code"),
            calculator_row.get("vendor_code"),
        ),
        "brand_name": _first_present(
            funnel_row.get("brand_name"),
            cost_row.get("brand_name"),
            calculator_row.get("brand_name"),
        ),
        "subject_name": _first_present(
            funnel_row.get("subject_name"),
            calculator_row.get("subject_name"),
        ),
        "price_before_discount": _first_money(
            supplier_price_row.get("price"),
            calculator_row.get("price_before_wb_discount"),
            onepager_row.get("price_before_discount"),
        ),
        "price_after_discount": _first_money(
            supplier_price_row.get("discounted_price"),
            calculator_row.get("price_after_discount_rub"),
            onepager_row.get("price_after_discount"),
        ),
        "seller_discount_pct": _optional_number(supplier_price_row.get("discount")),
        "discount_pct": _optional_number(onepager_row.get("discount_pct")),
        "club_discount_pct": _optional_number(supplier_price_row.get("club_discount")),
        "price_with_wb_club": _optional_number(supplier_price_row.get("club_discounted_price")),
        "spp_pct": _optional_number(onepager_row.get("spp_pct")),
        "price_with_spp": _optional_number(onepager_row.get("price_with_spp")),
        "buyout_percent": _first_money(
            funnel_row.get("buyout_percent"),
            calculator_row.get("buyout_pct"),
        ),
        "cost_price_rub": _first_money(
            cost_row.get("cost_price_rub"),
            calculator_row.get("cost_price_rub"),
        ),
        "calculator_commission_pct": _optional_number(calculator_row.get("kvv_pct")),
        "calculator_commission_rub": _optional_number(calculator_row.get("kvv_rub")),
        "calculator_logistics_rub": _first_money(
            calculator_row.get("expected_outbound_per_buyout_rub"),
            calculator_row.get("outbound_dispatch_rub"),
        ),
        "calculator_storage_rub": _optional_number(calculator_row.get("storage_total_rub")),
        "calculator_advertising_rub": _optional_number(calculator_row.get("ad_rub")),
        "calculator_tax_pct": _optional_number(calculator_row.get("tax_percent")),
        "calculator_tax_rub": _optional_number(calculator_row.get("tax_rub")),
    }
    if not any(value is not None for value in context.values()):
        return None
    return context


def _fetch_sku_finance_metrics(
    nm_id: int,
    date_from: str,
    date_to: str,
) -> dict[str, Any] | None:
    query = """
        select
            max(nullif(sa_name, '')) as vendor_code,
            max(nullif(brand_name, '')) as brand_name,
            max(nullif(subject_name, '')) as subject_name,
            coalesce(sum(quantity) filter (
                where doc_type_name = 'Продажа'
            ), 0) as delivered_qty,
            coalesce(sum(retail_price_withdisc_rub) filter (
                where doc_type_name = 'Продажа'
            ), 0) as revenue_sum,
            coalesce(sum(ppvz_sales_commission) filter (
                where doc_type_name = 'Продажа'
            ), 0) as commission_sum,
            coalesce(sum(acquiring_fee) filter (
                where doc_type_name = 'Продажа'
            ), 0) as acquiring_sum,
            coalesce(sum(delivery_rub) filter (
                where supplier_oper_name = 'Логистика'
            ), 0) as logistics_sum,
            coalesce(sum(storage_fee) filter (
                where supplier_oper_name = 'Хранение'
            ), 0) as storage_sum,
            coalesce(sum(acceptance), 0) as acceptance_sum,
            coalesce(sum(deduction) filter (
                where supplier_oper_name = 'Удержание'
            ), 0) as advertising_sum,
            coalesce(sum(penalty), 0) as penalty_sum,
            round((
                sum(retail_price * quantity) filter (
                    where doc_type_name = 'Продажа'
                      and quantity > 0
                )
                / nullif(sum(quantity) filter (
                    where doc_type_name = 'Продажа'
                      and quantity > 0
                ), 0)
            )::numeric, 2) as base_price_rub,
            round((
                sum(retail_price_withdisc_rub) filter (
                    where doc_type_name = 'Продажа'
                      and quantity > 0
                )
                / nullif(sum(quantity) filter (
                    where doc_type_name = 'Продажа'
                      and quantity > 0
                ), 0)
            )::numeric, 2) as sale_price_without_spp_rub,
            round((
                sum(retail_amount) filter (
                    where doc_type_name = 'Продажа'
                      and quantity > 0
                )
                / nullif(sum(quantity) filter (
                    where doc_type_name = 'Продажа'
                      and quantity > 0
                ), 0)
            )::numeric, 2) as price_with_spp_rub,
            round((avg(sale_percent) filter (
                where doc_type_name = 'Продажа'
                  and quantity > 0
            ))::numeric, 2) as discount_pct,
            round((avg(product_discount_for_report) filter (
                where doc_type_name = 'Продажа'
                  and quantity > 0
            ))::numeric, 2) as wb_site_discount_pct,
            round((avg(ppvz_spp_prc) filter (
                where doc_type_name = 'Продажа'
                  and quantity > 0
            ))::numeric, 2) as spp_pct
        from finance.wb_report_detail_daily
        where rr_dt between %s and %s
          and nm_id = %s
    """

    try:
        with db.get_connection() as conn, conn.cursor() as cur:
            cur.execute(query, [date_from, date_to, nm_id])
            row = db.serialize_value(cur.fetchone() or {})
    except Exception as exc:
        raise CommissionCalculatorDataError("failed to load SKU finance metrics") from exc

    has_finance_values = any(
        _optional_number(row.get(key)) not in (None, 0)
        for key in (
            "delivered_qty",
            "revenue_sum",
            "commission_sum",
            "acquiring_sum",
            "logistics_sum",
            "storage_sum",
            "acceptance_sum",
            "advertising_sum",
            "penalty_sum",
            "base_price_rub",
            "sale_price_without_spp_rub",
            "price_with_spp_rub",
            "discount_pct",
            "wb_site_discount_pct",
            "spp_pct",
        )
    )
    has_identity = any(
        _normalize_optional_string(row.get(key)) is not None
        for key in ("vendor_code", "brand_name", "subject_name")
    )
    if not has_finance_values and not has_identity:
        return None

    return {
        "vendor_code": _normalize_optional_string(row.get("vendor_code")),
        "brand_name": _normalize_optional_string(row.get("brand_name")),
        "subject_name": _normalize_optional_string(row.get("subject_name")),
        "delivered_qty": _optional_number(row.get("delivered_qty")),
        "revenue_sum": _optional_number(row.get("revenue_sum")),
        "commission_sum": _optional_number(row.get("commission_sum")),
        "acquiring_sum": _optional_number(row.get("acquiring_sum")),
        "logistics_sum": _optional_number(row.get("logistics_sum")),
        "storage_sum": _optional_number(row.get("storage_sum")),
        "acceptance_sum": _optional_number(row.get("acceptance_sum")),
        "advertising_sum": _optional_number(row.get("advertising_sum")),
        "penalty_sum": _optional_number(row.get("penalty_sum")),
        "base_price_rub": _optional_number(row.get("base_price_rub")),
        "sale_price_without_spp_rub": _optional_number(row.get("sale_price_without_spp_rub")),
        "price_with_spp_rub": _optional_number(row.get("price_with_spp_rub")),
        "discount_pct": _optional_number(row.get("discount_pct")),
        "wb_site_discount_pct": _optional_number(row.get("wb_site_discount_pct")),
        "spp_pct": _optional_number(row.get("spp_pct")),
    }


def _build_row(
    *,
    nm_id: int,
    context: dict[str, Any],
    finance: dict[str, Any],
) -> CommissionCalculatorRow:
    vendor_article = _first_present(
        finance.get("vendor_code"),
        context.get("vendor_code"),
    )
    brand = _first_present(
        finance.get("brand_name"),
        context.get("brand_name"),
    )
    subject = _first_present(
        finance.get("subject_name"),
        context.get("subject_name"),
    )
    base_price = _first_money(
        context.get("price_before_discount"),
        finance.get("base_price_rub"),
    )
    sale_price = _first_money(
        context.get("price_after_discount"),
        finance.get("sale_price_without_spp_rub"),
        _per_unit(finance.get("revenue_sum"), finance.get("delivered_qty")),
    )
    price_with_spp = _first_money(
        context.get("price_with_spp"),
        finance.get("price_with_spp_rub"),
    )
    actual_commission_pct = _percent(
        finance.get("commission_sum"),
        finance.get("revenue_sum"),
    )
    acquiring_pct = _percent(finance.get("acquiring_sum"), finance.get("revenue_sum"))
    actual_total_commission_pct = _percent(
        _sum_numbers(finance.get("commission_sum"), finance.get("acquiring_sum")),
        finance.get("revenue_sum"),
    )
    total_commission_pct = _first_money(
        actual_total_commission_pct,
        context.get("calculator_commission_pct"),
    )
    total_commission_rub = _first_money(
        _value_from_percent(sale_price, actual_total_commission_pct),
        _per_unit(
            _sum_numbers(finance.get("commission_sum"), finance.get("acquiring_sum")),
            finance.get("delivered_qty"),
        ),
        context.get("calculator_commission_rub"),
        _value_from_percent(sale_price, context.get("calculator_commission_pct")),
    )
    logistics_rub = _first_money(
        _per_unit(finance.get("logistics_sum"), finance.get("delivered_qty")),
        context.get("calculator_logistics_rub"),
    )
    storage_rub = _first_money(
        _per_unit(finance.get("storage_sum"), finance.get("delivered_qty")),
        context.get("calculator_storage_rub"),
    )
    advertising_rub = _first_money(
        _per_unit(finance.get("advertising_sum"), finance.get("delivered_qty")),
        context.get("calculator_advertising_rub"),
        0,
    )
    acceptance_rub = _first_money(
        _per_unit(finance.get("acceptance_sum"), finance.get("delivered_qty")),
        0,
    )
    defect_rub = _first_money(
        _per_unit(finance.get("penalty_sum"), finance.get("delivered_qty")),
        0,
    )
    unit_cost_rub = _optional_number(context.get("cost_price_rub"))
    total_cost_rub = _sum_numbers(unit_cost_rub, defect_rub)
    tax_rub = _first_money(
        context.get("calculator_tax_rub"),
        _value_from_percent(sale_price, context.get("calculator_tax_pct")),
    )
    wb_payout_rub = _subtract_numbers(
        sale_price,
        total_commission_rub,
        logistics_rub,
        storage_rub,
        advertising_rub,
        acceptance_rub,
    )
    profit_rub = _subtract_numbers(wb_payout_rub, total_cost_rub, tax_rub)

    return CommissionCalculatorRow(
        rowId=f"commission_{nm_id}",
        brand=brand,
        subject=subject,
        productName=_normalize_optional_string(context.get("product_name")),
        vendorArticle=vendor_article,
        article=nm_id,
        basePriceRub=base_price,
        discountPct=_first_money(
            context.get("seller_discount_pct"),
            finance.get("discount_pct"),
        ),
        wbSiteDiscountPct=_first_money(
            context.get("discount_pct"),
            finance.get("wb_site_discount_pct"),
        ),
        checkStatus=_check_status(
            {
                **context,
                "price_before_discount": base_price,
                "price_after_discount": sale_price,
                "price_with_spp": price_with_spp,
            }
        ),
        salePriceWithoutSppRub=sale_price,
        wbClubDiscountPct=_optional_number(context.get("club_discount_pct")),
        priceWithWbClubRub=_optional_number(context.get("price_with_wb_club")),
        sppPct=_first_money(context.get("spp_pct"), finance.get("spp_pct")),
        priceWithSppRub=price_with_spp,
        priceWithWalletRub=None,
        commissionPct=_first_money(
            actual_commission_pct,
            context.get("calculator_commission_pct"),
        ),
        acquiringPct=acquiring_pct,
        totalCommissionPct=total_commission_pct,
        commissionRub=total_commission_rub,
        logisticsTariffRub=logistics_rub,
        isMonopallet=None,
        monopalletItemsQty=None,
        logisticsTariffMonopalletRub=None,
        buyoutPct=_optional_number(context.get("buyout_percent")),
        logisticsRub=logistics_rub,
        logisticsSalePct=_percent(logistics_rub, sale_price),
        unitCostRub=unit_cost_rub,
        defectRub=defect_rub,
        totalCostRub=total_cost_rub,
        costSalePct=_percent(total_cost_rub, sale_price),
        storageRub=storage_rub,
        storageSalePct=_percent(storage_rub, sale_price),
        advertisingRub=advertising_rub,
        advertisingSalePct=_percent(advertising_rub, sale_price),
        taxRub=tax_rub,
        taxSalePct=_first_money(
            context.get("calculator_tax_pct"),
            _percent(tax_rub, sale_price),
        ),
        vatRub=None,
        vatSalePct=None,
        acceptanceRub=acceptance_rub,
        acceptanceSalePct=_percent(acceptance_rub, sale_price),
        profitRub=profit_rub,
        marginSalePct=_percent(profit_rub, sale_price),
        profitCostPct=_percent(profit_rub, total_cost_rub),
        wbPayoutRub=wb_payout_rub,
    )


def _check_status(context: dict[str, Any]) -> str:
    if context.get("price_before_discount") is not None and context.get("price_after_discount") is not None:
        return "Цены заполнены"
    if context:
        return "Частично заполнено"
    return "Нет данных"


def _first_present(*values: Any) -> str | None:
    for value in values:
        normalized = _normalize_optional_string(value)
        if normalized is not None:
            return normalized
    return None


def _first_money(*values: Any) -> int | float | None:
    for value in values:
        number = _optional_number(value)
        if number is not None:
            return number
    return None


def _per_unit(total: Any, qty: Any) -> int | float | None:
    total_number = _optional_number(total)
    qty_number = _optional_number(qty)
    if total_number is None or qty_number in (None, 0):
        return None
    value = round(float(total_number) / float(qty_number), 2)
    return int(value) if value.is_integer() else value


def _sum_numbers(*values: Any) -> int | float | None:
    normalized = []
    for value in values:
        number = _optional_number(value)
        if number is not None:
            normalized.append(number)
    if not normalized:
        return None
    total = round(sum(float(value) for value in normalized), 2)
    return int(total) if total.is_integer() else total


def _subtract_numbers(base: Any, *deductions: Any) -> int | float | None:
    base_number = _optional_number(base)
    if base_number is None:
        return None
    result = float(base_number)
    for deduction in deductions:
        deduction_number = _optional_number(deduction)
        if deduction_number is not None:
            result -= float(deduction_number)
    result = round(result, 2)
    return int(result) if result.is_integer() else result


def _value_from_percent(base: Any, percent: Any) -> int | float | None:
    base_number = _optional_number(base)
    percent_number = _optional_number(percent)
    if base_number is None or percent_number is None:
        return None
    value = round(float(base_number) * float(percent_number) / 100, 2)
    return int(value) if value.is_integer() else value


def _percent(numerator: Any, denominator: Any) -> int | float | None:
    numerator_number = _optional_number(numerator)
    denominator_number = _optional_number(denominator)
    if numerator_number is None or denominator_number in (None, 0):
        return None
    value = round(100 * float(numerator_number) / float(denominator_number), 2)
    return int(value) if value.is_integer() else value


def _coerce_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str) and value.strip():
        return date.fromisoformat(value.strip())
    return None


def _normalize_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    stripped = str(value).strip()
    return stripped or None


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
