from typing import Any, Dict, List, Optional
from agents import function_tool

from agi_ecom_app.services import db


@function_tool
def list_tables(schema_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """List available tables and views in the database, optionally filtered by schema."""
    return db.list_tables(schema=schema_name)


@function_tool
def describe_table(table_name: str, schema_name: str = "public") -> List[Dict[str, Any]]:
    """Describe columns for a specific table."""
    return db.describe_table(table_name=table_name, schema=schema_name)


@function_tool
def search_columns(
    pattern: str, schema_name: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Find columns whose names match a pattern, optionally filtered by schema."""
    return db.search_columns(pattern=pattern, schema=schema_name)


@function_tool
def sample_rows(
    table_name: str, schema_name: str = "public", limit: int = 5
) -> List[Dict[str, Any]]:
    """Return a small sample of rows from a table to understand its semantics."""
    return db.sample_rows(table_name=table_name, schema=schema_name, limit=limit)


@function_tool
def run_readonly_sql(query: str, limit: int = 200) -> List[Dict[str, Any]]:
    """Execute a readonly SQL query. Only SELECT/WITH queries are allowed."""
    return db.run_readonly_sql(query=query, limit=limit)


@function_tool
def get_database_time() -> Dict[str, Any]:
    """Return the database current date and current timestamp."""
    return db.get_database_time()


@function_tool
def get_sales_summary(
    date_from: str,
    date_to: Optional[str] = None,
    product_keyword: Optional[str] = None,
    limit: int = 20,
) -> Dict[str, Any]:
    """Return aggregated supplier sales for a date range, optionally filtered by product keyword."""
    return db.get_sales_summary(
        date_from=date_from,
        date_to=date_to,
        product_keyword=product_keyword,
        limit=limit,
    )


@function_tool
def get_sales_report_text(
    date_from: str,
    date_to: Optional[str] = None,
    product_keyword: Optional[str] = None,
    top_n: int = 10,
) -> str:
    """Return a preformatted sales report with exact totals and top rows from the database."""
    return db.get_sales_report_text(
        date_from=date_from,
        date_to=date_to,
        product_keyword=product_keyword,
        top_n=top_n,
    )


@function_tool
def get_finance_marketplace_report(
    date_from: str,
    date_to: Optional[str] = None,
) -> Dict[str, Any]:
    """Return the WB marketplace finance breakdown for a requested date range from finance.wb_report_detail_daily."""
    return db.get_finance_marketplace_report(
        date_from=date_from,
        date_to=date_to,
    )


@function_tool
def get_finance_marketplace_report_text(
    date_from: str,
    date_to: Optional[str] = None,
) -> str:
    """Return a preformatted WB marketplace finance report for a requested date range."""
    return db.get_finance_marketplace_report_text(
        date_from=date_from,
        date_to=date_to,
    )


@function_tool
def get_finance_sku_report(
    date_from: str,
    date_to: Optional[str] = None,
    product_keyword: Optional[str] = None,
    limit: int = 100,
    sort_by: str = "net_revenue",
) -> Dict[str, Any]:
    """Return SKU-level WB marketplace finance economics for a requested date range."""
    return db.get_finance_sku_report(
        date_from=date_from,
        date_to=date_to,
        product_keyword=product_keyword,
        limit=limit,
        sort_by=sort_by,
    )


@function_tool
def get_finance_sku_report_text(
    date_from: str,
    date_to: Optional[str] = None,
    product_keyword: Optional[str] = None,
    limit: int = 50,
    sort_by: str = "net_revenue",
) -> str:
    """Return a preformatted SKU-level WB marketplace finance report for a requested date range."""
    return db.get_finance_sku_report_text(
        date_from=date_from,
        date_to=date_to,
        product_keyword=product_keyword,
        limit=limit,
        sort_by=sort_by,
    )


@function_tool
def get_finance_sku_profit_change_report(
    current_date_from: str,
    current_date_to: Optional[str] = None,
    previous_date_from: Optional[str] = None,
    previous_date_to: Optional[str] = None,
    product_keyword: Optional[str] = None,
    limit: int = 20,
) -> Dict[str, Any]:
    """Return SKU profit/margin change between current and previous finance periods."""
    return db.get_finance_sku_profit_change_report(
        current_date_from=current_date_from,
        current_date_to=current_date_to,
        previous_date_from=previous_date_from,
        previous_date_to=previous_date_to,
        product_keyword=product_keyword,
        limit=limit,
    )


@function_tool
def get_finance_sku_profit_change_report_text(
    current_date_from: str,
    current_date_to: Optional[str] = None,
    previous_date_from: Optional[str] = None,
    previous_date_to: Optional[str] = None,
    product_keyword: Optional[str] = None,
    limit: int = 20,
) -> str:
    """Return a preformatted SKU profit/margin change report between two finance periods."""
    return db.get_finance_sku_profit_change_report_text(
        current_date_from=current_date_from,
        current_date_to=current_date_to,
        previous_date_from=previous_date_from,
        previous_date_to=previous_date_to,
        product_keyword=product_keyword,
        limit=limit,
    )


@function_tool
def get_1pager_ai_summary(
    report_date: Optional[str] = None,
    product_keyword: Optional[str] = None,
    limit: int = 20,
) -> Dict[str, Any]:
    """Return rows from the analytics view reports.v_1pager_ai for a date and optional product filter."""
    return db.get_1pager_ai_summary(
        report_date=report_date,
        product_keyword=product_keyword,
        limit=limit,
    )


@function_tool
def get_1pager_ai_report_text(
    report_date: Optional[str] = None,
    product_keyword: Optional[str] = None,
    top_n: int = 10,
) -> str:
    """Return a preformatted report from reports.v_1pager_ai."""
    return db.get_1pager_ai_report_text(
        report_date=report_date,
        product_keyword=product_keyword,
        top_n=top_n,
    )


@function_tool
def get_1pager_ai_sales_summary(
    report_date: Optional[str] = None,
    product_keyword: Optional[str] = None,
    limit: int = 20,
) -> Dict[str, Any]:
    """Return sales-focused aggregates from reports.v_1pager_ai for a date and optional product filter."""
    return db.get_1pager_ai_sales_summary(
        report_date=report_date,
        product_keyword=product_keyword,
        limit=limit,
    )


@function_tool
def get_1pager_ai_sales_report_text(
    report_date: Optional[str] = None,
    product_keyword: Optional[str] = None,
    top_n: int = 10,
) -> str:
    """Return a preformatted sales report from reports.v_1pager_ai."""
    return db.get_1pager_ai_sales_report_text(
        report_date=report_date,
        product_keyword=product_keyword,
        top_n=top_n,
    )
