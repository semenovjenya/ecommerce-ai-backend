from __future__ import annotations

from typing import Any, Dict, List, Optional

from agents import Tool, function_tool

from agi_ecom_app.services import db

SALES_SCHEMAS = ("sales",)
SUPPLY_SCHEMAS = ("supply",)
MARKETING_SCHEMAS = ("marketing",)
FINANCE_SCHEMAS = ("finance",)
PRICE_SCHEMAS = ("analytics_ai", "pricing")


def build_role_db_tools(role_name: str, allowed_schemas: tuple[str, ...]) -> list[Tool]:
    schema_list = ", ".join(allowed_schemas)

    @function_tool(
        name_override=f"{role_name}_list_tables",
        description_override=f"List tables and views only in the allowed schemas: {schema_list}.",
    )
    def list_allowed_tables() -> List[Dict[str, Any]]:
        return db.list_tables_in_schemas(allowed_schemas)

    @function_tool(
        name_override=f"{role_name}_describe_table",
        description_override=f"Describe a table only if its schema is one of: {schema_list}.",
    )
    def describe_allowed_table(table_name: str, schema_name: str) -> List[Dict[str, Any]]:
        return db.describe_table_in_schemas(
            table_name=table_name,
            schema=schema_name,
            allowed_schemas=allowed_schemas,
        )

    @function_tool(
        name_override=f"{role_name}_search_columns",
        description_override=f"Search columns only in the allowed schemas: {schema_list}.",
    )
    def search_allowed_columns(
        pattern: str,
        schema_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return db.search_columns_in_schemas(
            pattern=pattern,
            allowed_schemas=allowed_schemas,
            schema=schema_name,
        )

    @function_tool(
        name_override=f"{role_name}_sample_rows",
        description_override=f"Sample rows only from the allowed schemas: {schema_list}.",
    )
    def sample_allowed_rows(
        table_name: str,
        schema_name: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        return db.sample_rows_in_schemas(
            table_name=table_name,
            schema=schema_name,
            allowed_schemas=allowed_schemas,
            limit=limit,
        )

    @function_tool(
        name_override=f"{role_name}_run_sql",
        description_override=(
            f"Run readonly SQL only inside the allowed schemas: {schema_list}. "
            "Any explicit reference to a schema outside this role scope is blocked."
        ),
    )
    def run_allowed_sql(query: str, limit: int = 200) -> List[Dict[str, Any]]:
        return db.run_scoped_readonly_sql(
            query=query,
            allowed_schemas=allowed_schemas,
            limit=limit,
        )

    return [
        list_allowed_tables,
        describe_allowed_table,
        search_allowed_columns,
        sample_allowed_rows,
        run_allowed_sql,
    ]
