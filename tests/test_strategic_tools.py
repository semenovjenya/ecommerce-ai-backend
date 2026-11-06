from agi_ecom_app.tools.strategic_tools import (
    get_finance_overview,
    get_price_context,
    get_sales_overview,
    get_supply_overview,
)


def test_portfolio_overview_tools_expose_full_matrix_limit_to_the_model() -> None:
    tools = (get_sales_overview, get_supply_overview, get_finance_overview)

    for tool in tools:
        limit_schema = tool.params_json_schema["properties"]["limit"]
        assert limit_schema["minimum"] == 1
        assert limit_schema["maximum"] == 100
        assert limit_schema["default"] == 100


def test_price_context_tool_exposes_explicit_price_lookup_inputs() -> None:
    properties = get_price_context.params_json_schema["properties"]

    assert "nm_id" in properties
    assert "vendor_code" in properties
    assert "product_keyword" in properties
    assert "include_evidence" in properties
