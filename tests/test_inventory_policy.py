from agi_ecom_app.inventory_policy import (
    calculate_coverage_days,
    classify_availability,
    classify_coverage_days,
    normalize_turnover_row,
)


def test_coverage_formula_is_deterministic() -> None:
    assert calculate_coverage_days(stock_qty=20, sales_units=10, period_days=30) == 60.0
    assert calculate_coverage_days(stock_qty=20, sales_units=0, period_days=30) is None


def test_coverage_bands_do_not_overlap() -> None:
    assert classify_coverage_days(13.9) == "дефицит <14 дней"
    assert classify_coverage_days(14) == "норма 14–30 дней"
    assert classify_coverage_days(30) == "норма 14–30 дней"
    assert classify_coverage_days(30.1) == "умеренный избыток 31–60 дней"
    assert classify_coverage_days(60) == "умеренный избыток 31–60 дней"
    assert classify_coverage_days(60.1) == "критический избыток >60 дней"


def test_turnover_band_and_current_availability_are_separate_axes() -> None:
    row = normalize_turnover_row(
        {
            "avg_stock_qty": 9.2,
            "sales_units": 15,
            "current_stock_qty": 0,
        },
        period_days=30,
    )

    assert row["days_on_hand"] == 18.4
    assert row["turnover_band"] == "норма 14–30 дней"
    assert row["availability_status"] == "нет доступного остатка"
    assert classify_availability(None) == "остаток не подтверждён"
