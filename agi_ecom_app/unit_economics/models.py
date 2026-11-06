from __future__ import annotations

from pydantic import BaseModel, Field


class FinanceReportOption(BaseModel):
    reportId: str
    label: str
    sortOrder: int


class FinanceReportOptionsResponse(BaseModel):
    status: str = "completed"
    items: list[FinanceReportOption]


class UnitEconomicsPeriod(BaseModel):
    dateFrom: str | None = None
    dateTo: str | None = None
    label: str | None = None
    isDefaultPeriod: bool


class UnitEconomicsColumn(BaseModel):
    key: str
    label: str
    type: str
    align: str | None = None


class UnitEconomicsRow(BaseModel):
    rowId: str
    nmId: str | None = None
    vendorCode: str | None = None
    brandName: str | None = None
    subjectName: str | None = None
    salePriceRub: int | float | None = None
    unitCostRub: int | float | None = None
    commissionPct: int | float | None = None
    commissionRub: int | float | None = None
    logisticsRub: int | float | None = None
    returnLogisticsRub: int | float | None = None
    storageRub: int | float | None = None
    adRub: int | float | None = None
    otherExpensesRub: int | float | None = None
    taxRub: int | float | None = None
    profitRub: int | float | None = None
    marginPct: int | float | None = None


class UnitEconomicsTotals(BaseModel):
    totalRowCount: int
    returnedRowCount: int
    sourceTable: str = "finance.wb_profit_calculator_results"


class UnitEconomicsResponse(BaseModel):
    status: str = "completed"
    report: FinanceReportOption
    period: UnitEconomicsPeriod
    columns: list[UnitEconomicsColumn] = Field(default_factory=list)
    rows: list[UnitEconomicsRow] = Field(default_factory=list)
    totals: UnitEconomicsTotals
