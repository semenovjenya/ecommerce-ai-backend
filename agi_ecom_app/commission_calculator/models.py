from __future__ import annotations

from pydantic import BaseModel, Field


class CommissionCalculatorReport(BaseModel):
    reportId: str
    label: str
    sortOrder: int


class CommissionCalculatorPeriod(BaseModel):
    dateFrom: str
    dateTo: str
    isDefaultPeriod: bool


class CommissionCalculatorColumn(BaseModel):
    key: str
    label: str
    type: str


class CommissionCalculatorRow(BaseModel):
    rowId: str
    brand: str | None = None
    subject: str | None = None
    productName: str | None = None
    vendorArticle: str | None = None
    article: int | None = None
    basePriceRub: int | float | None = None
    discountPct: int | float | None = None
    wbSiteDiscountPct: int | float | None = None
    checkStatus: str | None = None
    salePriceWithoutSppRub: int | float | None = None
    wbClubDiscountPct: int | float | None = None
    priceWithWbClubRub: int | float | None = None
    sppPct: int | float | None = None
    priceWithSppRub: int | float | None = None
    priceWithWalletRub: int | float | None = None
    commissionPct: int | float | None = None
    acquiringPct: int | float | None = None
    totalCommissionPct: int | float | None = None
    commissionRub: int | float | None = None
    logisticsTariffRub: int | float | None = None
    isMonopallet: bool | None = None
    monopalletItemsQty: int | float | None = None
    logisticsTariffMonopalletRub: int | float | None = None
    buyoutPct: int | float | None = None
    logisticsRub: int | float | None = None
    logisticsSalePct: int | float | None = None
    unitCostRub: int | float | None = None
    defectRub: int | float | None = None
    totalCostRub: int | float | None = None
    costSalePct: int | float | None = None
    storageRub: int | float | None = None
    storageSalePct: int | float | None = None
    advertisingRub: int | float | None = None
    advertisingSalePct: int | float | None = None
    taxRub: int | float | None = None
    taxSalePct: int | float | None = None
    vatRub: int | float | None = None
    vatSalePct: int | float | None = None
    acceptanceRub: int | float | None = None
    acceptanceSalePct: int | float | None = None
    profitRub: int | float | None = None
    marginSalePct: int | float | None = None
    profitCostPct: int | float | None = None
    wbPayoutRub: int | float | None = None


class CommissionCalculatorTotals(BaseModel):
    requestedNmId: int | None = None
    returnedRowCount: int
    hasPriceData: bool
    hasFinanceData: bool
    hasCostData: bool


class CommissionCalculatorResponse(BaseModel):
    status: str = "completed"
    report: CommissionCalculatorReport
    period: CommissionCalculatorPeriod
    columns: list[CommissionCalculatorColumn] = Field(default_factory=list)
    rows: list[CommissionCalculatorRow] = Field(default_factory=list)
    totals: CommissionCalculatorTotals
