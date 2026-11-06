from __future__ import annotations

from fastapi import APIRouter, HTTPException

from agi_ecom_app.unit_economics.models import (
    FinanceReportOptionsResponse,
    UnitEconomicsResponse,
)
from agi_ecom_app.unit_economics.service import (
    get_finance_report_options_payload,
    get_unit_economics_payload,
)

router = APIRouter()


@router.get("/v1/finance/report-options", response_model=FinanceReportOptionsResponse)
def get_finance_report_options() -> FinanceReportOptionsResponse:
    return get_finance_report_options_payload()


@router.get("/v1/finance/unit-economics", response_model=UnitEconomicsResponse)
def get_unit_economics(
    date_from: str | None = None,
    date_to: str | None = None,
    product_keyword: str | None = None,
    limit: int = 100,
) -> UnitEconomicsResponse:
    try:
        return get_unit_economics_payload(
            date_from=date_from,
            date_to=date_to,
            product_keyword=product_keyword,
            limit=limit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
