from __future__ import annotations

from fastapi import APIRouter, HTTPException

from agi_ecom_app.commission_calculator.models import CommissionCalculatorResponse
from agi_ecom_app.commission_calculator.service import get_commission_calculator_payload

router = APIRouter()


@router.get(
    "/v1/finance/commission-calculator",
    response_model=CommissionCalculatorResponse,
)
def get_commission_calculator(
    nm_id: int | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> CommissionCalculatorResponse:
    try:
        return get_commission_calculator_payload(
            nm_id=nm_id,
            date_from=date_from,
            date_to=date_to,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
