from __future__ import annotations

from fastapi import APIRouter, HTTPException

from agi_ecom_app.sku_previews.models import SkuPreviewsResponse
from agi_ecom_app.sku_previews.service import get_sku_previews_payload

router = APIRouter()


@router.get("/v1/skus/previews", response_model=SkuPreviewsResponse)
def get_sku_previews(ids: str) -> SkuPreviewsResponse:
    try:
        return get_sku_previews_payload(ids)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
