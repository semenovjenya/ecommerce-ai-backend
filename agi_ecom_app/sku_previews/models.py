from __future__ import annotations

from pydantic import BaseModel


class SkuPreviewItem(BaseModel):
    sku: str
    title: str | None = None
    imageUrl: str | None = None
    productUrl: str | None = None


class SkuPreviewsResponse(BaseModel):
    items: dict[str, SkuPreviewItem]

