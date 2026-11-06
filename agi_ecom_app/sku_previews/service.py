from __future__ import annotations

import re
from typing import Any

from psycopg import sql

from agi_ecom_app.config import get_settings
from agi_ecom_app.services.db import get_connection, serialize_rows
from agi_ecom_app.sku_previews.models import SkuPreviewItem, SkuPreviewsResponse

MAX_SKU_PREVIEW_IDS = 100
SKU_ID_PATTERN = re.compile(r"^[0-9A-Za-z_-]+$")

SKU_COLUMN_CANDIDATES = ("vendor_code", "supplier_article", "sa_name", "sku")
WB_NM_ID_COLUMN_CANDIDATES = ("nm_id", "nmID", "nmId", "wb_id", "product_id")
TITLE_COLUMN_CANDIDATES = ("title", "name", "product_name", "subject_name", "vendor_code")
IMAGE_URL_COLUMN_CANDIDATES = (
    "main_photo_url",
    "analytics_main_photo_url",
    "photo_url",
    "image_url",
    "main_photo",
    "photo",
    "image",
)
PRODUCT_URL_COLUMN_CANDIDATES = ("product_url", "card_url", "url", "link")


def get_sku_previews_payload(ids: str) -> SkuPreviewsResponse:
    sku_ids = _parse_sku_ids(ids)
    if not sku_ids:
        return SkuPreviewsResponse(items={})

    rows = _fetch_sku_preview_rows(sku_ids)
    return SkuPreviewsResponse(
        items={
            row["sku"]: SkuPreviewItem(
                sku=row["sku"],
                title=row.get("title"),
                imageUrl=row.get("image_url"),
                productUrl=row.get("product_url") or _wildberries_product_url(row.get("wb_nm_id")),
            )
            for row in rows
        }
    )


def _parse_sku_ids(ids: str) -> list[str]:
    cleaned_ids = []
    seen = set()

    for raw_id in ids.split(","):
        sku_id = raw_id.strip()
        if not sku_id:
            continue
        if not SKU_ID_PATTERN.fullmatch(sku_id):
            raise ValueError(f"invalid SKU id: {sku_id}")
        if sku_id not in seen:
            cleaned_ids.append(sku_id)
            seen.add(sku_id)

    if len(cleaned_ids) > MAX_SKU_PREVIEW_IDS:
        raise ValueError(f"ids must contain at most {MAX_SKU_PREVIEW_IDS} SKU values")

    return cleaned_ids


def _fetch_sku_preview_rows(sku_ids: list[str]) -> list[dict[str, Any]]:
    settings = get_settings()
    preview_config = settings.sku_preview

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            select column_name
            from information_schema.columns
            where table_schema = %s and table_name = %s
            """,
            [preview_config.schema, preview_config.table],
        )
        available_columns = {row["column_name"] for row in cur.fetchall()}

        sku_column = _resolve_required_column(
            configured=preview_config.sku_column,
            candidates=SKU_COLUMN_CANDIDATES,
            available_columns=available_columns,
            purpose="SKU",
        )
        wb_nm_id_column = _resolve_optional_column(
            configured=preview_config.wb_nm_id_column,
            candidates=WB_NM_ID_COLUMN_CANDIDATES,
            available_columns=available_columns,
        )
        title_column = _resolve_optional_column(
            configured=preview_config.title_column,
            candidates=TITLE_COLUMN_CANDIDATES,
            available_columns=available_columns,
        )
        image_url_column = _resolve_optional_column(
            configured=preview_config.image_url_column,
            candidates=IMAGE_URL_COLUMN_CANDIDATES,
            available_columns=available_columns,
        )
        product_url_column = _resolve_optional_column(
            configured=preview_config.product_url_column,
            candidates=PRODUCT_URL_COLUMN_CANDIDATES,
            available_columns=available_columns,
        )

        query = sql.SQL(
            """
            select distinct on ({sku_column}::text)
                {sku_column}::text as sku,
                {wb_nm_id_expr} as wb_nm_id,
                {title_expr} as title,
                {image_url_expr} as image_url,
                {product_url_expr} as product_url
            from {table_ref}
            where {sku_column}::text = any(%s)
            order by {sku_column}::text
            """
        ).format(
            sku_column=sql.Identifier(sku_column),
            wb_nm_id_expr=_nullable_text_column(wb_nm_id_column),
            title_expr=_nullable_text_column(title_column),
            image_url_expr=_nullable_text_column(image_url_column),
            product_url_expr=_nullable_text_column(product_url_column),
            table_ref=sql.Identifier(preview_config.schema, preview_config.table),
        )

        cur.execute(query, [sku_ids])
        return serialize_rows(cur.fetchall())


def _resolve_required_column(
    configured: str,
    candidates: tuple[str, ...],
    available_columns: set[str],
    purpose: str,
) -> str:
    column = _resolve_optional_column(configured, candidates, available_columns)
    if column is None:
        raise RuntimeError(f"SKU preview table does not contain a {purpose} column")
    return column


def _resolve_optional_column(
    configured: str | None,
    candidates: tuple[str, ...],
    available_columns: set[str],
) -> str | None:
    ordered_candidates = [column for column in (configured, *candidates) if column]
    for column in ordered_candidates:
        if column in available_columns:
            return column
    return None


def _nullable_text_column(column: str | None) -> sql.Composed:
    if column is None:
        return sql.SQL("null::text")
    return sql.SQL("nullif({column}::text, '')").format(column=sql.Identifier(column))


def _wildberries_product_url(wb_nm_id: str | None) -> str | None:
    if wb_nm_id is None or not wb_nm_id.isdigit():
        return None
    return f"https://www.wildberries.ru/catalog/{wb_nm_id}/detail.aspx"
