from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from io import BytesIO
from pathlib import Path
from typing import Any

from openpyxl import load_workbook
from psycopg import sql

from agi_ecom_app.services.db import get_connection
from agi_ecom_app.document_import.models import ProductCoastImportResponse

PRODUCT_COAST_SCHEMA = "finance"
PRODUCT_COAST_TABLE = "wb_product_coast"
SUPPORTED_EXTENSIONS = {".xlsx", ".xlsm"}
MAX_ERROR_DETAILS = 10


def _normalize_header(value: Any) -> str:
    if value is None:
        return ""
    cleaned = str(value).strip().lower().replace("ё", "е")
    cleaned = re.sub(r"[\W_]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


HEADER_ALIASES = {
    "nm_id": {
        "nm_id",
        "nm id",
        "nmid",
        "nm",
        "sku",
        "wb sku",
        "wildberries sku",
        "артикул wb",
        "артикул вб",
        "wb артикул",
        "вб артикул",
        "код номенклатуры",
        "номенклатура wb",
        "номенклатура вб",
    },
    "vendor_code": {
        "vendor_code",
        "vendor code",
        "vendorcode",
        "seller_article",
        "supplier_article",
        "sa_name",
        "артикул",
        "артикул продавца",
        "артикул поставщика",
        "вендор код",
        "код продавца",
    },
    "product_name": {
        "product_name",
        "product name",
        "name",
        "title",
        "название",
        "название товара",
        "наименование",
        "наименование товара",
        "товар",
    },
    "brand_name": {
        "brand_name",
        "brand name",
        "brand",
        "бренд",
        "бренд товара",
    },
    "cost_price_rub": {
        "cost_price_rub",
        "cost price rub",
        "cost price",
        "cost",
        "product cost",
        "себестоимость",
        "себестоимость руб",
        "себестоимость товара",
        "закупочная цена",
        "закупочная себестоимость",
        "цена закупки",
    },
}

NORMALIZED_HEADER_ALIASES = {
    field: {_normalize_header(alias) for alias in aliases}
    for field, aliases in HEADER_ALIASES.items()
}


@dataclass(frozen=True)
class ProductCoastRow:
    product_key: str
    vendor_code: str | None
    nm_id: int | None
    product_name: str | None
    brand_name: str | None
    cost_price_rub: Decimal
    source_row_number: int


@dataclass(frozen=True)
class ProductCoastParseResult:
    rows: list[ProductCoastRow]
    rows_parsed: int
    rows_skipped: int


def import_product_coast_file(
    content: bytes,
    filename: str,
    upload_date: date | None = None,
) -> ProductCoastImportResponse:
    source_filename = _validate_filename(filename)
    parse_result = parse_product_coast_workbook(content)
    effective_upload_date = upload_date or date.today()

    _save_product_coast_rows(
        rows=parse_result.rows,
        upload_date=effective_upload_date,
        source_filename=source_filename,
    )

    return ProductCoastImportResponse(
        uploadDate=effective_upload_date,
        rowsParsed=parse_result.rows_parsed,
        rowsSaved=len(parse_result.rows),
        rowsSkipped=parse_result.rows_skipped,
        sourceFilename=source_filename,
    )


def parse_product_coast_workbook(content: bytes) -> ProductCoastParseResult:
    if not content:
        raise ValueError("Excel file is empty")

    try:
        workbook = load_workbook(BytesIO(content), read_only=True, data_only=True)
    except Exception as exc:
        raise ValueError("file must be a valid .xlsx or .xlsm Excel workbook") from exc

    try:
        worksheet = workbook.worksheets[0]
        header_row_number, header_values = _find_header_row(worksheet.iter_rows(values_only=True))
        column_indexes = _resolve_columns(header_values)

        rows: list[ProductCoastRow] = []
        errors: list[str] = []
        rows_parsed = 0
        rows_skipped = 0

        for row_number, values in enumerate(
            worksheet.iter_rows(min_row=header_row_number + 1, values_only=True),
            start=header_row_number + 1,
        ):
            if _is_empty_row(values):
                rows_skipped += 1
                continue

            rows_parsed += 1
            try:
                parsed_row = _parse_data_row(row_number, values, column_indexes)
            except ValueError as exc:
                errors.append(f"row {row_number}: {exc}")
                continue
            rows.append(parsed_row)
    finally:
        workbook.close()

    if errors:
        shown_errors = "; ".join(errors[:MAX_ERROR_DETAILS])
        if len(errors) > MAX_ERROR_DETAILS:
            shown_errors += f"; and {len(errors) - MAX_ERROR_DETAILS} more"
        raise ValueError(f"invalid product cost rows: {shown_errors}")

    if not rows:
        raise ValueError("Excel file does not contain product cost rows")

    return ProductCoastParseResult(
        rows=rows,
        rows_parsed=rows_parsed,
        rows_skipped=rows_skipped,
    )


def _validate_filename(filename: str) -> str:
    cleaned = Path(filename).name.strip()
    if not cleaned:
        raise ValueError("file name must not be empty")
    extension = Path(cleaned).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError("only .xlsx and .xlsm Excel files are supported")
    return cleaned


def _find_header_row(rows: Any) -> tuple[int, tuple[Any, ...]]:
    for row_number, values in enumerate(rows, start=1):
        if not _is_empty_row(values):
            return row_number, values
    raise ValueError("Excel file does not contain a header row")


def _resolve_columns(header_values: tuple[Any, ...]) -> dict[str, int]:
    resolved: dict[str, int] = {}
    for index, raw_header in enumerate(header_values):
        normalized = _normalize_header(raw_header)
        if not normalized:
            continue
        for field, aliases in NORMALIZED_HEADER_ALIASES.items():
            if field not in resolved and normalized in aliases:
                resolved[field] = index

    if "cost_price_rub" not in resolved:
        raise ValueError("Excel file must contain a product cost column")
    if "nm_id" not in resolved and "vendor_code" not in resolved:
        raise ValueError("Excel file must contain nm_id/SKU or vendor_code column")

    return resolved


def _parse_data_row(
    row_number: int,
    values: tuple[Any, ...],
    column_indexes: dict[str, int],
) -> ProductCoastRow:
    nm_id = _parse_nm_id(_get_cell(values, column_indexes.get("nm_id")))
    vendor_code = _clean_text(_get_cell(values, column_indexes.get("vendor_code")))
    cost_price_rub = _parse_cost(_get_cell(values, column_indexes["cost_price_rub"]))

    if nm_id is None and vendor_code is None:
        raise ValueError("nm_id/SKU or vendor_code is required")

    product_key = str(nm_id) if nm_id is not None else str(vendor_code)
    return ProductCoastRow(
        product_key=product_key,
        vendor_code=vendor_code,
        nm_id=nm_id,
        product_name=_clean_text(_get_cell(values, column_indexes.get("product_name"))),
        brand_name=_clean_text(_get_cell(values, column_indexes.get("brand_name"))),
        cost_price_rub=cost_price_rub,
        source_row_number=row_number,
    )


def _save_product_coast_rows(
    rows: list[ProductCoastRow],
    upload_date: date,
    source_filename: str,
) -> None:
    with get_connection() as conn, conn.cursor() as cur:
        _ensure_product_coast_table_exists(cur)
        query = sql.SQL(
            """
            insert into {table_ref} (
                upload_date,
                product_key,
                vendor_code,
                nm_id,
                product_name,
                brand_name,
                cost_price_rub,
                source_filename,
                source_row_number
            )
            values (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            on conflict (upload_date, product_key) do update set
                vendor_code = excluded.vendor_code,
                nm_id = excluded.nm_id,
                product_name = excluded.product_name,
                brand_name = excluded.brand_name,
                cost_price_rub = excluded.cost_price_rub,
                source_filename = excluded.source_filename,
                source_row_number = excluded.source_row_number,
                updated_at = now()
            """
        ).format(table_ref=sql.Identifier(PRODUCT_COAST_SCHEMA, PRODUCT_COAST_TABLE))
        cur.executemany(
            query,
            [
                (
                    upload_date,
                    row.product_key,
                    row.vendor_code,
                    row.nm_id,
                    row.product_name,
                    row.brand_name,
                    row.cost_price_rub,
                    source_filename,
                    row.source_row_number,
                )
                for row in rows
            ],
        )


def _ensure_product_coast_table_exists(cur: Any) -> None:
    cur.execute(
        "select to_regclass(%s) as table_regclass",
        [f"{PRODUCT_COAST_SCHEMA}.{PRODUCT_COAST_TABLE}"],
    )
    row = cur.fetchone()
    if not row or not row.get("table_regclass"):
        raise RuntimeError(
            "finance.wb_product_coast is missing. Apply sql/create_wb_product_coast.sql first."
        )


def _get_cell(values: tuple[Any, ...], index: int | None) -> Any:
    if index is None or index >= len(values):
        return None
    return values[index]


def _parse_nm_id(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("nm_id/SKU must be numeric")
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float):
        if value <= 0 or not value.is_integer():
            raise ValueError("nm_id/SKU must be a positive integer")
        return int(value)

    cleaned = _clean_text(value)
    if cleaned is None:
        return None
    normalized = re.sub(r"[\s\u00a0]+", "", cleaned)
    if not normalized.isdigit():
        raise ValueError("nm_id/SKU must be numeric")
    parsed = int(normalized)
    return parsed if parsed > 0 else None


def _parse_cost(value: Any) -> Decimal:
    if value is None:
        raise ValueError("cost_price_rub is required")
    if isinstance(value, bool):
        raise ValueError("cost_price_rub must be numeric")
    if isinstance(value, (int, float, Decimal)):
        parsed = Decimal(str(value))
    else:
        cleaned = _clean_text(value)
        if cleaned is None:
            raise ValueError("cost_price_rub is required")
        normalized = _normalize_money_text(cleaned)
        try:
            parsed = Decimal(normalized)
        except InvalidOperation as exc:
            raise ValueError("cost_price_rub must be numeric") from exc

    if parsed < 0:
        raise ValueError("cost_price_rub must be greater than or equal to 0")
    return parsed


def _normalize_money_text(value: str) -> str:
    cleaned = value.replace("\u00a0", " ").replace(" ", "")
    cleaned = re.sub(r"[^\d,.\-]", "", cleaned)
    if "," in cleaned and "." in cleaned:
        if cleaned.rfind(",") > cleaned.rfind("."):
            return cleaned.replace(".", "").replace(",", ".")
        return cleaned.replace(",", "")
    if "," in cleaned:
        return cleaned.replace(",", ".")
    if cleaned.count(".") > 1:
        head, tail = cleaned.rsplit(".", 1)
        return head.replace(".", "") + "." + tail
    return cleaned


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _is_empty_row(values: tuple[Any, ...]) -> bool:
    return all(_clean_text(value) is None for value in values)
