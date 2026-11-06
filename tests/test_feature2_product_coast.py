import unittest
from datetime import date
from decimal import Decimal
from io import BytesIO
from unittest.mock import patch

from fastapi.testclient import TestClient
from openpyxl import Workbook

from agi_ecom_app.api import app
from agi_ecom_app.document_import.models import ProductCoastImportResponse
from agi_ecom_app.document_import.service import (
    import_product_coast_file,
    parse_product_coast_workbook,
)


def _workbook_bytes(rows):
    workbook = Workbook()
    worksheet = workbook.active
    for row in rows:
        worksheet.append(row)
    output = BytesIO()
    workbook.save(output)
    workbook.close()
    return output.getvalue()


class FakeProductCoastCursor:
    def __init__(self, table_exists=True):
        self.table_exists = table_exists
        self.executed = []
        self.executemany_calls = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, query, params=None):
        self.executed.append((query, params))

    def executemany(self, query, rows):
        self.executemany_calls.append((query, list(rows)))

    def fetchone(self):
        if not self.table_exists:
            return {"table_regclass": None}
        return {"table_regclass": "finance.wb_product_coast"}


class FakeProductCoastConnection:
    def __init__(self, table_exists=True):
        self.cursor_obj = FakeProductCoastCursor(table_exists=table_exists)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def cursor(self):
        return self.cursor_obj


class ProductCoastParserTests(unittest.TestCase):
    def test_parser_accepts_flexible_headers_and_money_strings(self):
        content = _workbook_bytes(
            [
                ["Артикул WB", "Артикул продавца", "Название товара", "Бренд", "Себестоимость, руб"],
                [70195339, "46230", "Сыворотка", "Lifecode", "1 234,50 ₽"],
                [None, None, None, None, None],
            ]
        )

        result = parse_product_coast_workbook(content)

        self.assertEqual(result.rows_parsed, 1)
        self.assertEqual(result.rows_skipped, 1)
        self.assertEqual(result.rows[0].product_key, "70195339")
        self.assertEqual(result.rows[0].vendor_code, "46230")
        self.assertEqual(result.rows[0].product_name, "Сыворотка")
        self.assertEqual(result.rows[0].brand_name, "Lifecode")
        self.assertEqual(result.rows[0].cost_price_rub, Decimal("1234.50"))

    def test_parser_accepts_english_headers_and_vendor_code_identity(self):
        content = _workbook_bytes(
            [
                ["vendor_code", "product name", "brand", "cost"],
                ["A-100", "Product", "Brand", 99.9],
            ]
        )

        result = parse_product_coast_workbook(content)

        self.assertEqual(result.rows[0].product_key, "A-100")
        self.assertIsNone(result.rows[0].nm_id)
        self.assertEqual(result.rows[0].cost_price_rub, Decimal("99.9"))

    def test_parser_rejects_invalid_rows_all_or_nothing(self):
        content = _workbook_bytes(
            [
                ["SKU", "Cost"],
                ["bad sku", "10"],
                [70195339, "-1"],
            ]
        )

        with self.assertRaises(ValueError) as ctx:
            parse_product_coast_workbook(content)

        self.assertIn("invalid product cost rows", str(ctx.exception))
        self.assertIn("row 2", str(ctx.exception))
        self.assertIn("row 3", str(ctx.exception))


class ProductCoastImportServiceTests(unittest.TestCase):
    @patch("agi_ecom_app.document_import.service.get_connection")
    def test_import_upserts_rows_by_upload_date_and_product_key(self, mock_get_connection):
        connection = FakeProductCoastConnection()
        mock_get_connection.return_value = connection
        content = _workbook_bytes(
            [
                ["SKU", "vendor_code", "product name", "brand", "cost"],
                [70195339, "46230", "Сыворотка", "Lifecode", "344,00"],
            ]
        )

        response = import_product_coast_file(
            content=content,
            filename="costs.xlsx",
            upload_date=date(2026, 4, 6),
        )

        self.assertEqual(response.uploadDate, date(2026, 4, 6))
        self.assertEqual(response.rowsParsed, 1)
        self.assertEqual(response.rowsSaved, 1)
        rows = connection.cursor_obj.executemany_calls[0][1]
        self.assertEqual(
            rows[0],
            (
                date(2026, 4, 6),
                "70195339",
                "46230",
                70195339,
                "Сыворотка",
                "Lifecode",
                Decimal("344.00"),
                "costs.xlsx",
                2,
            ),
        )

    @patch("agi_ecom_app.document_import.service.get_connection")
    def test_import_reports_missing_target_table(self, mock_get_connection):
        mock_get_connection.return_value = FakeProductCoastConnection(table_exists=False)
        content = _workbook_bytes([["SKU", "Cost"], [70195339, "344"]])

        with self.assertRaises(RuntimeError) as ctx:
            import_product_coast_file(
                content=content,
                filename="costs.xlsx",
                upload_date=date(2026, 4, 6),
            )

        self.assertIn("finance.wb_product_coast is missing", str(ctx.exception))


class ProductCoastApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch("agi_ecom_app.document_import.router.import_product_coast_file")
    def test_import_endpoint_accepts_multipart_excel(self, mock_import):
        mock_import.return_value = ProductCoastImportResponse(
            uploadDate=date(2026, 4, 6),
            rowsParsed=1,
            rowsSaved=1,
            rowsSkipped=0,
            sourceFilename="costs.xlsx",
        )

        response = self.client.post(
            "/v1/finance/product-coast/import",
            files={
                "file": (
                    "costs.xlsx",
                    b"excel-bytes",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["uploadDate"], "2026-04-06")
        self.assertEqual(response.json()["rowsSaved"], 1)
        mock_import.assert_called_once_with(content=b"excel-bytes", filename="costs.xlsx")

    @patch("agi_ecom_app.document_import.router.import_product_coast_file")
    def test_import_endpoint_returns_422_for_invalid_file(self, mock_import):
        mock_import.side_effect = ValueError("only .xlsx and .xlsm Excel files are supported")

        response = self.client.post(
            "/v1/finance/product-coast/import",
            files={"file": ("costs.xls", b"legacy", "application/vnd.ms-excel")},
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"], "only .xlsx and .xlsm Excel files are supported")


if __name__ == "__main__":
    unittest.main()
