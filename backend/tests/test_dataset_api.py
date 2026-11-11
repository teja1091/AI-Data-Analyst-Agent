
import unittest
from io import BytesIO

import pandas as pd
from fastapi.testclient import TestClient

from app.main import app


class TestDatasetUploadAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def create_excel_file(self):
        df = pd.DataFrame({
            "order_id": [101, 102, 103],
            "amount": [250.0, 400.0, 150.0],
            "status": ["completed", "pending", "completed"]
        })

        buffer = BytesIO()
        df.to_excel(buffer, index=False, engine="openpyxl")
        return buffer.getvalue()

    def test_valid_csv_upload(self):
        csv_content = (
            b"order_id,amount,status\n"
            b"101,250,completed\n"
            b"102,400,pending\n"
            b"103,150,completed\n"
        )

        response = self.client.post(
            "/api/datasets/upload",
            files={
                "file": (
                    "orders.csv",
                    csv_content,
                    "text/csv"
                )
            }
        )

        self.assertEqual(response.status_code, 200)

        data = response.json()

        self.assertEqual(data["status"], "success")
        self.assertEqual(data["dataset"]["filename"], "orders.csv")
        self.assertEqual(data["dataset"]["total_rows"], 3)
        self.assertEqual(data["dataset"]["total_columns"], 3)

        self.assertIn("structure", data)
        self.assertIn("quality", data)

    def test_valid_excel_upload(self):
        excel_content = self.create_excel_file()

        response = self.client.post(
            "/api/datasets/upload",
            files={
                "file": (
                    "orders.xlsx",
                    excel_content,
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            }
        )

        self.assertEqual(response.status_code, 200)

        data = response.json()

        self.assertEqual(data["status"], "success")
        self.assertEqual(data["dataset"]["total_rows"], 3)
        self.assertEqual(data["dataset"]["total_columns"], 3)

    def test_unsupported_file_format(self):
        response = self.client.post(
            "/api/datasets/upload",
            files={
                "file": (
                    "orders.txt",
                    b"some text",
                    "text/plain"
                )
            }
        )

        self.assertEqual(response.status_code, 400)

    def test_empty_file_upload(self):
        response = self.client.post(
            "/api/datasets/upload",
            files={
                "file": (
                    "empty.csv",
                    b"",
                    "text/csv"
                )
            }
        )

        self.assertEqual(response.status_code, 400)

    def test_invalid_csv_upload(self):
        invalid_csv = b'order_id,amount\n101,"unterminated\n'

        response = self.client.post(
            "/api/datasets/upload",
            files={
                "file": (
                    "invalid.csv",
                    invalid_csv,
                    "text/csv"
                )
            }
        )

        self.assertEqual(response.status_code, 400)

    def test_invalid_excel_upload(self):
        response = self.client.post(
            "/api/datasets/upload",
            files={
                "file": (
                    "invalid.xlsx",
                    b"this is not a valid Excel workbook",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            }
        )

        self.assertEqual(response.status_code, 400)

    def test_file_exceeding_size_limit(self):
        oversized_content = b"a" * (10 * 1024 * 1024 + 1)

        response = self.client.post(
            "/api/datasets/upload",
            files={
                "file": (
                    "large.csv",
                    oversized_content,
                    "text/csv"
                )
            }
        )

        self.assertEqual(response.status_code, 413)

    def test_missing_file(self):
        response = self.client.post(
            "/api/datasets/upload"
        )

        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
