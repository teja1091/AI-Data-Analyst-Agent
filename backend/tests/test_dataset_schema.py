
import unittest

from pydantic import ValidationError

from app.schemas.dataset import (
    DatasetMetadata,
    DatasetUploadResponse,
)


class TestDatasetMetadata(unittest.TestCase):

    def test_valid_metadata(self):
        metadata = DatasetMetadata(
            filename="orders.csv",
            total_rows=100,
            total_columns=5
        )

        self.assertEqual(metadata.filename, "orders.csv")
        self.assertEqual(metadata.total_rows, 100)
        self.assertEqual(metadata.total_columns, 5)

    def test_negative_rows_are_rejected(self):
        with self.assertRaises(ValidationError):
            DatasetMetadata(
                filename="orders.csv",
                total_rows=-1,
                total_columns=5
            )

    def test_negative_columns_are_rejected(self):
        with self.assertRaises(ValidationError):
            DatasetMetadata(
                filename="orders.csv",
                total_rows=100,
                total_columns=-1
            )

    def test_empty_filename_is_rejected(self):
        with self.assertRaises(ValidationError):
            DatasetMetadata(
                filename="",
                total_rows=100,
                total_columns=5
            )


class TestDatasetUploadResponse(unittest.TestCase):

    
    def setUp(self):
        self.valid_response = {
            "status": "success",
            "message": "Upload successful",
            "dataset": {
                "filename": "orders.csv",
                "total_rows": 3,
                "total_columns": 3
            },
            "structure": {
                "row_count": 3,
                "column_count": 3,
                "columns": [
                    {
                        "column_index": 0,
                        "name": "order_id",
                        "pandas_dtype": "int64",
                        "analytical_type": "numeric"
                    }
                ]
            },
            "quality": {
                "dataset_summary": {
                    "total_rows": 3,
                    "total_columns": 3,
                    "total_cells": 9
                },
                "missing_value_analysis": {
                    "total_rows": 3,
                    "columns": []
                },
                "duplicate_analysis": {
                    "total_rows": 3,
                    "duplicate_rows": 0,
                    "unique_rows": 3,
                    "duplicate_percentage": 0.0
                },
                "empty_columns": [],
                "constant_columns": [],
                "unique_value_analysis": [],
                "dominant_value_analysis": {
                    "threshold_percentage": 80.0,
                    "columns": []
                },
                "quality_summary": {
                    "total_missing_values": 0,
                    "total_blank_values": 0,
                    "total_missing_and_blank_cells": 0,
                    "missing_and_blank_percentage": 0.0,
                    "duplicate_rows": 0,
                    "empty_column_count": 0,
                    "constant_column_count": 0,
                    "dominant_value_column_count": 0
                },
                "warnings": []
            }
        }


    def test_valid_response(self):
        response = DatasetUploadResponse(**self.valid_response)

        self.assertEqual(response.status, "success")
        self.assertIsInstance(response.dataset, DatasetMetadata)

    def test_invalid_status_is_rejected(self):
        data = self.valid_response.copy()
        data["status"] = "failed"

        with self.assertRaises(ValidationError):
            DatasetUploadResponse(**data)

    def test_invalid_analytical_type_is_rejected(self):
        data = self.valid_response.copy()
        data["structure"] = {
            **data["structure"],
            "columns": [{
                "column_index": 0,
                "name": "order_id",
                "pandas_dtype": "int64",
                "analytical_type": "invalid_type"
            }]
        }

        with self.assertRaises(ValidationError):
            DatasetUploadResponse(**data)

if __name__ == "__main__":
    unittest.main()
