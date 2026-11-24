
import unittest
from unittest.mock import Mock

import pandas as pd

from app.services.dataset_service import DatasetService
from app.schemas.dataset import DatasetUploadResponse


class TestDatasetService(unittest.TestCase):

    
    def setUp(self):
        self.df = pd.DataFrame({
            "order_id": [101, 102, 103],
            "amount": [250.0, 400.0, 150.0],
            "status": ["completed", "pending", "completed"]
        })

        self.loader = Mock()
        self.structure_analyzer = Mock()
        self.quality_analyzer = Mock()

        self.loader.load.return_value = self.df

        self.structure_analyzer.analyze.return_value = {
            "row_count": 3,
            "column_count": 3,
            "columns": [
                {
                    "column_index": 0,
                    "name": "order_id",
                    "pandas_dtype": "int64",
                    "analytical_type": "numeric"
                },
                {
                    "column_index": 1,
                    "name": "amount",
                    "pandas_dtype": "float64",
                    "analytical_type": "numeric"
                },
                {
                    "column_index": 2,
                    "name": "status",
                    "pandas_dtype": "object",
                    "analytical_type": "categorical_or_text"
                }
            ]
        }

        self.quality_analyzer.analyze.return_value = {
            "dataset_summary": {
                "total_rows": 3,
                "total_columns": 3,
                "total_cells": 9
            },
            "missing_value_analysis": {
                "total_rows": 3,
                "columns": [
                    {
                        "column_index": 0,
                        "name": "order_id",
                        "pandas_dtype": "int64",
                        "missing_count": 0,
                        "blank_count": 0,
                        "valid_value_count": 3,
                        "missing_percentage": 0.0,
                        "blank_percentage": 0.0,
                        "issue_percentage": 0.0
                    },
                    {
                        "column_index": 1,
                        "name": "amount",
                        "pandas_dtype": "float64",
                        "missing_count": 0,
                        "blank_count": 0,
                        "valid_value_count": 3,
                        "missing_percentage": 0.0,
                        "blank_percentage": 0.0,
                        "issue_percentage": 0.0
                    },
                    {
                        "column_index": 2,
                        "name": "status",
                        "pandas_dtype": "object",
                        "missing_count": 0,
                        "blank_count": 0,
                        "valid_value_count": 3,
                        "missing_percentage": 0.0,
                        "blank_percentage": 0.0,
                        "issue_percentage": 0.0
                    }
                ]
            },
            "duplicate_analysis": {
                "total_rows": 3,
                "duplicate_rows": 0,
                "unique_rows": 3,
                "duplicate_percentage": 0.0
            },
            "empty_columns": [],
            "constant_columns": [],
            "unique_value_analysis": [
                {
                    "column_index": 0,
                    "name": "order_id",
                    "unique_count": 3
                },
                {
                    "column_index": 1,
                    "name": "amount",
                    "unique_count": 3
                },
                {
                    "column_index": 2,
                    "name": "status",
                    "unique_count": 2
                }
            ],
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

        self.service = DatasetService(
            loader=self.loader,
            structure_analyzer=self.structure_analyzer,
            quality_analyzer=self.quality_analyzer
        )


    def test_valid_upload_returns_response(self):
        response = self.service.process_upload(
            "orders.csv",
            b"sample content"
        )

        self.assertIsInstance(response, DatasetUploadResponse)
        self.assertEqual(response.status, "success")

    def test_dataset_metadata_is_correct(self):
        response = self.service.process_upload(
            "orders.csv",
            b"sample content"
        )

        self.assertEqual(response.dataset.filename, "orders.csv")
        self.assertEqual(response.dataset.total_rows, 3)
        self.assertEqual(response.dataset.total_columns, 3)

    def test_loader_is_called(self):
        self.service.process_upload(
            "orders.csv",
            b"sample content"
        )

        self.loader.load.assert_called_once_with(
            filename="orders.csv",
            file_content=b"sample content"
        )

    def test_structure_analyzer_is_called(self):
        self.service.process_upload(
            "orders.csv",
            b"sample content"
        )

        self.structure_analyzer.analyze.assert_called_once_with(self.df)

    def test_quality_analyzer_is_called(self):
        self.service.process_upload(
            "orders.csv",
            b"sample content"
        )

        self.quality_analyzer.analyze.assert_called_once_with(self.df)

    def test_loader_exception_propagates(self):
        self.loader.load.side_effect = ValueError("Invalid dataset")

        with self.assertRaises(ValueError):
            self.service.process_upload(
                "orders.csv",
                b"invalid content"
            )

        self.structure_analyzer.analyze.assert_not_called()
        self.quality_analyzer.analyze.assert_not_called()

    def test_analyzer_exception_propagates(self):
        self.structure_analyzer.analyze.side_effect = RuntimeError(
            "Structure analysis failed"
        )

        with self.assertRaises(RuntimeError):
            self.service.process_upload(
                "orders.csv",
                b"sample content"
            )

        self.quality_analyzer.analyze.assert_not_called()


if __name__ == "__main__":
    unittest.main()
