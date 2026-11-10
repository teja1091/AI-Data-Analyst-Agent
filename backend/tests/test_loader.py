
import unittest
from io import BytesIO

import pandas as pd

from app.data_engine.loader import DatasetLoader


class TestDatasetLoader(unittest.TestCase):

    def setUp(self):
        self.csv_content = (
            b"Product,Quantity,Country\n"
            b"Apple,10,India\n"
            b"Banana,20,USA\n"
        )

    def test_load_valid_csv(self):
        df = DatasetLoader.load(
            "sales.csv",
            self.csv_content
        )

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, (2, 3))
        self.assertEqual(df.iloc[0]["Product"], "Apple")
        self.assertEqual(df.iloc[1]["Quantity"], 20)

    def test_load_valid_xlsx(self):
        excel_buffer = BytesIO()

        source_df = pd.DataFrame({
            "Product": ["Apple", "Banana"],
            "Quantity": [10, 20]
        })

        source_df.to_excel(
            excel_buffer,
            index=False,
            engine="openpyxl"
        )

        df = DatasetLoader.load(
            "sales.xlsx",
            excel_buffer.getvalue()
        )

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, (2, 2))
        self.assertEqual(df.iloc[0]["Product"], "Apple")

    def test_unsupported_file_format(self):
        with self.assertRaises(ValueError):
            DatasetLoader.load(
                "sales.json",
                self.csv_content
            )

    def test_empty_file_content(self):
        with self.assertRaises(ValueError):
            DatasetLoader.load(
                "sales.csv",
                b""
            )

    def test_invalid_filename(self):
        with self.assertRaises(ValueError):
            DatasetLoader.load(
                "",
                self.csv_content
            )

    def test_missing_file_extension(self):
        with self.assertRaises(ValueError):
            DatasetLoader.load(
                "sales",
                self.csv_content
            )

    def test_invalid_file_content_type(self):
        with self.assertRaises(TypeError):
            DatasetLoader.load(
                "sales.csv",
                "Product,Quantity"
            )

    def test_invalid_csv_structure(self):
        invalid_csv = (
            b"Product,Quantity\n"
            b'"Apple,10\n'
        )

        with self.assertRaises(ValueError):
            DatasetLoader.load(
                "sales.csv",
                invalid_csv
            )

    def test_empty_csv(self):
        csv_content = b"Product,Quantity\n"

        with self.assertRaises(ValueError):
            DatasetLoader.load(
                "sales.csv",
                csv_content
            )

    def test_invalid_excel_content(self):
        with self.assertRaises(ValueError):
            DatasetLoader.load(
                "sales.xlsx",
                b"This is not an Excel workbook"
            )

    def test_csv_with_utf8_bom(self):
        csv_content = (
            "\ufeffProduct,Quantity\n"
            "Apple,10\n"
        ).encode("utf-8")

        df = DatasetLoader.load(
            "sales.csv",
            csv_content
        )

        self.assertEqual(
            df.columns.tolist(),
            ["Product", "Quantity"]
        )

    def test_csv_with_windows_encoding(self):
        csv_content = (
            "Product,Country\n"
            "Café,India\n"
        ).encode("cp1252")

        df = DatasetLoader.load(
            "sales.csv",
            csv_content
        )

        self.assertEqual(
            df.iloc[0]["Product"],
            "Café"
        )


if __name__ == "__main__":
    unittest.main()
