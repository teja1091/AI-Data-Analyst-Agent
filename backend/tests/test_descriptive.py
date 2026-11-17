
import unittest

import numpy as np
import pandas as pd

from app.data_engine.eda.descriptive import DescriptiveAnalyzer


class TestDescriptiveAnalyzer(unittest.TestCase):

    def test_basic_descriptive_statistics(self):
        df = pd.DataFrame({
            "sales": [10, 20, 30, 40, 50]
        })

        result = DescriptiveAnalyzer.analyze(df)
        column = result["columns"][0]
        stats = column["statistics"]

        self.assertEqual(column["name"], "sales")
        self.assertEqual(column["valid_value_count"], 5)
        self.assertEqual(stats["mean"], 30.0)
        self.assertEqual(stats["median"], 30.0)
        self.assertEqual(stats["minimum"], 10.0)
        self.assertEqual(stats["maximum"], 50.0)
        self.assertEqual(stats["range"], 40.0)

    def test_missing_values(self):
        df = pd.DataFrame({
            "sales": [10.0, 20.0, np.nan, 40.0]
        })

        result = DescriptiveAnalyzer.analyze(df)
        column = result["columns"][0]

        self.assertEqual(column["missing_count"], 1)
        self.assertEqual(column["valid_value_count"], 3)
        self.assertEqual(column["statistics"]["mean"], 70 / 3)

    def test_non_numeric_columns_are_excluded(self):
        df = pd.DataFrame({
            "sales": [10, 20, 30],
            "category": ["A", "B", "A"],
            "active": [True, False, True]
        })

        result = DescriptiveAnalyzer.analyze(df)

        self.assertEqual(
            result["dataset_summary"]["numeric_column_count"],
            1
        )
        self.assertEqual(result["columns"][0]["name"], "sales")

    def test_empty_numeric_column(self):
        df = pd.DataFrame({
            "sales": [np.nan, np.nan, np.nan]
        })

        result = DescriptiveAnalyzer.analyze(df)
        column = result["columns"][0]

        self.assertEqual(column["valid_value_count"], 0)
        self.assertEqual(column["missing_count"], 3)
        self.assertIsNone(column["statistics"])

    def test_non_finite_values(self):
        df = pd.DataFrame({
            "measurement": [10.0, 20.0, np.inf, -np.inf]
        })

        result = DescriptiveAnalyzer.analyze(df)
        column = result["columns"][0]

        self.assertEqual(column["valid_value_count"], 2)
        self.assertEqual(column["non_finite_count"], 2)
        self.assertEqual(column["statistics"]["mean"], 15.0)

    def test_duplicate_column_names(self):
        df = pd.DataFrame(
            [[10, 100], [20, 200]],
            columns=["value", "value"]
        )

        result = DescriptiveAnalyzer.analyze(df)

        self.assertEqual(len(result["columns"]), 2)
        self.assertEqual(
            result["columns"][0]["column_index"], 0
        )
        self.assertEqual(
            result["columns"][1]["column_index"], 1
        )

    def test_empty_dataframe(self):
        df = pd.DataFrame()

        result = DescriptiveAnalyzer.analyze(df)

        self.assertEqual(
            result["dataset_summary"]["total_rows"], 0
        )
        self.assertEqual(
            result["dataset_summary"]["numeric_column_count"], 0
        )
        self.assertEqual(result["columns"], [])

    def test_invalid_input(self):
        with self.assertRaises(TypeError):
            DescriptiveAnalyzer.analyze([10, 20, 30])

    def test_original_dataframe_is_not_modified(self):
        df = pd.DataFrame({
            "sales": [10.0, np.nan, 30.0]
        })

        original = df.copy(deep=True)

        DescriptiveAnalyzer.analyze(df)

        pd.testing.assert_frame_equal(df, original)


if __name__ == "__main__":
    unittest.main()
