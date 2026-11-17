
import unittest

import numpy as np
import pandas as pd

from app.data_engine.eda.categorical import CategoricalAnalyzer


class TestCategoricalAnalyzer(unittest.TestCase):

    def test_basic_category_analysis(self):

        df = pd.DataFrame({
            "membership": [
                "member",
                "casual",
                "member",
                "member",
                "casual"
            ]
        })

        result = CategoricalAnalyzer.analyze(df)

        column = result["columns"][0]

        self.assertEqual(
            column["unique_category_count"], 2
        )

        self.assertEqual(
            column["valid_value_count"], 5
        )

        self.assertEqual(
            column["most_frequent_category"], "member"
        )

        self.assertEqual(
            column["least_frequent_category"], "casual"
        )

        self.assertEqual(
            column["categories"][0]["percentage"], 60.0
        )

    def test_missing_and_blank_values(self):

        df = pd.DataFrame({
            "category": [
                "A",
                "B",
                None,
                np.nan,
                "",
                "   "
            ]
        })

        result = CategoricalAnalyzer.analyze(df)

        column = result["columns"][0]

        self.assertEqual(column["missing_count"], 2)
        self.assertEqual(column["blank_count"], 2)
        self.assertEqual(column["valid_value_count"], 2)

    def test_numeric_columns_are_excluded(self):

        df = pd.DataFrame({
            "sales": [100, 200, 300],
            "category": ["A", "B", "A"]
        })

        result = CategoricalAnalyzer.analyze(df)

        self.assertEqual(
            result["dataset_summary"]["categorical_column_count"],
            1
        )

        self.assertEqual(
            result["columns"][0]["name"],
            "category"
        )

    def test_empty_category_column(self):

        df = pd.DataFrame({
            "category": [None, None, None]
        })

        result = CategoricalAnalyzer.analyze(df)

        column = result["columns"][0]

        self.assertEqual(column["valid_value_count"], 0)
        self.assertEqual(column["unique_category_count"], 0)
        self.assertEqual(column["categories"], [])
        self.assertIsNone(
            column["most_frequent_category"]
        )

    def test_duplicate_column_names(self):

        df = pd.DataFrame(
            [["A", "X"], ["B", "Y"]],
            columns=["category", "category"]
        )

        result = CategoricalAnalyzer.analyze(df)

        self.assertEqual(len(result["columns"]), 2)

        self.assertEqual(
            result["columns"][0]["column_index"], 0
        )

        self.assertEqual(
            result["columns"][1]["column_index"], 1
        )

    def test_invalid_input(self):

        with self.assertRaises(TypeError):
            CategoricalAnalyzer.analyze(["A", "B"])

    def test_original_dataframe_is_not_modified(self):

        df = pd.DataFrame({
            "category": ["A", "B", None, "A"]
        })

        original = df.copy(deep=True)

        CategoricalAnalyzer.analyze(df)

        pd.testing.assert_frame_equal(df, original)


if __name__ == "__main__":
    unittest.main()
