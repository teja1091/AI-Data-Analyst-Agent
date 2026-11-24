
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

        self.assertEqual(column["unique_category_count"], 2)
        self.assertEqual(column["valid_value_count"], 5)

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
                "A", "B", None, np.nan, "", "   "
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

        self.assertEqual(result["columns"][0]["name"], "category")

    def test_empty_category_column(self):

        df = pd.DataFrame({
            "category": [None, None, None]
        })

        result = CategoricalAnalyzer.analyze(df)
        column = result["columns"][0]

        self.assertEqual(column["valid_value_count"], 0)
        self.assertEqual(column["unique_category_count"], 0)
        self.assertEqual(column["categories"], [])
        self.assertEqual(column["total_category_count"], 0)
        self.assertEqual(column["returned_category_count"], 0)
        self.assertEqual(column["omitted_category_count"], 0)
        self.assertFalse(column["categories_truncated"])

        self.assertIsNone(column["most_frequent_category"])
        self.assertIsNone(column["least_frequent_category"])

    def test_duplicate_column_names(self):

        df = pd.DataFrame(
            [["A", "X"], ["B", "Y"]],
            columns=["category", "category"]
        )

        result = CategoricalAnalyzer.analyze(df)

        self.assertEqual(len(result["columns"]), 2)
        self.assertEqual(result["columns"][0]["column_index"], 0)
        self.assertEqual(result["columns"][1]["column_index"], 1)

    def test_boolean_columns_are_included(self):

        df = pd.DataFrame({
            "active": [True, False, True, True, False]
        })

        result = CategoricalAnalyzer.analyze(df)
        column = result["columns"][0]

        self.assertEqual(
            result["dataset_summary"]["categorical_column_count"],
            1
        )

        self.assertEqual(column["valid_value_count"], 5)
        self.assertEqual(column["unique_category_count"], 2)
        self.assertEqual(column["most_frequent_category"], "True")
        self.assertEqual(column["least_frequent_category"], "False")

    def test_pandas_string_dtype_with_missing_values(self):

        df = pd.DataFrame({
            "category": pd.Series(
                ["A", "B", pd.NA, "A"],
                dtype="string"
            )
        })

        result = CategoricalAnalyzer.analyze(df)
        column = result["columns"][0]

        self.assertEqual(column["valid_value_count"], 3)
        self.assertEqual(column["missing_count"], 1)
        self.assertEqual(column["unique_category_count"], 2)
        self.assertEqual(column["most_frequent_category"], "A")

    def test_pandas_category_dtype(self):

        df = pd.DataFrame({
            "membership": pd.Series(
                ["Gold", "Silver", "Gold", "Bronze"],
                dtype="category"
            )
        })

        result = CategoricalAnalyzer.analyze(df)
        column = result["columns"][0]

        self.assertEqual(column["valid_value_count"], 4)
        self.assertEqual(column["unique_category_count"], 3)
        self.assertEqual(column["most_frequent_category"], "Gold")

    def test_category_percentages_sum_to_100(self):

        df = pd.DataFrame({
            "membership": [
                "Gold", "Silver", "Gold",
                "Bronze", "Silver", "Gold"
            ]
        })

        result = CategoricalAnalyzer.analyze(df)
        column = result["columns"][0]

        total_percentage = sum(
            category["percentage"]
            for category in column["categories"]
        )

        self.assertAlmostEqual(total_percentage, 100.0, places=2)

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

    # New response-limit tests

    def test_default_category_limit_is_20(self):

        df = pd.DataFrame({
            "category": [f"Category_{i}" for i in range(30)]
        })

        result = CategoricalAnalyzer.analyze(df)
        column = result["columns"][0]

        self.assertEqual(len(column["categories"]), 20)
        self.assertEqual(column["total_category_count"], 30)
        self.assertEqual(column["returned_category_count"], 20)
        self.assertEqual(column["omitted_category_count"], 10)
        self.assertTrue(column["categories_truncated"])

    def test_custom_category_limit(self):

        df = pd.DataFrame({
            "category": [
                "A", "A", "A",
                "B", "B",
                "C", "C",
                "D"
            ]
        })

        result = CategoricalAnalyzer.analyze(
            df,
            max_categories=2
        )

        column = result["columns"][0]

        self.assertEqual(len(column["categories"]), 2)
        self.assertEqual(column["categories"][0]["value"], "A")
        self.assertEqual(column["categories"][1]["value"], "B")
        self.assertEqual(column["returned_category_count"], 2)
        self.assertEqual(column["omitted_category_count"], 2)
        self.assertTrue(column["categories_truncated"])

    def test_most_and_least_frequent_use_full_distribution(self):

        df = pd.DataFrame({
            "category": [
                "A", "A", "A", "A",
                "B", "B",
                "C",
                "D"
            ]
        })

        result = CategoricalAnalyzer.analyze(
            df,
            max_categories=2
        )

        column = result["columns"][0]

        self.assertEqual(column["most_frequent_category"], "A")
        self.assertEqual(column["least_frequent_category"], "D")

        returned_values = [
            category["value"]
            for category in column["categories"]
        ]

        self.assertNotIn(
            column["least_frequent_category"],
            returned_values
        )

    def test_no_truncation_when_categories_fit_limit(self):

        df = pd.DataFrame({
            "category": ["A", "B", "A", "C"]
        })

        result = CategoricalAnalyzer.analyze(
            df,
            max_categories=5
        )

        column = result["columns"][0]

        self.assertEqual(column["total_category_count"], 3)
        self.assertEqual(column["returned_category_count"], 3)
        self.assertEqual(column["omitted_category_count"], 0)
        self.assertFalse(column["categories_truncated"])

    def test_none_returns_all_categories(self):

        df = pd.DataFrame({
            "category": [f"Category_{i}" for i in range(30)]
        })

        result = CategoricalAnalyzer.analyze(
            df,
            max_categories=None
        )

        column = result["columns"][0]

        self.assertEqual(len(column["categories"]), 30)
        self.assertEqual(column["returned_category_count"], 30)
        self.assertEqual(column["omitted_category_count"], 0)
        self.assertFalse(column["categories_truncated"])

    def test_invalid_category_limit(self):

        df = pd.DataFrame({
            "category": ["A", "B", "C"]
        })

        for invalid_limit in [0, -1]:
            with self.subTest(limit=invalid_limit):
                with self.assertRaises(ValueError):
                    CategoricalAnalyzer.analyze(
                        df,
                        max_categories=invalid_limit
                    )

    def test_boolean_category_limit_is_invalid(self):

        df = pd.DataFrame({
            "category": ["A", "B", "C"]
        })

        with self.assertRaises(TypeError):
            CategoricalAnalyzer.analyze(
                df,
                max_categories=True
            )

    def test_non_integer_category_limit_is_invalid(self):

        df = pd.DataFrame({
            "category": ["A", "B", "C"]
        })

        with self.assertRaises(TypeError):
            CategoricalAnalyzer.analyze(
                df,
                max_categories=2.5
            )


if __name__ == "__main__":
    unittest.main()
