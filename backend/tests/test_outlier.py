
import unittest

import numpy as np
import pandas as pd

from app.data_engine.eda.outlier import OutlierAnalyzer


class TestOutlierAnalyzer(unittest.TestCase):

    def setUp(self):
        self.analyzer = OutlierAnalyzer()

    def test_iqr_detects_outlier(self):

        df = pd.DataFrame({
            "sales": [100, 110, 120, 130, 900]
        })

        result = self.analyzer.analyze(df)

        column = result["columns"][0]

        self.assertEqual(column["iqr_outlier_count"], 1)
        self.assertEqual(column["iqr_outlier_indices"], [4])

    def test_zscore_detects_extreme_outlier(self):

        df = pd.DataFrame({
            "sales": [
                100, 101, 102, 103, 104,
                105, 106, 107, 108, 109,
                1000
            ]
        })

        result = self.analyzer.analyze(df)

        column = result["columns"][0]

        self.assertEqual(column["zscore_outlier_count"], 1)
        self.assertEqual(column["zscore_outlier_indices"], [10])

    def test_custom_thresholds(self):

        df = pd.DataFrame({
            "sales": [10, 20, 30, 40, 100]
        })

        result = self.analyzer.analyze(
            df,
            iqr_multiplier=1.0,
            zscore_threshold=1.0
        )

        self.assertEqual(
            result["configuration"]["iqr_multiplier"],
            1.0
        )

        self.assertEqual(
            result["configuration"]["zscore_threshold"],
            1.0
        )

    def test_missing_values_are_excluded(self):

        df = pd.DataFrame({
            "sales": [100, 110, np.nan, 130, 900]
        })

        result = self.analyzer.analyze(df)

        column = result["columns"][0]

        self.assertEqual(column["missing_count"], 1)
        self.assertEqual(column["valid_value_count"], 4)

    def test_non_finite_values_are_excluded(self):

        df = pd.DataFrame({
            "sales": [100, 110, np.inf, 130, 900]
        })

        result = self.analyzer.analyze(df)

        column = result["columns"][0]

        self.assertEqual(column["non_finite_count"], 1)
        self.assertEqual(column["valid_value_count"], 4)

    def test_constant_column(self):

        df = pd.DataFrame({
            "discount": [5, 5, 5, 5, 5]
        })

        result = self.analyzer.analyze(df)

        column = result["columns"][0]

        self.assertEqual(column["iqr_outlier_count"], 0)
        self.assertEqual(column["zscore_outlier_count"], 0)

        self.assertTrue(
            any("zero variance" in warning for warning in result["warnings"])
        )

    def test_empty_numeric_column(self):

        df = pd.DataFrame({
            "sales": [np.nan, np.nan, np.nan]
        })

        result = self.analyzer.analyze(df)

        column = result["columns"][0]

        self.assertEqual(column["valid_value_count"], 0)
        self.assertIsNone(column["statistics"])

    def test_boolean_and_non_numeric_columns_are_excluded(self):

        df = pd.DataFrame({
            "sales": [10, 20, 30],
            "active": [True, False, True],
            "category": ["A", "B", "C"]
        })

        result = self.analyzer.analyze(df)

        self.assertEqual(
            result["dataset_summary"]["numeric_column_count"],
            1
        )

    def test_duplicate_column_names(self):

        df = pd.DataFrame(
            [
                [10, 100],
                [20, 200],
                [30, 300],
                [40, 400],
                [900, 9000]
            ],
            columns=["value", "value"]
        )

        result = self.analyzer.analyze(df)

        self.assertEqual(len(result["columns"]), 2)

        self.assertEqual(
            result["columns"][0]["column_index"],
            0
        )

        self.assertEqual(
            result["columns"][1]["column_index"],
            1
        )

    def test_invalid_input(self):

        with self.assertRaises(TypeError):
            self.analyzer.analyze([1, 2, 3])

    def test_invalid_iqr_multiplier(self):

        with self.assertRaises(ValueError):
            self.analyzer.analyze(
                pd.DataFrame({"x": [1, 2, 3]}),
                iqr_multiplier=0
            )

    def test_invalid_zscore_threshold(self):

        with self.assertRaises(ValueError):
            self.analyzer.analyze(
                pd.DataFrame({"x": [1, 2, 3]}),
                zscore_threshold=-1
            )

    def test_boolean_threshold_is_invalid(self):

        with self.assertRaises(TypeError):
            self.analyzer.analyze(
                pd.DataFrame({"x": [1, 2, 3]}),
                iqr_multiplier=True
            )

    def test_original_dataframe_is_not_modified(self):

        df = pd.DataFrame({
            "sales": [100, 110, np.nan, 130, 900]
        })

        original = df.copy(deep=True)

        self.analyzer.analyze(df)

        pd.testing.assert_frame_equal(df, original)
    def test_custom_dataframe_index_returns_row_positions(self):

        df = pd.DataFrame(
            {
                "sales": [100, 110, 120, 130, 900]
            },
            index=[100, 200, 300, 400, 500]
        )

        result = self.analyzer.analyze(df)
        column = result["columns"][0]

        # Returned indices represent zero-based row positions,
        # not the original DataFrame index labels.
        self.assertEqual(
            column["iqr_outlier_indices"],
            [4]
        )

        self.assertNotEqual(
            column["iqr_outlier_indices"],
            [500]
        )

    def test_nullable_integer_column(self):

        df = pd.DataFrame({
            "sales": pd.Series(
                [100, 110, 120, 130, 900],
                dtype="Int64"
            )
        })

        result = self.analyzer.analyze(df)
        column = result["columns"][0]

        self.assertEqual(
            column["valid_value_count"],
            5
        )

        self.assertEqual(
            column["iqr_outlier_count"],
            1
        )

        self.assertEqual(
            column["iqr_outlier_indices"],
            [4]
        )

    def test_single_observation_has_no_outliers(self):

        df = pd.DataFrame({
            "sales": [100]
        })

        result = self.analyzer.analyze(df)
        column = result["columns"][0]

        self.assertEqual(
            column["valid_value_count"],
            1
        )

        self.assertEqual(
            column["iqr_outlier_count"],
            0
        )

        self.assertEqual(
            column["zscore_outlier_count"],
            0
        )

        self.assertTrue(
            any(
                "zero variance" in warning
                for warning in result["warnings"]
            )
        )



if __name__ == "__main__":
    unittest.main()
