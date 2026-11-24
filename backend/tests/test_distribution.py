
import unittest

import numpy as np
import pandas as pd

from app.data_engine.eda.distribution import DistributionAnalyzer


class TestDistributionAnalyzer(unittest.TestCase):

    def setUp(self):
        self.analyzer = DistributionAnalyzer()

    def test_basic_distribution_analysis(self):

        df = pd.DataFrame({
            "sales": [10, 20, 30, 40, 50, 60]
        })

        result = self.analyzer.analyze(df)

        column = result["columns"][0]

        self.assertEqual(
            column["valid_value_count"],
            6
        )

        self.assertEqual(
            column["percentiles"]["p50"],
            35.0
        )

        self.assertEqual(
            column["statistics"]["minimum"],
            10.0
        )

        self.assertEqual(
            column["statistics"]["maximum"],
            60.0
        )

        self.assertIsNotNone(column["skewness"])
        self.assertIsNotNone(column["excess_kurtosis"])

    def test_positive_skewness(self):

        df = pd.DataFrame({
            "sales": [
                10, 11, 12, 13, 14,
                15, 16, 17, 18, 100
            ]
        })

        result = self.analyzer.analyze(df)

        column = result["columns"][0]

        self.assertGreater(column["skewness"], 0)

        self.assertEqual(
            column["skewness_interpretation"],
            "highly_positive_skew"
        )

    def test_negative_skewness(self):

        df = pd.DataFrame({
            "scores": [
                0, 83, 84, 85, 86,
                87, 88, 89, 90, 91
            ]
        })

        result = self.analyzer.analyze(df)

        column = result["columns"][0]

        self.assertLess(column["skewness"], 0)

        self.assertEqual(
            column["skewness_interpretation"],
            "highly_negative_skew"
        )

    def test_approximately_symmetric_distribution(self):

        df = pd.DataFrame({
            "values": [-3, -2, -1, 0, 1, 2, 3]
        })

        result = self.analyzer.analyze(df)

        column = result["columns"][0]

        self.assertAlmostEqual(
            column["skewness"],
            0.0
        )

        self.assertEqual(
            column["skewness_interpretation"],
            "approximately_symmetric"
        )

    def test_histogram_counts_match_valid_values(self):

        df = pd.DataFrame({
            "sales": [10, 20, 30, 40, 50, 60]
        })

        result = self.analyzer.analyze(
            df,
            bins=3
        )

        histogram = result["columns"][0]["histogram"]

        self.assertEqual(
            len(histogram["counts"]),
            3
        )

        self.assertEqual(
            sum(histogram["counts"]),
            6
        )

        self.assertEqual(
            len(histogram["bin_edges"]),
            4
        )

    def test_custom_histogram_bins(self):

        df = pd.DataFrame({
            "sales": [10, 20, 30, 40, 50]
        })

        result = self.analyzer.analyze(
            df,
            bins=5
        )

        self.assertEqual(
            result["configuration"]["histogram_bins"],
            5
        )

        self.assertEqual(
            len(result["columns"][0]["histogram"]["counts"]),
            5
        )

    def test_missing_values_are_excluded(self):

        df = pd.DataFrame({
            "sales": [10, 20, np.nan, 40, 50]
        })

        result = self.analyzer.analyze(df)

        column = result["columns"][0]

        self.assertEqual(column["missing_count"], 1)
        self.assertEqual(column["valid_value_count"], 4)

        self.assertEqual(
            sum(column["histogram"]["counts"]),
            4
        )

    def test_non_finite_values_are_excluded(self):

        df = pd.DataFrame({
            "sales": [10, 20, np.inf, 40, 50]
        })

        result = self.analyzer.analyze(df)

        column = result["columns"][0]

        self.assertEqual(column["non_finite_count"], 1)
        self.assertEqual(column["valid_value_count"], 4)

        self.assertEqual(
            sum(column["histogram"]["counts"]),
            4
        )

    def test_empty_numeric_column(self):

        df = pd.DataFrame({
            "sales": [np.nan, np.nan, np.nan]
        })

        result = self.analyzer.analyze(df)

        column = result["columns"][0]

        self.assertEqual(column["valid_value_count"], 0)
        self.assertIsNone(column["statistics"])
        self.assertIsNone(column["percentiles"])
        self.assertIsNone(column["histogram"])
        self.assertIsNone(column["skewness"])
        self.assertIsNone(column["excess_kurtosis"])

    def test_small_sample_size(self):

        df = pd.DataFrame({
            "sales": [10, 20, 30]
        })

        result = self.analyzer.analyze(df)

        column = result["columns"][0]

        self.assertIsNotNone(column["skewness"])
        self.assertIsNone(column["excess_kurtosis"])

    def test_constant_column(self):

        df = pd.DataFrame({
            "discount": [5, 5, 5, 5, 5]
        })

        result = self.analyzer.analyze(df)

        column = result["columns"][0]

        self.assertEqual(
            column["unique_value_count"],
            1
        )

        self.assertEqual(
            column["skewness_interpretation"],
            "unavailable"
        )

        self.assertEqual(
            column["kurtosis_interpretation"],
            "unavailable"
        )

    def test_boolean_and_non_numeric_columns_are_excluded(self):

        df = pd.DataFrame({
            "sales": [10, 20, 30, 40],
            "active": [True, False, True, False],
            "category": ["A", "B", "C", "D"]
        })

        result = self.analyzer.analyze(df)

        self.assertEqual(
            result["dataset_summary"]["numeric_column_count"],
            1
        )

        self.assertEqual(
            result["columns"][0]["name"],
            "sales"
        )

    def test_duplicate_column_names(self):

        df = pd.DataFrame(
            [
                [10, 100],
                [20, 200],
                [30, 300],
                [40, 400]
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

    def test_no_numeric_columns(self):

        df = pd.DataFrame({
            "name": ["A", "B", "C"],
            "category": ["X", "Y", "Z"]
        })

        result = self.analyzer.analyze(df)

        self.assertEqual(result["columns"], [])

        self.assertEqual(
            result["dataset_summary"]["numeric_column_count"],
            0
        )

    def test_invalid_input(self):

        with self.assertRaises(TypeError):
            self.analyzer.analyze([1, 2, 3])

    def test_invalid_bins_type(self):

        with self.assertRaises(TypeError):
            self.analyzer.analyze(
                pd.DataFrame({"x": [1, 2, 3]}),
                bins=2.5
            )

    def test_boolean_bins_are_invalid(self):

        with self.assertRaises(TypeError):
            self.analyzer.analyze(
                pd.DataFrame({"x": [1, 2, 3]}),
                bins=True
            )

    def test_zero_bins_are_invalid(self):

        with self.assertRaises(ValueError):
            self.analyzer.analyze(
                pd.DataFrame({"x": [1, 2, 3]}),
                bins=0
            )

    def test_original_dataframe_is_not_modified(self):

        df = pd.DataFrame({
            "sales": [10, 20, np.nan, 40, 100]
        })

        original = df.copy(deep=True)

        self.analyzer.analyze(df)

        pd.testing.assert_frame_equal(df, original)


if __name__ == "__main__":
    unittest.main()
