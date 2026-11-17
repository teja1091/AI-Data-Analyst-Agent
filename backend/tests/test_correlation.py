import unittest

import numpy as np
import pandas as pd

from app.data_engine.eda.correlation import CorrelationAnalyzer


class TestCorrelationAnalyzer(unittest.TestCase):

    def setUp(self):
        self.analyzer = CorrelationAnalyzer()

    def test_basic_correlation(self):

        df = pd.DataFrame({
            "sales": [10, 20, 30, 40, 50],
            "profit": [20, 40, 60, 80, 100],
            "temperature": [50, 40, 30, 20, 10]
        })

        result = self.analyzer.analyze(df)

        self.assertEqual(
            result["dataset_summary"]["numeric_column_count"],
            3
        )

        self.assertAlmostEqual(
            result["pearson"][0][1],
            1.0
        )

        self.assertAlmostEqual(
            result["pearson"][0][2],
            -1.0
        )

        self.assertAlmostEqual(
            result["spearman"][0][1],
            1.0
        )

        self.assertEqual(
            result["pair_counts"][0][1],
            5
        )

    def test_spearman_detects_monotonic_relationship(self):

        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5],
            "y": [1, 4, 9, 16, 25]
        })

        result = self.analyzer.analyze(df)

        self.assertAlmostEqual(
            result["spearman"][0][1],
            1.0
        )

        self.assertLess(
            result["pearson"][0][1],
            1.0
        )

    def test_pairwise_missing_values(self):

        df = pd.DataFrame({
            "sales": [10, 20, np.nan, 40, 50],
            "profit": [20, 40, 60, np.nan, 100]
        })

        result = self.analyzer.analyze(df)

        self.assertEqual(
            result["pair_counts"][0][1],
            3
        )

        self.assertAlmostEqual(
            result["pearson"][0][1],
            1.0
        )

    def test_non_finite_values_are_excluded(self):

        df = pd.DataFrame({
            "x": [1, 2, np.inf, 4, 5],
            "y": [2, 4, 6, -np.inf, 10]
        })

        result = self.analyzer.analyze(df)

        self.assertEqual(
            result["pair_counts"][0][1],
            3
        )

        self.assertAlmostEqual(
            result["pearson"][0][1],
            1.0
        )

    def test_constant_column_has_undefined_correlation(self):

        df = pd.DataFrame({
            "sales": [10, 20, 30, 40],
            "discount": [5, 5, 5, 5]
        })

        result = self.analyzer.analyze(df)

        self.assertIsNone(
            result["pearson"][0][1]
        )

        self.assertIsNone(
            result["spearman"][0][1]
        )

        self.assertTrue(
            any("constant" in warning for warning in result["warnings"])
        )

    def test_boolean_and_non_numeric_columns_are_excluded(self):

        df = pd.DataFrame({
            "sales": [10, 20, 30],
            "active": [True, False, True],
            "category": ["A", "B", "A"]
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
                [1, 2],
                [2, 4],
                [3, 6]
            ],
            columns=["value", "value"]
        )

        result = self.analyzer.analyze(df)

        self.assertEqual(
            result["dataset_summary"]["numeric_column_count"],
            2
        )

        self.assertEqual(
            result["columns"][0]["column_index"],
            0
        )

        self.assertEqual(
            result["columns"][1]["column_index"],
            1
        )

        self.assertAlmostEqual(
            result["pearson"][0][1],
            1.0
        )

    def test_no_numeric_columns(self):

        df = pd.DataFrame({
            "name": ["A", "B", "C"],
            "category": ["X", "Y", "Z"]
        })

        result = self.analyzer.analyze(df)

        self.assertEqual(
            result["pearson"],
            []
        )

        self.assertEqual(
            result["spearman"],
            []
        )

        self.assertEqual(
            result["pair_counts"],
            []
        )

        self.assertEqual(
            result["dataset_summary"]["numeric_column_count"],
            0
        )

    def test_empty_dataframe(self):

        df = pd.DataFrame()

        result = self.analyzer.analyze(df)

        self.assertEqual(
            result["dataset_summary"]["total_rows"],
            0
        )

        self.assertEqual(
            result["columns"],
            []
        )

    def test_invalid_input(self):

        with self.assertRaises(TypeError):
            self.analyzer.analyze([1, 2, 3])

    def test_original_dataframe_is_not_modified(self):

        df = pd.DataFrame({
            "sales": [10, 20, np.nan, 40],
            "profit": [20, 40, 60, 80]
        })

        original = df.copy(deep=True)

        self.analyzer.analyze(df)

        pd.testing.assert_frame_equal(df, original)


if __name__ == "__main__":
    unittest.main()