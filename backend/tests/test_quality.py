
import unittest
import pandas as pd

from app.data_engine.quality import QualityAnalyzer


class TestQualityAnalyzer(unittest.TestCase):

    def setUp(self):

        self.df = pd.DataFrame({
            "Product": [
                "Apple", "Banana", "Apple", None, ""
            ],
            "Quantity": [
                10, 20, 10, None, 30
            ],
            "Revenue": [
                100, 200, 100, 300, 400
            ],
            "Country": [
                "India", "India", "India", "India", "India"
            ],
            "Empty_Column": [
                None, None, None, None, None
            ]
        })

    def test_missing_and_blank_values(self):

        result = QualityAnalyzer.analyze_missing_values(
            self.df
        )

        columns = {
            column["name"]: column
            for column in result["columns"]
        }

        self.assertEqual(
            columns["Product"]["missing_count"], 1
        )

        self.assertEqual(
            columns["Product"]["blank_count"], 1
        )

        self.assertEqual(
            columns["Quantity"]["missing_count"], 1
        )

    def test_duplicate_rows(self):

        result = QualityAnalyzer.analyze_duplicates(
            self.df
        )

        self.assertEqual(
            result["duplicate_rows"], 1
        )

    def test_empty_columns(self):

        result = QualityAnalyzer.identify_empty_columns(
            self.df
        )

        self.assertEqual(
            result[0]["name"], "Empty_Column"
        )

    def test_constant_columns(self):

        result = QualityAnalyzer.identify_constant_columns(
            self.df
        )

        names = [
            column["name"]
            for column in result
        ]

        self.assertIn("Country", names)
        self.assertNotIn("Empty_Column", names)

    def test_unique_value_analysis(self):

        result = QualityAnalyzer.analyze_unique_values(
            self.df
        )

        columns = {
            column["name"]: column["unique_count"]
            for column in result
        }

        self.assertEqual(columns["Product"], 2)
        self.assertEqual(columns["Country"], 1)

    def test_complete_quality_report(self):

        result = QualityAnalyzer.analyze(self.df)

        self.assertIn("dataset_summary", result)
        self.assertIn("quality_summary", result)
        self.assertIn("warnings", result)

        self.assertEqual(
            result["dataset_summary"]["total_rows"], 5
        )

    def test_empty_dataframe(self):

        result = QualityAnalyzer.analyze(
            pd.DataFrame()
        )

        self.assertEqual(
            result["quality_summary"]["total_missing_and_blank_cells"], 0
        )

        self.assertEqual(
            result["duplicate_analysis"]["duplicate_rows"], 0
        )

    def test_invalid_input(self):

        with self.assertRaises(TypeError):
            QualityAnalyzer.analyze(
                ["Apple", "Banana"]
            )

    
    def test_dominant_value_detection(self):

        df = pd.DataFrame({
            "Country": [
                "India", "India", "India", "India", "USA",
                "India", "India", "India", "India", "India"
            ],
            "Order_ID": list(range(101, 111)),
            "Category": [
                "A", "A", "A", "B", "A",
                "A", "A", "A", "A", "A"
            ]
        })

        result = QualityAnalyzer.analyze_value_distribution(df)

        names = [
            column["name"]
            for column in result
        ]

        self.assertIn("Country", names)
        self.assertIn("Category", names)
        self.assertNotIn("Order_ID", names)

    def test_dominant_threshold(self):

        df = pd.DataFrame({
            "Country": [
                "India", "India", "India", "India", "USA"
            ]
        })

        result = QualityAnalyzer.analyze_value_distribution(
            df,
            threshold=0.80
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(
            result[0]["dominant_percentage"], 80.0
        )

    def test_custom_dominant_threshold(self):

        df = pd.DataFrame({
            "Country": [
                "India", "India", "India", "India", "USA"
            ]
        })

        result = QualityAnalyzer.analyze_value_distribution(
            df,
            threshold=0.90
        )

        self.assertEqual(len(result), 0)

    def test_invalid_dominant_threshold(self):

        df = pd.DataFrame({
            "Country": ["India", "USA"]
        })

        with self.assertRaises(ValueError):
            QualityAnalyzer.analyze_value_distribution(
                df,
                threshold=1.5
            )
    
    # --------------------------------------------------
    # EDGE CASE TESTS
    # --------------------------------------------------

    def test_multiple_missing_value_types(self):

        df = pd.DataFrame({
            "Values": pd.Series(
                ["Apple", None, float("nan"), pd.NA],
                dtype="object"
            )
        })

        result = QualityAnalyzer.analyze_missing_values(df)

        column = result["columns"][0]

        self.assertEqual(column["missing_count"], 3)
        self.assertEqual(column["valid_value_count"], 1)

    def test_whitespace_only_values(self):

        df = pd.DataFrame({
            "Category": ["Apple", "", "   ", "Banana"]
        })

        result = QualityAnalyzer.analyze_missing_values(df)

        column = result["columns"][0]

        self.assertEqual(column["blank_count"], 2)
        self.assertEqual(column["missing_count"], 0)
        self.assertEqual(column["valid_value_count"], 2)

    def test_duplicate_column_names(self):

        df = pd.DataFrame(
            [
                ["Apple", None],
                ["Banana", "Orange"]
            ],
            columns=["Product", "Product"]
        )

        result = QualityAnalyzer.analyze_missing_values(df)

        self.assertEqual(len(result["columns"]), 2)

        self.assertEqual(
            result["columns"][0]["missing_count"], 0
        )

        self.assertEqual(
            result["columns"][1]["missing_count"], 1
        )

        self.assertEqual(
            result["columns"][0]["column_index"], 0
        )

        self.assertEqual(
            result["columns"][1]["column_index"], 1
        )

    def test_zero_rows_with_columns(self):

        df = pd.DataFrame({
            "Product": pd.Series(dtype="object"),
            "Quantity": pd.Series(dtype="float64")
        })

        result = QualityAnalyzer.analyze(df)

        self.assertEqual(
            result["dataset_summary"]["total_rows"], 0
        )

        self.assertEqual(
            result["dataset_summary"]["total_columns"], 2
        )

        self.assertEqual(
            result["empty_columns"][0]["name"], "Product"
        )

        self.assertEqual(len(result["empty_columns"]), 2)

        self.assertEqual(
            result["quality_summary"][
                "missing_and_blank_percentage"
            ], 0.0
        )

    def test_rows_without_columns(self):

        df = pd.DataFrame(index=range(3))

        result = QualityAnalyzer.analyze(df)

        self.assertEqual(
            result["dataset_summary"]["total_rows"], 3
        )

        self.assertEqual(
            result["dataset_summary"]["total_columns"], 0
        )

        self.assertEqual(
            result["dataset_summary"]["total_cells"], 0
        )

        self.assertEqual(result["empty_columns"], [])
        self.assertEqual(result["constant_columns"], [])

    def test_all_missing_column_is_not_constant(self):

        df = pd.DataFrame({
            "Empty": [None, None, None],
            "Country": ["India", "India", "India"]
        })

        empty_columns = QualityAnalyzer.identify_empty_columns(df)

        constant_columns = QualityAnalyzer.identify_constant_columns(df)

        empty_names = [
            column["name"] for column in empty_columns
        ]

        constant_names = [
            column["name"] for column in constant_columns
        ]

        self.assertIn("Empty", empty_names)
        self.assertNotIn("Empty", constant_names)

        self.assertIn("Country", constant_names)

    def test_dominant_analysis_with_missing_values(self):

        df = pd.DataFrame({
            "Country": [
                "India", "India", "India", "India",
                None, "", "USA", "India", "India", "India"
            ]
        })

        result = QualityAnalyzer.analyze_value_distribution(df)

        self.assertEqual(len(result), 1)

        self.assertEqual(
            result[0]["total_valid_values"], 8
        )

        self.assertEqual(
            result[0]["dominant_count"], 7
        )

        self.assertEqual(
            result[0]["dominant_percentage"], 87.5
        )

    def test_original_dataframe_is_not_modified(self):

        df = pd.DataFrame({
            "Product": ["Apple", None, "Banana", ""],
            "Quantity": [10, 20, 30, 40]
        })

        original_df = df.copy(deep=True)

        QualityAnalyzer.analyze(df)

        pd.testing.assert_frame_equal(df, original_df)

    def test_duplicate_rows_containing_missing_values(self):

        df = pd.DataFrame({
            "Product": ["Apple", "Apple", "Banana"],
            "Quantity": [10, 10, 20],
            "Discount": [None, None, 5]
        })

        result = QualityAnalyzer.analyze_duplicates(df)

        self.assertEqual(result["duplicate_rows"], 1)
        self.assertEqual(result["unique_rows"], 2)

    def test_non_finite_dominant_threshold(self):

        df = pd.DataFrame({
            "Country": ["India", "USA"]
        })

        with self.assertRaises(ValueError):
            QualityAnalyzer.analyze_value_distribution(
                df,
                threshold=float("nan")
            )

        with self.assertRaises(ValueError):
            QualityAnalyzer.analyze_value_distribution(
                df,
                threshold=float("inf")
            )

    def test_boolean_threshold_is_invalid(self):

        df = pd.DataFrame({
            "Country": ["India", "USA"]
        })

        with self.assertRaises(TypeError):
            QualityAnalyzer.analyze_value_distribution(
                df,
                threshold=True
            )


if __name__ == "__main__":
    unittest.main()
