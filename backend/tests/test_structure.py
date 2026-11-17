
import unittest
import pandas as pd

from app.data_engine.structure import StructureAnalyzer


class TestStructureAnalyzer(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            "Product": ["Apple", "Banana", "Orange"],
            "Quantity": [10, 15, 8],
            "Revenue": [200.5, 150.0, 120.0],
            "Is_Active": [True, False, True],
            "Order_Date": pd.to_datetime([
                "2026-01-01",
                "2026-01-02",
                "2026-01-03"
            ])
        })

    def test_row_and_column_count(self):
        result = StructureAnalyzer.analyze(self.df)

        self.assertEqual(result["row_count"], 3)
        self.assertEqual(result["column_count"], 5)

    def test_column_classification(self):
        result = StructureAnalyzer.analyze(self.df)

        types = {
            (column["column_index"], column["name"]):
            column["analytical_type"]
            for column in result["columns"]
        }

        self.assertEqual(types[(0, "Product")], "categorical_or_text")
        self.assertEqual(types[(1, "Quantity")], "numeric")
        self.assertEqual(types[(2, "Revenue")], "numeric")
        self.assertEqual(types[(3, "Is_Active")], "boolean")
        self.assertEqual(types[(4, "Order_Date")], "datetime")

    def test_empty_dataframe(self):
        result = StructureAnalyzer.analyze(pd.DataFrame())

        self.assertEqual(result["row_count"], 0)
        self.assertEqual(result["column_count"], 0)
        self.assertEqual(result["columns"], [])

    def test_invalid_input(self):
        with self.assertRaises(TypeError):
            StructureAnalyzer.analyze(["Apple", "Banana"])

    def test_duplicate_column_names(self):
        df = pd.DataFrame(
            [[100, "Food"], [200, "Travel"]],
            columns=["category", "category"]
        )

        result = StructureAnalyzer.analyze(df)

        self.assertEqual(result["column_count"], 2)
        self.assertEqual(result["columns"][0]["column_index"], 0)
        self.assertEqual(result["columns"][1]["column_index"], 1)
        self.assertEqual(result["columns"][0]["name"], "category")
        self.assertEqual(result["columns"][1]["name"], "category")
        self.assertEqual(
            result["columns"][0]["analytical_type"], "numeric"
        )
        self.assertEqual(
            result["columns"][1]["analytical_type"], "categorical_or_text"
        )

    def test_complex_numeric_column(self):
        series = pd.Series([1 + 2j, 3 + 4j])

        result = StructureAnalyzer.classify_column(series)

        self.assertEqual(result, "complex")

    def test_nullable_integer_column(self):
        series = pd.Series([10, 20, None], dtype="Int64")

        result = StructureAnalyzer.classify_column(series)

        self.assertEqual(result, "numeric")

    def test_nullable_boolean_column(self):
        series = pd.Series([True, False, None], dtype="boolean")

        result = StructureAnalyzer.classify_column(series)

        self.assertEqual(result, "boolean")

    def test_timezone_aware_datetime(self):
        series = pd.Series(
            pd.to_datetime([
                "2026-01-01",
                "2026-01-02"
            ]).tz_localize("UTC")
        )

        result = StructureAnalyzer.classify_column(series)

        self.assertEqual(result, "datetime")

    def test_timedelta_column(self):
        series = pd.Series(
            pd.to_timedelta([1, 2, 3], unit="D")
        )

        result = StructureAnalyzer.classify_column(series)

        self.assertEqual(result, "timedelta")

    def test_invalid_series_input(self):
        with self.assertRaises(TypeError):
            StructureAnalyzer.classify_column([10, 20, 30])

    def test_original_dataframe_is_not_modified(self):
        original = self.df.copy(deep=True)

        StructureAnalyzer.analyze(self.df)

        pd.testing.assert_frame_equal(self.df, original)


if __name__ == "__main__":
    unittest.main()
