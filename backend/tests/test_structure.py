
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
            column["name"]: column["analytical_type"]
            for column in result["columns"]
        }

        self.assertEqual(types["Product"], "categorical_or_text")
        self.assertEqual(types["Quantity"], "numeric")
        self.assertEqual(types["Revenue"], "numeric")
        self.assertEqual(types["Is_Active"], "boolean")
        self.assertEqual(types["Order_Date"], "datetime")

    def test_empty_dataframe(self):
        result = StructureAnalyzer.analyze(pd.DataFrame())

        self.assertEqual(result["row_count"], 0)
        self.assertEqual(result["column_count"], 0)

    def test_invalid_input(self):
        with self.assertRaises(TypeError):
            StructureAnalyzer.analyze(["Apple", "Banana"])


if __name__ == "__main__":
    unittest.main()
