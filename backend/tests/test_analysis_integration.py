
import unittest

import pandas as pd

from app.data_engine.orchestrator.analysis_orchestrator import (
    AnalysisOrchestrator,
)


class TestAnalysisIntegration(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            "revenue": [
                120, 150, 180, 200, 220,
                250, 280, 300, 350, 400
            ],
            "units_sold": [
                10, 12, 15, 17, 19,
                21, 24, 25, 29, 32
            ],
            "category": [
                "A", "B", "A", "C", "B",
                "A", "C", "B", "A", "C"
            ],
        })

        self.orchestrator = AnalysisOrchestrator()

    def test_all_real_analyzers_execute_successfully(self):
        result = self.orchestrator.analyze(self.df)

        self.assertEqual(result.overall_status, "success")

        expected_analyses = {
            "structure",
            "quality",
            "descriptive",
            "distribution",
            "categorical",
            "correlation",
            "outlier",
        }

        self.assertEqual(
            set(result.analyses.keys()),
            expected_analyses,
        )

        for name in expected_analyses:
            self.assertEqual(
                result.execution[name].status,
                "success",
                f"{name} analyzer did not succeed.",
            )

    def test_dataset_summary_is_correct(self):
        result = self.orchestrator.analyze(self.df)

        self.assertEqual(
            result.dataset_summary["total_rows"],
            10,
        )

        self.assertEqual(
            result.dataset_summary["total_columns"],
            3,
        )

    def test_analysis_results_are_dictionaries(self):
        result = self.orchestrator.analyze(self.df)

        for name, output in result.analyses.items():
            self.assertIsInstance(
                output,
                dict,
                f"{name} output should be a dictionary.",
            )

    def test_selective_real_analysis(self):
        from app.data_engine.orchestrator.analysis_config import (
            AnalysisConfig,
        )

        config = AnalysisConfig(
            run_distribution=False,
            run_categorical=False,
            run_correlation=False,
            run_outlier=False,
        )

        result = self.orchestrator.analyze(
            self.df,
            config=config,
        )

        self.assertEqual(result.overall_status, "success")

        self.assertEqual(
            result.execution["descriptive"].status,
            "success",
        )

        self.assertEqual(
            result.execution["distribution"].status,
            "skipped",
        )

        self.assertNotIn(
            "distribution",
            result.analyses,
        )


if __name__ == "__main__":
    unittest.main()
