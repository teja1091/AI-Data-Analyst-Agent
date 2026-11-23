
import unittest

from app.data_engine.orchestrator.analysis_result import (
    AnalysisExecution,
    AnalysisResult,
)


class TestAnalysisResult(unittest.TestCase):

    def test_successful_analysis_result(self):
        result = AnalysisResult(
            overall_status="success",
            dataset_summary={
                "total_rows": 100,
                "total_columns": 5
            },
            analyses={
                "descriptive": {
                    "columns": []
                }
            },
            execution={
                "descriptive": AnalysisExecution(
                    status="success"
                )
            }
        )

        self.assertEqual(
            result.overall_status,
            "success"
        )

        self.assertEqual(
            result.dataset_summary["total_rows"],
            100
        )

        self.assertEqual(
            result.execution["descriptive"].status,
            "success"
        )

    def test_partial_failure_result(self):
        result = AnalysisResult(
            overall_status="partial_failure",
            execution={
                "descriptive": AnalysisExecution(
                    status="success"
                ),
                "correlation": AnalysisExecution(
                    status="failed",
                    error="Correlation calculation failed."
                )
            }
        )

        self.assertEqual(
            result.overall_status,
            "partial_failure"
        )

        self.assertEqual(
            result.execution["correlation"].status,
            "failed"
        )

    def test_skipped_analysis(self):
        result = AnalysisResult(
            overall_status="success",
            execution={
                "outlier": AnalysisExecution(
                    status="skipped"
                )
            }
        )

        self.assertEqual(
            result.execution["outlier"].status,
            "skipped"
        )

    def test_invalid_execution_status_is_rejected(self):
        with self.assertRaises(ValueError):
            AnalysisExecution(status="completed")

    def test_invalid_overall_status_is_rejected(self):
        with self.assertRaises(ValueError):
            AnalysisResult(overall_status="completed")


if __name__ == "__main__":
    unittest.main()
