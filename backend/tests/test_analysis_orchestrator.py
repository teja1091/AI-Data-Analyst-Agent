
import unittest
from unittest.mock import Mock
from unittest.mock import Mock
import pandas as pd

from app.data_engine.orchestrator.analysis_config import AnalysisConfig
from app.data_engine.orchestrator.analysis_orchestrator import (
    AnalysisOrchestrator,
)


class TestAnalysisOrchestrator(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            "sales": [100, 200, 300],
            "category": ["A", "B", "A"],
        })

        self.analyzers = {
            name: Mock()
            for name in (
                "structure",
                "quality",
                "descriptive",
                "distribution",
                "categorical",
                "correlation",
                "outlier",
            )
        }

        for name, analyzer in self.analyzers.items():
            analyzer.analyze.return_value = {
                "analyzer": name,
                "warnings": [],
            }

        self.orchestrator = AnalysisOrchestrator(
            structure_analyzer=self.analyzers["structure"],
            quality_analyzer=self.analyzers["quality"],
            descriptive_analyzer=self.analyzers["descriptive"],
            distribution_analyzer=self.analyzers["distribution"],
            categorical_analyzer=self.analyzers["categorical"],
            correlation_analyzer=self.analyzers["correlation"],
            outlier_analyzer=self.analyzers["outlier"],
        )

    def test_all_analyses_execute_by_default(self):
        result = self.orchestrator.analyze(self.df)

        self.assertEqual(result.overall_status, "success")

        for name, analyzer in self.analyzers.items():
            analyzer.analyze.assert_called_once_with(self.df)
            self.assertEqual(
                result.execution[name].status,
                "success",
            )

        self.assertEqual(len(result.analyses), 7)

    def test_disabled_optional_analysis_is_skipped(self):
        config = AnalysisConfig(
            run_correlation=False,
            run_outlier=False,
        )

        result = self.orchestrator.analyze(
            self.df,
            config=config,
        )

        self.assertEqual(result.overall_status, "success")

        self.assertEqual(
            result.execution["correlation"].status,
            "skipped",
        )

        self.assertEqual(
            result.execution["outlier"].status,
            "skipped",
        )

        self.analyzers["correlation"].analyze.assert_not_called()
        self.analyzers["outlier"].analyze.assert_not_called()

    def test_optional_failure_does_not_stop_pipeline(self):
        self.analyzers["correlation"].analyze.side_effect = (
            RuntimeError("Correlation calculation failed.")
        )

        result = self.orchestrator.analyze(self.df)

        self.assertEqual(
            result.overall_status,
            "partial_failure",
        )

        self.assertEqual(
            result.execution["correlation"].status,
            "failed",
        )

        self.assertIn(
            "Correlation calculation failed.",
            result.execution["correlation"].error,
        )

        self.assertEqual(
            result.execution["outlier"].status,
            "success",
        )

        self.analyzers["outlier"].analyze.assert_called_once()

    def test_structure_failure_stops_pipeline(self):
        self.analyzers["structure"].analyze.side_effect = (
            RuntimeError("Structure validation failed.")
        )

        result = self.orchestrator.analyze(self.df)

        self.assertEqual(result.overall_status, "failed")

        self.assertEqual(
            result.execution["structure"].status,
            "failed",
        )

        self.analyzers["quality"].analyze.assert_not_called()
        self.analyzers["descriptive"].analyze.assert_not_called()

    def test_quality_failure_stops_optional_analyses(self):
        self.analyzers["quality"].analyze.side_effect = (
            RuntimeError("Quality analysis failed.")
        )

        result = self.orchestrator.analyze(self.df)

        self.assertEqual(result.overall_status, "failed")

        self.assertEqual(
            result.execution["structure"].status,
            "success",
        )

        self.assertEqual(
            result.execution["quality"].status,
            "failed",
        )

        self.analyzers["descriptive"].analyze.assert_not_called()

    def test_invalid_dataframe_is_rejected(self):
        with self.assertRaises(TypeError):
            self.orchestrator.analyze([1, 2, 3])

    def test_invalid_config_is_rejected(self):
        with self.assertRaises(TypeError):
            self.orchestrator.analyze(
                self.df,
                config={"run_quality": True},
            )

    def test_essential_analyses_cannot_be_disabled(self):
        with self.assertRaises(ValueError):
            self.orchestrator.analyze(
                self.df,
                config=AnalysisConfig(run_structure=False),
            )

        with self.assertRaises(ValueError):
            self.orchestrator.analyze(
                self.df,
                config=AnalysisConfig(run_quality=False),
            )

    def test_analyzer_must_return_dictionary(self):
        self.analyzers["descriptive"].analyze.return_value = ["invalid"]

        result = self.orchestrator.analyze(self.df)

        self.assertEqual(
            result.overall_status,
            "partial_failure",
        )

        self.assertEqual(
            result.execution["descriptive"].status,
            "failed",
        )

    def test_original_dataframe_is_not_modified(self):
        original = self.df.copy(deep=True)

        self.orchestrator.analyze(self.df)

        pd.testing.assert_frame_equal(self.df, original)

    def test_structure_analyzer_invalid_output(self):
        structure = Mock()
        structure.analyze.return_value = []

        orchestrator = AnalysisOrchestrator(
            structure_analyzer=structure
        )

        result = orchestrator.analyze(
            pd.DataFrame({"value": [10, 20]})
        )

        self.assertEqual(result.overall_status, "failed")
        self.assertEqual(
            result.execution["structure"].status,
            "failed",
        )
        self.assertIn("dictionary", result.execution["structure"].error)
        self.assertNotIn("structure", result.analyses)


    def test_quality_analyzer_invalid_output(self):
        structure = Mock()
        structure.analyze.return_value = {"columns": []}

        quality = Mock()
        quality.analyze.return_value = []

        orchestrator = AnalysisOrchestrator(
            structure_analyzer=structure,
            quality_analyzer=quality,
        )

        result = orchestrator.analyze(
            pd.DataFrame({"value": [10, 20]})
        )

        self.assertEqual(result.overall_status, "failed")
        self.assertEqual(
            result.execution["structure"].status,
            "success",
        )
        self.assertEqual(
            result.execution["quality"].status,
            "failed",
        )
        self.assertNotIn("quality", result.analyses)
        self.assertEqual(
            result.execution["descriptive"].status,
            "skipped",
        )


    def test_structure_cannot_be_disabled_before_execution(self):
        structure = Mock()
        quality = Mock()

        orchestrator = AnalysisOrchestrator(
            structure_analyzer=structure,
            quality_analyzer=quality,
        )

        config = AnalysisConfig(run_structure=False)

        with self.assertRaises(ValueError):
            orchestrator.analyze(
                pd.DataFrame({"value": [10, 20]}),
                config=config,
            )

        structure.analyze.assert_not_called()
        quality.analyze.assert_not_called()


    def test_quality_cannot_be_disabled_before_execution(self):
        structure = Mock()
        quality = Mock()

        orchestrator = AnalysisOrchestrator(
            structure_analyzer=structure,
            quality_analyzer=quality,
        )

        config = AnalysisConfig(run_quality=False)

        with self.assertRaises(ValueError):
            orchestrator.analyze(
                pd.DataFrame({"value": [10, 20]}),
                config=config,
            )

        structure.analyze.assert_not_called()
        quality.analyze.assert_not_called()


if __name__ == "__main__":
    unittest.main()
