
import unittest

from app.data_engine.orchestrator.analysis_config import AnalysisConfig


class TestAnalysisConfig(unittest.TestCase):

    def test_all_analyses_enabled_by_default(self):
        config = AnalysisConfig()

        expected = {
            "structure": True,
            "quality": True,
            "descriptive": True,
            "distribution": True,
            "categorical": True,
            "correlation": True,
            "outlier": True,
        }

        self.assertEqual(
            config.enabled_analyses(),
            expected
        )

    def test_individual_analysis_can_be_disabled(self):
        config = AnalysisConfig(
            run_correlation=False
        )

        self.assertFalse(
            config.enabled_analyses()["correlation"]
        )

        self.assertTrue(
            config.enabled_analyses()["descriptive"]
        )

    def test_multiple_analyses_can_be_disabled(self):
        config = AnalysisConfig(
            run_correlation=False,
            run_outlier=False,
            run_categorical=False
        )

        enabled = config.enabled_analyses()

        self.assertFalse(enabled["correlation"])
        self.assertFalse(enabled["outlier"])
        self.assertFalse(enabled["categorical"])

    def test_configuration_is_immutable(self):
        config = AnalysisConfig()

        with self.assertRaises(AttributeError):
            config.run_quality = False

    def test_non_boolean_configuration_is_rejected(self):
        with self.assertRaises(TypeError):
            AnalysisConfig(run_quality="yes")


if __name__ == "__main__":
    unittest.main()
