
import pandas as pd

from app.data_engine.structure import StructureAnalyzer
from app.data_engine.quality import QualityAnalyzer

from app.data_engine.eda.descriptive import DescriptiveAnalyzer
from app.data_engine.eda.distribution import DistributionAnalyzer
from app.data_engine.eda.categorical import CategoricalAnalyzer
from app.data_engine.eda.correlation import CorrelationAnalyzer
from app.data_engine.eda.outlier import OutlierAnalyzer

from app.data_engine.orchestrator.analysis_config import AnalysisConfig
from app.data_engine.orchestrator.analysis_result import (
    AnalysisExecution,
    AnalysisResult,
)


class AnalysisOrchestrator:
    """
    Coordinates deterministic dataset analysis.

    Structure and quality are essential analyses.
    EDA analyses are independently recoverable.
    """

    ANALYSIS_NAMES = (
        "structure",
        "quality",
        "descriptive",
        "distribution",
        "categorical",
        "correlation",
        "outlier",
    )

    def __init__(
        self,
        structure_analyzer=None,
        quality_analyzer=None,
        descriptive_analyzer=None,
        distribution_analyzer=None,
        categorical_analyzer=None,
        correlation_analyzer=None,
        outlier_analyzer=None,
    ):
        """Allow analyzer dependencies to be injected for testing."""

        self.structure_analyzer = (
            structure_analyzer
            if structure_analyzer is not None
            else StructureAnalyzer()
        )

        self.quality_analyzer = (
            quality_analyzer
            if quality_analyzer is not None
            else QualityAnalyzer()
        )

        self.descriptive_analyzer = (
            descriptive_analyzer
            if descriptive_analyzer is not None
            else DescriptiveAnalyzer()
        )

        self.distribution_analyzer = (
            distribution_analyzer
            if distribution_analyzer is not None
            else DistributionAnalyzer()
        )

        self.categorical_analyzer = (
            categorical_analyzer
            if categorical_analyzer is not None
            else CategoricalAnalyzer()
        )

        self.correlation_analyzer = (
            correlation_analyzer
            if correlation_analyzer is not None
            else CorrelationAnalyzer()
        )

        self.outlier_analyzer = (
            outlier_analyzer
            if outlier_analyzer is not None
            else OutlierAnalyzer()
        )

    @staticmethod
    def _collect_warnings(
        analysis_name: str,
        analysis_output: dict,
    ) -> list[str]:
        """Collect warnings returned by an individual analyzer."""

        analyzer_warnings = analysis_output.get("warnings", [])

        if not isinstance(analyzer_warnings, list):
            return [
                f"{analysis_name}: Analyzer returned an invalid warnings field."
            ]

        return [
            f"{analysis_name}: {warning}"
            for warning in analyzer_warnings
        ]

    def _create_initial_execution(self) -> dict:
        """Initialize every analyzer as skipped."""

        return {
            name: AnalysisExecution(status="skipped")
            for name in self.ANALYSIS_NAMES
        }

    def _create_result(
        self,
        overall_status: str,
        dataset_summary: dict,
        analyses: dict,
        execution: dict,
        warnings: list[str],
    ) -> AnalysisResult:
        """Build the validated unified result."""

        return AnalysisResult(
            overall_status=overall_status,
            dataset_summary=dataset_summary,
            analyses=analyses,
            execution=execution,
            warnings=warnings,
        )

    def analyze(
        self,
        df: pd.DataFrame,
        config: AnalysisConfig | None = None,
    ) -> AnalysisResult:
        """
        Run the configured analysis pipeline.

        Essential analysis failure stops execution.
        Optional analysis failure is recorded and execution continues.
        """

        # 1. Validate input
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a Pandas DataFrame.")

        if config is None:
            config = AnalysisConfig()

        if not isinstance(config, AnalysisConfig):
            raise TypeError("config must be an AnalysisConfig instance.")

        # 2. Validate configuration before executing anything
        enabled = config.enabled_analyses()

        for essential_name in ("structure", "quality"):
            if not enabled[essential_name]:
                raise ValueError(
                    f"{essential_name.capitalize()} analysis "
                    "is essential and cannot be disabled."
                )

        # 3. Initialize pipeline state
        dataset_summary = {
            "total_rows": int(df.shape[0]),
            "total_columns": int(df.shape[1]),
        }

        analyses = {}
        execution = self._create_initial_execution()
        warnings = []

        # 4. Structure analysis (essential)
        try:
            structure_output = self.structure_analyzer.analyze(df)

            if not isinstance(structure_output, dict):
                raise TypeError(
                    "Structure analyzer must return a dictionary."
                )

            analyses["structure"] = structure_output

            execution["structure"] = AnalysisExecution(
                status="success"
            )

            warnings.extend(
                self._collect_warnings(
                    "structure",
                    structure_output,
                )
            )

        except Exception as exc:
            execution["structure"] = AnalysisExecution(
                status="failed",
                error=str(exc),
            )

            warnings.append(
                "Structure analysis failed. "
                "The remaining pipeline was not executed."
            )

            return self._create_result(
                overall_status="failed",
                dataset_summary=dataset_summary,
                analyses=analyses,
                execution=execution,
                warnings=warnings,
            )

        # 5. Quality analysis (essential)
        try:
            quality_output = self.quality_analyzer.analyze(df)

            if not isinstance(quality_output, dict):
                raise TypeError(
                    "Quality analyzer must return a dictionary."
                )

            analyses["quality"] = quality_output

            execution["quality"] = AnalysisExecution(
                status="success"
            )

            warnings.extend(
                self._collect_warnings(
                    "quality",
                    quality_output,
                )
            )

        except Exception as exc:
            execution["quality"] = AnalysisExecution(
                status="failed",
                error=str(exc),
            )

            warnings.append(
                "Quality analysis failed. "
                "The remaining pipeline was not executed."
            )

            return self._create_result(
                overall_status="failed",
                dataset_summary=dataset_summary,
                analyses=analyses,
                execution=execution,
                warnings=warnings,
            )

        # 6. Optional EDA analyses
        optional_analyzers = {
            "descriptive": (
                enabled["descriptive"],
                self.descriptive_analyzer,
            ),
            "distribution": (
                enabled["distribution"],
                self.distribution_analyzer,
            ),
            "categorical": (
                enabled["categorical"],
                self.categorical_analyzer,
            ),
            "correlation": (
                enabled["correlation"],
                self.correlation_analyzer,
            ),
            "outlier": (
                enabled["outlier"],
                self.outlier_analyzer,
            ),
        }

        for analysis_name, (is_enabled, analyzer) in (
            optional_analyzers.items()
        ):

            if not is_enabled:
                continue

            try:
                analysis_output = analyzer.analyze(df)

                if not isinstance(analysis_output, dict):
                    raise TypeError(
                        f"{analysis_name} analyzer must return a dictionary."
                    )

                analyses[analysis_name] = analysis_output

                execution[analysis_name] = AnalysisExecution(
                    status="success"
                )

                warnings.extend(
                    self._collect_warnings(
                        analysis_name,
                        analysis_output,
                    )
                )

            except Exception as exc:
                execution[analysis_name] = AnalysisExecution(
                    status="failed",
                    error=str(exc),
                )

                warnings.append(
                    f"{analysis_name} analysis failed: {exc}"
                )

        # 7. Determine overall execution status
        optional_failures = any(
            execution[name].status == "failed"
            for name in optional_analyzers
        )

        overall_status = (
            "partial_failure"
            if optional_failures
            else "success"
        )

        return self._create_result(
            overall_status=overall_status,
            dataset_summary=dataset_summary,
            analyses=analyses,
            execution=execution,
            warnings=warnings,
        )
