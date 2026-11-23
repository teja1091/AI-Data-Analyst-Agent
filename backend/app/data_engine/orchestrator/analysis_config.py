
from dataclasses import dataclass


@dataclass(frozen=True)
class AnalysisConfig:
    """
    Controls which analysis components run.

    All analyses are enabled by default.
    The configuration is immutable after creation.
    """

    run_structure: bool = True
    run_quality: bool = True
    run_descriptive: bool = True
    run_distribution: bool = True
    run_categorical: bool = True
    run_correlation: bool = True
    run_outlier: bool = True

    def __post_init__(self):
        for field_name, value in self.__dict__.items():
            if not isinstance(value, bool):
                raise TypeError(
                    f"{field_name} must be a boolean."
                )

    def enabled_analyses(self) -> dict[str, bool]:
        """Return analyzer names and their execution settings."""
        return {
            "structure": self.run_structure,
            "quality": self.run_quality,
            "descriptive": self.run_descriptive,
            "distribution": self.run_distribution,
            "categorical": self.run_categorical,
            "correlation": self.run_correlation,
            "outlier": self.run_outlier,
        }
