
import numpy as np
import pandas as pd

from pandas.api.types import (
    is_numeric_dtype,
    is_bool_dtype,
    is_complex_dtype,
)


class DistributionAnalyzer:
    """
    Analyzes the distribution of numeric columns.

    Calculates:
    - Percentiles
    - Skewness
    - Excess kurtosis
    - Histogram frequency distributions
    - Distribution shape interpretation

    Missing and non-finite values are excluded.
    Boolean and complex columns are excluded.
    The original DataFrame is never modified.
    """

    PERCENTILES = [0, 5, 10, 25, 50, 75, 90, 95, 100]

    @staticmethod
    def _validate_bins(bins):

        if isinstance(bins, bool) or not isinstance(bins, int):
            raise TypeError("Histogram bins must be an integer.")

        if bins < 1:
            raise ValueError("Histogram bins must be at least 1.")

    @staticmethod
    def _to_json_number(value):
        """Convert numerical values into JSON-safe Python values."""

        if pd.isna(value):
            return None

        result = float(value)

        if not np.isfinite(result):
            return None

        return result

    @staticmethod
    def _classify_skewness(skewness):
        """
        Classify distribution asymmetry using practical thresholds.

        These thresholds are descriptive heuristics, not statistical tests.
        """

        if skewness is None:
            return "unavailable"

        absolute_skewness = abs(skewness)

        if absolute_skewness <= 0.5:
            return "approximately_symmetric"

        if skewness > 0:
            if absolute_skewness <= 1:
                return "moderately_positive_skew"

            return "highly_positive_skew"

        if absolute_skewness <= 1:
            return "moderately_negative_skew"

        return "highly_negative_skew"

    @staticmethod
    def _classify_kurtosis(kurtosis):
        """
        Interpret Fisher excess kurtosis relative to a normal distribution.
        """

        if kurtosis is None:
            return "unavailable"

        if kurtosis > 0.5:
            return "heavier_tails_than_normal"

        if kurtosis < -0.5:
            return "lighter_tails_than_normal"

        return "similar_tail_weight_to_normal"

    @classmethod
    def analyze(
        cls,
        df: pd.DataFrame,
        bins: int = 10
    ) -> dict:

        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a Pandas DataFrame.")

        cls._validate_bins(bins)

        columns = []
        warnings = []

        for column_index in range(df.shape[1]):

            series = df.iloc[:, column_index]

            if (
                not is_numeric_dtype(series.dtype)
                or is_bool_dtype(series.dtype)
                or is_complex_dtype(series.dtype)
            ):
                continue

            values = pd.to_numeric(
                series,
                errors="coerce"
            ).to_numpy(
                dtype="float64",
                na_value=np.nan,
                copy=True
            )

            finite_mask = np.isfinite(values)
            valid_values = values[finite_mask]

            valid_count = len(valid_values)

            missing_count = int(series.isna().sum())

            non_finite_count = int(
                (~finite_mask & ~np.isnan(values)).sum()
            )

            if valid_count == 0:

                columns.append({
                    "column_index": column_index,
                    "name": str(df.columns[column_index]),
                    "pandas_dtype": str(series.dtype),
                    "valid_value_count": 0,
                    "missing_count": missing_count,
                    "non_finite_count": non_finite_count,
                    "unique_value_count": 0,
                    "statistics": None,
                    "percentiles": None,
                    "skewness": None,
                    "skewness_interpretation": "unavailable",
                    "excess_kurtosis": None,
                    "kurtosis_interpretation": "unavailable",
                    "histogram": None
                })

                warnings.append(
                    f"Column at position {column_index} "
                    "has no finite numeric values."
                )

                continue

            # Calculate the number of distinct finite values.
            # This is also used to identify constant columns.
            unique_value_count = int(
                np.unique(valid_values).size
            )

            # Percentile calculations.
            percentile_values = np.percentile(
                valid_values,
                cls.PERCENTILES
            )

            percentiles = {
                f"p{percentile}": cls._to_json_number(value)
                for percentile, value in zip(
                    cls.PERCENTILES,
                    percentile_values
                )
            }

            # Constant columns have no variation.
            # Therefore, skewness and kurtosis are mathematically undefined.
            if unique_value_count <= 1:

                skewness = None
                excess_kurtosis = None

            else:

                # Pandas skewness uses its adjusted sample estimator.
                skewness_value = pd.Series(valid_values).skew()

                skewness = cls._to_json_number(
                    skewness_value
                )

                # Pandas kurtosis returns Fisher excess kurtosis.
                # A normal distribution has an excess kurtosis of approximately 0.
                kurtosis_value = pd.Series(valid_values).kurt()

                excess_kurtosis = cls._to_json_number(
                    kurtosis_value
                )

            # Histogram calculation.
            histogram_counts, bin_edges = np.histogram(
                valid_values,
                bins=bins
            )

            histogram = {
                "bin_edges": [
                    cls._to_json_number(edge)
                    for edge in bin_edges
                ],
                "counts": [
                    int(count)
                    for count in histogram_counts
                ]
            }

            # Basic numerical statistics.
            statistics = {
                "mean": cls._to_json_number(
                    np.mean(valid_values)
                ),
                "median": cls._to_json_number(
                    np.median(valid_values)
                ),
                "minimum": cls._to_json_number(
                    np.min(valid_values)
                ),
                "maximum": cls._to_json_number(
                    np.max(valid_values)
                ),
                "standard_deviation": cls._to_json_number(
                    np.std(valid_values, ddof=0)
                )
            }

            if valid_count < 3:
                warnings.append(
                    f"Column '{series.name}' has fewer than three "
                    "finite observations. Skewness may be unavailable."
                )

            if valid_count < 4:
                warnings.append(
                    f"Column '{series.name}' has fewer than four "
                    "finite observations. Kurtosis may be unavailable."
                )

            if skewness is None:
                warnings.append(
                    f"Column '{series.name}' has undefined skewness."
                )

            if excess_kurtosis is None:
                warnings.append(
                    f"Column '{series.name}' has undefined kurtosis."
                )

            columns.append({
                "column_index": column_index,
                "name": str(df.columns[column_index]),
                "pandas_dtype": str(series.dtype),
                "valid_value_count": int(valid_count),
                "missing_count": missing_count,
                "non_finite_count": non_finite_count,
                "unique_value_count": unique_value_count,
                "statistics": statistics,
                "percentiles": percentiles,
                "skewness": skewness,
                "skewness_interpretation": (
                    cls._classify_skewness(skewness)
                ),
                "excess_kurtosis": excess_kurtosis,
                "kurtosis_interpretation": (
                    cls._classify_kurtosis(excess_kurtosis)
                ),
                "histogram": histogram
            })

        return {
            "dataset_summary": {
                "total_rows": int(df.shape[0]),
                "total_columns": int(df.shape[1]),
                "numeric_column_count": len(columns)
            },
            "configuration": {
                "histogram_bins": bins,
                "percentiles": cls.PERCENTILES.copy()
            },
            "columns": columns,
            "warnings": warnings
        }
