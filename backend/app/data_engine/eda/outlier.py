
import numpy as np
import pandas as pd

from pandas.api.types import (
    is_numeric_dtype,
    is_bool_dtype,
    is_complex_dtype,
)


class OutlierAnalyzer:
    """
    Detects potential outliers in numeric columns using IQR and Z-score.

    Missing and non-finite values are excluded from calculations.
    Boolean and complex columns are excluded.
    The original DataFrame is never modified.
    """

    @staticmethod
    def _validate_thresholds(iqr_multiplier, zscore_threshold):

        if (
            isinstance(iqr_multiplier, bool)
            or not isinstance(iqr_multiplier, (int, float))
        ):
            raise TypeError("IQR multiplier must be a number.")

        if (
            isinstance(zscore_threshold, bool)
            or not isinstance(zscore_threshold, (int, float))
        ):
            raise TypeError("Z-score threshold must be a number.")

        if not np.isfinite(iqr_multiplier) or iqr_multiplier <= 0:
            raise ValueError("IQR multiplier must be greater than zero.")

        if not np.isfinite(zscore_threshold) or zscore_threshold <= 0:
            raise ValueError("Z-score threshold must be greater than zero.")

    @staticmethod
    def _to_json_number(value):

        if pd.isna(value):
            return None

        result = float(value)

        if not np.isfinite(result):
            return None

        return result

    @classmethod
    def analyze(
        cls,
        df: pd.DataFrame,
        iqr_multiplier: float = 1.5,
        zscore_threshold: float = 3.0,
    ) -> dict:

        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a Pandas DataFrame.")

        cls._validate_thresholds(
            iqr_multiplier,
            zscore_threshold
        )

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

            if valid_count == 0:

                columns.append({
                    "column_index": column_index,
                    "name": str(df.columns[column_index]),
                    "pandas_dtype": str(series.dtype),
                    "valid_value_count": 0,
                    "missing_count": int(series.isna().sum()),
                    "non_finite_count": int(
                        (~finite_mask & ~np.isnan(values)).sum()
                    ),
                    "iqr": None,
                    "iqr_bounds": None,
                    "iqr_outlier_count": 0,
                    "iqr_outlier_indices": [],
                    "zscore_outlier_count": 0,
                    "zscore_outlier_indices": [],
                    "statistics": None
                })

                warnings.append(
                    f"Column '{series.name}' has no finite numeric values."
                )

                continue

            q1 = float(np.percentile(valid_values, 25))
            median = float(np.median(valid_values))
            q3 = float(np.percentile(valid_values, 75))

            iqr = q3 - q1

            lower_bound = q1 - iqr_multiplier * iqr
            upper_bound = q3 + iqr_multiplier * iqr

            iqr_mask = (
                (values < lower_bound)
                | (values > upper_bound)
            ) & finite_mask

            mean = float(np.mean(valid_values))
            std = float(np.std(valid_values, ddof=0))

            if std > 0:

                zscores = np.zeros(len(values), dtype="float64")

                zscores[finite_mask] = (
                    values[finite_mask] - mean
                ) / std

                zscore_mask = (
                    np.abs(zscores) > zscore_threshold
                ) & finite_mask

            else:
                zscore_mask = np.zeros(
                    len(values),
                    dtype=bool
                )

                warnings.append(
                    f"Column '{series.name}' has zero variance. "
                    "Z-score detection is unavailable."
                )

            iqr_indices = np.flatnonzero(iqr_mask).tolist()
            zscore_indices = np.flatnonzero(zscore_mask).tolist()

            columns.append({
                "column_index": column_index,
                "name": str(df.columns[column_index]),
                "pandas_dtype": str(series.dtype),
                "valid_value_count": int(valid_count),
                "missing_count": int(series.isna().sum()),
                "non_finite_count": int(
                    (~finite_mask & ~np.isnan(values)).sum()
                ),
                "iqr": cls._to_json_number(iqr),
                "iqr_bounds": {
                    "lower": cls._to_json_number(lower_bound),
                    "upper": cls._to_json_number(upper_bound)
                },
                "iqr_outlier_count": len(iqr_indices),
                "iqr_outlier_indices": iqr_indices,
                "zscore_outlier_count": len(zscore_indices),
                "zscore_outlier_indices": zscore_indices,
                "statistics": {
                    "mean": cls._to_json_number(mean),
                    "median": cls._to_json_number(median),
                    "standard_deviation": cls._to_json_number(std),
                    "minimum": cls._to_json_number(
                        np.min(valid_values)
                    ),
                    "maximum": cls._to_json_number(
                        np.max(valid_values)
                    )
                }
            })

        return {
            "dataset_summary": {
                "total_rows": int(df.shape[0]),
                "total_columns": int(df.shape[1]),
                "numeric_column_count": len(columns)
            },
            "configuration": {
                "iqr_multiplier": float(iqr_multiplier),
                "zscore_threshold": float(zscore_threshold)
            },
            "columns": columns,
            "warnings": warnings
        }
