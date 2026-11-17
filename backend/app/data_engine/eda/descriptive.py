
import numpy as np
import pandas as pd

from pandas.api.types import (
    is_numeric_dtype,
    is_bool_dtype,
    is_complex_dtype,
)


class DescriptiveAnalyzer:
    """
    Calculates descriptive statistics for real-valued numeric columns.

    The analyzer does not modify the original DataFrame.
    Missing and non-finite values are excluded from statistical
    calculations and reported separately.
    """

    @staticmethod
    def _to_json_number(value):
        """Convert a numerical result into a JSON-safe Python value."""
        if pd.isna(value):
            return None

        result = float(value)

        if not np.isfinite(result):
            return None

        return result

    @classmethod
    def analyze(cls, df: pd.DataFrame) -> dict:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a Pandas DataFrame.")

        columns = []

        for column_index in range(df.shape[1]):
            series = df.iloc[:, column_index]

            # Boolean and complex columns are not treated as
            # ordinary numerical measurement columns.
            if (
                not is_numeric_dtype(series.dtype)
                or is_bool_dtype(series.dtype)
                or is_complex_dtype(series.dtype)
            ):
                continue

            numeric_values = pd.to_numeric(
                series,
                errors="coerce"
            )

            values_array = numeric_values.to_numpy(
                dtype="float64",
                na_value=np.nan
            )

            finite_mask = np.isfinite(values_array)

            finite_values = numeric_values.iloc[
                np.flatnonzero(finite_mask)
            ]

            missing_count = int(series.isna().sum())

            non_finite_count = int(
                (~finite_mask & ~np.isnan(values_array)).sum()
            )

            statistics = None

            if not finite_values.empty:
                q1 = finite_values.quantile(0.25)
                median = finite_values.median()
                q3 = finite_values.quantile(0.75)
                minimum = finite_values.min()
                maximum = finite_values.max()

                statistics = {
                    "mean": cls._to_json_number(
                        finite_values.mean()
                    ),
                    "median": cls._to_json_number(median),
                    "standard_deviation": cls._to_json_number(
                        finite_values.std()
                    ),
                    "variance": cls._to_json_number(
                        finite_values.var()
                    ),
                    "minimum": cls._to_json_number(minimum),
                    "q1": cls._to_json_number(q1),
                    "q3": cls._to_json_number(q3),
                    "maximum": cls._to_json_number(maximum),
                    "range": cls._to_json_number(
                        maximum - minimum
                    ),
                }

            columns.append({
                "column_index": column_index,
                "name": str(df.columns[column_index]),
                "pandas_dtype": str(series.dtype),
                "valid_value_count": int(len(finite_values)),
                "missing_count": missing_count,
                "non_finite_count": non_finite_count,
                "unique_value_count": int(
                    finite_values.nunique()
                ),
                "statistics": statistics,
            })

        return {
            "dataset_summary": {
                "total_rows": int(df.shape[0]),
                "total_columns": int(df.shape[1]),
                "numeric_column_count": len(columns),
            },
            "columns": columns,
        }
