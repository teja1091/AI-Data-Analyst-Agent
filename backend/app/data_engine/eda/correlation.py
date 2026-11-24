
import numpy as np
import pandas as pd

from pandas.api.types import (
    is_numeric_dtype,
    is_bool_dtype,
    is_complex_dtype,
)


class CorrelationAnalyzer:
    """
    Calculates Pearson and Spearman correlations between numeric columns.

    Pairwise deletion is used: each pair of columns is evaluated using
    rows where both values are finite.

    Boolean and complex columns are excluded.
    The original DataFrame is never modified.

    A configurable maximum number of numeric columns prevents
    uncontrolled quadratic growth in correlation matrices.
    """

    DEFAULT_MAX_NUMERIC_COLUMNS = 50

    @staticmethod
    def _to_json_number(value):
        """Convert a numerical result into a JSON-safe Python value."""
        if pd.isna(value):
            return None

        result = float(value)

        if not np.isfinite(result):
            return None

        return result

    @staticmethod
    def _validate_max_numeric_columns(max_numeric_columns):
        if (
            isinstance(max_numeric_columns, bool)
            or not isinstance(max_numeric_columns, int)
        ):
            raise TypeError(
                "max_numeric_columns must be a positive integer."
            )

        if max_numeric_columns <= 0:
            raise ValueError(
                "max_numeric_columns must be greater than zero."
            )

    @classmethod
    def analyze(
        cls,
        df: pd.DataFrame,
        max_numeric_columns: int = DEFAULT_MAX_NUMERIC_COLUMNS
    ) -> dict:

        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                "Input must be a Pandas DataFrame."
            )

        cls._validate_max_numeric_columns(max_numeric_columns)

        # Identify eligible numeric columns before converting their data
        # or allocating any correlation matrices.
        eligible_columns = []

        for column_index in range(df.shape[1]):

            series = df.iloc[:, column_index]

            if (
                not is_numeric_dtype(series.dtype)
                or is_bool_dtype(series.dtype)
                or is_complex_dtype(series.dtype)
            ):
                continue

            eligible_columns.append(
                (column_index, series)
            )

        column_count = len(eligible_columns)

        # Fail explicitly instead of silently returning a partial matrix.
        if column_count > max_numeric_columns:
            raise ValueError(
                f"Dataset contains {column_count} eligible numeric "
                f"columns, exceeding the configured limit of "
                f"{max_numeric_columns}. Reduce the number of numeric "
                f"columns or increase max_numeric_columns."
            )

        numeric_columns = []
        numeric_data = {}

        for position, (column_index, series) in enumerate(
            eligible_columns
        ):

            values = pd.to_numeric(
                series,
                errors="coerce"
            ).to_numpy(
                dtype="float64",
                na_value=np.nan,
                copy=True
            )

            values[~np.isfinite(values)] = np.nan

            numeric_data[position] = values

            numeric_columns.append({
                "column_index": column_index,
                "name": str(df.columns[column_index]),
                "pandas_dtype": str(series.dtype),
                "valid_value_count": int(
                    np.isfinite(values).sum()
                ),
                "unique_value_count": int(
                    pd.Series(values).dropna().nunique()
                )
            })

        if column_count == 0:
            return {
                "dataset_summary": {
                    "total_rows": int(df.shape[0]),
                    "total_columns": int(df.shape[1]),
                    "numeric_column_count": 0
                },
                "configuration": {
                    "max_numeric_columns": max_numeric_columns
                },
                "columns": [],
                "pearson": [],
                "spearman": [],
                "pair_counts": [],
                "warnings": [
                    "No eligible numeric columns were found."
                ]
            }

        numeric_df = pd.DataFrame(numeric_data)

        # Pairwise number of valid observations.
        valid_mask = numeric_df.notna().to_numpy(
            dtype=np.int64
        )

        pair_counts = valid_mask.T @ valid_mask

        # Pandas uses pairwise complete observations when calculating
        # correlations. min_periods=2 requires at least two valid pairs.
        pearson_df = numeric_df.corr(
            method="pearson",
            min_periods=2
        )

        spearman_df = numeric_df.corr(
            method="spearman",
            min_periods=2
        )

        def convert_matrix(matrix):
            return [
                [
                    cls._to_json_number(
                        matrix.iloc[row, column]
                    )
                    for column in range(column_count)
                ]
                for row in range(column_count)
            ]

        warnings = []

        for column in numeric_columns:

            if column["valid_value_count"] < 2:
                warnings.append(
                    f"Column '{column['name']}' has fewer than two "
                    "finite values. Correlations may be unavailable."
                )

            elif column["unique_value_count"] <= 1:
                warnings.append(
                    f"Column '{column['name']}' is constant. "
                    "Its correlations are undefined."
                )

        return {
            "dataset_summary": {
                "total_rows": int(df.shape[0]),
                "total_columns": int(df.shape[1]),
                "numeric_column_count": column_count
            },
            "configuration": {
                "max_numeric_columns": max_numeric_columns
            },
            "columns": numeric_columns,
            "pearson": convert_matrix(pearson_df),
            "spearman": convert_matrix(spearman_df),
            "pair_counts": pair_counts.tolist(),
            "warnings": warnings
        }
