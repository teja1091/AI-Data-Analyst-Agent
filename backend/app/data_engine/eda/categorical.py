
import pandas as pd

from pandas.api.types import (
    is_bool_dtype,
    is_object_dtype,
    is_string_dtype,
    is_categorical_dtype,
)


class CategoricalAnalyzer:

    @staticmethod
    def _is_categorical(series: pd.Series) -> bool:
        dtype = series.dtype

        return (
            is_bool_dtype(dtype)
            or isinstance(dtype, pd.CategoricalDtype)
            or is_object_dtype(dtype)
            or is_string_dtype(dtype)
        )

    @classmethod
    def analyze(cls, df: pd.DataFrame) -> dict:

        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a Pandas DataFrame.")

        columns = []

        for column_index in range(df.shape[1]):

            series = df.iloc[:, column_index]

            if not cls._is_categorical(series):
                continue

            # Separate missing and blank values from valid categories.
            missing_mask = series.isna()

            blank_mask = (
                series.astype("string")
                .str.strip()
                .eq("")
                .fillna(False)
            )

            invalid_mask = missing_mask | blank_mask

            valid_values = series[~invalid_mask]

            total_valid = len(valid_values)

            value_counts = valid_values.value_counts(
                dropna=True,
                sort=True
            )

            categories = []

            for value, count in value_counts.items():

                frequency = int(count)

                categories.append({
                    "value": str(value),
                    "count": frequency,
                    "percentage": (
                        round(frequency / total_valid * 100, 2)
                        if total_valid > 0
                        else 0.0
                    )
                })

            most_frequent = (
                categories[0]["value"]
                if categories else None
            )

            least_frequent = (
                categories[-1]["value"]
                if categories else None
            )

            columns.append({
                "column_index": column_index,
                "name": str(df.columns[column_index]),
                "pandas_dtype": str(series.dtype),
                "unique_category_count": int(
                    valid_values.nunique()
                ),
                "valid_value_count": int(total_valid),
                "missing_count": int(missing_mask.sum()),
                "blank_count": int(
                    (blank_mask & ~missing_mask).sum()
                ),
                "most_frequent_category": most_frequent,
                "least_frequent_category": least_frequent,
                "categories": categories
            })

        return {
            "dataset_summary": {
                "total_rows": int(df.shape[0]),
                "total_columns": int(df.shape[1]),
                "categorical_column_count": len(columns)
            },
            "columns": columns
        }
