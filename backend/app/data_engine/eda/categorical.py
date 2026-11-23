
import pandas as pd

from pandas.api.types import (
    is_bool_dtype,
    is_object_dtype,
    is_string_dtype,
)


class CategoricalAnalyzer:
    DEFAULT_MAX_CATEGORIES = 20

    @staticmethod
    def _is_categorical(series: pd.Series) -> bool:
        dtype = series.dtype

        return (
            is_bool_dtype(dtype)
            or isinstance(dtype, pd.CategoricalDtype)
            or is_object_dtype(dtype)
            or is_string_dtype(dtype)
        )

    @staticmethod
    def _validate_max_categories(max_categories):
        if max_categories is None:
            return

        if isinstance(max_categories, bool) or not isinstance(
            max_categories, int
        ):
            raise TypeError(
                "max_categories must be a positive integer or None."
            )

        if max_categories <= 0:
            raise ValueError(
                "max_categories must be greater than zero."
            )

    @classmethod
    def analyze(
        cls,
        df: pd.DataFrame,
        max_categories: int | None = DEFAULT_MAX_CATEGORIES
    ) -> dict:

        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                "Input must be a Pandas DataFrame."
            )

        cls._validate_max_categories(max_categories)

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

            total_category_count = int(len(value_counts))

            # Determine the most and least frequent categories
            # from the complete distribution.
            most_frequent = (
                str(value_counts.index[0])
                if total_category_count > 0
                else None
            )

            least_frequent = (
                str(value_counts.index[-1])
                if total_category_count > 0
                else None
            )

            # Limit only the returned category details.
            returned_counts = (
                value_counts
                if max_categories is None
                else value_counts.head(max_categories)
            )

            categories = []

            for value, count in returned_counts.items():

                frequency = int(count)

                categories.append({
                    "value": str(value),
                    "count": frequency,
                    "percentage": (
                        round(
                            frequency / total_valid * 100,
                            2
                        )
                        if total_valid > 0
                        else 0.0
                    )
                })

            returned_category_count = len(categories)

            omitted_category_count = (
                total_category_count - returned_category_count
            )

            columns.append({
                "column_index": column_index,
                "name": str(df.columns[column_index]),
                "pandas_dtype": str(series.dtype),
                "unique_category_count": int(
                    valid_values.nunique()
                ),
                "total_category_count": total_category_count,
                "returned_category_count": returned_category_count,
                "omitted_category_count": omitted_category_count,
                "categories_truncated": omitted_category_count > 0,
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
