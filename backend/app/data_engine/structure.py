
import pandas as pd

from pandas.api.types import (
    is_bool_dtype,
    is_numeric_dtype,
    is_datetime64_any_dtype,
    is_timedelta64_dtype,
)


class StructureAnalyzer:

    @staticmethod
    def classify_column(series: pd.Series) -> str:
        """
        Identify the initial analytical type of a Pandas Series.
        """

        dtype = series.dtype

        if is_bool_dtype(dtype):
            return "boolean"

        if is_numeric_dtype(dtype):
            return "numeric"

        if is_datetime64_any_dtype(dtype):
            return "datetime"

        if is_timedelta64_dtype(dtype):
            return "timedelta"

        if isinstance(dtype, pd.CategoricalDtype):
            return "categorical"

        if pd.api.types.is_string_dtype(dtype):
            return "categorical_or_text"

        return "other"

    @classmethod
    def analyze(cls, df: pd.DataFrame) -> dict:
        """
        Generate structural information about a DataFrame.
        """

        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a Pandas DataFrame")

        columns = []

        for column_name, series in df.items():

            columns.append({
                "name": str(column_name),
                "pandas_dtype": str(series.dtype),
                "analytical_type": cls.classify_column(series)
            })

        return {
            "row_count": int(df.shape[0]),
            "column_count": int(df.shape[1]),
            "columns": columns
        }
