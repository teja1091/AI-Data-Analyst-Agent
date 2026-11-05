
import pandas as pd

from pandas.api.types import (
    is_bool_dtype,
    is_object_dtype,
    is_string_dtype,
)


class QualityAnalyzer:
    """
    Analyze data-quality issues in a Pandas DataFrame.

    Responsibilities:
    - Missing-value analysis
    - Blank-value analysis
    - Duplicate-row detection
    - Empty-column detection
    - Constant-column detection
    - Unique-value analysis
    - Dominant-value analysis
    - Quality summary and warnings

    This class only analyzes data.
    It never modifies the original DataFrame.
    """

    # --------------------------------------------------
    # 1. Input Validation
    # --------------------------------------------------

    @staticmethod
    def _validate_dataframe(df: pd.DataFrame) -> None:

        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                "Input must be a Pandas DataFrame"
            )

    # --------------------------------------------------
    # 2. Blank Value Detection
    # --------------------------------------------------

    @staticmethod
    def _identify_blank_values(
        series: pd.Series
    ) -> pd.Series:
        """
        Identify empty strings and whitespace-only strings.
        """

        return series.map(
            lambda value: (
                isinstance(value, str)
                and value.strip() == ""
            )
        )

    # --------------------------------------------------
    # 3. Categorical Column Detection
    # --------------------------------------------------

    @staticmethod
    def _is_categorical_column(
        series: pd.Series,
        column_name: str
    ) -> bool:
        """
        Identify columns suitable for categorical
        dominant-value analysis.

        Rules:
        - Boolean columns are categorical.
        - Pandas category columns are categorical.
        - String/object columns are categorical candidates
          when their distinct-value count is at most 50.
        - Identifier-like column names are excluded.

        This is an initial heuristic, not a permanent
        semantic classification system.
        """

        name = str(column_name).strip().lower()

        identifier_names = {
            "id",
            "uuid",
            "identifier",
            "order_id",
            "customer_id",
            "product_id",
            "user_id",
            "transaction_id",
        }

        if name in identifier_names or name.endswith("_id"):
            return False

        if (
            is_bool_dtype(series.dtype)
            or isinstance(series.dtype, pd.CategoricalDtype)
        ):
            return True

        if (
            is_object_dtype(series.dtype)
            or is_string_dtype(series.dtype)
        ):
            missing_mask = series.isna()
            blank_mask = QualityAnalyzer._identify_blank_values(
                series
            )

            valid_values = series[
                ~(missing_mask | blank_mask)
            ]

            return valid_values.nunique(dropna=True) <= 50

        return False

    # --------------------------------------------------
    # 4. Missing and Blank Value Analysis
    # --------------------------------------------------

    @classmethod
    def analyze_missing_values(
        cls,
        df: pd.DataFrame
    ) -> dict:

        cls._validate_dataframe(df)

        total_rows = len(df)
        columns = []

        for column_index in range(df.shape[1]):

            series = df.iloc[:, column_index]

            missing_mask = series.isna()

            blank_mask = cls._identify_blank_values(
                series
            )

            missing_count = int(missing_mask.sum())

            blank_count = int(blank_mask.sum())

            valid_value_count = int(
                (~(missing_mask | blank_mask)).sum()
            )

            if total_rows > 0:

                missing_percentage = (
                    missing_count / total_rows
                ) * 100

                blank_percentage = (
                    blank_count / total_rows
                ) * 100

                issue_percentage = (
                    (missing_count + blank_count)
                    / total_rows
                ) * 100

            else:
                missing_percentage = 0.0
                blank_percentage = 0.0
                issue_percentage = 0.0

            columns.append({
                "column_index": column_index,
                "name": str(df.columns[column_index]),
                "pandas_dtype": str(series.dtype),
                "missing_count": missing_count,
                "blank_count": blank_count,
                "valid_value_count": valid_value_count,
                "missing_percentage": round(
                    missing_percentage, 2
                ),
                "blank_percentage": round(
                    blank_percentage, 2
                ),
                "issue_percentage": round(
                    issue_percentage, 2
                )
            })

        return {
            "total_rows": int(total_rows),
            "columns": columns
        }

    # --------------------------------------------------
    # 5. Duplicate Row Analysis
    # --------------------------------------------------

    @classmethod
    def analyze_duplicates(
        cls,
        df: pd.DataFrame
    ) -> dict:

        cls._validate_dataframe(df)

        total_rows = len(df)

        duplicate_mask = df.duplicated()

        duplicate_count = int(duplicate_mask.sum())

        unique_row_count = total_rows - duplicate_count

        duplicate_percentage = (
            duplicate_count / total_rows * 100
            if total_rows > 0
            else 0.0
        )

        return {
            "total_rows": int(total_rows),
            "duplicate_rows": duplicate_count,
            "unique_rows": int(unique_row_count),
            "duplicate_percentage": round(
                duplicate_percentage, 2
            )
        }

    # --------------------------------------------------
    # 6. Empty Column Detection
    # --------------------------------------------------

    @classmethod
    def identify_empty_columns(
        cls,
        df: pd.DataFrame
    ) -> list:

        cls._validate_dataframe(df)

        empty_columns = []

        for column_index in range(df.shape[1]):

            series = df.iloc[:, column_index]

            missing_mask = series.isna()

            blank_mask = cls._identify_blank_values(
                series
            )

            valid_mask = ~(
                missing_mask | blank_mask
            )

            if not valid_mask.any():

                empty_columns.append({
                    "column_index": column_index,
                    "name": str(df.columns[column_index])
                })

        return empty_columns

    # --------------------------------------------------
    # 7. Constant Column Detection
    # --------------------------------------------------

    @classmethod
    def identify_constant_columns(
        cls,
        df: pd.DataFrame
    ) -> list:

        cls._validate_dataframe(df)

        constant_columns = []

        for column_index in range(df.shape[1]):

            series = df.iloc[:, column_index]

            missing_mask = series.isna()

            blank_mask = cls._identify_blank_values(
                series
            )

            valid_values = series[
                ~(missing_mask | blank_mask)
            ]

            unique_count = int(
                valid_values.nunique(dropna=True)
            )

            if unique_count == 1:

                constant_columns.append({
                    "column_index": column_index,
                    "name": str(df.columns[column_index]),
                    "unique_value_count": unique_count,
                    "constant_value": str(
                        valid_values.iloc[0]
                    )
                })

        return constant_columns

    # --------------------------------------------------
    # 8. Unique Value Analysis
    # --------------------------------------------------

    @classmethod
    def analyze_unique_values(
        cls,
        df: pd.DataFrame
    ) -> list:

        cls._validate_dataframe(df)

        unique_analysis = []

        for column_index in range(df.shape[1]):

            series = df.iloc[:, column_index]

            missing_mask = series.isna()

            blank_mask = cls._identify_blank_values(
                series
            )

            valid_values = series[
                ~(missing_mask | blank_mask)
            ]

            unique_count = int(
                valid_values.nunique(dropna=True)
            )

            unique_analysis.append({
                "column_index": column_index,
                "name": str(df.columns[column_index]),
                "unique_count": unique_count
            })

        return unique_analysis

    # --------------------------------------------------
    # 9. Dominant Value Analysis
    # --------------------------------------------------

    @classmethod
    def analyze_value_distribution(
        cls,
        df: pd.DataFrame,
        threshold: float = 0.80
    ) -> list:
        """
        Detect dominant values in categorical columns.

        Default threshold:
            0.80 = 80%

        Missing and blank values are excluded.

        A dominant value is an observation, not
        automatically a data-quality problem.
        """

        cls._validate_dataframe(df)

        if not isinstance(threshold, (int, float)):
            raise TypeError(
                "Threshold must be a number between 0 and 1"
            )

        if not 0 < threshold <= 1:
            raise ValueError(
                "Threshold must be greater than 0 and at most 1"
            )

        if isinstance(threshold, bool) or not isinstance(
                threshold, (int, float)
            ):
                raise TypeError(
                    "Threshold must be a number between 0 and 1"
                )

        dominant_columns = []

        for column_index in range(df.shape[1]):

            series = df.iloc[:, column_index]

            column_name = str(
                df.columns[column_index]
            )

            if not cls._is_categorical_column(
                series,
                column_name
            ):
                continue

            missing_mask = series.isna()

            blank_mask = cls._identify_blank_values(
                series
            )

            valid_values = series[
                ~(missing_mask | blank_mask)
            ]

            total_valid_values = len(valid_values)

            if total_valid_values == 0:
                continue

            value_counts = valid_values.value_counts(
                dropna=True
            )

            dominant_value = value_counts.index[0]

            dominant_count = int(value_counts.iloc[0])

            dominant_percentage = (
                dominant_count / total_valid_values
            )

            if dominant_percentage >= threshold:

                dominant_columns.append({
                    "column_index": column_index,
                    "name": column_name,
                    "dominant_value": str(dominant_value),
                    "dominant_count": dominant_count,
                    "total_valid_values": total_valid_values,
                    "dominant_percentage": round(
                        dominant_percentage * 100, 2
                    ),
                    "threshold_percentage": round(
                        threshold * 100, 2
                    )
                })

        return dominant_columns

    # --------------------------------------------------
    # 10. Overall Quality Analysis
    # --------------------------------------------------

    @classmethod
    def analyze(
        cls,
        df: pd.DataFrame,
        dominant_threshold: float = 0.80
    ) -> dict:
        """
        Execute all quality checks and return
        a consolidated quality report.
        """

        cls._validate_dataframe(df)

        total_rows = len(df)
        total_columns = len(df.columns)

        total_cells = total_rows * total_columns

        missing_analysis = cls.analyze_missing_values(df)

        duplicate_analysis = cls.analyze_duplicates(df)

        empty_columns = cls.identify_empty_columns(df)

        constant_columns = cls.identify_constant_columns(df)

        unique_analysis = cls.analyze_unique_values(df)

        dominant_columns = cls.analyze_value_distribution(
            df,
            threshold=dominant_threshold
        )

        total_missing_values = sum(
            column["missing_count"]
            for column in missing_analysis["columns"]
        )

        total_blank_values = sum(
            column["blank_count"]
            for column in missing_analysis["columns"]
        )

        total_missing_and_blank_cells = (
            total_missing_values + total_blank_values
        )

        if total_cells > 0:

            missing_and_blank_percentage = (
                total_missing_and_blank_cells
                / total_cells
            ) * 100

        else:
            missing_and_blank_percentage = 0.0

        # ----------------------------------------------
        # Generate Data Quality Warnings
        # ----------------------------------------------

        warnings = []

        if total_missing_values > 0:
            warnings.append(
                f"Dataset contains {total_missing_values} missing values."
            )

        if total_blank_values > 0:
            warnings.append(
                f"Dataset contains {total_blank_values} blank values."
            )

        if duplicate_analysis["duplicate_rows"] > 0:
            warnings.append(
                f"Dataset contains "
                f"{duplicate_analysis['duplicate_rows']} duplicate rows."
            )

        if empty_columns:
            warnings.append(
                f"Dataset contains {len(empty_columns)} empty columns."
            )

        if constant_columns:
            warnings.append(
                f"Dataset contains {len(constant_columns)} constant columns."
            )

        # Dominant values are observations,
        # not automatically quality warnings.

        return {
            "dataset_summary": {
                "total_rows": int(total_rows),
                "total_columns": int(total_columns),
                "total_cells": int(total_cells)
            },

            "missing_value_analysis": missing_analysis,

            "duplicate_analysis": duplicate_analysis,

            "empty_columns": empty_columns,

            "constant_columns": constant_columns,

            "unique_value_analysis": unique_analysis,

            "dominant_value_analysis": {
                "threshold_percentage": round(
                    dominant_threshold * 100, 2
                ),
                "columns": dominant_columns
            },

            "quality_summary": {
                "total_missing_values": total_missing_values,
                "total_blank_values": total_blank_values,
                "total_missing_and_blank_cells": (
                    total_missing_and_blank_cells
                ),
                "missing_and_blank_percentage": round(
                    missing_and_blank_percentage, 2
                ),
                "duplicate_rows": (
                    duplicate_analysis["duplicate_rows"]
                ),
                "empty_column_count": len(empty_columns),
                "constant_column_count": len(constant_columns),
                "dominant_value_column_count": len(
                    dominant_columns
                )
            },

            "warnings": warnings
        }
