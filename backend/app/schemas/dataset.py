
from typing import Literal

from pydantic import BaseModel, Field


# ======================================================
# 1. Dataset Metadata
# ======================================================

class DatasetMetadata(BaseModel):
    filename: str = Field(min_length=1)
    total_rows: int = Field(ge=0)
    total_columns: int = Field(ge=0)


# ======================================================
# 2. Structure Analysis Schemas
# ======================================================

AnalyticalType = Literal[
    "boolean",
    "datetime",
    "timedelta",
    "categorical",
    "complex",
    "numeric",
    "categorical_or_text",
    "other"
]


class ColumnStructure(BaseModel):
    column_index: int = Field(ge=0)
    name: str
    pandas_dtype: str
    analytical_type: AnalyticalType


class DatasetStructure(BaseModel):
    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    columns: list[ColumnStructure]


# ======================================================
# 3. Missing Value Analysis Schemas
# ======================================================

class MissingValueColumn(BaseModel):
    column_index: int = Field(ge=0)
    name: str
    pandas_dtype: str

    missing_count: int = Field(ge=0)
    blank_count: int = Field(ge=0)
    valid_value_count: int = Field(ge=0)

    missing_percentage: float = Field(ge=0, le=100)
    blank_percentage: float = Field(ge=0, le=100)
    issue_percentage: float = Field(ge=0, le=100)


class MissingValueAnalysis(BaseModel):
    total_rows: int = Field(ge=0)
    columns: list[MissingValueColumn]


# ======================================================
# 4. Duplicate Analysis Schema
# ======================================================

class DuplicateAnalysis(BaseModel):
    total_rows: int = Field(ge=0)
    duplicate_rows: int = Field(ge=0)
    unique_rows: int = Field(ge=0)
    duplicate_percentage: float = Field(ge=0, le=100)


# ======================================================
# 5. Empty Column Schema
# ======================================================

class EmptyColumn(BaseModel):
    column_index: int = Field(ge=0)
    name: str


# ======================================================
# 6. Constant Column Schema
# ======================================================

class ConstantColumn(BaseModel):
    column_index: int = Field(ge=0)
    name: str
    unique_value_count: Literal[1]
    constant_value: str


# ======================================================
# 7. Unique Value Analysis Schema
# ======================================================

class UniqueValueColumn(BaseModel):
    column_index: int = Field(ge=0)
    name: str
    unique_count: int = Field(ge=0)


# ======================================================
# 8. Dominant Value Analysis Schemas
# ======================================================

class DominantValueColumn(BaseModel):
    column_index: int = Field(ge=0)
    name: str
    dominant_value: str

    dominant_count: int = Field(ge=0)
    total_valid_values: int = Field(ge=0)

    dominant_percentage: float = Field(ge=0, le=100)
    threshold_percentage: float = Field(gt=0, le=100)


class DominantValueAnalysis(BaseModel):
    threshold_percentage: float = Field(gt=0, le=100)
    columns: list[DominantValueColumn]


# ======================================================
# 9. Dataset Quality Summary Schema
# ======================================================

class DatasetQualitySummary(BaseModel):
    total_missing_values: int = Field(ge=0)
    total_blank_values: int = Field(ge=0)
    total_missing_and_blank_cells: int = Field(ge=0)

    missing_and_blank_percentage: float = Field(
        ge=0,
        le=100
    )

    duplicate_rows: int = Field(ge=0)
    empty_column_count: int = Field(ge=0)
    constant_column_count: int = Field(ge=0)
    dominant_value_column_count: int = Field(ge=0)


# ======================================================
# 10. Complete Quality Analysis Schema
# ======================================================

class QualityDatasetSummary(BaseModel):
    total_rows: int = Field(ge=0)
    total_columns: int = Field(ge=0)
    total_cells: int = Field(ge=0)


class DatasetQuality(BaseModel):
    dataset_summary: QualityDatasetSummary

    missing_value_analysis: MissingValueAnalysis
    duplicate_analysis: DuplicateAnalysis

    empty_columns: list[EmptyColumn]
    constant_columns: list[ConstantColumn]
    unique_value_analysis: list[UniqueValueColumn]

    dominant_value_analysis: DominantValueAnalysis

    quality_summary: DatasetQualitySummary
    warnings: list[str]


# ======================================================
# 11. Dataset Upload Response
# ======================================================

class DatasetUploadResponse(BaseModel):
    status: Literal["success"]
    message: str

    dataset: DatasetMetadata
    structure: DatasetStructure
    quality: DatasetQuality
