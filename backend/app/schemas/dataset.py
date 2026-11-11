
from pydantic import BaseModel, Field


class DatasetMetadata(BaseModel):
    filename: str
    total_rows: int = Field(ge=0)
    total_columns: int = Field(ge=0)


class DatasetUploadResponse(BaseModel):
    status: str
    message: str
    dataset: DatasetMetadata
    structure: dict
    quality: dict
