
from app.data_engine.loader import DatasetLoader
from app.data_engine.structure import StructureAnalyzer
from app.data_engine.quality import QualityAnalyzer

from app.schemas.dataset import (
    DatasetMetadata,
    DatasetUploadResponse,
)


class DatasetService:
    """
    Coordinates dataset loading, structural analysis,
    and data quality analysis.
    """

    def __init__(
        self,
        loader=None,
        structure_analyzer=None,
        quality_analyzer=None,
    ):
        self.loader = (
            loader if loader is not None else DatasetLoader()
        )
        self.structure_analyzer = (
            structure_analyzer
            if structure_analyzer is not None
            else StructureAnalyzer()
        )
        self.quality_analyzer = (
            quality_analyzer
            if quality_analyzer is not None
            else QualityAnalyzer()
        )

    def process_upload(
        self,
        filename: str,
        file_content: bytes,
    ) -> DatasetUploadResponse:

        df = self.loader.load(
            filename=filename,
            file_content=file_content,
        )

        structure = self.structure_analyzer.analyze(df)
        quality = self.quality_analyzer.analyze(df)

        metadata = DatasetMetadata(
            filename=filename,
            total_rows=int(df.shape[0]),
            total_columns=int(df.shape[1]),
        )

        return DatasetUploadResponse(
            status="success",
            message="Dataset uploaded and analyzed successfully.",
            dataset=metadata,
            structure=structure,
            quality=quality,
        )
