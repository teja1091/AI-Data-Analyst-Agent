
from app.data_engine.loader import DatasetLoader
from app.data_engine.structure import StructureAnalyzer
from app.data_engine.quality import QualityAnalyzer

from app.schemas.dataset import (
    DatasetMetadata,
    DatasetUploadResponse,
)


class DatasetService:

    def __init__(self):
        self.loader = DatasetLoader()
        self.structure_analyzer = StructureAnalyzer()
        self.quality_analyzer = QualityAnalyzer()

    def process_upload(
        self,
        filename: str,
        file_content: bytes
    ) -> DatasetUploadResponse:

        # 1. Load uploaded file into a DataFrame
        df = self.loader.load(
            filename=filename,
            file_content=file_content
        )

        # 2. Analyze dataset structure
        structure = self.structure_analyzer.analyze(df)

        # 3. Analyze dataset quality
        quality = self.quality_analyzer.analyze(df)

        # 4. Create dataset metadata
        metadata = DatasetMetadata(
            filename=filename,
            total_rows=int(df.shape[0]),
            total_columns=int(df.shape[1])
        )

        # 5. Build structured API response
        return DatasetUploadResponse(
            status="success",
            message="Dataset uploaded and analyzed successfully.",
            dataset=metadata,
            structure=structure,
            quality=quality
        )
