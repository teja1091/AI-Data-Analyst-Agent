
import logging

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException,
    Depends
)
from fastapi.concurrency import run_in_threadpool

from app.services.dataset_service import DatasetService
from app.schemas.dataset import DatasetUploadResponse


logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/api/datasets",
    tags=["Datasets"]
)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


def get_dataset_service() -> DatasetService:
    return DatasetService()


@router.post(
    "/upload",
    response_model=DatasetUploadResponse
)
async def upload_dataset(
    file: UploadFile = File(...),
    service: DatasetService = Depends(get_dataset_service)
):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required."
        )

    try:
        file_content = await file.read(MAX_FILE_SIZE + 1)

        if not file_content:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty."
            )

        if len(file_content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="File size exceeds the 10 MB limit."
            )

        response = await run_in_threadpool(
            service.process_upload,
            file.filename,
            file_content
        )

        return response

    except HTTPException:
        raise

    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        ) from exc

    except Exception:
        logger.exception(
            "Unexpected dataset processing error"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "An unexpected error occurred "
                "while processing the dataset."
            )
        )

    finally:
        await file.close()
