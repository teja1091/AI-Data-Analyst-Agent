
from fastapi import APIRouter, UploadFile, File, HTTPException

from app.services.dataset_service import DatasetService
from app.schemas.dataset import DatasetUploadResponse


router = APIRouter(
    prefix="/api/datasets",
    tags=["Datasets"]
)

dataset_service = DatasetService()

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@router.post(
    "/upload",
    response_model=DatasetUploadResponse
)
async def upload_dataset(
    file: UploadFile = File(...)
):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required."
        )

    try:
        # Read at most 10 MB + 1 byte to detect oversized uploads.
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

        response = dataset_service.process_upload(
            filename=file.filename,
            file_content=file_content
        )

        return response

    except HTTPException:
        raise

    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while processing the dataset."
        ) from exc

    finally:
        await file.close()
