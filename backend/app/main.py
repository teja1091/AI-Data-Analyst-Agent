
from fastapi import FastAPI

from app.api.dataset_routes import router as dataset_router


APP_VERSION = "0.1.0"

app = FastAPI(
    title="AI Data Analyst Agent",
    description=(
        "An AI-powered platform for dataset analysis, "
        "data quality assessment, and insights."
    ),
    version=APP_VERSION
)

app.include_router(dataset_router)


@app.get("/", tags=["System"])
async def root():
    return {
        "status": "success",
        "message": "AI Data Analyst Agent API is running."
    }


@app.get("/health", tags=["System"])
async def health_check():
    return {
        "status": "healthy"
    }
