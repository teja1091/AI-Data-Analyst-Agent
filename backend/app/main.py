
from fastapi import FastAPI

from app.api.dataset_routes import router as dataset_router


app = FastAPI(
    title="AI Data Analyst Agent",
    description="An AI-powered platform for dataset analysis and insights.",
    version="0.1.0"
)

app.include_router(dataset_router)


@app.get("/")
async def root():
    return {
        "status": "success",
        "message": "AI Data Analyst Agent API is running."
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy"
    }
