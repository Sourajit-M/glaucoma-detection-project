from fastapi import APIRouter
from app.schemas.prediction import HealthResponse
from app.core.model_loader import registry

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if registry.is_ready else "degraded",
        models_loaded=registry.loaded_models,
    )